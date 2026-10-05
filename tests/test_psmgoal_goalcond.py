"""Goal-conditioned psmgoal (policy_index=goal, measure_loss=squared|softmax, actor_input=goal).

Spec: docs/design/2026-10-01-psmgoal-goal-conditioned.md. Checks: the new keys default off in
both configs; the goal sampler's shares; the goal-mode loss against a hand computation, and
that it ignores the policy code; the Polyak step of the h target; the bootstrap is the actor
at the row's measure goal; `softmax_td_loss` against numpy and its shift invariance; the
actor loss's Q by hand and its gradients; the actor's input width; the four test-time
readouts act with finite actions; each `create` assert fires; a checkpoint written before
`target_w_star` existed restores.

Run in its own process: JAX_PLATFORMS=cpu .venv/bin/python -m pytest
tests/test_psmgoal_goalcond.py -q -p no:cacheprovider
"""
import os
import pickle

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import flax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml

import agents.psmgoal as pg
from agents.psmgoal import PSMGoalAgent, get_config
from utils.datasets import Dataset, hindsight_goal_idxs
from utils.flax_utils import restore_agent, save_agent
from utils.psm_networks import tanh_gaussian_sample

OB, DA, Z, CODE, N = 6, 3, 8, 4, 8

NEW_KEYS = {"policy_index": "code", "measure_loss": "squared", "measure_temp": 1.0,
            "actor_input": "w", "goal_cur_frac": 0.0, "eval_goal_source": "relabel"}

# The two training arms of the spec's "Runs" table.
GC = {"policy_index": "goal", "train_actor": True, "actor_input": "goal", "goal_cur_frac": 0.2,
      "goal_random_frac": 0.3}
ARMS = {"sq": {**GC, "measure_loss": "squared"}, "sm": {**GC, "measure_loss": "softmax"}}

# The four test-time readouts, as overrides on top of the training config.
READOUTS = {
    "hgoal": {"coef_source": "amortized", "acting": "gpi"},
    "lp": {"coef_source": "lp", "acting": "gpi"},
    "actor_rel": {"acting": "distill", "eval_redistill": False, "eval_goal_source": "relabel"},
    "actor_env": {"acting": "distill", "eval_redistill": False, "eval_goal_source": "env"},
}


def _cfg(**over):
    c = get_config()
    c.z_dim = Z
    c.max_log_seed = CODE
    c.batch_size = N
    c.k_goals = 4
    c.gpi_num_u = 5
    c.num_inference_steps = 2
    c.infer_batch = N
    c.allow_untrained_flow = True
    c.measure.hidden_dim = 16
    c.w.hidden_dim = 16
    c.l.hidden_dim = 16
    c.actor.hidden_dim = 16
    c.w_star.hidden_dim = 16
    c.flow.hidden_dims = (16, 16)
    c.fb_hit_spec = "0:3:0.5"
    for k, v in over.items():
        setattr(c, k, v)
    return c


def _agent(seed=0, **over):
    return PSMGoalAgent.create(seed, np.zeros((2, OB), np.float32), np.zeros((2, DA), np.float32),
                               _cfg(**over))


def _batch(n=N, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "observations": rng.standard_normal((n, OB)).astype(np.float32),
        "next_observations": rng.standard_normal((n, OB)).astype(np.float32),
        "noise_preimage": rng.standard_normal((n, DA)).astype(np.float32),
        "goals": rng.standard_normal((n, OB)).astype(np.float32),
        "fb_goals": rng.standard_normal((n, OB)).astype(np.float32),
        "index": np.arange(n).astype(np.int32),
        "rewards": -np.ones((n,), np.float32),
    }


def _relabel(n=32, seed=5, n_rewarding=6):
    b = _batch(n=n, seed=seed)
    rew = np.zeros((n,), np.float32)
    rew[:n_rewarding] = 1.0
    return b, rew


def _leaves(tree):
    return jax.tree_util.tree_leaves(tree)


def _changed(t0, t1):
    return any(not jnp.array_equal(a, b) for a, b in zip(_leaves(t0), _leaves(t1)))


def _same(t0, t1):
    return all(jnp.array_equal(a, b) for a, b in zip(_leaves(t0), _leaves(t1)))


def _code(seed):
    return jax.random.bernoulli(jax.random.PRNGKey(seed), 0.5, (N, CODE)).astype(jnp.float32)


# ------------------------------------------------------------------ defaults

def test_new_keys_default_off_in_get_config_and_yaml():
    c = get_config()
    path = os.path.join(os.path.dirname(__file__), "..", "configs", "agent", "psmgoal.yaml")
    with open(path) as f:
        y = yaml.safe_load(f)
    for k, v in NEW_KEYS.items():
        assert k in c and c[k] == v, f"get_config()[{k!r}]"
        assert k in y and y[k] == v, f"psmgoal.yaml[{k!r}]"


def test_default_agent_restores_checkpoint_without_new_fields(tmp_path):
    """A checkpoint written before `target_w_star` (and the eval goal fields) existed restores
    into the default agent, keeps its trained measure and acts."""
    ag = _agent(seed=1)
    ag, _ = ag.update(_batch())
    saved = flax.serialization.to_state_dict(ag)
    for k in ("target_w_star", "eval_ref", "eval_actor_goal"):
        saved.pop(k)                               # KeyError here = the field does not exist yet
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with open(run_dir / "params_10.pkl", "wb") as f:
        pickle.dump({"agent": saved}, f)
    restored = restore_agent(_agent(seed=2), str(run_dir), 10)
    assert _same(restored.basis.params, ag.basis.params)
    assert _same(restored.w.params, ag.w.params)
    b, rew = _relabel()
    ev = restored.infer_eval_goals(b, rew)
    a = ev.sample_actions(jnp.asarray(b["observations"][0]), seed=jax.random.PRNGKey(0))
    assert a.shape == (DA,) and np.all(np.isfinite(np.asarray(a)))


# ------------------------------------------------------------------ goal sampler

def test_goal_shares_own_future_random():
    """cur_frac of rows take their own row, random_frac a random row, the rest a future row.
    discount ~ 1 sends every future goal to the trajectory's last row, so the three kinds
    are told apart by the goal row alone."""
    size = 20000
    terminal_locs = np.array([size - 1])
    idxs = np.random.default_rng(1).integers(0, 5000, size=40000)
    goal, is_random = hindsight_goal_idxs(idxs, terminal_locs, size, 1.0 - 1e-9, 0.3,
                                          rng=np.random.default_rng(0), cur_frac=0.2)
    own = goal == idxs
    future = goal == size - 1
    assert abs(own.mean() - 0.2) < 0.01
    assert abs(future.mean() - 0.5) < 0.01
    assert abs(is_random.mean() - 0.3) < 0.01


def test_cur_frac_zero_draws_no_extra_random_numbers():
    """At cur_frac=0 the goals and the generator's state after the call equal the call
    without the argument."""
    terminal_locs = np.array([49, 99, 149])
    idxs = np.random.default_rng(2).integers(0, 150, size=500)
    r0, r1 = np.random.default_rng(7), np.random.default_rng(7)
    g0, m0 = hindsight_goal_idxs(idxs, terminal_locs, 150, 0.9, 0.3, rng=r0)
    g1, m1 = hindsight_goal_idxs(idxs, terminal_locs, 150, 0.9, 0.3, rng=r1, cur_frac=0.0)
    np.testing.assert_array_equal(g0, g1)
    np.testing.assert_array_equal(m0, m1)
    assert r0.integers(0, 1 << 30) == r1.integers(0, 1 << 30)


def test_dataset_serves_own_next_state_as_goal_at_cur_frac_one():
    rng = np.random.default_rng(0)
    n = 30
    terminals = np.zeros((n,), np.float32)
    terminals[[9, 19, 29]] = 1.0
    ds = Dataset.create(observations=rng.standard_normal((n, OB)).astype(np.float32),
                        next_observations=rng.standard_normal((n, OB)).astype(np.float32),
                        actions=rng.standard_normal((n, DA)).astype(np.float32),
                        terminals=terminals)
    ds.return_goals = True
    ds.goal_random_frac = 0.0
    ds.goal_cur_frac = 1.0
    b = ds.sample(64)
    np.testing.assert_array_equal(b["goals"], b["next_observations"])


# ------------------------------------------------------------------ goal-indexed measure loss

def test_coef_is_h_of_goal_in_goal_mode_and_w_of_code_otherwise():
    b = _batch()
    g, z = jnp.asarray(b["goals"]), _code(0)
    ag = _agent(**ARMS["sq"])
    np.testing.assert_allclose(ag._coef(z, g), ag.w_star(g), rtol=1e-6)
    np.testing.assert_allclose(ag._coef(z, g, target=True),
                               ag.w_star(g, params=ag.target_w_star), rtol=1e-6)
    ad = _agent()
    np.testing.assert_allclose(ad._coef(z, g), ad.w(z), rtol=1e-6)
    np.testing.assert_allclose(ad._coef(z, g, target=True), ad.w(z, params=ad.target_w), rtol=1e-6)


def test_goal_mode_squared_loss_matches_hand_and_ignores_code():
    """loss = 0.5 mean_{i!=j} (M_ij - discount*Mbar_ij)^2 - (1-discount) mean_i M_ii with
    w_i = h(g_i) online in M and the target h in Mbar; the policy code z does not enter."""
    ag = _agent(**ARMS["sq"])
    ag, _ = ag.update(_batch(seed=1))               # so the targets differ from the online nets
    b = _batch()
    u_next = ag._select_u_next(b, _code(0))
    loss, _ = ag.measure_loss(ag.basis.params, ag.w_star.params, b, _code(0), u_next)
    loss_other_code, _ = ag.measure_loss(ag.basis.params, ag.w_star.params, b, _code(1), u_next)
    assert float(loss) == float(loss_other_code)

    obs, nobs = jnp.asarray(b["observations"]), jnp.asarray(b["next_observations"])
    u = jnp.clip(jnp.asarray(b["noise_preimage"]), -3.0, 3.0)
    g = jnp.asarray(b["goals"])
    w = ag.w_star(g)
    w_t = ag.w_star(g, params=ag.target_w_star)
    M, Mbar = np.zeros((N, N)), np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            M[i, j] = float(ag.M(obs[i:i + 1], u[i:i + 1], nobs[j:j + 1], w[i])[0])
            Mbar[i, j] = float(ag.M(nobs[i:i + 1], u_next[i:i + 1], nobs[j:j + 1], w_t[i],
                                    params=ag.target_basis)[0])
    off = 1.0 - np.eye(N)
    hand = 0.5 * (((M - 0.98 * Mbar) ** 2) * off).sum() / off.sum() - 0.02 * np.diag(M).mean()
    np.testing.assert_allclose(float(loss), hand, rtol=1e-4, atol=1e-6)


def test_goal_mode_update_trains_h_and_polyaks_its_target():
    ag = _agent(**ARMS["sq"])
    new, info = ag.update(_batch())
    assert _changed(ag.w_star.params, new.w_star.params), "h must take the measure-loss gradient"
    assert _changed(ag.basis.params, new.basis.params)
    assert _same(ag.w.params, new.w.params), "the code net w(z) is unused in goal mode"
    assert _same(ag.target_w, new.target_w)
    tau = float(ag.config["tau"])
    for t_old, t_new, online in zip(_leaves(ag.target_w_star), _leaves(new.target_w_star),
                                    _leaves(new.w_star.params)):
        np.testing.assert_allclose(t_new, tau * online + (1.0 - tau) * t_old, rtol=1e-6, atol=1e-7)
    assert "goal_obj" not in info, "goal_head_loss is not used in goal mode"


def test_goal_mode_bootstrap_is_actor_mode_at_measure_goal():
    ag = _agent(**ARMS["sq"])
    ag, _ = ag.update(_batch(seed=1))
    b = _batch()
    nobs = jnp.asarray(b["next_observations"])
    u_sel = ag._select_u_next(b, _code(0))
    mu, _ = ag.actor(nobs, jnp.asarray(b["goals"]))
    u_goal = jnp.clip(3.0 * jnp.tanh(mu), -3.0, 3.0)
    mu_fb, _ = ag.actor(nobs, jnp.asarray(b["fb_goals"]))
    np.testing.assert_allclose(u_sel, u_goal, atol=1e-6)
    assert not np.allclose(u_sel, 3.0 * jnp.tanh(mu_fb)), "bootstrap must use the measure goal"
    np.testing.assert_allclose(u_sel, ag._select_u_next(b, _code(1)), atol=0.0)


# ------------------------------------------------------------------ softmax loss

def _np_softmax(x):
    e = np.exp(x - x.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


@pytest.mark.parametrize("temp", [1.0, 0.5])
def test_softmax_td_loss_matches_numpy(temp):
    rng = np.random.default_rng(0)
    M = rng.standard_normal((5, 5)).astype(np.float32) * 2.0
    Mbar = rng.standard_normal((5, 5)).astype(np.float32) * 2.0
    target = 0.02 * np.eye(5) + 0.98 * _np_softmax(Mbar / temp)
    want = -(target * np.log(_np_softmax(M / temp))).sum(1).mean()
    got = pg.softmax_td_loss(jnp.asarray(M), jnp.asarray(Mbar), 0.98, temp)
    np.testing.assert_allclose(float(got), want, rtol=1e-5)


def test_softmax_td_loss_is_invariant_to_a_per_row_shift():
    rng = np.random.default_rng(1)
    M = jnp.asarray(rng.standard_normal((5, 5)), jnp.float32)
    Mbar = jnp.asarray(rng.standard_normal((5, 5)), jnp.float32)
    shift = jnp.asarray(rng.standard_normal((5, 1)) * 3.0, jnp.float32)
    base = float(pg.softmax_td_loss(M, Mbar, 0.98, 1.0))
    np.testing.assert_allclose(float(pg.softmax_td_loss(M + shift, Mbar, 0.98, 1.0)), base, rtol=1e-5)
    np.testing.assert_allclose(float(pg.softmax_td_loss(M, Mbar - shift, 0.98, 1.0)), base, rtol=1e-5)


def test_softmax_td_loss_gives_the_target_no_gradient():
    rng = np.random.default_rng(2)
    M = jnp.asarray(rng.standard_normal((4, 4)), jnp.float32)
    Mbar = jnp.asarray(rng.standard_normal((4, 4)), jnp.float32)
    g = jax.grad(lambda mb: pg.softmax_td_loss(M, mb, 0.98, 1.0))(Mbar)
    assert float(jnp.max(jnp.abs(g))) == 0.0


def test_softmax_measure_loss_is_softmax_td_on_the_mesh():
    ag = _agent(**ARMS["sm"], measure_temp=0.7)
    ag, _ = ag.update(_batch(seed=1))
    b = _batch()
    u_next = ag._select_u_next(b, _code(0))
    loss, info = ag.measure_loss(ag.basis.params, ag.w_star.params, b, _code(0), u_next)
    obs, nobs = jnp.asarray(b["observations"]), jnp.asarray(b["next_observations"])
    u = jnp.clip(jnp.asarray(b["noise_preimage"]), -3.0, 3.0)
    g = jnp.asarray(b["goals"])
    M = ag.mesh_M(obs, u, nobs, ag.w_star(g))
    Mbar = ag.mesh_M(nobs, u_next, nobs, ag.w_star(g, params=ag.target_w_star),
                     params=ag.target_basis)
    np.testing.assert_allclose(float(loss), float(pg.softmax_td_loss(M, Mbar, 0.98, 0.7)), rtol=1e-5)
    p = _np_softmax(np.asarray(M) / 0.7)
    np.testing.assert_allclose(float(info["m_softmax_max"]), p.max(1).mean(), rtol=1e-4)
    spread = (np.asarray(M).max(1) - np.asarray(M).min(1)) / 0.7
    np.testing.assert_allclose(float(info["m_logit_spread"]), spread.mean(), rtol=1e-4)


# ------------------------------------------------------------------ actor

@pytest.mark.parametrize("arm", ["sq", "sm"])
def test_actor_loss_goal_matches_hand(arm):
    """Q_i = sum_j M(s_i,u_i,s+_j) k(g_i,s+_j) (squared run) or the same with
    softmax_j(M/measure_temp) in place of M (softmax run), M at w_i = h(g_i), u the actor's
    draw at (s_i, g_i). loss = -q_coeff*mean Q/|Q| + actor_temp*mean logp + fb_bc_coeff*bc."""
    ag = _agent(**ARMS[arm], measure_temp=0.7, actor_temp=0.1, q_coeff=2.0, fb_bc_coeff=3.0)
    b = _batch()
    obs, pool = jnp.asarray(b["observations"]), jnp.asarray(b["next_observations"])
    g = jnp.asarray(b["fb_goals"])
    noise = jnp.asarray(np.random.default_rng(3).standard_normal((N, DA)), jnp.float32)
    u_data = jnp.asarray(b["noise_preimage"])
    loss, info = ag.actor_loss_goal(ag.actor.params, obs, pool, g, noise, u_data)

    mu, ls = ag.actor(obs, g)
    u, logp = tanh_gaussian_sample(mu, ls, noise, 3.0)
    w = ag.w_star(g)
    M = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            M[i, j] = float(ag.M(obs[i:i + 1], u[i:i + 1], pool[j:j + 1], w[i])[0])
    k = np.asarray(pg.kernel_reward(g, pool, pg.parse_hit_spec("0:3:0.5")))
    q = (M * k).sum(1) if arm == "sq" else (_np_softmax(M / 0.7) * k).sum(1)
    bc = float(jnp.mean((u - u_data) ** 2))
    hand = -2.0 * q.mean() / np.abs(q).mean() + 0.1 * float(jnp.mean(logp)) + 3.0 * bc
    np.testing.assert_allclose(float(info["actor_q"]), q.mean(), rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(float(info["actor_bc_err"]), bc, rtol=1e-5)
    np.testing.assert_allclose(float(loss), hand, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("arm", ["sq", "sm"])
def test_actor_loss_goal_moves_only_the_actor(arm):
    ag = _agent(**ARMS[arm])
    b = _batch()
    args = (jnp.asarray(b["observations"]), jnp.asarray(b["next_observations"]),
            jnp.asarray(b["fb_goals"]),
            jnp.asarray(np.random.default_rng(4).standard_normal((N, DA)), jnp.float32),
            jnp.asarray(b["noise_preimage"]))

    def of_basis(bp):
        a = ag.replace(basis=ag.basis.replace(params=bp))
        return a.actor_loss_goal(a.actor.params, *args)[0]

    def of_h(hp):
        a = ag.replace(w_star=ag.w_star.replace(params=hp))
        return a.actor_loss_goal(a.actor.params, *args)[0]

    for g in (jax.grad(of_basis)(ag.basis.params), jax.grad(of_h)(ag.w_star.params)):
        assert all(float(jnp.max(jnp.abs(x))) == 0.0 for x in _leaves(g))
    ga = jax.grad(lambda ap: ag.actor_loss_goal(ap, *args)[0])(ag.actor.params)
    assert any(float(jnp.max(jnp.abs(x))) > 0.0 for x in _leaves(ga))


def _actor_input_widths(agent):
    """Input widths of the actor's first-layer kernels (one takes s, the other [s, input])."""
    return sorted({int(x.shape[0]) for x in _leaves(agent.actor.params) if x.ndim == 2})


def test_actor_input_width_follows_actor_input():
    assert 2 * OB in _actor_input_widths(_agent(**ARMS["sq"]))            # [s, g]
    assert OB + Z not in _actor_input_widths(_agent(**ARMS["sq"]))
    assert OB + Z in _actor_input_widths(_agent())                          # [s, w]
    mixed = _agent(policy_index="goal", train_actor=True)                   # goal index, actor fed h(g)
    assert OB + Z in _actor_input_widths(mixed)
    new, info = mixed.update(_batch())
    assert np.isfinite(float(info["actor_q"])) and _changed(mixed.actor.params, new.actor.params)


@pytest.mark.parametrize("arm", ["sq", "sm"])
def test_goal_mode_update_is_finite_and_steps_the_actor(arm):
    ag = _agent(**ARMS[arm])
    for step in range(2):
        new, info = ag.update(_batch(seed=step))
        for k in ("psm_loss", "m_diag_mean", "m_offdiag_mean", "actor_loss", "actor_q",
                  "actor_bc_err"):
            assert np.isfinite(float(info[k])), k
        assert _changed(ag.actor.params, new.actor.params)
        ag = new
    assert ("m_softmax_max" in info) == (arm == "sm")
    loss, _ = ag.total_loss(_batch(seed=9))
    assert np.isfinite(float(loss))


# ------------------------------------------------------------------ test-time readouts

def _trained_run(arm, tmp_path):
    ag = _agent(**ARMS[arm])
    for step in range(2):
        ag, _ = ag.update(_batch(seed=step))
    run_dir = tmp_path / f"run_{arm}"
    run_dir.mkdir()
    save_agent(ag, str(run_dir), 2)
    return ag, str(run_dir)


@pytest.mark.parametrize("arm", ["sq", "sm"])
@pytest.mark.parametrize("readout", list(READOUTS))
def test_four_readouts_restore_and_give_finite_actions(arm, readout, tmp_path):
    trained, run_dir = _trained_run(arm, tmp_path)
    ev = restore_agent(_agent(seed=3, **{**ARMS[arm], **READOUTS[readout]}), run_dir, 2)
    assert _same(ev.actor.params, trained.actor.params), "the trained actor must be restored"
    assert _same(ev.w_star.params, trained.w_star.params)
    b, rew = _relabel()
    env_goal = np.random.default_rng(11).standard_normal((OB,)).astype(np.float32)
    ev = ev.infer_eval_goals(b, rew, env_goal=env_goal)
    obs0 = jnp.asarray(b["observations"][0])
    a = ev.sample_actions(obs0, seed=jax.random.PRNGKey(0))
    assert a.shape == (DA,) and np.all(np.isfinite(np.asarray(a)))

    rewarding = b["next_observations"][:6]
    if readout == "hgoal":
        want = ev._project(jnp.mean(ev.w_star(ev.eval_goals), axis=0))
        np.testing.assert_allclose(ev.eval_w_star, want, atol=1e-5)
    if readout == "lp":
        np.testing.assert_allclose(float(jnp.linalg.norm(ev.eval_w_star)), np.sqrt(Z), rtol=1e-4)
    if readout.startswith("actor"):
        # no coefficient is inferred; the actor takes one goal state
        assert float(jnp.max(jnp.abs(ev.eval_w_star))) == 0.0
        goal = np.asarray(ev.eval_actor_goal)
        if readout == "actor_env":
            np.testing.assert_array_equal(goal, env_goal)
        else:
            assert any(np.array_equal(goal, r) for r in rewarding)
        mu, _ = ev.actor(obs0[None], jnp.asarray(goal)[None])
        want = ev.decode(obs0[None], 3.0 * jnp.tanh(mu))[0]
        np.testing.assert_allclose(a, want, atol=1e-6)


def test_env_goal_readout_needs_the_env_goal():
    ev = _agent(**{**ARMS["sq"], **READOUTS["actor_env"]})
    b, rew = _relabel()
    with pytest.raises(AssertionError):
        ev.infer_eval_goals(b, rew)


def test_softmax_readout_stores_non_rewarding_reference_columns():
    ev = _agent(**{**ARMS["sm"], **READOUTS["hgoal"]})
    b, rew = _relabel()
    ev = ev.infer_eval_goals(b, rew)
    refs = np.asarray(ev.eval_ref)
    assert refs.shape == (N - 4, OB)                       # batch_size - k_goals
    non_rewarding = b["next_observations"][6:]
    assert all(any(np.array_equal(r, x) for x in non_rewarding) for r in refs)


def test_softmax_goal_score_is_mass_on_goal_columns():
    """Score = sum over goal columns of softmax_j(M/measure_temp) over [goals + references];
    select_latent returns the candidate with the largest score."""
    ev = _agent(**{**ARMS["sm"], **READOUTS["hgoal"]}, measure_temp=0.7)
    b, rew = _relabel()
    ev = ev.infer_eval_goals(b, rew)
    obs0 = jnp.asarray(b["observations"][0])
    seed = jax.random.PRNGKey(4)
    K = int(ev.config["gpi_num_u"])
    u_cand = jnp.clip(jax.random.normal(seed, (K, DA)), -3.0, 3.0)
    obs = jnp.broadcast_to(obs0, (K, OB))
    w = jnp.broadcast_to(ev.eval_w_star, (K, Z))
    cols = jnp.concatenate([ev.eval_goals, ev.eval_ref], axis=0)
    p = _np_softmax(np.asarray(ev.mesh_M(obs, u_cand, cols, w)) / 0.7)
    want = p[:, :ev.eval_goals.shape[0]].sum(1)
    got = ev._goal_score(obs, u_cand, ev.eval_goals, ev.eval_ref, w)
    np.testing.assert_allclose(got, want, rtol=1e-4)
    np.testing.assert_array_equal(ev.select_latent(obs0, seed), u_cand[int(np.argmax(want))])


def test_squared_goal_score_is_mean_over_goal_columns():
    ev = _agent()
    b, rew = _relabel()
    ev = ev.infer_eval_goals(b, rew)
    obs = jnp.asarray(b["observations"][:5])
    u = jnp.asarray(b["noise_preimage"][:5])
    w = jnp.broadcast_to(ev.eval_w_star, (5, Z))
    want = jnp.mean(ev.mesh_M(obs, u, ev.eval_goals, w), axis=1)
    np.testing.assert_array_equal(ev._goal_score(obs, u, ev.eval_goals, ev.eval_ref, w), want)


def test_softmax_lagrangian_maximises_the_mass_with_the_penalty_off():
    """Under measure_loss=softmax the inference objective is the mean softmax mass on the goal
    columns, and the multiplier is not stepped (the mass is non-negative for every w)."""
    ev = _agent(**{**ARMS["sm"], **READOUTS["lp"]})
    b, _ = _relabel()
    obs = jnp.asarray(b["observations"][:N])
    u = jnp.asarray(b["noise_preimage"][:N])
    goals = jnp.asarray(b["next_observations"][:4])
    refs = jnp.asarray(b["next_observations"][6:10])
    st = ev.init_inference(jax.random.PRNGKey(0))
    st2, info = ev.inference_step(st, obs, u, goals, jax.random.PRNGKey(1), refs=refs)
    w_rows = jnp.broadcast_to(st.w, (N, Z))
    want = jnp.mean(ev._goal_score(obs, u, goals, refs, w_rows))
    np.testing.assert_allclose(float(info["obj"]), float(want), rtol=1e-5)
    assert _same(st.l_params, st2.l_params), "the multiplier must not move under softmax"
    assert _changed(st.w, st2.w)


# ------------------------------------------------------------------ create asserts

@pytest.mark.parametrize("over", [
    {**GC, "train_goal_head": True},                                   # goal index + goal head
    {**GC, "actor_value": "measure_reward_raw"},                       # goal index + actor_value
    {"measure_loss": "softmax", "measure_form": "factorized"},
    {"measure_loss": "softmax", "ortho_coef": 1.0},
    {**GC, "acting": "distill"},                                       # goal actor + eval re-distillation
    {**GC, "acting": "sfbc"},
    {"policy_index": "goal"},                                          # goal index without train_actor
    {"actor_input": "goal", "train_actor": True},                      # goal actor without the goal index
    {"measure_loss": "softmax", "k_goals": N},                         # no room for reference columns
    {"policy_index": "task"}, {"measure_loss": "ce"}, {"actor_input": "state"},
    {"eval_goal_source": "dataset"},
])
def test_create_asserts_fire(over):
    with pytest.raises(AssertionError):
        _agent(**over)


def test_spec_run_configs_build():
    """The two training arms, and each of their four readouts, pass every assert."""
    for arm in ARMS.values():
        _agent(**arm)
        for readout in READOUTS.values():
            _agent(**{**arm, **readout})

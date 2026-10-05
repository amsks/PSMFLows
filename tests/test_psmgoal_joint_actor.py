"""psmgoal, joint actor training inside the goal-conditioned measure (2026-10-05).

The actor supplies the bootstrap move (bootstrap_source=actor, policy_index=goal). Two new
actor heads (`actor_kind`) and a second value the actor climbs (`actor_value_kind`):

- flowbc: u = clip(eps + delta(s, g, eps)), eps a prior draw; loss -q_coeff*Q/sg|Q| +
  fb_bc_coeff*mean delta^2 (anchored at eps, which is itself a BC sample). Deterministic
  given eps; the bootstrap and the acting draw a fresh eps each.
- dsrl: the tanh-Gaussian actor with no BC term, -q_coeff*Q/sg|Q| + alpha*logp, alpha
  learned toward target entropy 0 (the f_psmflow dsrl_sac head). Mode in bootstrap and acting.
- actor_value_kind=point: M(s,u,g) at the actor goal with w = h(g); softmax: the share of g
  among [g + the batch's next states].

Checks: defaults unchanged (both configs; the digest file covers bit-identity); the flowbc
actor returns eps exactly when delta is zero and its BC term is zero there; the BC gradient
pulls delta to zero; the value gradient reaches delta only; the point value equals a hand
computation for both losses; `_select_u_next` uses the selected actor at (s', measure goal);
the dsrl path has no BC term and its alpha moves; each kind updates with finite stats; acting
draws one eps per step (flowbc) or the mode (dsrl); the asserts fire.

Run in its own process: JAX_PLATFORMS=cpu .venv/bin/python -m pytest
tests/test_psmgoal_joint_actor.py -q -p no:cacheprovider
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
from utils.flax_utils import restore_agent

OB, DA, Z, CODE, N = 6, 3, 8, 4, 8

NEW_KEYS = {"actor_kind": "tanh", "actor_value_kind": "shaped"}

# The joint-actor base: the data-bootstrap goal setup with the actor back in the bootstrap.
JA = {"policy_index": "goal", "bootstrap_source": "actor", "train_actor": True, "actor_input": "goal",
      "goal_random_frac": 0.0, "goal_cur_frac": 0.0, "measure_loss": "softmax",
      "acting": "distill", "eval_redistill": False}
ARMS = {
    "fbc3_sh": {**JA, "actor_kind": "flowbc", "fb_bc_coeff": 3.0, "actor_value_kind": "shaped"},
    "fbc0_sh": {**JA, "actor_kind": "flowbc", "fb_bc_coeff": 0.0, "actor_value_kind": "shaped"},
    "dsrl_sh": {**JA, "actor_kind": "dsrl", "fb_bc_coeff": 0.0, "actor_value_kind": "shaped"},
    "fbc3_pt": {**JA, "actor_kind": "flowbc", "fb_bc_coeff": 3.0, "actor_value_kind": "point"},
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


def _perturbed_delta(ag, bias=0.5):
    """delta with its output bias moved off zero, so delta != 0 everywhere."""
    params = jax.tree_util.tree_map(lambda x: x, ag.delta.params)      # same container types
    name = pg.DELTA_OUT_LAYER
    params[name]["bias"] = jnp.full_like(params[name]["bias"], bias)
    params[name]["kernel"] = 0.1 * jax.random.normal(jax.random.PRNGKey(9), params[name]["kernel"].shape)
    return ag.replace(delta=ag.delta.replace(params=params))


# ------------------------------------------------------------------ defaults

def test_new_keys_default_off_in_get_config_and_yaml():
    c = get_config()
    path = os.path.join(os.path.dirname(__file__), "..", "configs", "agent", "psmgoal.yaml")
    with open(path) as f:
        y = yaml.safe_load(f)
    for k, v in NEW_KEYS.items():
        assert k in c and c[k] == v, f"get_config()[{k!r}]"
        assert k in y and y[k] == v, f"psmgoal.yaml[{k!r}]"
    for k, v in {"target_entropy": 0.0, "init_alpha": 1.0, "lr_alpha": 3.0e-4}.items():
        assert float(c.dsrl_actor[k]) == v
        assert float(y["dsrl_actor"][k]) == v


def test_default_update_has_no_actor_keys_and_leaves_new_heads_untouched():
    ag = _agent()
    d0, a0 = ag.delta.params, ag.log_alpha.params
    ag, info = ag.update(_batch())
    assert not any(k.startswith("actor_") for k in info)
    assert _same(ag.delta.params, d0) and _same(ag.log_alpha.params, a0)


# ------------------------------------------------------------------ flowbc head

def test_flowbc_returns_eps_when_delta_is_zero_and_bc_term_is_zero():
    ag = _agent(**ARMS["fbc3_sh"])
    b = _batch()
    obs, g = jnp.asarray(b["observations"]), jnp.asarray(b["fb_goals"])
    eps = jax.random.normal(jax.random.PRNGKey(3), (N, DA)) * 2.0
    # delta's output layer is zero-initialised, so delta == 0 at init
    u, delta = ag.flowbc_latent(obs, g, eps)
    assert np.allclose(np.asarray(delta), 0.0)
    assert np.allclose(np.asarray(u), np.clip(np.asarray(eps), -3.0, 3.0))
    u_data = jnp.asarray(b["noise_preimage"])
    _, info = ag.actor_loss_flowbc(ag.delta.params, obs, jnp.asarray(b["next_observations"]), g, eps, u_data)
    assert float(info["actor_bc_err"]) == 0.0
    assert float(info["actor_delta_norm"]) == 0.0


def test_flowbc_bc_gradient_pulls_delta_to_zero():
    ag = _perturbed_delta(_agent(q_coeff=0.0, **ARMS["fbc3_sh"]), bias=0.5)
    b = _batch()
    obs, g = jnp.asarray(b["observations"]), jnp.asarray(b["fb_goals"])
    eps = jax.random.normal(jax.random.PRNGKey(3), (N, DA))
    u_data = jnp.asarray(b["noise_preimage"])
    pool = jnp.asarray(b["next_observations"])
    (loss, info), grads = jax.value_and_grad(ag.actor_loss_flowbc, has_aux=True)(
        ag.delta.params, obs, pool, g, eps, u_data)
    bias_grad = np.asarray(grads[pg.DELTA_OUT_LAYER]["bias"])
    # d/d bias of mean delta^2 = 2 mean(delta) / d_a; delta's mean is positive here
    assert np.all(bias_grad > 0.0)                     # the step moves the bias back toward 0
    assert float(info["actor_bc_err"]) > 0.0
    # with q_coeff=0 the loss IS the BC term
    assert np.isclose(float(loss), 3.0 * float(info["actor_bc_err"]), rtol=1e-5)
    # several steps shrink delta
    for i in range(20):
        upd, a_info = ag.goal_actor_step(_batch(seed=i))
        ag = ag.replace(**upd)
    assert float(a_info["actor_bc_err"]) < float(info["actor_bc_err"])


def test_flowbc_value_gradient_reaches_delta_only():
    ag = _perturbed_delta(_agent(**ARMS["fbc0_sh"]), bias=0.3)
    b = _batch()
    obs, g = jnp.asarray(b["observations"]), jnp.asarray(b["fb_goals"])
    eps = jax.random.normal(jax.random.PRNGKey(3), (N, DA))
    u_data = jnp.asarray(b["noise_preimage"])
    pool = jnp.asarray(b["next_observations"])
    g_delta = jax.grad(lambda p: ag.actor_loss_flowbc(p, obs, pool, g, eps, u_data)[0])(ag.delta.params)
    assert any(float(jnp.abs(x).max()) > 0.0 for x in _leaves(g_delta))

    def through(bp, hp):
        a = ag.replace(basis=ag.basis.replace(params=bp), w_star=ag.w_star.replace(params=hp))
        return a.actor_loss_flowbc(a.delta.params, obs, pool, g, eps, u_data)[0]

    g_b, g_h = jax.grad(through, argnums=(0, 1))(ag.basis.params, ag.w_star.params)
    assert all(float(jnp.abs(x).max()) == 0.0 for x in _leaves(g_b))
    assert all(float(jnp.abs(x).max()) == 0.0 for x in _leaves(g_h))


def test_flowbc_update_moves_delta_only_and_logs_stats():
    ag = _agent(**ARMS["fbc3_sh"])
    a0, d0 = ag.actor.params, ag.delta.params
    ag, info = ag.update(_batch())
    assert _changed(ag.delta.params, d0)
    assert _same(ag.actor.params, a0)                 # the tanh head is not trained under flowbc
    for k in ("actor_u_absmean", "actor_delta_norm", "actor_bc_err", "actor_q", "actor_p_own", "psm_loss"):
        assert k in info and np.isfinite(float(info[k])), k
    assert "actor_logp" not in info


# ------------------------------------------------------------------ dsrl head

def test_dsrl_loss_has_no_bc_term_and_alpha_moves():
    ag = _agent(**ARMS["dsrl_sh"])
    b = _batch()
    obs, g = jnp.asarray(b["observations"]), jnp.asarray(b["fb_goals"])
    noise = jax.random.normal(jax.random.PRNGKey(3), (N, DA))
    pool = jnp.asarray(b["next_observations"])
    loss, info = ag.actor_loss_dsrl(ag.actor.params, obs, pool, g, noise)
    alpha = float(jnp.exp(ag.log_alpha()))
    assert alpha == 1.0                                # init_alpha
    expect = -float(info["actor_q"]) / float(info["actor_q_absmean"]) + alpha * float(info["actor_logp"])
    assert np.isclose(float(loss), expect, rtol=1e-5)
    assert "actor_bc_err" not in info
    la0 = ag.log_alpha.params
    ag, info = ag.update(b)
    assert _changed(ag.log_alpha.params, la0)
    assert "actor_alpha" in info and np.isfinite(float(info["actor_alpha"]))
    # alpha loss direction: -(log_alpha * (logp + target)); logp > 0 here moves log_alpha down
    for k in ("actor_u_absmean", "actor_q", "actor_p_own", "actor_logp"):
        assert np.isfinite(float(info[k])), k


def test_dsrl_bootstrap_and_acting_use_the_mode():
    ag = _agent(**ARMS["dsrl_sh"])
    b = _batch()
    z = jnp.zeros((N, CODE))
    u_next = ag._select_u_next(b, z)
    expect = ag.actor_backup(jnp.asarray(b["next_observations"]), jnp.asarray(b["goals"]))
    assert np.allclose(np.asarray(u_next), np.asarray(expect))
    rb, rew = _relabel()
    ag = ag.infer_eval_goals(rb, rew)
    o = jnp.asarray(rb["observations"][0])
    a1 = ag.sample_actions(o, seed=jax.random.PRNGKey(1))
    a2 = ag.sample_actions(o, seed=jax.random.PRNGKey(2))
    assert np.allclose(np.asarray(a1), np.asarray(a2))   # deterministic mode
    assert np.all(np.isfinite(np.asarray(a1)))


# ------------------------------------------------------------------ bootstrap and acting

def test_flowbc_bootstrap_is_the_actor_at_next_state_and_measure_goal():
    ag = _perturbed_delta(_agent(**ARMS["fbc3_sh"]), bias=0.4)
    b = _batch()
    z = jnp.zeros((N, CODE))
    u_next = ag._select_u_next(b, z)
    eps = jax.random.normal(jax.random.fold_in(ag.rng, pg.FLOWBC_BOOT_KEY), (N, DA))
    expect, _ = ag.flowbc_latent(jnp.asarray(b["next_observations"]), jnp.asarray(b["goals"]), eps)
    assert np.allclose(np.asarray(u_next), np.asarray(expect))
    # a different measure goal changes the bootstrap (the actor reads the goal)
    b2 = dict(b)
    b2["goals"] = np.asarray(b["fb_goals"])
    assert not np.allclose(np.asarray(ag._select_u_next(b2, z)), np.asarray(u_next))
    # the loss' own eps at the actor goal, and the measure loss consumes u_next
    loss, _ = ag.measure_loss(ag.basis.params, ag.w_star.params, b, z, u_next)
    assert np.isfinite(float(loss))


def test_flowbc_acting_draws_one_eps_per_step():
    ag = _perturbed_delta(_agent(**ARMS["fbc3_sh"]), bias=0.4)
    rb, rew = _relabel()
    ag = ag.infer_eval_goals(rb, rew)
    o = jnp.asarray(rb["observations"][0])
    a1 = ag.sample_actions(o, seed=jax.random.PRNGKey(1))
    a2 = ag.sample_actions(o, seed=jax.random.PRNGKey(2))
    assert np.all(np.isfinite(np.asarray(a1)))
    assert not np.allclose(np.asarray(a1), np.asarray(a2))
    # the same seed gives the same action: deterministic given eps
    assert np.allclose(np.asarray(a1), np.asarray(ag.sample_actions(o, seed=jax.random.PRNGKey(1))))
    # and equals clip(eps + delta) decoded, eps drawn from the seed
    eps = jax.random.normal(jax.random.PRNGKey(1), (1, DA))
    u, _ = ag.flowbc_latent(o[None], ag.eval_actor_goal[None], eps)
    assert np.allclose(np.asarray(a1), np.asarray(ag.decode(o[None], u)[0]), atol=1e-6)


# ------------------------------------------------------------------ point value

@pytest.mark.parametrize("loss", ["softmax", "squared"])
def test_point_value_matches_hand_computation(loss):
    ag = _agent(**{**ARMS["fbc3_pt"], "measure_loss": loss})
    b = _batch()
    obs, g = jnp.asarray(b["observations"]), jnp.asarray(b["fb_goals"])
    pool = jnp.asarray(b["next_observations"])
    u = jnp.asarray(b["noise_preimage"])
    w = ag.w_star(g)
    q = ag.actor_value(obs, u, pool, g, w, ag.basis.params)
    m_goal = np.asarray(ag.M(obs, u, g, w))                              # (N,) M at the goal
    m_pool = np.asarray(ag.mesh_M(obs, u, pool, w))                      # (N, N)
    if loss == "softmax":
        t = float(ag.config["measure_temp"])
        e_g = np.exp(m_goal / t)
        expect = e_g / (e_g + np.exp(m_pool / t).sum(1))
    else:
        expect = m_goal
    assert np.allclose(np.asarray(q), expect, atol=1e-5)


@pytest.mark.parametrize("loss", ["softmax", "squared"])
def test_shaped_value_matches_hand_computation(loss):
    ag = _agent(**{**ARMS["fbc3_sh"], "measure_loss": loss})
    b = _batch()
    obs, g = jnp.asarray(b["observations"]), jnp.asarray(b["fb_goals"])
    pool = jnp.asarray(b["next_observations"])
    u = jnp.asarray(b["noise_preimage"])
    w = ag.w_star(g)
    q = ag.actor_value(obs, u, pool, g, w, ag.basis.params)
    m_pool = np.asarray(ag.mesh_M(obs, u, pool, w))
    k = np.asarray(pg.kernel_reward(g, pool, pg.parse_hit_spec("0:3:0.5")))
    if loss == "softmax":
        p = np.exp(m_pool - m_pool.max(1, keepdims=True))
        p = p / p.sum(1, keepdims=True)
        expect = (p * k).sum(1)
    else:
        expect = (m_pool * k).sum(1)
    assert np.allclose(np.asarray(q), expect, atol=1e-5)


def test_point_arm_updates_with_finite_stats():
    ag = _agent(**ARMS["fbc3_pt"])
    ag, info = ag.update(_batch())
    for k in ("actor_q", "actor_p_own", "actor_bc_err", "actor_u_absmean"):
        assert np.isfinite(float(info[k])), k


# ------------------------------------------------------------------ tanh head unchanged

def test_tanh_kind_is_the_existing_goal_actor():
    base = {**JA, "actor_kind": "tanh"}
    ag = _agent(**base)
    b = _batch()
    obs, g = jnp.asarray(b["observations"]), jnp.asarray(b["fb_goals"])
    noise = jax.random.normal(jax.random.PRNGKey(3), (N, DA))
    u_data = jnp.asarray(b["noise_preimage"])
    pool = jnp.asarray(b["next_observations"])
    loss, info = ag.actor_loss_goal(ag.actor.params, obs, pool, g, noise, u_data)
    assert "actor_bc_err" in info and np.isfinite(float(loss))
    ag2, _ = ag.update(b)
    assert _changed(ag2.actor.params, ag.actor.params)
    assert _same(ag2.delta.params, ag.delta.params)


# ------------------------------------------------------------------ restore

def test_checkpoint_roundtrip_keeps_delta_and_alpha(tmp_path):
    ag = _agent(**ARMS["fbc3_sh"])
    ag, _ = ag.update(_batch())
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with open(run_dir / "params_10.pkl", "wb") as f:
        pickle.dump({"agent": flax.serialization.to_state_dict(ag)}, f)
    restored = restore_agent(_agent(seed=2, **ARMS["fbc3_sh"]), str(run_dir), 10)
    assert _same(restored.delta.params, ag.delta.params)
    assert _same(restored.log_alpha.params, ag.log_alpha.params)


# ------------------------------------------------------------------ asserts

@pytest.mark.parametrize("over", [
    {"actor_kind": "nope"},
    {"actor_value_kind": "nope"},
    {**ARMS["fbc3_sh"], "policy_index": "code", "actor_input": "w"},   # flowbc needs policy_index=goal
    {**ARMS["fbc3_sh"], "actor_input": "w"},                           # flowbc takes the goal
    {**ARMS["fbc3_sh"], "train_actor": False, "bootstrap_source": "data"},  # flowbc trains an actor
    {**ARMS["dsrl_sh"], "fb_bc_coeff": 3.0},                            # dsrl has no BC term
    {"actor_value_kind": "point"},                                      # point needs policy_index=goal
    {**ARMS["fbc3_sh"], "eval_redistill": True},                        # goal actor cannot be redistilled
])
def test_create_asserts_fire(over):
    with pytest.raises(AssertionError):
        _agent(**over)


@pytest.mark.parametrize("name", sorted(ARMS))
def test_each_arm_creates_and_updates(name):
    ag = _agent(**ARMS[name])
    ag, info = ag.update(_batch())
    assert np.isfinite(float(info["psm_loss"]))
    assert np.isfinite(float(info["actor_q"]))

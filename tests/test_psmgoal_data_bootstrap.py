"""psmgoal, goal-conditioned measure bootstrapped from the data's next latent
(bootstrap_source=data) and the per-goal readout coef_source=hgoal_each.

Spec: docs/design/2026-10-03-psmgoal-data-bootstrap.md. Checks: the new keys default off in
both configs; `Dataset.return_next_preimage` serves row i+1's latent; terminal rows and rows
whose next latent is invalid are never sampled, and the default sampler is bit-identical;
`_select_u_next` returns the next latent; the goal-mode measure loss with the data bootstrap
equals a hand computation (both losses); no actor update happens; `hgoal_each` equals a hand
computation (both losses) and reduces to `amortized` when all goals are identical; the
`create` asserts fire.

Run in its own process: JAX_PLATFORMS=cpu .venv/bin/python -m pytest
tests/test_psmgoal_data_bootstrap.py -q -p no:cacheprovider
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml

import agents.psmgoal as pg
from agents.psmgoal import PSMGoalAgent, get_config
from utils.datasets import Dataset
from utils.flax_utils import restore_agent, save_agent
from utils.flow_inversion import PREIMAGE_VALID_KEY

OB, DA, Z, CODE, N = 6, 3, 8, 4, 8

NEW_KEYS = {"bootstrap_source": "actor"}

# The two training arms of the spec's "Runs" table.
DB = {"policy_index": "goal", "bootstrap_source": "data", "train_actor": False,
      "goal_random_frac": 0.0, "goal_cur_frac": 0.0}
ARMS = {"sq": {**DB, "measure_loss": "squared"}, "sm": {**DB, "measure_loss": "softmax"}}

READOUTS = {
    "hgoal": {"coef_source": "amortized", "acting": "gpi"},
    "hgoal_each": {"coef_source": "hgoal_each", "acting": "gpi"},
    "lp": {"coef_source": "lp", "acting": "gpi"},
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
        "next_noise_preimage": rng.standard_normal((n, DA)).astype(np.float32),
        "goals": rng.standard_normal((n, OB)).astype(np.float32),
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


def _np_softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


# ------------------------------------------------------------------ defaults

def test_new_key_defaults_off_in_get_config_and_yaml():
    c = get_config()
    path = os.path.join(os.path.dirname(__file__), "..", "configs", "agent", "psmgoal.yaml")
    with open(path) as f:
        y = yaml.safe_load(f)
    for k, v in NEW_KEYS.items():
        assert k in c and c[k] == v, f"get_config()[{k!r}]"
        assert k in y and y[k] == v, f"psmgoal.yaml[{k!r}]"
    assert "hgoal_each" not in pg.COEF_SOURCES          # eval-only, goal-mode source
    assert "hgoal_each" in pg.GOAL_COEF_SOURCES


def test_dataset_flag_defaults_off():
    ds = _dataset(n=12, ends=(5, 11))
    assert ds.return_next_preimage is False
    b = ds.sample(4)
    assert "next_noise_preimage" not in b


# ------------------------------------------------------------------ dataset

def _dataset(n=40, ends=(9, 19, 29, 39), invalid=()):
    rng = np.random.default_rng(0)
    terminals = np.zeros((n,), np.float32)
    terminals[list(ends)] = 1.0
    valid = np.ones((n,), np.float32)
    valid[list(invalid)] = 0.0
    point = rng.standard_normal((n, DA)).astype(np.float32)
    point[list(invalid)] = 0.0
    fields = {"observations": rng.standard_normal((n, OB)).astype(np.float32),
              "next_observations": rng.standard_normal((n, OB)).astype(np.float32),
              "actions": rng.standard_normal((n, DA)).astype(np.float32),
              "terminals": terminals, "noise_preimage_point": point}
    fields[PREIMAGE_VALID_KEY] = valid
    ds = Dataset.create(**fields)
    ds.return_preimage_noise = True
    ds.preimage_point_mode = True
    return ds


def test_next_noise_preimage_is_row_plus_one_latent_inside_trajectories():
    ds = _dataset()
    ds.return_next_preimage = True
    idxs = np.array([0, 3, 8, 10, 25, 38])
    b = ds.get_subset(idxs)
    np.testing.assert_array_equal(b["next_noise_preimage"], ds["noise_preimage_point"][idxs + 1])
    np.testing.assert_array_equal(b["noise_preimage"], ds["noise_preimage_point"][idxs])


def test_sampler_skips_terminal_rows_and_rows_with_an_invalid_next_latent():
    ds = _dataset(invalid=(5, 22))
    ds.return_next_preimage = True
    idxs = ds.get_random_idxs(5000)
    assert idxs.min() >= 0 and idxs.max() < 40
    seen = set(idxs.tolist())
    for bad in (9, 19, 29, 39):           # trajectory ends: no next row
        assert bad not in seen
    for bad in (4, 21):                   # the next row's latent is invalid (0)
        assert bad not in seen
    for bad in (5, 22):                   # the row's own latent is invalid, as before
        assert bad not in seen
    allowed = set(range(40)) - {9, 19, 29, 39, 4, 21, 5, 22}
    assert seen == allowed                # every other row is drawn
    b = ds.sample(64)
    assert np.all(np.abs(b["next_noise_preimage"]).sum(-1) > 0)


def test_default_sampler_is_bit_identical_with_the_flag_off():
    ds0, ds1 = _dataset(invalid=(5,)), _dataset(invalid=(5,))
    np.random.seed(3)
    b0 = ds0.sample(64)
    np.random.seed(3)
    b1 = ds1.sample(64)
    for k in b0:
        np.testing.assert_array_equal(b0[k], b1[k])
    assert "next_noise_preimage" not in b0
    seen = set(ds0.get_random_idxs(5000).tolist())
    assert {9, 19, 29, 39, 4} <= seen           # the default sampler still draws terminal rows
    assert 5 not in seen                        # and still skips the invalid row


def test_sampler_feeds_the_hindsight_goal_draw_rows_with_a_future():
    """With goals on, every sampled row's goal is its own s' or a later state of the same
    trajectory (goal_random_frac=0, goal_cur_frac=0), and the next latent rides along."""
    ds = _dataset(invalid=(5, 22))
    ds.return_next_preimage = True
    ds.return_goals = True
    ds.goal_random_frac = 0.0
    ds.goal_cur_frac = 0.0
    ds.goal_discount = 0.9
    np.random.seed(11)
    idxs = ds.get_random_idxs(256)
    b = ds.get_subset(idxs)
    ends = np.array([9, 19, 29, 39])
    end_of = ends[np.searchsorted(ends, idxs)]
    nobs = np.asarray(ds["next_observations"])
    for i, (row, e) in enumerate(zip(idxs, end_of)):
        cands = nobs[row:e + 1]
        assert any(np.array_equal(b["goals"][i], c) for c in cands)
    np.testing.assert_array_equal(b["next_noise_preimage"], ds["noise_preimage_point"][idxs + 1])


# ------------------------------------------------------------------ bootstrap and loss

def test_bootstrap_is_the_next_latent_under_data_source():
    ag = _agent(**ARMS["sq"])
    b = _batch()
    u_sel = ag._select_u_next(b, _code(0))
    np.testing.assert_array_equal(u_sel, np.clip(b["next_noise_preimage"], -3.0, 3.0))
    np.testing.assert_array_equal(u_sel, ag._select_u_next(b, _code(1)))
    # the actor path (the 2026-10-01 arm) is untouched
    ac = _agent(policy_index="goal", train_actor=True, actor_input="goal")
    mu, _ = ac.actor(jnp.asarray(b["next_observations"]), jnp.asarray(b["goals"]))
    np.testing.assert_allclose(ac._select_u_next(b, _code(0)), jnp.clip(3.0 * jnp.tanh(mu), -3.0, 3.0),
                               atol=1e-6)


def test_data_bootstrap_squared_loss_matches_hand():
    ag = _agent(**ARMS["sq"])
    ag, _ = ag.update(_batch(seed=1))
    b = _batch()
    u_next = ag._select_u_next(b, _code(0))
    loss, info = ag.measure_loss(ag.basis.params, ag.w_star.params, b, _code(0), u_next)
    obs, nobs = jnp.asarray(b["observations"]), jnp.asarray(b["next_observations"])
    u = jnp.clip(jnp.asarray(b["noise_preimage"]), -3.0, 3.0)
    u1 = jnp.clip(jnp.asarray(b["next_noise_preimage"]), -3.0, 3.0)
    g = jnp.asarray(b["goals"])
    w, w_t = ag.w_star(g), ag.w_star(g, params=ag.target_w_star)
    M, Mbar = np.zeros((N, N)), np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            M[i, j] = float(ag.M(obs[i:i + 1], u[i:i + 1], nobs[j:j + 1], w[i])[0])
            Mbar[i, j] = float(ag.M(nobs[i:i + 1], u1[i:i + 1], nobs[j:j + 1], w_t[i],
                                    params=ag.target_basis)[0])
    off = 1.0 - np.eye(N)
    hand = 0.5 * (((M - 0.98 * Mbar) ** 2) * off).sum() / off.sum() - 0.02 * np.diag(M).mean()
    np.testing.assert_allclose(float(loss), hand, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(float(info["td_target_absmean"]), np.abs(Mbar).mean(), rtol=1e-4)


def test_data_bootstrap_softmax_loss_matches_hand():
    ag = _agent(**ARMS["sm"], measure_temp=0.7)
    ag, _ = ag.update(_batch(seed=1))
    b = _batch()
    u_next = ag._select_u_next(b, _code(0))
    loss, _ = ag.measure_loss(ag.basis.params, ag.w_star.params, b, _code(0), u_next)
    obs, nobs = jnp.asarray(b["observations"]), jnp.asarray(b["next_observations"])
    u = jnp.clip(jnp.asarray(b["noise_preimage"]), -3.0, 3.0)
    u1 = jnp.clip(jnp.asarray(b["next_noise_preimage"]), -3.0, 3.0)
    g = jnp.asarray(b["goals"])
    M = np.asarray(ag.mesh_M(obs, u, nobs, ag.w_star(g)))
    Mbar = np.asarray(ag.mesh_M(nobs, u1, nobs, ag.w_star(g, params=ag.target_w_star),
                                params=ag.target_basis))
    target = 0.02 * np.eye(N) + 0.98 * _np_softmax(Mbar / 0.7)
    hand = -(target * np.log(_np_softmax(M / 0.7))).sum(1).mean()
    np.testing.assert_allclose(float(loss), hand, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("arm", ["sq", "sm"])
def test_update_trains_measure_and_h_but_never_the_actor(arm):
    ag = _agent(**ARMS[arm])
    for step in range(2):
        new, info = ag.update(_batch(seed=step))
        assert _same(ag.actor.params, new.actor.params), "no actor step under bootstrap_source=data"
        assert not any(k.startswith("actor") for k in info), sorted(info)
        assert _changed(ag.basis.params, new.basis.params)
        assert _changed(ag.w_star.params, new.w_star.params)
        assert _same(ag.w.params, new.w.params)
        for k in ("psm_loss", "m_diag_mean", "m_offdiag_mean", "td_target_absmean"):
            assert np.isfinite(float(info[k])), k
        ag = new
    loss, _ = ag.total_loss(_batch(seed=9))
    assert np.isfinite(float(loss))


def test_update_does_not_need_fb_goals():
    ag = _agent(**ARMS["sq"])
    b = _batch()
    b.pop("fb_goals", None)
    _, info = ag.update(b)
    assert np.isfinite(float(info["psm_loss"]))


# ------------------------------------------------------------------ hgoal_each readout

def _trained(arm, tmp_path, n_steps=2):
    ag = _agent(**ARMS[arm])
    for step in range(n_steps):
        ag, _ = ag.update(_batch(seed=step))
    run_dir = tmp_path / f"run_{arm}"
    run_dir.mkdir()
    save_agent(ag, str(run_dir), n_steps)
    return ag, str(run_dir)


@pytest.mark.parametrize("arm", ["sq", "sm"])
@pytest.mark.parametrize("readout", list(READOUTS))
def test_readouts_restore_and_give_finite_actions(arm, readout, tmp_path):
    trained, run_dir = _trained(arm, tmp_path)
    ev = restore_agent(_agent(seed=3, **{**ARMS[arm], **READOUTS[readout]}), run_dir, 2)
    assert _same(ev.w_star.params, trained.w_star.params)
    assert _same(ev.basis.params, trained.basis.params)
    b, rew = _relabel()
    ev = ev.infer_eval_goals(b, rew)
    obs0 = jnp.asarray(b["observations"][0])
    a = ev.sample_actions(obs0, seed=jax.random.PRNGKey(0))
    assert a.shape == (DA,) and np.all(np.isfinite(np.asarray(a)))
    if readout == "hgoal":
        want = ev._project(jnp.mean(ev.w_star(ev.eval_goals), axis=0))
        np.testing.assert_allclose(ev.eval_w_star, want, atol=1e-5)
    if readout == "hgoal_each":
        np.testing.assert_allclose(ev.eval_goal_w, ev.w_star(ev.eval_goals), atol=1e-6)


def test_hgoal_each_squared_score_matches_hand():
    """score(u) = mean_j M(s,u,g_j; h(g_j))."""
    ev = _agent(**{**ARMS["sq"], **READOUTS["hgoal_each"]})
    ev, _ = ev.update(_batch(seed=1))
    b, rew = _relabel()
    ev = ev.infer_eval_goals(b, rew)
    obs0 = jnp.asarray(b["observations"][0])
    K = int(ev.config["gpi_num_u"])
    seed = jax.random.PRNGKey(4)
    u_cand = jnp.clip(jax.random.normal(seed, (K, DA)), -3.0, 3.0)
    obs = jnp.broadcast_to(obs0, (K, OB))
    H = ev.w_star(ev.eval_goals)                                    # (k_goals, Z)
    G = ev.eval_goals.shape[0]
    hand = np.zeros((K,))
    for m in range(K):
        for j in range(G):
            hand[m] += float(ev.M(obs[m:m + 1], u_cand[m:m + 1], ev.eval_goals[j:j + 1], H[j])[0]) / G
    got = ev._hgoal_each_score(obs, u_cand, ev.eval_goals, H, ev.eval_ref)
    np.testing.assert_allclose(np.asarray(got), hand, rtol=1e-4, atol=1e-6)
    np.testing.assert_array_equal(ev.select_latent(obs0, seed), u_cand[int(np.argmax(hand))])


def test_hgoal_each_softmax_score_matches_hand():
    """For each j: the share of g_j among [g_j + refs] with w = h(g_j); averaged over j."""
    ev = _agent(**{**ARMS["sm"], **READOUTS["hgoal_each"]}, measure_temp=0.7)
    ev, _ = ev.update(_batch(seed=1))
    b, rew = _relabel()
    ev = ev.infer_eval_goals(b, rew)
    obs0 = jnp.asarray(b["observations"][0])
    K = int(ev.config["gpi_num_u"])
    seed = jax.random.PRNGKey(4)
    u_cand = jnp.clip(jax.random.normal(seed, (K, DA)), -3.0, 3.0)
    obs = jnp.broadcast_to(obs0, (K, OB))
    H = ev.w_star(ev.eval_goals)
    G, R = ev.eval_goals.shape[0], ev.eval_ref.shape[0]
    assert R == N - 4
    hand = np.zeros((K,))
    for m in range(K):
        for j in range(G):
            logits = np.zeros((1 + R,))
            logits[0] = float(ev.M(obs[m:m + 1], u_cand[m:m + 1], ev.eval_goals[j:j + 1], H[j])[0])
            for r in range(R):
                logits[1 + r] = float(ev.M(obs[m:m + 1], u_cand[m:m + 1], ev.eval_ref[r:r + 1], H[j])[0])
            hand[m] += _np_softmax(logits / 0.7)[0] / G
    got = ev._hgoal_each_score(obs, u_cand, ev.eval_goals, H, ev.eval_ref)
    np.testing.assert_allclose(np.asarray(got), hand, rtol=1e-4, atol=1e-6)
    np.testing.assert_array_equal(ev.select_latent(obs0, seed), u_cand[int(np.argmax(hand))])


def test_hgoal_each_chunked_equals_unchunked():
    ev = _agent(**{**ARMS["sm"], **READOUTS["hgoal_each"]})
    ev, _ = ev.update(_batch(seed=1))
    rng = np.random.default_rng(6)
    goals = jnp.asarray(rng.standard_normal((11, OB)), jnp.float32)
    refs = jnp.asarray(rng.standard_normal((5, OB)), jnp.float32)
    obs = jnp.asarray(rng.standard_normal((7, OB)), jnp.float32)
    u = jnp.asarray(rng.standard_normal((7, DA)), jnp.float32)
    H = ev.w_star(goals)
    full = ev._hgoal_each_score(obs, u, goals, H, refs)
    chunked = ev._hgoal_each_score(obs, u, goals, H, refs, chunk=4)
    np.testing.assert_allclose(np.asarray(chunked), np.asarray(full), rtol=1e-5, atol=1e-7)
    sq = ev.replace(config=ev.config.copy({"measure_loss": "squared"}))
    full_sq = sq._hgoal_each_score(obs, u, goals, H, refs)
    chunked_sq = sq._hgoal_each_score(obs, u, goals, H, refs, chunk=4)
    np.testing.assert_allclose(np.asarray(chunked_sq), np.asarray(full_sq), rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("arm", ["sq", "sm"])
def test_hgoal_each_reduces_to_amortized_when_all_goals_are_identical(arm):
    """One goal repeated: h(g_j) = h(g) for all j, and the amortized w is h(g). Squared: the
    two scores are equal. Softmax: the per-goal share counts the goal column once where the
    amortized share counts it k_goals times, a monotone map of the same odds, so the two
    scores agree with the single-column amortized score and rank the candidates alike."""
    ev = _agent(**{**ARMS[arm], **READOUTS["hgoal_each"]})
    ev, _ = ev.update(_batch(seed=1))
    b, rew = _relabel()
    b = {**b, "next_observations": b["next_observations"].copy()}
    b["next_observations"][:6] = b["next_observations"][0]       # all rewarding rows the same state
    ev = ev.infer_eval_goals(b, rew)
    assert np.all(np.asarray(ev.eval_goals) == np.asarray(ev.eval_goals[0]))
    am = ev.replace(config=ev.config.copy({"coef_source": "amortized"}))
    am = am.infer_eval_goals(b, rew)        # same goal state; its reference columns are redrawn
    np.testing.assert_allclose(am.eval_w_star, ev.eval_goal_w[0], atol=1e-5)
    K = int(ev.config["gpi_num_u"])
    seed = jax.random.PRNGKey(2)
    u_cand = jnp.clip(jax.random.normal(seed, (K, DA)), -3.0, 3.0)
    obs = jnp.broadcast_to(jnp.asarray(b["observations"][0]), (K, OB))
    each = ev._hgoal_each_score(obs, u_cand, ev.eval_goals, ev.eval_goal_w, ev.eval_ref)
    w_rows = jnp.broadcast_to(am.eval_w_star, (K, Z))
    one = am._goal_score(obs, u_cand, ev.eval_goals[:1], ev.eval_ref, w_rows)
    np.testing.assert_allclose(np.asarray(each), np.asarray(one), rtol=1e-4, atol=1e-6)
    many = am._goal_score(obs, u_cand, ev.eval_goals, ev.eval_ref, w_rows)
    assert np.argsort(np.asarray(each)).tolist() == np.argsort(np.asarray(many)).tolist()
    if arm == "sq":
        np.testing.assert_allclose(np.asarray(each), np.asarray(many), rtol=1e-4, atol=1e-6)


# ------------------------------------------------------------------ create asserts

@pytest.mark.parametrize("over", [
    {"bootstrap_source": "data"},                                       # data bootstrap without the goal index
    {"bootstrap_source": "data", "train_actor": True},
    {**DB, "goal_random_frac": 0.3},                                    # random goals are rejected
    {**DB, "goal_cur_frac": 0.2},                                       # own-s' goals are rejected
    {"policy_index": "goal"},                                           # actor bootstrap without train_actor
    {"policy_index": "goal", "bootstrap_source": "actor"},
    {"coef_source": "hgoal_each"},                                      # per-goal h without the goal index
    {"bootstrap_source": "next"},
])
def test_create_asserts_fire(over):
    with pytest.raises(AssertionError):
        _agent(**over)


def test_spec_run_configs_build():
    for arm in ARMS.values():
        _agent(**arm)
        for readout in READOUTS.values():
            _agent(**{**arm, **readout})
    # the 2026-10-01 actor arm still builds, with and without the explicit default
    _agent(policy_index="goal", train_actor=True, actor_input="goal", bootstrap_source="actor")
    # data bootstrap with a trained actor is allowed by the spec ("train_actor may be false")
    _agent(**{**DB, "train_actor": True})

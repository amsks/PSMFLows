"""psmgoal with a large eval goal set (eval_goal_pool=dataset, k_goals up to 10,000).

Eval-time only. The default (eval_goal_pool=relabel) keeps the goal set and the coefficient
inference exactly as before; `GOLDEN` pins that path's outputs, computed on CPU with
agents/psmgoal.py as it stood before this option was added (2026-10-01).

eval_goal_pool=dataset:
  * the goal set is k_goals rewarding next states of the task dataset, drawn without
    replacement;
  * the Lagrangian for w runs on phi and b computed once on the (pairs x goals) mesh: the
    objective reads the per-pair mean over ALL goals, the constraint and the multiplier l read
    `LP_CONS_COLS` fixed random goal columns per pair (every column when k_goals is at most
    that), and with every column kept the result equals the uncached Lagrangian;
  * acting and coef_source=regression score the goal set in chunks.

Run in its own process: JAX_PLATFORMS=cpu .venv/bin/python -m pytest
tests/test_psmgoal_many_goals.py -q -p no:cacheprovider
"""
import hashlib
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml

import agents.psmgoal as pg
from agents.psmgoal import PSMGoalAgent, get_config

OB, DA, Z, CODE, N = 6, 3, 8, 4, 16
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# sha256 of eval_goals, eval_w_star and one deployed action on the default relabel path.
GOLDEN = {
    "lp": "158298b06714c070bdf61f3411586c9dea58f8bf243f595a043df0aa53fe3728",
    "regression": "5be38ba145105e3023ff8295a1404535dc3d0a29bba3520a35a4761b1dc533ff",
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
    c.flow.hidden_dims = (16, 16)
    for k, v in over.items():
        setattr(c, k, v)
    return c


def _agent(seed=0, **over):
    return PSMGoalAgent.create(seed, np.zeros((2, OB), np.float32), np.zeros((2, DA), np.float32),
                               _cfg(**over))


def _batch(reward_rows, n=N, seed=0):
    """A relabel batch whose shifted reward is 1 on `reward_rows` and 0 elsewhere."""
    rng = np.random.default_rng(seed)
    r = np.zeros((n,), np.float32)
    r[list(reward_rows)] = 1.0
    return {
        "observations": rng.standard_normal((n, OB)).astype(np.float32),
        "next_observations": rng.standard_normal((n, OB)).astype(np.float32),
        "noise_preimage": rng.standard_normal((n, DA)).astype(np.float32),
        "index": np.arange(n).astype(np.int32),
        "rewards": r,
    }


def _pool(n_rows=200, n_rewarding=60, seed=5):
    """The task dataset's next states and shifted rewards; the first n_rewarding rows reward."""
    rng = np.random.default_rng(seed)
    r = np.zeros((n_rows,), np.float32)
    r[:n_rewarding] = 1.0
    return {"next_observations": rng.standard_normal((n_rows, OB)).astype(np.float32), "rewards": r}


def _mesh_inputs(n=N, g=6, seed=3):
    rng = np.random.default_rng(seed)
    obs = jnp.asarray(rng.standard_normal((n, OB)), jnp.float32)
    u = jnp.clip(jnp.asarray(rng.standard_normal((n, DA)), jnp.float32), -3.0, 3.0)
    goals = jnp.asarray(rng.standard_normal((g, OB)), jnp.float32)
    return obs, u, goals


def _default_digest(coef_source):
    ag = _agent(coef_source=coef_source, num_inference_steps=20)
    b = _batch(reward_rows=(1, 3, 7, 8, 12))
    np.random.seed(0)
    ev = ag.infer_eval_goals(b, b["rewards"])
    action = ev.sample_actions(jnp.asarray(b["observations"][0]), seed=jax.random.PRNGKey(4))
    h = hashlib.sha256()
    for leaf in (ev.eval_goals, ev.eval_w_star, action):
        h.update(np.asarray(leaf, np.float32).tobytes())
    return h.hexdigest()


# --------------------------------------------------------------------- default path unchanged

@pytest.mark.parametrize("coef_source", ["lp", "regression"])
def test_default_relabel_path_is_unchanged(coef_source):
    assert _default_digest(coef_source) == GOLDEN[coef_source]


def test_new_key_default_in_config_and_yaml():
    with open(os.path.join(REPO, "configs", "agent", "psmgoal.yaml")) as fh:
        y = yaml.safe_load(fh)
    c = get_config()
    assert c["eval_goal_pool"] == "relabel" and y["eval_goal_pool"] == "relabel"
    assert c["k_goals"] == 32 and y["k_goals"] == 32


def test_unknown_goal_pool_is_rejected():
    with pytest.raises(AssertionError):
        _agent(eval_goal_pool="everything")


def test_relabel_ignores_a_passed_pool():
    b = _batch(reward_rows=(1, 3, 7, 8, 12))
    ag = _agent()
    np.random.seed(0)
    a = ag.infer_eval_goals(b, b["rewards"])
    np.random.seed(0)
    c = ag.infer_eval_goals(b, b["rewards"], goal_pool=_pool())
    assert np.array_equal(np.asarray(a.eval_goals), np.asarray(c.eval_goals))
    assert np.array_equal(np.asarray(a.eval_w_star), np.asarray(c.eval_w_star))


# --------------------------------------------------------------------- the dataset goal pool

def test_dataset_pool_draws_k_distinct_rewarding_rows():
    pool = _pool(n_rows=200, n_rewarding=60)
    b = _batch(reward_rows=(2,))                       # one rewarding row in the relabel batch
    ag = _agent(eval_goal_pool="dataset", k_goals=40)
    np.random.seed(0)
    ev = ag.infer_eval_goals(b, b["rewards"], goal_pool=pool)
    goals = np.asarray(ev.eval_goals)
    assert goals.shape == (40, OB)
    rewarding = pool["next_observations"][:60]
    rows = [int(np.nonzero((rewarding == g).all(axis=1))[0][0]) for g in goals]   # each is a rewarding row
    assert len(set(rows)) == 40                        # without replacement
    assert jnp.allclose(jnp.linalg.norm(ev.eval_w_star), jnp.sqrt(float(Z)), atol=1e-3)


def test_dataset_pool_with_too_few_rewarding_rows_raises():
    b = _batch(reward_rows=(2,))
    ag = _agent(eval_goal_pool="dataset", k_goals=40)
    with pytest.raises(ValueError, match="39 rewarding rows"):
        ag.infer_eval_goals(b, b["rewards"], goal_pool=_pool(n_rows=200, n_rewarding=39))


def test_dataset_pool_must_be_passed():
    b = _batch(reward_rows=(2,))
    ag = _agent(eval_goal_pool="dataset")
    with pytest.raises(AssertionError, match="goal_pool"):
        ag.infer_eval_goals(b, b["rewards"])


# --------------------------------------------------------------------- the cached Lagrangian

@pytest.mark.parametrize("use_dgd", [True, False])
def test_cached_lagrangian_matches_uncached_when_every_column_is_kept(use_dgd):
    """k_goals <= LP_CONS_COLS: the cached path reads the whole mesh, so w is the same."""
    ag = _agent(num_inference_steps=300, use_dgd=use_dgd)
    obs, u, goals = _mesh_inputs(g=6)
    key = jax.random.PRNGKey(11)
    w_old = np.asarray(ag.run_inference(obs, u, goals, key))
    w_new = np.asarray(ag.run_inference_cached(obs, u, goals, key))
    assert not np.allclose(w_old, np.asarray(ag.init_inference(key).w), atol=1e-2)   # w moved
    assert np.allclose(w_new, w_old, atol=1e-3), np.abs(w_new - w_old).max()


def test_lp_cache_holds_goal_means_and_the_fixed_columns(monkeypatch):
    """phi_mean, b_mean are the per-pair means over ALL goals; phi_cons, b_cons are the mesh
    at LP_CONS_COLS distinct columns per pair. Row-chunked with a padded last chunk."""
    monkeypatch.setattr(pg, "LP_CONS_COLS", 3)
    monkeypatch.setattr(pg, "MESH_ROWS", 50)           # 5 pairs per call at 10 goals; 16 = 3*5 + 1
    ag = _agent()
    obs, u, goals = _mesh_inputs(g=10)
    cache = ag._lp_cache(obs, u, goals, jax.random.PRNGKey(2))
    phi, b = ag.mesh_phi_b(obs, u, goals)
    cols = np.asarray(cache["cols"])
    assert cols.shape == (N, 3) and cols.min() >= 0 and cols.max() < 10
    assert all(len(set(row)) == 3 for row in cols.tolist())
    assert len({tuple(row) for row in cols.tolist()}) > 1          # not one shared column set
    assert jnp.allclose(cache["phi_mean"], phi.mean(axis=1), atol=1e-5)
    assert jnp.allclose(cache["b_mean"], b.mean(axis=1), atol=1e-5)
    assert jnp.allclose(cache["phi_cons"], jnp.take_along_axis(phi, cache["cols"][:, :, None], axis=1), atol=1e-5)
    assert jnp.allclose(cache["b_cons"], jnp.take_along_axis(b, cache["cols"], axis=1), atol=1e-5)
    # the multiplier's inputs are the same (pair, column) entries, row-major
    assert cache["l_obs"].shape == (N * 3, OB) and cache["l_u"].shape == (N * 3, DA)
    assert jnp.allclose(cache["l_obs"].reshape(N, 3, OB), obs[:, None, :])
    assert jnp.allclose(cache["l_goals"].reshape(N, 3, OB), goals[cache["cols"]])


def test_lp_cache_keeps_every_column_in_order_when_goals_fit():
    ag = _agent()
    obs, u, goals = _mesh_inputs(g=6)
    cache = ag._lp_cache(obs, u, goals, jax.random.PRNGKey(2))
    assert np.array_equal(np.asarray(cache["cols"]), np.tile(np.arange(6), (N, 1)))


def test_cached_lagrangian_runs_on_a_column_subsample(monkeypatch):
    monkeypatch.setattr(pg, "LP_CONS_COLS", 3)
    monkeypatch.setattr(pg, "MESH_ROWS", 50)
    ag = _agent(num_inference_steps=20)
    obs, u, goals = _mesh_inputs(g=10)
    w = ag.run_inference_cached(obs, u, goals, jax.random.PRNGKey(11))
    assert w.shape == (Z,) and bool(jnp.all(jnp.isfinite(w)))
    assert jnp.allclose(jnp.linalg.norm(w), jnp.sqrt(float(Z)), atol=1e-3)


def test_dataset_pool_lp_uses_the_cached_lagrangian():
    pool = _pool()
    b = _batch(reward_rows=(2,))
    ag = _agent(eval_goal_pool="dataset", k_goals=6, num_inference_steps=20)
    np.random.seed(0)
    ev = ag.infer_eval_goals(b, b["rewards"], goal_pool=pool)
    # the (s, u) pairs are the relabel batch's rows and preimages (a permutation of all N)
    np.random.seed(0)
    np.random.choice(np.arange(60), size=6, replace=False)
    sub = np.random.choice(np.arange(N), size=N, replace=False)
    obs = jnp.asarray(b["observations"][sub])
    u = jnp.clip(jnp.asarray(b["noise_preimage"][sub]), -3.0, 3.0)
    expect = ag.run_inference_cached(obs, u, ev.eval_goals, jax.random.fold_in(ag.rng, 7))
    assert jnp.allclose(ev.eval_w_star, expect, atol=1e-5)


# --------------------------------------------------------------------- chunked scoring

def test_goal_mean_is_the_same_in_chunks():
    ag = _agent()
    obs, u, goals = _mesh_inputs(g=10)
    w_rows = jnp.broadcast_to(ag._project(jnp.arange(1.0, Z + 1.0)), (N, Z))
    full = jnp.mean(ag.mesh_M(obs, u, goals, w_rows), axis=1)
    for chunk in (3, 5, 10, 64):                       # remainder, exact split, one chunk, larger
        assert jnp.allclose(ag._goal_mean_M(obs, u, goals, w_rows, chunk=chunk), full, atol=1e-5)


def test_select_latent_is_the_same_in_chunks(monkeypatch):
    ag = _agent(k_goals=10, gpi_num_u=16)
    _, _, goals = _mesh_inputs(g=10)
    ag = ag.replace(eval_goals=goals, eval_w_star=ag._project(jnp.arange(1.0, Z + 1.0)))
    obs = jnp.asarray(np.random.default_rng(9).standard_normal((OB,)), jnp.float32)
    whole = [np.asarray(ag.select_latent(obs, jax.random.PRNGKey(s))) for s in range(8)]
    monkeypatch.setattr(pg, "GOAL_CHUNK", 3)
    chunked = [np.asarray(ag.select_latent(obs, jax.random.PRNGKey(s))) for s in range(8)]
    assert all(np.array_equal(a, c) for a, c in zip(whole, chunked))


def test_jitted_acting_with_a_chunked_goal_set(monkeypatch):
    monkeypatch.setattr(pg, "GOAL_CHUNK", 16)
    ag = _agent(k_goals=40, eval_goal_pool="dataset", num_inference_steps=4)
    b = _batch(reward_rows=(2,))
    ev = ag.infer_eval_goals(b, b["rewards"], goal_pool=_pool())
    a = np.asarray(ev.sample_actions(jnp.asarray(b["observations"][0]), seed=jax.random.PRNGKey(4)))
    assert a.shape == (DA,) and np.all(np.isfinite(a)) and np.all(np.abs(a) <= 1.0)
    assert np.array_equal(a, np.asarray(ev.decode(
        jnp.asarray(b["observations"][:1]),
        ev.select_latent(jnp.asarray(b["observations"][0]), jax.random.PRNGKey(4))[None])[0]))


# --------------------------------------------------------------------- regression, chunked

@pytest.mark.parametrize("measure_form", ["joint", "factorized"])
def test_regression_on_a_dataset_pool_is_the_least_squares_fit(monkeypatch, measure_form):
    """w = project(lstsq(Phi, r)), Phi[i] = mean over the goal set of phi(s'_i, u_i, g)."""
    monkeypatch.setattr(pg, "MESH_ROWS", 100)          # 2 rows per call at 40 goals
    rng = np.random.default_rng(1)
    b = _batch(reward_rows=(2,))
    r = rng.standard_normal((N,)).astype(np.float32)   # regression target over the batch rows
    ag = _agent(coef_source="regression", eval_goal_pool="dataset", k_goals=40, measure_form=measure_form)
    ev = ag.infer_eval_goals(b, r, goal_pool=_pool())
    next_obs = jnp.asarray(b["next_observations"])
    u = jnp.clip(jnp.asarray(b["noise_preimage"]), -3.0, 3.0)
    Phi = ag.mesh_phi_b(next_obs, u, ev.eval_goals)[0].mean(axis=1)
    expect = ag._project(jnp.linalg.lstsq(Phi, jnp.asarray(r), rcond=None)[0])
    assert jnp.allclose(ev.eval_w_star, expect, atol=1e-3)

"""psmgoal measure_loss=pool (docs/design/2026-10-19-pool-measure.md): the candidate set of the
cross-entropy TD loss is a fixed reference pool of dataset states carried on the agent; the one-hot
term sits on the pool state nearest the row's next state. CPU, allow_untrained_flow.

Run in its own process: JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/test_psmgoal_pool_loss.py -q
"""
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml

import utils.xla_guard  # noqa: F401  (must precede jax)
from agents.psmgoal import PSMGoalAgent, get_config
from tests.test_psmgoal_bootstrap_delta_scale import OVER, _z
from tests.test_psmgoal_joint_actor import DA, OB, _agent, _batch, _cfg
from utils.psm_common import pool_hash

POOL_KEYS = {"pool_size": 1024, "pool_seed": 0, "pool_hash": None}


def _pool_agent(pool_obs, seed=0, indices=None, **over):
    pool_obs = np.asarray(pool_obs, np.float32)
    indices = np.arange(pool_obs.shape[0]) if indices is None else np.asarray(indices)
    cfg = _cfg(measure_loss="pool", pool_size=int(pool_obs.shape[0]), **over)
    return PSMGoalAgent.create(seed, np.zeros((2, OB), np.float32), np.zeros((2, DA), np.float32), cfg,
                               pool_obs=pool_obs, pool_indices=indices)


def _goal_over(**extra):
    return {**{k: v for k, v in OVER.items() if k != "measure_loss"}, **extra}


def _leaves(tree):
    return jax.tree_util.tree_leaves(tree)


# ------------------------------------------------------------------ defaults

def test_pool_keys_default_in_get_config_and_yaml():
    c = get_config()
    path = os.path.join(os.path.dirname(__file__), "..", "configs", "agent", "psmgoal.yaml")
    with open(path) as f:
        y = yaml.safe_load(f)
    for k, v in POOL_KEYS.items():
        assert k in c and c[k] == v, f"get_config()[{k!r}] = {c.get(k)!r}"
        assert k in y and y[k] == v, f"psmgoal.yaml[{k!r}] = {y.get(k)!r}"
    assert c["measure_loss"] == "squared" and y["measure_loss"] == "squared"


def test_default_agent_carries_no_pool():
    ag = _agent()
    assert ag.pool_states is None and ag.pool_indices is None
    assert ag.config.get("pool_hash") is None


# ------------------------------------------------------------------ the loss

def test_pool_loss_matches_softmax_loss_when_pool_is_the_batch_next_states():
    """Pool = the batch's own next states in batch order: every anchor is the row's own column and the
    pool loss equals softmax_td_loss on the same mesh."""
    batch = _batch()
    soft = _agent(**OVER)
    pool = _pool_agent(batch["next_observations"], **_goal_over())
    # same seed, same nets: the pool enters only through the candidate set
    for a, b in zip(_leaves(soft.basis.params), _leaves(pool.basis.params)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    z = _z(soft, batch)
    u_soft = soft._select_u_next(batch, z)
    u_pool = pool._select_u_next(batch, z)
    np.testing.assert_allclose(np.asarray(u_soft), np.asarray(u_pool))
    assert np.array_equal(np.asarray(pool.pool_anchors(jnp.asarray(batch["next_observations"]))), np.arange(len(batch["observations"])))
    l_soft, _ = soft.measure_loss(soft.basis.params, soft.w_star.params, batch, z, u_soft)
    l_pool, info = pool.measure_loss(pool.basis.params, pool.w_star.params, batch, z, u_pool)
    np.testing.assert_allclose(float(l_pool), float(l_soft), rtol=1e-5, atol=1e-6)
    assert 0.0 < float(info["measure_p_anchor"]) < 1.0 and np.isfinite(float(info["pool_u_tv"]))


def test_pool_td_loss_target_rows_sum_to_one_and_anchor_gets_one_minus_gamma_when_target_is_flat():
    rng = np.random.default_rng(0)
    N, P, g, t = 5, 7, 0.9, 0.5
    M = rng.standard_normal((N, P)).astype(np.float32)
    anchor = np.array([3, 0, 6, 6, 1], np.int32)
    target = g / P * np.ones((N, P), np.float32)
    target[np.arange(N), anchor] += 1.0 - g
    np.testing.assert_allclose(target.sum(1), 1.0, rtol=1e-6)
    logp = M / t - np.log(np.exp(M / t).sum(1, keepdims=True))
    expect = -np.mean((target * logp).sum(1))
    got = PSMGoalAgent.pool_td_loss(jnp.asarray(M), jnp.zeros((N, P)), jnp.asarray(anchor), g, t)
    np.testing.assert_allclose(float(got), expect, rtol=1e-5)
    # the minimum is at softmax(M / t) = target, where the loss is the target's entropy
    M_star = t * np.log(target)
    at_min = float(PSMGoalAgent.pool_td_loss(jnp.asarray(M_star), jnp.zeros((N, P)), jnp.asarray(anchor), g, t))
    np.testing.assert_allclose(at_min, -np.mean((target * np.log(target)).sum(1)), rtol=1e-5)
    assert at_min < float(got)


def test_pool_anchor_is_nearest_pool_state_in_pool_scale():
    pool_obs = np.zeros((4, OB), np.float32)
    pool_obs[1, 0] = 10.0
    pool_obs[2, 1] = 1.0
    pool_obs[3, 2] = 3.0
    ag = _pool_agent(pool_obs, **_goal_over())
    sigma = np.maximum(pool_obs.std(0), 1e-6)
    np.testing.assert_allclose(np.asarray(ag.pool_scale), sigma, rtol=1e-6)
    nxt = np.zeros((3, OB), np.float32)
    nxt[0, 0] = 7.0          # nearest x=10
    nxt[1, 1] = 0.6          # nearest y=1
    nxt[2, 2] = 1.0          # 1/sigma_z from z=0 and 2/sigma_z from z=3 -> pool 0
    assert np.asarray(ag.pool_anchors(jnp.asarray(nxt))).tolist() == [1, 2, 0]


# ------------------------------------------------------------------ asserts

@pytest.mark.parametrize("over", [
    {"norm_future_k": 1},
    {"ortho_coef": 0.1},
    {"measure_form": "factorized"},
])
def test_pool_refuses_incompatible_settings(over):
    with pytest.raises(AssertionError):
        _pool_agent(np.zeros((16, OB), np.float32), **_goal_over(**over))


def test_pool_refuses_a_missing_or_mis_sized_pool():
    with pytest.raises(AssertionError):
        _agent(measure_loss="pool", pool_size=16, **_goal_over())            # no pool_obs handed in
    with pytest.raises(AssertionError):
        PSMGoalAgent.create(0, np.zeros((2, OB), np.float32), np.zeros((2, DA), np.float32),
                            _cfg(measure_loss="pool", pool_size=8, **_goal_over()),
                            pool_obs=np.zeros((16, OB), np.float32), pool_indices=np.arange(16))


def test_pool_hash_is_recorded_and_checked():
    idx = np.array([3, 10, 25, 41], np.int64)
    ag = _pool_agent(np.zeros((4, OB), np.float32), indices=idx, **_goal_over())
    assert ag.config["pool_hash"] == pool_hash(idx) and len(ag.config["pool_hash"]) == 40
    np.testing.assert_array_equal(np.asarray(ag.pool_indices), idx)
    # a config that already records a hash (eval, from flags.json) must match the pool handed in
    with pytest.raises(AssertionError):
        _pool_agent(np.zeros((4, OB), np.float32), indices=idx, pool_hash=pool_hash(idx + 1), **_goal_over())
    ok = _pool_agent(np.zeros((4, OB), np.float32), indices=idx, pool_hash=pool_hash(idx), **_goal_over())
    assert ok.config["pool_hash"] == pool_hash(idx)


# ------------------------------------------------------------------ update

def test_pool_update_runs_and_logs_keys():
    pool_obs = np.random.default_rng(1).standard_normal((16, OB)).astype(np.float32)
    ag = _pool_agent(pool_obs, **_goal_over())
    ag, info = ag.update(_batch())
    for k in ("psm_loss", "measure_p_anchor", "pool_entropy", "pool_u_tv", "phi_eff_rank", "td_target_absmean",
              "w_norm", "actor_q"):
        assert k in info and np.isfinite(float(info[k])), k
    assert 0.0 <= float(info["measure_p_anchor"]) <= 1.0
    assert 0.0 <= float(info["pool_entropy"]) <= np.log(16) + 1e-5
    assert 0.0 <= float(info["pool_u_tv"]) <= 1.0
    assert "measure_p_own_next" not in info


def test_pool_states_are_not_trainable_and_persist_across_update():
    pool_obs = np.random.default_rng(2).standard_normal((16, OB)).astype(np.float32)
    ag = _pool_agent(pool_obs, **_goal_over())
    b0 = _leaves(ag.basis.params)
    new, _ = ag.update(_batch())
    np.testing.assert_array_equal(np.asarray(new.pool_states), pool_obs)
    np.testing.assert_array_equal(np.asarray(new.pool_indices), np.arange(16))
    np.testing.assert_array_equal(np.asarray(new.pool_scale), np.asarray(ag.pool_scale))
    assert any(not np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(b0, _leaves(new.basis.params)))
    # the pool has no optimiser and no gradient: it is not among the trainable states
    assert all(not hasattr(x, "opt_state") for x in (new.pool_states, new.pool_indices))


def test_pool_u_tv_is_positive_when_the_measure_reads_u():
    """At init phi, b read u, so the row's own u and a fresh prior draw give different distributions over
    the pool: pool_u_tv is strictly positive (and at most 1)."""
    pool_obs = np.random.default_rng(3).standard_normal((16, OB)).astype(np.float32)
    ag = _pool_agent(pool_obs, **_goal_over())
    batch = _batch()
    z = _z(ag, batch)
    u_next = ag._select_u_next(batch, z)
    _, info = ag.measure_loss(ag.basis.params, ag.w_star.params, batch, z, u_next)
    assert 0.0 < float(info["pool_u_tv"]) <= 1.0

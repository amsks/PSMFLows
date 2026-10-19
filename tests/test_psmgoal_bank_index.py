"""psmgoal policy_index=bank (docs/design/2026-10-19-pool-measure.md): one policy, the behaviour
policy (a clipped prior draw), with one shared learned coefficient w; a plain policy evaluation of BC.
CPU, allow_untrained_flow.

Run in its own process: JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/test_psmgoal_bank_index.py -q
"""
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml

import agents.psmgoal as pg
import utils.xla_guard  # noqa: F401  (must precede jax)
from agents.psmgoal import get_config
from tests.test_psmgoal_bootstrap_delta_scale import _z
from tests.test_psmgoal_joint_actor import OB, Z, _agent, _batch
from tests.test_psmgoal_pool_loss import _pool_agent

BANK = {"policy_index": "bank", "train_actor": False, "measure_loss": "softmax"}


def _leaves(tree):
    return jax.tree_util.tree_leaves(tree)


def _same(a, b):
    return all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(_leaves(a), _leaves(b)))


def test_n_bank_defaults_to_one_in_get_config_and_yaml():
    c = get_config()
    path = os.path.join(os.path.dirname(__file__), "..", "configs", "agent", "psmgoal.yaml")
    with open(path) as f:
        y = yaml.safe_load(f)
    assert c["n_bank"] == 1 and y["n_bank"] == 1
    assert c["policy_index"] == "code" and y["policy_index"] == "code"


def test_bank_w_is_shared_across_rows():
    ag = _agent(**BANK)
    batch = _batch()
    z = _z(ag, batch)
    w = np.asarray(ag._coef(z, None))
    assert w.shape == (len(batch["observations"]), Z)
    np.testing.assert_allclose(w, np.broadcast_to(w[:1], w.shape), rtol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(w, axis=-1), np.sqrt(Z), rtol=1e-4)
    # the policy code is ignored: any z gives the same w
    z2 = jnp.ones_like(z)
    np.testing.assert_allclose(np.asarray(ag._coef(z2, None)), w, rtol=1e-6)
    # bank_w is that single coefficient, (1, z); the target reads the Polyak copy
    np.testing.assert_allclose(np.asarray(ag.bank_w()), w[:1], rtol=1e-6)
    np.testing.assert_allclose(np.asarray(ag._coef(z, None, target=True)), w, rtol=1e-6)   # targets start equal
    assert np.all(np.isfinite(w)) and np.abs(w).max() > 0.0


def test_bank_bootstrap_is_a_clipped_prior_draw():
    ag = _agent(**BANK)
    batch = _batch()
    z = _z(ag, batch)
    u = np.asarray(ag._select_u_next(batch, z))
    n, d = len(batch["observations"]), int(ag.config["action_dim"])
    eps = jax.random.normal(jax.random.fold_in(ag.rng, pg.FLOWBC_BOOT_KEY), (n, d))
    clip = float(ag.config["u_clip"])
    np.testing.assert_allclose(u, np.clip(np.asarray(eps), -clip, clip), atol=1e-6)
    # it does not read the batch's goals, the actor or the policy code
    other = ag.replace(rng=jax.random.PRNGKey(99))
    assert not np.allclose(np.asarray(other._select_u_next(batch, z)), u)
    np.testing.assert_allclose(np.asarray(ag._select_u_next(batch, jnp.ones_like(z))), u, atol=1e-6)


@pytest.mark.parametrize("over", [
    {**BANK, "train_actor": True},
    {**BANK, "n_bank": 2},
    {**BANK, "bootstrap_source": "data"},
    {**BANK, "bootstrap_source": "argmax"},
    {**BANK, "bootstrap_source": "bc_mean"},
    {**BANK, "coef_source": "hgoal_each"},
    {**BANK, "actor_input": "goal"},
    {**BANK, "train_goal_head": True},
])
def test_bank_refuses_incompatible_settings(over):
    with pytest.raises(AssertionError):
        _agent(**over)


def test_bank_update_runs_with_the_pool_loss():
    pool_obs = np.random.default_rng(5).standard_normal((16, OB)).astype(np.float32)
    ag = _pool_agent(pool_obs, policy_index="bank", train_actor=False)
    w0, tw0, h0, d0, a0 = ag.w.params, ag.target_w, ag.w_star.params, ag.delta.params, ag.actor.params
    new, info = ag.update(_batch())
    for k in ("psm_loss", "measure_p_anchor", "pool_entropy", "pool_u_tv", "phi_eff_rank", "w_norm"):
        assert k in info and np.isfinite(float(info[k])), k
    assert not any(k.startswith("actor_") for k in info)
    assert not _same(new.w.params, w0) and not _same(new.target_w, tw0)      # the shared w trains
    assert _same(new.w_star.params, h0) and _same(new.delta.params, d0) and _same(new.actor.params, a0)
    np.testing.assert_allclose(float(info["w_norm"]), np.sqrt(Z), rtol=1e-4)


def test_bank_update_runs_with_the_softmax_loss():
    ag = _agent(**BANK)
    new, info = ag.update(_batch())
    assert np.isfinite(float(info["psm_loss"])) and "measure_p_own_next" in info
    assert not _same(new.w.params, ag.w.params)

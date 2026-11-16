"""measure_form=factorized with critic_heads=2 and/or factorized_index_input (2026-11-16).

Checks, on a tiny synthetic dataset with the frozen-flow stand-in used by the other psmgoal tests:
- two factorised heads give two different scores and the pool-softmax readout equals their min;
- with factorized_index_input the operators change when w changes, and without it they do not;
- one jitted update runs on each combination and changes the basis params.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils.xla_guard  # noqa: F401

# isort: split

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agents.psmgoal import PSMGoalAgent
from tests.psmgoal_golden import DA, OB, Z, tiny_batch
from tests.test_psmgoal_dfb_recipe import _dfb_cfg


def make_agent_and_batch(over):
    c = _dfb_cfg(**{k: v for k, v in over.items() if k not in ('f_dim', 'z_dim')})
    c.measure_form = 'factorized'
    c.f_dim = Z
    c.measure.layer_norm = False   # the recipe's LayerNorm is joint-only
    agent = PSMGoalAgent.create(0, np.zeros((2, OB), np.float32), np.zeros((2, DA), np.float32), c)
    return agent, tiny_batch(seed=0)


def _cfg(**over):
    base = dict(measure_form="factorized", policy_index="reward", coef_source="ridge", measure_loss="softmax",
                train_actor=True, actor_kind="flowbc_vf", reward_q_center=True, ridge_score="pool_softmax",
                goal_random_frac=0.0, acting="distill", eval_redistill=False)
    base.update(over)
    return base


@pytest.mark.parametrize("heads,index_input", [(2, False), (1, True), (2, True)])
def test_update_runs_and_moves_basis(heads, index_input):
    agent, batch = make_agent_and_batch(_cfg(critic_heads=heads, factorized_index_input=index_input))
    before = jax.tree_util.tree_leaves(agent.basis.params)
    if heads > 1:
        assert before[0].shape[0] == heads, "stacked head params expected"
    agent2, info = agent.update(batch)
    after = jax.tree_util.tree_leaves(agent2.basis.params)
    assert any(not np.allclose(np.asarray(a), np.asarray(b)) for a, b in zip(before, after))
    assert np.isfinite(float(info["psm_loss"]))
    if heads > 1:
        assert "psm_loss_h1" in info


def test_two_heads_differ_and_readout_is_min():
    agent, batch = make_agent_and_batch(_cfg(critic_heads=2))
    obs = jnp.asarray(batch["observations"][:4])
    u = jnp.clip(jnp.asarray(batch["noise_preimage"][:4]), -3, 3)
    pool = jnp.asarray(batch["next_observations"][:8])
    w = jnp.ones((4, Z))
    m0 = agent.mesh_M(obs, u, pool, w, head=0)
    m1 = agent.mesh_M(obs, u, pool, w, head=1)
    assert not np.allclose(np.asarray(m0), np.asarray(m1)), "independent heads must differ at init"
    f0 = agent._features_fwd(pool, head=0)
    f1 = agent._features_fwd(pool, head=1)
    assert not np.allclose(np.asarray(f0), np.asarray(f1))


def test_index_input_changes_operators_only_when_on():
    for on in (False, True):
        agent, batch = make_agent_and_batch(_cfg(factorized_index_input=on))
        obs = jnp.asarray(batch["observations"][:4])
        u = jnp.clip(jnp.asarray(batch["noise_preimage"][:4]), -3, 3)
        pool = jnp.asarray(batch["next_observations"][:8])
        w_a = jnp.ones((4, Z)) * 0.5
        w_b = -jnp.ones((4, Z)) * 0.5
        phi_a, b_a = agent.mesh_phi_b(obs, u, pool, w_rows=w_a)
        phi_b, b_b = agent.mesh_phi_b(obs, u, pool, w_rows=w_b)
        same = np.allclose(np.asarray(phi_a), np.asarray(phi_b)) and np.allclose(np.asarray(b_a), np.asarray(b_b))
        assert same == (not on), f"index_input={on}: operators {'must' if on else 'must not'} depend on w"

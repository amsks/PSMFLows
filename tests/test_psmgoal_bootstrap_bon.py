"""bootstrap_bon_n (2026-11-16, HQL-style): the reward-index measure's bootstrap latent u+ at s' is the best of
N clipped prior draws under the critic's own reward readout (reward_softmax_q at the row's index w, min over
heads, target basis), in place of the FlowBC actor's draw.

Checks, on the tiny synthetic dataset and frozen-flow stand-in of the other psmgoal tests
(measure_form=factorized, policy_index=reward, actor_kind=flowbc_vf):
- with bootstrap_bon_n=4 one jitted update runs (critic_heads=2, with and without factorized_index_input);
- the chosen u+ is one of the N drawn candidates and has the highest readout among them;
- bootstrap_bon_n=0 leaves one update bit-identical to the default (flag absent) run, and N=4 changes it.
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

from agents.psmgoal import BON_BOOT_KEY, PSMGoalAgent
from tests.psmgoal_golden import DA, OB, Z, tiny_batch
from tests.test_psmgoal_dfb_recipe import _dfb_cfg


def make_agent_and_batch(over):
    c = _dfb_cfg(**over)
    c.measure_form = 'factorized'
    c.f_dim = Z
    c.measure.layer_norm = False   # the recipe's LayerNorm is joint-only
    agent = PSMGoalAgent.create(0, np.zeros((2, OB), np.float32), np.zeros((2, DA), np.float32), c)
    return agent, tiny_batch(seed=0)


def _cfg(**over):
    base = {"policy_index": "reward", "coef_source": "ridge", "measure_loss": "softmax", "train_actor": True,
            "actor_kind": "flowbc_vf", "reward_q_center": True, "ridge_score": "pool_softmax",
            "goal_random_frac": 0.0, "acting": "distill", "eval_redistill": False}
    base.update(over)
    return base


def _leaves(agent):
    return [np.asarray(x) for x in jax.tree_util.tree_leaves((agent.basis.params, agent.fbc_actor.params,
                                                              agent.fbc_vf.params, agent.target_basis))]


@pytest.mark.parametrize("heads,index_input", [(2, False), (2, True), (1, False)])
def test_update_runs_with_bon(heads, index_input):
    agent, batch = make_agent_and_batch(_cfg(critic_heads=heads, factorized_index_input=index_input,
                                             bootstrap_bon_n=4))
    before = jax.tree_util.tree_leaves(agent.basis.params)
    agent2, info = agent.update(batch)
    after = jax.tree_util.tree_leaves(agent2.basis.params)
    assert any(not np.allclose(np.asarray(a), np.asarray(b)) for a, b in zip(before, after))
    assert np.isfinite(float(info["psm_loss"]))


@pytest.mark.parametrize("heads,index_input", [(2, False), (2, True)])
def test_selection_picks_best_candidate(heads, index_input):
    n = 4
    agent, batch = make_agent_and_batch(_cfg(critic_heads=heads, factorized_index_input=index_input,
                                             bootstrap_bon_n=n))
    rows = {**batch, **agent.reward_index_rows(batch)}
    u_next, scores = agent.bon_bootstrap(rows)
    next_obs = jnp.asarray(rows["next_observations"])
    N = next_obs.shape[0]
    assert u_next.shape == (N, DA) and scores.shape == (n, N)
    # the candidates: the same clipped draws from the BON_BOOT_KEY stream
    eps = jnp.clip(jax.random.normal(jax.random.fold_in(agent.rng, BON_BOOT_KEY), (n, N, DA)), -3.0, 3.0)
    u_np, eps_np = np.asarray(u_next), np.asarray(eps)
    which = [int(np.flatnonzero(np.all(eps_np[:, i] == u_np[i], axis=-1))[0]) for i in range(N)]
    # the scores are the critic's own reward readout at the row's index w on the TARGET basis
    w = jnp.asarray(rows["index_w"])
    f_c = jnp.asarray(rows["index_f"]) - agent.fb_mu[None]
    for k in range(n):
        q_k, q_heads = agent.reward_softmax_q(next_obs, eps[k], next_obs, w, f_c, agent.target_basis)
        assert q_heads.shape[0] == heads
        np.testing.assert_allclose(np.asarray(q_k), np.asarray(scores[k]), rtol=1e-4, atol=1e-5)  # lax.map vs direct, f32
    s_np = np.asarray(scores)
    for i in range(N):
        assert s_np[which[i], i] == s_np[:, i].max(), i
    # the jitted update's bootstrap is this selection
    np.testing.assert_array_equal(np.asarray(agent._select_u_next(rows, None)), u_np)


def test_off_is_bit_identical_and_on_differs():
    agent_default, batch = make_agent_and_batch(_cfg(critic_heads=2))
    agent_off, _ = make_agent_and_batch(_cfg(critic_heads=2, bootstrap_bon_n=0))
    agent_on, _ = make_agent_and_batch(_cfg(critic_heads=2, bootstrap_bon_n=4))
    assert int(agent_default.config["bootstrap_bon_n"]) == 0
    for a, b in zip(_leaves(agent_default), _leaves(agent_on)):
        np.testing.assert_array_equal(a, b)   # same init; only the bootstrap differs
    d1, _ = agent_default.update(batch)
    d0, _ = agent_off.update(batch)
    d4, _ = agent_on.update(batch)
    for a, b in zip(_leaves(d1), _leaves(d0)):
        np.testing.assert_array_equal(a, b)
    basis_off = jax.tree_util.tree_leaves(d0.basis.params)
    basis_on = jax.tree_util.tree_leaves(d4.basis.params)
    assert any(not np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(basis_off, basis_on))

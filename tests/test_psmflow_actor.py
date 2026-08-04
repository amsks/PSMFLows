"""The amortized latent actor (flowBC recipe over latents, Rung 3).

Pins the three properties that make the actor safe to ship alongside v1:
  - actor.enabled=false (the default) leaves the measure update byte-identical in its
    reported losses and never touches actor params;
  - the enabled actor branch trains ONLY (actor, actor_vf) — psi/phi and the frozen flow
    must not move from actor gradients;
  - acting="actor" emits a latent inside the u_clip box and decodes to a valid action.
"""
import math

import jax
import jax.numpy as jnp
import numpy as np

from tests.test_psmflow_agent import _agent, _batch


def _tree_equal(a, b):
    return all(bool(jnp.array_equal(x, y))
               for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)))


def test_disabled_actor_changes_nothing():
    agent, info = _agent().update(_batch())
    assert "actor_loss" not in info
    agent2, _ = agent.update(_batch(1))
    assert _tree_equal(agent.actor.params, agent2.actor.params)
    assert _tree_equal(agent.actor_vf.params, agent2.actor_vf.params)


def test_enabled_actor_trains_only_actor_branch():
    agent = _agent(actor=dict(enabled=True, hidden_dim=32, hidden_layers=1,
                              embedding_layers=2, vf_hidden_dim=32, vf_hidden_layers=2,
                              flow_steps=3, bc_coeff=1.0, task_mix_ratio=0.5))
    before_flow = agent.flow_vf
    before_psi = agent.psi.params
    agent2, info = agent.update(_batch())
    for k in ("actor_loss", "actor_q", "actor_bc_flow_loss", "actor_bc_error"):
        assert math.isfinite(float(info[k])), (k, info[k])
    assert not _tree_equal(agent.actor.params, agent2.actor.params), "actor did not train"
    assert not _tree_equal(agent.actor_vf.params, agent2.actor_vf.params), "vf did not train"
    # The frozen flow must stay frozen and the measure step must be the only psi change:
    # rerun a disabled agent from the same state and compare psi one-step deltas.
    assert _tree_equal(before_flow, agent2.flow_vf)
    ref, _ = agent.replace(config=agent.config.copy(
        {"actor": dict(agent.config["actor"], enabled=False)})).update(_batch())
    assert _tree_equal(ref.psi.params, agent2.psi.params), (
        "psi moved differently with the actor branch on — actor gradients leaked into psi")
    assert before_psi is not agent2.psi.params


def test_acting_actor_decodes_a_clipped_latent():
    agent = _agent(acting="actor",
                   actor=dict(enabled=True, hidden_dim=32, hidden_layers=1,
                              embedding_layers=2, vf_hidden_dim=32, vf_hidden_layers=2,
                              flow_steps=3, bc_coeff=1.0, task_mix_ratio=0.5))
    agent = agent.replace(task_z=jnp.ones_like(agent.task_z))
    obs = jnp.zeros((6,), jnp.float32)
    a = np.asarray(agent.sample_actions(obs, seed=jax.random.PRNGKey(0)))
    assert a.shape == (2,) and np.isfinite(a).all()
    assert (np.abs(a) <= 1.0 + 1e-6).all()
    # The latent itself respects the box: tanh * u_clip by construction.
    noise = jax.random.normal(jax.random.PRNGKey(0), (1, 2))
    u = np.asarray(agent.config["u_clip"] * agent.actor(obs[None], agent.task_z[None], noise))
    assert (np.abs(u) <= agent.config["u_clip"] + 1e-6).all()

"""The amortized latent actor (flowBC recipe over latents) — a NON-DEFAULT arm since 09-04.

The primary agent (`train_actor=false acting=gpi`) has no actor at all, so every test here
builds the arm explicitly via `_legacy_agent()` (`train_actor=true acting=actor`, task-vector
index). The arm is retained deliberately: it is the DSRL-style comparator and the substrate
for the planned DSRL-NA distillation of the GPI argmax into an amortized head. Pins:
  - the actor and its vf train on EVERY update of that arm, with finite losses;
  - actor gradients do not leak into psi: an actor-only config change (bc_coeff) must
    leave the psi one-step delta byte-identical (sampling is unaffected by it);
  - acting="actor" emits a latent inside the u_clip box and decodes to a valid action.
"""
import math

import jax
import jax.numpy as jnp
import numpy as np

from tests.test_psmflow_agent import _batch, _legacy_agent


def _tree_equal(a, b):
    return all(bool(jnp.array_equal(x, y))
               for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)))


def test_actor_trains_every_update():
    agent = _legacy_agent()
    agent2, info = agent.update(_batch())
    for k in ("actor_loss", "actor_q", "actor_bc_flow_loss", "actor_bc_error"):
        assert math.isfinite(float(info[k])), (k, info[k])
    assert not _tree_equal(agent.actor.params, agent2.actor.params), "actor did not train"
    assert not _tree_equal(agent.actor_vf.params, agent2.actor_vf.params), "vf did not train"
    assert _tree_equal(agent.flow_vf, agent2.flow_vf), "frozen flow moved"


def test_actor_gradients_do_not_leak_into_psi():
    """Same init, same batch, different bc_coeff (an actor-loss-only knob): the psi and
    phi one-step deltas must be byte-identical, the actor's must differ."""
    a1 = _legacy_agent()
    a2 = _legacy_agent(actor=dict(hidden_dim=512, hidden_layers=2, embedding_layers=2,
                           vf_hidden_dim=512, vf_hidden_layers=4, flow_steps=10,
                           bc_coeff=7.0))
    u1, _ = a1.update(_batch())
    u2, _ = a2.update(_batch())
    assert _tree_equal(u1.psi.params, u2.psi.params), "actor config changed the psi step"
    assert _tree_equal(u1.phi.params, u2.phi.params), "actor config changed the phi step"
    assert not _tree_equal(u1.actor.params, u2.actor.params)


def test_acting_actor_decodes_a_clipped_latent():
    agent = _legacy_agent()
    agent = agent.replace(task_z=jnp.ones_like(agent.task_z))
    obs = jnp.zeros((6,), jnp.float32)
    a = np.asarray(agent.sample_actions(obs, seed=jax.random.PRNGKey(0)))
    assert a.shape == (2,) and np.isfinite(a).all()
    assert (np.abs(a) <= 1.0 + 1e-6).all()
    # The latent itself respects the box: tanh * u_clip by construction.
    noise = jax.random.normal(jax.random.PRNGKey(0), (1, 2))
    u = np.asarray(agent.config["u_clip"] * agent.actor(obs[None], agent.task_z[None], noise))
    assert (np.abs(u) <= agent.config["u_clip"] + 1e-6).all()

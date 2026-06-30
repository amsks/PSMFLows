"""Validation + plumbing tests for the BC-flow inversion utilities.

Task 1 (round-trip) characterizes Claas's inversion methods on FQLAgent. The code
under test already exists, so these are validation/characterization tests: they may
pass on first run. Round-trip consistency is a property of the inverter vs. the
network and holds regardless of flow training quality, as long as the forward and
inverse maps use the SAME step discretization.
"""
import jax
import jax.numpy as jnp
import numpy as np

from agents.fql import FQLAgent, get_config


def _tiny_agent(obs_dim=4, act_dim=2, flow_steps=100):
    """Build a minimal real FQLAgent. `create` derives ob_dims/action_dim and
    `encoder` defaults to None, so we only override flow_steps for speed/accuracy."""
    cfg = get_config()
    cfg['flow_steps'] = flow_steps
    ex_obs = jnp.zeros((1, obs_dim))
    ex_act = jnp.zeros((1, act_dim))
    return FQLAgent.create(0, ex_obs, ex_act, cfg)


def test_roundtrip_recovers_action():
    flow_steps = 100
    agent = _tiny_agent(obs_dim=4, act_dim=2, flow_steps=flow_steps)
    obs = jax.random.normal(jax.random.PRNGKey(0), (8, 4))
    noise = jax.random.normal(jax.random.PRNGKey(1), (8, 2))

    actions = agent.compute_flow_actions(obs, noises=noise)  # forward (uses cfg flow_steps)
    # invert each (single-example methods -> vmap); match n_steps to the forward discretization
    preimage = jax.vmap(
        lambda s, a: agent._get_preimage_and_jacobian(s, a, flow_steps)[0]
    )(obs, actions)
    recon = agent.compute_flow_actions(obs, noises=preimage)  # forward again
    err = float(jnp.mean(jnp.linalg.norm(recon - actions, axis=-1)))
    assert err < 1e-2, f"round-trip L2 {err} too high"

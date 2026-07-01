"""Tests for energy-proposal flow steering (WP2 grounding).

steer_actions grounds a (possibly OOD) target action onto the BC flow's in-distribution
manifold by fitting the target's noise preimage proposal and pushing it forward through
the flow. Runs in float32 (the inversion scan in agents/fql.py is not x64-safe)."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agents.fql import FQLAgent, get_config


@pytest.fixture(autouse=True)
def _force_float32():
    prev = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", False)
    yield
    jax.config.update("jax_enable_x64", prev)


def _tiny_agent(obs_dim=4, act_dim=2, flow_steps=100):
    cfg = get_config()
    cfg['flow_steps'] = flow_steps
    return FQLAgent.create(0, jnp.zeros((1, obs_dim)), jnp.zeros((1, act_dim)), cfg)


def test_steer_mean_grounds_in_distribution_target():
    """For an in-distribution target (produced by the flow itself), mean-mode steering
    with a sharp energy (high alpha) grounds it closely — the energy-proposal mean
    concentrates on the true preimage. Grounding tightness is governed by alpha (the
    inverse temperature): measured L2 ~0.57 @ alpha=1, ~0.18 @ alpha=5, ~0.008 @ alpha=20."""
    from utils.flow_steering import steer_actions

    agent = _tiny_agent(obs_dim=4, act_dim=2, flow_steps=100)
    obs = jax.random.normal(jax.random.PRNGKey(0), (8, 4))
    noise = jax.random.normal(jax.random.PRNGKey(1), (8, 2))
    targets = agent.compute_flow_actions(obs, noises=noise)  # in-distribution targets

    steered = steer_actions(agent, obs, targets, jax.random.PRNGKey(2), mode='mean',
                            n_steps=10, n_initial_steps=100, alpha=20.0)
    err = float(jnp.mean(jnp.linalg.norm(steered - targets, axis=-1)))
    assert steered.shape == targets.shape
    assert np.all(np.isfinite(np.asarray(steered)))
    assert err < 0.05, f"sharp-energy mean-mode steering did not ground the target (L2 {err})"


def test_steer_sample_shapes_and_finite():
    """Sample-mode steering returns finite steered actions of the right shape."""
    from utils.flow_steering import steer_actions

    agent = _tiny_agent(obs_dim=4, act_dim=2)
    obs = jax.random.normal(jax.random.PRNGKey(0), (6, 4))
    targets = jax.random.uniform(jax.random.PRNGKey(3), (6, 2), minval=-1.0, maxval=1.0)

    steered = steer_actions(agent, obs, targets, jax.random.PRNGKey(4), mode='sample',
                            num_samples=30, n_steps=3, n_initial_steps=10, num_clusters=2, alpha=1.0)
    assert steered.shape == (6, 2)
    assert np.all(np.isfinite(np.asarray(steered)))

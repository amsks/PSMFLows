"""The preimage EM target carries the flow's N(0, I) latent prior.

`prior_scale` weights it: 1.0 fits the posterior pi(u) ~ N(u; 0, I) exp(-alpha ||G(s,u) - a||),
0.0 the likelihood-only target the published npz files were computed under, whose fit
wandered out of the prior's typical set (docs/PREIMAGES.md).
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agents.fql import FQLAgent, get_config

OBS, ACT = 4, 3


@pytest.fixture(autouse=True)
def _force_float32():
    """The BC-flow inversion scan is not x64-safe; see tests/test_flow_inversion.py."""
    prev = jax.config.read('jax_enable_x64')
    jax.config.update('jax_enable_x64', False)
    yield
    jax.config.update('jax_enable_x64', prev)


def _agent(flow_steps=100):
    cfg = get_config()
    cfg['actor_hidden_dims'] = (32, 32)
    cfg['value_hidden_dims'] = (32, 32)
    cfg['flow_steps'] = flow_steps
    return FQLAgent.create(0, jnp.zeros((1, OBS)), jnp.zeros((1, ACT)), cfg)


def _fit(agent, prior_scale, n=16, seed=0):
    """Mixture means for `n` transitions under the given prior weight."""
    rng = jax.random.PRNGKey(seed)
    obs = jax.random.normal(jax.random.PRNGKey(seed + 1), (n, OBS))
    act = jnp.clip(jax.random.normal(jax.random.PRNGKey(seed + 2), (n, ACT)), -1, 1)
    keys = jax.random.split(rng, n)
    means, covs, weights, ess = jax.jit(jax.vmap(
        lambda s, a, k: agent.compute_full_proposal_distribution_em(
            s, a, k, num_samples=32, n_steps=3, n_initial_steps=100, alpha=20.0,
            n_components=1, prior_scale=prior_scale)
    ))(obs, act, keys)
    return np.asarray(means), np.asarray(covs), np.asarray(ess)


def test_prior_pulls_the_fit_toward_the_typical_set():
    agent = _agent()
    with_prior, _, _ = _fit(agent, prior_scale=1.0)
    without, _, _ = _fit(agent, prior_scale=0.0)

    sq_with = (with_prior[:, 0] ** 2).sum(-1)
    sq_without = (without[:, 0] ** 2).sum(-1)
    assert sq_with.mean() < sq_without.mean(), (
        f'the prior must shrink the fit toward 0: {sq_with.mean():.3f} vs {sq_without.mean():.3f}')
    assert np.isfinite(with_prior).all()


def test_prior_scale_zero_reproduces_the_likelihood_only_target():
    """The published npz files were computed under this branch; it must stay reachable."""
    agent = _agent()
    a, _, _ = _fit(agent, prior_scale=0.0, seed=3)
    b, _, _ = _fit(agent, prior_scale=0.0, seed=3)
    np.testing.assert_array_equal(a, b)


def test_default_is_the_posterior():
    """Callers that pass no prior_scale get the prior, including the precompute path."""
    agent = _agent()
    explicit, _, _ = _fit(agent, prior_scale=1.0, seed=5)

    rng = jax.random.PRNGKey(5)
    obs = jax.random.normal(jax.random.PRNGKey(6), (16, OBS))
    act = jnp.clip(jax.random.normal(jax.random.PRNGKey(7), (16, ACT)), -1, 1)
    keys = jax.random.split(rng, 16)
    default, _, _, _ = jax.jit(jax.vmap(
        lambda s, a, k: agent.compute_full_proposal_distribution_em(
            s, a, k, num_samples=32, n_steps=3, n_initial_steps=100, alpha=20.0,
            n_components=1)
    ))(obs, act, keys)
    np.testing.assert_allclose(np.asarray(default), explicit, rtol=1e-5, atol=1e-6)

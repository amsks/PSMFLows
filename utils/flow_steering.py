"""Energy-proposal flow steering (WP2 grounding).

Wraps Claas's preimage-proposal methods on `FQLAgent` into an inference-time action
path: given a (possibly out-of-distribution) target action, fit its noise preimage
distribution under the BC flow and push that noise forward, yielding action(s) that
lie on the flow's IN-DISTRIBUTION manifold near the target.

The target's preimage is defined by the energy pi(x) ~ exp(-alpha * ||flow(s,x) - a||):
- mode='mean'   : deterministic grounding — use the refined single-Gaussian proposal
                  mean as the noise, so steered ≈ nearest in-distribution action to the
                  target. This is the "project this action onto the behavior manifold" op.
- mode='sample' : stochastic — draw noise from the EM Gaussian-mixture preimage, giving
                  a distribution over in-distribution actions consistent with the target.

These are single-example agent methods; we vmap over the batch. Runs in float32 (the
inversion scan in agents/fql.py is not x64-safe).
"""
import jax
import jax.numpy as jnp
import numpy as np

from utils.flow_inversion import sample_preimage_noise


def steer_actions(agent, states, target_actions, rng, *, mode='mean',
                  num_samples=100, n_steps=10, n_initial_steps=100, alpha=1.0,
                  num_clusters=3):
    """Ground target actions onto the BC flow manifold via their noise preimage.

    Args:
        agent: an FQLAgent exposing compute_full_proposal_distribution[_em] and
            compute_flow_actions.
        states: (B, obs_dim) observations.
        target_actions: (B, action_dim) target actions to steer toward.
        rng: a jax PRNGKey (split per row).
        mode: 'mean' (deterministic, single-Gaussian proposal mean) or 'sample'
            (draw from the EM mixture preimage).
        num_samples, n_steps, n_initial_steps, alpha: proposal-refinement hyperparameters.
        num_clusters: mixture components for mode='sample' (ignored for mode='mean').

    Returns:
        steered: (B, action_dim) actions on the flow's in-distribution manifold.
    """
    states = jnp.asarray(states)
    target_actions = jnp.asarray(target_actions)
    keys = jax.random.split(rng, states.shape[0])

    if mode == 'mean':
        x0, _cov, _ess = jax.vmap(
            lambda s, a, k: agent.compute_full_proposal_distribution(
                s, a, k, num_samples=num_samples, n_steps=n_steps,
                n_initial_steps=n_initial_steps, alpha=alpha)
        )(states, target_actions, keys)
        noise = x0
    elif mode == 'sample':
        means, covs, weights, _ess = jax.vmap(
            lambda s, a, k: agent.compute_full_proposal_distribution_em(
                s, a, k, num_samples=num_samples, n_steps=n_steps,
                n_initial_steps=n_initial_steps, alpha=alpha, n_components=num_clusters)
        )(states, target_actions, keys)
        noise = jnp.asarray(
            sample_preimage_noise(np.asarray(means), np.asarray(covs), np.asarray(weights))
        )
    else:
        raise ValueError(f"mode must be 'mean' or 'sample', got {mode!r}")

    return agent.compute_flow_actions(states, noises=noise)

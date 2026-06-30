"""Utility functions to invert the flow model on a dataset"""

import jax
import numpy as np
from tqdm import trange

from utils.datasets import get_noise_preimage_dataset, get_size


def save_augmented_dataset(path, dataset):
    """Persist a preimage-augmented dataset (dict of arrays) to a compressed .npz."""
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in dataset.items()})


def load_augmented_dataset(path):
    """Load a preimage-augmented dataset previously written by `save_augmented_dataset`."""
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def sample_preimage_noise(means, covs, weights, rng=None):
    """Sample one preimage-noise vector per transition from its EM Gaussian mixture.

    For each row the component is drawn from the categorical mixture `weights`, then the noise is
    drawn from that component's Gaussian as `mu + L z` with `L = cholesky(cov)` and `z ~ N(0, I)`.

    Args:
        means: (B, K, A) component means.
        covs: (B, K, A, A) component covariances.
        weights: (B, K) mixture weights (rows should sum to 1).
        rng: Optional `np.random.Generator`; defaults to the global `np.random` state.

    Returns:
        noise: (B, A) float32 array, one sampled latent noise per transition.
    """
    rand = np.random if rng is None else rng
    B, K, A = means.shape

    # Pick a component per row via the categorical mixture weights.
    cdf = np.cumsum(weights, axis=1)
    cdf = cdf / cdf[:, -1:]  # guard against small normalization drift
    comp = (rand.random((B, 1)) > cdf).sum(axis=1)
    comp = np.clip(comp, 0, K - 1)

    rows = np.arange(B)
    chosen_mean = means[rows, comp]  # (B, A)
    chosen_cov = covs[rows, comp]  # (B, A, A)

    # x = mu + L z, with L the Cholesky factor of the chosen covariance.
    L = np.linalg.cholesky(chosen_cov)  # (B, A, A)
    z = rand.standard_normal((B, A))
    noise = chosen_mean + np.einsum('bij,bj->bi', L, z)
    return noise.astype(np.float32)


def augment_dataset_with_preimage_distribution(agent, dataset, config):
    """Precompute the noise preimage of each action under the BC flow model.

    For every transition the preimage of `action` (in the latent noise space) is fit with a
    Gaussian mixture via `agent.compute_full_proposal_distribution_em` and stored back into the
    dataset under the `noise_preimage_{mean,cov,weights}` slots.

    Args:
        agent: A trained agent exposing `compute_full_proposal_distribution_em`.
        dataset: A dataset (dict-like) with `observations` and `actions`.
        config: Inversion config (its own namespaced group; read via `.get`).

    Returns:
        The dataset (plain dict) with the noise-preimage slots populated.
    """
    # Hyperparameters (from the dedicated inversion config, with defaults).
    num_clusters = config.get('num_clusters', 1)
    alpha = config.get('alpha', 1.0)
    num_samples = config.get('num_samples', 100)
    n_steps = config.get('n_steps', 10)
    n_initial_steps = config.get('n_initial_steps', 100)
    batch_size = config.get('batch_size', 256)
    seed = config.get('seed', 0)

    assert num_samples >= num_clusters, (
        f'num_samples ({num_samples}) must be >= num_clusters ({num_clusters}); '
        'EM draws num_samples // n_components samples per component.'
    )

    # Allocate the noise-preimage slots (writable numpy arrays).
    dataset = get_noise_preimage_dataset(dataset, num_clusters=num_clusters)
    size = get_size(dataset)

    def _em_single(state, action, rng):
        return agent.compute_full_proposal_distribution_em(
            state, action, rng,
            num_samples=num_samples,
            n_steps=n_steps,
            n_initial_steps=n_initial_steps,
            alpha=alpha,
            n_components=num_clusters,
        )

    _em_batch = jax.jit(jax.vmap(_em_single))

    rng = jax.random.PRNGKey(seed)
    for start in trange(0, size, batch_size, desc='Inverting flow'):
        end = min(start + batch_size, size)
        rng, batch_rng = jax.random.split(rng)
        keys = jax.random.split(batch_rng, end - start)
        means, covs, weights, _ess = _em_batch(
            dataset['observations'][start:end],
            dataset['actions'][start:end],
            keys,
        )
        dataset['noise_preimage_mean'][start:end] = np.asarray(means)
        dataset['noise_preimage_cov'][start:end] = np.asarray(covs)
        dataset['noise_preimage_weights'][start:end] = np.asarray(weights)

    return dataset

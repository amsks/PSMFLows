"""Utility functions to invert the flow model on a dataset"""

import jax
import jax.numpy as jnp
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


#: Per-row 1.0/0.0 mask over every preimage product. Written by the precompute, recomputed
#: on the fly for npz files that predate it.
PREIMAGE_VALID_KEY = 'preimage_valid'

#: Above this invalid fraction, a few diverged rows have stopped being a tail artifact and
#: the inversion itself is wrong; repairing that many rows would hide a real failure.
MAX_INVALID_PREIMAGE_FRAC = 0.01


def compute_preimage_validity(dataset):
    """Per-row mask: is every preimage product for this transition finite?

    The flow inverse can diverge (see `FQLAgent._get_preimage_and_jacobian`), and NaN
    propagates from there into the mixture, the point preimage and the round-trip alike.
    The mask is the CONJUNCTION across all arms present, deliberately: `use_point_preimage`
    selects an arm at training time, and a shared row set keeps the point-vs-mixture
    ablation a comparison of representations rather than of two different datasets.

    Returns:
        valid: (N,) float32 in {0.0, 1.0}.
    """
    size = get_size(dataset)
    valid = np.ones((size,), bool)
    for key in ('noise_preimage_mean', 'noise_preimage_cov', 'noise_preimage_weights',
                'noise_preimage_point', 'preimage_roundtrip', 'preimage_ess'):
        if key in dataset:
            arr = np.asarray(dataset[key])
            valid &= np.isfinite(arr).reshape(size, -1).all(axis=1)
    if 'noise_preimage_point' in dataset:
        # Finiteness is not enough for the point arm: the implicit-Euler inverse can blow
        # up to huge FINITE values (measured ||u*||^2 ~ 1e59 on 8/1M alpha=1 cube rows)
        # without reaching NaN. chi^2_{d_a} mass above 100 is ~0 for any action dim here,
        # so every such row is a diverged inverse, not an atypical-but-real latent.
        sq = (np.asarray(dataset['noise_preimage_point'], np.float64) ** 2).sum(-1)
        valid &= ~(sq > 100.0)
    return valid.astype(np.float32)


def repair_invalid_preimages(dataset, max_invalid_frac=MAX_INVALID_PREIMAGE_FRAC):
    """Make rows whose inversion diverged safe to train on, loudly and in bounded number.

    A NaN latent means the inverse diverged, so we do not know which `u` produced that
    action. Rows CANNOT simply be dropped: `Dataset.sample` pairs each transition with row
    `idx + 1` for the next-policy index, so deleting rows would silently re-pair transitions
    across the gap. Instead the mixture arm is reset to the standard normal PRIOR
    (mean 0, cov I) — the honest posterior when the data carries no information about `u` —
    and the point arm, which has no distributional fallback, to 0. Both are recorded in
    `preimage_valid` so a caller can mask them.

    This is only defensible because the count is tiny; past `max_invalid_frac` it would be
    papering over a broken inversion, so that aborts instead.

    Returns:
        (dataset, valid): the dataset with `preimage_valid` set and invalid rows reset.
    """
    valid = dataset.get(PREIMAGE_VALID_KEY)
    valid = compute_preimage_validity(dataset) if valid is None else np.asarray(valid)
    bad = valid < 0.5
    n_bad = int(bad.sum())
    if n_bad:
        frac = n_bad / len(valid)
        assert frac <= max_invalid_frac, (
            f'{n_bad}/{len(valid)} preimages ({frac:.4%}) are non-finite, above the '
            f'{max_invalid_frac:.2%} ceiling. That is no longer a divergence tail — re-run '
            'tools/precompute_preimages.py (a higher inversion.n_initial_steps reduces it) '
            'rather than training on repaired rows.'
        )
        print(f'WARNING: {n_bad}/{len(valid)} preimages ({frac:.4%}) diverged to non-finite '
              f'and were reset to the N(0, I) prior; see preimage_valid.')
        if 'noise_preimage_mean' in dataset:
            dataset['noise_preimage_mean'][bad] = 0.0
        if 'noise_preimage_cov' in dataset:
            d_a = dataset['noise_preimage_cov'].shape[-1]
            dataset['noise_preimage_cov'][bad] = np.eye(d_a, dtype=np.float32)
        if 'noise_preimage_weights' in dataset:
            k = dataset['noise_preimage_weights'].shape[-1]
            dataset['noise_preimage_weights'][bad] = 1.0 / k
        if 'noise_preimage_point' in dataset:
            dataset['noise_preimage_point'][bad] = 0.0
        for key in ('preimage_roundtrip', 'preimage_ess'):
            if key in dataset:
                dataset[key][bad] = 0.0
    dataset[PREIMAGE_VALID_KEY] = valid
    return dataset, valid


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
    # Per-transition health scalar: how informative the epsilon-relaxed preimage posterior
    # is. Persisted so a training run can tell whether its inversion was trustworthy
    # without re-running diagnostic D3. Reference: mean ~7/100 on cube-single at every
    # flow_steps setting, so treat a much lower value as the anomaly, not 7 itself.
    dataset['preimage_ess'] = np.zeros((size,), np.float32)

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
        means, covs, weights, ess = _em_batch(
            dataset['observations'][start:end],
            dataset['actions'][start:end],
            keys,
        )
        dataset['noise_preimage_mean'][start:end] = np.asarray(means)
        dataset['noise_preimage_cov'][start:end] = np.asarray(covs)
        dataset['noise_preimage_weights'][start:end] = np.asarray(weights)
        # `ess` is (B, n_steps) — the scan stacks one value per EM iteration. Keep the
        # LAST, which is the health of the mixture actually stored above; an earlier
        # iterate describes a proposal that was then discarded.
        dataset['preimage_ess'][start:end] = np.asarray(ess[:, -1])

    return dataset


def augment_dataset_with_point_preimage(agent, dataset, config):
    """Store the exact backward-ODE preimage of each action and its decode round-trip error.

    Adds `noise_preimage_point (N, A)` and `preimage_roundtrip (N,)`. The point preimage
    enables the point-vs-mixture ablation (agent config `use_point_preimage`): the mixture
    indexes a policy by a DISTRIBUTION over latents, the point by the single exact preimage.

    `n_initial_steps` sets the inversion discretization. It must be >= 100: the
    implicit-Euler fixed point diverges outright at the training default of 10 (diagnostic
    D3, cube-single: NaN round-trip at 10, KS 0.217 at 30, 1.2e-4 at 100).

    NOTE: inversion runs at `n_initial_steps` but the round-trip decode runs at the
    AGENT's `flow_steps`. Round-trip consistency only holds when the forward and inverse
    maps share a discretization, so if the two disagree `preimage_roundtrip` measures that
    mismatch rather than inversion quality. tools/precompute_preimages.py asserts they are
    equal before writing a dataset; callers that bypass it must match them by hand.

    Args:
        agent: A trained agent exposing `_get_preimage_and_jacobian` and `compute_flow_actions`.
        dataset: A dataset (dict-like) with `observations` and `actions`.
        config: Inversion config (its own namespaced group; read via `.get`).

    Returns:
        A new dict: the input dataset plus the point-preimage and round-trip slots.
    """
    n_steps = config.get('n_initial_steps', 100)
    batch_size = config.get('batch_size', 256)
    size = get_size(dataset)
    d_a = dataset['actions'].shape[-1]
    point = np.zeros((size, d_a), np.float32)
    roundtrip = np.zeros((size,), np.float32)

    # jit(vmap), NOT bare vmap: on GPU an un-jitted vmap of this function (lax.scan over an
    # inner fori_loop, plus jacfwd) returns all-NaN, reproducibly, while the same call
    # jitted / python-looped / on CPU is correct.
    _points = jax.jit(jax.vmap(lambda s, a: agent._get_preimage_and_jacobian(s, a, n_steps)[0]))
    for start in trange(0, size, batch_size, desc='Point preimages'):
        end = min(start + batch_size, size)
        obs = dataset['observations'][start:end]
        act = dataset['actions'][start:end]
        x0 = _points(obs, act)
        recon = agent.compute_flow_actions(obs, noises=x0)
        point[start:end] = np.asarray(x0)
        # Compare against the CLIPPED action: compute_flow_actions clips its output to
        # [-1, 1], so an unclipped target would charge the inverter for the clip.
        roundtrip[start:end] = np.asarray(
            jnp.linalg.norm(recon - jnp.clip(jnp.asarray(act), -1, 1), axis=-1))
    return {**dataset, 'noise_preimage_point': point, 'preimage_roundtrip': roundtrip}

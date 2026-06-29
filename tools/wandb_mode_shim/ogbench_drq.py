"""JAX/Flax DrQ-v2 visual front-end for vendored OGBench (GCIQL/GCIVL).

Imported by the sitecustomize shim inside the OGBench child process (jax env).
Mirrors FB's PyTorch DrQEncoder (nn_models.py:480) and Augmentator
(nn_models.py:530) so OGBench's pixel baselines share FB's encoder, augmentation,
frame-stacking and normalization. Never edits vendored OGBench.
"""
import flax.linen as nn
import jax.numpy as jnp
import numpy as np


class DrQEncoder(nn.Module):
    """DrQ-v2 RGB encoder (Flax), numerically matching FB's torch DrQEncoder.

    Normalizes internally (x/255 - 0.5, == FB's RGBNorm) because OGBench encoders
    own normalization (cf. ImpalaEncoder's x/255). __call__ takes (train, cond_var)
    for signature parity with ImpalaEncoder; both are ignored.
    """

    feature_dim: int = 256

    @nn.compact
    def __call__(self, x, train=True, cond_var=None):
        init = nn.initializers.xavier_uniform()
        x = x.astype(jnp.float32) / 255.0 - 0.5
        x = nn.relu(nn.Conv(32, (3, 3), strides=2, padding="VALID",
                            kernel_init=init, name="conv0")(x))
        x = nn.relu(nn.Conv(32, (3, 3), strides=1, padding="VALID",
                            kernel_init=init, name="conv1")(x))
        x = nn.relu(nn.Conv(32, (3, 3), strides=1, padding="VALID",
                            kernel_init=init, name="conv2")(x))
        x = nn.relu(nn.Conv(32, (3, 3), strides=1, padding="VALID",
                            kernel_init=init, name="conv3")(x))
        x = x.reshape((*x.shape[:-3], -1))
        x = nn.Dense(self.feature_dim, kernel_init=init, name="proj")(x)
        x = nn.LayerNorm(epsilon=1e-5, name="proj_ln")(x)  # 1e-5 == torch default
        return jnp.tanh(x)


def _shift_batch_np(arr, shifts, pad):
    """Replicate-pad by `pad`, then integer-crop each image at its own shift.

    arr: (N, H, W, C); shifts: (N, 2) ints in [0, 2*pad]. Pure NumPy (CPU): a
    single vectorized gather, bit-identical to FB's random_shifts (edge-pad +
    crop). Staying on the host lets augmentation overlap the jitted device step
    instead of forcing a per-key host<->device round-trip every train step.
    """
    n, h, w, _ = arr.shape
    padded = np.pad(arr, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode="edge")
    rows = shifts[:, 0:1] + np.arange(h)  # (N, H)
    cols = shifts[:, 1:2] + np.arange(w)  # (N, W)
    ni = np.arange(n)[:, None, None]
    return padded[ni, rows[:, :, None], cols[:, None, :], :]


def random_shifts_batch(batch, keys, pad=2, rng=None):
    """In-place FB-faithful random_shifts: replicate-pad by `pad`, integer-crop.

    Independent per-key shifts (matches FB's separate _augmentator calls). Only
    4D arrays (N,H,W,C) are augmented; other keys pass through. `rng` is an
    np.random.Generator (None -> fresh) so callers/tests can fix the seed.
    """
    g = np.random.default_rng() if rng is None else rng
    for key in keys:
        arr = batch[key]
        if getattr(arr, "ndim", 0) != 4:
            continue
        n = arr.shape[0]
        shifts = g.integers(0, 2 * pad + 1, size=(n, 2))
        batch[key] = _shift_batch_np(np.asarray(arr), shifts, pad)
    return batch


def gcdataset_augment(self, batch, keys):
    """Drop-in replacement for GCDataset.augment (pad=2, FB-faithful)."""
    random_shifts_batch(batch, keys, pad=2)


gcdataset_augment._ogbench_drq_patched = True  # sentinel for the injection test

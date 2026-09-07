"""Pure helpers shared by the measure agents — no agent state, no networks.

`contrastive_loss` and `ortho_loss` are the PSM/FB measure objective and its
orthonormality regulariser (PSM Eq. 7, Table 2); `targets_uncertainty` is the ensemble
mean and spread the pessimism penalties are built from; `project_z`, `off_diagonal_mask`
and `polyak_update` are one-liners with a single sensible definition. `proto_sample`,
`_HashableDict`, `_step` and `_soft` are read only by the agents under archive/.
"""

import jax
import jax.numpy as jnp
import optax


def contrastive_loss(M, target_M, discount, off_diag, off_diag_sum):
    """PSM Eq. 7. The diagonal term takes diag(diff) x num_parallel, not (1-g)*diag(M)."""
    diff = M - discount * target_M
    offdiag = 0.5 * jnp.sum((diff * off_diag) ** 2) / off_diag_sum
    diag = -jnp.mean(jnp.diagonal(diff, axis1=1, axis2=2)) * M.shape[0]
    return offdiag + diag, diag, offdiag


def ortho_loss(phi, off_diag, off_diag_sum):
    """Orthonormality regulariser: enforces E_rho[phi phi^T] = I.

    That identity is the condition under which the closed-form task vector
    w = E_D[r phi] is the correct least-squares projection (Cor. `reward-inference`).
    """
    cov = phi @ phi.T
    offdiag = 0.5 * jnp.sum((cov * off_diag) ** 2) / off_diag_sum
    diag = -jnp.mean(jnp.diagonal(cov))
    return offdiag + diag, diag, offdiag


def targets_uncertainty(preds, num_parallel):
    """Ensemble mean over axis 0, and the mean pairwise absolute spread around it."""
    mean = preds.mean(axis=0)
    d1 = preds[None]
    d2 = preds[:, None]
    unc = jnp.sum(jnp.abs(d1 - d2), axis=(0, 1)) / (num_parallel ** 2 - num_parallel)
    return mean, unc


def proto_sample(seed_to_action, powers, obs_hash, z, max_seed):
    """PSM Eq. 8, the codebook policy pi(a|s,z) = UniformSample(seed = z + hash(s))."""
    seed_long = jnp.sum(z * powers, axis=1)
    final = ((seed_long + obs_hash.reshape(-1)) % max_seed).astype(jnp.int32)
    return seed_to_action[final].astype(jnp.float32)


def project_z(z, norm_z):
    """Project a task vector onto the sphere of radius sqrt(d), or pass it through."""
    if not norm_z:
        return z
    d = z.shape[-1]
    return jnp.sqrt(d) * z / (jnp.linalg.norm(z, axis=-1, keepdims=True) + 1e-12)


def off_diagonal_mask(B):
    """(1 - I, its sum): selects the s+ != s (off-diagonal) entries of the BxB grid."""
    off = 1.0 - jnp.eye(B)
    return off, off.sum()


def polyak_update(online, target, tau):
    """Soft-update a target param tree toward `online` at rate `tau`."""
    return jax.tree_util.tree_map(lambda p, tp: p * tau + tp * (1 - tau), online, target)


def _plain_config(x):
    """Recursively convert ConfigDict/dict -> plain dict and list -> tuple, so the
    resulting FrozenDict is hashable (required for the jitted `update`'s static aux)."""
    if isinstance(x, (list, tuple)):
        return tuple(_plain_config(v) for v in x)
    if hasattr(x, "items"):
        return {k: _plain_config(v) for k, v in x.items()}
    return x


class _HashableDict(dict):
    """A dict that hashes/compares by identity so it can live in a jit static aux."""

    __hash__ = object.__hash__

    def __eq__(self, other):
        return self is other


def _step(tx, grad, params, opt_state):
    """One optax step on a raw (params, opt_state) pair."""
    updates, new_opt = tx.update(grad, opt_state, params)
    return optax.apply_updates(params, updates), new_opt


def _soft(online, target, tau):
    """`polyak_update` under the name the archived agents import."""
    return jax.tree_util.tree_map(lambda p, tp: p * tau + tp * (1 - tau), online, target)

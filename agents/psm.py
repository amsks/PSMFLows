"""PSM (Proto Successor Measure) agent — JAX/Flax port of the PyTorch reference.

This module hosts the pure loss/uncertainty/proto helpers (Task 5) and the
PSMAgent (Task 6). Math/op-order is transcribed verbatim from the reference
agents/psm/{agent,model,proto_sampler}.py.
"""

import jax.numpy as jnp


def contrastive_loss(M, target_M, discount, off_diag, off_diag_sum):
    """Reference SM-TD contrastive loss (agent.py:170-173).

    M: [P, B, B], target_M: [B, B], discount: [B, 1] (row-wise) or scalar.
    Returns (loss, diag, offdiag).
    """
    diff = M - discount * target_M
    offdiag = 0.5 * jnp.sum((diff * off_diag) ** 2) / off_diag_sum
    diag = -jnp.mean(jnp.diagonal(diff, axis1=1, axis2=2)) * M.shape[0]
    return offdiag + diag, diag, offdiag


def ortho_loss(phi, off_diag, off_diag_sum):
    """Reference phi orthonormality loss (agent.py:176-179). phi: [B, z_dim]."""
    cov = phi @ phi.T
    offdiag = 0.5 * jnp.sum((cov * off_diag) ** 2) / off_diag_sum
    diag = -jnp.mean(jnp.diagonal(cov))
    return offdiag + diag, diag, offdiag


def targets_uncertainty(preds, num_parallel):
    """Reference get_targets_uncertainty (model.py:112-116). preds: [P, ...]."""
    mean = preds.mean(axis=0)
    d1 = preds[None]
    d2 = preds[:, None]
    scaling = num_parallel ** 2 - num_parallel
    unc = jnp.sum(jnp.abs(d1 - d2), axis=(0, 1)) / scaling
    return mean, unc


def proto_sample(seed_to_action, powers, obs_hash, z, max_seed):
    """Reference ProtoBehaviorSampler.forward (proto_sampler.py:33-38)."""
    seed_long = jnp.sum(z * powers, axis=1)
    final = ((seed_long + obs_hash.reshape(-1)) % max_seed).astype(jnp.int32)
    # Reference returns torch.FloatTensor(...) i.e. float32; match that cast exactly.
    return seed_to_action[final].astype(jnp.float32)

"""The reference PSM proto policy family, on latent inputs. Pure functions, no agent state.

PSM (arXiv 2411.19418) learns its basis phi from the successor measures of a whole family
of fixed pseudo-random behaviour policies at once. A policy in that family is indexed by a
binary code `z_bin` of width `max_log_seed`, and its action at a dataset row is a
deterministic function of (row, code): the reference (`ProtoBehaviorSampler`) forms an
integer seed `(z_bin . powers) + row mod max_seed` and gathers a precomputed uniform action
table at that seed. `archive/agents/psm.py` and Factored-FB `impls/critics/psm.py` are the
two ports this file copies.

The one change for LatentFlowPSM: the proto policy emits a LATENT, not an action. The
same seed indexes a clipped standard-normal draw in the u box instead of a row of a
uniform action table, so the proto policy is `pi_z(s) = G(s, u_proto(row, z_bin))` -- a
fixed pseudo-random latent policy -- and every bootstrap action stays a flow decode of a
prior-typical latent. Everything else (code layout, powers, the seed arithmetic and the
`+ 20000` in max_seed) is the reference's, verbatim.
"""

import jax
import jax.numpy as jnp


def proto_max_seed(max_log_seed):
    """Reference `ProtoBehaviorSampler.max_seed`: 2**max_log_seed + 20000."""
    return 2 ** int(max_log_seed) + 20000


def sample_z_bin(key, batch_size, max_log_seed):
    """Binary proto codes, (B, max_log_seed) float32 in {0, 1}, reference layout.

    Reference `PSMModel.sample_z_psm`: a uniform integer in [0, 2**max_log_seed), unpacked
    LSB-first -- column k is bit k of the code.
    """
    codes = jax.random.randint(key, (batch_size,), 0, 2 ** int(max_log_seed))
    return ((codes[:, None] >> jnp.arange(int(max_log_seed))) & 1).astype(jnp.float32)


def proto_seed_ints(z_bin, row_index, max_log_seed):
    """Reference `ProtoBehaviorSampler.forward` seed arithmetic, verbatim.

    `powers` is the REVERSED power list, so the dot product reads the LSB-first bit array
    MSB-first; the row index is the dataset ROW (batch['index']), not the batch position,
    so the same transition meets the same proto policy on every resample. Returns (B,) int32.
    """
    w = int(max_log_seed)
    powers = 2 ** jnp.arange(w)[::-1]
    row = jnp.asarray(row_index).reshape(-1).astype(jnp.int32)
    return ((z_bin.astype(jnp.int32) * powers).sum(1) + row) % proto_max_seed(w)


def proto_latents(seed_ints, action_dim, u_clip, base_key):
    """The proto policy's LATENT at each seed: clip(N(0, I) at fold_in(base_key, seed), +-u_clip).

    A pure function of the integer seed, hence of (row, z_bin): the same table lookup the
    reference does, evaluated lazily with threefry instead of materialised. `base_key` is
    PRNGKey(proto.proto_seed), a constant of the code and NOT the run seed, so the proto
    policy family is identical across seeds as the reference's table is. Returns (B, d_a).
    """
    d_a = int(action_dim)

    def one(s):
        return jnp.clip(jax.random.normal(jax.random.fold_in(base_key, s), (d_a,)), -u_clip, u_clip)

    return jax.vmap(one)(seed_ints.astype(jnp.uint32))

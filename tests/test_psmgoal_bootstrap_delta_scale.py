"""bootstrap_delta_scale: scales the flowbc actor's step inside the TD bootstrap only (2026-10-15)."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import utils.xla_guard  # noqa: F401  (must precede jax)
from tests.test_psmgoal_joint_actor import _agent, _batch

OVER = dict(policy_index='goal', bootstrap_source='actor', train_actor=True, actor_input='goal',
            measure_loss='softmax', actor_kind='flowbc', fb_bc_coeff=3.0, actor_value_kind='point')


def _nonzero_delta(agent):
    """The flowbc delta is zero at init; perturb its params so the step is non-zero."""
    leaves, tree = jax.tree_util.tree_flatten(agent.delta.params)
    keys = jax.random.split(jax.random.PRNGKey(3), len(leaves))
    new = [x + 0.05 * jax.random.normal(k, x.shape, x.dtype) for x, k in zip(leaves, keys)]
    return agent.replace(delta=agent.delta.replace(params=jax.tree_util.tree_unflatten(tree, new)))


def _z(agent, batch):
    n = np.asarray(batch['observations']).shape[0]
    return jnp.zeros((n, int(agent.config['z_dim'])), jnp.float32)


@pytest.mark.parametrize('m', [0.5, 2.0, 4.0])
def test_bootstrap_scaled_step(m):
    base = _nonzero_delta(_agent(**OVER))
    scaled = _agent(bootstrap_delta_scale=m, **OVER).replace(delta=base.delta, rng=base.rng)
    batch = _batch()
    u1 = np.asarray(base._select_u_next(batch, _z(base, batch)))
    um = np.asarray(scaled._select_u_next(batch, _z(scaled, batch)))
    assert u1.shape == um.shape and np.all(np.isfinite(um))
    assert not np.allclose(u1, um)
    clip = float(base.config['u_clip'])
    assert np.all(np.abs(um) <= clip + 1e-6)


def test_scale_one_is_identical():
    base = _nonzero_delta(_agent(**OVER))
    one = _agent(bootstrap_delta_scale=1.0, **OVER).replace(delta=base.delta, rng=base.rng)
    batch = _batch()
    np.testing.assert_array_equal(np.asarray(base._select_u_next(batch, _z(base, batch))),
                                  np.asarray(one._select_u_next(batch, _z(one, batch))))

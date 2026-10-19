"""Pure helpers of the pool measure (docs/design/2026-10-19-pool-measure.md): the reference pool R
drawn from the valid-preimage rows, its hash, the per-dimension scale, and the nearest-pool anchor."""
import jax.numpy as jnp
import numpy as np
import pytest

import utils.xla_guard  # noqa: F401  (must precede jax)
from utils.psm_common import draw_pool_indices, pool_anchor, pool_hash, pool_sigma


def test_draw_pool_is_deterministic_sorted_without_replacement():
    rows = np.arange(100, 1100)
    a = draw_pool_indices(rows, 64, 0)
    b = draw_pool_indices(rows, 64, 0)
    c = draw_pool_indices(rows, 64, 1)
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    assert a.dtype == np.int64 and len(a) == 64
    assert len(set(a.tolist())) == 64 and np.all(np.diff(a) > 0) and set(a.tolist()) <= set(rows.tolist())


def test_draw_pool_refuses_oversize_and_empty():
    with pytest.raises(AssertionError):
        draw_pool_indices(np.arange(10), 11, 0)
    with pytest.raises(AssertionError):
        draw_pool_indices(np.arange(10), 0, 0)


def test_pool_hash_changes_with_indices():
    assert pool_hash(np.arange(5)) != pool_hash(np.arange(1, 6))
    assert len(pool_hash(np.arange(5))) == 40
    # dtype-insensitive: the same rows as int32 or int64 hash the same
    assert pool_hash(np.arange(5, dtype=np.int32)) == pool_hash(np.arange(5, dtype=np.int64))


def test_anchor_is_nearest_in_standardised_units_and_first_on_ties():
    pool = jnp.asarray([[0.0, 0.0], [10.0, 0.0], [0.0, 1.0]])
    sigma = jnp.asarray([10.0, 1.0])              # x is 10x wider than y
    # row 0: x=6 is 0.6 sd from x=0 and 0.4 sd from x=10 -> pool 1 (not a tie: nearest wins)
    # row 1: 0.5 sd from pool 0 and from pool 2 -> tie -> first index 0
    # row 2: exactly between pool 0 and pool 1 in x -> tie -> 0
    nxt = jnp.asarray([[6.0, 0.0], [0.0, 0.5], [5.0, 0.0]])
    a = pool_anchor(nxt, pool, sigma)
    assert a.dtype == jnp.int32
    assert a.tolist() == [1, 0, 0]
    # the scale decides: (6, 0.9) is nearest (10, 0) in raw units and nearest (0, 1) when y is expensive
    a2 = pool_anchor(jnp.asarray([[6.0, 0.9]]), pool, jnp.asarray([1.0, 1.0]))
    assert a2.tolist() == [1]
    a3 = pool_anchor(jnp.asarray([[6.0, 0.9]]), pool, jnp.asarray([100.0, 0.01]))
    assert a3.tolist() == [2]


def test_pool_sigma_floor_and_value():
    s = pool_sigma(jnp.zeros((4, 3)))
    assert s.shape == (3,) and float(s.min()) >= float(np.float32(1e-6))
    x = jnp.asarray([[0.0, 1.0], [2.0, 1.0]])
    np.testing.assert_allclose(np.asarray(pool_sigma(x)), [1.0, 1e-6], rtol=1e-6)

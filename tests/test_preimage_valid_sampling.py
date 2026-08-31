"""Rows whose inversion diverged must not reach a preimage-consuming loss.

`repair_invalid_preimages` (utils/flow_inversion) neutralizes diverged rows so a NaN
latent cannot poison an update: the point preimage becomes u = 0, the mixture becomes
N(0, I). That keeps training alive, but u = 0 is a wrong value rather than a missing
one -- G(s, 0) != a for those transitions -- and until now `preimage_valid` was recorded
and then never read by anything downstream, so those rows trained as if they were real.
Counts are small (cube 13/1M, antmaze 881/1M) and antmaze is the one that matters.

Pinned here: invalid rows are never sampled, the all-valid case is untouched (so every
run without diverged rows is bit-identical), and the filter is off unless the batch
actually carries preimage latents.
"""
import numpy as np
import pytest

from utils.datasets import Dataset
from utils.flow_inversion import PREIMAGE_VALID_KEY

N, OBS, ACT = 64, 3, 2
BAD = np.array([5, 17, 40])


def _dataset(with_valid=True, all_valid=False):
    rng = np.random.default_rng(0)
    valid = np.ones(N, np.float32)
    if not all_valid:
        valid[BAD] = 0.0
    fields = dict(
        observations=rng.standard_normal((N, OBS)).astype(np.float32),
        next_observations=rng.standard_normal((N, OBS)).astype(np.float32),
        actions=rng.standard_normal((N, ACT)).astype(np.float32),
        terminals=np.zeros(N, np.float32),
        noise_preimage_point=rng.standard_normal((N, ACT)).astype(np.float32),
    )
    if with_valid:
        fields[PREIMAGE_VALID_KEY] = valid
    ds = Dataset.create(**fields)
    ds.return_preimage_noise = True
    ds.preimage_point_mode = True
    return ds


def test_invalid_rows_are_never_sampled():
    ds = _dataset()
    np.random.seed(0)
    seen = set()
    for _ in range(200):
        seen.update(np.asarray(ds.sample(32, )["index"]) if ds.return_index
                    else np.asarray(ds.get_random_idxs(32)))
    assert seen, "sampled nothing"
    assert not (seen & set(BAD.tolist())), sorted(seen & set(BAD.tolist()))
    # ... and the valid rows are still all reachable, i.e. this is a filter, not a slice.
    assert len(seen) == N - len(BAD)


def test_all_valid_leaves_the_draw_untouched():
    """No invalid row -> the numpy stream and the indices are exactly the old ones."""
    ds = _dataset(all_valid=True)
    np.random.seed(0)
    got = ds.get_random_idxs(64)
    np.random.seed(0)
    want = np.random.randint(N, size=64)
    assert np.array_equal(got, want)
    assert ds._valid_preimage_rows() is None


def test_filter_is_off_without_preimages():
    """An agent not consuming preimage latents keeps the plain uniform draw."""
    ds = _dataset()
    ds.return_preimage_noise = False
    np.random.seed(0)
    got = ds.get_random_idxs(64)
    np.random.seed(0)
    assert np.array_equal(got, np.random.randint(N, size=64))


def test_missing_valid_key_is_tolerated():
    """npz files written before preimage_valid existed still train."""
    ds = _dataset(with_valid=False)
    assert ds._valid_preimage_rows() is None
    assert ds.get_random_idxs(8).shape == (8,)


def test_sampled_batches_carry_only_valid_preimages():
    """End to end: the served noise_preimage never comes from a repaired row."""
    ds = _dataset()
    # Mark the repaired rows the way repair_invalid_preimages does, then check no batch
    # ever serves that sentinel.
    pt = np.array(ds["noise_preimage_point"], copy=True)
    pt[BAD] = 0.0
    fields = {k: (pt if k == "noise_preimage_point" else v) for k, v in ds.items()}
    ds2 = Dataset.create(**fields)
    ds2.return_preimage_noise = True
    ds2.preimage_point_mode = True
    np.random.seed(0)
    for _ in range(100):
        u = np.asarray(ds2.sample(32)["noise_preimage"])
        assert not np.any(np.all(u == 0.0, axis=-1)), "a repaired row reached the batch"

"""Neighbourhood sampling for tuning batches.

A uniform sample of a large buffer has no neighbourhood structure left, which silently
turns any "what do nearby transitions share" metric into a statement about arbitrary
pairs. `neighborhood_sample` draws whole neighbourhoods instead, so the local density of
the batch matches the buffer's.
"""
import numpy as np

from utils.nn_search import knn_indices, neighborhood_sample


def _clustered_points(n_clusters=20, per_cluster=50, dim=3, seed=0):
    """Well-separated clusters: local density is unambiguous."""
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-50, 50, (n_clusters, dim)).astype(np.float32)
    return (centers[:, None] + 0.05 * rng.standard_normal((n_clusters, per_cluster, dim))
            ).reshape(-1, dim).astype(np.float32)


def test_sample_size_and_uniqueness():
    points = _clustered_points()
    rows = neighborhood_sample(points, n_rows=200, k=20, seed=0)
    assert len(rows) == len(np.unique(rows))
    assert np.all(np.diff(rows) > 0), 'indices must come back sorted'
    # 200 // 20 = 10 anchors x 20 neighbours, minus whatever the overlaps merge away.
    assert 0 < len(rows) <= 200


def test_batch_preserves_local_density_where_uniform_does_not():
    """The gap grows as (N/n)^(1/d), so it needs a realistically sparse sample to show.

    Measured on pointmaze (d=2, 512 rows of 1M) the uniform batch's nearest neighbour sits
    40x further out than the buffer's; this fixture reproduces that regime in miniature.
    """
    rng = np.random.default_rng(0)
    points = rng.uniform(-1, 1, (40_000, 2)).astype(np.float32)
    n_rows, k = 200, 20

    nbh = neighborhood_sample(points, n_rows, k, seed=0)
    uni = np.sort(rng.choice(len(points), n_rows, replace=False))

    def in_batch_nn(rows):
        return np.median(knn_indices(points[rows], points[rows], 2)[1][:, 1])

    full_nn = np.median(knn_indices(points, points, 2)[1][:, 1])
    assert in_batch_nn(nbh) < 3 * full_nn, 'neighbourhood batch must keep the buffer density'
    assert in_batch_nn(uni) > 5 * in_batch_nn(nbh), 'uniform batch should be far sparser'


def test_neighbourhoods_are_actual_neighbourhoods():
    """Every sampled row belongs to some anchor's k nearest, so rows come in clumps."""
    points = _clustered_points(n_clusters=20, per_cluster=50)
    rows = neighborhood_sample(points, n_rows=100, k=20, seed=1)
    # Cluster id is the row index // per_cluster; a neighbourhood sample of k=20 from
    # 50-point clusters must concentrate in few clusters, unlike a uniform draw.
    assert len(np.unique(rows // 50)) <= 100 // 20


def test_scale_changes_which_rows_are_neighbours():
    rng = np.random.default_rng(0)
    points = rng.standard_normal((400, 2)).astype(np.float32)
    plain = neighborhood_sample(points, 40, 10, seed=0)
    skewed = neighborhood_sample(points, 40, 10, seed=0,
                                 scale=np.array([1.0, 100.0], np.float32))
    assert not np.array_equal(plain, skewed)

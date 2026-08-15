"""Shared k-NN neighbourhood protocol (utils/geometry, P0.3 2026-08-14).

Three diagnostics quote "the local data" and used to build the neighbourhood three
different ways; these pin the properties the harmonised helper has to have for their
numbers to be comparable: standardization actually applied, exact self-exclusion, and
neighbours returned nearest-first.
"""
import numpy as np

from utils.geometry import NeighbourIndex


def _data(n=400, d_s=3, d_a=2, seed=0):
    rng = np.random.default_rng(seed)
    obs = rng.standard_normal((n, d_s)) * np.array([1.0, 100.0, 0.01])[:d_s]
    act = rng.standard_normal((n, d_a))
    return obs, act


def test_standardization_changes_which_neighbours_are_selected():
    """The point of the harmonisation: on data whose dimensions differ by orders of
    magnitude (cube mixes positions and velocities), raw distances are dominated by the
    largest-scale block and pick different neighbours than the standardized geometry."""
    obs, act = _data()
    index = NeighbourIndex(obs, act, index_rows=None, seed=0)
    _, pos = index.query(obs[:1], k=8)
    raw = np.argsort(np.linalg.norm(obs - obs[0], axis=1))[:8]
    assert set(pos[0].tolist()) != set(raw.tolist())


def test_query_is_nearest_first_in_standardized_space():
    obs, act = _data()
    index = NeighbourIndex(obs, act, index_rows=None, seed=0)
    d, pos = index.query(obs[:4], k=10)
    assert (np.diff(d, axis=1) >= -1e-12).all()
    ref = np.linalg.norm(index.standardize(obs[:4])[:, None] - index.standardize(obs)[None],
                         axis=-1)
    np.testing.assert_allclose(d, np.sort(ref, axis=1)[:, :10], rtol=1e-6, atol=1e-8)
    np.testing.assert_array_equal(pos[:, 0], np.arange(4))  # each row is its own NN


def test_self_exclusion_drops_exactly_the_query_row():
    obs, act = _data()
    index = NeighbourIndex(obs, act, index_rows=None, seed=0)
    rows = np.array([0, 5, 17, 99])
    _, pos = index.query(obs[rows], k=6, exclude_rows=rows)
    assert pos.shape == (4, 6)
    for i, r in enumerate(rows):
        assert r not in index.rows[pos[i]]
    # Without exclusion the row IS its own nearest neighbour, so the baselines that skip
    # it are measuring something different -- which is the whole reason for the flag.
    _, kept = index.query(obs[rows], k=6)
    assert (index.rows[kept[:, 0]] == rows).all()


def test_self_exclusion_is_a_no_op_for_rows_outside_the_index():
    """Query points that are not dataset rows (rollout states) must keep all k."""
    obs, act = _data()
    index = NeighbourIndex(obs, act, index_rows=200, seed=1)
    outside = np.setdiff1d(np.arange(len(obs)), index.rows)[:3]
    _, pos = index.query(obs[outside], k=5, exclude_rows=outside)
    _, plain = index.query(obs[outside], k=5)
    np.testing.assert_array_equal(pos, plain)


def test_min_action_dist_matches_a_brute_force_computation():
    obs, act = _data()
    index = NeighbourIndex(obs, act, index_rows=None, seed=0)
    q = obs[10:14]
    a = np.zeros((4, act.shape[1]))
    got = index.min_action_dist(q, a, k=12)
    neigh = index.neighbour_actions(q, k=12)
    want = np.linalg.norm(neigh - a[:, None], axis=-1).min(1) / index.action_scale
    np.testing.assert_allclose(got, want, rtol=1e-9)


def test_index_subsample_is_seeded_and_recorded():
    obs, act = _data(n=1000)
    a = NeighbourIndex(obs, act, index_rows=100, seed=3)
    b = NeighbourIndex(obs, act, index_rows=100, seed=3)
    c = NeighbourIndex(obs, act, index_rows=100, seed=4)
    np.testing.assert_array_equal(a.rows, b.rows)
    assert not np.array_equal(a.rows, c.rows)
    p = a.protocol()
    assert p["index_rows"] == 100 and p["k_neighbours"] == 32 and p["standardized"]

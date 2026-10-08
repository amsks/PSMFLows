import numpy as np

from tools.diag_actor_ranking import goal_rows, rank_among, rank_of_first, spread_ratio, summarise, trajectory_ends


def test_goal_rows_skip_trajectory_end():
    end = trajectory_ends(np.array([0, 0, 0, 1, 0, 0, 0, 0]))   # trajectories rows 0-3 and 4-7
    ok, g = goal_rows(np.array([0, 1, 2, 4]), end, 3)
    assert ok.tolist() == [True, True, False, True]
    assert g.tolist() == [2, 3, 4, 6]


def test_rank_of_first_best_worst_ties():
    s = np.array([[5.0, 1.0, 2.0, 3.0],
                  [0.0, 1.0, 2.0, 3.0],
                  [1.0, 1.0, 1.0, 1.0]])
    assert rank_of_first(s).tolist() == [1.0, 4.0, 2.5]


def test_rank_among_counts_all_columns():
    s = np.array([[1.0, 2.0, 3.0]])
    assert rank_among(s, np.array([2.5])).tolist() == [2.0]
    assert rank_among(s, np.array([9.0])).tolist() == [1.0]


def test_spread_ratio():
    s = np.array([[0.0, 2.0], [1.0, 5.0]])
    # within-row std 1 and 2 -> mean 1.5; best scores 2, 5 -> std 1.5
    assert np.isclose(spread_ratio(s), 1.0)


def test_summarise_chance_and_actor():
    rng = np.random.default_rng(0)
    s = rng.standard_normal((4000, 64))
    m = summarise(s, actor=s[:, 1] + 10.0)
    assert abs(m['data_mean_rank'] - 32.5) < 1.0
    assert abs(m['data_top8_rate'] - 0.125) < 0.02
    assert m['actor_mean_rank'] == 1.0 and m['actor_beats_eps_rate'] == 1.0
    assert abs(m['eps_mean_rank'] - 32.0) < 1.0

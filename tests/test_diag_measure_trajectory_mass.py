"""CPU tests for the pure half of tools/diag_measure_trajectory_mass.py: the eligible rows and
own-offset rows respect trajectory ends, the reference-set assembly dedups the goal slot, and
the rank-fraction / aggregation numbers come out as hand-computed."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.diag_measure_trajectory_mass import (
    aggregate,
    build_reference_rows,
    eligible_rows,
    in_top_fraction,
    offset_rows,
    slot_stats,
)

# two trajectories: rows 0..9 (terminal at 9) and rows 10..14 (terminal at 14)
TERMINALS = np.array([9, 14])
SIZE = 15


def test_eligible_rows_respects_trajectory_ends():
    # end - i >= 3: rows 0..6 of the first trajectory, rows 10..11 of the second
    assert eligible_rows(TERMINALS, SIZE, 3).tolist() == [0, 1, 2, 3, 4, 5, 6, 10, 11]
    assert eligible_rows(TERMINALS, SIZE, 0).tolist() == list(range(SIZE))
    assert eligible_rows(TERMINALS, SIZE, 10).tolist() == []
    # a dataset whose last row is not marked terminal ends at the last row
    assert eligible_rows(np.array([9]), SIZE, 4).tolist() == [0, 1, 2, 3, 4, 5, 10]


def test_offset_rows_clip_at_end_and_flag_invalid():
    rows, ok = offset_rows(np.array([2, 8, 12]), TERMINALS, SIZE, (1, 2, 5))
    # k = 1 is the row itself (its own next state); row 2: 2, 3, 6 all inside the trajectory
    assert rows[0].tolist() == [2, 3, 6] and ok[0].tolist() == [True, True, True]
    # row 8: k = 2 is row 9 (the terminal row, valid); k = 5 would be row 12, past the end -> clipped
    assert rows[1].tolist() == [8, 9, 9] and ok[1].tolist() == [True, True, False]
    # row 12 of the second trajectory never crosses into the first
    assert rows[2].tolist() == [12, 13, 14] and ok[2].tolist() == [True, True, False]


def test_build_reference_rows_layout_and_goal_dedup():
    pool = np.arange(100, 130)
    rows, slots = build_reference_rows(2, [2, 3, 6], goal_row=7, random_rows=pool, n_ref=8,
                                       excluded_rows=[2, 3, 4, 5, 6, 7, 8, 9])
    assert rows.tolist() == [100, 101, 102, 103, 2, 3, 6, 7]
    assert slots['random'] == slice(0, 4) and slots['own'].tolist() == [4, 5, 6]
    assert slots['goal'] == 7 and slots['goal_dup_offset'] is None
    # the goal coincides with the k = 2 slot: no goal slot, one more random row, goal -> that slot
    rows, slots = build_reference_rows(2, [2, 3, 6], goal_row=3, random_rows=pool, n_ref=8,
                                       excluded_rows=[2, 3, 4, 5, 6, 7, 8, 9])
    assert rows.tolist() == [100, 101, 102, 103, 104, 2, 3, 6]
    assert slots['goal'] == 6 and slots['goal_dup_offset'] == 1
    # excluded rows and the special rows are never drawn as random rows
    pool2 = np.array([3, 5, 7, 100, 2, 101, 102, 103, 104])
    rows, _ = build_reference_rows(2, [2, 3, 6], goal_row=7, random_rows=pool2, n_ref=8,
                                   excluded_rows=[5])
    assert rows.tolist() == [100, 101, 102, 103, 2, 3, 6, 7]


def test_in_top_fraction_strict_rank():
    p = np.array([0.5, 0.2, 0.1, 0.1, 0.05, 0.05, 0.0, 0.0, 0.0, 0.0])  # n = 10
    assert in_top_fraction(p, 0, 0.1)            # ceil(1) = 1: the argmax only
    assert not in_top_fraction(p, 1, 0.1)
    assert in_top_fraction(p, 1, 0.2)            # ceil(2) = 2
    assert in_top_fraction(p, 2, 0.3) and in_top_fraction(p, 3, 0.3)   # ties: both tied entries have 2 above
    assert not in_top_fraction(p, 4, 0.3)
    assert in_top_fraction(p, 9, 1.0)
    assert in_top_fraction(p, 0, 0.001)          # ceil(0.01) = 1


def test_slot_stats_and_aggregate_hand_values():
    # 8 states: 5 random slots (ids 0..4), 2 own slots (5, 6), goal slot 7
    slots = {'random': slice(0, 5), 'own': np.array([5, 6]), 'goal': 7, 'goal_dup_offset': None}
    p1 = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.3, 0.1, 0.1])
    p2 = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.65])
    s1, s2 = slot_stats(p1, slots, top_fracs=(0.2,)), slot_stats(p2, slots, top_fracs=(0.2,))
    assert np.isclose(s1['p_random_mean'], 0.1) and s1['p_own'] == [0.3, 0.1] and s1['p_goal'] == 0.1
    assert s1['own_top']['0.2'] == [True, True]    # ceil(1.6) = 2: the 0.3 entry, and 0.1 ties with 1 above
    assert s1['goal_top']['0.2'] is True
    assert s2['own_top']['0.2'] == [True, True] and s2['goal_top']['0.2'] is True
    agg = aggregate([s1, s2], offsets=(1, 2), top_fracs=(0.2,))
    assert agg['n'] == 2
    assert np.isclose(agg['per_offset']['1']['p']['mean'], 0.175)
    # mean p_own(k=1) / mean p_random = 0.175 / 0.075
    assert np.isclose(agg['per_offset']['1']['ratio_to_random'], 0.175 / 0.075)
    assert np.isclose(agg['per_offset']['2']['ratio_to_random'], 0.075 / 0.075)
    assert np.isclose(agg['goal']['ratio_to_random'], 0.375 / 0.075)
    assert agg['per_offset']['1']['top0.2'] == 1.0 and agg['goal']['top0.2'] == 1.0
    assert agg['per_offset']['1']['p']['n'] == 2
    assert aggregate([], offsets=(1,)) == {}

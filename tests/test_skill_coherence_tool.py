"""Unit tests for the PURE functions in tools/validate_skill_coherence.py.

validate_skill_coherence is the GO/NO-GO gate for hindsight skill conditioning: does
commanding c actually steer the rollout more than an arbitrary c would? These tests
cover only the logic that doesn't touch an env/checkpoint (all hydra/env work is
guarded behind main(), so importing the module is side-effect free) -- the episode-safe
c bookkeeping, the gap-closed metric, and the coherence_ratio/gate arithmetic.
"""
import numpy as np
import pytest

from tools.validate_skill_coherence import (
    coherence_ratio,
    episode_ends,
    gap_closed_frac,
    gate_decision,
    nearest_indices,
    rows_with_full_horizon,
    state_distance,
)


# --- episode-safe c computation --------------------------------------------------------

def _two_episode_dataset():
    """Episode 1: rows 0-4 (terminal at 4). Episode 2: rows 5-8 (terminal at 8)."""
    terminals = np.array([0, 0, 0, 0, 1, 0, 0, 0, 1], dtype=np.float32)
    observations = np.arange(9, dtype=np.float32).reshape(9, 1)
    return terminals, observations


def test_episode_ends_matches_hand_computed_boundaries():
    terminals, _ = _two_episode_dataset()
    ends = episode_ends(terminals)
    # every row in episode 1 (0-4) ends at 4; every row in episode 2 (5-8) ends at 8
    np.testing.assert_array_equal(ends, [4, 4, 4, 4, 4, 8, 8, 8, 8])


def test_episode_ends_handles_untermianted_final_row():
    """The dataset may not mark the last row as terminal; it still ends an episode."""
    terminals = np.array([0, 0, 1, 0, 0, 0], dtype=np.float32)  # no terminal at row 5
    ends = episode_ends(terminals)
    np.testing.assert_array_equal(ends, [2, 2, 2, 5, 5, 5])


def test_rows_with_full_horizon_matches_hand_computed_mask():
    terminals, _ = _two_episode_dataset()
    mask = rows_with_full_horizon(terminals, window=2)
    # idx+2 <= end: [0:2<=4 T, 1:3<=4 T, 2:4<=4 T, 3:5<=4 F, 4:6<=4 F,
    #                5:7<=8 T, 6:8<=8 T, 7:9<=8 F, 8:10<=8 F]
    np.testing.assert_array_equal(mask, [True, True, True, False, False,
                                         True, True, False, False])


def test_rows_with_full_horizon_agrees_with_add_skill_targets_clipping():
    """Cross-check against the ACTUAL hindsight-training rule (utils.datasets.
    add_skill_targets), not just a hand-derived mask: rows_with_full_horizon must be
    exactly the rows where add_skill_targets returns the UNCLIPPED i+window read, never
    a value clipped to (or worse, silently read across) the episode boundary.
    """
    from utils.datasets import add_skill_targets

    terminals, observations = _two_episode_dataset()
    window = 2
    skills = add_skill_targets({'observations': observations, 'terminals': terminals}, window)
    mask = rows_with_full_horizon(terminals, window)

    idx = np.arange(len(terminals))
    in_range = (idx + window) < len(terminals)
    unclipped_read = np.where(in_range, observations[np.minimum(idx + window, len(terminals) - 1), 0], np.nan)
    # Where the mask is True, add_skill_targets must equal the raw (unclipped) read.
    assert np.all(skills[mask, 0] == unclipped_read[mask])
    # Row 3 is the case this whole rule exists for: raw i+window=5 would read INTO
    # episode 2 (observations[5]==5), but it must be clipped to this episode's end (4).
    assert mask[3] == False  # noqa: E712 -- explicit bool comparison reads clearer here
    assert skills[3, 0] == 4.0
    assert skills[3, 0] != observations[5, 0]


def test_rows_with_full_horizon_all_true_for_zero_window():
    terminals, _ = _two_episode_dataset()
    mask = rows_with_full_horizon(terminals, window=0)
    assert np.all(mask)


# --- gap-closed metric ------------------------------------------------------------------

def test_state_distance_uses_shared_leading_dims():
    a = np.array([0.0, 0.0, 0.0, 99.0, 99.0])  # extra trailing dims must be ignored
    b = np.array([3.0, 4.0, 0.0])
    assert state_distance(a, b) == pytest.approx(5.0)


def test_gap_closed_frac_on_synthetic_trajectory():
    # trajectory distances-to-c over a rollout: starts at 5, gets as close as 1.
    dists = [5.0, 4.0, 3.0, 2.0, 1.0, 2.0, 3.0]
    initial_dist, min_dist = dists[0], min(dists)
    assert gap_closed_frac(initial_dist, min_dist) == pytest.approx(0.8)


def test_gap_closed_frac_reaching_c_exactly_is_one():
    assert gap_closed_frac(initial_dist=10.0, min_dist=0.0) == pytest.approx(1.0)


def test_gap_closed_frac_never_getting_closer_is_zero():
    assert gap_closed_frac(initial_dist=10.0, min_dist=10.0) == pytest.approx(0.0)


def test_gap_closed_frac_eps_guard_zero_initial_dist_no_drift():
    # "stay" condition: c == reset obs, so initial_dist == 0. No drift -> min_dist == 0
    # too, and the eps guard must not blow this up or divide by zero.
    val = gap_closed_frac(initial_dist=0.0, min_dist=0.0, eps=1e-8)
    assert np.isfinite(val)
    assert val == pytest.approx(1.0)


def test_gap_closed_frac_eps_guard_zero_initial_dist_with_drift():
    # Same "stay" edge case, but the rollout DID drift away from the reset: must stay
    # finite (not raise ZeroDivisionError) and register as a strong failure to stay put.
    val = gap_closed_frac(initial_dist=0.0, min_dist=0.5, eps=1e-8)
    assert np.isfinite(val)
    assert val < -1e4


# --- coherence_ratio arithmetic ----------------------------------------------------------

def test_coherence_ratio_basic():
    assert coherence_ratio(0.8, 0.2) == pytest.approx(4.0)


def test_coherence_ratio_eps_guard_zero_shuffled():
    # A shuffled control with a zero (or negative) mean gap-closed must not divide by
    # zero or flip sign -- it should read as a large-but-finite ratio.
    val = coherence_ratio(0.8, 0.0, eps=1e-8)
    assert np.isfinite(val)
    assert val > 1e6


def test_coherence_ratio_eps_guard_negative_shuffled():
    val = coherence_ratio(0.8, -0.3, eps=1e-8)
    assert np.isfinite(val)
    assert val > 1e6


def test_coherence_ratio_both_zero():
    val = coherence_ratio(0.0, 0.0, eps=1e-8)
    assert np.isfinite(val)
    assert val == pytest.approx(0.0)


# --- gate_decision -----------------------------------------------------------------------

def test_gate_passes_when_both_thresholds_clear():
    assert gate_decision(mean_gap_true=0.8, ratio=4.0) is True


def test_gate_fails_on_low_ratio_even_with_high_gap():
    assert gate_decision(mean_gap_true=0.9, ratio=1.5) is False


def test_gate_fails_on_low_gap_even_with_high_ratio():
    assert gate_decision(mean_gap_true=0.3, ratio=10.0) is False


def test_gate_boundary_is_strict_not_inclusive():
    assert gate_decision(mean_gap_true=0.5, ratio=4.0) is False  # gap == threshold
    assert gate_decision(mean_gap_true=0.8, ratio=2.0) is False  # ratio == threshold


# --- nearest_indices (reset-plausibility pool) --------------------------------------------

def test_nearest_indices_orders_by_distance():
    query = np.array([0.0, 0.0])
    candidates = np.array([[5.0, 0.0], [1.0, 0.0], [3.0, 0.0], [0.1, 0.0]])
    nn = nearest_indices(query, candidates, k=2)
    assert set(nn.tolist()) == {1, 3}  # the two closest: rows 1 (dist 1.0) and 3 (dist 0.1)


def test_nearest_indices_clips_k_to_pool_size():
    query = np.array([0.0])
    candidates = np.array([[1.0], [2.0]])
    nn = nearest_indices(query, candidates, k=10)
    assert len(nn) == 2

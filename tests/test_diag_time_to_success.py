"""CPU tests for the pure half of tools/diag_time_to_success.py: the first success step, the
survival curve, the per-episode record from an evaluate() trajectory, the summary numbers, and the
conversion of a diag_failure_stages cell (reset state counted in its length)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.diag_time_to_success import (
    episode_record,
    first_success_step,
    records_from_failure_stages,
    summarise,
    survival_curve,
)


def test_first_success_step_is_one_based_and_none_when_never():
    assert first_success_step([0, 0, 1, 1]) == 3
    assert first_success_step([1]) == 1
    assert first_success_step([0, 0, 0]) is None
    assert first_success_step([]) is None
    # booleans and floats both count; only the first nonzero matters
    assert first_success_step([False, True, False]) == 2
    assert first_success_step(np.array([0.0, 0.0, 1.0])) == 3


def test_survival_curve_counts_episodes_succeeded_by_t():
    s = survival_curve([3, 5, None, 5], horizon=6)
    assert s.shape == (7,)
    assert s.tolist() == [0.0, 0.0, 0.0, 0.25, 0.25, 0.75, 0.75]
    # S(0) is 0 even when every episode succeeds; S(horizon) is the success rate
    s = survival_curve([1, 2], horizon=4)
    assert s[0] == 0.0 and s[-1] == 1.0
    # a success beyond the horizon never shows; no episodes -> all zeros
    assert survival_curve([10], horizon=4).tolist() == [0.0] * 5
    assert survival_curve([], horizon=3).tolist() == [0.0] * 4


def test_episode_record_from_evaluate_trajectory():
    # success flag fires after the 3rd step; the env terminates there (3 transitions)
    traj = {'info': [{'success': False}, {'success': False}, {'success': True}]}
    r = episode_record(traj)
    assert r == {'success': 1.0, 'success_final': 1.0, 't_success': 3, 'length': 3}
    # a flag that fires and then drops: success (max) is 1, success_final is 0
    traj = {'info': [{'success': False}, {'success': True}, {'success': False}]}
    r = episode_record(traj)
    assert r['success'] == 1.0 and r['success_final'] == 0.0 and r['t_success'] == 2 and r['length'] == 3
    # never: t_success None, length is the number of transitions
    traj = {'info': [{'success': 0.0}] * 5}
    r = episode_record(traj)
    assert r == {'success': 0.0, 'success_final': 0.0, 't_success': None, 'length': 5}


def test_summarise_hand_computed():
    recs = [
        {'success': 1.0, 'success_final': 1.0, 't_success': 40, 'length': 40},
        {'success': 1.0, 'success_final': 1.0, 't_success': 120, 'length': 120},
        {'success': 0.0, 'success_final': 0.0, 't_success': None, 'length': 200},
        {'success': 1.0, 'success_final': 1.0, 't_success': 80, 'length': 80},
    ]
    s = summarise(recs, horizon=200)
    assert s['n_episodes'] == 4 and s['n_success'] == 3
    assert s['success']['mean'] == 0.75
    assert abs(s['success']['se'] - np.std([1, 1, 0, 1], ddof=1) / 2) < 1e-12
    assert s['t_success_mean'] == 80.0 and s['t_success_median'] == 80.0
    assert s['length_mean'] == 110.0
    assert s['by_step'] == {'50': 0.25, '100': 0.5, '150': 0.75}
    assert len(s['survival']) == 201 and s['survival'][-1] == 0.75 and s['survival'][0] == 0.0
    # no successes: the time-to-success fields are None, the curve is flat zero
    s0 = summarise([recs[2]], horizon=10)
    assert s0['t_success_mean'] is None and s0['t_success_median'] is None and s0['n_success'] == 0
    assert s0['survival'] == [0.0] * 11


def test_records_from_failure_stages_drops_reset_row():
    cell = {'episodes': {
        'a': [{'success': True, 'stats': {'length': 55}}, {'success': False, 'stats': {'length': 201}}],
        'b': [{'success': True, 'stats': {'length': 101}}],
    }}
    recs = records_from_failure_stages(cell)
    assert recs['a'][0] == {'success': 1.0, 'success_final': 1.0, 't_success': 54, 'length': 54}
    assert recs['a'][1] == {'success': 0.0, 'success_final': 0.0, 't_success': None, 'length': 200}
    assert recs['b'][0]['t_success'] == 100

"""Hindsight goal sampling for the preimage-augmented dataset (psmgoal).

OGBench GCDataset convention: a goal for row i is a state later on the same trajectory,
offset d ~ Geometric(p = 1 - discount), d >= 1, capped at the trajectory end; the goal
state is `next_observations[min(i + d - 1, end)]`, so d = 1 is the row's own s' and the
cap is the trajectory's final state. A `random_frac` of goals are next states of uniformly
random rows instead. Off by default: `Dataset.sample` is unchanged for every other agent.
"""
import numpy as np

from utils.datasets import Dataset, hindsight_goal_idxs

OB = 3
LENS = (5, 9, 7)      # three trajectories


def _toy():
    n = sum(LENS)
    obs = np.arange(n, dtype=np.float32)[:, None].repeat(OB, 1)
    terminals = np.zeros(n, np.float32)
    ends = np.cumsum(LENS) - 1
    terminals[ends] = 1.0
    return {'observations': obs, 'next_observations': obs + 0.5,
            'actions': np.zeros((n, 2), np.float32), 'rewards': np.zeros(n, np.float32),
            'masks': np.ones(n, np.float32), 'terminals': terminals}, ends


def _end_of(idx, ends):
    return ends[np.searchsorted(ends, idx)]


def test_hindsight_goals_lie_on_the_same_trajectory_and_after_the_row():
    d, ends = _toy()
    term = np.nonzero(d['terminals'] > 0)[0]
    n = d['terminals'].shape[0]
    idxs = np.repeat(np.arange(n), 50)
    g, is_rand = hindsight_goal_idxs(idxs, term, n, discount=0.5, random_frac=0.0,
                                     rng=np.random.default_rng(0))
    assert not is_rand.any()
    assert np.all(g >= idxs), "goal row precedes the row"
    assert np.all(g <= _end_of(idxs, ends)), "goal row crosses a trajectory boundary"
    assert np.any(g > idxs), "every goal is the row's own s'; the geometric offset is dead"
    # the cap is reached: rows near the end land on the terminal row
    assert np.any(g == _end_of(idxs, ends))
    # the terminal row's goal is itself (its next state is the trajectory's final state)
    assert np.all(g[np.isin(idxs, ends)] == idxs[np.isin(idxs, ends)])


def test_random_goal_fraction_is_honoured():
    d, _ = _toy()
    term = np.nonzero(d['terminals'] > 0)[0]
    n = d['terminals'].shape[0]
    idxs = np.random.default_rng(1).integers(0, n, size=20000)
    g, is_rand = hindsight_goal_idxs(idxs, term, n, discount=0.98, random_frac=0.3,
                                     rng=np.random.default_rng(2))
    frac = float(is_rand.mean())
    assert 0.27 < frac < 0.33, frac
    assert np.all((g >= 0) & (g < n))
    # the random goals are not confined to the row's trajectory
    _, ends = _toy()
    assert np.any(g[is_rand] > _end_of(idxs[is_rand], ends)) or np.any(g[is_rand] < idxs[is_rand])
    _, r1 = hindsight_goal_idxs(idxs, term, n, discount=0.98, random_frac=1.0,
                                rng=np.random.default_rng(3))
    assert r1.all()


def test_dataset_goals_are_opt_in_and_row_aligned():
    d, ends = _toy()
    ds = Dataset.create(**d)
    np.random.seed(0)
    b = ds.sample(8)
    assert 'goals' not in b, "goals must be opt-in; other agents' batches are unchanged"
    ds.return_goals = True
    ds.goal_discount = 0.5
    ds.goal_random_frac = 0.0
    idxs = np.arange(ds.size)
    b = ds.sample(ds.size, idxs=idxs)
    assert b['goals'].shape == (ds.size, OB)
    # each goal is some next state on the row's own trajectory, at or after the row
    g_row = np.rint(b['goals'][:, 0] - 0.5).astype(int)
    assert np.all(g_row >= idxs) and np.all(g_row <= _end_of(idxs, ends))
    np.testing.assert_array_equal(b['goals'], d['next_observations'][g_row])

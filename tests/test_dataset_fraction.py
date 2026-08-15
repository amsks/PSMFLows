"""dataset_fraction: episode-level subsampling keeps episodes whole and is seeded."""
import numpy as np

from envs.env_utils import subsample_episodes


def _toy(n_ep=10, ep_len=5):
    n = n_ep * ep_len
    terminals = np.zeros(n)
    terminals[ep_len - 1::ep_len] = 1.0
    return {
        'observations': np.arange(n, dtype=np.float32)[:, None],
        'actions': np.arange(n, dtype=np.float32)[:, None],
        'terminals': terminals,
    }


def test_keeps_whole_episodes():
    d = _toy()
    out = subsample_episodes(d, 0.3, seed=0)
    # 3 of 10 episodes, 5 rows each.
    assert out['terminals'].shape[0] == 15
    assert out['terminals'].sum() == 3
    # Every kept episode is contiguous in the original ordering: within an episode the
    # observation ids increase by exactly 1; terminals mark each seam.
    obs = out['observations'][:, 0]
    seams = np.nonzero(out['terminals'] > 0.5)[0]
    prev = 0
    for s in seams:
        ep = obs[prev:s + 1]
        assert np.all(np.diff(ep) == 1), 'episode rows were not kept contiguously'
        prev = s + 1


def test_seeded_and_distinct():
    d = _toy()
    a = subsample_episodes(d, 0.5, seed=0)
    b = subsample_episodes(d, 0.5, seed=0)
    c = subsample_episodes(d, 0.5, seed=1)
    assert np.array_equal(a['observations'], b['observations'])
    assert not np.array_equal(a['observations'], c['observations'])


def test_at_least_one_episode():
    d = _toy()
    out = subsample_episodes(d, 0.001, seed=0)
    assert out['terminals'].sum() == 1

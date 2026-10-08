import numpy as np

from utils.datasets import apply_row_drop


def test_row_drop_subsets_and_cuts(tmp_path):
    n = 8
    d = {'observations': np.arange(n, dtype=np.float32)[:, None],
         'next_observations': np.arange(1, n + 1, dtype=np.float32)[:, None],
         'terminals': np.array([0, 0, 0, 1, 0, 0, 0, 1], np.float32)}
    keep = np.array([1, 1, 0, 1, 1, 0, 0, 1], bool)
    p = tmp_path / 'm.npz'
    np.savez(p, keep=keep)
    out = apply_row_drop(d, str(p))
    np.testing.assert_array_equal(out['observations'][:, 0], [0, 1, 3, 4, 7])
    np.testing.assert_array_equal(out['next_observations'][:, 0], [1, 2, 4, 5, 8])
    # row 1 precedes a cut, row 3 and 7 were ends already, row 4 precedes a cut
    np.testing.assert_array_equal(out['terminals'], [0, 1, 1, 1, 1])


def test_stitch_mask_leaves_no_direct_stretch():
    from tools.make_stitch_mask import build_keep, remaining_direct
    # one trajectory: cube at start, travels, reaches goal, leaves, comes back to start
    s, g = np.array([0., 0., 0.]), np.array([1., 0., 0.])
    path = np.array([s, s, [.5, 0, 0], g, g, [.5, 0, 0], s, [.5, 0, 0], g])
    term = np.zeros(len(path)); term[-1] = 1
    keep, per = build_keep(path, term, {1: (s, g)}, 0.1, 0.1)
    assert per[1]['stretches'] == 2
    assert remaining_direct(path, term, keep, {1: (s, g)}, 0.1, 0.1)[1] == 0
    np.testing.assert_array_equal(keep, [1, 1, 0, 1, 1, 1, 1, 0, 1])

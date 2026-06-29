import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.figures import outcome_clips as oc


def _frames(val, T=8):
    return [np.full((16, 16, 3), val, np.uint8) for _ in range(T)]


def test_pick_success_failure_first_of_each():
    frames = [_frames(10), _frames(20), _frames(30)]
    succ, fail = oc.pick_success_failure([False, True, False], frames)
    assert int(succ[0][0, 0, 0]) == 20   # first success = ep1
    assert int(fail[0][0, 0, 0]) == 10   # first failure = ep0


def test_pick_handles_missing_outcome():
    succ, fail = oc.pick_success_failure([False, False],
                                         [_frames(10), _frames(20)])
    assert succ is None and fail is not None


def test_two_row_filmstrip_stacks():
    strip = oc.two_row_filmstrip(_frames(200), _frames(50),
                                 n_frames=4, pad=2, gap=8)
    assert strip.width == 16 * 4 + 2 * 3   # 4 frames, pad 2
    assert strip.height == 16 * 2 + 8      # two rows + gap


def test_two_row_filmstrip_single_and_empty():
    strip = oc.two_row_filmstrip(_frames(200), None, n_frames=4)
    assert strip.height == 16              # only the success row
    assert oc.two_row_filmstrip(None, None) is None

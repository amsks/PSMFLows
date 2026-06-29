import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.figures import pixel_filmstrips as pf


def test_sample_frames_count_and_endpoints():
    frames = [np.full((4, 4, 3), i, dtype=np.uint8) for i in range(10)]
    out = pf.sample_frames(frames, 4)
    assert len(out) == 4
    assert int(out[0][0, 0, 0]) == 0
    assert int(out[-1][0, 0, 0]) == 9


def test_make_filmstrip_width():
    frames = [np.zeros((8, 6, 3), dtype=np.uint8) for _ in range(3)]
    strip = pf.make_filmstrip(frames, pad=2)
    assert strip.height == 8
    assert strip.width == 6 * 3 + 2 * 2  # 22


def test_build_writes_five_filmstrips(tmp_path):
    import imageio.v2 as imageio
    root = tmp_path / "fb-pixel-results"
    run = root / "2026-05-20_00-00-00__cube-single-play-v0__fb_flowbc__s0"
    vid = run / "eval_videos" / "step_1000000"
    vid.mkdir(parents=True)
    frames = [np.full((16, 16, 3), 30 * i, dtype=np.uint8) for i in range(8)]
    for t in range(1, 6):
        imageio.mimsave(
            str(vid / f"cube-single-play-singletask-task{t}-v0.gif"), frames)
    out = tmp_path / "figures"
    written = pf.build(root, seed=0, step="step_1000000", n_frames=6,
                       out_dir=out)
    assert len(written) == 5
    for p in written:
        assert p.exists()

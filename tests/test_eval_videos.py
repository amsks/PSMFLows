"""Rollout videos from `evaluate` (2026-09-16).

`utils.evaluation.evaluate` already renders `num_video_episodes` extra episodes and keeps them
out of the statistics, but it drops their `info`, so a rendered episode could not be labelled
success/fail, and nothing wrote the frames anywhere (main.py only ships them to wandb).

Pinned here, with a mock env that renders a solid frame -- no GPU, env or checkpoint:
  - `evaluate(..., return_render_infos=True)` returns a 4th value: one flattened final `info`
    per rendered episode; the 3-tuple signature is unchanged when the flag is absent;
  - rendered episodes still stay out of `stats`;
  - `write_render_videos` writes one mp4 per episode named by its success, one contact-sheet
    PNG per episode (up to `sheet_frames` frames side by side), and a summary JSON.
"""
import json
import os

import numpy as np

from tests.test_eval_seeding import _MockAgent, _MockEnv
from utils.evaluation import evaluate, write_render_videos

H, W = 8, 12


class _RenderEnv(_MockEnv):
    def render(self):
        return np.full((H, W, 3), int(255 * self._init), np.uint8)


def test_default_signature_is_unchanged():
    out = evaluate(_MockAgent(), _RenderEnv(entropy=1), num_eval_episodes=2, seed=0)
    assert len(out) == 3


def test_render_infos_are_returned_and_rendered_episodes_stay_out_of_stats():
    env = _RenderEnv(entropy=1)
    stats, trajs, renders, infos = evaluate(_MockAgent(), env, num_eval_episodes=2,
                                            num_video_episodes=3, video_frame_skip=1, seed=0,
                                            return_render_infos=True)
    assert len(trajs) == 2 and len(renders) == 3 and len(infos) == 3
    assert renders[0].shape == (3, H, W, 3)            # 3 steps, every frame kept
    assert all("success" in i for i in infos)
    # stats came from the first two episodes only
    expected = np.mean([float(x > 0.5) for x in env.inits[:2]])
    assert stats["success"] == expected
    # each rendered episode's label matches that episode's own init
    for i, info in enumerate(infos):
        assert info["success"] == float(env.inits[2 + i] > 0.5)


def test_write_render_videos_writes_mp4_sheet_and_summary(tmp_path):
    renders = [np.random.default_rng(0).integers(0, 255, (6, H, W, 3), dtype=np.uint8),
               np.random.default_rng(1).integers(0, 255, (4, H, W, 3), dtype=np.uint8)]
    infos = [{"success": 1.0}, {"success": 0.0}]
    summary = write_render_videos(renders, infos, str(tmp_path), "demo", fps=10, sheet_frames=5)
    assert [e["success"] for e in summary["episodes"]] == [1.0, 0.0]
    p0 = os.path.join(tmp_path, "demo_ep0_success.mp4")
    p1 = os.path.join(tmp_path, "demo_ep1_fail.mp4")
    assert os.path.exists(p0) and os.path.getsize(p0) > 0
    assert os.path.exists(p1) and os.path.getsize(p1) > 0
    assert summary["episodes"][0]["video"] == p0 and summary["episodes"][0]["n_frames"] == 6
    import imageio.v3 as iio
    s0 = iio.imread(os.path.join(tmp_path, "demo_ep0_sheet.png"))
    s1 = iio.imread(os.path.join(tmp_path, "demo_ep1_sheet.png"))
    assert s0.shape == (H, 5 * W, 3), "5 evenly spaced frames side by side"
    assert s1.shape == (H, 4 * W, 3), "fewer frames than sheet_frames -> all of them"
    with open(os.path.join(tmp_path, "demo_videos.json")) as f:
        on_disk = json.load(f)
    assert on_disk == summary

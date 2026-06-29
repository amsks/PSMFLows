"""Verifies tools/wandb_mode_shim video GIF re-encoding + FB re-key.

Run with the JAX venv (has wandb/imageio/numpy):
    /dev/shm/.venv-jax/bin/python scripts/dev/test_wandb_shim_video.py
"""
import importlib.util
import os
import sys
import tempfile
from pathlib import Path

import imageio
import numpy as np
import wandb

SHIM = (
    Path(__file__).resolve().parent.parent.parent
    / "tools" / "wandb_mode_shim" / "sitecustomize.py"
)

_ORIG_VIDEO = wandb.Video
_ORIG_LOG = wandb.log
_ORIG_INIT = wandb.init
_GATE_VARS = (
    "GCIQL_WANDB_FORCE_MODE",
    "GCIQL_FB_ENV",
    "GCIQL_WANDB_PROJECT",
    "GCIQL_WANDB_ENTITY",
)
_load_count = 0


def load_shim(env, log=None):
    """Restore real wandb attrs, set env, exec the shim fresh.

    `log` lets a test inject the callable the shim should wrap as its
    "original" wandb.log (defaults to the real one).
    """
    global _load_count
    wandb.Video = _ORIG_VIDEO
    wandb.log = log if log is not None else _ORIG_LOG
    wandb.init = _ORIG_INIT
    for k in _GATE_VARS:
        os.environ.pop(k, None)
    os.environ.update(env)
    _load_count += 1
    spec = importlib.util.spec_from_file_location(
        f"_shim_under_test_{_load_count}", SHIM
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)


# Case 1: OGBench-style ndarray+mp4 -> GIF-backed wandb.Video
load_shim({"GCIQL_FB_ENV": "cube-single-play-v0"})
arr = np.zeros((4, 3, 8, 8), dtype=np.uint8)  # (t, c, H, W) like OGBench
v = wandb.Video(arr, fps=15, format="mp4")
assert isinstance(v, _ORIG_VIDEO), type(v)
assert getattr(v, "_format", None) == "gif", getattr(v, "_format", None)
assert str(getattr(v, "_path", "")).endswith(".gif"), getattr(v, "_path", None)
print("CASE 1 ok: ndarray+mp4 -> gif")

# Case 2: passthrough for a non-mp4 path arg (no re-encode, identity of path)
gp = tempfile.NamedTemporaryFile(suffix=".gif", delete=False)
gp.close()
imageio.mimsave(gp.name, np.zeros((2, 8, 8, 3), np.uint8), format="GIF",
                fps=30, loop=0)
load_shim({"GCIQL_FB_ENV": "cube-single-play-v0"})
v2 = wandb.Video(gp.name, fps=5, format="gif")
assert str(getattr(v2, "_path", "")) == gp.name, getattr(v2, "_path", None)
os.unlink(gp.name)
print("CASE 2 ok: non-mp4 path passthrough")

# Case 3: inert when no gate var set (wandb.Video unchanged identity)
load_shim({})
assert wandb.Video is _ORIG_VIDEO, "Video should be untouched when inert"
print("CASE 3 ok: inert without GCIQL_FB_ENV")

print("ALL VIDEO-WRAPPER CASES PASS")

# Case 4: wandb.log({'video': ...}) also emits eval_video/all_tasks,
# keeps the original 'video', and still mirrors scalar FB keys.
_captured = {}


def _recorder(data=None, *args, **kwargs):
    _captured.clear()
    if isinstance(data, dict):
        _captured.update(data)
    return None


load_shim({"GCIQL_FB_ENV": "cube-single-play-v0"}, log=_recorder)
_SENTINEL = object()
wandb.log(
    {
        "video": _SENTINEL,
        "evaluation/overall_success": 1.0,
        "evaluation/task1_pick_success": 0.5,
    },
    step=7,
)
assert _captured.get("eval_video/all_tasks") is _SENTINEL, _captured
assert _captured.get("video") is _SENTINEL, "original video key must remain"
assert _captured.get("eval/reward/eval/success") == 1.0, _captured
assert (
    _captured.get("eval/reward/cube-single-play-singletask-task1-v0/success")
    == 0.5
), _captured
print("CASE 4 ok: video re-keyed to eval_video/all_tasks (+ scalars)")

print("ALL CASES PASS")

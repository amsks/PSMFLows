"""Regression tests for tools/wandb_mode_shim/sitecustomize.py.

The shim monkeypatches wandb in the OGBench child process. Replacing the
``wandb.Video`` *class* with a function broke ``isinstance()`` checks in
OGBench's CsvLogger (``log_utils.py:26``), crashing every GCIQL pixel run at
the first eval with::

    TypeError: isinstance() arg 2 must be a type, a tuple of types, or a union
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SHIM_DIR = REPO / "tools" / "wandb_mode_shim"


def _run_with_shim(code: str, env_extra: dict) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SHIM_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    env.update(env_extra)
    return subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )


def test_wandb_video_is_still_a_class_after_shim():
    """``wandb.Video`` must remain a class so CsvLogger's ``isinstance()`` works.

    OGBench's CsvLogger stores ``(wandb.Image, wandb.Video, wandb.Histogram)``
    as ``disallowed_types`` and calls ``isinstance(v, self.disallowed_types)``
    to filter non-scalar metrics out of the CSV header. If the shim replaces
    ``wandb.Video`` with a function, the tuple contains a non-type and the
    isinstance call raises TypeError on the first eval.
    """
    code = textwrap.dedent(
        """
        import wandb
        assert isinstance(wandb.Video, type), (
            f"wandb.Video must remain a class after the shim, got "
            f"{type(wandb.Video).__name__}"
        )
        # Exact pattern from third_party/ogbench/impls/utils/log_utils.py:26
        disallowed = (wandb.Image, wandb.Video, wandb.Histogram)
        isinstance(0, disallowed)  # must not raise
        print("OK")
        """
    )
    proc = _run_with_shim(code, {"GCIQL_FB_ENV": "cube-single-play-v0"})
    assert proc.returncode == 0, (
        f"shim broke wandb.Video class identity\n"
        f"stdout: {proc.stdout!r}\nstderr: {proc.stderr!r}"
    )
    assert "OK" in proc.stdout


def test_wandb_video_call_still_returns_video_instance():
    """``wandb.Video(np.ndarray, format='mp4')`` must still produce an instance
    recognised by ``isinstance(v, wandb.Video)`` — that's how CsvLogger drops
    it from the CSV header.
    """
    code = textwrap.dedent(
        """
        import os, tempfile
        os.environ["WANDB_MODE"] = "offline"
        import numpy as np
        import wandb
        wandb.init(project="shim_test", dir=tempfile.mkdtemp(), mode="offline")
        # (t, c, h, w) uint8 like OGBench's get_wandb_video output.
        frames = np.zeros((4, 3, 8, 8), dtype=np.uint8)
        v = wandb.Video(frames, fps=15, format="mp4")
        assert isinstance(v, wandb.Video), (
            f"wandb.Video(...) returned {type(v).__name__}, "
            f"not isinstance of wandb.Video"
        )
        wandb.finish()
        print("OK")
        """
    )
    proc = _run_with_shim(code, {"GCIQL_FB_ENV": "cube-single-play-v0"})
    assert proc.returncode == 0, (
        f"wandb.Video instance check broke\n"
        f"stdout: {proc.stdout!r}\nstderr: {proc.stderr!r}"
    )
    assert "OK" in proc.stdout

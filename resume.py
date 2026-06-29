"""Crash-recovery resume helpers for train.py.

Pure and torch-free: sidecar I/O, latest-checkpoint discovery, resume-plan
resolution, and the clobber guard. Kept out of train.py so they unit-test
without importing torch/hydra/agents. See
PAPER/specs/2026-06-09-training-resume-design.md.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import NamedTuple, Optional, Union

TRAIN_STATE_FILE = "train_state.json"
_CKPT_RE = re.compile(r"^step_(\d+)\.pt$")

PathLike = Union[str, Path]


def write_train_state(save_dir: PathLike, step: int, wandb_run_id: Optional[str],
                      latest: str) -> None:
    """Write/overwrite the single sidecar that points at the newest checkpoint."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    payload = {"step": int(step), "wandb_run_id": wandb_run_id, "latest": latest}
    (save_dir / TRAIN_STATE_FILE).write_text(json.dumps(payload, indent=2))


def read_train_state(save_dir: PathLike) -> Optional[dict]:
    """Return the sidecar dict, or None if it does not exist."""
    p = Path(save_dir) / TRAIN_STATE_FILE
    if not p.exists():
        return None
    return json.loads(p.read_text())


def find_latest_checkpoint(save_dir: PathLike):
    """Return (Path, step) of the highest-numbered step_<N>.pt, or None.

    Ignores final.pt and any non step_<N>.pt file.
    """
    save_dir = Path(save_dir)
    if not save_dir.is_dir():
        return None
    best = None
    for f in save_dir.glob("step_*.pt"):
        m = _CKPT_RE.match(f.name)
        if m is None:
            continue
        n = int(m.group(1))
        if best is None or n > best[1]:
            best = (f, n)
    return best


class ResumePlan(NamedTuple):
    ckpt_path: Optional[Path]   # None => fresh run
    start_step: int             # 0 for a fresh run
    wandb_run_id: Optional[str]


def _step_from_ckpt_name(path: PathLike) -> int:
    m = _CKPT_RE.match(Path(path).name)
    return int(m.group(1)) if m else 0


def resolve_resume(save_dir: PathLike, resume: bool,
                   resume_from: Optional[str]) -> ResumePlan:
    """Decide how to start training.

    Priority: explicit resume_from > resume=true (auto-find latest) > fresh.
    Raises FileNotFoundError if a resume is requested but nothing is found.
    """
    if resume_from:
        ckpt = Path(resume_from)
        if not ckpt.exists():
            raise FileNotFoundError(f"resume_from checkpoint not found: {ckpt}")
        ts = read_train_state(ckpt.parent) or {}
        step = int(ts["step"]) if ts.get("latest") == ckpt.name \
            else _step_from_ckpt_name(ckpt)
        return ResumePlan(ckpt, step, ts.get("wandb_run_id"))

    if resume:
        latest = find_latest_checkpoint(save_dir)
        if latest is None:
            raise FileNotFoundError(
                f"resume=true but no step_*.pt found in {save_dir}")
        ckpt, step = latest
        ts = read_train_state(save_dir) or {}
        if ts.get("latest") == ckpt.name:
            step = int(ts["step"])
        return ResumePlan(ckpt, step, ts.get("wandb_run_id"))

    return ResumePlan(None, 0, None)


def assert_no_clobber(save_dir: PathLike, resume: bool, force: bool) -> None:
    """Refuse to start a fresh run on top of an existing run's checkpoints."""
    if resume or force:
        return
    save_dir = Path(save_dir)
    existing = list(save_dir.glob("step_*.pt")) + list(save_dir.glob("final.pt"))
    if existing:
        raise FileExistsError(
            f"{save_dir} already holds {len(existing)} checkpoint file(s). "
            f"Pass resume=true to continue, or force=true to overwrite.")

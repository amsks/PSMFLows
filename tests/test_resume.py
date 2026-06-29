import json
from pathlib import Path

import pytest

from resume import write_train_state, read_train_state


def test_write_then_read_roundtrip(tmp_path):
    write_train_state(tmp_path, step=400000, wandb_run_id="abc123", latest="step_400000.pt")
    got = read_train_state(tmp_path)
    assert got == {"step": 400000, "wandb_run_id": "abc123", "latest": "step_400000.pt"}


def test_write_creates_dir_and_overwrites(tmp_path):
    sub = tmp_path / "checkpoints"
    write_train_state(sub, step=10, wandb_run_id=None, latest="step_10.pt")
    write_train_state(sub, step=20, wandb_run_id=None, latest="step_20.pt")
    got = read_train_state(sub)
    assert got["step"] == 20 and got["latest"] == "step_20.pt"
    # exactly one sidecar file, valid JSON
    assert json.loads((sub / "train_state.json").read_text())["step"] == 20


def test_read_missing_returns_none(tmp_path):
    assert read_train_state(tmp_path) is None


# ── Task 2: latest-checkpoint discovery ──────────────────────────────────────
from resume import find_latest_checkpoint


def _touch(p: Path):
    p.write_bytes(b"x")


def test_find_latest_picks_highest_step(tmp_path):
    for n in (200000, 400000, 600000):
        _touch(tmp_path / f"step_{n}.pt")
    _touch(tmp_path / "final.pt")          # must be ignored
    _touch(tmp_path / "notes.txt")         # must be ignored
    path, step = find_latest_checkpoint(tmp_path)
    assert step == 600000
    assert path == tmp_path / "step_600000.pt"


def test_find_latest_empty_or_missing(tmp_path):
    assert find_latest_checkpoint(tmp_path) is None
    assert find_latest_checkpoint(tmp_path / "does_not_exist") is None


# ── Task 3: resume-plan resolution ───────────────────────────────────────────
from resume import resolve_resume, ResumePlan


def test_resolve_fresh_when_not_resuming(tmp_path):
    plan = resolve_resume(tmp_path, resume=False, resume_from=None)
    assert plan == ResumePlan(ckpt_path=None, start_step=0, wandb_run_id=None)


def test_resolve_auto_uses_latest_and_sidecar(tmp_path):
    (tmp_path / "step_200000.pt").write_bytes(b"x")
    (tmp_path / "step_400000.pt").write_bytes(b"x")
    write_train_state(tmp_path, step=400000, wandb_run_id="run42", latest="step_400000.pt")
    plan = resolve_resume(tmp_path, resume=True, resume_from=None)
    assert plan.ckpt_path == tmp_path / "step_400000.pt"
    assert plan.start_step == 400000
    assert plan.wandb_run_id == "run42"


def test_resolve_auto_errors_when_nothing_to_resume(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_resume(tmp_path, resume=True, resume_from=None)


def test_resolve_explicit_path(tmp_path):
    ckpt = tmp_path / "step_600000.pt"
    ckpt.write_bytes(b"x")
    write_train_state(tmp_path, step=600000, wandb_run_id="rid", latest="step_600000.pt")
    plan = resolve_resume(tmp_path, resume=False, resume_from=str(ckpt))
    assert plan.ckpt_path == ckpt
    assert plan.start_step == 600000
    assert plan.wandb_run_id == "rid"


def test_resolve_explicit_path_missing_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_resume(tmp_path, resume=False, resume_from=str(tmp_path / "nope.pt"))


def test_resolve_explicit_path_no_sidecar_parses_step(tmp_path):
    ckpt = tmp_path / "step_300000.pt"
    ckpt.write_bytes(b"x")
    plan = resolve_resume(tmp_path, resume=False, resume_from=str(ckpt))
    assert plan.start_step == 300000
    assert plan.wandb_run_id is None


# ── Task 4: clobber guard ────────────────────────────────────────────────────
from resume import assert_no_clobber


def test_clobber_raises_on_existing_checkpoints(tmp_path):
    (tmp_path / "step_200000.pt").write_bytes(b"x")
    with pytest.raises(FileExistsError):
        assert_no_clobber(tmp_path, resume=False, force=False)


def test_clobber_raises_on_existing_final(tmp_path):
    (tmp_path / "final.pt").write_bytes(b"x")
    with pytest.raises(FileExistsError):
        assert_no_clobber(tmp_path, resume=False, force=False)


def test_clobber_allows_when_empty(tmp_path):
    assert_no_clobber(tmp_path, resume=False, force=False)  # no raise


def test_clobber_skipped_when_resume_or_force(tmp_path):
    (tmp_path / "step_200000.pt").write_bytes(b"x")
    assert_no_clobber(tmp_path, resume=True, force=False)   # no raise
    assert_no_clobber(tmp_path, resume=False, force=True)   # no raise

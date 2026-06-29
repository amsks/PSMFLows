import importlib.util
from pathlib import Path

import pytest

# Load scripts/train/sweep_lib.py directly by file path. A plain `from scripts.train.sweep_lib
# import ...` is unreliable under full-suite collection: tests/scripts/__init__.py
# makes `tests/scripts` importable as top-level `scripts`, shadowing the repo-root
# `scripts/` package. File-path loading sidesteps that name collision entirely.
_SWEEP_LIB = Path(__file__).resolve().parents[1] / "scripts" / "train" / "sweep_lib.py"
_spec = importlib.util.spec_from_file_location("_sweep_lib_under_test", _SWEEP_LIB)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
parse_final_success = _mod.parse_final_success
aggregate_grid = _mod.aggregate_grid
pick_winner = _mod.pick_winner


def test_parse_final_success_takes_last(tmp_path):
    log = tmp_path / "run.log"
    log.write_text(
        "[step 0] eval/success=0.0000\n"
        "noise line\n"
        "[step 250000] eval/success=0.3000\n"
        "[step 500000] eval/success=0.4200\n"
    )
    assert parse_final_success(str(log)) == pytest.approx(0.42)


def test_parse_final_success_missing_returns_none(tmp_path):
    log = tmp_path / "empty.log"
    log.write_text("no eval here\n")
    assert parse_final_success(str(log)) is None


def test_aggregate_grid_means_over_seeds():
    rows = {("1", "1e-4"): [0.4, 0.6, 0.5], ("10", "1e-4"): [0.0, 0.1, 0.05]}
    agg = aggregate_grid(rows)
    assert agg[("1", "1e-4")]["mean"] == pytest.approx(0.5)
    assert agg[("10", "1e-4")]["mean"] == pytest.approx(0.05)
    assert agg[("1", "1e-4")]["n"] == 3


def test_pick_winner_is_highest_mean():
    agg = {("1", "1e-4"): {"mean": 0.5, "std": 0.08, "n": 3},
           ("10", "1e-4"): {"mean": 0.05, "std": 0.04, "n": 3}}
    assert pick_winner(agg) == ("1", "1e-4")

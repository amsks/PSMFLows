import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.figures import fb_gciql_curves as fc


def test_interp_tensor_forward_fills_failed_seed():
    # seed 1 stops at 950k (failed); seed 2 reaches 1M.
    data = {
        1: pd.DataFrame({"step": [0, 950_000], "overall": [0.0, 0.8]}),
        2: pd.DataFrame({"step": [0, 1_000_000], "overall": [0.0, 0.9]}),
    }
    ten = fc.interp_tensor(data, ["overall"], grid=np.array([0, 950_000,
                                                             1_000_000]))
    # seed 1 holds its 950k value (0.8) at 1M, not extrapolated upward.
    assert ten[0, 0, 1] == 0.8
    assert ten[0, 0, 2] == 0.8
    assert ten[1, 0, 2] == 0.9


def test_final_scores_uses_last_logged_row():
    data = {
        1: pd.DataFrame({"step": [0, 950_000], "overall": [0.1, 0.7]}),
        2: pd.DataFrame({"step": [0, 1_000_000], "overall": [0.1, 0.95]}),
    }
    fs = fc.final_scores(data, ["overall"])
    assert fs[0, 0] == 0.7   # failed seed -> its 950k checkpoint value
    assert fs[1, 0] == 0.95


def test_load_gciql_maps_task_columns(tmp_path):
    d = tmp_path / "sd007_20260518_201036"
    d.mkdir()
    pd.DataFrame({
        "evaluation/task1_horizontal_success": [0.0, 0.5],
        "evaluation/task2_vertical1_success": [0.0, 0.6],
        "evaluation/task3_vertical2_success": [0.0, 0.4],
        "evaluation/task4_diagonal1_success": [0.0, 0.3],
        "evaluation/task5_diagonal2_success": [0.0, 0.2],
        "evaluation/overall_success": [0.0, 0.4],
        "step": [1, 1_000_000],
    }).to_csv(d / "eval.csv", index=False)
    out = fc.load_gciql(tmp_path)
    assert sorted(out) == [7]
    assert list(out[7].columns) == ["step", "overall", *fc.TASKS]
    assert out[7].iloc[-1]["task2"] == 0.6


def test_load_fb_reads_canonical_keys(tmp_path):
    rid = "abc123"
    cols = {"_step": [0, 1_000_000], fc.FB_OVERALL: [0.0, 0.7]}
    for t in fc.TASKS:
        cols[fc._fb_task_key(t)] = [0.0, 0.7]
    pd.DataFrame(cols).to_parquet(tmp_path / f"{rid}.parquet")
    (tmp_path / "_meta.json").write_text(json.dumps(
        [{"id": rid, "config": {"seed": 3}}]))
    out = fc.load_fb(tmp_path)
    assert sorted(out) == [3]
    assert out[3].iloc[-1]["overall"] == 0.7


def test_render_generic_single_method(tmp_path):
    # Three seeds, FB-only; render must emit all four artifacts with one row.
    fb = {}
    for s in (0, 1, 2):
        df = pd.DataFrame({"step": [0, 500_000, 1_000_000],
                           "overall": [0.0, 0.3, 0.5]})
        for t in fc.TASKS:
            df[t] = [0.0, 0.25, 0.45]
        fb[s] = df
    out = tmp_path / "out"
    fc.render({"FB": fb}, out, title_prefix="FB (pixels)",
              n_seeds_note="n=3 seeds")
    for f in ("curves_pertask.png", "curve_aggregate.png",
              "iqm_table.tex", "iqm_table.md"):
        assert (out / f).exists()
    tex = (out / "iqm_table.tex").read_text()
    assert tex.count(r" \\") == 2  # header row + one FB data row
    assert "GCIQL" not in tex

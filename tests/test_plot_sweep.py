import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.figures import plot_sweep as ps


def _cache(tmp_path):
    # two seeds, same HP cell -> aggregate should average them
    pd.DataFrame({"_step": [0, 100], "eval/reward/eval/success": [0.0, 0.4]}
                 ).to_parquet(tmp_path / "r1.parquet")
    pd.DataFrame({"_step": [0, 100], "eval/reward/eval/success": [0.0, 0.6]}
                 ).to_parquet(tmp_path / "r2.parquet")
    meta = [
        {"id": "r1", "name": "s1", "group": "g", "state": "finished",
         "config": {"domain": "cube-single-play-v0", "ortho_coef": 1000,
                    "lr_b": 1e-4, "seed": 1}, "summary": {}},
        {"id": "r2", "name": "s2", "group": "g", "state": "finished",
         "config": {"domain": "cube-single-play-v0", "ortho_coef": 1000,
                    "lr_b": 1e-4, "seed": 2}, "summary": {}},
    ]
    (tmp_path / "_meta.json").write_text(json.dumps(meta))
    return tmp_path


def test_load_cache_returns_meta_and_histories(tmp_path):
    _cache(tmp_path)
    meta, hist = ps.load_cache(tmp_path)
    assert {m["id"] for m in meta} == {"r1", "r2"}
    assert set(hist) == {"r1", "r2"}
    assert "eval/reward/eval/success" in hist["r1"].columns


def test_hp_cell():
    assert ps.hp_cell({"domain": "cube-single-play-v0", "ortho_coef": 1000,
                        "lr_b": 1e-4}) == ("cube-single-play-v0", "1000", "1e-4")


def test_aggregate_mean_std_across_seeds(tmp_path):
    _cache(tmp_path)
    meta, hist = ps.load_cache(tmp_path)
    agg = ps.aggregate(hist, meta, "eval/reward/eval/success")
    cell = ("cube-single-play-v0", "1000", "1e-4")
    assert cell in agg
    df = agg[cell].sort_values("step").reset_index(drop=True)
    assert list(df["step"]) == [0, 100]
    assert df.loc[1, "mean"] == 0.5          # (0.4 + 0.6) / 2
    assert df.loc[1, "n"] == 2
    assert abs(df.loc[1, "std"] - 0.1) < 1e-9  # population std


def test_render_all_writes_four_outputs(tmp_path):
    _cache(tmp_path)
    out = tmp_path / "plots"
    meta, hist = ps.load_cache(tmp_path)
    ps.render_all(meta, hist, out, domains=["cube-single-play-v0"])
    assert (out / "sweep-cube-single-play-v0__eval.png").exists()
    assert (out / "sweep-cube-single-play-v0__eval_pertask.png").exists()
    assert (out / "sweep-cube-single-play-v0__train.png").exists()
    assert (out / "sweep-cube-single-play-v0__final.png").exists()
    assert (out / "final_summary.md").exists()
    assert "cube-single-play-v0" in (out / "final_summary.md").read_text()

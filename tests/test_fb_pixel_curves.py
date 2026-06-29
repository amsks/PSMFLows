import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.figures import fb_gciql_curves as fc
from scripts.figures import fb_pixel_curves as fp


def _write_fb_cache(cache_dir: Path, seeds):
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta = []
    for s in seeds:
        rid = f"run{s}"
        cols = {"_step": [0, 500_000, 1_000_000],
                fc.FB_OVERALL: [0.0, 0.3, 0.5]}
        for t in fc.TASKS:
            cols[fc._fb_task_key(t)] = [0.0, 0.25, 0.45]
        pd.DataFrame(cols).to_parquet(cache_dir / f"{rid}.parquet")
        meta.append({"id": rid, "config": {"seed": s}})
    (cache_dir / "_meta.json").write_text(json.dumps(meta))


def test_fb_pixel_curves_outputs(tmp_path, monkeypatch):
    cache = tmp_path / "wandb_data_fbpixel"
    _write_fb_cache(cache, seeds=[0, 1, 2, 3, 4])
    out = tmp_path / "fb_pixel"
    monkeypatch.setattr(sys, "argv",
                        ["fb_pixel_curves.py", "--fb", str(cache),
                         "--out", str(out)])
    fp.main()
    for f in ("curves_pertask.png", "curve_aggregate.png",
              "iqm_table.tex", "iqm_table.md"):
        assert (out / f).exists()
    tex = (out / "iqm_table.tex").read_text()
    assert "FB" in tex
    assert "GCIQL" not in tex  # FB-only until GCIQL pixel data is supplied

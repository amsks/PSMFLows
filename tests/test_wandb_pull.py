import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval import wandb_pull as wp


def test_metric_keys_cube_includes_canonical_eval_and_train():
    keys = wp.metric_keys("cube-single-play-v0")
    assert "eval/reward/eval/success" in keys
    assert "eval/reward/eval/reward" in keys
    assert "eval/reward/cube-single-play-singletask-task1-v0/success" in keys
    assert "eval/reward/cube-single-play-singletask-task5-v0/success" in keys
    assert "train/fb_loss" in keys
    assert "train/orth_loss" in keys
    assert "train/actor_loss" in keys
    assert "train/bc_flow_loss" in keys
    # old nested style must NOT appear
    assert "train/loss/fb" not in keys


def _run(group, tags, created, domain):
    return SimpleNamespace(
        id="x", group=group, tags=tags,
        created_at=created.isoformat().replace("+00:00", "Z"),
        config={"domain": domain, "ortho_coef": 1000, "lr_b": 1e-4, "seed": 1},
    )


def test_filter_runs_by_group_tag_hours_domain():
    now = datetime(2026, 5, 18, tzinfo=timezone.utc)
    recent = _run("g1", ["sweep"], now - timedelta(hours=1), "cube-single-play-v0")
    old = _run("g1", ["sweep"], now - timedelta(hours=99), "cube-single-play-v0")
    wrong_group = _run("g2", ["sweep"], now - timedelta(hours=1), "cube-single-play-v0")
    wrong_domain = _run("g1", ["sweep"], now - timedelta(hours=1), "foo-v0")
    kept = wp.filter_runs(
        [recent, old, wrong_group, wrong_domain],
        groups=["g1"], tags=["sweep"], hours=36,
        domains=["cube-single-play-v0"], now=now,
    )
    assert [r.id for r in kept] == ["x"]
    assert kept[0] is recent


import pandas as pd


def test_has_canonical_schema_true_when_eval_agg_present():
    df_ok = pd.DataFrame({"_step": [0, 100], "eval/reward/eval/success": [0.0, 0.1]})
    df_bad = pd.DataFrame({"_step": [0], "eval/cube-x/success": [0.0]})
    assert wp.has_canonical_schema(df_ok) is True
    assert wp.has_canonical_schema(df_bad) is False


class _FakeRun:
    def __init__(self, rid, df, domain="cube-single-play-v0"):
        self.id = rid
        self.name = f"run-{rid}"
        self.group = "g1"
        self.tags = ["sweep"]
        self.state = "finished"
        self.created_at = "2026-05-18T00:00:00Z"
        self.config = {"domain": domain, "ortho_coef": 1000, "lr_b": 1e-4, "seed": 1}
        self.summary = {"eval/reward/eval/success": 0.5}
        self._df = df

    def history(self, keys=None, samples=None, pandas=True):
        return self._df


class _FakeApi:
    def __init__(self, runs):
        self._runs = runs

    def runs(self, project, order=None, per_page=None):
        return list(self._runs)


def test_pull_writes_parquet_and_meta(tmp_path):
    good = _FakeRun("aaa", pd.DataFrame(
        {"_step": [0, 100], "eval/reward/eval/success": [0.0, 0.4],
         "train/fb_loss": [9.0, 3.0]}))
    bad = _FakeRun("bbb", pd.DataFrame({"_step": [0], "eval/old/success": [0.0]}))
    n = wp.pull(
        entity=None, project="amsks/factored-fb",
        groups=["g1"], tags=None, hours=999, domains=["cube-single-play-v0"],
        out_dir=tmp_path, api=_FakeApi([good, bad]),
    )
    assert n == 1  # only the canonical-schema run is written
    assert (tmp_path / "aaa.parquet").exists()
    assert not (tmp_path / "bbb.parquet").exists()
    meta = json.loads((tmp_path / "_meta.json").read_text())
    assert [m["id"] for m in meta] == ["aaa"]
    assert meta[0]["config"]["seed"] == 1
    back = pd.read_parquet(tmp_path / "aaa.parquet")
    assert "eval/reward/eval/success" in back.columns

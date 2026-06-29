"""scripts/eval/wandb_pull.py — pull canonical-schema run histories from wandb.

Targets EXACTLY the metric names the current code logs (train.py +
evals/ogbench.py). No back-compat remapping. See
docs/superpowers/specs/2026-05-17-cube-analysis-pipeline-design.md §4.3.

Usage:
    python scripts/eval/wandb_pull.py --project amsks/factored-fb \
        --groups cube-single-video-replay-ortho1000-lrb1e-4 --hours 96
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, List, Optional

if TYPE_CHECKING:
    import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from envs.ogbench import ALL_TASKS

# Aggregate eval metrics: train.py:228 logs evaluator.run() with prefix
# "eval/reward/"; evals/ogbench.py:100-102 emits eval/{success,reward,reward#std}
EVAL_AGG_KEYS = [
    "eval/reward/eval/success",
    "eval/reward/eval/reward",
    "eval/reward/eval/reward#std",
]

# Train metrics: train.py:247 logs agent.update() with prefix "train/";
# agents/fb/agent.py + flow_bc/agent.py return a FLAT dict (not nested loss/).
TRAIN_KEYS = [
    "train/fb_loss",
    "train/orth_loss",
    "train/actor_loss",
    "train/bc_flow_loss",
    "train/q",
    "train/M1",
    "train/B_norm",
    "train/z_norm",
]


def per_task_eval_keys(domain: str) -> List[str]:
    """eval/reward/<task>/{success,reward} for every task of the domain.

    evals/ogbench.py:85-86 emits "<task>/success" and "<task>/reward";
    train.py:228 adds the "eval/reward/" prefix.
    """
    keys: List[str] = []
    for task in ALL_TASKS.get(domain, []):
        keys.append(f"eval/reward/{task}/success")
        keys.append(f"eval/reward/{task}/reward")
    return keys


def metric_keys(domain: str) -> List[str]:
    """Full canonical key list for a domain (single source of truth)."""
    return EVAL_AGG_KEYS + per_task_eval_keys(domain) + TRAIN_KEYS


def _created(run) -> datetime:
    c = run.created_at
    if isinstance(c, str):
        return datetime.fromisoformat(c.replace("Z", "+00:00"))
    return c


def filter_runs(
    runs: Iterable,
    *,
    groups: Optional[List[str]],
    tags: Optional[List[str]],
    hours: int,
    domains: Optional[List[str]],
    obs_type: Optional[str] = None,
    now: Optional[datetime] = None,
) -> List:
    """Pure run filter: group in groups, any tag in tags, within `hours`,
    config.domain in domains, config.obs_type == obs_type. Empty/None filter
    means 'do not filter on it'."""
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=hours)
    kept = []
    for r in runs:
        if groups and r.group not in groups:
            continue
        if tags and not (set(r.tags or []) & set(tags)):
            continue
        if _created(r) < cutoff:
            continue
        if domains and r.config.get("domain") not in domains:
            continue
        if obs_type and r.config.get("obs_type") != obs_type:
            continue
        kept.append(r)
    return kept


def has_canonical_schema(history: "pd.DataFrame") -> bool:
    """A run is canonical iff the aggregate eval key is present.

    The discriminating signal vs. the old schema is the canonical aggregate
    eval prefix ``eval/reward/eval/`` (train.py:228 + evals/ogbench.py:100).
    ``#std`` is dropped by some parquet/history paths, so requiring the
    primary aggregate success key (``EVAL_AGG_KEYS[0]``) keeps old-schema
    runs (e.g. ``eval/<task>/success`` only) correctly skipped without
    false-negatives on canonical runs whose ``#std`` column was not logged.
    """
    return EVAL_AGG_KEYS[0] in history.columns


def pull(
    *,
    entity: Optional[str],
    project: str,
    groups: Optional[List[str]],
    tags: Optional[List[str]],
    hours: int,
    domains: Optional[List[str]],
    out_dir: Path,
    obs_type: Optional[str] = None,
    api=None,
) -> int:
    """Pull canonical-schema runs into out_dir. Returns #runs written.

    Per-run try/except: one failure warns and is skipped, never aborts.
    """
    import pandas as pd

    if api is None:
        import wandb
        api = wandb.Api()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = list(api.runs(project, order="-created_at", per_page=300))
    kept = filter_runs(runs, groups=groups, tags=tags, hours=hours,
                       domains=domains, obs_type=obs_type)

    meta = []
    written = 0
    for r in kept:
        domain = r.config.get("domain")
        try:
            hist = r.history(keys=metric_keys(domain), samples=2000, pandas=True)
        except Exception as e:  # noqa: BLE001 - resilience over precision
            print(f"  [warn] history fetch failed for {r.id}: {e}")
            continue
        if hist is None or hist.empty:
            print(f"  [warn] empty history for {r.id}, skipping")
            continue
        if not has_canonical_schema(hist):
            print(f"  [warn] {r.id} ({getattr(r, 'name', '?')}) lacks canonical "
                  f"eval keys — old schema, skipping")
            continue
        hist.to_parquet(out_dir / f"{r.id}.parquet")
        meta.append({
            "id": r.id,
            "name": getattr(r, "name", r.id),
            "state": getattr(r, "state", None),
            "group": r.group,
            "config": dict(r.config),
            "summary": dict(getattr(r, "summary", {}) or {}),
        })
        written += 1

    (out_dir / "_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"[wandb_pull] wrote {written} run(s) to {out_dir}")
    return written


def _csv(value: Optional[str]) -> Optional[List[str]]:
    return [x for x in value.split(",") if x] if value else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--entity", default=None)
    ap.add_argument("--project", default="amsks/factored-fb")
    ap.add_argument("--groups", default=None, help="comma-separated")
    ap.add_argument("--tags", default=None, help="comma-separated")
    ap.add_argument("--hours", type=int, default=96)
    ap.add_argument("--domains", default=None, help="comma-separated")
    ap.add_argument("--obs-type", default=None,
                    help="filter on config.obs_type, e.g. pixels or state")
    ap.add_argument("--out", default=str(REPO_ROOT / "analysis" / "wandb" / "wandb_data"))
    args = ap.parse_args()
    pull(
        entity=args.entity,
        project=args.project,
        groups=_csv(args.groups),
        tags=_csv(args.tags),
        hours=args.hours,
        domains=_csv(args.domains),
        obs_type=args.obs_type,
        out_dir=Path(args.out),
    )


if __name__ == "__main__":
    main()

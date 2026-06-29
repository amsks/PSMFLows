"""scripts/figures/plot_sweep.py — replot wandb cache (canonical schema only).

Reads analysis/wandb/wandb_data/*.parquet + _meta.json, groups by
(domain, ortho_coef, lr_b), and renders 4 plot types per domain.
Metric key schema is imported from scripts.eval.wandb_pull (single source).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval.wandb_pull import EVAL_AGG_KEYS, TRAIN_KEYS, per_task_eval_keys  # noqa: E402

Cell = Tuple[str, str, str]  # (domain, ortho_coef, lr_b)


def _fmt_lr(lr: float) -> str:
    return f"{float(lr):.0e}".replace("e-0", "e-")


def hp_cell(config: dict) -> Cell:
    return (
        str(config.get("domain")),
        str(int(config.get("ortho_coef"))),
        _fmt_lr(config.get("lr_b")),
    )


def load_cache(in_dir: Path | str) -> Tuple[List[dict], Dict[str, pd.DataFrame]]:
    in_dir = Path(in_dir)
    meta = json.loads((in_dir / "_meta.json").read_text())
    histories: Dict[str, pd.DataFrame] = {}
    for m in meta:
        p = in_dir / f"{m['id']}.parquet"
        if p.exists():
            histories[m["id"]] = pd.read_parquet(p)
    return meta, histories


def aggregate(
    histories: Dict[str, pd.DataFrame],
    meta: List[dict],
    metric_key: str,
) -> Dict[Cell, pd.DataFrame]:
    """For each HP cell, align runs on `_step` and compute mean/std/n.

    Pure function — no matplotlib, no IO. Runs lacking `metric_key` are
    skipped for that metric.
    """
    by_cell: Dict[Cell, List[pd.DataFrame]] = defaultdict(list)
    for m in meta:
        h = histories.get(m["id"])
        if h is None or metric_key not in h.columns or "_step" not in h.columns:
            continue
        sub = h[["_step", metric_key]].dropna()
        if sub.empty:
            continue
        sub = sub.rename(columns={"_step": "step", metric_key: "value"})
        by_cell[hp_cell(m["config"])].append(sub)

    out: Dict[Cell, pd.DataFrame] = {}
    for cell, frames in by_cell.items():
        cat = pd.concat(frames, ignore_index=True)
        g = cat.groupby("step")["value"]
        out[cell] = pd.DataFrame({
            "step": list(g.groups.keys()),
            "mean": g.mean().values,
            "std": g.std(ddof=0).fillna(0.0).values,
            "n": g.count().values,
        })
    return out


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _label(cell: Cell) -> str:
    return f"ortho={cell[1]}, lr_b={cell[2]}"


def _plot_metric(ax, agg: Dict[Cell, pd.DataFrame], domain: str, title: str):
    plotted = False
    for cell, df in sorted(agg.items()):
        if cell[0] != domain:
            continue
        d = df.sort_values("step")
        ax.plot(d["step"], d["mean"], label=_label(cell))
        ax.fill_between(d["step"], d["mean"] - d["std"], d["mean"] + d["std"],
                        alpha=0.2)
        plotted = True
    ax.set_title(title)
    ax.set_xlabel("step")
    if plotted:
        ax.legend(fontsize=7)
    return plotted


def render_all(
    meta: List[dict],
    histories: Dict[str, pd.DataFrame],
    out_dir: Path | str,
    domains: List[str],
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_lines = ["# Final eval success per HP cell", ""]

    for domain in domains:
        safe = domain

        # 1. eval success + reward
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        _plot_metric(axes[0], aggregate(histories, meta, "eval/reward/eval/success"),
                     domain, "eval/success")
        _plot_metric(axes[1], aggregate(histories, meta, "eval/reward/eval/reward"),
                     domain, "eval/reward")
        fig.tight_layout()
        fig.savefig(out_dir / f"sweep-{safe}__eval.png", dpi=120)
        plt.close(fig)

        # 2. per-task success
        task_keys = [k for k in per_task_eval_keys(domain) if k.endswith("/success")]
        ncols = max(1, len(task_keys))
        fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4), squeeze=False)
        for ax, key in zip(axes[0], task_keys):
            _plot_metric(ax, aggregate(histories, meta, key), domain,
                         key.split("/")[-2])
        fig.tight_layout()
        fig.savefig(out_dir / f"sweep-{safe}__eval_pertask.png", dpi=120)
        plt.close(fig)

        # 3. train losses
        loss_keys = ["train/fb_loss", "train/orth_loss", "train/actor_loss",
                     "train/bc_flow_loss"]
        fig, axes = plt.subplots(1, 4, figsize=(16, 4), squeeze=False)
        for ax, key in zip(axes[0], loss_keys):
            _plot_metric(ax, aggregate(histories, meta, key), domain, key)
        fig.tight_layout()
        fig.savefig(out_dir / f"sweep-{safe}__train.png", dpi=120)
        plt.close(fig)

        # 4. final-performance bar + summary md
        agg = aggregate(histories, meta, "eval/reward/eval/success")
        cells = sorted(c for c in agg if c[0] == domain)
        fig, ax = plt.subplots(figsize=(max(4, 1.5 * len(cells)), 4))
        summary_lines.append(f"## {domain}")
        if cells:
            means, stds, labels = [], [], []
            for c in cells:
                last = agg[c].sort_values("step").iloc[-1]
                means.append(last["mean"])
                stds.append(last["std"])
                labels.append(_label(c))
                summary_lines.append(
                    f"- {_label(c)}: {last['mean']*100:.2f} ± "
                    f"{last['std']*100:.2f} (n={int(last['n'])}, "
                    f"@step {int(last['step'])})")
            ax.bar(range(len(cells)), means, yerr=stds)
            ax.set_xticks(range(len(cells)))
            ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
        else:
            summary_lines.append("- (no data)")
        ax.set_title(f"{domain} final eval/success")
        fig.tight_layout()
        fig.savefig(out_dir / f"sweep-{safe}__final.png", dpi=120)
        plt.close(fig)
        summary_lines.append("")

    (out_dir / "final_summary.md").write_text("\n".join(summary_lines))
    print(f"[plot_sweep] wrote plots to {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="in_dir",
                    default=str(REPO_ROOT / "analysis" / "wandb" / "wandb_data"))
    ap.add_argument("--out", default=str(REPO_ROOT / "analysis" / "legacy" / "plots"))
    ap.add_argument("--domains", default="cube-single-play-v0,antmaze-medium-navigate-v0",
                    help="comma-separated")
    args = ap.parse_args()
    meta, hist = load_cache(args.in_dir)
    domains = [d for d in args.domains.split(",") if d]
    render_all(meta, hist, args.out, domains)


if __name__ == "__main__":
    main()

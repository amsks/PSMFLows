#!/usr/bin/env python
"""scripts/figures/plot_cf_value_curve.py — does climbing the value buy you success?

The policy follows the value, so the decisive question is: among in-hand grasps,
do higher-value grasps actually have a higher counterfactual success rate (the
agent reaching the goal from them)? For each method we rank grasps by value
within each goal (to remove the goal's value offset), pool, bin into value
deciles, and plot the counterfactual success rate per decile.

A value that tracks controllability rises steeply; an aliased value is flat. The
slope is the value's usable signal; the annotation is the rank-AUC.

Reads analysis/value/repsep/cf_records_<method>.parquet and cf_value_<method>.json.
Run under .venv.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

REPO = Path(__file__).resolve().parent.parent.parent
REPSEP = REPO / "analysis" / "value" / "repsep"
OUT = REPO / "PAPER" / "rlbrew" / "figures"

sns.set_palette("colorblind")
_CB = sns.color_palette("colorblind").as_hex()
METHODS = [("gciql", "GCIQL", _CB[1]), ("crl", "CRL", _CB[2]),
           ("fb", "FB", _CB[0]), ("rldp", "RLDP", _CB[3])]
NBINS = 6


def _seed_curve(df):
    """Per-bin counterfactual success rate for one (seed's) records frame."""
    df = df.copy()
    df["pct"] = df.groupby("task")["value"].rank(pct=True)
    bins = np.linspace(0, 1, NBINS + 1)
    df["b"] = np.clip(np.digitize(df["pct"], bins) - 1, 0, NBINS - 1)
    g = df.groupby("b")["success"].mean()
    y = np.full(NBINS, np.nan)
    y[g.index.to_numpy()] = g.to_numpy()
    return y


def main() -> int:
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    x = (np.linspace(0, 1, NBINS + 1)[:-1] + np.linspace(0, 1, NBINS + 1)[1:]) / 2
    ms = json.loads((REPSEP / "cf_value_multiseed.json").read_text()) \
        if (REPSEP / "cf_value_multiseed.json").exists() else {}
    for m, lab, col in METHODS:
        seed_files = sorted(REPSEP.glob(f"cf_records_{m}_ms*.parquet"))
        if seed_files:                                    # multiseed: mean +/- across-seed 95% CI
            curves = np.vstack([_seed_curve(pd.read_parquet(p)) for p in seed_files])
            y = np.nanmean(curves, axis=0)
            n = np.sum(~np.isnan(curves), axis=0)
            sem = np.nanstd(curves, axis=0, ddof=1) / np.sqrt(np.maximum(n, 1))
            lo, hi = y - 1.96 * sem, y + 1.96 * sem
            a = ms.get(m, {})
            note = (f"AUC {a['auc_iqm']:.2f}$\\pm${a['auc_ci_halfwidth']:.2f}"
                    if a else "")
            ax.fill_between(x, lo, hi, color=col, alpha=0.18)
            ax.plot(x, y, "o-", color=col, lw=2.2, ms=7,
                    label=f"{lab}  ({note})")
        else:                                             # fallback: single primary run
            rp = REPSEP / f"cf_records_{m}.parquet"
            if not rp.exists():
                continue
            y = _seed_curve(pd.read_parquet(rp))
            auc = json.loads((REPSEP / f"cf_value_{m}.json").read_text())["cf_value_auc_mean"]
            ax.plot(x, y, "o-", color=col, lw=2.2, ms=7, label=f"{lab}  (AUC {auc:.2f})")

    ax.set_xlabel("grasp value percentile  (low $\\rightarrow$ high value)", fontsize=11)
    ax.set_ylabel("counterfactual success rate\n(agent reaches goal from grasp)", fontsize=11)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("Does climbing the value buy success?\nGCIQL's value tracks "
                 "controllability; the FB family's is nearly flat (aliased)\n"
                 "mean over 10 seeds; band = 95% CI across seeds",
                 fontsize=11)
    ax.grid(True, alpha=0.3); ax.legend(fontsize=10, loc="upper left", title="method")
    fig.tight_layout()
    out = OUT / "fig_cf_value_curve.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

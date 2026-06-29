#!/usr/bin/env python
"""scripts/figures/plot_cf_aliasing.py — counterfactual value-aliasing visualization.

For each method, a 2D PCA of in-hand grasp states (the most outcome-balanced
goal), shown twice: coloured by the agent's COUNTERFACTUAL success (does the
agent reach the goal from this grasp) and by the agent's VALUE at the grasp. If
the value tracked controllability, the value field would line up with the
success colours. It does for the method that works and not for the ones that
fail -- the visual form of value aliasing.

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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parent.parent.parent
REPSEP = REPO / "analysis" / "value" / "repsep"
OUT = REPO / "PAPER" / "rlbrew" / "figures"

sns.set_palette("colorblind")
_CB = sns.color_palette("colorblind").as_hex()
SUCC, FAIL = _CB[2], _CB[1]
METHODS = [("fb", "FB"), ("rldp", "RLDP"), ("gciql", "GCIQL"), ("crl", "CRL")]


def main() -> int:
    fig, axes = plt.subplots(2, 4, figsize=(15.0, 7.4))
    for j, (m, lab) in enumerate(METHODS):
        rp = REPSEP / f"cf_records_{m}.parquet"
        jp = REPSEP / f"cf_value_{m}.json"
        if not rp.exists():
            for r in (0, 1):
                axes[r, j].text(0.5, 0.5, f"{lab}\n(no data)", ha="center",
                                va="center", transform=axes[r, j].transAxes)
                axes[r, j].set_xticks([]); axes[r, j].set_yticks([])
            continue
        df = pd.read_parquet(rp)
        js = json.loads(jp.read_text())
        # most outcome-balanced goal (clearest scatter)
        sr = df.groupby("task")["success"].mean()
        t = int((sr - 0.5).abs().idxmin())
        d = df[df["task"] == t].reset_index(drop=True)
        cols = [c for c in d.columns if c.startswith("obs_")]
        Z = PCA(2, random_state=0).fit_transform(StandardScaler().fit_transform(d[cols].to_numpy()))
        auc = js["cf_value_auc_per_task"][t - 1]
        succ = d["success"].to_numpy(bool)
        val = d["value"].to_numpy()

        ax = axes[0, j]
        ax.scatter(Z[~succ, 0], Z[~succ, 1], s=12, c=FAIL, alpha=0.6, linewidths=0, label="agent fails")
        ax.scatter(Z[succ, 0], Z[succ, 1], s=12, c=SUCC, alpha=0.7, linewidths=0, label="agent succeeds")
        ax.set_title(lab, fontsize=13, fontweight="bold")
        ax.text(0.5, 1.005, f"goal {t} · success {succ.mean():.0%} · value-AUC {auc:.2f}",
                transform=ax.transAxes, ha="center", va="bottom", fontsize=9)
        if j == 0:
            ax.set_ylabel("counterfactual\noutcome", fontsize=11, fontweight="bold")

        ax = axes[1, j]
        sciter = ax.scatter(Z[:, 0], Z[:, 1], s=12, c=val, cmap="viridis", alpha=0.85, linewidths=0)
        fig.colorbar(sciter, ax=ax, fraction=0.046, pad=0.04)
        if j == 0:
            ax.set_ylabel("agent's value\n$V(s\\,|\\,g)$", fontsize=11, fontweight="bold")
        for ax in (axes[0, j], axes[1, j]):
            ax.set_xticks([]); ax.set_yticks([])

    h = [plt.Line2D([], [], marker="o", ls="", ms=7, c=SUCC, label="agent succeeds"),
         plt.Line2D([], [], marker="o", ls="", ms=7, c=FAIL, label="agent fails")]
    fig.legend(handles=h, loc="lower center", ncol=2, frameon=False, fontsize=11,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Does the value line up with the agent's own success? "
                 "(PCA of in-hand grasps; top = counterfactual outcome, bottom = value)",
                 fontsize=13, y=1.0)
    fig.tight_layout(rect=(0, 0.02, 1, 0.99))
    out = OUT / "fig_cf_aliasing.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

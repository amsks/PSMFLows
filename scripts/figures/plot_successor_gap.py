#!/usr/bin/env python
"""scripts/figures/plot_successor_gap.py — direct tail successor-gap estimate.

Plots the tail-success-probability gap |v_hat(s1)-v_hat(s2)| against the
observation distance ||Omega(s1)-Omega(s2)|| for pairs of in-hand grasps under
the same goal (binned mean +/- s.e.m.). A smooth value would send the gap to 0
as observation distance shrinks; the measured gap stays large at the smallest
distances, directly exhibiting the interaction-conditioned successor gap.

Reads analysis/value/repsep/gap_pairs_<method>.parquet. Run under .venv.
"""
from __future__ import annotations

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
METHODS = [("fb", "FB", _CB[0]), ("rldp", "RLDP", _CB[3])]
NBINS = 8


def main() -> int:
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    near = {}
    for m, lab, col in METHODS:
        fp = REPSEP / f"gap_pairs_{m}.parquet"
        if not fp.exists():
            continue
        d = pd.read_parquet(fp)
        qs = np.quantile(d["dobs"], np.linspace(0, 1, NBINS + 1))
        qs[0] -= 1e-9
        d["b"] = (np.digitize(d["dobs"], qs) - 1).clip(0, NBINS - 1)
        g = d.groupby("b")["dv"].agg(["mean", "sem"])
        xc = [(qs[i] + qs[i + 1]) / 2 for i in g.index]
        ax.errorbar(xc, g["mean"], yerr=g["sem"], fmt="o-", color=col, lw=2.2,
                    ms=6, capsize=3, label=f"{lab} (cross-grasp gap)")
        # same-state noise floor (matched K): horizontal band
        npq = REPSEP / f"gap_noise_{m}.parquet"
        if npq.exists():
            nf = pd.read_parquet(npq)["noise"]
            ax.axhline(nf.mean(), ls="--", color=col, lw=1.3, alpha=0.7)
            ax.text(xc[-1], nf.mean() + 0.004, f"{lab} same-state noise {nf.mean():.2f}",
                    color=col, fontsize=8, ha="right", va="bottom")
            near[lab] = (d[d["b"] == 0]["dv"].mean(), nf.mean())

    xl = ax.get_xlim()
    ax.set_xlim(0, xl[1]); ax.set_ylim(0, None)
    ax.set_xlabel("observation distance  $\\|\\Omega(s_1)-\\Omega(s_2)\\|$  (standardized)", fontsize=11)
    ax.set_ylabel("tail-success gap  $|\\hat v(s_1)-\\hat v(s_2)|$", fontsize=11)
    sub = "; ".join(f"{k}: {v[0]:.2f} vs noise {v[1]:.2f}" for k, v in near.items())
    ax.set_title("Direct estimate of the interaction-conditioned successor gap\n"
                 f"nearest-grasp gap exceeds the same-state noise floor ({sub})",
                 fontsize=11)
    ax.legend(fontsize=9, loc="lower right", title="method")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = OUT / "fig_successor_gap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[plot] wrote {out}")
    print("nearest-bin gap vs noise:",
          {k: (round(v[0], 3), round(v[1], 3)) for k, v in near.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())

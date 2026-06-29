#!/usr/bin/env python
"""scripts/figures/plot_cf_ladder.py — cross-method counterfactual ladder (all four methods).

Per method, against the agent's OWN counterfactual success: the raw-state linear
baseline (how state-predictable the agent's outcome is), the best-linear readout of the
representation the value reads (GCIQL penultimate value features; CRL phi(s,pi);
FB/RLDP forward map F(s,pi,z)), and the value Q's own ranking. The rep->Q gap is
how much usable signal the value discards.

Reads analysis/value/repsep/cf_ladder_<method>.json. Run under .venv.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

REPO = Path(__file__).resolve().parent.parent.parent
REPSEP = REPO / "analysis" / "value" / "repsep"
OUT = REPO / "PAPER" / "rlbrew" / "figures"

sns.set_palette("colorblind")
_CB = sns.color_palette("colorblind").as_hex()
RAW, REP, Q = "0.6", _CB[0], _CB[3]
METHODS = ["gciql", "crl", "fb", "rldp"]
LAB = {"gciql": "GCIQL", "crl": "CRL", "fb": "FB", "rldp": "RLDP"}


def _vals(m):
    d = json.loads((REPSEP / f"cf_ladder_{m}.json").read_text())["means"]
    rep = d["rep"] if "rep" in d else d["F"]   # value's input representation
    return d["raw"], rep, d["Q"]


def main() -> int:
    raw = [_vals(m)[0] for m in METHODS]
    rep = [_vals(m)[1] for m in METHODS]
    qv = [_vals(m)[2] for m in METHODS]
    x = np.arange(len(METHODS)); w = 0.26
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.bar(x - w, raw, w, color=RAW, edgecolor="black", linewidth=0.4, label="raw-state linear baseline")
    ax.bar(x, rep, w, color=REP, edgecolor="black", linewidth=0.4, label="representation (value's input)")
    b3 = ax.bar(x + w, qv, w, color=Q, edgecolor="black", linewidth=0.4, label="value $Q$ (policy follows)")
    for bars in (ax.containers):
        ax.bar_label(bars, fmt="%.2f", fontsize=7.5, padding=1)
    ax.axhline(0.5, color="black", ls="--", lw=1.0)
    ax.text(len(METHODS) - 0.5, 0.505, "chance", fontsize=8.5, va="bottom", ha="right")
    ax.set_xticks(x); ax.set_xticklabels([LAB[m] for m in METHODS], fontsize=11, fontweight="bold")
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("AUC vs agent's counterfactual success", fontsize=10)
    ax.set_title("Counterfactual representation–value ladder\n"
                 "GCIQL: value reads all available signal · FB: representation has it, "
                 "value doesn't · RLDP: signal absent", fontsize=11)
    ax.legend(fontsize=9, loc="upper right", frameon=False)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = OUT / "fig_cf_ladder.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[plot] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

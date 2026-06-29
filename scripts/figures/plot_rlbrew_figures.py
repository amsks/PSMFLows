#!/usr/bin/env python
"""scripts/figures/plot_rlbrew_figures.py — main-paper figures for PAPER/rlbrew.

Individual single-panel figures for the experiments section; layout/subcaptions
are handled in LaTeX (subcaption):
  fig_funnel.png       — P1: 4-stage phase funnel
  fig_outcomes.png     — P2: stacked terminal-outcome composition
  fig_support.png      — P3: (failure - success) nearest-neighbour distance heatmap
  fig_aliasing.png     — P4: matched-pair aliasing ranking

The representation-aliasing figures (fig_repsep, fig_repsep_auc) are generated
separately by scripts/figures/plot_repsep.py (they need the per-grasp embeddings).

Headline numbers are paper-canonical (match the tables in content/results.tex and
the appendix), so figures and tables cannot disagree. The aliasing panel is read
from the probe output so its error bars stay in sync.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Colorblind palette (Okabe-Ito), the de-facto standard for RL figures; also set
# as the default cycle so the by-regime aliasing bars use it.
sns.set_palette("colorblind")
_CB = sns.color_palette("colorblind").as_hex()
# blue, orange, green, vermillion, purple = _CB[0..4]

REPO = Path(__file__).resolve().parent.parent.parent
ANALYSIS = REPO / "analysis"
OUT = REPO / "PAPER" / "rlbrew" / "figures"

METHODS = ["FB", "GCIQL", "CRL", "RLDP", "TDMPC2"]
MC = {"FB": _CB[0], "GCIQL": _CB[1], "CRL": _CB[2], "RLDP": _CB[3],
      "TDMPC2": _CB[5]}


def _present(d):
    """Methods (in canonical order) that have an entry in data dict `d`. Lets
    a newly-added method (e.g. TDMPC2) appear only in the panels whose numbers
    have been filled in, without breaking panels that still lack it."""
    return [m for m in METHODS if m in d]

# S0 terminal-outcome distribution (% episodes, mean over seeds).
OUTCOMES = {
    "FB":    {"success": 49.4, "approach": 0.0, "grasp": 9.8,  "maintain": 23.8, "transport": 17.0},
    "GCIQL": {"success": 69.2, "approach": 0.2, "grasp": 12.2, "maintain": 9.2,  "transport": 9.2},
    "CRL":   {"success": 23.2, "approach": 0.4, "grasp": 23.8, "maintain": 28.6, "transport": 24.0},
    "RLDP":  {"success": 25.4, "approach": 0.2, "grasp": 9.6,  "maintain": 44.0, "transport": 20.8},
}
OUTCOME_KEYS = ["success", "approach", "grasp", "maintain", "transport"]
OUTCOME_LAB = ["Success", "Approach fail", "Grasp fail", "Maintain fail", "Transport fail"]
# keep the semantics (success=green, maintain=vermillion, transport=blue, ...)
# but draw from the colorblind palette
OUTCOME_COL = {"success": _CB[2], "approach": _CB[4], "grasp": _CB[1],
               "maintain": _CB[3], "transport": _CB[0]}

# Failure-minus-success nearest-neighbour distance (paper-canonical; cf. appendix
# support table). Columns: pickup Mf-S, carry Mf-S, pickup Tf-S, carry Tf-S.
SUPPORT_DELTA = {
    "FB":    [0.51, 0.13, 0.19, -0.04],
    "GCIQL": [0.82, 0.35, 0.75, 0.34],
    "CRL":   [1.42, 0.12, 0.39, 0.31],
    "RLDP":  [0.47, 0.41, 0.44, 0.34],
}
SUPPORT_COLS = ["Pickup\nMf$-$S", "Carry\nMf$-$S", "Pickup\nTf$-$S", "Carry\nTf$-$S"]

PHASES = ["approach", "pickup", "carry"]


def _funnel_stages(o):
    """Cumulative survival through the pipeline (% episodes)."""
    reached = 100 - o["approach"]
    acquired = reached - o["grasp"]
    maintained = acquired - o["maintain"]
    return [reached, acquired, maintained, o["success"]]


def fig_funnel(out: Path) -> None:
    """4-stage phase funnel (survival lines), single panel."""
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    stages = ["Reached", "Control\nacquired", "Control\nmaintained", "Goal\nreached"]
    x = np.arange(4)
    for m in _present(OUTCOMES):
        ax.plot(x, _funnel_stages(OUTCOMES[m]), "o-", color=MC[m], lw=2, label=m)
    ax.set_xticks(x); ax.set_xticklabels(stages, fontsize=9)
    ax.set_ylabel("% of episodes", fontsize=10); ax.set_ylim(0, 102)
    ax.grid(True, axis="y", alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[plot] wrote {out}")


def fig_outcomes(out: Path) -> None:
    """Stacked terminal-outcome composition, single panel."""
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    methods = _present(OUTCOMES)
    xb = np.arange(len(methods)); bottom = np.zeros(len(methods))
    for k, lab in zip(OUTCOME_KEYS, OUTCOME_LAB):
        h = np.array([OUTCOMES[m][k] for m in methods])
        ax.bar(xb, h, bottom=bottom, color=OUTCOME_COL[k], label=lab,
               edgecolor="black", linewidth=0.4)
        for xi, (hi, bi) in enumerate(zip(h, bottom)):
            if hi >= 4:
                ax.text(xi, bi + hi / 2, f"{hi:.0f}", ha="center", va="center",
                        fontsize=8, color="white" if hi >= 8 else "black")
        bottom += h
    ax.set_xticks(xb); ax.set_xticklabels(methods, fontsize=10, fontweight="bold")
    ax.set_ylabel("% of episodes", fontsize=10); ax.set_ylim(0, 102)
    # horizontal legend above the axes so it never overlaps the bars
    ax.legend(fontsize=7, ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.01),
              frameon=False, columnspacing=1.0, handletextpad=0.4)
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[plot] wrote {out}")


def fig_support(out: Path) -> None:
    methods = _present(SUPPORT_DELTA)
    M = np.array([SUPPORT_DELTA[m] for m in methods])
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    vmax = np.abs(M).max()
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(SUPPORT_COLS))); ax.set_xticklabels(SUPPORT_COLS, fontsize=9)
    ax.set_yticks(range(len(methods))); ax.set_yticklabels(methods, fontsize=10, fontweight="bold")
    for i in range(len(methods)):
        for j in range(len(SUPPORT_COLS)):
            ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center",
                    fontsize=9, color="black")
    ax.set_title("Failure $-$ success nearest-neighbour distance", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="$\\Delta m_{\\mathrm{dist}}$")
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[plot] wrote {out}")


def _aliasing_table():
    f = ANALYSIS / "value/training_value_multiseed/aggregate_rldp/aliasing_matched_pairs.parquet"
    d = pd.read_parquet(f)
    ml = {"fb": "FB", "gciql": "GCIQL", "crl": "CRL", "rldp": "RLDP",
          "tdmpc2": "TDMPC2"}
    g = d.groupby(["method", "region"])["rank_acc"].agg(["mean", "std"])
    return {ml[m]: {ph: (g.loc[(m, ph), "mean"], g.loc[(m, ph), "std"])
                    for ph in PHASES if (m, ph) in g.index}
            for m in ml}


def fig_aliasing(out: Path) -> None:
    """Matched-pair aliasing ranking by regime, single panel."""
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    al = _aliasing_table()
    methods = [m for m in METHODS if al.get(m)]
    x = np.arange(len(methods)); w = 0.25
    for i, ph in enumerate(PHASES):
        vals = [al[m].get(ph, (np.nan, 0))[0] for m in methods]
        errs = [al[m].get(ph, (np.nan, 0))[1] for m in methods]
        ax.bar(x + (i - 1) * w, vals, w, yerr=errs, capsize=3, label=ph.capitalize(),
               edgecolor="black", linewidth=0.3)
    ax.axhline(0.5, color="black", ls="--", lw=1.0, label="chance")
    ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=10, fontweight="bold")
    ax.set_ylabel("success-bound ranked higher", fontsize=10)
    ax.set_ylim(0.4, 0.7); ax.legend(fontsize=8, ncol=2); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[plot] wrote {out}")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    fig_funnel(OUT / "fig_funnel.png")
    fig_outcomes(OUT / "fig_outcomes.png")
    fig_support(OUT / "fig_support.png")
    fig_aliasing(OUT / "fig_aliasing.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""scripts/figures/plot_value_aliasing.py — value-function aliasing (rlbrew P4).

The policy follows the value, so the most behaviorally direct aliasing test is
whether the scalar value Q = F(s,pi,z).z separates success-bound from fail-bound
in-hand grasps. It does not (see the value rung of the aliasing ladder). This
figure shows the overlap directly.

Top row (train, robust): per-task-standardized value for success-bound vs
fail-bound grasps -- the distributions sit on top of each other (AUC ~0.5).
Bottom row (eval, illustrative): value at the policy's own in-hand grasps split
by terminal outcome (success / maintain-fail / transport-fail); high-variance
over ~50 episodes.

Reads analysis/value/repsep/value_{train,eval}_{fb,rldp}.parquet and
forward_probe_{fb,rldp}.json. Run under .venv.
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
SUCC, MAINT, TRANS = _CB[2], _CB[3], _CB[0]      # green / vermillion / blue
METHODS = [("fb", "FB"), ("rldp", "RLDP")]
EVAL_ORDER = ["success", "maintain_fail", "transport_fail"]
EVAL_LAB = {"success": "Success", "maintain_fail": "Maintain\nfail",
            "transport_fail": "Transport\nfail"}
EVAL_COL = {"success": SUCC, "maintain_fail": MAINT, "transport_fail": TRANS}


def main() -> int:
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.0))
    for j, (m, lab) in enumerate(METHODS):
        js = json.loads((REPSEP / f"forward_probe_{m}.json").read_text())

        # ---- train (success-bound vs fail-bound), per-task standardized ----
        tr = pd.read_parquet(REPSEP / f"value_train_{m}.parquet")
        ax = axes[0, j]
        tr["lab"] = np.where(tr["success"], "Success-bound", "Fail-bound")
        sns.violinplot(data=tr, x="lab", y="q_z", hue="lab",
                       order=["Success-bound", "Fail-bound"],
                       palette={"Success-bound": SUCC, "Fail-bound": MAINT},
                       cut=0, inner="quartile", legend=False, ax=ax)
        ax.set_title(f"{lab} — train", fontsize=12, fontweight="bold")
        ax.set_xlabel(""); ax.set_ylabel("value $Q$ (per-task $z$-scored)" if j == 0 else "")
        ax.text(0.5, 0.97, f"AUC {js['train']['V_mean']:.2f}", transform=ax.transAxes,
                ha="center", va="top", fontsize=11, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.5", alpha=0.85))

        # ---- eval (by terminal outcome), z-scored within method ----
        ev = pd.read_parquet(REPSEP / f"value_eval_{m}.parquet")
        ev = ev[ev["outcome"].isin(EVAL_ORDER)].copy()
        ev["q_z"] = (ev["q"] - ev["q"].mean()) / (ev["q"].std() + 1e-8)
        order = [o for o in EVAL_ORDER if (ev["outcome"] == o).any()]
        ax = axes[1, j]
        sns.violinplot(data=ev, x="outcome", y="q_z", hue="outcome", order=order,
                       palette={o: EVAL_COL[o] for o in order}, cut=0,
                       inner="quartile", legend=False, ax=ax)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([EVAL_LAB[o] for o in order], fontsize=9)
        ax.set_title(f"{lab} — eval (illustrative)", fontsize=12, fontweight="bold")
        ax.set_xlabel(""); ax.set_ylabel("value $Q$ ($z$-scored)" if j == 0 else "")

    fig.suptitle("Value aliasing: the score the policy follows barely separates "
                 "success- from fail-bound grasps", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out = OUT / "fig_value_aliasing.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

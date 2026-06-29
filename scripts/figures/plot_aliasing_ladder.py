#!/usr/bin/env python
"""scripts/figures/plot_aliasing_ladder.py — the P4 aliasing ladder (single clean figure).

Per-task linear-probe AUC (train grasps, independent states) for separating
success-bound from fail-bound in-hand grasps, across the FB-family representation
stack: the forward-side state embedding phi(s), the backward/successor embedding
B(s), the forward map F(s,pi,z) (which predicts successor occupancy), and the
scalar value Q = F.z that the policy follows.

Two reference lines: chance (0.5) and the raw full-state ceiling (~0.64, the best
a linear probe does from the ground-truth observation). Every learned rung sits
between them, and separability decays to chance at the value -- no representation,
input/backward/forward/value, tells the grasps apart, because the deciding
variable (post-contact controllability) is not in the instantaneous state.

Reads analysis/value/repsep/grasp_feature.json and forward_probe_{fb,rldp}.json.
Run under .venv.
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
FB_C, RLDP_C = _CB[0], _CB[3]

RUNGS = [("phi_mean", r"$\phi(s)$" + "\ninput"),
         ("B_mean",   r"$B(s)$" + "\nbackward"),
         ("F_mean",   r"$F(s,\pi,z)$" + "\nforward"),
         ("V_mean",   r"$Q{=}F{\cdot}z$" + "\nvalue")]


def main() -> int:
    gf = json.loads((REPSEP / "grasp_feature.json").read_text())
    fb = json.loads((REPSEP / "forward_probe_fb.json").read_text())["train"]
    rl = json.loads((REPSEP / "forward_probe_rldp.json").read_text())["train"]
    ceiling = gf["full state"]["mean"]

    x = np.arange(len(RUNGS)); w = 0.38
    fbv = [fb[k] for k, _ in RUNGS]
    rlv = [rl[k] for k, _ in RUNGS]

    fig, ax = plt.subplots(figsize=(8.0, 4.3))
    b1 = ax.bar(x - w / 2, fbv, w, color=FB_C, edgecolor="black", linewidth=0.5,
                label="FB")
    b2 = ax.bar(x + w / 2, rlv, w, color=RLDP_C, edgecolor="black", linewidth=0.5,
                label="RLDP")
    for b in (b1, b2):
        ax.bar_label(b, fmt="%.2f", fontsize=8, padding=2)

    ax.axhline(ceiling, color="0.4", ls="--", lw=1.2)
    ax.text(len(RUNGS) - 0.5, ceiling + 0.006, f"raw full-state ceiling ({ceiling:.2f})",
            fontsize=8.5, color="0.3", va="bottom", ha="right")
    ax.axhline(0.5, color="black", ls="--", lw=1.0)
    ax.text(len(RUNGS) - 0.5, 0.505, "chance", fontsize=8.5, va="bottom", ha="right")

    ax.set_xticks(x); ax.set_xticklabels([lab for _, lab in RUNGS], fontsize=10)
    ax.set_ylim(0.4, 0.78)
    ax.set_ylabel("Linear-probe AUC\n(success- vs fail-bound, per task)", fontsize=10)
    ax.legend(fontsize=10, loc="upper right", frameon=False)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = OUT / "fig_repsep_ladder.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out}")
    print("FB:", {lab.split(chr(10))[0]: round(v, 3) for (_, lab), v in zip(RUNGS, fbv)})
    print("RLDP:", {lab.split(chr(10))[0]: round(v, 3) for (_, lab), v in zip(RUNGS, rlv)})
    return 0


if __name__ == "__main__":
    sys.exit(main())

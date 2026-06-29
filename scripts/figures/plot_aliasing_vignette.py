#!/usr/bin/env python
"""scripts/figures/plot_aliasing_vignette.py — candidate Figure 1 (interaction-gap vignette).

For each fail-bound state in the contact regimes (pickup+carry), find its nearest
success-bound state in observation space and compare the learned value of the two.
If the value separated controllable from about-to-fail states, every point would
sit above the diagonal (success ranked higher); interaction-mode aliasing puts the
cloud ON the diagonal. z-scored value per method. Also prints one concrete
exemplar pair (observation-near, opposite outcome, near-equal value).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parent.parent.parent
PAIR = REPO / "analysis/value/training_value_multiseed/p0"
OUT = REPO / "PAPER" / "rlbrew" / "figures" / "fig_vignette.png"
ML = {"fb": ("FB", "V_policy"), "gciql": ("GCIQL", "V"),
      "crl": ("CRL", "V"), "rldp": ("RLDP", "V"), "tdmpc2": ("TDMPC2", "V")}
CONTACT = {"grasp", "transport"}   # pickup + carry regimes
RNG = np.random.default_rng(0)


def main() -> int:
    st = np.load(PAIR / "training_states.npz", allow_pickle=True)
    obs = np.asarray(st["obs"], np.float32)
    region = np.array([str(r) for r in st["region"]])
    outcome = np.asarray(st["outcome"], bool)
    cube = np.asarray(st["cube"], np.float64)
    goals = np.asarray(st["goals"], np.float64)
    mu, sd = obs.mean(0), obs.std(0) + 1e-8
    oz = (obs - mu) / sd
    contact = np.isin(region, list(CONTACT))

    # one panel per method whose value parquet exists in this pair dir
    panels = [(m, lv) for m, lv in ML.items()
              if (PAIR / f"{m}_values.parquet").exists()]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.25 * len(panels), 3.4))
    if len(panels) == 1:
        axes = [axes]
    exemplar = None
    for ax, (m, (lab, vc)) in zip(axes, panels):
        V = pd.read_parquet(PAIR / f"{m}_values.parquet")[vc].to_numpy(float)
        n = V.shape[0] // 5
        Vt = V.reshape(5, n).T  # [n_states, 5 tasks]
        vs, vf, dists, cubed = [], [], [], []
        for ti in range(5):
            idx = np.where(contact)[0]
            su = outcome[idx, ti]
            pos, neg = idx[su], idx[~su]
            if len(pos) < 10 or len(neg) < 10:
                continue
            tree = cKDTree(oz[pos])
            dd, nb = tree.query(oz[neg], k=1)
            mp = pos[np.asarray(nb).ravel()]
            vs.append(Vt[mp, ti]); vf.append(Vt[neg, ti])
            dists.append(np.asarray(dd));
            cubed.append(np.linalg.norm(cube[neg] - cube[mp], axis=1))
        vs = np.concatenate(vs); vf = np.concatenate(vf)
        dd = np.concatenate(dists); cd = np.concatenate(cubed)
        # z-score jointly for this method
        allv = np.concatenate([vs, vf]); z = lambda a: (a - allv.mean()) / (allv.std() + 1e-9)
        zs, zf = z(vs), z(vf)
        k = min(600, len(zs)); sel = RNG.choice(len(zs), k, replace=False)
        ax.scatter(zf[sel], zs[sel], s=6, alpha=0.3, color="#1f77b4", edgecolors="none")
        lim = [min(zf.min(), zs.min()), max(zf.max(), zs.max())]
        ax.plot(lim, lim, "k--", lw=1)
        acc = float(np.mean(vs > vf))
        ax.set_title(f"{lab}  (rank acc {acc:.2f})", fontsize=10)
        ax.set_xlabel("value (fail-bound)", fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("value (success-bound\nnearest neighbour)", fontsize=9)
        ax.set_aspect("equal", "box")
        # capture an exemplar from FB: observation-near, opposite outcome, near-equal value
        if m == "fb":
            score = dd + 5 * np.abs(zs - zf)  # close in obs AND value
            j = int(np.argmin(score))
            exemplar = dict(obs_dist=float(dd[j]), cube_dist_m=float(cd[j]),
                            v_succ=float(vs[j]), v_fail=float(vf[j]))
    fig.suptitle("Observation-near states with opposite futures receive nearly the same value",
                 fontsize=12)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[vignette] wrote {OUT}")
    if exemplar:
        print(f"[exemplar FB] obs-dist(std) {exemplar['obs_dist']:.2f} "
              f"(median pair ~7); cube positions differ by {exemplar['cube_dist_m']*100:.1f} cm; "
              f"V(success)={exemplar['v_succ']:.1f} vs V(fail)={exemplar['v_fail']:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

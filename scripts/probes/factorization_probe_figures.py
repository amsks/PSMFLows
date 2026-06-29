"""Visualize the FB terminal-geometry failure mode (run in .venv).

Two views of "can you see the cube-to-goal geometry in the representation?", on
transport-phase states, for FB state-B vs FB pixel-B vs GCIQL state-value-phi:
  Row 1 (PCA):     2-D PCA of the representation, points colored by cube->goal
                   distance d. Organized gradient = geometry encoded; scrambled
                   = not. R^2(ridge: rep->d) in each title.
  Row 2 (spatial): cube-xy field of the ridge-predicted distance (held-out),
                   goal marked with a star. Bullseye = geometry recoverable;
                   flat blur = not.

  .venv/bin/python -m scripts.probes.factorization_probe_figures \
    --fb-state analysis/probes/factorization_probe/s3/embeddings.npz \
    --fb-pixel analysis/probes/factorization_probe_pixel/s0/embeddings.npz \
    --gciql-state analysis/features_raw/gciql_feature/state_sd001/task1.npz \
    --out analysis/probes/factorization_probe/failure_mode.png
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evals.training_value import region_labels, cube_to_goal_dist
from evals.phase_probe import Thresholds
from evals.factorization_probe import _ridge_predict, _standardize, r2_score


def load_fb(npz_path, task=0):
    z = np.load(npz_path, allow_pickle=True)
    cube = z["cube"].astype(np.float64); goal = z["goals"][task].astype(np.float64)
    region = np.array([str(r) for r in z["region"]])
    return dict(rep=z["B"].astype(np.float64), cube=cube, goal=goal, region=region,
                d=cube_to_goal_dist(cube, goal))


def load_gciql(npz_path):
    z = np.load(npz_path, allow_pickle=True); thr = Thresholds()
    cube = z["cube"].astype(np.float64); goal = z["goal_xyz"].astype(np.float64)
    grip = np.clip(z["grip"].astype(np.float64) / 0.8, 0, 1)
    lift = z["lift"].astype(np.float64) - float(z["table_z"])
    region = np.array([str(r) for r in region_labels(grip, lift, thr)])
    return dict(rep=z["phi"].astype(np.float64), cube=cube, goal=goal, region=region,
                d=cube_to_goal_dist(cube, goal))


def _pca2(X):
    Xc = X - X.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:2].T


def _heldout(data, rng):
    idx = np.where(data["region"] == "transport")[0]
    rng.shuffle(idx); cut = len(idx) // 2
    tr, te = idx[:cut], idx[cut:]
    Xtr, Xte = _standardize(data["rep"][tr], data["rep"][te])
    dhat = _ridge_predict(Xtr, data["d"][tr], Xte)
    return te, dhat, r2_score(data["d"][te], dhat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fb-state", required=True)
    ap.add_argument("--fb-pixel", required=True)
    ap.add_argument("--gciql-state", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cols = [("FB state-B", load_fb(args.fb_state)),
            ("FB pixel-B", load_fb(args.fb_pixel)),
            ("GCIQL state-φ", load_gciql(args.gciql_state))]
    rng = np.random.default_rng(0)
    # Shared color scale (cube->goal distance) so panels are directly comparable.
    dvals = np.concatenate([d["d"][d["region"] == "transport"] for _, d in cols])
    vmin, vmax = 0.0, float(np.percentile(dvals, 97))
    fig, axes = plt.subplots(3, 3, figsize=(12, 11.5))

    for j, (label, data) in enumerate(cols):
        te, dhat, r2 = _heldout(data, np.random.default_rng(0))

        # Row 0: PCA colored by cube->goal distance (transport states).
        m = np.where(data["region"] == "transport")[0]
        sub = rng.choice(m, min(2500, len(m)), replace=False)
        Z = _pca2(data["rep"][sub])
        sc = axes[0, j].scatter(Z[:, 0], Z[:, 1], c=data["d"][sub], cmap="viridis",
                                s=7, alpha=0.75, vmin=vmin, vmax=vmax)
        axes[0, j].set_title(f"{label}  (R²={r2:.2f})\nPCA colored by cube→goal dist", fontsize=11)
        axes[0, j].set_xticks([]); axes[0, j].set_yticks([])
        fig.colorbar(sc, ax=axes[0, j], fraction=0.046)

        # Row 1: calibration — held-out predicted vs true distance (blob vs diagonal).
        axes[1, j].scatter(data["d"][te], dhat, s=6, alpha=0.35, color="steelblue")
        axes[1, j].plot([vmin, vmax], [vmin, vmax], "k--", lw=1)
        axes[1, j].set_xlim(vmin, vmax); axes[1, j].set_ylim(vmin, vmax)
        axes[1, j].set_title("predicted vs true distance", fontsize=11)
        axes[1, j].set_xlabel("true cube→goal dist"); axes[1, j].set_ylabel("predicted")

        # Row 2: cube-xy field of held-out predicted distance.
        cx, cy = data["cube"][te, 0], data["cube"][te, 1]
        nb = 12
        xr = (cx.min(), cx.max()); yr = (cy.min(), cy.max())
        ix = np.clip(((cx - xr[0]) / (xr[1] - xr[0] + 1e-9) * nb).astype(int), 0, nb - 1)
        iy = np.clip(((cy - yr[0]) / (yr[1] - yr[0] + 1e-9) * nb).astype(int), 0, nb - 1)
        s = np.zeros((nb, nb)); cnt = np.zeros((nb, nb))
        np.add.at(s, (iy, ix), dhat); np.add.at(cnt, (iy, ix), 1)
        H = np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)
        im = axes[2, j].imshow(H, origin="lower", extent=[xr[0], xr[1], yr[0], yr[1]],
                               cmap="viridis", aspect="auto", vmin=vmin, vmax=vmax)
        axes[2, j].plot(data["goal"][0], data["goal"][1], marker="*", color="red",
                        ms=18, mec="white", mew=1.2)
        axes[2, j].set_title("cube-xy: predicted cube→goal dist", fontsize=11)
        axes[2, j].set_xticks([]); axes[2, j].set_yticks([])
        fig.colorbar(im, ax=axes[2, j], fraction=0.046)

    fig.suptitle("Terminal-geometry resolution: FB state-B discards cube→goal geometry; "
                 "pixel-B and GCIQL value retain it", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"[figures] wrote {args.out}")


if __name__ == "__main__":
    main()

"""B0 — precompute the goal-agnostic cube-position density for Experiment B.

Builds a 3D histogram over cube xyz (physics[:, 14:17]) across the whole offline
buffer and writes analysis/cube_density.npz (H, ex, ey, ez). This is SELF-SUPERVISED:
derived only from the offline data's own cube-position density, never from eval/task
goals. agents/fb/agent.py loads it when reweight_alpha > 0.

Run on the box that has the dataset:
    .venv/bin/python -m scripts.data.cube_density \
        --data datasets/cube-single-play-v0/buffer --out analysis/cube_density.npz
See docs/superpowers/specs/2026-05-25-fb-Bg-cause-isolation.md (Experiment B, B0).
"""
import argparse
import glob
import os

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/cube-single-play-v0/buffer",
                    help="dir of episode *.npz files (each with a 'physics' array)")
    ap.add_argument("--out", default="analysis/cube_density.npz")
    ap.add_argument("--bins", type=int, nargs=3, default=[16, 16, 8])
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.data, "*.npz")))
    if not files:
        raise SystemExit(f"no .npz episodes found at {args.data}")
    cubes = np.concatenate(
        [np.load(f)["physics"][:, 14:17] for f in files]
    ).astype(np.float64)

    nb = tuple(args.bins)
    edges = [np.linspace(cubes[:, i].min(), cubes[:, i].max(), nb[i] + 1) for i in range(3)]
    H, _ = np.histogramdd(cubes, bins=edges)
    H = H / H.sum()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, H=H, ex=edges[0], ey=edges[1], ez=edges[2])
    print(f"wrote {args.out}: grid {H.shape}, nonzero cells {int((H > 0).sum())}, "
          f"{len(cubes)} states from {len(files)} episodes")


if __name__ == "__main__":
    main()

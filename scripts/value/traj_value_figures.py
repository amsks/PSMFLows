"""Plot the trajectory value profiles (run in .venv). Shows whether the
goal-conditioned value RISES toward the hindsight goal along coherent real play
trajectories — the marginalization-free version of "does the value peak at the
goal". Value is min-max normalized PER TRAJECTORY so FB and GCIQL shapes compare.

  .venv/bin/python -m scripts.value.traj_value_figures \
    --fb analysis/value/traj_value/fb_s3.parquet \
    --gciql analysis/value/traj_value/gciql_sd001.parquet \
    --out analysis/value/traj_value/traj_value.png
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _norm(df):
    df = df.copy()
    g = df.groupby("traj")["V"]
    vmin, vmax = g.transform("min"), g.transform("max")
    r = (vmax - vmin)
    df["Vn"] = np.where(r > 1e-9, (df["V"] - vmin) / r.where(r > 1e-9, 1.0), 0.5)
    return df


def _binned(x, y, bins):
    idx = np.clip(np.digitize(x, bins) - 1, 0, len(bins) - 2)
    ctr = 0.5 * (bins[:-1] + bins[1:])
    m = np.full(len(bins) - 1, np.nan); s = np.full(len(bins) - 1, np.nan)
    for b in range(len(bins) - 1):
        yb = y[idx == b]
        if len(yb) > 5:
            m[b] = yb.mean(); s[b] = yb.std() / np.sqrt(len(yb))
    return ctr, m, s


def _workspace(ax, df, title, n_show=10, seed=0):
    rng = np.random.default_rng(seed)
    trajs = rng.choice(df["traj"].unique(), min(n_show, df["traj"].nunique()), replace=False)
    for tr in trajs:
        g = df[df.traj == tr]
        ax.scatter(g["cube_x"], g["cube_y"], c=g["Vn"], cmap="viridis",
                   s=8, alpha=0.7, vmin=0, vmax=1)
        gp = g.loc[g["d"].idxmin()]          # the (hindsight) goal end of this traj
        ax.plot(gp["cube_x"], gp["cube_y"], marker="*", color="red", ms=14, mec="white", mew=0.8)
    ax.set_title(title, fontsize=11); ax.set_xlabel("cube x"); ax.set_ylabel("cube y")
    ax.set_aspect("equal", "box")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fb", required=True); ap.add_argument("--gciql", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    fb, gq = _norm(pd.read_parquet(args.fb)), _norm(pd.read_parquet(args.gciql))
    series = [(fb, "FB", "tab:red"), (gq, "GCIQL", "tab:blue")]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # (0,0) value vs cube->goal distance — rises as d->0 if value peaks at goal.
    dmax = float(np.percentile(np.concatenate([fb["d"], gq["d"]]), 97))
    dbins = np.linspace(0, dmax, 18)
    for df, lab, c in series:
        ctr, m, s = _binned(df["d"].to_numpy(), df["Vn"].to_numpy(), dbins)
        axes[0, 0].plot(ctr, m, color=c, label=lab, lw=2)
        axes[0, 0].fill_between(ctr, m - s, m + s, color=c, alpha=0.2)
    axes[0, 0].invert_xaxis()
    axes[0, 0].set_xlabel("cube→goal distance  (→ approaching goal)")
    axes[0, 0].set_ylabel("normalized value (per-traj)")
    axes[0, 0].set_title("Value vs distance-to-goal along trajectories\n(rise toward d→0 = value peaks at the goal)")
    axes[0, 0].legend()

    # (0,1) value vs progress to the goal step.
    pbins = np.linspace(0, 1, 18)
    for df, lab, c in series:
        ctr, m, s = _binned(df["progress"].to_numpy(), df["Vn"].to_numpy(), pbins)
        axes[0, 1].plot(ctr, m, color=c, label=lab, lw=2)
        axes[0, 1].fill_between(ctr, m - s, m + s, color=c, alpha=0.2)
    axes[0, 1].set_xlabel("progress to goal step  (t / t_goal)")
    axes[0, 1].set_ylabel("normalized value (per-traj)")
    axes[0, 1].set_title("Value vs progress to goal\n(rise toward 1 = goal-conditioned)")
    axes[0, 1].legend()

    # (1,*) workspace: real cube paths colored by value, hindsight goal = ★.
    _workspace(axes[1, 0], fb, "FB — cube paths colored by value (★ = goal)")
    _workspace(axes[1, 1], gq, "GCIQL — cube paths colored by value (★ = goal)")

    fig.suptitle("Goal-conditioned value along real play trajectories (hindsight goal) — "
                 "coherent states, no cube-xy marginalization", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"[traj_value_figures] wrote {args.out}")


if __name__ == "__main__":
    main()

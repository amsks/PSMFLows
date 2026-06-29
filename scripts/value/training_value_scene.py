"""scripts/value/training_value_scene.py — eval phase_scene analog on TRAINING
states. Per task per method (FB policy, GCIQL, CRL): value-saliency heatmap (from
the scored multiseed parquets) over the top-down scene, gridded by phase x
outcome, + a dataset-support panel. Pure data
+ matplotlib; no model loading. Run under .venv.

Usage:
    .venv/bin/python scripts/value/training_value_scene.py \
        --root analysis/value/training_value_multiseed --data-path datasets \
        --mujoco-gl glfw
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals.training_value import region_labels, flow_step_labels
from evals.phase_probe import Thresholds
from evals.dataset_support import support_kde
from scripts.profiles.gciql_profile_aggregate import (_bin_vz_grid, _phase_action_fields,
                                             _nan_gaussian)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import matplotlib.image as mpimg         # noqa: E402
from matplotlib.lines import Line2D      # noqa: E402
from matplotlib.patches import Patch     # noqa: E402

CUBE_SLICE = slice(14, 17)
GRIP_QPOS_IDX = 6
TASK_TMPL = "cube-single-play-singletask-task{n}-v0"
REGIONS = [("reach", "reach"), ("grasp", "grasp"), ("transport", "transport")]
OUTCOMES = ["success_bound", "fail_bound"]
NBINS, NARROW, LOW_N = 16, 10, 30


def _slug(s: str) -> str:
    return s.replace(" ", "_")


def _pool(root: Path, name: str) -> pd.DataFrame:
    frames = []
    for pdir in sorted(d for d in root.glob("p*") if d.is_dir()):
        f = pdir / name
        if f.exists():
            frames.append(pd.read_parquet(f))
    if not frames:
        raise SystemExit(f"no {name} under {root}/p*/")
    return pd.concat(frames, ignore_index=True)


def _value_frame(root: Path) -> pd.DataFrame:
    fb = _pool(root, "fb_values.parquet")[
        ["task", "region", "outcome", "cube_x", "cube_y", "V_policy"]
    ].rename(columns={"V_policy": "V"})
    fb["method"] = "FB policy"
    gq = _pool(root, "gciql_values.parquet")[
        ["task", "region", "outcome", "cube_x", "cube_y", "V"]
    ].copy()
    gq["method"] = "GCIQL"
    frames = [fb, gq]
    if any((d / "crl_values.parquet").exists() for d in root.glob("p*")) \
            or (root / "crl_values.parquet").exists():
        cr = _pool(root, "crl_values.parquet")[
            ["task", "region", "outcome", "cube_x", "cube_y", "V"]
        ].copy()
        cr["method"] = "CRL"
        frames.append(cr)
    return pd.concat(frames, ignore_index=True)


def _table_z(seed, obs_type) -> float:
    from envs.ogbench import create_ogbench_env
    e0, _ = create_ogbench_env(TASK_TMPL.format(n=1), seed=seed,
                               obs_type=obs_type)
    tb = int(getattr(e0.unwrapped, "_target_block", 0) or 0)
    tz = float(e0.unwrapped.cur_task_info["init_xyzs"][tb][2])
    e0.close()
    return tz


def _flow_frame(data_path, goals_xyz, thr, table_z, horizon, thresh,
                n_eps, seed=0):
    """Long flow DataFrame (task, outcome, region, episode, t, cube_x, cube_y)
    + raw play cube-xy [N,2] for support."""
    files = sorted(glob.glob(str(Path(data_path) / "cube-single-play-v0"
                                  / "buffer" / "episode_*.npz")))
    rng = np.random.default_rng(seed)
    if len(files) > n_eps:
        files = [files[i] for i in sorted(rng.choice(len(files), n_eps,
                                                     replace=False))]
    rows, sup = [], []
    for ei, f in enumerate(files):
        phys = np.asarray(np.load(f)["physics"], np.float32)
        cube = phys[:, CUBE_SLICE].astype(np.float64)
        grip = np.clip(phys[:, GRIP_QPOS_IDX] / 0.8, 0, 1)
        region = region_labels(grip, phys[:, 16] - table_z, thr)  # [T]
        oc = flow_step_labels(cube, goals_xyz, horizon, thresh)    # [T, 5]
        sup.append(cube[:, :2])
        T = len(cube)
        for ti in range(goals_xyz.shape[0]):
            df = pd.DataFrame({
                "task": f"task{ti + 1}",
                "outcome": np.where(oc[:, ti], "success_bound", "fail_bound"),
                "region": region, "episode": ei, "t": np.arange(T),
                "cube_x": cube[:, 0], "cube_y": cube[:, 1]})
            rows.append(df)
    return pd.concat(rows, ignore_index=True), np.concatenate(sup)


def _extent(vdf_task):
    xs, ys = vdf_task["cube_x"], vdf_task["cube_y"]
    xlo, xhi = np.percentile(xs, [1, 99])
    ylo, yhi = np.percentile(ys, [1, 99])
    px, py = 0.06 * (xhi - xlo), 0.06 * (yhi - ylo)
    xlo, xhi, ylo, yhi = xlo - px, xhi + px, ylo - py, yhi + py
    xr, yr = xhi - xlo, yhi - ylo
    if xr < yr:
        pad = 0.5 * (yr - xr)
        xlo, xhi = xlo - pad, xhi + pad
    elif yr < xr:
        pad = 0.5 * (xr - yr)
        ylo, yhi = ylo - pad, yhi + pad
    return xlo, xhi, ylo, yhi


def _render(task, method, vdf, fdf, support_xy, anchors, bg_img, out_dir):
    vt = vdf[(vdf.task == task) & (vdf.method == method)]
    if vt.empty:
        return None
    ft = fdf[fdf.task == task]
    xlo, xhi, ylo, yhi = _extent(vt)
    xe = np.linspace(xlo, xhi, NBINS + 1)
    ye = np.linspace(ylo, yhi, NBINS + 1)
    ext = [xlo, xhi, ylo, yhi]
    anc = anchors.get(task, {})
    goal_xy = tuple(anc["goal"]) if "goal" in anc else None
    kx, ky = 0.5 * (xe[:-1] + xe[1:]), 0.5 * (ye[:-1] + ye[1:])
    kz = support_kde(support_xy, kx, ky)

    fig = plt.figure(figsize=(20, 9.5))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1],
                          wspace=0.06, hspace=0.10)
    axes = np.empty((2, 3), dtype=object)
    for rr in range(2):
        for cc in range(3):
            axes[rr, cc] = fig.add_subplot(gs[rr, cc])
    ax_data = fig.add_subplot(gs[0, 3])

    for r, oc in enumerate(OUTCOMES):
        for c, (rg, lbl) in enumerate(REGIONS):
            ax = axes[r, c]
            dv = vt[(vt.outcome == oc) & (vt.region == rg)]
            n = len(dv)
            low = n < LOW_N
            ax.set_xlim(xlo, xhi)
            ax.set_ylim(ylo, yhi)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            if bg_img is not None:
                ax.imshow(bg_img, extent=ext, aspect="auto", zorder=0,
                          interpolation="bilinear")
            else:
                ax.set_facecolor("0.92")
            ax.set_title(f"{oc} — {lbl} (n={n}{', low' if low else ''})",
                         fontsize=9, color="grey" if low else "black")
            if n >= 5:
                ax.scatter([dv["cube_x"].mean()], [dv["cube_y"].mean()],
                           marker="s", s=170, c="white", edgecolors="black",
                           linewidths=1.4, zorder=6)
            if goal_xy is not None:
                ax.scatter([goal_xy[0]], [goal_xy[1]], marker="*", s=900,
                           c="white", alpha=0.55, zorder=6)
                ax.scatter([goal_xy[0]], [goal_xy[1]], marker="*", s=520,
                           c="gold", edgecolors="black", linewidths=1.6,
                           zorder=7)
            if n < 5:
                continue
            a = 0.35 if low else 1.0
            grid, _, _ = _bin_vz_grid(dv.assign(Vz=dv["V"]),
                                      xlo, xhi, ylo, yhi, NBINS)
            vz = _nan_gaussian(grid, sigma=1.0)
            fin = np.isfinite(vz)
            if fin.any():
                lo, hi = np.percentile(vz[fin], [2, 98])
                if hi - lo < 1e-9:
                    hi = lo + 1.0
                norm = np.clip((vz - lo) / (hi - lo), 0.0, 1.0)
                rgba = matplotlib.colormaps["Reds"](norm)
                rgba[..., 3] = np.where(fin, norm * (0.55 * a), 0.0)
                ax.imshow(rgba, origin="lower", extent=ext, zorder=2,
                          aspect="auto", interpolation="bilinear")

    ax_data.set_xlim(xlo, xhi)
    ax_data.set_ylim(ylo, yhi)
    ax_data.set_aspect("equal")
    ax_data.set_xticks([])
    ax_data.set_yticks([])
    if bg_img is not None:
        ax_data.imshow(bg_img, extent=ext, aspect="auto", zorder=0,
                       interpolation="bilinear")
    else:
        ax_data.set_facecolor("0.92")
    in_ext = ((support_xy[:, 0] >= xlo) & (support_xy[:, 0] <= xhi)
              & (support_xy[:, 1] >= ylo) & (support_xy[:, 1] <= yhi))
    sup_clip = support_xy[in_ext]
    n_sub = min(8000, len(sup_clip))
    if n_sub:
        idx = np.random.default_rng(0).choice(len(sup_clip), n_sub,
                                              replace=False)
        ax_data.scatter(sup_clip[idx, 0], sup_clip[idx, 1], s=2.0,
                        c="#ffd86e", alpha=0.18, zorder=1, linewidths=0)
    ax_data.contourf(kx, ky, kz, levels=[0.10, 0.30, 0.60, 1.01],
                     colors=["#f0c0a0", "#e89060", "#d05050"], alpha=0.45,
                     zorder=2)
    ax_data.contour(kx, ky, kz, levels=[0.15, 0.6], colors="white",
                    linewidths=1.2, alpha=0.9, zorder=3)
    if goal_xy is not None:
        ax_data.scatter([goal_xy[0]], [goal_xy[1]], marker="*", s=900,
                        c="white", alpha=0.55, zorder=6)
        ax_data.scatter([goal_xy[0]], [goal_xy[1]], marker="*", s=520,
                        c="gold", edgecolors="black", linewidths=1.6, zorder=7)
    ax_data.set_title("offline dataset support\n(cube xy in the play data)",
                      fontsize=9)

    handles = [
        Patch(facecolor=matplotlib.colormaps["Reds"](0.85),
              edgecolor="black", linewidth=0.4,
              label="value V(s): low → high (per-panel)"),
        Line2D([0], [0], marker="s", color="white", markeredgecolor="black",
               markersize=10, linestyle="", label="cube (mean this phase)"),
        Line2D([0], [0], marker="*", color="gold", markeredgecolor="black",
               markersize=15, linestyle="", label="goal (task target)")]
    axes[0, 0].legend(handles=handles, loc="upper left", fontsize=7,
                      framealpha=0.85)
    fig.suptitle(
        f"{task} — {method}  [training states]    "
        "red = value V(s) (per-panel saliency, low→high)",
        fontsize=12)
    fig.tight_layout()
    fp = out_dir / f"{task}__{_slug(method)}.png"
    fig.savefig(fp, dpi=125)
    plt.close(fig)
    return fp.name


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="analysis/value/training_value_multiseed")
    ap.add_argument("--data-path", default="datasets")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tasks", nargs="*",
                    default=[f"task{i}" for i in range(1, 6)])
    ap.add_argument("--methods", nargs="*", default=["FB policy", "GCIQL", "CRL"])
    ap.add_argument("--n-eps", type=int, default=300)
    ap.add_argument("--horizon", type=int, default=200)
    ap.add_argument("--thresh", type=float, default=0.04)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--obs-type", default="state")
    ap.add_argument("--mujoco-gl", default=None)
    args = ap.parse_args()
    if args.mujoco_gl:
        os.environ["MUJOCO_GL"] = args.mujoco_gl

    root = Path(args.root)
    out_dir = Path(args.out) if args.out else root / "aggregate" / "scene"
    out_dir.mkdir(parents=True, exist_ok=True)

    sx = REPO_ROOT / "analysis" / "misc" / "scene"
    cal = json.loads((sx / "calib.json").read_text())
    bg_img = (mpimg.imread(str(sx / "topdown.png"))
              if (sx / "topdown.png").exists() and cal.get("photoreal")
              else None)
    anchors = (json.loads((sx / "task_anchors.json").read_text())
               if (sx / "task_anchors.json").exists() else {})

    npz = next(iter(sorted(root.glob("p*/training_states.npz"))), None)
    if npz is None:
        raise SystemExit(f"no p*/training_states.npz under {root}")
    goals_xyz = np.asarray(np.load(npz)["goals"], np.float64)  # [5,3]

    vdf = _value_frame(root)
    fdf, support_xy = _flow_frame(args.data_path, goals_xyz, Thresholds(),
                                  _table_z(args.seed, args.obs_type),
                                  args.horizon, args.thresh, args.n_eps,
                                  seed=args.seed)
    written = []
    for task in args.tasks:
        for method in args.methods:
            name = _render(task, method, vdf, fdf, support_xy, anchors,
                           bg_img, out_dir)
            if name:
                written.append(name)
    (out_dir / "_index.md").write_text(
        "# training_value scene (phase x outcome)\n\n"
        + "\n".join(f"- {n}" for n in sorted(written)) + "\n")
    print(f"[training_value_scene] wrote {len(written)} figs -> {out_dir}")


if __name__ == "__main__":
    main()

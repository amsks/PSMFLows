"""Master CSV + figure for the trajectory value profile (run in .venv).

Combines the per-step FB/GCIQL trajectory parquets (from traj_value_profile, whole
training dataset, hindsight goal) into one master CSV, and renders:
  - aggregate value-vs-distance + value-vs-progress over ALL relabelled trajectories,
  - workspace overlays of example trajectories (success: value rises near goal;
    failure: it does not), value-colored cube paths on the MuJoCo top-down + goal ★.

  .venv/bin/python -m scripts.value.traj_value_master \
    --fb analysis/value/traj_value/fb.parquet --gciql analysis/value/traj_value/gciql.parquet \
    --out-dir analysis/value/traj_value
"""
import argparse
import json
from pathlib import Path

import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# canonical scene-overlay machinery (same as training_value_scene / _phase_action_scene)
from scripts.profiles.gciql_profile_aggregate import (_bin_vz_grid, _phase_action_fields,
                                             _nan_gaussian)
from evals._profile_core import _spearman


def _summary(master):
    """Per-(method, trajectory) scores, incl. PHASE-specific value alignment:
      approach_rho  = Spearman(V, -d_effector→cube)  over reach phase
      grasp_rho     = Spearman(V, -d_cube→goal)      over grasp phase
      transport_rho = Spearman(V, -d_cube→goal)      over transport phase
    (approach asks if value tracks reaching the cube; transport if it tracks
    placing it at the goal once in hand.)"""
    def prho(g, phase, col):
        s = g[g.region == phase]
        if len(s) < 5 or s[col].isna().all():
            return float("nan"), int(len(s))
        return float(_spearman(s["V"].to_numpy(), -s[col].to_numpy())), int(len(s))

    rows = []
    for (mth, tr), g in master.groupby(["method", "traj"]):
        ar, nr = prho(g, "reach", "d_eff_cube")
        gr, ng = prho(g, "grasp", "d")
        tp, nt = prho(g, "transport", "d")
        rows.append(dict(method=mth, traj=int(tr), n=len(g),
                         goal_peak_rho=float(_spearman(g["V"].to_numpy(), -g["d"].to_numpy())),
                         approach_rho=ar, grasp_rho=gr, transport_rho=tp,
                         n_reach=nr, n_grasp=ng, n_transport=nt,
                         V_near=float(g.loc[g["d"].idxmin(), "V"]),
                         V_far=float(g.loc[g["d"].idxmax(), "V"])))
    return pd.DataFrame(rows)


def _norm(df):
    g = df.groupby(["method", "traj"])["V"]
    vmin, vmax = g.transform("min"), g.transform("max")
    r = vmax - vmin
    df["Vn"] = np.where(r > 1e-9, (df["V"] - vmin) / r.where(r > 1e-9, 1.0), 0.5)
    return df


def _binned(x, y, bins):
    idx = np.clip(np.digitize(x, bins) - 1, 0, len(bins) - 2)
    ctr = 0.5 * (bins[:-1] + bins[1:])
    m = np.full(len(bins) - 1, np.nan); s = np.full(len(bins) - 1, np.nan)
    for b in range(len(bins) - 1):
        yb = y[idx == b]
        if len(yb) > 20:
            m[b] = yb.mean(); s[b] = yb.std() / np.sqrt(len(yb))
    return ctr, m, s


def _bg_ext():
    sx = REPO_ROOT / "analysis" / "misc" / "scene"
    cal = json.loads((sx / "calib.json").read_text())
    bg = (mpimg.imread(str(sx / "topdown.png"))
          if (sx / "topdown.png").exists() and cal.get("photoreal") else None)
    ws = cal["workspace"]
    return bg, [ws["xmin"], ws["xmax"], ws["ymin"], ws["ymax"]]


def _select_pair(master, summ, min_span=0.12, min_n=20):
    """Two trajectories (same episode shown for BOTH algos): A = FB & GCIQL both
    succeed (value rises near goal); B = FB fails but GCIQL succeeds. Selection
    is over per-trajectory goal_peak_rho for each method (traj i = same episode)."""
    piv = summ.pivot(index="traj", columns="method", values="goal_peak_rho")
    if not {"fb", "gciql"}.issubset(piv.columns):
        return None, None
    ref = master[master.method == "fb"]
    span = ref.groupby("traj")["d"].agg(lambda x: x.max() - x.min())
    cnt = ref.groupby("traj")["t"].count()
    keep = piv.index[(span.reindex(piv.index).fillna(0) >= min_span)
                     & (cnt.reindex(piv.index).fillna(0) >= min_n)]
    v = piv.loc[keep].dropna(subset=["fb", "gciql"])
    if v.empty:
        return None, None
    traj_a = int(v[["fb", "gciql"]].min(axis=1).idxmax())      # both high
    traj_b = int((v["gciql"] - v["fb"]).idxmax())              # GCIQL >> FB
    return traj_a, traj_b


def _rho(summ, method, traj):
    m = summ[(summ.method == method) & (summ.traj == traj)]
    return float(m["goal_peak_rho"].iloc[0]) if len(m) else float("nan")


def _scene_overlay(ax, g, bg, ext, title, nbins=16, narrow=10, arrows=True):
    """Value heatmap (binned, smoothed, Reds) + cube-flow arrows + hindsight
    goal ★ on the MuJoCo top-down — the canonical _phase_action_scene style.
    arrows=False drops the cube-flow quiver layer (value heatmap + goal only)."""
    xlo, xhi, ylo, yhi = ext
    xe = np.linspace(xlo, xhi, nbins + 1); ye = np.linspace(ylo, yhi, nbins + 1)
    if bg is not None:
        ax.imshow(bg, extent=ext, aspect="auto", zorder=0, interpolation="bilinear")
    else:
        ax.set_facecolor("0.9")
    # value heatmap (per-traj normalized value binned over cube-xy)
    grid, _, _ = _bin_vz_grid(g.assign(Vz=g["Vn"]), xlo, xhi, ylo, yhi, nbins)
    try:
        vz = _nan_gaussian(grid, 1.0)
    except Exception:
        vz = grid
    fin = np.isfinite(vz)
    if fin.any():
        lo, hi = np.percentile(vz[fin], [2, 98]); hi = max(hi, lo + 1e-9)
        norm = np.clip((vz - lo) / (hi - lo), 0.0, 1.0)
        rgba = matplotlib.colormaps["Reds"](norm)
        rgba[..., 3] = np.where(fin, norm * 0.6, 0.0)
        ax.imshow(rgba, origin="lower", extent=ext, zorder=2, aspect="auto",
                  interpolation="bilinear")
    # cube-flow arrows (mean cube displacement per cell)
    if arrows:
        ac = 0.5 * (np.linspace(xlo, xhi, narrow + 1)[:-1] + np.linspace(xlo, xhi, narrow + 1)[1:])
        ar = 0.5 * (np.linspace(ylo, yhi, narrow + 1)[:-1] + np.linspace(ylo, yhi, narrow + 1)[1:])
        AX, AY = np.meshgrid(ac, ar)
        _, (U, Vv), _ = _phase_action_fields(
            g.rename(columns={"traj": "episode"}).assign(Vz=0.0), (xe, ye),
            n_arrow=narrow, n_min=5)
        mag = np.hypot(U, Vv); ok = np.isfinite(mag) & (mag > 0)
        if ok.any():
            un = np.where(ok, U / np.maximum(mag, 1e-9), np.nan)
            vn = np.where(ok, Vv / np.maximum(mag, 1e-9), np.nan)
            ax.quiver(AX[ok], AY[ok], un[ok], vn[ok], color="white", edgecolor="black",
                      linewidth=0.5, scale=narrow * 1.1, width=0.007, headwidth=4,
                      zorder=4, pivot="mid")
    # hindsight goal endpoints of the group
    goals = g.loc[g.groupby("traj")["d"].idxmin(), ["cube_x", "cube_y"]]
    ax.scatter(goals["cube_x"], goals["cube_y"], marker="*", s=900, c="gold",
               edgecolors="black", linewidths=1.6, zorder=5)
    ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=10)


def _select_single(master, summ, method, min_span=0.12, min_n=20):
    """One success (value rises toward goal) + one failure (it doesn't) example
    trajectory for a single method, by its own goal_peak_rho with span/count
    filters. Used for the standalone per-method (e.g. CRL) scene overlay."""
    ref = master[master.method == method]
    if ref.empty:
        return None, None
    span = ref.groupby("traj")["d"].agg(lambda x: x.max() - x.min())
    cnt = ref.groupby("traj")["t"].count()
    keep = span.index[(span >= min_span) & (cnt.reindex(span.index).fillna(0) >= min_n)]
    cand = summ[(summ.method == method) & (summ.traj.isin(keep))].dropna(
        subset=["goal_peak_rho"])
    if cand.empty:
        return None, None
    return (int(cand.loc[cand.goal_peak_rho.idxmax(), "traj"]),
            int(cand.loc[cand.goal_peak_rho.idxmin(), "traj"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fb", required=True); ap.add_argument("--gciql", required=True)
    ap.add_argument("--crl", default=None,
                    help="optional CRL per-step parquet (adds CRL to master/summary, "
                         "the aggregate curves, and a standalone CRL scene overlay)")
    ap.add_argument("--rldp", default=None,
                    help="optional RLDP per-step parquet (adds RLDP to master/summary "
                         "and the aggregate curves)")
    ap.add_argument("--effector", default="analysis/value/traj_value/effector.parquet")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    fb = pd.read_parquet(args.fb); gq = pd.read_parquet(args.gciql)
    frames = [fb, gq]
    if args.crl:
        frames.append(pd.read_parquet(args.crl))
    if args.rldp:
        frames.append(pd.read_parquet(args.rldp))
    master = _norm(pd.concat(frames, ignore_index=True))
    # join FK effector + phase (model-independent) by (traj, t)
    eff = pd.read_parquet(args.effector)[["traj", "t", "region", "d_eff_cube",
                                          "eff_x", "eff_y", "eff_z", "cube_z"]]
    master = master.merge(eff, on=["traj", "t"], how="left")
    master.to_csv(out / "master.csv", index=False)
    summ = _summary(master)
    summ.to_csv(out / "summary.csv", index=False)

    bg, ext = _bg_ext()
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.15], hspace=0.28, wspace=0.12)
    axd = fig.add_subplot(gs[0, 0:2]); axp = fig.add_subplot(gs[0, 2:4])
    series = [("fb", "FB", "tab:red"), ("gciql", "GCIQL", "tab:blue")]
    if args.crl:
        series.append(("crl", "CRL", "tab:green"))
    if args.rldp:
        series.append(("rldp", "RLDP", "tab:purple"))

    # Aggregate value vs distance-to-goal (whole dataset).
    dmax = float(np.percentile(master["d"], 97)); dbins = np.linspace(0, dmax, 18)
    for mth, lab, c in series:
        d = master[master.method == mth]
        ctr, m, s = _binned(d["d"].to_numpy(), d["Vn"].to_numpy(), dbins)
        axd.plot(ctr, m, color=c, lw=2, label=lab); axd.fill_between(ctr, m - s, m + s, color=c, alpha=0.2)
    axd.invert_xaxis(); axd.set_xlabel("cube→goal distance  (→ approaching goal)")
    axd.set_ylabel("normalized value (per-traj)"); axd.legend()
    axd.set_title("Value vs distance-to-goal — ALL training trajectories (hindsight)\n"
                  "rise toward d→0 = value peaks at goal")

    # Aggregate value vs progress.
    pbins = np.linspace(0, 1, 18)
    for mth, lab, c in series:
        d = master[master.method == mth]
        ctr, m, s = _binned(d["progress"].to_numpy(), d["Vn"].to_numpy(), pbins)
        axp.plot(ctr, m, color=c, lw=2, label=lab); axp.fill_between(ctr, m - s, m + s, color=c, alpha=0.2)
    axp.set_xlabel("progress to goal step (t / t_goal)"); axp.set_ylabel("normalized value")
    axp.legend(); axp.set_title("Value vs progress — ALL training trajectories")

    # Example overlays: success (value rises) + failure (it doesn't) per method.
    traj_a, traj_b = _select_pair(master, summ)
    cells = [(traj_a, "fb", "A — FB & GCIQL succeed"),
             (traj_a, "gciql", "A — FB & GCIQL succeed"),
             (traj_b, "fb", "B — FB fails, GCIQL succeeds"),
             (traj_b, "gciql", "B — FB fails, GCIQL succeeds")]
    for j, (tr, mth, tag) in enumerate(cells):
        ax = fig.add_subplot(gs[1, j])
        lab = dict(fb="FB", gciql="GCIQL")[mth]
        if tr is None:
            ax.set_title("no trajectory found"); ax.set_xticks([]); ax.set_yticks([]); continue
        g = master[(master.method == mth) & (master.traj == tr)].sort_values("t")
        _scene_overlay(ax, g, bg, ext,
                       f"{lab} · traj {tr}  ({tag})\nρ(V,−d)={_rho(summ, mth, tr):.2f}  ·  red=value · ★=goal")

    fig.suptitle("Goal-conditioned value along real training trajectories (hindsight goal): "
                 "GCIQL value rises toward the goal; FB stays flat (B can't resolve the goal)",
                 fontsize=12)
    fig.savefig(out / "traj_value_master.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # Standalone CRL scene overlay (success + failure example), same style as the
    # FB/GCIQL bottom panels — keeps the FB-vs-GCIQL figure above untouched.
    methods_present = (["fb", "gciql"] + (["crl"] if args.crl else [])
                       + (["rldp"] if args.rldp else []))
    if args.crl:
        cs, cf = _select_single(master, summ, "crl")
        cfig, caxes = plt.subplots(1, 2, figsize=(9, 5))
        for ax, tr, tag in ((caxes[0], cs, "success: value rises near goal"),
                            (caxes[1], cf, "failure: value stays flat")):
            if tr is None:
                ax.set_title("no trajectory found"); ax.set_xticks([]); ax.set_yticks([]); continue
            g = master[(master.method == "crl") & (master.traj == tr)].sort_values("t")
            _scene_overlay(ax, g, bg, ext,
                           f"CRL · traj {tr}  ({tag})\nρ(V,−d)={_rho(summ, 'crl', tr):.2f}"
                           "  ·  red=value · ★=goal")
        cfig.suptitle("Goal-conditioned value along real training trajectories (hindsight goal) — CRL",
                      fontsize=12)
        cfig.savefig(out / "traj_value_crl_scene.png", dpi=130, bbox_inches="tight")
        plt.close(cfig)

    print(f"[master] {master['traj'].nunique()} trajs/method; wrote {out}/master.csv, "
          f"summary.csv, traj_value_master.png"
          + (", traj_value_crl_scene.png" if args.crl else ""))
    print("  goal_peak_rho mean:  " +
          "  ".join(f"{m}={summ[summ.method==m].goal_peak_rho.mean():.3f}" for m in methods_present))


if __name__ == "__main__":
    main()

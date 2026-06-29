"""scripts/profiles/gciql_profile_aggregate.py — 10-seed GCIQL aggregate
(T1/T4/funnel) + FB-vs-GCIQL comparison. Reuses _profile_core verdicts
so GCIQL and FB are scored identically."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from evals._profile_core import (  # noqa: E402
    _synthesis, _verdict_t1, _verdict_t4)

_SEED_RE = re.compile(r"s(\d+)_final")


def _load(root: Path, name: str) -> pd.DataFrame:
    frames = []
    for p in sorted(Path(root).glob(f"s*_final/{name}.parquet")):
        m = _SEED_RE.search(p.parent.name)
        if not m:
            continue
        df = pd.read_parquet(p)
        df["seed"] = int(m.group(1))
        frames.append(df)
    if not frames:
        raise FileNotFoundError(
            f"No s*_final/{name}.parquet under {root}. Run "
            f"scripts/profiles/gciql_profile.py for each seed first.")
    return pd.concat(frames, ignore_index=True)


def _ms(s: pd.Series) -> str:
    return f"{s.mean():.3g} ± {s.std(ddof=0):.2g}"


def aggregate(root, out, fb_aggregate=None, fb_seed_root=None,
              method_label="GCIQL") -> None:
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    vl = _load(root, "value_landscape")
    cv = _load(root, "coverage")
    fun = _load(root, "phase_funnel")

    t1 = (vl.groupby(["task", "outcome"])["rho_V_negd"]
          .agg(["mean", "std"]).reset_index())
    t1.to_parquet(out / "T1_value_gradient.parquet")
    t4 = (cv.groupby(["region", "outcome"])["nn_dist"]
          .agg(["mean", "std"]).reset_index())
    t4.to_parquet(out / "T4_coverage.parquet")
    fun.to_parquet(out / "funnel.parquet")

    rho_s = float(vl[vl.outcome == "success"]["rho_V_negd"].mean())
    rho_f = float(vl[vl.outcome == "transport_fail"]["rho_V_negd"].mean())
    cvt = cv[cv.region == "transport"]
    fail_nn = float(cvt[cvt.outcome == "transport_fail"]["nn_dist"].mean())
    succ_nn = float(cvt[cvt.outcome == "success"]["nn_dist"].mean())
    verdicts = {"T1": _verdict_t1(rho_s, rho_f),
                "T4": _verdict_t4(fail_nn, succ_nn)}

    fig, ax = plt.subplots(figsize=(6, 4))
    g = vl.groupby("outcome")["rho_V_negd"].mean()
    ax.bar([str(i) for i in g.index], g.values)
    ax.set_title(f"{method_label} value gradient")
    ax.set_ylabel("Spearman rho(V, -d)")
    fig.tight_layout(); fig.savefig(out / "value_rho_bars.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    cc = cv.groupby(["region", "outcome"])["nn_dist"].mean().unstack()
    cc.plot(kind="bar", ax=ax)
    ax.set_title(f"{method_label} coverage"); ax.set_ylabel("nn_dist")
    fig.tight_layout(); fig.savefig(out / "coverage_curves.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    fc = fun.groupby("furthest_phase").size()
    ax.bar([str(i) for i in fc.index], fc.values)
    ax.set_title(f"{method_label} phase funnel")
    fig.tight_layout(); fig.savefig(out / "funnel.png", dpi=120)
    plt.close(fig)

    lines = [
        f"# {method_label} cross-seed story", "",
        f"Seeds: {', '.join('s'+str(s) for s in sorted(vl['seed'].unique()))}",
        "", "## Value gradient (T1)",
        f"- Spearman ρ(V vs −d) success: "
        f"{_ms(vl[vl.outcome=='success']['rho_V_negd'])}",
        f"- Spearman ρ(V vs −d) transport-fail: "
        f"{_ms(vl[vl.outcome=='transport_fail']['rho_V_negd'])}",
        "", "## Coverage (T4)",
        f"- transport nn_dist (transport_fail): "
        f"{_ms(cvt[cvt.outcome=='transport_fail']['nn_dist'])}",
        "", "## Readout",
        f"- T1 value gradient: rho_success={rho_s:.3g}, "
        f"rho_fail={rho_f:.3g} -> {verdicts['T1'][0]} "
        f"({verdicts['T1'][1]})",
        f"- T4 coverage: fail={fail_nn:.3g}, success={succ_nn:.3g} "
        f"-> {verdicts['T4'][0]} ({verdicts['T4'][1]})",
        _synthesis(verdicts),
    ]
    (out / "story.md").write_text("\n".join(lines))

    if fb_aggregate is not None:
        _comparison(out / "comparison", vl, cvt, fb_aggregate,
                    rho_s, rho_f, fail_nn, succ_nn, verdicts,
                    gciql_root=root, fb_seed_root=fb_seed_root,
                    method_label=method_label)
    print(f"[gciql_profile_aggregate] {vl['seed'].nunique()} seeds -> {out}")


def _zscore_v(df):
    """Per-call standardised V column 'Vz' = (V-mean)/std (ddof=0);
    constant V -> all zeros. Makes FB (V~2e3) and GCIQL (V~-60)
    value curves comparable on one axis without changing curve shape."""
    out = df.copy()
    v = out["V"].to_numpy(dtype=float)
    sd = v.std()
    out["Vz"] = (v - v.mean()) / sd if sd > 1e-12 else 0.0
    return out


def _bin_vz_grid(df, xmin, xmax, ymin, ymax, nbins):
    """2D mean of df['Vz'] over an (nbins x nbins) cube-(x,y) grid;
    cells with no samples are NaN. Returns (grid[ny,nx], x_edges,
    y_edges). grid[j,i] = mean Vz for x-bin i, y-bin j."""
    xe = np.linspace(xmin, xmax, nbins + 1)
    ye = np.linspace(ymin, ymax, nbins + 1)
    gx = np.clip(np.digitize(df["cube_x"], xe) - 1, 0, nbins - 1)
    gy = np.clip(np.digitize(df["cube_y"], ye) - 1, 0, nbins - 1)
    grid = np.full((nbins, nbins), np.nan)
    tmp = pd.DataFrame({"gx": gx, "gy": gy, "Vz": df["Vz"].to_numpy()})
    for (i, j), s in tmp.groupby(["gx", "gy"])["Vz"]:
        grid[j, i] = float(s.mean())
    return grid, xe, ye


def _phase_action_fields(df, grid, n_arrow: int = 12, n_min: int = 5):
    """Pure: (Vz_grid, (U,V), counts) for one panel's rows.

    `df` has columns episode,t,cube_x,cube_y,Vz (already filtered to one
    task/method/outcome/region). `grid` = (x_edges, y_edges). Vz_grid is
    the per-cell mean of Vz on the fine grid (NaN where empty). (U,V) is
    a coarse `n_arrow x n_arrow` mean of per-step cube displacement
    (NaN where < n_min samples). counts is the coarse sample count.
    """
    xe, ye = grid
    nx, ny = len(xe) - 1, len(ye) - 1
    vz = np.full((ny, nx), np.nan)
    if not df.empty:
        gx = np.clip(np.digitize(df["cube_x"], xe) - 1, 0, nx - 1)
        gy = np.clip(np.digitize(df["cube_y"], ye) - 1, 0, ny - 1)
        tmp = pd.DataFrame({"gx": gx, "gy": gy,
                            "Vz": df["Vz"].to_numpy()})
        for (i, j), s in tmp.groupby(["gx", "gy"])["Vz"]:
            vz[j, i] = float(s.mean())

    axe = np.linspace(xe[0], xe[-1], n_arrow + 1)
    aye = np.linspace(ye[0], ye[-1], n_arrow + 1)
    U = np.full((n_arrow, n_arrow), np.nan)
    Vv = np.full((n_arrow, n_arrow), np.nan)
    counts = np.zeros((n_arrow, n_arrow), int)
    if not df.empty:
        d = df.sort_values(["episode", "t"])
        dx = d.groupby("episode")["cube_x"].diff().shift(-1)
        dy = d.groupby("episode")["cube_y"].diff().shift(-1)
        ok = dx.notna() & dy.notna()
        cx, cy = d["cube_x"][ok], d["cube_y"][ok]
        ai = np.clip(np.digitize(cx, axe) - 1, 0, n_arrow - 1)
        aj = np.clip(np.digitize(cy, aye) - 1, 0, n_arrow - 1)
        agg = pd.DataFrame({"ai": ai, "aj": aj,
                            "dx": dx[ok].to_numpy(),
                            "dy": dy[ok].to_numpy()})
        for (i, j), s in agg.groupby(["ai", "aj"]):
            counts[j, i] = len(s)
            if len(s) >= n_min:
                U[j, i] = float(s["dx"].mean())
                Vv[j, i] = float(s["dy"].mean())
    return vz, (U, Vv), counts


def _cube_distribution_scatter(out_dir, gciql_root, fb_seed_root,
                               buffer_dir=None,
                               n_episodes_dataset: int = 500) -> None:
    """One scatter PNG comparing (state, goal) coverage between
    training and evaluation. Training (offline play data): per-episode
    cube xy at start (light blue) and end (dark blue) — these are the
    (state, goal-reached) pairs the model saw. Evaluation: per-episode
    cube xy at start (light orange) and the explicit task goal we hand
    the policy (dark orange) — what we ASK the policy to do. We want
    to see whether the eval (start, goal) pairs sit inside the
    training (start, end) distribution."""
    import json
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    buf = Path(buffer_dir or
               REPO_ROOT / "datasets" / "cube-single-play-v0" / "buffer")

    # Dataset: first / last cube xy per episode (physics[:, 14:16])
    ds_s, ds_e = [], []
    for f in sorted(buf.glob("episode_*.npz"))[:n_episodes_dataset]:
        with np.load(f) as z:
            p = np.asarray(z["physics"], np.float64)
            ds_s.append(p[0, 14:16])
            ds_e.append(p[-1, 14:16])
    if not ds_s:
        print("  [warn] _cube_distribution_scatter: no buffer episodes")
        return
    ds_s = np.asarray(ds_s, float)
    ds_e = np.asarray(ds_e, float)

    def _eval_starts(root):
        """Pooled cube xy at t=0 across all rollouts (methods, seeds,
        tasks, episodes). Skips old-schema parquet."""
        out = []
        if root is None:
            return np.zeros((0, 2))
        need = {"task", "episode", "t", "cube_x", "cube_y"}
        for p in sorted(Path(root).glob("s*_final/value_steps.parquet")):
            df = pd.read_parquet(p)
            if not need <= set(df.columns):
                continue
            df = df.sort_values(["task", "episode", "t"])
            for _, g in df.groupby(["task", "episode"]):
                out.append(g.iloc[0][["cube_x", "cube_y"]].to_numpy())
        return np.asarray(out, float) if out else np.zeros((0, 2))

    eval_starts = np.vstack([_eval_starts(gciql_root),
                              _eval_starts(fb_seed_root)])
    # Eval goals from env-extracted anchors (5 task targets).
    sx = REPO_ROOT / "analysis" / "misc" / "scene"
    anchors_json = sx / "task_anchors.json"
    eval_goals = []
    if anchors_json.exists():
        anc = json.loads(anchors_json.read_text())
        eval_goals = [v["goal"] for v in anc.values() if "goal" in v]
    eval_goals = (np.asarray(eval_goals, float)
                  if eval_goals else np.zeros((0, 2)))

    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    # training (state, end) pairs from offline data
    ax.scatter(ds_s[:, 0], ds_s[:, 1], s=12, c="#9ecae1", alpha=0.55,
               edgecolors="none",
               label=f"training: cube start (n={len(ds_s)})")
    ax.scatter(ds_e[:, 0], ds_e[:, 1], s=12, c="#08306b", alpha=0.65,
               edgecolors="none",
               label=f"training: cube end (n={len(ds_e)})")
    # evaluation (state, goal) pairs
    if len(eval_starts):
        ax.scatter(eval_starts[:, 0], eval_starts[:, 1], s=22,
                   c="#fdae6b", alpha=0.85, edgecolors="black",
                   linewidths=0.4,
                   label=f"eval: cube start (n={len(eval_starts)})")
    if len(eval_goals):
        ax.scatter(eval_goals[:, 0], eval_goals[:, 1], s=180,
                   c="#a63603", alpha=0.95, edgecolors="black",
                   linewidths=1.2, marker="X",
                   label=f"eval: task goal (n={len(eval_goals)})")
    ax.set_aspect("equal")
    ax.set_xlabel("cube x (m)")
    ax.set_ylabel("cube y (m)")
    ax.set_title("(state, goal) coverage: training data vs evaluation\n"
                 "do eval (start, goal) pairs sit inside the training "
                 "(start, end) distribution?",
                 fontsize=11)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9,
              markerscale=1.3)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fp = out_dir / "cube_distribution_scatter.png"
    fig.savefig(fp, dpi=140)
    plt.close(fig)
    print(f"[cube_distribution_scatter] wrote -> {fp}")


def _decode_support_xy(gciql_root, fb_seed_root):
    """Decode dataset cube xy once, cached to
    analysis/misc/dataset_support_cube_xy.npy."""
    from evals.dataset_support import dataset_cube_xy

    cache = REPO_ROOT / "analysis" / "misc" / "dataset_support_cube_xy.npy"
    if cache.exists():
        return np.load(cache)
    ref = None
    for rt in (gciql_root, fb_seed_root):
        if rt is None:
            continue
        fr = [pd.read_parquet(p) for p in
              sorted(Path(rt).glob("s*_final/value_steps.parquet"))]
        if fr:
            cat = pd.concat(fr, ignore_index=True)
            ref = cat[["cube_x", "cube_y"]].to_numpy()
            break
    if ref is None:
        raise ValueError("no value_steps to derive ref cube xy")
    buf = REPO_ROOT / "datasets" / "cube-single-play-v0" / "buffer"
    xy = dataset_cube_xy(buf, ref_cube_xy=ref)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, xy)
    return xy


def _nan_gaussian(grid, sigma: float = 1.0):
    """NaN-aware Gaussian smooth: empty cells stay NaN, occupied cells
    are blurred using only finite neighbours (renders a continuous
    field instead of confetti)."""
    from scipy.ndimage import gaussian_filter

    g = np.asarray(grid, float)
    m = np.isfinite(g)
    if not m.any():
        return np.full_like(g, np.nan)
    filled = np.where(m, g, 0.0)
    num = gaussian_filter(filled, sigma, mode="nearest")
    den = gaussian_filter(m.astype(float), sigma, mode="nearest")
    # paint only where enough finite mass diffused in (keeps the field
    # near the data, not smeared across empty workspace)
    return np.where(den > 1e-3, num / np.maximum(den, 1e-9), np.nan)


REGIONS = [("reach", "approach"), ("lift", "contact"),
           ("transport", "transport")]


def _canon_task(s: str) -> str:
    """Canonical 'taskN' for either recorder's task label.

    GCIQL writes 'task1'..'task5'; FB writes the full OGBench env id
    'cube-single-play-singletask-task{N}-v0'. Normalize so FB and
    GCIQL pair per task.
    """
    m = re.search(r"task(\d+)", str(s))
    return f"task{m.group(1)}" if m else str(s)


def _phase_action_scene(out_dir, gciql_root, fb_seed_root,
                        tasks=None, method_label="GCIQL") -> None:
    """10 figures (task x method); rows {success,transport_fail}, cols
    {approach,contact,transport}. Pure data + matplotlib; no model
    loading. Skips a method with no value_steps (warns)."""
    import json

    import matplotlib.image as mpimg

    from evals.dataset_support import support_kde

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sx = REPO_ROOT / "analysis" / "misc" / "scene"
    scene_png, calib_json = sx / "topdown.png", sx / "calib.json"
    bg_img = None
    if scene_png.exists() and calib_json.exists():
        cal = json.loads(calib_json.read_text())
        if cal.get("photoreal"):
            bg_img = mpimg.imread(str(scene_png))
    anchors_json = sx / "task_anchors.json"
    anchors = (json.loads(anchors_json.read_text())
               if anchors_json.exists() else {})

    def _seed_concat(rt):
        if rt is None:
            return None
        fr = [pd.read_parquet(p) for p in
              sorted(Path(rt).glob("s*_final/value_steps.parquet"))]
        return pd.concat(fr, ignore_index=True) if fr else None

    src = {"FB": _seed_concat(fb_seed_root),
           method_label: _seed_concat(gciql_root)}
    src = {k: v for k, v in src.items()
           if v is not None and "region" in v.columns}
    if not src:
        print("  [warn] _phase_action_scene: no value_steps w/ region")
        return
    for v in src.values():
        v["task"] = v["task"].map(_canon_task)
    tasks = tasks or sorted(
        set().union(*[set(v["task"].unique()) for v in src.values()]))

    support_xy = _decode_support_xy(gciql_root, fb_seed_root)
    NBINS, NARROW, LOW_N = 16, 10, 30
    written = []
    for task in tasks:
        for method, vs in src.items():
            sub_all = vs[vs.task == task]
            if sub_all.empty:
                continue
            # shared world extent for all 6 panels (data-driven, robust)
            xs, ys = sub_all["cube_x"], sub_all["cube_y"]
            xlo, xhi = np.percentile(xs, [1, 99])
            ylo, yhi = np.percentile(ys, [1, 99])
            px, py = 0.06 * (xhi - xlo), 0.06 * (yhi - ylo)
            xlo, xhi, ylo, yhi = xlo - px, xhi + px, ylo - py, yhi + py
            # Square-pad so every panel is visually square (and the
            # 4th column has the SAME axes/aspect as the phase panels).
            xr, yr = xhi - xlo, yhi - ylo
            if xr < yr:
                pad = 0.5 * (yr - xr)
                xlo, xhi = xlo - pad, xhi + pad
            elif yr < xr:
                pad = 0.5 * (xr - yr)
                ylo, yhi = ylo - pad, yhi + pad
            xe = np.linspace(xlo, xhi, NBINS + 1)
            ye = np.linspace(ylo, yhi, NBINS + 1)
            ext = [xlo, xhi, ylo, yhi]
            # Per-task anchors from the real env (authoritative). Fall
            # back to data-derived values if the cache is missing.
            anc = anchors.get(task, {})
            if "cube_init" in anc:
                cube_init_xy = tuple(anc["cube_init"])
            else:
                init = sub_all[sub_all["t"] == 0]
                cube_init_xy = ((float(init["cube_x"].mean()),
                                 float(init["cube_y"].mean()))
                                if not init.empty else None)
            if "goal" in anc:
                goal_xy = tuple(anc["goal"])
            else:
                succ = sub_all[sub_all.outcome == "success"]
                if not succ.empty:
                    thr = np.percentile(succ["d"], 5)
                    g = succ[succ["d"] <= thr]
                    goal_xy = (float(g["cube_x"].mean()),
                               float(g["cube_y"].mean()))
                else:
                    goal_xy = None
            eef_init_xy = (tuple(anc["eef_init"])
                           if "eef_init" in anc else None)
            kx = 0.5 * (xe[:-1] + xe[1:])
            ky = 0.5 * (ye[:-1] + ye[1:])
            kz = support_kde(support_xy, kx, ky)

            # 2 rows x 4 cols: cols 0-2 are phases; col 3 spans both
            # rows as a dedicated "offline dataset support" panel.
            fig = plt.figure(figsize=(20, 9.5))
            gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1],
                                  wspace=0.06, hspace=0.10)
            axes = np.empty((2, 3), dtype=object)
            for rr in range(2):
                for cc in range(3):
                    axes[rr, cc] = fig.add_subplot(gs[rr, cc])
            # Single cell for the data-support panel (same size as a
            # phase panel). gs[1, 3] is intentionally left blank.
            ax_data = fig.add_subplot(gs[0, 3])
            for r, oc in enumerate(("success", "transport_fail")):
                for c, (rg, lbl) in enumerate(REGIONS):
                    ax = axes[r, c]
                    d = sub_all[(sub_all.outcome == oc)
                                & (sub_all.region == rg)]
                    n = len(d)
                    low = n < LOW_N
                    ax.set_xlim(xlo, xhi)
                    ax.set_ylim(ylo, yhi)
                    ax.set_aspect("equal")
                    ax.set_xticks([])
                    ax.set_yticks([])
                    if bg_img is not None:
                        ax.imshow(bg_img, extent=[xlo, xhi, ylo, yhi],
                                  aspect="auto", zorder=0,
                                  interpolation="bilinear")
                    else:
                        ax.set_facecolor("0.92")
                    tag = f"{oc} — {lbl} (n={n}{', low' if low else ''})"
                    ax.set_title(tag, fontsize=9,
                                 color="grey" if low else "black")
                    # per-phase mean cube + EEF positions (this filter)
                    if n >= 5:
                        cube_p = (float(d["cube_x"].mean()),
                                  float(d["cube_y"].mean()))
                        eef_p = (float(d["eef_x"].mean()),
                                 float(d["eef_y"].mean()))
                        ax.scatter([eef_p[0]], [eef_p[1]], marker="D",
                                   s=160, c="cyan", edgecolors="black",
                                   linewidths=1.4, zorder=5)
                        ax.scatter([cube_p[0]], [cube_p[1]], marker="s",
                                   s=170, c="white", edgecolors="black",
                                   linewidths=1.4, zorder=6)
                    if goal_xy is not None:
                        # goal: bright gold star with halo + black edge
                        ax.scatter([goal_xy[0]], [goal_xy[1]],
                                   marker="*", s=900, c="white",
                                   alpha=0.55, zorder=6)
                        ax.scatter([goal_xy[0]], [goal_xy[1]],
                                   marker="*", s=520, c="gold",
                                   edgecolors="black", linewidths=1.6,
                                   zorder=7, label="goal")
                    if n < 5:
                        continue
                    a = 0.35 if low else 1.0
                    vz, (U, Vv), cnt = _phase_action_fields(
                        d.assign(Vz=d["V"]), (xe, ye), n_arrow=NARROW,
                        n_min=5)
                    vz = _nan_gaussian(vz, sigma=1.0)
                    fin = np.isfinite(vz)
                    if fin.any():
                        lo, hi = np.percentile(vz[fin], [2, 98])
                        if hi - lo < 1e-9:
                            hi = lo + 1.0
                        # red-saliency overlay: alpha proportional to
                        # normalized value (low V transparent -> table
                        # shows; high V bright red). RGB from 'Reds'.
                        from matplotlib.cm import get_cmap
                        norm = np.clip((vz - lo) / (hi - lo), 0.0, 1.0)
                        rgba = get_cmap("Reds")(norm)
                        # Lighter "saliency" overlay: linear-in-value
                        # alpha (no aggressive low-V fade) capped at
                        # 0.55 so the photoreal table reads through.
                        rgba[..., 3] = np.where(
                            fin, norm * (0.55 * a), 0.0)
                        ax.imshow(rgba, origin="lower", extent=ext,
                                  zorder=2, aspect="auto",
                                  interpolation="bilinear")
                    # unit-normalized arrows, colored by displacement
                    ac = 0.5 * (np.linspace(xlo, xhi, NARROW + 1)[:-1]
                                + np.linspace(xlo, xhi,
                                               NARROW + 1)[1:])
                    ar = 0.5 * (np.linspace(ylo, yhi, NARROW + 1)[:-1]
                                + np.linspace(ylo, yhi,
                                               NARROW + 1)[1:])
                    AX, AY = np.meshgrid(ac, ar)
                    mag = np.hypot(U, Vv)
                    ok = np.isfinite(mag) & (mag > 0)
                    if ok.any():
                        un = np.where(ok, U / np.maximum(mag, 1e-9),
                                      np.nan)
                        vn = np.where(ok, Vv / np.maximum(mag, 1e-9),
                                      np.nan)
                        ax.quiver(AX[ok], AY[ok], un[ok], vn[ok],
                                  color="white", edgecolor="black",
                                  linewidth=0.5, alpha=a,
                                  scale=NARROW * 1.1, width=0.007,
                                  headwidth=4, zorder=4, pivot="mid")
            # ---- 4th column: dataset-support reference panel -----
            # Use the SAME extent as the phase panels so axes are
            # directly comparable (no transposed/portrait artifact).
            ax_data.set_xlim(xlo, xhi)
            ax_data.set_ylim(ylo, yhi)
            ax_data.set_aspect("equal")
            ax_data.set_xticks([])
            ax_data.set_yticks([])
            if bg_img is not None:
                ax_data.imshow(bg_img,
                               extent=[xlo, xhi, ylo, yhi],
                               aspect="auto", zorder=0,
                               interpolation="bilinear")
            else:
                ax_data.set_facecolor("0.92")
            # raw dataset cube xy scatter clipped to the phase extent
            in_ext = ((support_xy[:, 0] >= xlo)
                      & (support_xy[:, 0] <= xhi)
                      & (support_xy[:, 1] >= ylo)
                      & (support_xy[:, 1] <= yhi))
            sup_clip = support_xy[in_ext]
            n_sub = min(8000, len(sup_clip))
            if n_sub:
                idx = np.random.default_rng(0).choice(
                    len(sup_clip), n_sub, replace=False)
                ax_data.scatter(sup_clip[idx, 0], sup_clip[idx, 1],
                                s=2.0, c="#ffd86e", alpha=0.18,
                                zorder=1, linewidths=0)
            # filled KDE region + iso-lines on the phase-panel grid
            ax_data.contourf(kx, ky, kz,
                             levels=[0.10, 0.30, 0.60, 1.01],
                             colors=["#f0c0a0", "#e89060", "#d05050"],
                             alpha=0.45, zorder=2)
            ax_data.contour(kx, ky, kz, levels=[0.15, 0.6],
                            colors="white", linewidths=1.2,
                            alpha=0.9, zorder=3)
            # task reference markers (env anchors)
            anc_here = anchors.get(task, {})
            if "eef_init" in anc_here:
                eii = anc_here["eef_init"]
                ax_data.scatter([eii[0]], [eii[1]], marker="D", s=160,
                                c="cyan", edgecolors="black",
                                linewidths=1.4, zorder=5)
            if "cube_init" in anc_here:
                cii = anc_here["cube_init"]
                ax_data.scatter([cii[0]], [cii[1]], marker="s", s=170,
                                c="white", edgecolors="black",
                                linewidths=1.4, zorder=6)
            if "goal" in anc_here:
                gi = anc_here["goal"]
                ax_data.scatter([gi[0]], [gi[1]], marker="*", s=900,
                                c="white", alpha=0.55, zorder=6)
                ax_data.scatter([gi[0]], [gi[1]], marker="*", s=520,
                                c="gold", edgecolors="black",
                                linewidths=1.6, zorder=7)
            ax_data.set_title("offline dataset support\n"
                              "(cube xy in the play data)", fontsize=9)

            from matplotlib.lines import Line2D
            from matplotlib.patches import Patch
            handles = []
            handles.append(Patch(
                facecolor="#de2d26", edgecolor="black",
                linewidth=0.4, label="value V(s): low → high (per-panel)"))
            handles.append(Line2D(
                [0], [0], color="white", marker=r"$\rightarrow$",
                markeredgecolor="black", markersize=12, linestyle="",
                label="cube-flow direction (rollouts)"))
            handles.append(Line2D(
                [0], [0], marker="D", color="cyan",
                markeredgecolor="black", markersize=8,
                linestyle="", label="EEF (mean this phase)"))
            handles.append(Line2D(
                [0], [0], marker="s", color="white",
                markeredgecolor="black", markersize=10,
                linestyle="", label="cube (mean this phase)"))
            if goal_xy is not None:
                handles.append(Line2D(
                    [0], [0], marker="*", color="gold",
                    markeredgecolor="black", markersize=15,
                    linestyle="", label="goal (task target)"))
            handles.append(Line2D(
                [0], [0], color="0.55", linewidth=1.0,
                label="offline data support"))
            axes[0, 0].legend(handles=handles, loc="upper left",
                              fontsize=7, framealpha=0.85, ncol=1)
            fig.suptitle(
                f"{task} — {method}  "
                "red = value V(s) (per-panel saliency, low→high)    ·    "
                "white arrows = cube-flow direction (rollouts)", fontsize=12)
            fig.tight_layout()
            fp = out_dir / f"{task}__{method}.png"
            fig.savefig(fp, dpi=125)
            plt.close(fig)
            written.append(fp.name)
    (out_dir / "_index.md").write_text(
        "# phase_scene\n\n"
        + "\n".join(f"- {n}" for n in sorted(written)) + "\n")
    print(f"[phase_action_scene] wrote {len(written)} figs -> "
          f"{out_dir}")


def _medoid_episode(df) -> int:
    """Episode id whose per-step (cube_x,cube_y) path is closest (mean
    L2) to the across-episode mean path (truncated to the shortest)."""
    eps = sorted(df["episode"].unique())
    paths = {e: df[df.episode == e][["cube_x", "cube_y"]].to_numpy()
             for e in eps}
    if not paths:
        return -1
    L = min(len(p) for p in paths.values())
    if L == 0:
        return eps[0]
    stack = np.stack([paths[e][:L] for e in eps], axis=0)  # [E,L,2]
    mean_path = stack.mean(axis=0)                          # [L,2]
    dists = [float(np.linalg.norm(stack[k] - mean_path, axis=1).mean())
             for k in range(len(eps))]
    return int(eps[int(np.argmin(dists))])


def _value_scene(cdir, gciql_root, fb_seed_root, scene_png, calib,
                 method_label="GCIQL"):
    """Four {FB,GCIQL}x{success,transport_fail} translucent V heatmaps
    over the shared top-down scene + medoid trajectory + a 2x2 montage.
    Skipped (with a warning) if inputs are missing."""
    import json

    import matplotlib.image as mpimg
    from evals.scene_calib import world_to_px

    if not (Path(scene_png).exists() and Path(calib).exists()):
        print("  [warn] _value_scene skipped (no scene/calib)")
        return
    cal = json.loads(Path(calib).read_text())
    ws = cal["workspace"]
    bg = mpimg.imread(str(scene_png))

    def _seed_concat(rt, name):
        if rt is None:
            return None
        fr = [pd.read_parquet(p)
              for p in sorted(Path(rt).glob(f"s*_final/{name}.parquet"))]
        return pd.concat(fr, ignore_index=True) if fr else None

    sources = {"FB": _seed_concat(fb_seed_root, "value_steps"),
               method_label: _seed_concat(gciql_root, "value_steps")}
    panels = []
    for method, vs in sources.items():
        if vs is None or "cube_x" not in vs.columns:
            print(f"  [warn] _value_scene: no value_steps for {method}")
            continue
        vsz = _zscore_v(vs)
        for oc in ("success", "transport_fail"):
            sub = vsz[(vsz.outcome == oc) & (vsz.transport)]
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(bg, extent=[0, cal["img_w"], cal["img_h"], 0])
            ax.set_xlim(0, cal["img_w"]); ax.set_ylim(cal["img_h"], 0)
            ax.axis("off")
            title = f"{method} — {oc}"
            if not sub.empty:
                grid, xe, ye = _bin_vz_grid(
                    sub, ws["xmin"], ws["xmax"], ws["ymin"], ws["ymax"],
                    nbins=40)
                p00 = world_to_px(ws["xmin"], ws["ymin"], cal)
                p11 = world_to_px(ws["xmax"], ws["ymax"], cal)
                ax.imshow(np.ma.masked_invalid(grid), origin="lower",
                          extent=[p00[0], p11[0], p00[1], p11[1]],
                          cmap="viridis", alpha=0.55, aspect="auto")
                med = _medoid_episode(sub)
                mp = sub[sub.episode == med]
                pts = [world_to_px(x, y, cal) for x, y in
                       zip(mp["cube_x"], mp["cube_y"])]
                if pts:
                    xs, ys = zip(*pts)
                    ax.plot(xs, ys, "-", color="red", lw=2.0,
                            label="medoid path")
                    ax.scatter([xs[0]], [ys[0]], c="white",
                               edgecolors="k", s=40, zorder=5)
                ax.legend(loc="lower right", fontsize=7)
            else:
                title += " (no transport steps)"
            ax.set_title(title, fontsize=9)
            fig.tight_layout()
            fp = cdir / f"cmp_scene_{method}_{oc}.png"
            fig.savefig(fp, dpi=120)
            plt.close(fig)
            panels.append(fp)

    if panels:
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        for axx, fp in zip(axes.ravel(), panels):
            axx.imshow(mpimg.imread(str(fp)))
            axx.axis("off")
        for axx in list(axes.ravel())[len(panels):]:
            axx.axis("off")
        fig.tight_layout()
        fig.savefig(cdir / "cmp_value_scene.png", dpi=120)
        plt.close(fig)


def _comparison(cdir, vl, cvt, fb_aggregate, g_rs, g_rf, g_fnn, g_snn,
                g_verd, gciql_root=None, fb_seed_root=None,
                method_label="GCIQL") -> None:
    cdir.mkdir(parents=True, exist_ok=True)
    gl = method_label
    fb_t1 = pd.read_parquet(
        Path(fb_aggregate) / "T1_value_gradient.parquet")

    def _fb_rho(oc):
        sub = fb_t1[fb_t1.outcome == oc]
        col = "mean" if "mean" in sub.columns else "rho_V_negd"
        return float(sub[col].mean()) if len(sub) else float("nan")

    fb_rs, fb_rf = _fb_rho("success"), _fb_rho("transport_fail")
    fbv = _verdict_t1(fb_rs, fb_rf)
    rows = [
        {"method": "FB", "rho_success": fb_rs, "rho_fail": fb_rf,
         "T1_verdict": fbv[0]},
        {"method": gl, "rho_success": g_rs, "rho_fail": g_rf,
         "T1_verdict": g_verd["T1"][0]},
    ]
    cmp = pd.DataFrame(rows)
    cmp.to_parquet(cdir / "fb_vs_gciql.parquet")

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(2)
    ax.bar(x - 0.2, [fb_rs, g_rs], 0.4, label="success")
    ax.bar(x + 0.2, [fb_rf, g_rf], 0.4, label="transport_fail")
    ax.set_xticks(x); ax.set_xticklabels(["FB", gl])
    ax.set_ylabel("Spearman rho(V, -d)"); ax.legend()
    ax.set_title(f"FB vs {gl}: value gradient")
    fig.tight_layout(); fig.savefig(cdir / "cmp_value_rho.png", dpi=120)
    plt.close(fig)

    def _seed_concat(rt, name):
        if rt is None:
            return None
        fr = []
        for p in sorted(Path(rt).glob(f"s*_final/{name}.parquet")):
            fr.append(pd.read_parquet(p))
        return pd.concat(fr, ignore_index=True) if fr else None

    g_vs = _seed_concat(gciql_root, "value_steps")
    f_vs = _seed_concat(fb_seed_root, "value_steps")
    g_vl = _seed_concat(gciql_root, "value_landscape")
    f_vl = _seed_concat(fb_seed_root, "value_landscape")
    g_cv = _seed_concat(gciql_root, "coverage")
    f_cv = _seed_concat(fb_seed_root, "coverage")

    # 1. value vs cube->goal distance (FB & GCIQL x success/fail)
    if g_vs is not None and f_vs is not None:
        pool = np.concatenate([g_vs["d"].to_numpy(),
                               f_vs["d"].to_numpy()])
        edges = np.quantile(pool, np.linspace(0, 1, 10))
        edges = np.unique(edges)
        ctr = 0.5 * (edges[:-1] + edges[1:])
        fig, ax = plt.subplots(figsize=(6, 4))
        for nm, df in (("FB", _zscore_v(f_vs)), (gl, _zscore_v(g_vs))):
            for oc, ls in (("success", "-"),
                           ("transport_fail", "--")):
                sub = df[df.outcome == oc]
                if sub.empty:
                    continue
                idx = np.clip(np.digitize(sub["d"], edges) - 1, 0,
                              len(ctr) - 1)
                mv = (pd.DataFrame({"b": idx, "V": sub["Vz"].to_numpy()})
                      .groupby("b")["V"].mean())
                ax.plot(ctr[mv.index], mv.values, ls, marker="o",
                        label=f"{nm} {oc}")
        ax.set_xlabel("cube -> goal distance")
        ax.set_ylabel("standardised V (per method, z-scored)")
        ax.set_title(f"Value vs goal distance: FB vs {gl}")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(cdir / "cmp_value_vs_dist.png", dpi=120)
        plt.close(fig)
    else:
        print("  [warn] cmp_value_vs_dist skipped (missing value_steps)")

    # 2. per-episode rho distribution (box) for the four groups
    if g_vl is not None and f_vl is not None:
        groups, labels = [], []
        for nm, df in (("FB", f_vl), (gl, g_vl)):
            for oc in ("success", "transport_fail"):
                v = df[df.outcome == oc]["rho_V_negd"].dropna()
                groups.append(v.to_numpy() if len(v) else np.array([0.0]))
                labels.append(f"{nm}\n{oc}")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.boxplot(groups, labels=labels, showmeans=True)
        ax.set_ylabel("Spearman rho(V, -d)")
        ax.set_title(f"Per-episode value gradient: FB vs {gl}")
        ax.tick_params(axis="x", labelsize=7)
        fig.tight_layout()
        fig.savefig(cdir / "cmp_rho_box.png", dpi=120)
        plt.close(fig)

    # 3. outcome funnel (normalized fraction) FB vs GCIQL
    if g_vl is not None and f_vl is not None:
        order = ["success", "transport_fail", "other"]
        fig, ax = plt.subplots(figsize=(6, 4))
        x = np.arange(len(order))
        for i, (nm, df) in enumerate((("FB", f_vl), (gl, g_vl))):
            fr = (df["outcome"].value_counts(normalize=True)
                  .reindex(order).fillna(0.0))
            ax.bar(x + (i - 0.5) * 0.4, fr.values, 0.4, label=nm)
        ax.set_xticks(x); ax.set_xticklabels(order)
        ax.set_ylabel("episode fraction")
        ax.set_title(f"Outcome funnel: FB vs {gl}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(cdir / "cmp_funnel.png", dpi=120)
        plt.close(fig)

    # 4. coverage nn_dist by region FB vs GCIQL
    if g_cv is not None and f_cv is not None:
        regs = sorted(set(g_cv["region"]) | set(f_cv["region"]))
        fig, ax = plt.subplots(figsize=(6, 4))
        x = np.arange(len(regs))
        for i, (nm, df) in enumerate((("FB", f_cv), (gl, g_cv))):
            m = df.groupby("region")["nn_dist"].mean()
            ax.bar(x + (i - 0.5) * 0.4,
                   [float(m.get(r, 0.0)) for r in regs], 0.4, label=nm)
        ax.set_xticks(x); ax.set_xticklabels(regs)
        ax.set_ylabel("nn_dist"); ax.legend()
        ax.set_title(f"Off-support coverage: FB vs {gl}")
        fig.tight_layout()
        fig.savefig(cdir / "cmp_coverage.png", dpi=120)
        plt.close(fig)

    (cdir / "comparison.md").write_text("\n".join([
        f"# FB vs {gl} — shared-axis comparison", "",
        f"- FB    value gradient: rho_s={fb_rs:.3g}, rho_f={fb_rf:.3g} "
        f"-> {fbv[0]}",
        f"- {gl} value gradient: rho_s={g_rs:.3g}, rho_f={g_rf:.3g} "
        f"-> {g_verd['T1'][0]}",
        f"- {gl} coverage transport nn_dist: fail={g_fnn:.3g}, "
        f"success={g_snn:.3g} -> {g_verd['T4'][0]}",
        "",
        "Both methods are compared only on axes that transfer "
        "(value gradient, coverage, phase funnel); FB-specific "
        f"z-decoding (T2) and B-resolution (T3) have no {gl} analog.",
    ]))

    _value_scene(cdir, gciql_root, fb_seed_root,
                 REPO_ROOT / "analysis" / "misc" / "scene" / "topdown.png",
                 REPO_ROOT / "analysis" / "misc" / "scene" / "calib.json",
                 method_label=gl)

    _phase_action_scene(cdir / "phase_scene", gciql_root, fb_seed_root,
                        method_label=gl)
    _cube_distribution_scatter(cdir / "phase_scene",
                                gciql_root, fb_seed_root)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root",
                    default=str(REPO_ROOT / "analysis" / "legacy" / "gciql_profile"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--fb-aggregate",
                    default=str(REPO_ROOT / "analysis"
                               / "probes"
                               / "representation_profile" / "aggregate"))
    ap.add_argument("--fb-seed-root",
                    default=str(REPO_ROOT / "analysis"
                               / "probes"
                               / "representation_profile"))
    ap.add_argument("--method-label", default="GCIQL",
                    help="display name for the non-FB method (e.g. GCIVL)")
    args = ap.parse_args()
    root = Path(args.root)
    fb = Path(args.fb_aggregate)
    fbsr = Path(args.fb_seed_root)
    aggregate(root, Path(args.out) if args.out else root / "aggregate",
              fb_aggregate=fb if fb.exists() else None,
              fb_seed_root=fbsr if fbsr.exists() else None,
              method_label=args.method_label)


if __name__ == "__main__":
    main()

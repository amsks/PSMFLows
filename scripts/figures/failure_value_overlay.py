"""scripts/figures/failure_value_overlay.py — value-function overlays for two failure modes.

For both FB (state) and GCIQL (state), pick 1–2 example eval-rollout episodes per
failure mode (maintain-grasp = picked-up-then-dropped; transport = held-but-far)
and render the canonical scene overlay: value heatmap binned over cube-xy,
cube-flow arrows showing where the cube actually moved during the rollout, the
goal star, and a workspace top-down background.

Inputs (already produced by representation_profile / gciql_profile + phase_probe_v2):
  FB:    analysis/probes/representation_profile/sN_final/value_steps.parquet
         analysis/probes/phase_probe_v2/sN_final/per_episode.parquet  (final-step signals)
  GCIQL: analysis/profiles/gciql_profile_v2/sN_final/value_steps.parquet
         analysis/profiles/gciql_profile_v2/sN_final/phase_funnel.parquet

Output:
  analysis/value/failure_overlays/failure_value_overlays.png
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from scripts.value.traj_value_master import _bg_ext, _scene_overlay  # noqa: E402

DELTA_LIFT = 0.03
EPS_NEAR = 0.05

# fb task strings are full URLs; gciql uses short form. Map to short form for join.
def _short_task(s: str) -> str:
    if s.startswith("cube-single-play-singletask-"):
        return s.split("-")[-2]   # ...-taskN-v0 -> taskN
    return s


def _classify(row: pd.Series) -> str:
    """Bucket an eval episode by failure mode using final-step signals."""
    if bool(row.get("success", False)):
        return "success"
    fp = row["furthest_phase"] if "furthest_phase" in row else row["fail_phase"]
    if fp == "reach":
        return "approach"
    if fp == "grasp":
        return "grasp"
    # transport sub-split via final_cube_lift and final_cube_goal_dist
    if row["final_cube_lift"] < DELTA_LIFT:
        return "maintain"
    if row["final_cube_goal_dist"] <= EPS_NEAR:
        return "near"
    return "transport"


def _load_fb() -> pd.DataFrame:
    vs_paths = sorted(glob.glob("analysis/probes/representation_profile/s*_final/value_steps.parquet"))
    pe_paths = sorted(glob.glob("analysis/probes/phase_probe_v2/s*_final/per_episode.parquet"))
    pe_by_seed = {p.split("/")[-2]: pd.read_parquet(p) for p in pe_paths}
    rows = []
    for vp in vs_paths:
        seed = vp.split("/")[-2]
        if seed not in pe_by_seed:
            continue
        vs = pd.read_parquet(vp)
        vs["task_short"] = vs["task"].map(_short_task)
        pe = pe_by_seed[seed][pe_by_seed[seed]["scenario"] == "S0"].copy()
        pe["task_short"] = pe["task"].map(_short_task)
        pe["mode"] = pe.apply(_classify, axis=1)
        keep = pe[["task_short", "episode", "mode", "success", "final_cube_lift",
                   "final_cube_goal_dist"]].drop_duplicates()
        merged = vs.merge(keep, on=["task_short", "episode"], how="inner")
        merged["seed"] = seed.replace("_final", "")
        merged["method"] = "fb"
        rows.append(merged)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _load_gciql() -> pd.DataFrame:
    vs_paths = sorted(glob.glob("analysis/profiles/gciql_profile_v2/s*_final/value_steps.parquet"))
    rows = []
    for vp in vs_paths:
        seed = vp.split("/")[-2]
        pf = pd.read_parquet(Path(vp).parent / "phase_funnel.parquet")
        pf["mode"] = pf.apply(_classify, axis=1)
        pf["success"] = (pf["mode"] == "success")
        keep = pf[["task", "episode", "mode", "success", "final_cube_lift",
                   "final_cube_goal_dist"]]
        vs = pd.read_parquet(vp)
        merged = vs.merge(keep, on=["task", "episode"], how="inner")
        merged["task_short"] = merged["task"]
        merged["seed"] = seed.replace("_final", "")
        merged["method"] = "gciql"
        rows.append(merged)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _pick_examples(df: pd.DataFrame, mode: str, n: int = 2) -> list[tuple[str, str, int]]:
    """Pick n (seed, task_short, episode) tuples for the given failure mode.
    We prefer episodes with rich cube motion (large cube-xy span) so the arrows
    have something to show. Stable ordering across runs (sorted)."""
    sub = df[df["mode"] == mode]
    if sub.empty:
        return []
    span = (sub.groupby(["seed", "task_short", "episode"])
              .agg(span_x=("cube_x", lambda s: s.max() - s.min()),
                   span_y=("cube_y", lambda s: s.max() - s.min()),
                   n=("t", "count")))
    span["span"] = np.hypot(span["span_x"], span["span_y"])
    span = span[(span["n"] >= 30) & (span["span"] >= 0.05)]
    span = span.sort_values("span", ascending=False)
    # spread examples across distinct (seed, task) where possible
    seen, out = set(), []
    for (seed, task, ep), _ in span.iterrows():
        key = (seed, task)
        if key in seen:
            continue
        seen.add(key)
        out.append((seed, task, int(ep)))
        if len(out) >= n:
            break
    return out


def _norm_one(g: pd.DataFrame) -> pd.DataFrame:
    """Per-trajectory min-max normalize V into Vn for the scene heatmap.
    Rename episode -> traj because _scene_overlay groups by 'traj' and then
    internally renames traj -> episode for _phase_action_fields."""
    g = g.copy()
    lo, hi = g["V"].min(), g["V"].max()
    g["Vn"] = (g["V"] - lo) / (hi - lo + 1e-12)
    g = g.rename(columns={"episode": "traj"})
    return g


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="analysis/value/failure_overlays")
    ap.add_argument("--n-per-cell", type=int, default=2)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    fb = _load_fb(); gq = _load_gciql()
    if fb.empty or gq.empty:
        raise SystemExit("missing FB or GCIQL per-step data")

    bg, ext = _bg_ext()
    n = args.n_per_cell
    fig, axes = plt.subplots(2, 2 * n, figsize=(4.6 * n * 2, 9.2))
    if n == 1:
        axes = axes[:, None]   # keep 2-D

    rows = [("maintain", "Failure to maintain grasp  (dropped)"),
            ("transport", "Failure to transport  (held but >5 cm from goal)")]
    cols = [(fb, "FB", "tab:red"), (gq, "GCIQL", "tab:blue")]

    for ri, (mode, row_title) in enumerate(rows):
        for ci, (data, mlabel, _) in enumerate(cols):
            picks = _pick_examples(data, mode, n=n)
            for k in range(n):
                ax = axes[ri, ci * n + k]
                if k >= len(picks):
                    ax.set_title(f"{mlabel} · no example"); ax.set_xticks([]); ax.set_yticks([])
                    continue
                seed, task, ep = picks[k]
                g = data[(data["seed"] == seed) & (data["task_short"] == task)
                         & (data["episode"] == ep)].sort_values("t")
                g = _norm_one(g)
                ti = (f"{mlabel} · {seed} · {task} · ep{ep}\n"
                      f"final lift={g['final_cube_lift'].iloc[0]*100:.1f} cm  "
                      f"final d(cube,goal)={g['final_cube_goal_dist'].iloc[0]*100:.1f} cm")
                _scene_overlay(ax, g, bg, ext, ti, nbins=14, narrow=8, arrows=True)
        # row label
        axes[ri, 0].set_ylabel(row_title, fontsize=11)

    fig.suptitle("Eval rollouts: value-function overlay (cube-xy) for the two FB failure modes\n"
                 "left two = FB,  right two = GCIQL   ·   red = value (per-traj normalized) ·  white arrows = cube path  ·  ★ = goal",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = out / "failure_value_overlays.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[failure_overlay] -> {path}")

    # also write the example list for traceability
    rows_out = []
    for mode, _ in rows:
        for data, mlabel, _ in cols:
            for seed, task, ep in _pick_examples(data, mode, n=n):
                rows_out.append(dict(mode=mode, method=mlabel.lower(),
                                     seed=seed, task=task, episode=ep))
    pd.DataFrame(rows_out).to_csv(out / "examples.csv", index=False)


if __name__ == "__main__":
    main()

"""scripts/eval/eval_pickup_cycle_counts.py — for each eval rollout in each of the
four (method, regime) cells, count the number of distinct pick-and-place
cycles (= held-segments of >= K_STEPS), then break the distribution down by
the 4-phase failure mode.

Inputs (per-step eval rollouts + per-episode classification):
  FB state         : analysis/probes/representation_profile/sN_final/value_steps.parquet
                     analysis/probes/phase_probe_v2/sN_final/per_episode.parquet
  GCIQL state      : analysis/profiles/gciql_profile_v2/sN_final/{value_steps,phase_funnel}.parquet
  FB pixel         : analysis/probes/representation_profile_pixel/sN_final/value_steps.parquet
                     analysis/probes/phase_probe_pixel_v2/sN_final/per_episode.parquet
  GCIQL pixel DrQ  : analysis/profiles/gciql_profile_pixel_drq_v2/sN_final/{value_steps,phase_funnel}.parquet

Output:
  analysis/misc/eval_pickup_cycles/summary.md  + per_episode.parquet
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

K_STEPS = 5
DELTA_LIFT = 0.03
EPS_GOAL = 0.05


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    out, n, i = [], len(mask), 0
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if j - i >= K_STEPS:
                out.append((i, j))
            i = j
        else:
            i += 1
    return out


def _classify_fb(r: pd.Series) -> str:
    if r["success"]:
        return "success"
    if r["fail_phase"] == "reach":
        return "approach"
    if r["fail_phase"] == "grasp":
        return "grasp"
    if r["final_cube_lift"] < DELTA_LIFT:
        return "maintain"
    if r["final_cube_goal_dist"] <= EPS_GOAL:
        return "near"
    return "transport"


def _classify_gciql(r: pd.Series) -> str:
    fp = r["furthest_phase"]
    if fp == "success":
        return "success"
    if fp == "reach":
        return "approach"
    if fp == "grasp":
        return "grasp"
    if r["final_cube_lift"] < DELTA_LIFT:
        return "maintain"
    if r["final_cube_goal_dist"] <= EPS_GOAL:
        return "near"
    return "transport"


def _short_task(s: str) -> str:
    if "singletask-task" in s:
        return "task" + s.split("task")[-1].rstrip("v-0")[0]
    return s


def _load_cell(name: str, vs_glob: str, pe_glob: str, kind: str) -> pd.DataFrame:
    vs_paths = sorted(glob.glob(vs_glob))
    pe_paths = sorted(glob.glob(pe_glob))
    if not vs_paths or not pe_paths:
        print(f"[skip] {name}: missing vs/pe data")
        return pd.DataFrame()
    pe_by_seed = {p.split("/")[-2]: pd.read_parquet(p) for p in pe_paths}
    rows = []
    for vp in vs_paths:
        seed = vp.split("/")[-2]
        if seed not in pe_by_seed:
            continue
        vs = pd.read_parquet(vp).sort_values(["task", "episode", "t"])
        vs["task_short"] = vs["task"].map(_short_task)
        pe = pe_by_seed[seed].copy()
        if "scenario" in pe.columns:
            pe = pe[pe["scenario"] == "S0"].copy()
        pe["task_short"] = pe["task"].map(_short_task)
        pe["mode"] = pe.apply(_classify_fb if kind == "fb" else _classify_gciql, axis=1)
        seg = (vs.groupby(["task_short", "episode"])
                 .apply(lambda g: len(_runs(g["transport"].to_numpy(bool))))
                 .reset_index(name="n_segments"))
        merged = seg.merge(
            pe[["task_short", "episode", "mode"]],
            on=["task_short", "episode"], how="inner")
        merged["seed"] = seed.replace("_final", "")
        merged["cell"] = name
        rows.append(merged)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _summarize(df: pd.DataFrame, name: str) -> tuple[pd.DataFrame, list[str]]:
    if df.empty:
        return df, [f"## {name}\n\n(no data)\n"]
    g = (df.groupby("mode")
           .agg(n=("n_segments", "count"),
                mean=("n_segments", "mean"),
                median=("n_segments", "median"),
                max=("n_segments", "max"),
                pct_ge2=("n_segments", lambda x: (x >= 2).mean() * 100),
                pct_ge3=("n_segments", lambda x: (x >= 3).mean() * 100),
                pct_eq1=("n_segments", lambda x: (x == 1).mean() * 100),
                pct_eq0=("n_segments", lambda x: (x == 0).mean() * 100))
           .round(2))
    order = ["success", "approach", "grasp", "maintain", "transport", "near"]
    g = g.reindex([o for o in order if o in g.index])
    lines = [f"## {name}", "",
             "| outcome | n eps | mean | median | % with 1 cycle | % with >= 2 cycles | % with >= 3 cycles |",
             "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for outcome, row in g.iterrows():
        lines.append(
            f"| {outcome} | {int(row['n'])} | {row['mean']:.2f} | "
            f"{int(row['median'])} | {row['pct_eq1']:.0f} | "
            f"{row['pct_ge2']:.0f} | {row['pct_ge3']:.0f} |"
        )
    lines.append("")
    return g, lines


def main() -> None:
    out = Path("analysis/misc/eval_pickup_cycles")
    out.mkdir(parents=True, exist_ok=True)

    cells = [
        ("FB state", "analysis/probes/representation_profile/s*_final/value_steps.parquet",
         "analysis/probes/phase_probe_v2/s*_final/per_episode.parquet", "fb"),
        ("GCIQL state", "analysis/profiles/gciql_profile_v2/s*_final/value_steps.parquet",
         "analysis/profiles/gciql_profile_v2/s*_final/phase_funnel.parquet", "gciql"),
        ("FB pixel (DrQ)", "analysis/probes/representation_profile_pixel/s*_final/value_steps.parquet",
         "analysis/probes/phase_probe_pixel_v2/s*_final/per_episode.parquet", "fb"),
        ("GCIQL pixel DrQ", "analysis/profiles/gciql_profile_pixel_drq_v2/s*_final/value_steps.parquet",
         "analysis/profiles/gciql_profile_pixel_drq_v2/s*_final/phase_funnel.parquet", "gciql"),
    ]
    all_df, all_lines = [], [
        "# Eval pickup-and-place cycle counts (FB vs GCIQL, state vs pixel)",
        "",
        f"A 'held-segment' / pick-and-place cycle = >= {K_STEPS} consecutive eval steps with "
        "cube lifted >3 cm above the table AND gripper closed. We count these per eval "
        "rollout and break the distribution down by the 4-phase failure mode.",
        "",
    ]
    for name, vs_g, pe_g, kind in cells:
        df = _load_cell(name, vs_g, pe_g, kind)
        if not df.empty:
            df["cell"] = name
            all_df.append(df)
        _, lines = _summarize(df, name)
        all_lines.extend(lines)
    if all_df:
        merged = pd.concat(all_df, ignore_index=True)
        merged.to_parquet(out / "per_episode.parquet")
    (out / "summary.md").write_text("\n".join(all_lines) + "\n")
    print("\n".join(all_lines))


if __name__ == "__main__":
    main()

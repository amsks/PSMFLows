"""scripts/probes/phase_probe_transport_buckets.py

Aggregate transport-failure breakdown across the v2 phase_probe runs.
Splits `fail_phase == "transport"` episodes into three buckets:

    dropped   : cube no longer lifted at the final step (final_cube_lift < delta_lift)
    near_goal : held to end, final_cube_goal_dist <= eps_near
    far_goal  : held to end, final_cube_goal_dist >  eps_near

Inputs : analysis/probes/phase_probe_v2/s*_final/per_episode.parquet
Outputs: analysis/probes/phase_probe_v2/aggregate/{transport_buckets.parquet, summary.md}
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd


BUCKET_ORDER = ["dropped", "near_goal", "far_goal"]


def bucket_row(row: pd.Series, delta_lift: float, eps_near: float) -> str:
    if row["final_cube_lift"] < delta_lift:
        return "dropped"
    if row["final_cube_goal_dist"] <= eps_near:
        return "near_goal"
    return "far_goal"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="analysis/probes/phase_probe_v2")
    ap.add_argument("--delta-lift", type=float, default=0.03,
                    help="cube above table_z by this -> still lifted")
    ap.add_argument("--eps-near", type=float, default=0.05,
                    help="held cube within this 3D dist of goal -> near miss")
    args = ap.parse_args()

    root = Path(args.root)
    out = root / "aggregate"
    out.mkdir(parents=True, exist_ok=True)

    dfs = []
    for p in sorted(glob.glob(str(root / "s*_final" / "per_episode.parquet"))):
        d = pd.read_parquet(p)
        d["seed"] = Path(p).parent.name
        dfs.append(d)
    if not dfs:
        raise SystemExit(f"no per_episode.parquet found under {root}")
    df = pd.concat(dfs, ignore_index=True)
    df = df[df["scenario"] == "S0"].copy()

    n_ep = len(df)
    n_success = int(df["success"].sum())
    n_grasp_fail = int((df["fail_phase"] == "grasp").sum())
    n_reach_fail = int((df["fail_phase"] == "reach").sum())
    n_transport = int((df["fail_phase"] == "transport").sum())

    t = df[df["fail_phase"] == "transport"].copy()
    t["bucket"] = t.apply(
        lambda r: bucket_row(r, args.delta_lift, args.eps_near), axis=1)
    bucket_counts = t["bucket"].value_counts().reindex(BUCKET_ORDER, fill_value=0)

    # Per-seed bucket fractions of total episodes (for cross-seed mean/std).
    per_seed = (
        t.groupby(["seed", "bucket"]).size().unstack(fill_value=0)
        .reindex(columns=BUCKET_ORDER, fill_value=0)
    )
    seed_total = df.groupby("seed").size()
    per_seed_pct = per_seed.divide(seed_total, axis=0) * 100.0

    per_seed_pct.to_csv(out / "per_seed_pct.csv")
    t.to_parquet(out / "transport_buckets.parquet")

    lines = [
        "# Transport-failure breakdown (v2 phase probe, S0 baseline)",
        "",
        f"- thresholds: delta_lift={args.delta_lift} m, eps_near={args.eps_near} m",
        f"- seeds: {sorted(df['seed'].unique())}",
        f"- episodes: {n_ep}",
        "",
        "## Outcome composition (% of all S0 episodes)",
        "",
        "| Outcome | count | % |",
        "| :--- | ---: | ---: |",
        f"| success | {n_success} | {100*n_success/n_ep:.1f} |",
        f"| fail @ reach | {n_reach_fail} | {100*n_reach_fail/n_ep:.1f} |",
        f"| fail @ grasp | {n_grasp_fail} | {100*n_grasp_fail/n_ep:.1f} |",
        f"| fail @ transport | {n_transport} | {100*n_transport/n_ep:.1f} |",
        "",
        "## Transport failure broken down",
        "",
        "| Bucket | count | % of all episodes | % of transport-fails |",
        "| :--- | ---: | ---: | ---: |",
    ]
    for b in BUCKET_ORDER:
        c = int(bucket_counts[b])
        lines.append(
            f"| {b} | {c} | {100*c/n_ep:.1f} | "
            f"{100*c/n_transport:.1f} |"
        )
    lines += [
        "",
        "## Per-seed bucket percentages (% of S0 episodes for that seed)",
        "",
        "| seed | " + " | ".join(BUCKET_ORDER) + " |",
        "| :--- | " + " | ".join(["---:"] * len(BUCKET_ORDER)) + " |",
    ]
    for seed_name, row in per_seed_pct.round(1).iterrows():
        lines.append(
            "| " + seed_name + " | "
            + " | ".join(f"{row[b]:.1f}" for b in BUCKET_ORDER) + " |"
        )
    lines += [
        "",
        "## Bucket means across seeds",
        "",
        "| bucket | mean (%) | std (%) |",
        "| :--- | ---: | ---: |",
    ]
    for b in BUCKET_ORDER:
        col = per_seed_pct[b]
        lines.append(f"| {b} | {col.mean():.1f} | {col.std():.1f} |")

    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print((out / "summary.md").read_text())


if __name__ == "__main__":
    main()

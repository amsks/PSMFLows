#!/usr/bin/env python
"""scripts/coverage/aggregate_crl_phase.py — aggregate per-seed CRL phase parquets.

Reads analysis/probes/phase_probe_crl/s*_final/per_episode.parquet, concatenates
them with a `seed` column, writes aggregate_per_episode.parquet, and
emits a per-task / per-fail-phase summary (overall.md + funnel.md) that
the LaTeX integration step consumes.

Computes the 4-bucket outcome breakdown that rlbrew.tex uses by
splitting fail_phase='transport' into 'maintain grasp' (cube ended on
the table, final_cube_lift < delta_lift) vs 'transport to goal' (cube
still held but not at goal).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent.parent
SRC = REPO / "analysis/probes/phase_probe_crl"

# Same delta_lift threshold as evals/phase_probe.py::Thresholds default.
DELTA_LIFT = 0.03


def _outcome(row) -> str:
    """4-bucket outcome label matching the rlbrew tables."""
    if row["success"]:
        return "success"
    if row["fail_phase"] == "reach":
        return "approach"
    if row["fail_phase"] == "grasp":
        return "grasp"
    # fail_phase == "transport": split by final_cube_lift
    if row["final_cube_lift"] < DELTA_LIFT:
        return "maintain"  # cube dropped on the table
    return "transport"     # cube still held but not at goal


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default=str(SRC),
                    help="root holding s*_final/per_episode.parquet "
                         "(default: the CRL+FlowBC run)")
    ap.add_argument("--label", default="CRL+FlowBC",
                    help="method name used in the markdown headings")
    args = ap.parse_args()
    src = Path(args.src)
    label = args.label

    parquets = sorted(src.glob("s*_final/per_episode.parquet"))
    if not parquets:
        print(f"[aggregate] no per-seed parquets under {src}", file=sys.stderr)
        return 1

    frames = []
    for p in parquets:
        df = pd.read_parquet(p)
        seed = int(p.parent.name.replace("s", "").replace("_final", ""))
        df.insert(0, "seed", seed)
        frames.append(df)
    big = pd.concat(frames, ignore_index=True)
    big["outcome"] = big.apply(_outcome, axis=1)

    out = src / "aggregate"; out.mkdir(exist_ok=True)
    big.to_parquet(out / "aggregate_per_episode.parquet")

    # Filter to the M1 baseline (S0) for the outcome-distribution table.
    s0 = big[big["scenario"] == "S0"].copy()

    # Per-seed success rates (overall + per-task)
    seed_overall = s0.groupby("seed")["success"].mean()
    seed_per_task = (s0.groupby(["seed", "task"])["success"].mean()
                       .unstack("task"))

    # 4-bucket outcome distribution: mean over (seed, task) cells -> then
    # describe across seeds for IQM-style reporting.
    outcome_counts = (
        s0.groupby(["seed", "outcome"])
          .size()
          .unstack("outcome", fill_value=0)
          .reindex(columns=["success", "approach", "grasp",
                            "maintain", "transport"], fill_value=0))
    # Each (seed) row has 50 episodes (5 tasks * 10 eps) — convert to %.
    outcome_pct = outcome_counts.div(outcome_counts.sum(axis=1), axis=0) * 100
    outcome_summary = outcome_pct.agg(["mean", "std"]).T

    lines = [
        f"# {label} phase-probe — aggregate",
        "",
        f"- per-seed parquets: {len(parquets)}",
        f"- total episodes: {len(big)}",
        f"- S0-only episodes: {len(s0)}",
        f"- seeds: {sorted(big['seed'].unique().tolist())}",
        f"- S0 overall success (mean across seeds): "
        f"{seed_overall.mean():.3f} (std {seed_overall.std():.3f})",
        "",
        "## Per-seed overall success (S0)",
        "",
        seed_overall.round(3).to_string(),
        "",
        "## Per-task success rate (S0, % episodes)",
        "",
        (seed_per_task * 100).round(1).to_string(),
        "",
        "## Outcome distribution (S0, mean ± std across seeds, % episodes)",
        "",
        outcome_summary.round(2).to_string(),
        "",
    ]
    (out / "overall.md").write_text("\n".join(lines))

    # M2 intervention summary (S0/S1/S2 reached/secured/success)
    m2 = (big.groupby(["seed", "scenario"])[
              ["reached", "secured", "success"]
          ].mean() * 100)
    m2_summary = m2.groupby("scenario").agg(["mean", "std"]).round(2)
    (out / "intervention.md").write_text(
        f"# {label} intervention study (S0/S1/S2) — % episodes\n\n"
        + m2_summary.to_string()
    )

    print(f"[aggregate] wrote {out}/aggregate_per_episode.parquet")
    print(f"[aggregate]       {out}/overall.md")
    print(f"[aggregate]       {out}/intervention.md")
    print(f"[aggregate] overall S0 success: {seed_overall.mean():.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

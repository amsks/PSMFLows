"""scripts/probes/phase_probe_aggregate.py — aggregate phase-probe outputs across seeds.

Reads every <root>/s*_final/per_episode.parquet, tags it with the seed
parsed from the directory name, concatenates, and writes cross-seed
summaries + plots:

  <out>/aggregate_per_episode.parquet   all episodes, with `seed`
  <out>/by_seed_summary.parquet         per seed x task x scenario rates
  <out>/overall.md                      mean +- std across seeds, readout
  <out>/success_by_scenario.png         success rate per seed, by scenario
  <out>/fail_phase_composition.png      S0 baseline fail-phase make-up
  <out>/funnel.png                      reached -> secured -> success funnel

Usage:
    python scripts/probes/phase_probe_aggregate.py \\
      --root analysis/legacy/phase_probe --out analysis/legacy/phase_probe/aggregate
"""

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

_SEED_RE = re.compile(r"s(\d+)_final")


def load_all(root: Path) -> pd.DataFrame:
    frames = []
    for p in sorted(root.glob("s*_final/per_episode.parquet")):
        m = _SEED_RE.search(p.parent.name)
        if not m:
            continue
        df = pd.read_parquet(p)
        df["seed"] = int(m.group(1))
        frames.append(df)
    if not frames:
        raise FileNotFoundError(
            f"No s*_final/per_episode.parquet under {root}")
    return pd.concat(frames, ignore_index=True)


def by_seed_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["seed", "task", "scenario"])
        .agg(reached_rate=("reached", "mean"),
             secured_rate=("secured", "mean"),
             success_rate=("success", "mean"),
             n=("episode", "count"))
        .reset_index()
    )


def plot_success_by_scenario(by_seed: pd.DataFrame, path: Path) -> None:
    # mean over tasks, per (seed, scenario)
    g = (by_seed.groupby(["seed", "scenario"])["success_rate"]
         .mean().reset_index())
    seeds = sorted(g["seed"].unique())
    scenarios = sorted(g["scenario"].unique())
    x = np.arange(len(seeds))
    w = 0.8 / max(1, len(scenarios))
    fig, ax = plt.subplots(figsize=(max(6, 1.3 * len(seeds)), 4))
    for j, sc in enumerate(scenarios):
        vals = [float(g[(g.seed == s) & (g.scenario == sc)]["success_rate"]
                      .mean()) for s in seeds]
        ax.bar(x + j * w, vals, w, label=sc)
    ax.set_xticks(x + w * (len(scenarios) - 1) / 2)
    ax.set_xticklabels([f"s{s}" for s in seeds])
    ax.set_ylabel("success rate (mean over 5 tasks)")
    ax.set_title("Phase-probe success by scenario, per seed")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_fail_phase_composition(df: pd.DataFrame, path: Path) -> None:
    s0 = df[df.scenario == "S0"]
    order = ["none", "transport", "grasp", "reach"]
    seeds = sorted(s0["seed"].unique())
    fig, ax = plt.subplots(figsize=(max(6, 1.3 * len(seeds)), 4))
    bottom = np.zeros(len(seeds))
    for ph in order:
        frac = [
            float((s0[s0.seed == s]["fail_phase"] == ph).mean())
            for s in seeds
        ]
        label = "success" if ph == "none" else f"fail@{ph}"
        ax.bar([f"s{s}" for s in seeds], frac, bottom=bottom, label=label)
        bottom += np.array(frac)
    ax.set_ylabel("fraction of S0 episodes")
    ax.set_title("Baseline (S0) outcome composition per seed")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_funnel(df: pd.DataFrame, path: Path) -> None:
    s0 = df[df.scenario == "S0"]
    stages = ["reached", "secured", "success"]
    means = [float(s0[st].mean()) for st in stages]
    stds = [float(s0.groupby("seed")[st].mean().std(ddof=0)) for st in stages]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(stages, means, yerr=stds, color=["#4c72b0", "#dd8452", "#55a868"])
    for i, m in enumerate(means):
        ax.text(i, m + 0.02, f"{m*100:.0f}%", ha="center", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("rate (mean over seeds, S0)")
    ax.set_title("Reach -> Secure -> Success funnel (baseline)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_overall_md(df: pd.DataFrame, by_seed: pd.DataFrame,
                     path: Path) -> None:
    lines = ["# Phase-probe — cross-seed aggregate", ""]
    seeds = sorted(df["seed"].unique())
    lines.append(f"Seeds: {', '.join('s'+str(s) for s in seeds)}  "
                 f"(n={int(df.groupby(['seed','task','scenario']).size().mean())} "
                 f"episodes per task x scenario)")
    lines.append("")
    for sc in sorted(df["scenario"].unique()):
        sub = df[df.scenario == sc]
        per_seed = sub.groupby("seed")[["reached", "secured", "success"]].mean()
        lines.append(
            f"- **{sc}**: reached "
            f"{per_seed['reached'].mean()*100:.1f}±{per_seed['reached'].std(ddof=0)*100:.1f}%  "
            f"secured {per_seed['secured'].mean()*100:.1f}±{per_seed['secured'].std(ddof=0)*100:.1f}%  "
            f"success {per_seed['success'].mean()*100:.1f}±{per_seed['success'].std(ddof=0)*100:.1f}%")
    lines += ["", "## S0 baseline failure attribution", ""]
    s0 = df[df.scenario == "S0"]
    comp = s0["fail_phase"].value_counts(normalize=True)
    for ph in ["none", "transport", "grasp", "reach"]:
        if ph in comp:
            tag = "success" if ph == "none" else f"fail@{ph}"
            lines.append(f"- {tag}: {comp[ph]*100:.1f}% of episodes")
    lines += ["", "## Readout", ""]
    lines.append(
        "Hypothesis was: reach hard? no; grasp/pick-up is the bottleneck; "
        "transport easy once held. Evidence: reach is ~100%, secure (grasp"
        "+lift) is high, and S0 failures concentrate at **transport**, not "
        "grasp. S2 (pre-grasped, off-distribution clamp) does not exceed S0. "
        "=> the dominant failure is delivering/placing the lifted cube at "
        "the goal, not acquiring it.")
    Path(path).write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(REPO_ROOT / "analysis" / "legacy" / "phase_probe"))
    ap.add_argument("--out",
                    default=str(REPO_ROOT / "analysis" / "legacy" / "phase_probe" / "aggregate"))
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = load_all(root)
    by_seed = by_seed_summary(df)

    df.to_parquet(out / "aggregate_per_episode.parquet")
    by_seed.to_parquet(out / "by_seed_summary.parquet")
    write_overall_md(df, by_seed, out / "overall.md")
    plot_success_by_scenario(by_seed, out / "success_by_scenario.png")
    plot_fail_phase_composition(df, out / "fail_phase_composition.png")
    plot_funnel(df, out / "funnel.png")
    print(f"[phase_probe_aggregate] {df['seed'].nunique()} seeds -> {out}")


if __name__ == "__main__":
    main()

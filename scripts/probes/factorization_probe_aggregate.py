"""Pool per-seed factorization-probe outputs (each already pooled over tasks)
into one cross-seed verdict + R^2 table.

  .venv/bin/python -m scripts.probes.factorization_probe_aggregate \
    --root analysis/probes/factorization_probe --out analysis/probes/factorization_probe/aggregate
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from evals.factorization_probe import classify_bucket_regression


def aggregate(root: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    ceil = [pd.read_parquet(p).assign(seed=p.parent.name)
            for p in root.glob("*/readout_ceiling.parquet")]
    if not ceil:
        raise SystemExit("no per-seed readout_ceiling.parquet under root")
    ceil = pd.concat(ceil, ignore_index=True)

    # Mean R^2 per (feature, target, region, kind), pooled over tasks AND seeds.
    mean_ceil = (ceil.groupby(["feature", "target", "region", "kind"])["score"]
                 .mean().reset_index())
    verdict = classify_bucket_regression(mean_ceil, target="d")
    n_seeds = int(ceil["seed"].nunique())

    mean_ceil.to_parquet(out / "readout_ceiling_mean.parquet")
    payload = {**verdict, "n_seeds": n_seeds}
    (out / "verdict.json").write_text(json.dumps(payload, indent=2))

    tr = (mean_ceil[(mean_ceil.region == "transport") & (mean_ceil.target == "d")]
          .sort_values(["feature", "kind"]))
    lines = ["# Factorization probe — cross-seed verdict", "",
             f"**{verdict['bucket']}** — {verdict['why']}",
             f"(n_seeds {n_seeds}, R^2 pooled over tasks+seeds)", "",
             "## Transport-phase R^2(feature -> cube-to-goal d)", "",
             "| feature | kind | mean R^2 |", "| :-- | :-- | :-- |"]
    for _, r in tr.iterrows():
        lines.append(f"| {r.feature} | {r.kind} | {r.score:.3f} |")
    (out / "story.md").write_text("\n".join(lines) + "\n")
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    payload = aggregate(Path(args.root), Path(args.out))
    print(f"[aggregate] {payload['bucket']} (n_seeds {payload['n_seeds']})")


if __name__ == "__main__":
    main()

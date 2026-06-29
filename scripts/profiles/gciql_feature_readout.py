"""Torch consumer for the GCIQL/GCIVL representation probe (run in .venv).

Reads the phi(s,g) npz files dumped by scripts.profiles.gciql_feature_extract and runs
the SAME readout ceiling as the FB probe: R^2(phi -> cube-to-goal d), linear vs
MLP, transport phase, vs a cube-xyz "raw" reference. This is the positive
control: if the value representation RETAINS terminal geometry (high R^2(phi->d))
where FB's backward map B does not, that explains the performance gap.

  .venv/bin/python -m scripts.profiles.gciql_feature_readout \
    --root analysis/features_raw/gciql_feature/state_sd001 --out analysis/features_raw/gciql_feature/state_sd001/readout
The root is globbed recursively for task*.npz, so point it at one seed dir or at
a parent of many seed dirs to pool across seeds.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from evals.training_value import region_labels, cube_to_goal_dist
from evals.factorization_probe import readout_ceiling_table
from evals.phase_probe import Thresholds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    root = Path(args.root); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    thr = Thresholds()

    files = sorted(root.rglob("task*.npz"))
    if not files:
        raise SystemExit(f"no task*.npz under {root}")
    rows = []
    agent_name = "gciql"
    for f in files:
        z = np.load(f, allow_pickle=True)
        agent_name = str(z["agent"])
        cube = z["cube"].astype(np.float64)
        grip = np.clip(z["grip"].astype(np.float64) / 0.8, 0, 1)
        lift = z["lift"].astype(np.float64) - float(z["table_z"])
        region = [str(r) for r in region_labels(grip, lift, thr)]
        d = cube_to_goal_dist(cube, z["goal_xyz"].astype(np.float64))
        feats = {"raw": cube, "phi": z["phi"].astype(np.float64)}
        frame = pd.DataFrame({"region": region, "d": d})
        c = readout_ceiling_table(frame, feats, targets=("d",), seed=args.seed)
        c["seed_task"] = f"{f.parent.name}/{f.stem}"
        rows.append(c)

    ceiling = pd.concat(rows, ignore_index=True)
    pooled = (ceiling.groupby(["feature", "region", "kind"])["score"]
              .mean().reset_index())
    ceiling.to_parquet(out / "readout_ceiling.parquet")
    pooled.to_parquet(out / "readout_ceiling_pooled.parquet")

    tr = pooled[pooled.region == "transport"]
    def g(feat, kind):
        m = tr[(tr.feature == feat) & (tr.kind == kind)]
        return float(m["score"].iloc[0]) if len(m) else float("nan")
    phi_mlp, phi_lin, raw_mlp = g("phi", "mlp"), g("phi", "linear"), g("raw", "mlp")
    # Ceiling = best readout. phi is 512-d, so the MLP overfits small per-task
    # transport splits and the ridge (linear) is the reliable ceiling; max() is
    # robust either way (low-d B uses MLP, high-d phi uses linear).
    phi_ceiling = float(np.nanmax([phi_mlp, phi_lin]))
    retains = ("RETAINS terminal geometry" if phi_ceiling >= 0.6
               else "partially retains" if phi_ceiling >= 0.35 else "DISCARDS terminal geometry")
    lines = [f"# {agent_name.upper()} value-representation probe — R^2(phi -> cube-to-goal d)", "",
             f"**Value penultimate phi {retains}** (transport phase): ceiling R^2={phi_ceiling:.3f} "
             f"(linear {phi_lin:.3f}, MLP {phi_mlp:.3f}; MLP overfits 512-d phi), raw(cube) R^2={raw_mlp:.3f}",
             f"(n files {len(files)})", "",
             "## Transport-phase readout ceilings (mean R^2)", "",
             "| feature | kind | mean R^2 |", "| :-- | :-- | :-- |"]
    for _, r in tr.sort_values(["feature", "kind"]).iterrows():
        lines.append(f"| {r.feature} | {r.kind} | {r.score:.3f} |")
    (out / "story.md").write_text("\n".join(lines) + "\n")
    print(f"[gciql_readout:{agent_name}] phi ceiling R^2(transport)={phi_ceiling:.3f} "
          f"(linear {phi_lin:.3f}, MLP {phi_mlp:.3f}) -> {retains}")


if __name__ == "__main__":
    main()

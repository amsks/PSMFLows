#!/usr/bin/env python
"""scripts/probes/aliasing_probe.py — matched-observation aliasing test (paper P4).

Direct test of the interaction-gap proposition (Section theory): among states
that are CLOSE in observation but differ in controllability outcome, does the
learned value still separate them? For each method, phase region, and task we
take every fail-bound state, find its nearest success-bound state in
(standardized) observation space, and ask whether the value ranks the
success-bound one above the fail-bound one. A ranking accuracy near 0.5 means
the value aliases controllable and about-to-fail states that look alike --- the
mechanism the proposition predicts.

Reuses the shared multiseed training states + per-method value parquets written
by training_value_profile (so it is cross-method: FB/GCIQL/CRL/RLDP on the SAME
states). Run under .venv.

Usage:
    .venv/bin/python scripts/probes/aliasing_probe.py \
        --root analysis/value/training_value_multiseed \
        --out analysis/value/training_value_multiseed/aggregate_rldp
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parent.parent.parent
# Per-state instantaneous regimes (NOT the terminal failure phases). The
# maintain-vs-transport distinction is a terminal failure mode, not a state
# property, so at the state level we keep three clean regimes from (grip, lift):
# approach (pre-contact), pickup (gripper closed, cube on table), carry
# (cube lifted and held). The carry regime is where both maintain and transport
# failures originate.
PHASES = ["approach", "pickup", "carry"]
# stored region label (reach/grasp/transport) -> regime name used here
REGION_TO_REGIME = {"reach": "approach", "grasp": "pickup", "transport": "carry",
                    # tolerate the coverage-pipeline label and any 4-phase leftovers
                    "lift": "pickup", "approach": "approach", "pickup": "pickup",
                    "maintain": "carry", "carry": "carry"}
# value column per method file
METHOD_VCOL = {"fb": "V_policy", "gciql": "V", "crl": "V", "rldp": "V",
               "tdmpc2": "V"}
METHOD_LABEL = {"fb": "FB", "gciql": "GCIQL", "crl": "CRL", "rldp": "RLDP",
                "tdmpc2": "TDMPC2"}
N_TASKS = 5
RNG = np.random.default_rng(0)


def _regime(region: np.ndarray) -> np.ndarray:
    """Map the stored per-state region label to {approach, pickup, carry}."""
    return np.array([REGION_TO_REGIME.get(str(r), "approach") for r in region])


def _pair_dirs(root: Path):
    return sorted(d for d in root.iterdir()
                  if d.is_dir() and (d / "training_states.npz").exists())


def _method_values(pair_dir: Path, method: str) -> np.ndarray | None:
    """Return V as [n_states, n_tasks] for `method`, or None if absent."""
    f = pair_dir / f"{method}_values.parquet"
    if not f.exists():
        return None
    df = pd.read_parquet(f)
    v = df[METHOD_VCOL[method]].to_numpy(dtype=float)
    n = v.shape[0] // N_TASKS
    return v.reshape(N_TASKS, n).T  # [n_states, n_tasks] (rows were task-major)


def _matched_accuracy(obs_z, V_task, pos_idx, neg_idx, max_neg=2000):
    """For each fail-bound state, nearest success-bound state in obs space;
    fraction of pairs with V(success) > V(fail), and mean pair obs distance."""
    if len(pos_idx) < 5 or len(neg_idx) < 5:
        return np.nan, np.nan, 0
    if len(neg_idx) > max_neg:
        neg_idx = RNG.choice(neg_idx, max_neg, replace=False)
    tree = cKDTree(obs_z[pos_idx])
    dist, nbr = tree.query(obs_z[neg_idx], k=1)
    dist = np.asarray(dist).ravel()
    matched_pos = pos_idx[np.asarray(nbr).ravel()]
    vp, vn = V_task[matched_pos], V_task[neg_idx]
    # ties (vp == vn) count as 0.5
    acc = float(np.mean((vp > vn) + 0.5 * (vp == vn)))
    return acc, float(dist.mean()), len(neg_idx)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(REPO / "analysis/value/training_value_multiseed"))
    ap.add_argument("--out", default=str(REPO / "analysis/value/training_value_multiseed/aggregate_rldp"))
    args = ap.parse_args()
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    pairs = _pair_dirs(root)
    if not pairs:
        print(f"[aliasing] no pairs under {root}", file=sys.stderr)
        return 1

    rows = []
    for pd_dir in pairs:
        st = np.load(pd_dir / "training_states.npz", allow_pickle=True)
        obs = np.asarray(st["obs"], np.float32)
        region = np.array([str(r) for r in st["region"]])
        outcome = np.asarray(st["outcome"], dtype=bool)  # [n, n_tasks]
        # standardize obs per-dim so no coordinate dominates the metric
        mu, sd = obs.mean(0), obs.std(0) + 1e-8
        obs_z = (obs - mu) / sd
        regime = _regime(region)  # per-state {approach, pickup, carry}
        for method in METHOD_VCOL:
            V = _method_values(pd_dir, method)
            if V is None:
                continue
            for phase in PHASES:
                idx_phase = np.where(regime == phase)[0]
                accs, dists, ns = [], [], 0
                for ti in range(N_TASKS):
                    idx = idx_phase
                    if len(idx) < 10:
                        continue
                    succ = outcome[idx, ti]
                    pos = idx[succ]
                    neg = idx[~succ]
                    acc, d, n = _matched_accuracy(obs_z, V[:, ti], pos, neg)
                    if not np.isnan(acc):
                        accs.append(acc); dists.append(d); ns += n
                if accs:
                    rows.append({"pair": pd_dir.name, "method": method,
                                 "region": phase,
                                 "rank_acc": float(np.mean(accs)),
                                 "obs_dist": float(np.mean(dists)),
                                 "n_pairs": ns})
    df = pd.DataFrame(rows)
    df.to_parquet(out / "aliasing_matched_pairs.parquet")

    # cross-pair (seed) aggregate: mean rank_acc per method x region
    agg = (df.groupby(["method", "region"])["rank_acc"]
             .agg(["mean", "std"]).reset_index())
    piv = agg.pivot(index="method", columns="region", values="mean")
    piv = piv.reindex(columns=PHASES)
    piv = piv.reindex(["fb", "gciql", "crl", "rldp", "tdmpc2"]).dropna(how="all")
    piv.index = [METHOD_LABEL[m] for m in piv.index]

    lines = ["# Matched-observation aliasing test",
             "",
             "Ranking accuracy: fraction of (fail-bound state, nearest "
             "success-bound state in observation space) pairs where the value "
             "ranks the success-bound state higher. 0.5 = value cannot separate "
             "look-alike controllable vs about-to-fail states (aliasing).",
             "",
             "## Ranking accuracy by phase (mean across seeds)",
             "",
             "| method | " + " | ".join(PHASES) + " |",
             "| :--- | " + " | ".join([":---:"] * len(PHASES)) + " |"]
    for m, r in piv.iterrows():
        lines.append("| " + m + " | "
                     + " | ".join((f"{r[c]:.2f}" if pd.notna(r[c]) else "--")
                                  for c in PHASES) + " |")
    # mean matched-pair observation distance (sanity: pairs are close)
    dagg = df.groupby("method")["obs_dist"].mean()
    lines += ["", "## Mean matched-pair observation distance (standardized L2)",
              ""]
    for m in ["fb", "gciql", "crl", "rldp"]:
        if m in dagg.index:
            lines.append(f"- {METHOD_LABEL[m]}: {dagg[m]:.2f}")
    (out / "aliasing_matched_pairs.md").write_text("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\n[aliasing] wrote {out}/aliasing_matched_pairs.{{parquet,md}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

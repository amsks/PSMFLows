#!/usr/bin/env python
"""scripts/value/cf_ladder_from_feat.py — counterfactual ladder from dumped features.

Computes the counterfactual representation ladder for GCIQL/CRL from the features
dumped by cf_repfeat_jax.py: per task, the best-linear-readout AUC (stratified-CV
logistic) of the raw state and of the learned representation rep(s) for the
agent's own success, and the value's own rank-AUC. Run under .venv (sklearn).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--method", required=True)
    args = ap.parse_args()
    from sklearn.metrics import roc_auc_score
    from scripts.probes.repsep_forward_probe import _auc

    df = pd.read_parquet(REPO / f"analysis/value/repsep/cf_repfeat_{args.method}.parquet")
    obs_cols = [c for c in df.columns if c.startswith("obs_")]
    rep_cols = [c for c in df.columns if c.startswith("rep_")]
    rows = {"raw": [], "rep": [], "Q": []}
    for ti in sorted(df["task"].unique()):
        d = df[df["task"] == ti]
        y = d["success"].to_numpy(bool)
        if y.sum() < 5 or (~y).sum() < 5:
            for k in rows:
                rows[k].append(np.nan)
            continue
        rows["raw"].append(_auc(d[obs_cols].to_numpy(np.float32), y))
        rows["rep"].append(_auc(d[rep_cols].to_numpy(np.float32), y))
        rows["Q"].append(roc_auc_score(y.astype(int), d["value"].to_numpy()))
    means = {k: float(np.nanmean(v)) for k, v in rows.items()}
    out = REPO / f"analysis/value/repsep/cf_ladder_{args.method}.json"
    out.write_text(json.dumps({"method": args.method, "label": "counterfactual",
                               "means": means, "per_task": rows}, indent=2))
    print(f"=== {args.method.upper()} counterfactual ladder ===")
    print(f"  raw={means['raw']:.3f}  rep={means['rep']:.3f}  Q=value={means['Q']:.3f}  -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

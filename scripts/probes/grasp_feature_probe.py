#!/usr/bin/env python
"""scripts/probes/grasp_feature_probe.py — is the success signal in the grasp config?

Control for the representation-aliasing claim. Instead of the learned
representation, probe the RAW state sub-vectors for success-bound vs fail-bound
separability among in-hand (grasp/carry) training states. If the gripper+joint
configuration already separates the two outcomes but the learned representation
does not (Figure fig_repsep / tab:repsep), the network is discarding available
grasp information = genuine aliasing. If even the raw grasp config is at chance,
the instantaneous pose does not determine the outcome.

cube-single state layout (28-d), verified from the env:
  0:12  arm joints (pos+vel)   12:17 effector pos+yaw   17:19 gripper (open+contact)
  19:28 cube (pos+quat+yaw)

Probed per task (goal fixed, since success-bound is task-dependent but proprio is
not), AUC averaged across the 5 tasks. Run under .venv (sklearn).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
STATES = REPO / "analysis/value/training_value_multiseed/p0/training_states.npz"
CONTACT = {"grasp", "transport"}

# named column slices over the 28-d state
SUBSETS = {
    "joints (pos+vel)":     np.r_[0:12],
    "gripper (open+contact)": np.r_[17:19],
    "gripper + joints":     np.r_[0:12, 17:19],
    "proprio (full grasp config)": np.r_[0:19],
    "cube / object":        np.r_[19:28],
    "full state":           np.r_[0:28],
}


def _auc(X, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if len(np.unique(y)) < 2 or min(np.bincount(y.astype(int))) < 5:
        return np.nan
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000))
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    return float(cross_val_score(clf, X, y.astype(int), cv=cv,
                                 scoring="roc_auc").mean())


def main() -> int:
    st = np.load(STATES, allow_pickle=True)
    obs = np.asarray(st["obs"], np.float32)
    region = np.array([str(r) for r in st["region"]])
    outcome = np.asarray(st["outcome"], bool)          # [n, n_tasks]
    idx = np.where(np.isin(region, list(CONTACT)))[0]
    Xall = obs[idx]
    Y = outcome[idx]                                   # [n_contact, n_tasks]
    n_tasks = Y.shape[1]
    print(f"in-hand training states: {len(idx)}  "
          f"per-task success-bound rate: {Y.mean(0).round(3)}\n")

    rows = []
    for name, cols in SUBSETS.items():
        aucs = [_auc(Xall[:, cols], Y[:, ti]) for ti in range(n_tasks)]
        aucs = np.array(aucs, float)
        rows.append((name, len(cols), np.nanmean(aucs), np.nanstd(aucs)))

    w = max(len(n) for n, *_ in rows)
    print(f"{'feature subset':<{w}}  dims   AUC (mean±std over {n_tasks} tasks)")
    print("-" * (w + 34))
    for name, d, mu, sd in rows:
        print(f"{name:<{w}}  {d:>4}   {mu:.3f} ± {sd:.3f}")
    print("\n(0.5 = chance. Learned-rep train AUC for reference: "
          "FB 0.57, GCIQL 0.65, CRL 0.59, RLDP 0.56 — see tab:repsep.)")

    import json
    out = REPO / "analysis/value/repsep/grasp_feature.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {name: {"dims": d, "mean": mu, "std": sd}
         for name, d, mu, sd in rows}, indent=2))
    print(f"[grasp_feature] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""scripts/probes/cf_representation_probe.py — representation ladder on COUNTERFACTUAL labels.

The aliasing ladder used the offline reach-before-end label. This re-runs it with
the agent's OWN counterfactual outcome (from counterfactual_value_probe records):
for the in-hand grasps, does a linear readout of phi(s) / B(s) / F(s,pi,z)
separate the grasps the agent actually succeeds from? Compared with the value's
own ranking Q=F.z, this localizes the failure:
  - if phi/B/F (best linear readout) >> Q, the controllability signal IS in the
    representation but the value direction misses it (a fixable readout problem);
  - if phi/B/F ~ Q ~ chance, the signal is not linearly in the representation.

phi/B/F probed with stratified-CV logistic AUC; Q (1-D value) by direct rank-AUC.
Per task, averaged over the 5 goals. Reads cf_records_<method>.parquet.
Run under .venv (torch + sklearn). macOS: MUJOCO_GL=glfw.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

N_TASKS = 5


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--data-path", default="datasets")
    ap.add_argument("--mujoco-gl", default="glfw")
    args = ap.parse_args()
    os.environ["MUJOCO_GL"] = args.mujoco_gl

    from sklearn.metrics import roc_auc_score
    from evals.analysis import build_env_and_agent, load_cfg, load_checkpoint
    from evals.ogbench import OGBenchEvaluator
    from envs.ogbench import ALL_TASKS
    from data.ogbench import load_ogbench_dataset
    from scripts.probes.repsep_forward_probe import phi, B_rep, F_rep, _z_np, _auc

    rec = pd.read_parquet(REPO / f"analysis/value/repsep/cf_records_{args.method}.parquet")
    obs_cols = [c for c in rec.columns if c.startswith("obs_")]

    cfg = load_cfg(args.config, device="cpu"); cfg.data_path = args.data_path
    env, agent, _, _ = build_env_and_agent(cfg)
    load_checkpoint(agent, args.checkpoint)
    if hasattr(env, "close"):
        env.close()
    model = agent.model
    buffer = load_ogbench_dataset(domain=cfg.domain, data_path=cfg.data_path,
                                  load_n_episodes=cfg.load_n_episodes, device=cfg.device,
                                  n_transitions=cfg.n_transitions, obs_type=cfg.obs_type)
    evaluator = OGBenchEvaluator(domain=cfg.domain, agent=agent, offline_buffer=buffer,
                                 relabel_size=cfg.eval_relabel_size, n_episodes=1,
                                 shift_reward=cfg.eval_shift_reward, obs_type=cfg.obs_type,
                                 seed=cfg.seed, device=cfg.device, use_wandb=False)
    tasks = list(ALL_TASKS.get(cfg.domain, []))
    zs = {ti: evaluator._infer_z(t)[0] for ti, t in enumerate(tasks, start=1)}

    rows = {"raw": [], "phi": [], "B": [], "F": [], "Q": []}
    for ti in range(1, N_TASKS + 1):
        d = rec[rec["task"] == ti]
        obs = d[obs_cols].to_numpy(np.float32)
        y = d["success"].to_numpy(bool)
        if y.sum() < 5 or (~y).sum() < 5:
            for k in rows:
                rows[k].append(np.nan)
            continue
        Frep = F_rep(model, agent, obs, zs[ti])
        q = (Frep * _z_np(zs[ti])).sum(-1)
        rows["raw"].append(_auc(obs, y))                  # raw-state ceiling
        rows["phi"].append(_auc(phi(model, obs), y))
        rows["B"].append(_auc(B_rep(model, obs), y))
        rows["F"].append(_auc(Frep, y))
        rows["Q"].append(roc_auc_score(y.astype(int), q))

    means = {k: float(np.nanmean(v)) for k, v in rows.items()}
    out = REPO / f"analysis/value/repsep/cf_ladder_{args.method}.json"
    out.write_text(json.dumps({"method": args.method,
                               "label": "counterfactual",
                               "means": means,
                               "per_task": rows}, indent=2))
    print(f"\n=== {args.method.upper()}: representation ladder on COUNTERFACTUAL labels "
          f"(best-linear-readout AUC; Q = value's own ranking) ===")
    print(f"  raw={means['raw']:.3f}  phi(s)={means['phi']:.3f}  B(s)={means['B']:.3f}  "
          f"F(s,pi,z)={means['F']:.3f}  Q=F.z={means['Q']:.3f}")
    print(f"[cf_ladder:{args.method}] -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

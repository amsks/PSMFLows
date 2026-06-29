#!/usr/bin/env python
"""scripts/value/repsep_extract_torch.py — grasp-state representations (phi) for FB/RLDP.

phi(s) = left_encoder(fw_encoder(normalize(obs))) — the state representation the
forward map reads. Train grasps come from the shared training_states.npz
(labelled success-bound); eval grasps come from deterministic policy rollouts
(reusing evals.representation_profile.rollout_for_profile) labelled by the episode
outcome. Restricts to grasp/carry (cube-in-hand) states. Writes
analysis/value/repsep/<method>.parquet with columns {split, task, success, rep_*}.

Run under .venv (torch). macOS: --mujoco-gl glfw.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

CONTACT = {"grasp", "transport"}   # cube in hand (pickup + carry regimes)
N_TASKS = 5


def phi(model, obs_np):
    """Left-encoder state representation [n, L_dim]."""
    import torch
    with torch.no_grad():
        t = torch.as_tensor(np.asarray(obs_np), dtype=torch.float32)
        return model._left_encoder(model._fw_encoder(model._normalize(t))).cpu().numpy()


def train_rows(model, states_npz):
    st = np.load(states_npz, allow_pickle=True)
    obs = np.asarray(st["obs"], np.float32)
    region = np.array([str(r) for r in st["region"]])
    outcome = np.asarray(st["outcome"], bool)          # [n, n_tasks]
    idx = np.where(np.isin(region, list(CONTACT)))[0]
    rep = phi(model, obs[idx])
    rows = []
    for ti in range(N_TASKS):
        for j, i in enumerate(idx):
            rows.append(("train", ti + 1, -1, bool(outcome[i, ti]), rep[j]))
    return rows


def eval_rows(model, agent, cfg, episodes):
    from envs.ogbench import ALL_TASKS, create_ogbench_env
    from evals.ogbench import OGBenchEvaluator
    from evals.phase_probe import Thresholds
    from evals.representation_profile import rollout_for_profile
    from data.ogbench import load_ogbench_dataset

    thr = Thresholds()
    buffer = load_ogbench_dataset(domain=cfg.domain, data_path=cfg.data_path,
                                  load_n_episodes=cfg.load_n_episodes, device=cfg.device,
                                  n_transitions=cfg.n_transitions, obs_type=cfg.obs_type)
    evaluator = OGBenchEvaluator(domain=cfg.domain, agent=agent, offline_buffer=buffer,
                                 relabel_size=cfg.eval_relabel_size, n_episodes=1,
                                 shift_reward=cfg.eval_shift_reward, obs_type=cfg.obs_type,
                                 seed=cfg.seed, device=cfg.device, use_wandb=False)
    tasks = list(ALL_TASKS.get(cfg.domain, []))
    rows = []
    gep = 0                                  # global episode id (unique per rollout)
    for ti, task in enumerate(tasks, start=1):
        z, _ = evaluator._infer_z(task)
        env, _ = create_ogbench_env(task, seed=cfg.seed, obs_type=cfg.obs_type)
        try:
            eps = rollout_for_profile(env, agent, z, episodes, thr)
        finally:
            if hasattr(env, "close"):
                env.close()
        for ep in eps:
            mask = np.asarray(ep["transport_mask"], bool)
            if mask.sum() == 0:
                gep += 1
                continue
            success = ep["outcome"] == "success"
            rep = phi(model, np.asarray(ep["obs"])[mask])
            for r in rep:
                rows.append(("eval", ti, gep, bool(success), r))
            gep += 1
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--method", required=True, choices=["fb", "rldp"])
    ap.add_argument("--states",
                    default="analysis/value/training_value_multiseed/p0/training_states.npz")
    ap.add_argument("--data-path", default="datasets")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--out", default=None)
    ap.add_argument("--mujoco-gl", default="glfw")
    args = ap.parse_args()
    os.environ["MUJOCO_GL"] = args.mujoco_gl

    from evals.analysis import build_env_and_agent, load_cfg, load_checkpoint
    cfg = load_cfg(args.config, device="cpu")
    cfg.data_path = args.data_path
    env, agent, _, _ = build_env_and_agent(cfg)
    load_checkpoint(agent, args.checkpoint)
    if hasattr(env, "close"):
        env.close()
    if hasattr(agent, "eval"):
        agent.eval()

    rows = train_rows(agent.model, args.states) + eval_rows(agent.model, agent, cfg, args.episodes)
    df = pd.DataFrame([{"split": s, "task": t, "episode": e, "success": y,
                        **{f"rep_{i}": float(v) for i, v in enumerate(r)}}
                       for s, t, e, y, r in rows])
    out = Path(args.out or f"analysis/value/repsep/{args.method}.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    k = sum(c.startswith("rep_") for c in df.columns)
    print(f"[repsep:{args.method}] {len(df)} rows (dim {k}); "
          f"{df.groupby('split')['success'].agg(['size', 'mean']).to_dict()} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

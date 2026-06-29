#!/usr/bin/env python
"""scripts/probes/repsep_forward_probe.py — does the FORWARD MAP separate grasps?

phi(s) is only the input-side state embedding. The forward map F(s, a, z) is the
successor representation: F.B approximates the successor occupancy, and the value
is Q = F(s, a, z) . z. If anything separates success-bound from fail-bound
grasps it should be F, since the outcome lives in the future F is meant to
predict. This probe compares the linear-probe AUC of phi vs F on in-hand
(grasp/carry) states, train and eval, for the F-bearing methods (FB / RLDP).

F is task-conditioned: we set z = infer_z(task) and a = pi(s, z) (the policy
action), so F(s, pi(s,z), z) is the on-policy successor representation for that
task. Eval AUC uses episode-grouped CV (no trajectory spans folds).

Run under .venv (torch + sklearn). macOS: MUJOCO_GL=glfw.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

CONTACT = {"grasp", "transport"}
N_TASKS = 5


def phi(model, obs_np):
    """Forward-side state embedding phi(s) = left_encoder(fw_encoder(norm(s)))."""
    import torch
    with torch.no_grad():
        t = torch.as_tensor(np.asarray(obs_np), dtype=torch.float32)
        return model._left_encoder(model._fw_encoder(model._normalize(t))).cpu().numpy()


def B_rep(model, obs_np):
    """Backward/successor embedding B(s) (M = F.B, z = E[B(goal)])."""
    import torch
    with torch.no_grad():
        t = torch.as_tensor(np.asarray(obs_np), dtype=torch.float32)
        return model.backward_map(t).cpu().numpy()


def F_rep(model, agent, obs_np, z):
    """On-policy forward-map representation F(s, pi(s,z), z) -> [B, z_dim].

    Q = F . z is the value; since head-mean is linear, mean(F).z == Q, so the
    caller derives the scalar value from the returned (head-meaned) F."""
    import torch
    from evals.representation_profile import _tile_z
    with torch.no_grad():
        obs = torch.as_tensor(np.asarray(obs_np), dtype=torch.float32)
        zb = _tile_z(z, obs.shape[0]).to(obs.device).float()
        action = agent.act(obs, zb)
        if isinstance(action, np.ndarray):
            action = torch.as_tensor(action, dtype=torch.float32)
        F = model.forward_map(obs, zb, action.to(obs.device).float())
        if F.dim() == 3:                      # [P, B, z_dim] -> mean over heads
            F = F.mean(0)
        return F.cpu().numpy()


def _z_np(z):
    import torch
    return np.asarray(z.detach().cpu() if isinstance(z, torch.Tensor)
                      else z, dtype=np.float32).reshape(-1)


def _auc(X, y, groups=None):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import (StratifiedGroupKFold, StratifiedKFold,
                                         cross_val_score)
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2 or min(np.bincount(y)) < 5:
        return np.nan
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    if groups is not None:
        ng = len(np.unique(groups))
        if ng < 2:
            return np.nan
        cv = StratifiedGroupKFold(min(5, ng), shuffle=True, random_state=0)
        return float(cross_val_score(clf, X, y, groups=groups, cv=cv,
                                     scoring="roc_auc").mean())
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    return float(cross_val_score(clf, X, y, cv=cv, scoring="roc_auc").mean())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--states",
                    default="analysis/value/training_value_multiseed/p0/training_states.npz")
    ap.add_argument("--data-path", default="datasets")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--mujoco-gl", default="glfw")
    args = ap.parse_args()
    os.environ["MUJOCO_GL"] = args.mujoco_gl

    from evals.analysis import build_env_and_agent, load_cfg, load_checkpoint
    from envs.ogbench import ALL_TASKS, create_ogbench_env
    from evals.ogbench import OGBenchEvaluator
    from evals.phase_probe import Thresholds
    from evals.representation_profile import rollout_for_profile
    from data.ogbench import load_ogbench_dataset

    cfg = load_cfg(args.config, device="cpu")
    cfg.data_path = args.data_path
    env, agent, _, _ = build_env_and_agent(cfg)
    load_checkpoint(agent, args.checkpoint)
    if hasattr(env, "close"):
        env.close()
    if hasattr(agent, "eval"):
        agent.eval()
    model = agent.model

    thr = Thresholds()
    buffer = load_ogbench_dataset(domain=cfg.domain, data_path=cfg.data_path,
                                  load_n_episodes=cfg.load_n_episodes, device=cfg.device,
                                  n_transitions=cfg.n_transitions, obs_type=cfg.obs_type)
    evaluator = OGBenchEvaluator(domain=cfg.domain, agent=agent, offline_buffer=buffer,
                                 relabel_size=cfg.eval_relabel_size, n_episodes=1,
                                 shift_reward=cfg.eval_shift_reward, obs_type=cfg.obs_type,
                                 seed=cfg.seed, device=cfg.device, use_wandb=False)
    tasks = list(ALL_TASKS.get(cfg.domain, []))
    zs = {ti: evaluator._infer_z(task)[0] for ti, task in enumerate(tasks, start=1)}

    # ---- TRAIN grasps (per-task AUC, averaged) ----
    st = np.load(args.states, allow_pickle=True)
    obs = np.asarray(st["obs"], np.float32)
    region = np.array([str(r) for r in st["region"]])
    outcome = np.asarray(st["outcome"], bool)
    idx = np.where(np.isin(region, list(CONTACT)))[0]
    sub = obs[idx]
    phi_tr = phi(model, sub)                         # task-independent
    B_tr = B_rep(model, sub)                         # task-independent
    from sklearn.metrics import roc_auc_score
    tr_phi, tr_B, tr_F, tr_V = [], [], [], []
    vtrain = []                                      # (task, success, q, q_z)
    for ti in range(1, N_TASKS + 1):
        y = outcome[idx, ti - 1]
        Frep = F_rep(model, agent, sub, zs[ti])
        q = (Frep * _z_np(zs[ti])).sum(-1)          # scalar value Q = F.z
        tr_phi.append(_auc(phi_tr, y))
        tr_B.append(_auc(B_tr, y))
        tr_F.append(_auc(Frep, y))
        tr_V.append(roc_auc_score(y.astype(int), q)
                    if len(np.unique(y)) > 1 else np.nan)
        qz = (q - q.mean()) / (q.std() + 1e-8)
        vtrain += [(ti, bool(a), float(b), float(c)) for a, b, c in zip(y, q, qz)]
    tr_phi, tr_B = np.array(tr_phi), np.array(tr_B)
    tr_F, tr_V = np.array(tr_F), np.array(tr_V)

    # ---- EVAL grasps (pooled, episode-grouped) ----
    Xphi, XF, yev, gev, qev = [], [], [], [], []
    veval = []                                       # (episode, outcome, success, q)
    gep = 0
    for ti, task in enumerate(tasks, start=1):
        z = zs[ti]; z_np = _z_np(z)
        env_t, _ = create_ogbench_env(task, seed=cfg.seed, obs_type=cfg.obs_type)
        try:
            eps = rollout_for_profile(env_t, agent, z, args.episodes, thr)
        finally:
            if hasattr(env_t, "close"):
                env_t.close()
        for ep in eps:
            mask = np.asarray(ep["transport_mask"], bool)
            if mask.sum() == 0:
                gep += 1
                continue
            o = np.asarray(ep["obs"])[mask]
            succ = ep["outcome"] == "success"
            Frep = F_rep(model, agent, o, z)
            q = (Frep * z_np).sum(-1)
            Xphi.append(phi(model, o)); XF.append(Frep)
            yev.append(np.full(len(o), succ)); gev.append(np.full(len(o), gep))
            qev.append(q)
            veval += [(gep, str(ep["outcome"]), bool(succ), float(qi)) for qi in q]
            gep += 1
    Xphi = np.concatenate(Xphi); XF = np.concatenate(XF)
    yev = np.concatenate(yev); gev = np.concatenate(gev); qev = np.concatenate(qev)
    ev_phi = _auc(Xphi, yev, groups=gev)
    ev_F = _auc(XF, yev, groups=gev)
    ev_V = _auc(qev.reshape(-1, 1), yev, groups=gev)

    print(f"\n=== {args.method.upper()}: phi(s) | B(s) | F(s,pi,z) | value Q=F.z "
          f"separability (success- vs fail-bound, AUC) ===")
    print(f"{'':<8}{'phi(s)':>9}{'B(s)':>9}{'F(s,pi,z)':>11}{'Q=F.z':>9}")
    print(f"{'train':<8}{tr_phi.mean():>9.3f}{tr_B.mean():>9.3f}"
          f"{tr_F.mean():>11.3f}{tr_V.mean():>9.3f}   (per-task mean)")
    print(f"{'eval':<8}{ev_phi:>9.3f}{'--':>9}{ev_F:>11.3f}{ev_V:>9.3f}"
          f"   (pooled, episode-grouped; n={len(yev)})")
    print("0.5 = chance.  (eval is high-variance over ~50 episodes; trust train.)")

    import json
    import pandas as pd
    out = REPO / f"analysis/value/repsep/forward_probe_{args.method}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "method": args.method,
        "train": {"phi_mean": float(tr_phi.mean()), "phi_per_task": tr_phi.tolist(),
                  "B_mean": float(tr_B.mean()), "B_per_task": tr_B.tolist(),
                  "F_mean": float(tr_F.mean()), "F_per_task": tr_F.tolist(),
                  "V_mean": float(np.nanmean(tr_V)), "V_per_task": tr_V.tolist()},
        "eval": {"phi": ev_phi, "F": ev_F, "V": ev_V, "n": int(len(yev))},
    }, indent=2))
    print(f"[forward_probe] wrote {out}")

    pd.DataFrame(vtrain, columns=["task", "success", "q", "q_z"]).to_parquet(
        REPO / f"analysis/value/repsep/value_train_{args.method}.parquet")
    pd.DataFrame(veval, columns=["episode", "outcome", "success", "q"]).to_parquet(
        REPO / f"analysis/value/repsep/value_eval_{args.method}.parquet")
    print(f"[forward_probe] wrote value_train/value_eval parquets")
    return 0


if __name__ == "__main__":
    sys.exit(main())

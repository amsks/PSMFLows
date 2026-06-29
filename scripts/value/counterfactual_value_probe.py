#!/usr/bin/env python
"""scripts/value/counterfactual_value_probe.py — value aliasing vs the agent's OWN outcome.

The training-state value probe labels grasps "success-bound" using the play-data
trajectory (a proxy). This script removes that caveat: it resets the simulator to
each in-hand grasp state and rolls out the agent's own policy, labelling the grasp
by whether *the agent* reaches the goal from it. Then it measures whether the
value V(s|g) ranks grasps by the agent's true counterfactual success.

Unlike the eval probe, each grasp gets its OWN independent rollout outcome (no
shared-episode label / leakage) and the grasps span the play-data distribution
(the value's training distribution), not just the agent's self-generated states.

Mechanism (validated): the play data stores physics=qpos per step; we
env.reset(task) to set the goal, env.set_state(qpos, 0), and hold the gripper
actuator closed so the in-hand grasp is established, then roll out pi.

For FB-family agents: V(s|g) = Q(s, pi(s,z), z) = F(s,pi,z).z, z = infer_z(task).
Run under .venv (torch). macOS: MUJOCO_GL=glfw.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

N_TASKS = 5
# obs layout: 18 gripper_contact, 21 cube_z*10
GRIP_CONTACT, CUBE_Z = 18, 21


def _zb(z, B):
    import torch
    z = torch.as_tensor(z)
    if z.dim() == 1:
        z = z.unsqueeze(0)
    return z.expand(B, -1).float() if z.shape[0] == 1 else z.float()


def value_q(model, agent, obs_np, z):
    """V(s|z) = Q(s, pi(s,z), z) = F(s,pi,z) . z."""
    import torch
    with torch.no_grad():
        obs = torch.as_tensor(np.asarray(obs_np), dtype=torch.float32)
        zb = _zb(z, obs.shape[0])
        a = agent.act(obs, zb)
        if isinstance(a, np.ndarray):
            a = torch.as_tensor(a, dtype=torch.float32)
        F = model.forward_map(obs, zb, a.float())
        if F.dim() == 3:
            F = F.mean(0)
        return (F * zb).sum(-1).cpu().numpy()


def sample_grasps(data_path, domain, n, lift_thr=0.05, seed=0):
    """Sample n in-hand (contact + lifted) grasp states from the play data."""
    buf = Path(data_path) / domain / "buffer"
    if not buf.exists():
        buf = Path(data_path) / "buffer"
    files = sorted(glob.glob(str(buf / "episode_*.npz")))
    obs_l, phys_l = [], []
    for f in files[:60]:
        d = np.load(f)
        o = np.asarray(d["observation"], np.float32)
        ph = np.asarray(d["physics"], np.float32)
        m = (o[:, GRIP_CONTACT] > 0.5) & (o[:, CUBE_Z] / 10.0 > lift_thr)
        for i in np.where(m)[0]:
            obs_l.append(o[i]); phys_l.append(ph[i])
    obs_a = np.asarray(obs_l, np.float32); phys_a = np.asarray(phys_l, np.float32)
    rng = np.random.default_rng(seed)
    sel = rng.choice(len(obs_a), min(n, len(obs_a)), replace=False)
    return obs_a[sel], phys_a[sel]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--data-path", default="datasets")
    ap.add_argument("--n-states", type=int, default=200)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--mujoco-gl", default="glfw")
    ap.add_argument("--out-tag", default="",
                    help="suffix for output files, e.g. '_seed03' (avoids clobbering the primary run)")
    args = ap.parse_args()
    os.environ["MUJOCO_GL"] = args.mujoco_gl

    import mujoco
    import torch
    from sklearn.metrics import roc_auc_score
    from evals.analysis import build_env_and_agent, load_cfg, load_checkpoint
    from envs.ogbench import ALL_TASKS, create_ogbench_env
    from evals.ogbench import OGBenchEvaluator
    from data.ogbench import load_ogbench_dataset

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

    obs_g, phys_g = sample_grasps(args.data_path, cfg.domain, args.n_states)
    print(f"[cf] sampled {len(obs_g)} in-hand grasps; rolling out {len(tasks)} goals "
          f"x {args.max_steps} steps each")

    per_task_auc, per_task_sr = [], []
    records = []                                          # (task, success, value, obs)
    for ti, task in enumerate(tasks, start=1):
        z = zs[ti]
        val = value_q(model, agent, obs_g, z)            # value at each grasp
        e, _ = create_ogbench_env(task, seed=cfg.seed, obs_type=cfg.obs_type)
        u = e.unwrapped; nv = u._model.nv
        succ = np.zeros(len(phys_g), bool)
        for j, qpos in enumerate(phys_g):
            e.reset(options=dict(task_id=ti))
            u.set_state(np.asarray(qpos, np.float64), np.zeros(nv))
            u._data.ctrl[u._gripper_actuator_ids] = 0.0   # establish the grasp
            mujoco.mj_forward(u._model, u._data)
            ob = np.asarray(u.compute_observation(), np.float32)
            s = False
            for _ in range(args.max_steps):
                a = agent.act(torch.as_tensor(ob[None]), _zb(z, 1))
                a = np.clip(np.asarray(a).reshape(-1), -1.0, 1.0)
                ob_, _, term, trunc, info = e.step(a)
                ob = np.asarray(ob_, np.float32)
                s = s or bool(info.get("success", False))
                if term or trunc:
                    break
            succ[j] = s
        if hasattr(e, "close"):
            e.close()
        auc = (roc_auc_score(succ.astype(int), val)
               if 5 <= succ.sum() <= len(succ) - 5 else float("nan"))
        per_task_auc.append(auc); per_task_sr.append(float(succ.mean()))
        for j in range(len(obs_g)):
            records.append((ti, bool(succ[j]), float(val[j]), obs_g[j]))
        print(f"  task{ti}: cf success rate {succ.mean():.2f}  value-AUC {auc:.3f}")

    import pandas as pd
    rec = pd.DataFrame([{"task": t, "success": s, "value": v,
                         **{f"obs_{i}": float(o[i]) for i in range(o.shape[0])}}
                        for t, s, v, o in records])
    rec.to_parquet(REPO / f"analysis/value/repsep/cf_records_{args.method}{args.out_tag}.parquet")

    per_task_auc = np.array(per_task_auc, float)
    out = REPO / f"analysis/value/repsep/cf_value_{args.method}{args.out_tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "method": args.method, "n_states": int(len(obs_g)),
        "cf_value_auc_mean": float(np.nanmean(per_task_auc)),
        "cf_value_auc_per_task": per_task_auc.tolist(),
        "cf_success_rate": per_task_sr,
    }, indent=2))
    print(f"[cf:{args.method}] counterfactual value-AUC mean {np.nanmean(per_task_auc):.3f} "
          f"(cf success {np.mean(per_task_sr):.2f}) -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

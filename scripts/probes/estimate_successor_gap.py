#!/usr/bin/env python
"""scripts/probes/estimate_successor_gap.py — direct estimate of the tail successor gap.

The theory defines a tail successor gap: observation-near post-contact states can
have very different tail success probabilities. We estimate it directly.

For in-hand grasps reset into the simulator, we roll out the agent's own policy
K times per grasp to estimate the tail success probability v_hat(s)=P(reach goal
| s) (averaging out the stochastic flow actor, so the variation reflects the
state, not policy noise). For every pair of grasps under the same goal we then
record the observation distance ||Omega(s1)-Omega(s2)|| and the tail-probability
gap |v_hat(s1)-v_hat(s2)|. A smooth value would force the gap to 0 as the
observation distance shrinks; a real interaction gap does not.

Writes analysis/value/repsep/gap_pairs_<method>.parquet {task, dobs, dv} and
gap_points_<method>.parquet {task, vhat, ...}. Run under .venv. MUJOCO_GL=glfw.
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

N_TASKS = 5


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--n-states", type=int, default=60)
    ap.add_argument("--k-rollouts", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=160)
    args = ap.parse_args()
    os.environ["MUJOCO_GL"] = "glfw"

    import mujoco
    import torch
    import pandas as pd
    from evals.analysis import build_env_and_agent, load_cfg, load_checkpoint
    from evals.ogbench import OGBenchEvaluator
    from envs.ogbench import ALL_TASKS
    from data.ogbench import load_ogbench_dataset
    from scripts.value.counterfactual_value_probe import sample_grasps, _zb
    import ogbench

    cfg = load_cfg(args.config, device="cpu"); cfg.data_path = "datasets"
    env, agent, _, _ = build_env_and_agent(cfg)
    load_checkpoint(agent, args.checkpoint)
    if hasattr(env, "close"):
        env.close()
    buf = load_ogbench_dataset(domain=cfg.domain, data_path=cfg.data_path,
                               load_n_episodes=cfg.load_n_episodes, device=cfg.device,
                               n_transitions=cfg.n_transitions, obs_type=cfg.obs_type)
    ev = OGBenchEvaluator(domain=cfg.domain, agent=agent, offline_buffer=buf,
                          relabel_size=cfg.eval_relabel_size, n_episodes=1,
                          shift_reward=cfg.eval_shift_reward, obs_type=cfg.obs_type,
                          seed=cfg.seed, device=cfg.device, use_wandb=False)
    tasks = list(ALL_TASKS.get(cfg.domain, []))
    zs = {ti: ev._infer_z(t)[0] for ti, t in enumerate(tasks, start=1)}

    obs_g, phys_g = sample_grasps("datasets", cfg.domain, args.n_states)
    e = ogbench.make_env_and_datasets(cfg.domain, env_only=True)
    u = e.unwrapped; nv = u._model.nv

    def vbatch(qpos, ti, z, K):
        s = 0
        for _ in range(K):
            e.reset(options=dict(task_id=ti))
            u.set_state(np.asarray(qpos, np.float64), np.zeros(nv))
            u._data.ctrl[u._gripper_actuator_ids] = 0.0
            mujoco.mj_forward(u._model, u._data)
            ob = np.asarray(u.compute_observation(), np.float32)
            ok = False
            for _ in range(args.max_steps):
                a = np.clip(np.asarray(agent.act(torch.as_tensor(ob[None]), _zb(z, 1))).reshape(-1), -1, 1)
                ob_, _, term, trunc, info = e.step(a); ob = np.asarray(ob_, np.float32)
                ok = ok or bool(info.get("success", False))
                if term or trunc:
                    break
            s += int(ok)
        return s / K

    K = args.k_rollouts
    pair_rows, pt_rows, noise_rows = [], [], []
    for ti in range(1, N_TASKS + 1):
        z = zs[ti]
        # two independent K-rollout estimates per grasp: vA for the gap, |vA-vB|
        # for the matched same-state noise floor (same K, so noise is comparable).
        vA = np.array([vbatch(phys_g[j], ti, z, K) for j in range(len(phys_g))])
        vB = np.array([vbatch(phys_g[j], ti, z, K) for j in range(len(phys_g))])
        ostd = obs_g.std(0) + 1e-8
        On = obs_g / ostd
        for j in range(len(phys_g)):
            pt_rows.append((ti, float((vA[j] + vB[j]) / 2)))
            noise_rows.append((ti, float(abs(vA[j] - vB[j]))))   # same-state noise
        for a in range(len(phys_g)):
            for b in range(a + 1, len(phys_g)):
                d = float(np.linalg.norm(On[a] - On[b]) / np.sqrt(On.shape[1]))
                pair_rows.append((ti, d, float(abs(vA[a] - vA[b]))))  # cross-grasp gap
        print(f"  task{ti}: mean vhat {(vA.mean()+vB.mean())/2:.2f}  "
              f"same-state noise {np.mean([abs(vA[j]-vB[j]) for j in range(len(vA))]):.3f}  "
              f"(n={len(vA)}, K={K}x2)")

    pd.DataFrame(pair_rows, columns=["task", "dobs", "dv"]).to_parquet(
        REPO / f"analysis/value/repsep/gap_pairs_{args.method}.parquet")
    pd.DataFrame(pt_rows, columns=["task", "vhat"]).to_parquet(
        REPO / f"analysis/value/repsep/gap_points_{args.method}.parquet")
    pd.DataFrame(noise_rows, columns=["task", "noise"]).to_parquet(
        REPO / f"analysis/value/repsep/gap_noise_{args.method}.parquet")
    print(f"[gap:{args.method}] wrote gap_pairs/gap_points/gap_noise "
          f"({len(pair_rows)} pairs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

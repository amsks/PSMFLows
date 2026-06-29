#!/usr/bin/env python
"""scripts/coverage/crl_coverage.py — dataset-support coverage for CRL+FlowBC (JAX).

The FB/RLDP coverage comes from representation_profile and GCIQL's from
gciql_profile; CRL had no coverage because its phase-probe rollout
(scripts/probes/phase_probe_crl.py) keeps only eff/cube/grip signals, not the full
observation that the nearest-neighbour probe needs. This script rolls CRL out
keeping per-step observations, labels each step's instantaneous regime
(approach/pickup/carry via evals._profile_core), and writes the same
coverage.parquet schema (task/outcome/region/nn_dist) the other methods use.

Run under .venv-jax-cpu (jax + vendored ogbench). Importing phase_probe_crl
sets up sys.path for crl_flowbc + ogbench impls.

Usage:
    .venv-jax-cpu/bin/python scripts/coverage/crl_coverage.py \
        --checkpoint-dir results/factored-fb-crl-flowbc/sd000_20260525_080817 \
        --checkpoint-step 1000000 --seed 0 \
        --out analysis/probes/coverage_crl/s0_final --episodes 10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# importing phase_probe_crl injects ogbench impls + wandb_mode_shim onto sys.path
from scripts.probes.phase_probe_crl import EVAL_TASKS, load_agent, make_env  # noqa: E402


def rollout_episode_obs(agent, env, ep_seed: int, max_steps: int = 200) -> Dict[str, Any]:
    """One deterministic S0 episode, keeping per-step observation + signals."""
    import jax
    from evals.phase_probe import (episode_goal, episode_table_z,
                                   step_signals, _initial_obs)

    reset_obs, info = env.reset(seed=ep_seed)
    goal = info.get("goal")
    if goal is None:
        goal = episode_goal(env)
    goal_for_actor = np.asarray(goal, dtype=np.float32)
    table_z = episode_table_z(env)

    effs, cubes, grips, obses = [], [], [], []
    s0 = step_signals(info)
    effs.append(s0["eff"]); cubes.append(s0["cube"]); grips.append(s0["grip"])
    ep_success = bool(s0["success"])

    observation = _initial_obs(env, reset_obs, "S0")
    obses.append(np.asarray(observation, dtype=np.float32))
    rng = jax.random.PRNGKey(ep_seed)
    for _t in range(max_steps):
        rng, sub = jax.random.split(rng)
        obs_in = np.asarray(observation, dtype=np.float32)
        action = agent.sample_actions(observations=obs_in, goals=goal_for_actor,
                                      seed=sub, temperature=0.0)
        action = np.clip(np.asarray(action), -1.0, 1.0)
        observation, _r, terminated, truncated, info = env.step(action)
        s = step_signals(info)
        effs.append(s["eff"]); cubes.append(s["cube"]); grips.append(s["grip"])
        obses.append(np.asarray(observation, dtype=np.float32))
        ep_success = ep_success or s["success"]
        if terminated or truncated:
            break

    return {
        "obs": np.asarray(obses, dtype=np.float32),
        "eff": np.asarray(effs, dtype=np.float64),
        "cube": np.asarray(cubes, dtype=np.float64),
        "grip": np.asarray(grips, dtype=np.float64),
        "goal": episode_goal(env),
        "table_z": table_z,
        "success": ep_success,
        "length": len(effs),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--checkpoint-step", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--dataset-path", default="datasets/cube-single-play-v0")
    ap.add_argument("--tasks", default=None)
    args = ap.parse_args()

    import pandas as pd
    from evals.phase_probe import Thresholds, classify_phases, ensure_manip_env
    from evals._profile_core import probe_coverage, transport_mask
    from scripts.profiles.gciql_profile import _load_ref_obs

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    tasks = args.tasks.split(",") if args.tasks else EVAL_TASKS
    thr = Thresholds()
    ref_obs = _load_ref_obs(args.dataset_path, feature="obs")
    agent = load_agent(Path(args.checkpoint_dir).resolve(),
                       args.checkpoint_step, args.seed)
    print(f"[crl_coverage] seed={args.seed} step={args.checkpoint_step} "
          f"tasks={len(tasks)} ref_obs={ref_obs.shape}")

    cov_frames = []
    for task in tasks:
        env = make_env(task)
        try:
            ensure_manip_env(env)
            cov_eps = []
            for ep in range(args.episodes):
                sig = rollout_episode_obs(agent, env, args.seed * 100_000 + ep,
                                          max_steps=args.max_steps)
                cls = classify_phases(sig, thr)
                if cls["success"]:
                    outcome = "success"
                elif cls["fail_phase"] == "transport":
                    outcome = ("maintain_fail"
                               if cls["final_cube_lift"] < thr.delta_lift
                               else "transport_fail")
                else:
                    outcome = "other"
                cov_eps.append({
                    "obs": sig["obs"], "outcome": outcome,
                    "grip": sig["grip"], "cube": sig["cube"],
                    "table_z": sig["table_z"],
                    "transport_mask": transport_mask(sig, thr)})
            cov = probe_coverage(cov_eps, ref_obs, thr, feature="obs")
            cov.insert(0, "task", task)
            cov_frames.append(cov)
            print(f"  task={task}: "
                  f"{sum(e['outcome']=='success' for e in cov_eps)}/{args.episodes} success")
        finally:
            if hasattr(env, "close"):
                env.close()

    df = pd.concat(cov_frames, ignore_index=True) if cov_frames else pd.DataFrame()
    df.to_parquet(out / "coverage.parquet")
    (out / "metrics.json").write_text(json.dumps({
        "checkpoint_dir": args.checkpoint_dir, "checkpoint_step": args.checkpoint_step,
        "seed": args.seed, "episodes": args.episodes, "rows": len(df),
    }, indent=2, default=str))
    print(f"[crl_coverage] wrote {out}/coverage.parquet ({len(df)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

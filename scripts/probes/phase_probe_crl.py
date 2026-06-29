#!/usr/bin/env python
"""scripts/probes/phase_probe_crl.py — JAX-side phase-probe rollout for CRL+FlowBC.

Loads one seed's CRL+FlowBC checkpoint, runs N deterministic episodes on
each of the 5 cube-single eval tasks, captures per-step MuJoCo signals
(eff / cube / grip), and writes per_episode.parquet matching the FB
phase-probe schema.

Reuses the method-agnostic helpers from evals.phase_probe directly:
  - Thresholds, classify_phases  (classifier)
  - step_signals, episode_goal, episode_table_z  (signal extraction)
  - apply_scenario, _initial_obs  (M2 counterfactual support)
  - ensure_manip_env  (sanity)

The FB driver (scripts/probes/phase_probe.py) cannot be reused as-is because its
rollout loop calls agent.act(obs, z) — torch + FB-specific API. This
module mirrors its structure with the CRL sample_actions API instead.

Run inside .venv-jax-cpu (jax + ogbench). Invoked once per seed by
scripts/agents/crl_flowbc/phase.sh.

Usage:
    .venv-jax-cpu/bin/python scripts/probes/phase_probe_crl.py \\
        --checkpoint-dir results/factored-fb-crl-flowbc/sd000_20260525_080817 \\
        --checkpoint-step 700000 \\
        --seed 0 \\
        --out-dir analysis/probes/phase_probe_crl/s0_final \\
        --episodes 10
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "third_party" / "ogbench" / "impls"))
sys.path.insert(0, str(REPO_ROOT / "tools" / "wandb_mode_shim"))

# Silence gymnasium float-precision warnings around env construction.
warnings.filterwarnings("ignore", message=".*precision lowered.*")

import numpy as np
import pandas as pd

EVAL_TASKS = [
    "cube-single-play-singletask-task1-v0",
    "cube-single-play-singletask-task2-v0",
    "cube-single-play-singletask-task3-v0",
    "cube-single-play-singletask-task4-v0",
    "cube-single-play-singletask-task5-v0",
]
DEFAULT_SCENARIOS = ["S0", "S1", "S2"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint-dir", required=True,
                    help="run dir containing params_<step>.pkl files")
    ap.add_argument("--checkpoint-step", type=int, required=True,
                    help="step number to load (e.g. 700000, 1000000)")
    ap.add_argument("--seed", type=int, required=True,
                    help="seed id for output filename + RNGs")
    ap.add_argument("--out-dir", required=True,
                    help="output dir; will contain per_episode.parquet")
    ap.add_argument("--episodes", type=int, default=10,
                    help="rollouts per (task, scenario); paper protocol = 10")
    ap.add_argument("--scenarios", default=",".join(DEFAULT_SCENARIOS),
                    help="comma list of S0|S1|S2 (default all three)")
    ap.add_argument("--tasks", default=None,
                    help="comma list to subset EVAL_TASKS (default all 5)")
    return ap.parse_args()


def load_agent(checkpoint_dir: Path, checkpoint_step: int, seed: int):
    """Restore a CRLFlowBCAgent from a saved params_<step>.pkl.

    Uses OGBench's canonical restore path: build a template agent of the
    right architecture, then merge saved state in via flax serialization.
    """
    import jax
    import jax.numpy as jnp
    from crl_flowbc import CRLFlowBCAgent, get_config
    from utils.flax_utils import restore_agent

    # We need example obs + action of the right shape to call create(). The
    # easiest way is to build a temporary env, sample, then close.
    import ogbench
    tmp_env = ogbench.make_env_and_datasets(EVAL_TASKS[0], env_only=True)
    ex_obs_np, _ = tmp_env.reset()
    ex_obs = jnp.asarray(ex_obs_np, dtype=jnp.float32)[None]  # (1, obs_dim)
    ex_act = jnp.zeros((1, *tmp_env.action_space.shape), dtype=jnp.float32)
    cfg = dict(get_config())
    if hasattr(tmp_env, "close"):
        tmp_env.close()

    agent = CRLFlowBCAgent.create(seed, ex_obs, ex_act, cfg)
    # restore_agent expects the restore_path to glob to exactly one candidate
    # then appends /params_<step>.pkl. Pass an absolute path that matches one.
    agent = restore_agent(agent, str(checkpoint_dir), checkpoint_step)
    return agent


def make_env(task: str):
    """Construct the OGBench env bound to one cube-single eval task."""
    import ogbench
    env = ogbench.make_env_and_datasets(task, env_only=True)
    return env


def rollout_episode(agent, env, scenario: str, ep_seed: int,
                    max_steps: int = 200) -> Dict[str, Any]:
    """One deterministic episode under a scenario. Returns the signals dict
    that classify_phases consumes."""
    import jax
    from evals.phase_probe import (apply_scenario, episode_goal,
                                   episode_table_z, step_signals,
                                   _initial_obs)

    reset_obs, info = env.reset(seed=ep_seed)
    apply_scenario(env, scenario)
    goal = info.get("goal")
    if goal is None:
        # Fall back to xyz goal (S1/S2 scenarios where info["goal"] may be
        # stale; classify_phases only uses the xyz). For S0 with full goal,
        # we keep the obs-shaped goal for the actor.
        goal = episode_goal(env)
    goal_for_actor = np.asarray(goal, dtype=np.float32)
    table_z = episode_table_z(env)

    effs, cubes, grips = [], [], []
    s0 = step_signals(info)
    effs.append(s0["eff"]); cubes.append(s0["cube"]); grips.append(s0["grip"])
    ep_success = bool(s0["success"])

    observation = _initial_obs(env, reset_obs, scenario)
    rng = jax.random.PRNGKey(ep_seed)
    for _t in range(max_steps):
        rng, sub = jax.random.split(rng)
        obs_in = np.asarray(observation, dtype=np.float32)
        action = agent.sample_actions(
            observations=obs_in, goals=goal_for_actor,
            seed=sub, temperature=0.0)
        action = np.asarray(action)
        action = np.clip(action, -1.0, 1.0)
        observation, _reward, terminated, truncated, info = env.step(action)
        s = step_signals(info)
        effs.append(s["eff"]); cubes.append(s["cube"]); grips.append(s["grip"])
        ep_success = ep_success or s["success"]
        if terminated or truncated:
            break

    # For classify_phases we only need the xyz goal, not the full obs-goal.
    goal_xyz = episode_goal(env)
    return {
        "eff": np.asarray(effs, dtype=np.float64),
        "cube": np.asarray(cubes, dtype=np.float64),
        "grip": np.asarray(grips, dtype=np.float64),
        "goal": goal_xyz,
        "success": bool(ep_success),
        "table_z": table_z,
        "length": len(effs),
    }


def main() -> int:
    args = parse_args()
    ckpt_dir = Path(args.checkpoint_dir).resolve()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    scenarios = [s for s in args.scenarios.split(",") if s]
    tasks = (args.tasks.split(",") if args.tasks else EVAL_TASKS)

    print(f"[phase_probe_crl] seed={args.seed} step={args.checkpoint_step} "
          f"ckpt_dir={ckpt_dir}")
    print(f"[phase_probe_crl] out={out_dir} tasks={len(tasks)} "
          f"scenarios={scenarios} episodes={args.episodes}")

    from evals.phase_probe import (Thresholds, classify_phases,
                                   ensure_manip_env)
    agent = load_agent(ckpt_dir, args.checkpoint_step, args.seed)
    print("[phase_probe_crl] agent restored")
    thr = Thresholds()

    rows: List[Dict[str, Any]] = []
    for task in tasks:
        env = make_env(task)
        try:
            ensure_manip_env(env)
            for scenario in scenarios:
                for ep in range(args.episodes):
                    ep_seed = args.seed * 100_000 + ep
                    sig = rollout_episode(agent, env, scenario, ep_seed)
                    c = classify_phases(sig, thr)
                    rows.append({
                        "task": task, "scenario": scenario, "episode": ep,
                        "reached": c["reached"], "secured": c["secured"],
                        "success": c["success"],
                        "reached_raw": c["reached_raw"],
                        "secured_raw": c["secured_raw"],
                        "furthest_phase": c["furthest_phase"],
                        "fail_phase": c["fail_phase"],
                        "min_eff_cube_dist": c["min_eff_cube_dist"],
                        "max_cube_lift": c["max_cube_lift"],
                        # final_cube_lift needed to split fail_phase=transport
                        # into "maintain grasp" (cube on table) vs
                        # "transport to goal" (cube still held).
                        "final_cube_lift": c["final_cube_lift"],
                        "final_grip": c["final_grip"],
                        "final_cube_goal_dist": c["final_cube_goal_dist"],
                        "length": c["length"],
                    })
                print(f"  task={task} scenario={scenario}: "
                      f"{sum(r['success'] for r in rows[-args.episodes:])}/{args.episodes}")
        finally:
            if hasattr(env, "close"):
                env.close()

    df = pd.DataFrame(rows)
    out_parquet = out_dir / "per_episode.parquet"
    df.to_parquet(out_parquet)

    meta = out_dir / "metrics.json"
    meta.write_text(json.dumps({
        "checkpoint_dir": str(ckpt_dir),
        "checkpoint_step": args.checkpoint_step,
        "seed": args.seed,
        "episodes_per_cell": args.episodes,
        "tasks": tasks,
        "scenarios": scenarios,
        "rows": len(rows),
        "overall_success": float(df["success"].mean()) if len(df) else None,
    }, indent=2, default=str))
    print(f"[phase_probe_crl] wrote {out_parquet}: {len(rows)} rows; "
          f"overall_success={df['success'].mean():.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)

"""scripts/probes/representation_profile.py — per-checkpoint FB representation
profile. Caller loops seeds; the aggregate step runs all 7.

Usage:
    python scripts/probes/representation_profile.py \\
      --config RESULTS/.../__s3/.hydra/config.yaml \\
      --checkpoint RESULTS/.../__s3/checkpoints/final.pt \\
      --out analysis/probes/representation_profile/s3_final \\
      --data-path datasets --mujoco-gl glfw
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tasks", default=None, help="comma-separated; default=all")
    ap.add_argument("--n-episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--data-path", default=None)
    ap.add_argument("--mujoco-gl", default=None)
    ap.add_argument("--buffer-sample", type=int, default=20000)
    ap.add_argument("--topk", type=int, default=256)
    args = ap.parse_args()

    if args.mujoco_gl:
        os.environ["MUJOCO_GL"] = args.mujoco_gl

    import numpy as np
    import torch

    from evals.analysis import build_env_and_agent, load_cfg, load_checkpoint
    from evals.phase_probe import Thresholds
    from evals.representation_profile import run_representation_profile
    from envs.ogbench import ALL_TASKS, create_ogbench_env, get_relabel_fn
    from evals.ogbench import OGBenchEvaluator
    from data.ogbench import load_ogbench_dataset

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = load_cfg(args.config, device=args.device)
    if args.seed is not None:
        cfg.seed = args.seed
    if args.data_path is not None:
        cfg.data_path = args.data_path
    env, agent, _, _ = build_env_and_agent(cfg)
    load_checkpoint(agent, args.checkpoint, map_location=args.device)
    agent.eval() if hasattr(agent, "eval") else None
    if hasattr(env, "close"):
        env.close()

    tasks = (args.tasks.split(",") if args.tasks
             else list(ALL_TASKS.get(cfg.domain, [])))
    thr = Thresholds()
    frame_stack = int(getattr(cfg, "frame_stack", 1))

    buffer = load_ogbench_dataset(
        domain=cfg.domain, data_path=cfg.data_path,
        load_n_episodes=cfg.load_n_episodes, device=args.device,
        n_transitions=cfg.n_transitions, obs_type=cfg.obs_type,
        frame_stack=frame_stack)
    evaluator = OGBenchEvaluator(
        domain=cfg.domain, agent=agent, offline_buffer=buffer,
        relabel_size=cfg.eval_relabel_size, n_episodes=args.n_episodes,
        shift_reward=cfg.eval_shift_reward, obs_type=cfg.obs_type,
        frame_stack=frame_stack,
        seed=cfg.seed, device=args.device, save_videos=False,
        video_dir=None, use_wandb=False)

    def infer_z(task):
        return evaluator._infer_z(task)

    def make_env(task):
        e, _ = create_ogbench_env(task, seed=cfg.seed, obs_type=cfg.obs_type,
                                  frame_stack=frame_stack)
        return e

    def sample_buffer(n):
        b = buffer.sample(min(n, len(buffer)))
        return {"next_obs": b["next"]["observation"],
                "physics": b["next"]["physics"].detach().cpu().numpy(),
                "action": b["action"].detach().cpu().numpy()}

    def relabel_fn_for(task):
        return get_relabel_fn(cfg.domain, task)

    _goal_env, _ = create_ogbench_env(tasks[0], seed=cfg.seed,
                                      obs_type=cfg.obs_type)
    _tb = getattr(_goal_env.unwrapped, "_target_block", 0)

    def goal_for(task):
        e, _ = create_ogbench_env(task, seed=cfg.seed, obs_type=cfg.obs_type)
        try:
            tb = getattr(e.unwrapped, "_target_block", 0)
            return np.asarray(e.unwrapped.cur_task_info["goal_xyzs"][tb],
                              np.float64)
        finally:
            e.close()
    _goal_env.close()

    res = run_representation_profile(
        model=agent.model, agent=agent, infer_z=infer_z,
        make_env=make_env, sample_buffer=sample_buffer,
        relabel_fn_for=relabel_fn_for, goal_for=goal_for, tasks=tasks,
        n_episodes=args.n_episodes, thr=thr,
        buffer_sample=args.buffer_sample, topk=args.topk,
        seed=cfg.seed,
        coverage_feature=("cube" if cfg.obs_type == "pixels" else "obs"))

    res["value_landscape"].to_parquet(out / "value_landscape.parquet")
    res["value_steps"].to_parquet(out / "value_steps.parquet")
    res["z_decoding"].to_parquet(out / "z_decoding.parquet")
    res["b_resolution"].to_parquet(out / "b_resolution.parquet")
    res["coverage"].to_parquet(out / "coverage.parquet")
    (out / "metrics.json").write_text(json.dumps({
        "config": args.config, "checkpoint": args.checkpoint,
        "tasks": tasks, "buffer_sample": args.buffer_sample,
        "topk": args.topk,
        "q_aggregation": "ensemble-mean over F heads",
    }, indent=2, default=str))
    print(f"[representation_profile] done -> {out}")


if __name__ == "__main__":
    main()

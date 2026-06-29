"""scripts/probes/phase_probe.py — test where FB fails in cube episodes.

Loads one run's .hydra/config.yaml + a checkpoint, runs M1 (phase
decomposition) + M2 (S0/S1/S2 counterfactuals) over the cube tasks, and
writes per-episode parquet, a summary + hypothesis readout, and plots.
Caller loops seeds/checkpoints. Run on Linux with --mujoco-gl egl.

Usage:
    python scripts/probes/phase_probe.py \\
      --config RESULTS/.../__s3/.hydra/config.yaml \\
      --checkpoint RESULTS/.../__s3/checkpoints/final.pt \\
      --out analysis/legacy/phase_probe/s3_final --mujoco-gl egl
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
    ap.add_argument("--n-episodes", type=int, default=20)
    ap.add_argument("--scenarios", default="S0,S1,S2")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--data-path", default=None,
                    help="override cfg.data_path (offline buffer root); "
                         "the saved config may point at a cluster path")
    ap.add_argument("--mujoco-gl", default=None,
                    help="osmesa|glfw|egl; sets MUJOCO_GL before env import")
    ap.add_argument("--eps-reach", type=float, default=0.06)
    ap.add_argument("--delta-lift", type=float, default=0.03)
    ap.add_argument("--k-steps", type=int, default=5)
    ap.add_argument("--tau-grip", type=float, default=0.5)
    args = ap.parse_args()

    if args.mujoco_gl:
        os.environ["MUJOCO_GL"] = args.mujoco_gl

    from evals.analysis import build_env_and_agent, load_cfg, load_checkpoint
    from evals.phase_probe import (
        Thresholds, plot_phase_histogram, plot_scenario_success,
        run_phase_probe, write_summary_md,
    )
    from envs.ogbench import ALL_TASKS, create_ogbench_env
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
    # build_env_and_agent only needed env for obs/action dims; per-task
    # envs are created later. Release this one's renderer/physics resources.
    if hasattr(env, "close"):
        env.close()

    tasks = (args.tasks.split(",") if args.tasks
             else list(ALL_TASKS.get(cfg.domain, [])))
    scenarios = [s for s in args.scenarios.split(",") if s]
    thr = Thresholds(eps_reach=args.eps_reach, delta_lift=args.delta_lift,
                     k_steps=args.k_steps, tau_grip=args.tau_grip)
    frame_stack = int(getattr(cfg, "frame_stack", 1))

    buffer = load_ogbench_dataset(
        domain=cfg.domain, data_path=cfg.data_path,
        load_n_episodes=cfg.load_n_episodes, device=args.device,
        n_transitions=cfg.n_transitions, obs_type=cfg.obs_type,
        frame_stack=frame_stack,
    )
    evaluator = OGBenchEvaluator(
        domain=cfg.domain, agent=agent, offline_buffer=buffer,
        relabel_size=cfg.eval_relabel_size, n_episodes=args.n_episodes,
        shift_reward=cfg.eval_shift_reward, obs_type=cfg.obs_type,
        frame_stack=frame_stack,
        seed=cfg.seed, device=args.device, save_videos=False,
        video_dir=None, use_wandb=False,
    )

    def infer_z(task):
        # TD-MPC2 (and any agent exposing eval_context) reads its goal context
        # from the task env (cube goal xyz), not from the offline buffer; build
        # a short-lived env to extract it. FB/RLDP/CRL fall back to _infer_z.
        if hasattr(agent, "eval_context"):
            e, _ = create_ogbench_env(task, seed=cfg.seed, obs_type=cfg.obs_type,
                                      frame_stack=frame_stack)
            try:
                z, _ = agent.eval_context(env=e, domain=cfg.domain, task=task)
            finally:
                if hasattr(e, "close"):
                    e.close()
            return z
        z, _ = evaluator._infer_z(task)
        return z

    def make_env(task):
        e, _ = create_ogbench_env(task, seed=cfg.seed, obs_type=cfg.obs_type,
                                  frame_stack=frame_stack)
        return e

    per_ep, summary, _hist = run_phase_probe(
        agent=agent, infer_z=infer_z, make_env=make_env, tasks=tasks,
        scenarios=scenarios, n_episodes=args.n_episodes, thr=thr)

    per_ep.to_parquet(out / "per_episode.parquet")
    summary.to_parquet(out / "summary.parquet")
    write_summary_md(summary, out / "summary.md")
    plot_phase_histogram(per_ep, out / "phase_histogram.png")
    plot_scenario_success(summary, out / "scenario_success.png")
    (out / "metrics.json").write_text(json.dumps({
        "config": args.config, "checkpoint": args.checkpoint,
        "tasks": tasks, "scenarios": scenarios,
        "thresholds": vars(thr), "n_episodes": args.n_episodes,
    }, indent=2, default=str))
    print(f"[phase_probe] done -> {out}")


if __name__ == "__main__":
    main()

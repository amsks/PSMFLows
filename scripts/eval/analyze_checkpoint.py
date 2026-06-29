"""scripts/eval/analyze_checkpoint.py — re-run ONE checkpoint for behavior analysis.

Loads a run's .hydra/config.yaml + a checkpoint .pt, then writes trajectory
logs, z/task-conditioning probes, and (optionally) videos. Caller scripts any
looping over checkpoints.

Usage:
    python scripts/eval/analyze_checkpoint.py \\
      --config RESULTS/.../__s3/.hydra/config.yaml \\
      --checkpoint RESULTS/.../__s3/checkpoints/final.pt \\
      --out analysis/checkpoint_runs/s3_final --what traj,probe
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
    ap.add_argument("--what", default="traj,probe",
                    help="comma-separated subset of traj,probe,video")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--mujoco-gl", default=None,
                    help="osmesa|glfw|egl; sets MUJOCO_GL before env import")
    args = ap.parse_args()

    if args.mujoco_gl:
        os.environ["MUJOCO_GL"] = args.mujoco_gl

    import numpy as np
    import torch

    from evals.analysis import (
        build_env_and_agent, load_cfg, load_checkpoint,
        rollout_with_trajectory, save_trajectories, z_interp,
        z_probe_cross_task,
    )
    from envs.ogbench import ALL_TASKS, create_ogbench_env
    from evals.ogbench import OGBenchEvaluator
    from data.ogbench import load_ogbench_dataset

    what = set(x for x in args.what.split(",") if x)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = load_cfg(args.config, device=args.device)
    if args.seed is not None:
        cfg.seed = args.seed
    env, agent, _, _ = build_env_and_agent(cfg)
    load_checkpoint(agent, args.checkpoint, map_location=args.device)
    agent.eval() if hasattr(agent, "eval") else None
    # build_env_and_agent only needed env for obs/action dims; per-task
    # envs are created later. Release this one's renderer/physics resources.
    if hasattr(env, "close"):
        env.close()

    tasks = (args.tasks.split(",") if args.tasks
             else list(ALL_TASKS.get(cfg.domain, [])))

    # Reuse the evaluator only for its z-inference machinery.
    buffer = load_ogbench_dataset(
        domain=cfg.domain, data_path=cfg.data_path,
        load_n_episodes=cfg.load_n_episodes, device=args.device,
        n_transitions=cfg.n_transitions,
    )
    evaluator = OGBenchEvaluator(
        domain=cfg.domain, agent=agent, offline_buffer=buffer,
        relabel_size=cfg.eval_relabel_size, n_episodes=args.n_episodes,
        shift_reward=cfg.eval_shift_reward, obs_type=cfg.obs_type,
        seed=cfg.seed, device=args.device, save_videos=False,
        video_dir=None, use_wandb=False,
    )

    def infer_z(task):
        z, _ = evaluator._infer_z(task)
        return z

    summary = {"config": args.config, "checkpoint": args.checkpoint,
               "tasks": tasks, "per_task": {}}

    if "traj" in what or "video" in what:
        for task in tasks:
            tenv, _ = create_ogbench_env(task, seed=cfg.seed,
                                         obs_type=cfg.obs_type)
            try:
                z = infer_z(task)
                res = rollout_with_trajectory(
                    tenv, agent, args.n_episodes, z,
                    record=("video" in what))
            finally:
                tenv.close()
            if "traj" in what:
                save_trajectories(res, task=task, out_dir=out)
            if "video" in what and res["frames"]:
                try:
                    _save_videos(out, task, res["frames"])
                except Exception as e:  # noqa: BLE001 - render is best-effort
                    print(f"  [warn] video write failed for {task}: {e} "
                          f"(traj/probe unaffected)")
            summary["per_task"][task] = {
                "success": float(np.mean(res["success"])),
            }

    if "probe" in what:
        def make_env(task):
            e, _ = create_ogbench_env(task, seed=cfg.seed,
                                      obs_type=cfg.obs_type)
            return e
        probe_dir = out / "probes"
        probe_dir.mkdir(parents=True, exist_ok=True)
        df = z_probe_cross_task(tasks, agent, infer_z, make_env,
                                n_episodes=args.n_episodes)
        df.to_parquet(probe_dir / "cross_task.parquet")
        _heatmap(df, probe_dir / "cross_task.png")

        # z interpolation between the first two tasks' inferred z
        if len(tasks) >= 2:
            ienv = make_env(tasks[1])
            try:
                idf = z_interp(infer_z(tasks[0]), infer_z(tasks[1]), agent,
                               ienv, n_alpha=11, n_episodes=args.n_episodes)
            finally:
                if hasattr(ienv, "close"):
                    ienv.close()
            idf.to_parquet(probe_dir / "z_interp.parquet")
            _lineplot(idf, probe_dir / "z_interp.png",
                      title=f"z: {tasks[0]} -> {tasks[1]} (env={tasks[1]})")

    (out / "metrics.json").write_text(json.dumps(summary, indent=2,
                                                 default=str))
    print(f"[analyze_checkpoint] done -> {out}")


def _save_videos(out: Path, task: str, frames) -> None:
    import imageio
    import numpy as np
    vid_dir = out / "videos"
    vid_dir.mkdir(parents=True, exist_ok=True)
    for k, ep in enumerate(frames):
        imageio.mimsave(vid_dir / f"{task}_ep{k}.gif",
                        np.stack(ep), format="GIF", fps=30, loop=0)


def _heatmap(df, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pivot = df.pivot(index="env_task", columns="z_task", values="success")
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(pivot.values, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=7)
    ax.set_xlabel("z task")
    ax.set_ylabel("env task")
    fig.colorbar(im, ax=ax, label="success")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _lineplot(df, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["alpha"], df["success"], marker="o", label="success")
    ax.plot(df["alpha"], df["reward"], marker="s", label="reward")
    ax.set_xlabel("alpha (z_a -> z_b)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()

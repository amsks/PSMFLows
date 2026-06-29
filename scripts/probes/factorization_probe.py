"""Factorization-ablation + readout-ceiling probe on a frozen FB checkpoint.

Answers "why does FB fail at transport: data (B1), FB's use of data (B2), or the
bilinear FᵀB factorization (B3)?" by measuring, on frozen embeddings, the
readout ceiling R^2(feature -> cube-to-goal distance d) — the paper's R^2(B->d)
— per manipulation phase, for a geometry reference vs the forward branch's
left-encoder vs the backward map B, with linear vs MLP readouts.

Works for BOTH state and pixel FB: states are sampled from the offline buffer
(load_ogbench_dataset handles frame-stacking + normalization), so `next_obs` is
model-ready and `next_physics` carries the ground-truth cube position (cols
14:17) for the label. The "raw" reference is the state vector for state FB, or
the ground-truth cube xyz for pixels (raw images need a conv, not a flat MLP).

Static geometry (d) is the target: in PLAY data, future-outcome targets are
exploration-driven and ~unpredictable. Goal-independent embeddings (B, left_enc)
are computed once; only d depends on the task goal, so all tasks are swept
cheaply. Run under .venv (torch). macOS: prefix MUJOCO_GL=glfw. Example:

  MUJOCO_GL=glfw .venv/bin/python -m scripts.probes.factorization_probe \
    --config <run>/.hydra/config.yaml --checkpoint <run>/checkpoints/final.pt \
    --data-path datasets --out analysis/probes/factorization_probe/s3 \
    --n-states 6000 --load-episodes 300 --tasks all --device cpu
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from evals.training_value import region_labels, cube_to_goal_dist
from evals.factorization_probe import readout_ceiling_table, classify_bucket_regression
from scripts.value.training_value_profile import CUBE_SLICE, GRIP_QPOS_IDX, TASK_TMPL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-path", default="datasets")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-states", type=int, default=6000)
    ap.add_argument("--load-episodes", type=int, default=0, help="cap episodes loaded (0=cfg); use ~200-300 for pixels (memory)")
    ap.add_argument("--tasks", default="all", help="comma-sep task ids (1-N) or 'all'")
    ap.add_argument("--dump-embeddings", action="store_true", help="also save B/left_enc/cube/region/goals npz for figures")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    from evals.analysis import build_env_and_agent, load_cfg, load_checkpoint
    from envs.ogbench import create_ogbench_env, ALL_TASKS
    from evals.phase_probe import Thresholds
    from data.ogbench import load_ogbench_dataset

    cfg = load_cfg(args.config, device=args.device)
    if args.data_path:
        cfg.data_path = args.data_path
    if args.load_episodes:
        cfg.load_n_episodes = args.load_episodes
    frame_stack = int(getattr(cfg, "frame_stack", 1))
    is_pixel = cfg.obs_type == "pixels"

    env, agent, _, _ = build_env_and_agent(cfg)
    load_checkpoint(agent, args.checkpoint, map_location=args.device)
    agent.model.eval()
    if hasattr(env, "close"):
        env.close()

    thr = Thresholds()
    e0, _ = create_ogbench_env(TASK_TMPL.format(n=1), seed=cfg.seed,
                               obs_type=cfg.obs_type, frame_stack=frame_stack)
    table_z = float(e0.unwrapped.cur_task_info["init_xyzs"][
        int(getattr(e0.unwrapped, "_target_block", 0) or 0)][2])
    e0.close()

    # Sample model-ready next-states from the offline buffer (handles frame-stack
    # + normalization for both state and pixel); next-physics gives cube + phase.
    buffer = load_ogbench_dataset(domain=cfg.domain, data_path=cfg.data_path,
                                  load_n_episodes=cfg.load_n_episodes, device=args.device,
                                  n_transitions=cfg.n_transitions, obs_type=cfg.obs_type,
                                  frame_stack=frame_stack)
    smp = buffer.sample(min(args.n_states, len(buffer)))
    next_obs = smp["next"]["observation"].to(args.device).float()
    physics = smp["next"]["physics"].detach().cpu().numpy()
    cube = physics[:, CUBE_SLICE].astype(np.float64)
    grip = np.clip(physics[:, GRIP_QPOS_IDX] / 0.8, 0, 1)
    region = [str(r) for r in region_labels(grip, physics[:, 16] - table_z, thr)]

    # Frozen GOAL-INDEPENDENT embeddings: backward map B and forward left-encoder.
    m = agent.model
    with torch.no_grad():
        B = m.backward_map(next_obs).cpu().numpy()
        left = m._left_encoder(m._fw_encoder(m._normalize(next_obs))).cpu().numpy()
    # "raw" reference = full obs vector (state) or ground-truth cube xyz (pixels:
    # raw images are not a fair flat-MLP baseline). Both establish "geometry is
    # available in principle" so that a low B-score isolates B (not B1/data).
    raw = cube.copy() if is_pixel else next_obs.detach().cpu().numpy().astype(np.float64)
    feats = {"raw": raw, "B": B, "left_enc": left}

    all_tasks = list(ALL_TASKS[cfg.domain])
    task_ids = (list(range(1, len(all_tasks) + 1)) if args.tasks == "all"
                else [int(t) for t in args.tasks.split(",")])

    ceilings, per_task, goals = [], {}, []
    for ti in task_ids:
        e, _ = create_ogbench_env(TASK_TMPL.format(n=ti), seed=cfg.seed,
                                  obs_type=cfg.obs_type, frame_stack=frame_stack)
        g = np.asarray(e.unwrapped.cur_task_info["goal_xyzs"][
            int(getattr(e.unwrapped, "_target_block", 0) or 0)], np.float64)
        e.close()
        goals.append(g)
        frame = pd.DataFrame({"region": region, "d": cube_to_goal_dist(cube, g)})
        c = readout_ceiling_table(frame, feats, targets=("d",), seed=cfg.seed)
        c["task"] = f"task{ti}"
        ceilings.append(c)
        per_task[f"task{ti}"] = classify_bucket_regression(c, target="d")["bucket"]

    if args.dump_embeddings:
        np.savez(out / "embeddings.npz", B=B, left_enc=left, cube=cube,
                 region=np.asarray(region), goals=np.asarray(goals),
                 obs_type=cfg.obs_type)

    ceiling = pd.concat(ceilings, ignore_index=True)
    pooled = (ceiling.groupby(["feature", "target", "region", "kind"])["score"]
              .mean().reset_index())
    verdict = classify_bucket_regression(pooled, target="d")

    ceiling.to_parquet(out / "readout_ceiling.parquet")
    pooled.to_parquet(out / "readout_ceiling_pooled.parquet")
    (out / "verdict.json").write_text(json.dumps(
        {**verdict, "obs_type": cfg.obs_type, "per_task": per_task}, indent=2))
    _write_story(out, pooled, verdict, per_task, cfg.obs_type)
    print(f"[factorization_probe:{cfg.obs_type}] verdict={verdict['bucket']} "
          f"| per-task={per_task} :: {verdict['why']}")


def _write_story(out, pooled, verdict, per_task, obs_type):
    tr = pooled[(pooled.region == "transport") & (pooled.target == "d")]
    lines = [f"# Factorization probe ({obs_type}) — R^2(feature -> cube-to-goal d)", "",
             f"**Verdict: {verdict['bucket']}** — {verdict['why']}", "",
             f"Per-task verdicts: {per_task}", "",
             "## Transport-phase readout ceilings (mean R^2 over tasks)", "",
             "| feature | kind | mean R^2 |", "| :-- | :-- | :-- |"]
    for _, r in tr.sort_values(["feature", "kind"]).iterrows():
        lines.append(f"| {r.feature} | {r.kind} | {r.score:.3f} |")
    (out / "story.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

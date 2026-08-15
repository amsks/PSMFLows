"""Re-evaluate a saved checkpoint at high episode count, with a proper interval.

Why this exists: in-loop eval runs `eval_episodes` (50 in our Stage-C launches), which
puts a 95% CI of about +/-0.115 on any single success number -- wide enough that the
whole 0.22..0.36 band a run wanders through is one flat line plus noise. Comparing a
Stage-C score against its own behavior-cloning prior needs both sides measured tightly,
so this reloads weights and evaluates with N episodes and no training.

Two agents matter here and both work unchanged:
  * psmflow -- needs the reward-inferred task vector, so `infer_eval_z` is applied
    exactly as main.py's eval block does (same relabel size, same reward shift).
  * fql with bc_only -- `sample_actions` draws a fresh N(0, I) latent every step and
    decodes it, i.e. running the frozen Stage-A flow IS the per-step-prior BC control.
    No z inference, no preimages.

Reports mean success with a Wilson 95% interval (the normal approximation misbehaves
near 0 and 1, and the BC control could land anywhere), plus the per-episode successes
so several runs can be pooled later.

Run: MUJOCO_GL=egl .venv/bin/python tools/eval_checkpoint.py \
    agent=psmflow agent.flow_ckpt_path=<flow_dir> agent.flow_ckpt_epoch=500000 \
    agent.preimage_path=<npz> env_name=cube-single-play-singletask-v0 \
    restore_path=<run_dir> restore_epoch=500000 eval_episodes=500
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils.xla_guard  # noqa: F401  -- MUST precede jax (see module docstring)

import hydra
import ml_collections
import numpy as np
from omegaconf import OmegaConf

from agents import agents
from envs.env_utils import make_env_and_datasets
from main import _lists_to_tuples
from utils.datasets import Dataset
from utils.evaluation import evaluate
from utils.flax_utils import restore_agent
from utils.log_utils import write_report


def wilson(k, n, z=1.96):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(centre - half, 4), round(centre + half, 4))


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    # Dataset.sample() draws from the GLOBAL numpy stream, so the task-vector relabel
    # batch below (and anything else sampling here) was entropy-dependent: two runs of
    # this tool on the same checkpoint inferred w from different transitions. Pin it.
    np.random.seed(int(cfg.seed))

    env, eval_env, train_dataset, _ = make_env_and_datasets(cfg.env_name, frame_stack=cfg.frame_stack)
    ds = Dataset.create(**train_dataset)
    config = ml_collections.ConfigDict(_lists_to_tuples(OmegaConf.to_container(cfg.agent, resolve=True)))
    name = config["agent_name"]

    ex = ds.sample(1)
    agent = agents[name].create(cfg.seed, ex["observations"], ex["actions"], config)
    assert cfg.restore_path is not None, "needs a trained checkpoint (restore_path)"
    agent = restore_agent(agent, cfg.restore_path, cfg.restore_epoch)

    # Task vector, identical to main.py's eval block. fql/bc_only has no infer_eval_z
    # and acts straight off observations.
    if hasattr(agent, "infer_eval_z"):
        n_relabel = min(ds.size, int(cfg.get("eval_relabel_size", 10000)))
        zb = ds.sample(n_relabel)
        shift = float(cfg.get("eval_reward_shift", 1.0))
        agent = agent.infer_eval_z(zb["next_observations"], zb["rewards"] + shift)

    n_ep = int(cfg.eval_episodes)
    # Same seed convention as training eval: the episode-init stream starts where the
    # in-loop eval's does, and (since the 08-14 seeding fix) the action-noise stream is
    # pinned to cfg.seed too, so re-running this tool on a checkpoint reproduces its
    # number exactly. It does NOT reproduce an in-loop eval step-for-step -- the in-loop
    # relabel batch is drawn from a global numpy stream that training has advanced --
    # which is what the previous docstring claimed.
    info, trajs, _ = evaluate(agent=agent, env=eval_env, config=config,
                              num_eval_episodes=n_ep, num_video_episodes=0,
                              seed=int(cfg.seed))

    # `evaluate` averages info fields; recover per-episode successes from the
    # trajectories so the interval is computed on counts, not on a mean.
    per_ep = [float(np.max(np.asarray(t["info"][-1].get("success", 0.0)))) if "info" in t
              else None for t in trajs]
    per_ep = [p for p in per_ep if p is not None]
    if per_ep:
        k, n = int(sum(p > 0.5 for p in per_ep)), len(per_ep)
    else:  # fall back to the averaged field if the env does not expose per-step success
        k, n = int(round(float(info["success"]) * n_ep)), n_ep

    # Self-identifying metadata. The same checkpoint evaluated in three acting modes
    # produced three JSONs distinguishable only by their FILENAME, which is exactly how a
    # lambda-rank number gets quoted as the deployed one six weeks later. Record what was
    # actually run, in the report.
    ac = config.get("action_critic", {}) or {}
    rank_k = int(ac.get("eval_rank_k", 0) or 0)
    if not ac.get("enabled", False):
        mode = "decode(actor latent) — action branch disabled"
    elif rank_k == 1:
        mode = "decode-only control: one actor draw, decoded, no residual, no selection"
    elif rank_k > 1:
        mode = f"lambda-rank: {rank_k} decoded candidates scored by Q_a, argmax, no residual"
    else:
        mode = (f"deployed: actor draw + eps-bounded residual "
                f"(residual_eps={config.get('residual_eps')})")

    lo, hi = wilson(k, n)
    report = {
        "env": cfg.env_name,
        "agent": name,
        "acting": config.get("acting"),
        "acting_mode": mode,
        "action_critic": {"enabled": bool(ac.get("enabled", False)),
                          "eval_rank_k": rank_k,
                          "residual_eps": config.get("residual_eps"),
                          "fb_graft": bool(ac.get("fb_graft", False))},
        "dataset_fraction": float(cfg.get("dataset_fraction", 1.0)),
        "dataset_fraction_seed": int(cfg.get("dataset_fraction_seed", 0)),
        "flow_ckpt_path": str(config.get("flow_ckpt_path")),
        "preimage_path": str(config.get("preimage_path")),
        "restore_path": str(cfg.restore_path),
        "restore_epoch": int(cfg.restore_epoch),
        "seed": int(cfg.seed),
        "num_episodes": n,
        "num_success": k,
        "success": round(k / n, 4),
        "wilson95": [lo, hi],
        "half_width": round((hi - lo) / 2, 4),
        "eval_success_field": round(float(info["success"]), 4),
        "per_episode_success": [int(p > 0.5) for p in per_ep],
    }
    print(f"\n{name} [{cfg.env_name}] {k}/{n} = {report['success']:.3f}  "
          f"95% CI [{lo:.3f}, {hi:.3f}]\n  mode: {mode}")
    write_report(report, cfg, "eval_checkpoint.json")


if __name__ == "__main__":
    main()

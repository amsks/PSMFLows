"""Support budget actually spent: distance of EXECUTED actions to the local data.

The L1/W4 tradeoff curve's x-axis. Rolls N episodes from a checkpoint, and for every
executed action reports the minimum distance to the actions of the k nearest dataset
states, under the shared neighbourhood protocol in `utils/geometry` (per-dimension
standardized observations, k=32; action distances normalized by the dataset's mean |a|).

Also computes the data-matching-itself calibration baseline: the same statistic for
recorded dataset actions (each excluded from its own neighborhood), which defines what
"close" means for real data. The report quotes the fraction of executed actions beyond
that baseline's p95 -- the fraction of steps genuinely outside the local action cloud.

Run:
  MUJOCO_GL=egl .venv/bin/python tools/diag_action_distance.py agent=latentrl \
      env_name=cube-single-play-singletask-v0 \
      agent.flow_ckpt_path=<flow_dir> agent.flow_ckpt_epoch=500000 \
      agent.preimage_path=<npz> agent.use_point_preimage=true \
      agent.residual_eps=<eps> restore_path=<run_dir> restore_epoch=<step> \
      report_out=/data-local/amsks/PSMFLows/logs/<name>.json
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils.xla_guard  # noqa: F401  -- MUST precede jax (see module docstring)

import hydra
import jax
import ml_collections
import numpy as np
from omegaconf import OmegaConf
from agents import agents
from envs.env_utils import make_env_and_datasets
from main import _lists_to_tuples
from utils.datasets import Dataset
from utils.flax_utils import restore_agent
from utils.geometry import INDEX_ROWS, K_NEIGHBOURS, NeighbourIndex
from utils.log_utils import write_report

BASELINE_ROWS = 512


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    n_episodes = int(cfg.get("distance_episodes", 20))
    rng_np = np.random.default_rng(int(cfg.seed))

    env, eval_env, train_dataset, _ = make_env_and_datasets(
        cfg.env_name, frame_stack=cfg.frame_stack)
    ds = Dataset.create(**train_dataset)
    config = ml_collections.ConfigDict(
        _lists_to_tuples(OmegaConf.to_container(cfg.agent, resolve=True)))

    ex = ds.sample(1)
    agent = agents[config["agent_name"]].create(
        cfg.seed, ex["observations"], ex["actions"], config)
    assert cfg.restore_path is not None
    agent = restore_agent(agent, cfg.restore_path, cfg.restore_epoch)

    obs_all = np.asarray(ds["observations"])
    act_all = np.asarray(ds["actions"])

    # Shared geometry protocol (utils/geometry): standardized obs, k=32, seeded index.
    index = NeighbourIndex(obs_all, act_all, index_rows=INDEX_ROWS, seed=int(cfg.seed))
    a_scale = index.action_scale

    # Baseline: how well does REAL data match itself under the identical statistic,
    # each query row excluded from its own neighbourhood.
    brow = rng_np.choice(index.rows, size=BASELINE_ROWS, replace=False)
    base = index.min_action_dist(obs_all[brow], act_all[brow], exclude_rows=brow)
    base_p95 = float(np.percentile(base, 95))

    # Rollouts.
    eval_env.reset(seed=int(cfg.seed))
    rng = jax.random.PRNGKey(int(cfg.seed))
    dists, successes, ep_lens = [], [], []
    for ep in range(n_episodes):
        obs, _ = eval_env.reset()
        done, t = False, 0
        ep_obs, ep_act = [], []
        info = {}
        while not done:
            rng, key = jax.random.split(rng)
            act = np.asarray(agent.sample_actions(obs, seed=key))
            ep_obs.append(np.asarray(obs))
            ep_act.append(act)
            obs, r, terminated, truncated, info = eval_env.step(act)
            t += 1
            done = terminated or truncated
        dists.append(index.min_action_dist(np.stack(ep_obs), np.stack(ep_act)))
        successes.append(float(np.max(info.get("success", 0.0))))
        ep_lens.append(t)
    d = np.concatenate(dists)

    report = {
        "env": cfg.env_name,
        "agent": config["agent_name"],
        "residual_eps": float(config.get("residual_eps", 0.0)),
        "restore_path": str(cfg.restore_path),
        "restore_epoch": int(cfg.restore_epoch),
        "seed": int(cfg.seed),
        "n_episodes": n_episodes,
        "n_steps": int(d.size),
        "geometry": index.protocol(),
        "action_scale_mean_abs": a_scale,
        "executed_dist": {"mean": float(d.mean()), "median": float(np.median(d)),
                          "p90": float(np.percentile(d, 90)),
                          "p99": float(np.percentile(d, 99))},
        "baseline_dist": {"mean": float(base.mean()), "median": float(np.median(base)),
                          "p95": base_p95, "rows": BASELINE_ROWS},
        "frac_beyond_baseline_p95": float((d > base_p95).mean()),
        "success_rate_over_rollouts": float(np.mean(successes)),
        "mean_episode_length": float(np.mean(ep_lens)),
    }
    write_report(report, cfg, "diag_action_distance.json")


if __name__ == "__main__":
    main()

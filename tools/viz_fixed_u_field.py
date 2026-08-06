"""Why fixed-u policies fail: the vector field a = G(xy, u) over the maze, per u.

Companion to tools/latent_reachability.py (which found ZERO of 233 latents reach the
pointmaze goal) and to the latent-coherence stat (within-episode preimage variance ~= the
marginal — the expert's route is latent white noise, so no fixed u encodes a route).
This shows the mechanism: for a handful of fixed latents, decode the action at every cell
of an xy grid and quiver it over the dataset support, overlaying the actual fixed-u
rollout from the eval start. Attractors / constant headings that miss the goal are the
failure made visible. Also prints per-u heading circular variance and stall fraction,
and a per-cell multimodality probe (angle spread of 512 prior decodes at sample cells).

Run: MUJOCO_GL=egl OGBENCH_DATASET_DIR=/var/local/amsks/ogbench \
    .venv/bin/python tools/viz_fixed_u_field.py agent=fql \
    env_name=pointmaze-medium-navigate-singletask-task1-v0 \
    restore_path='/var/local/amsks/exp/<flow_run_dir>' restore_epoch=500000
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils.xla_guard  # noqa: F401  -- MUST precede jax (see module docstring)

import hydra
import jax
import ml_collections
import numpy as np
from omegaconf import OmegaConf

from agents.fql import FQLAgent
from envs.env_utils import make_env_and_datasets
from main import _lists_to_tuples
from utils.flax_utils import restore_agent
from utils.log_utils import write_report

GRID_N = 25
ROLLOUT_STEPS = 1000
US = [(-3.0, -3.0), (3.0, -3.0), (0.0, 0.0), (-3.0, 3.0), (3.0, 3.0), (2.0, 2.5)]
PROBE_CELLS = 4      # per-cell multimodality probe
PROBE_DRAWS = 512


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    env, eval_env, train_dataset, _ = make_env_and_datasets(cfg.env_name, frame_stack=cfg.frame_stack)
    obs_all = train_dataset['observations']
    ex_act = train_dataset['actions'][:1]
    assert obs_all.shape[-1] == 2, 'field viz assumes obs = (x, y)'
    assert cfg.agent.agent_name == 'fql', 'run with agent=fql (Stage-A flow ckpt)'
    agent_cfg = ml_collections.ConfigDict(_lists_to_tuples(OmegaConf.to_container(cfg.agent, resolve=True)))
    agent = FQLAgent.create(cfg.seed, obs_all[:1], ex_act, agent_cfg)
    assert cfg.restore_path is not None, 'needs the trained Stage-A flow (restore_path)'
    agent = restore_agent(agent, cfg.restore_path, cfg.restore_epoch)
    onestep = jax.jit(lambda o, u: agent.network.select('actor_onestep_flow')(o, u))

    lo, hi = obs_all.min(0), obs_all.max(0)
    xs = np.linspace(lo[0], hi[0], GRID_N)
    ys = np.linspace(lo[1], hi[1], GRID_N)
    gx, gy = np.meshgrid(xs, ys)
    cells = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)

    ob0, info0 = eval_env.reset(seed=int(cfg.seed) * 1000)
    goal = np.asarray(info0.get('goal'))[:2]

    fields, trajs, stats = [], [], []
    for u in US:
        ub = np.broadcast_to(np.asarray(u, np.float32), (len(cells), 2))
        a = np.clip(np.asarray(onestep(cells, ub)), -1, 1)
        fields.append(a)
        ang = np.arctan2(a[:, 1], a[:, 0])
        # circular variance: 0 = one global heading, 1 = uniform directions
        circ_var = 1.0 - float(np.hypot(np.cos(ang).mean(), np.sin(ang).mean()))
        stall = float((np.linalg.norm(a, axis=1) < 0.1).mean())
        ob, _ = eval_env.reset(seed=int(cfg.seed) * 1000)
        path = [np.asarray(ob)[:2]]
        for _ in range(ROLLOUT_STEPS):
            act = np.clip(np.asarray(onestep(np.asarray(ob)[None],
                                             np.asarray(u, np.float32)[None]))[0], -1, 1)
            ob, _, term, trunc, _ = eval_env.step(act)
            path.append(np.asarray(ob)[:2])
            if term or trunc:
                break
        path = np.array(path)
        trajs.append(path)
        stats.append({"u": list(u), "heading_circ_var": round(circ_var, 4),
                      "stall_frac": round(stall, 4),
                      "traj_min_goal_dist": round(float(np.linalg.norm(path - goal, axis=1).min()), 4),
                      "traj_net_displacement": round(float(np.linalg.norm(path[-1] - path[0])), 4),
                      "traj_path_length": round(float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()), 4)})
        print(json.dumps(stats[-1]), flush=True)

    # per-cell multimodality: angle spread of the flow's conditional at sample cells
    rng = np.random.default_rng(int(cfg.seed))
    probe_idx = rng.choice(len(obs_all), PROBE_CELLS, replace=False)
    probes = []
    for i in probe_idx:
        s = np.broadcast_to(obs_all[i].astype(np.float32), (PROBE_DRAWS, 2))
        udraw = np.clip(rng.standard_normal((PROBE_DRAWS, 2)), -3, 3).astype(np.float32)
        a = np.clip(np.asarray(onestep(s, udraw)), -1, 1)
        ang = np.arctan2(a[:, 1], a[:, 0])
        probes.append({"cell": [round(float(x), 3) for x in obs_all[i]],
                       "angle_circ_var": round(1.0 - float(np.hypot(np.cos(ang).mean(),
                                                                    np.sin(ang).mean())), 4)})

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, MUTED, ALERT, GREEN = "#1f2430", "#c9ced6", "#b3563d", "#3a7d44"
    sub = rng.choice(len(obs_all), 4000, replace=False)
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.6))
    for ax, u, a, path, st in zip(axes.ravel(), US, fields, trajs, stats):
        ax.scatter(obs_all[sub, 0], obs_all[sub, 1], s=2, c=MUTED, lw=0, zorder=0)
        ax.quiver(cells[:, 0], cells[:, 1], a[:, 0], a[:, 1], color=INK,
                  width=0.003, scale=30, zorder=1)
        ax.plot(path[:, 0], path[:, 1], c=ALERT, lw=1.6, zorder=2)
        ax.scatter([path[0, 0]], [path[0, 1]], marker="o", s=60, c=ALERT,
                   edgecolors="white", zorder=3, label="start")
        ax.scatter([goal[0]], [goal[1]], marker="*", s=180, c=GREEN,
                   edgecolors="white", zorder=3, label="goal")
        ax.set_title(f"u = {list(u)}   circ_var {st['heading_circ_var']}   "
                     f"min_dist {st['traj_min_goal_dist']}", fontsize=8.5, color=INK)
        ax.tick_params(labelsize=7)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("fixed-u policies as vector fields a = G(xy, u) — rollout in red\n"
                 f"{cfg.env_name}", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig_path = os.path.join(os.getcwd(), "fixed_u_fields.png")
    fig.savefig(fig_path, dpi=150)

    report = {"env": cfg.env_name, "seed": int(cfg.seed),
              "restore_path": str(cfg.restore_path), "restore_epoch": int(cfg.restore_epoch),
              "goal": [round(float(g), 3) for g in goal],
              "per_u": stats, "per_cell_multimodality": probes, "figure": fig_path}
    print(json.dumps(report, indent=2))
    write_report(report, cfg, "viz_fixed_u_field.json")


if __name__ == "__main__":
    main()

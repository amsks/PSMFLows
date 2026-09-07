"""Reachability: can ANY fixed-u policy reach the eval goal at all?

The calibration check (08-04) found every candidate latent realizes ZERO return — but it
sampled 16 prior draws and cut rollouts at 200 steps (eval episodes are 1000). This tool
settles whether the failure is candidate-sampling / horizon, or structural: for d_a=2
(pointmaze) it sweeps an exhaustive GRID over the whole latent box [-u_clip, u_clip]^2,
plus dataset preimage latents (including preimages of goal-reaching transitions, if any),
and rolls each u as a fixed-u policy for FULL episodes.

Needs only the Stage-A flow (the policy family is a = G(s, u); psi/phi only select).
Reports per-candidate: success (any step / final step), min agent-goal distance over the
episode. If nothing on the full grid ever reaches the goal, no selection mechanism can
save Rung-1 on this env — the family itself is not goal-covering from these starts.

Run: MUJOCO_GL=egl OGBENCH_DATASET_DIR=/var/local/amsks/ogbench \
    .venv/bin/python tools/latent_reachability.py agent=fql \
    env_name=pointmaze-medium-navigate-singletask-task1-v0 \
    restore_path='/var/local/amsks/exp/<flow_run_dir>' restore_epoch=500000 \
    +preimage_npz=/data-local/amsks/PSMFLows/preimages_pointmaze_medium_a20_n200.npz
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

GRID_N = 13          # d_a=2: GRID_N^2 exhaustive candidates over [-u_clip, u_clip]^2
NUM_PRIOR = 64       # d_a>2 fallback: clipped prior draws instead of a grid
NUM_PREIMAGE = 32    # dataset preimage candidates (random valid rows)
NUM_GOAL_PRE = 32    # preimages of goal-reaching transitions (reward > -0.5), if any
EPISODES_PER_U = 2
MAX_STEPS = 1000     # full eval horizon (env truncates itself; this is a backstop)
U_CLIP = 3.0         # matches configs/agent/psmflow.yaml u_clip


def _rollout(eval_env, onestep, u, reset_seed):
    ob, info = eval_env.reset(seed=reset_seed)
    # Distance over the first min(goal, obs) dims: xy for the maze envs (goal is a
    # position), full-obs for envs whose goal is a complete observation (e.g. cube,
    # where it mixes arm and object dims — auxiliary only; success is authoritative).
    goal = np.asarray(info["goal"]) if info.get("goal") is not None else None
    nd = min(goal.size, np.asarray(ob).size) if goal is not None else 0
    succ_any, succ_final, min_dist = 0.0, 0.0, np.inf
    for _ in range(MAX_STEPS):
        a = np.clip(np.asarray(onestep(ob[None], u[None]))[0], -1, 1)
        ob, _, term, trunc, info = eval_env.step(a)
        s = float(info.get("success", 0.0))
        succ_any = max(succ_any, s)
        succ_final = s
        if goal is not None:
            min_dist = min(min_dist, float(np.linalg.norm(np.asarray(ob)[:nd] - goal[:nd])))
        if term or trunc:
            break
    return succ_any, succ_final, min_dist


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    env, eval_env, train_dataset, _ = make_env_and_datasets(cfg.env_name, frame_stack=cfg.frame_stack)
    ex_obs = train_dataset['observations'][:1]
    ex_act = train_dataset['actions'][:1]
    assert cfg.agent.agent_name == 'fql', 'run with agent=fql (Stage-A flow ckpt)'
    agent_cfg = ml_collections.ConfigDict(_lists_to_tuples(OmegaConf.to_container(cfg.agent, resolve=True)))
    agent = FQLAgent.create(cfg.seed, ex_obs, ex_act, agent_cfg)
    assert cfg.restore_path is not None, 'needs the trained Stage-A flow (restore_path)'
    agent = restore_agent(agent, cfg.restore_path, cfg.restore_epoch)
    onestep = jax.jit(lambda o, u: agent.network.select('actor_onestep_flow')(o, u))

    rng = np.random.default_rng(int(cfg.seed))
    d_a = ex_act.shape[-1]
    groups = {}
    if d_a == 2:
        ax = np.linspace(-U_CLIP, U_CLIP, GRID_N, dtype=np.float32)
        gx, gy = np.meshgrid(ax, ax)
        groups["grid"] = np.stack([gx.ravel(), gy.ravel()], axis=1)
    else:
        groups["prior"] = np.clip(rng.standard_normal((NUM_PRIOR, d_a)),
                                  -U_CLIP, U_CLIP).astype(np.float32)

    npz_path = cfg.get("preimage_npz", None)
    if npz_path:
        npz = np.load(npz_path)
        valid = npz["preimage_valid"] > 0.5
        pts, rew = npz["noise_preimage_point"], npz["rewards"]
        vrows = np.flatnonzero(valid)
        groups["preimage"] = pts[rng.choice(vrows, size=min(NUM_PREIMAGE, len(vrows)),
                                            replace=False)].astype(np.float32)
        grows = np.flatnonzero(valid & (rew > -0.5))
        if len(grows):
            groups["goal_preimage"] = pts[rng.choice(grows, size=min(NUM_GOAL_PRE, len(grows)),
                                                     replace=False)].astype(np.float32)

    results = []
    for name, us in groups.items():
        for u in us:
            eps = [_rollout(eval_env, onestep, u, int(cfg.seed) * 1000 + ep)
                   for ep in range(EPISODES_PER_U)]
            sa, sf, md = (np.mean([e[i] for e in eps]) for i in range(3))
            results.append({"group": name, "u": [round(float(x), 4) for x in u],
                            "success_any": float(sa), "success_final": float(sf),
                            "min_goal_dist": round(float(md), 4)})
        done = sum(len(g) for g in list(groups.values())[:list(groups).index(name) + 1])
        print(f"[{name}] done ({done} candidates so far)", flush=True)

    succ = [r for r in results if r["success_any"] > 0]
    dists = np.array([r["min_goal_dist"] for r in results])
    report = {
        "env": cfg.env_name, "seed": int(cfg.seed),
        "restore_path": str(cfg.restore_path), "restore_epoch": int(cfg.restore_epoch),
        "episodes_per_u": EPISODES_PER_U, "max_steps": MAX_STEPS,
        "num_candidates": len(results),
        "groups": {k: len(v) for k, v in groups.items()},
        "num_success_any": len(succ),
        "num_success_final": sum(r["success_final"] > 0 for r in results),
        "min_goal_dist_overall": round(float(dists.min()), 4),
        "min_goal_dist_quartiles": [round(float(q), 4) for q in
                                    np.percentile(dists, [0, 25, 50, 75, 100])],
        "successes": succ,
        "closest_5": sorted(results, key=lambda r: r["min_goal_dist"])[:5],
        "candidates": results,
    }

    if d_a == 2:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        INK, ALERT = "#1f2430", "#b3563d"
        fig, ax_ = plt.subplots(figsize=(5.2, 4.6))
        for name, marker in [("grid", "s"), ("preimage", "o"), ("goal_preimage", "^")]:
            rs = [r for r in results if r["group"] == name]
            if not rs:
                continue
            uu = np.array([r["u"] for r in rs])
            dd = np.array([r["min_goal_dist"] for r in rs])
            sc = ax_.scatter(uu[:, 0], uu[:, 1], c=dd, marker=marker, s=55, cmap="viridis_r",
                             vmin=0, vmax=max(dists.max(), 1e-6), edgecolors="white",
                             lw=0.4, label=name)
        for r in succ:
            ax_.scatter([r["u"][0]], [r["u"][1]], marker="*", s=220, c=ALERT,
                        edgecolors="white", lw=0.8, zorder=5)
        ax_.set_xlabel("u[0]"), ax_.set_ylabel("u[1]")
        ax_.legend(frameon=False, fontsize=8)
        ax_.set_title(f"min goal distance per fixed-u policy (stars = reached goal)\n"
                      f"{cfg.env_name}", fontsize=9, color=INK)
        fig.colorbar(sc, ax=ax_, label="min goal dist over episode")
        fig.tight_layout()
        fig_path = os.path.join(os.getcwd(), "latent_reachability.png")
        fig.savefig(fig_path, dpi=150)
        report["figure"] = fig_path

    print(json.dumps({k: v for k, v in report.items() if k != "candidates"}, indent=2))
    write_report(report, cfg, "latent_reachability.json")


if __name__ == "__main__":
    main()

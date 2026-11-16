"""Does the task representation and the critic's action ranking respond to cube 1, to cube 2, or to neither?
(2026-11-16, cube-double; reuses the adapters and loader of diag_critic_compare.py.)

Per task, with the task vector inferred exactly as at eval:

1. REWARD VIEW. kg(y) = (f(y) - mu)^T w / sqrt(D) at N dataset next states y. Each cube block of y is moved to the
   task's goal position for that cube (the mean over the 32 rewarding goals), one cube at a time, both, or halfway.
   Reported: the mean change of kg in units of std(kg) over the N states. A view that ignores cube 2 shows
   ~0 for the cube-2 move and the same number for "cube 1" and "both".
2. ACTION RANKING. At N states, 64 clipped prior latents scored by the acting value; each decoded, the simulator
   stepped once. Per draw: progress of cube 1 toward its goal, of cube 2, and whether each cube moved.
   Reported: pooled Spearman (per-state centred) of the score with cube-1 progress, cube-2 progress, total cube
   displacement (dwell check: negative = the critic prefers actions that keep the cubes still), and the mean score
   of draws that move a cube minus draws that move none, in within-state stds.
3. ACTOR. The actor's one-step progress on cube 1 and cube 2 separately, and how often it moves each.

Run: as diag_critic_compare.py (same arguments).
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import utils.xla_guard  # noqa: F401  -- MUST precede jax

# isort: split

import jax
import jax.numpy as jnp
import numpy as np
from diag_critic_compare import load_agent

from tools.diag_index_ranking import task_env_name
from tools.eval_measure_vs_rollouts import restore_row, spearman
from tools.successor_flow import load_latent_dataset, parse_blocks, rewarding_goal_set
from utils.log_utils import write_report


def pooled_spearman(q, x):
    qc, xc = q - q.mean(1, keepdims=True), x - x.mean(1, keepdims=True)
    return float(spearman(qc.reshape(-1), xc.reshape(-1)))


def run_task(a, task):
    env_name = task_env_name(a.env, task)
    env, _, train_dataset, _ = load_latent_dataset(env_name, a.preimage_path, add_info=True)
    sim = {k: np.asarray(train_dataset[k]) for k in ("qpos", "qvel", "button_states") if k in train_dataset}
    env.unwrapped._terminate_at_goal = False
    from utils.datasets import Dataset
    if a.kind == "dfb":
        plain = Dataset.create(**{k: v for k, v in train_dataset.items()
                                  if not (k.startswith("noise_preimage") or k == "preimage_valid")})
    else:
        plain = Dataset.create(**{k: v for k, v in train_dataset.items() if k != "preimage_valid"})
        plain.return_preimage_noise = True
        plain.preimage_point_mode = True
    ad, zb_fit = load_agent(a.kind, a.run_dir, a.epoch, a.flow_ckpt, a.flow_epoch, a.preimage_path, plain, a.seed,
                            a.relabel_size, a.reward_shift)
    blocks = parse_blocks(ad.c["fb_hit_spec"])
    assert len(blocks) >= 1
    rng = np.random.default_rng(int(a.seed) + 100 * task)
    goals, n_rew = rewarding_goal_set(zb_fit["next_observations"], np.asarray(zb_fit["rewards"]) + a.reward_shift,
                                      a.k_goals, rng)
    goal_pos = [goals[:, b0:b1].mean(0) for b0, b1, _ in blocks]                     # per cube, (3,)
    obs_all = np.asarray(train_dataset["observations"], np.float32)
    nxt_all = np.asarray(train_dataset["next_observations"], np.float32)
    valid = np.nonzero(np.asarray(train_dataset["preimage_valid"]) > 0.5)[0]
    rows = np.sort(rng.choice(valid, size=int(a.n_states), replace=False))
    res = {"env_name": env_name, "n_rewarding_relabel": int(n_rew), "blocks": [list(b) for b in blocks]}

    # 1. reward view
    kgf = jax.jit(ad.kg)
    y = nxt_all[rows]
    kg0 = np.asarray(kgf(jnp.asarray(y)))
    sd = max(float(kg0.std()), 1e-12)
    view = {"kg_std": float(kg0.std()), "kg_mean": float(kg0.mean())}

    def moved(y, which, frac):
        z = y.copy()
        for i in which:
            b0, b1, _ = blocks[i]
            z[:, b0:b1] = (1 - frac) * z[:, b0:b1] + frac * goal_pos[i][None]
        return z
    for frac in (0.5, 1.0):
        for name, which in [("cube1", [0]), ("cube2", [1] if len(blocks) > 1 else []), ("both", list(range(len(blocks))))]:
            if not which:
                continue
            kg1 = np.asarray(kgf(jnp.asarray(moved(y, which, frac))))
            view[f"dkg_{name}_frac{frac}"] = float((kg1 - kg0).mean() / sd)
    # distance of the states' cubes to the goal (context: how far the moves are)
    for i, (b0, b1, _) in enumerate(blocks):
        view[f"cube{i + 1}_dist_to_goal_mean"] = float(np.linalg.norm(y[:, b0:b1] - goal_pos[i][None], axis=-1).mean())
    res["reward_view"] = view

    # 2. action ranking, 3. actor
    N, K, du = len(rows), int(a.n_u), ad.du
    s_raw = obs_all[rows]
    key = jax.random.PRNGKey(int(a.seed) + 11 + task)
    k_u, k_act = jax.random.split(key)
    u_draw = np.asarray(jnp.clip(jax.random.normal(k_u, (N, K, du)), -ad.u_clip, ad.u_clip))
    score = jax.jit(lambda o, u: ad.score_heads(o, u)[0])
    q = np.asarray(score(jnp.asarray(np.repeat(s_raw, K, 0)), jnp.asarray(u_draw.reshape(-1, du)))).reshape(N, K)
    u_a = np.asarray(jax.jit(ad.actor_u)(jnp.asarray(s_raw), jax.random.normal(k_act, (N, du))))
    u_all = np.concatenate([u_draw, u_a[:, None]], 1)
    dec = jax.jit(ad.a.decode)
    acts = np.asarray(dec(jnp.asarray(np.repeat(s_raw, K + 1, 0)), jnp.asarray(u_all.reshape(-1, du)))).reshape(N, K + 1, -1)
    nb = len(blocks)
    prog = np.zeros((N, K + 1, nb))
    disp = np.zeros((N, K + 1, nb))
    t0 = time.time()
    for i in range(N):
        d0 = [np.linalg.norm(s_raw[i, b0:b1] - goal_pos[k]) for k, (b0, b1, _) in enumerate(blocks)]
        for j in range(K + 1):
            restore_row(env, sim, int(rows[i]))
            ob, _, _, _, _ = env.step(np.asarray(acts[i, j], np.float32))
            ob = np.asarray(ob, np.float32)
            for k, (b0, b1, _) in enumerate(blocks):
                prog[i, j, k] = d0[k] - np.linalg.norm(ob[b0:b1] - goal_pos[k])
                disp[i, j, k] = np.linalg.norm(ob[b0:b1] - s_raw[i, b0:b1])
    within = np.maximum(q.std(1), 1e-12)
    qz = (q - q.mean(1, keepdims=True)) / within[:, None]
    rank = {"within_state_std": float(q.std(1).mean())}
    moved_any = (disp[:, :K].max(-1) > 1e-4)
    for k in range(nb):
        rank[f"spearman_q_vs_progress_cube{k + 1}"] = pooled_spearman(q, prog[:, :K, k])
        rank[f"frac_draws_moved_cube{k + 1}"] = float((disp[:, :K, k] > 1e-4).mean())
        mk = disp[:, :K, k] > 1e-4
        rank[f"q_moved_minus_still_cube{k + 1}"] = float(np.nanmean(
            [qz[i, mk[i]].mean() - qz[i, ~mk[i]].mean() for i in range(N) if mk[i].any() and (~mk[i]).any()]))
    rank["spearman_q_vs_total_displacement"] = pooled_spearman(q, disp[:, :K].sum(-1))
    rank["q_moved_any_minus_still"] = float(np.nanmean(
        [qz[i, moved_any[i]].mean() - qz[i, ~moved_any[i]].mean() for i in range(N) if moved_any[i].any() and (~moved_any[i]).any()]))
    rank["frac_states_with_still_and_moved_draws"] = float(np.mean([moved_any[i].any() and (~moved_any[i]).any() for i in range(N)]))
    rank["seconds_sim"] = round(time.time() - t0, 1)
    res["ranking"] = rank
    act = {}
    for k in range(nb):
        act[f"actor_progress_cube{k + 1}"] = float(prog[:, K, k].mean())
        act[f"rand_progress_cube{k + 1}"] = float(prog[:, :K, k].mean())
        act[f"best_draw_progress_cube{k + 1}"] = float(prog[:, :K, k].max(1).mean())
        act[f"actor_frac_moved_cube{k + 1}"] = float((disp[:, K, k] > 1e-4).mean())
        act[f"rand_frac_moved_cube{k + 1}"] = float((disp[:, :K, k] > 1e-4).mean())
    res["actor"] = act
    env.close()
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kind", choices=("psmgoal", "dfb"), required=True)
    p.add_argument("--run_dir", required=True)
    p.add_argument("--epoch", type=int, default=500000)
    p.add_argument("--tasks", default="1,2,3,4,5")
    p.add_argument("--n_states", type=int, default=256)
    p.add_argument("--n_u", type=int, default=64)
    p.add_argument("--env", default="cube-double-play-singletask-v0")
    p.add_argument("--flow_ckpt", default=os.path.join(os.environ.get("PSM_DATA", ""), "flow", "cube-double-play"))
    p.add_argument("--flow_epoch", type=int, default=500000)
    p.add_argument("--preimage_path",
                   default=os.path.join(os.environ.get("PSM_DATA", ""), "preimages", "cube-double-play.npz"))
    p.add_argument("--relabel_size", type=int, default=10000)
    p.add_argument("--reward_shift", type=float, default=1.0)
    p.add_argument("--k_goals", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--report_out", required=True)
    a = p.parse_args()
    t0 = time.time()
    tasks = [int(t) for t in a.tasks.split(",")]
    report = {"args": vars(a), "definitions": __doc__, "per_task": {}}
    for t in tasks:
        report["per_task"][str(t)] = run_task(a, t)
        print(f"task {t}: " + json.dumps(report["per_task"][str(t)]), flush=True)

    def mean_tree(xs):
        if isinstance(xs[0], dict):
            return {k: mean_tree([x[k] for x in xs]) for k in xs[0] if isinstance(xs[0][k], (dict, int, float)) and k != "blocks"}
        return float(np.nanmean(xs))
    report["mean_over_tasks"] = mean_tree([report["per_task"][str(t)] for t in tasks])
    report["seconds"] = round(time.time() - t0, 1)
    print("mean over tasks: " + json.dumps(report["mean_over_tasks"], indent=1), flush=True)
    write_report(report, {"report_out": a.report_out}, "diag_cube_sensitivity.json")


if __name__ == "__main__":
    main()

"""Privileged-z probe (frozen FB checkpoint, run under .venv).

Asks: does the value's terminal resolution improve if we replace the actual goal
vector z=B(g) (reward-inference) with the BEST-POSSIBLE z (a free 50-vector
optimized against the frozen forward map F)? This isolates B's goal-encoding from
the forward path:
  best-z >> z=B(g)  ->  the bottleneck is B's goal inference (z=B(g) is smeared)
  best-z ~= z=B(g)  ->  the forward path F can't resolve terminal regardless of z

F and the actor are frozen; only z is optimized (held-out eval). Metric:
Spearman(V_z(s), -cube→goal distance) over transport-phase states.

  MUJOCO_GL=glfw .venv/bin/python -m scripts.probes.privileged_z_probe \
    --config <run>/.hydra/config.yaml --checkpoint <run>/checkpoints/final.pt \
    --data-path datasets --task 1 --n-states 8000
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from evals._profile_core import _spearman

CUBE_SLICE = slice(14, 17)
TASK_TMPL = "cube-single-play-singletask-task{n}-v0"


def _pearson(x, y):
    x = x - x.mean(); y = y - y.mean()
    return (x * y).sum() / (x.norm() * y.norm() + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-path", default="datasets")
    ap.add_argument("--task", type=int, default=1)
    ap.add_argument("--n-states", type=int, default=8000)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--lr", type=float, default=3e-2)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from evals.analysis import build_env_and_agent, load_cfg, load_checkpoint
    from evals.ogbench import OGBenchEvaluator
    from envs.ogbench import create_ogbench_env, ALL_TASKS
    from evals.phase_probe import Thresholds
    from evals.training_value import region_labels, cube_to_goal_dist
    from data.ogbench import load_ogbench_dataset

    cfg = load_cfg(args.config, device=args.device)
    if args.data_path:
        cfg.data_path = args.data_path
    fs = int(getattr(cfg, "frame_stack", 1))
    env, agent, _, _ = build_env_and_agent(cfg)
    load_checkpoint(agent, args.checkpoint, map_location=args.device)
    m = agent.model; m.eval()
    if hasattr(env, "close"):
        env.close()

    e0, _ = create_ogbench_env(TASK_TMPL.format(n=args.task), seed=cfg.seed,
                               obs_type=cfg.obs_type, frame_stack=fs)
    tb = int(getattr(e0.unwrapped, "_target_block", 0) or 0)
    table_z = float(e0.unwrapped.cur_task_info["init_xyzs"][tb][2])
    goal = np.asarray(e0.unwrapped.cur_task_info["goal_xyzs"][tb], np.float64)
    e0.close()
    thr = Thresholds()

    buffer = load_ogbench_dataset(domain=cfg.domain, data_path=cfg.data_path,
                                  load_n_episodes=cfg.load_n_episodes, device=args.device,
                                  n_transitions=cfg.n_transitions, obs_type=cfg.obs_type,
                                  frame_stack=fs)
    evaluator = OGBenchEvaluator(domain=cfg.domain, agent=agent, offline_buffer=buffer,
                                 relabel_size=cfg.eval_relabel_size, n_episodes=1,
                                 shift_reward=cfg.eval_shift_reward, obs_type=cfg.obs_type,
                                 seed=cfg.seed, device=args.device, use_wandb=False)
    z0, _ = evaluator._infer_z(list(ALL_TASKS[cfg.domain])[args.task - 1])
    z0 = torch.as_tensor(np.asarray(z0), dtype=torch.float32, device=args.device).reshape(-1)

    smp = buffer.sample(min(args.n_states, len(buffer)))
    obs = smp["next"]["observation"].to(args.device).float()
    phys = smp["next"]["physics"].detach().cpu().numpy()
    cube = phys[:, CUBE_SLICE].astype(np.float64)
    grip = np.clip(phys[:, 6] / 0.8, 0, 1)
    region = np.array([str(r) for r in region_labels(grip, phys[:, 16] - table_z, thr)])
    d = cube_to_goal_dist(cube, goal)
    tr = region == "transport"
    obs_t = obs[tr]
    d_t = torch.as_tensor(d[tr], dtype=torch.float32, device=args.device)
    near = d[tr] <= np.percentile(d[tr], 40)         # near-terminal subset

    # Frozen forward features + (frozen) policy action at z0.
    with torch.no_grad():
        left = m._left_encoder(m._fw_encoder(m._normalize(obs_t)))
        a = m.act(obs_t, z0.reshape(1, -1).expand(len(obs_t), -1), mean=True)

    def value(z):
        F = m._forward_map(left, z.reshape(1, -1).expand(len(obs_t), -1), a).mean(0)  # [N, z_dim]
        return (F * z.reshape(1, -1)).sum(-1)

    N = len(obs_t); idx = np.random.default_rng(0).permutation(N); cut = N // 2
    trn, tst = idx[:cut], idx[cut:]

    def spear(zv):
        with torch.no_grad():
            V = value(zv).cpu().numpy()
        return (_spearman(V[tst], -d[tr][tst]),
                _spearman(V[tst][near[tst]], -d[tr][tst][near[tst]]))

    base_all, base_near = spear(z0)

    # Optimize a free z against frozen F (train half), eval held-out.
    z = z0.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=args.lr)
    mtr = torch.as_tensor(trn, device=args.device)
    for _ in range(args.steps):
        opt.zero_grad()
        V = value(z)
        loss = -_pearson(V[mtr], -d_t[mtr])
        loss.backward(); opt.step()
    opt_all, opt_near = spear(z.detach())

    # forward-featurizer ceiling (ridge linear of left_enc -> d) for reference
    from evals.factorization_probe import fit_eval_readout
    L = left.cpu().numpy()
    le = fit_eval_readout(L[trn], d[tr][trn], L[tst], d[tr][tst], kind="linear", task="regression")

    print(f"[privileged-z] task{args.task}  transport states n={N}")
    print(f"  Spearman(V, -d)   z=B(g) (actual):  all={base_all:+.3f}  near-goal={base_near:+.3f}")
    print(f"  Spearman(V, -d)   z* (best free z):  all={opt_all:+.3f}  near-goal={opt_near:+.3f}")
    print(f"  reference: ridge left_enc->d  R^2={le['score']:.3f}  (forward featurizer geometry)")
    verdict = ("goal-encoding (z=B(g)) is the bottleneck"
               if (opt_near - base_near) > 0.1 else
               "forward path F also can't resolve terminal (deeper than z)")
    print(f"  => {verdict}")


if __name__ == "__main__":
    main()

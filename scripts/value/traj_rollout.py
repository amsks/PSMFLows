"""Closed-loop rollout from a play trajectory's START toward its HINDSIGHT goal,
to test which agent actually SOLVES that specific instance (success rate).

For trajectory N (sorted episode-file index, matching scripts.value.traj_value_profile):
  init  = episode start state  (mujoco set_state on phys[0])
  goal  = the trajectory's later cube config: obs[tg] conditions the policy,
          phys[tg,14:17] is the target for success (OGBench cube_reward_fn, 4cm).
We roll out K stochastic episodes (success RATE) + 1 greedy episode, from the
fixed start, and report how often the cube reaches the goal.

  # FB (.venv):
  MUJOCO_GL=glfw .venv/bin/python -m scripts.value.traj_rollout --method fb \
    --config <run>/.hydra/config.yaml --checkpoint <run>/checkpoints/final.pt \
    --traj 236 841 --out analysis/value/traj_value/rollout.csv
  # GCIQL / CRL (.venv-jax-cpu):
  MUJOCO_GL=glfw PYTHONPATH=tools/wandb_mode_shim:third_party/ogbench/impls \
  .venv-jax-cpu/bin/python -m scripts.value.traj_rollout --method crl \
    --run-dir <crl_run> --step 1000000 --traj 236 841 \
    --out analysis/value/traj_value/rollout.csv
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CUBE_SLICE = slice(14, 17)
DOMAIN = "cube-single-play-v0"
TASK = "cube-single-play-singletask-task1-v0"


def _episode(traj: int):
    files = sorted(glob.glob(str(REPO_ROOT / "datasets" / DOMAIN / "buffer" / "episode_*.npz")))
    z = np.load(files[traj])
    return np.asarray(z["observation"], np.float32), np.asarray(z["physics"], np.float64)


def _run(make_action, traj, goal_offset, max_steps, thresh, k, perturb):
    """Each agent uses its EVAL-FAITHFUL (greedy) policy: FB mean-action, GCIQL
    temperature=0, CRL one-shot flow actor (noise seed varies = its natural eval).
    `greedy_*` = the exact start state; `success_rate` = k rollouts from the start
    cube xy jittered by N(0, perturb) (a robustness rate around the instance, with
    each method's true policy — NOT actor-temperature noise, which breaks GCIQL)."""
    import mujoco
    from envs.ogbench import create_ogbench_env
    env, _ = create_ogbench_env(TASK, seed=0, obs_type="state")
    u = env.unwrapped
    nv = u.model.nv
    obs_ep, phys = _episode(traj)
    T = len(obs_ep); tg = T - 1 - goal_offset
    goal_obs = obs_ep[tg].astype(np.float32)
    goal_cube = phys[tg, CUBE_SLICE]
    start_dist = float(np.linalg.norm(phys[0, CUBE_SLICE] - goal_cube))

    def one(act_fn, seed, jitter):
        env.reset(seed=seed)
        q = phys[0].copy()
        if jitter > 0:
            q[14:16] = q[14:16] + np.random.default_rng(seed).normal(0, jitter, 2)
        u.set_state(q, np.zeros(nv)); mujoco.mj_forward(u.model, u.data)
        ob = np.asarray(u.compute_observation(), np.float32)
        dmin = 1e9
        for _ in range(max_steps):
            ob, _, term, trunc, _ = env.step(act_fn(ob))
            ob = np.asarray(ob, np.float32)
            d = float(np.linalg.norm(np.asarray(u.data.qpos[CUBE_SLICE]) - goal_cube))
            dmin = min(dmin, d)
            if d < thresh or term or trunc:
                break
        return dmin

    greedy_d = one(make_action(0, goal_obs), 0, 0.0)
    mins = np.asarray([one(make_action(s + 1, goal_obs), s + 1, perturb) for s in range(k)])
    env.close()
    return dict(traj=traj, start_dist=round(start_dist, 4),
                greedy_success=int(greedy_d < thresh), greedy_min_dist=round(float(greedy_d), 4),
                success_rate=float((mins < thresh).mean()),
                mean_min_dist=float(mins.mean()), k=k, perturb=perturb)


def _fb_factory(args):
    import torch
    from evals.analysis import build_env_and_agent, load_cfg, load_checkpoint
    cfg = load_cfg(args.config, device=args.device); cfg.data_path = "datasets"
    e0, agent, _, _ = build_env_and_agent(cfg)
    load_checkpoint(agent, args.checkpoint, map_location=args.device)
    agent.model.eval()
    if hasattr(e0, "close"):
        e0.close()
    m = agent.model

    def make_action(seed, goal_obs):
        with torch.no_grad():
            z = m.project_z(m.backward_map(
                torch.as_tensor(goal_obs[None], dtype=torch.float32))).squeeze(0)

        def act(ob):  # eval-faithful: mean action (deterministic)
            with torch.no_grad():
                return m.act(torch.as_tensor(ob[None], dtype=torch.float32),
                             z.reshape(1, -1), mean=True).cpu().numpy()[0]
        return act
    return make_action


def _jax_factory(args):
    ogb = str(REPO_ROOT / "third_party" / "ogbench" / "impls")
    shim = str(REPO_ROOT / "tools" / "wandb_mode_shim")
    for p in (ogb, shim):
        if p in sys.path:
            sys.path.remove(p)
    sys.path.insert(0, ogb); sys.path.insert(0, shim)
    import jax, jax.numpy as jnp, ogbench  # noqa: F401
    from utils.flax_utils import restore_agent
    flags = json.loads((Path(args.run_dir) / "flags.json").read_text())
    saved = flags.get("agent")
    name = (saved.get("agent_name") if isinstance(saved, dict) else None) or args.method
    if name == "crl_flowbc":
        from crl_flowbc import CRLFlowBCAgent as Cls, get_config
    elif name == "gcivl":
        from agents.gcivl import GCIVLAgent as Cls, get_config
    else:
        from agents.gciql import GCIQLAgent as Cls, get_config
    config = get_config()
    if isinstance(saved, dict):
        for k, v in saved.items():
            if k in config:
                config[k] = v
    env = ogbench.make_env_and_datasets(flags["env_name"], env_only=True)
    ex_obs, _ = env.reset(options=dict(task_id=1))
    agent = Cls.create(int(flags["seed"]), np.asarray(ex_obs, np.float32)[None],
                       np.asarray(env.action_space.sample(), np.float32)[None], config)
    agent = restore_agent(agent, str(args.run_dir), args.step)
    env.close()

    def make_action(seed, goal_obs):
        # eval-faithful: temperature=0 (GCIQL/GCIVL greedy). CRL's one-shot flow
        # actor is noise-driven (temperature ignored); seed varies its draw.
        g = jnp.asarray(goal_obs[None]); key = jax.random.PRNGKey(seed)

        def act(ob):
            a = agent.sample_actions(jnp.asarray(ob[None]), g, seed=key, temperature=0.0)
            return np.asarray(a).reshape(-1)
        return act
    return make_action


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--method", required=True, choices=["fb", "gciql", "crl"])
    ap.add_argument("--config"); ap.add_argument("--checkpoint")
    ap.add_argument("--run-dir"); ap.add_argument("--step", type=int, default=1000000)
    ap.add_argument("--traj", type=int, nargs="+", required=True)
    ap.add_argument("--k", type=int, default=15, help="rollouts (init-jittered) per traj")
    ap.add_argument("--perturb", type=float, default=0.01,
                    help="std (m) of start-cube xy jitter for the robustness rate")
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--goal-offset", type=int, default=5)
    ap.add_argument("--thresh", type=float, default=0.04)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--mujoco-gl", default=None)
    ap.add_argument("--out", default="analysis/value/traj_value/rollout.csv")
    args = ap.parse_args()
    if args.mujoco_gl:
        import os; os.environ["MUJOCO_GL"] = args.mujoco_gl
    make_action = _fb_factory(args) if args.method == "fb" else _jax_factory(args)
    rows = []
    for tr in args.traj:
        r = _run(make_action, tr, args.goal_offset, args.max_steps, args.thresh,
                 args.k, args.perturb)
        r["method"] = args.method
        rows.append(r)
        print(f"[rollout:{args.method}] traj {tr}: start_dist={r['start_dist']}  "
              f"greedy={r['greedy_success']} (min_d={r['greedy_min_dist']})  "
              f"rate={r['success_rate']:.2f}  mean_min_dist={r['mean_min_dist']:.3f}")
    new = pd.DataFrame(rows)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        old = pd.read_csv(out)
        old = old[~old.set_index(["method", "traj"]).index.isin(
            new.set_index(["method", "traj"]).index)]
        new = pd.concat([old, new], ignore_index=True)
    new.sort_values(["traj", "method"]).to_csv(out, index=False)
    print(f"-> {out}")


if __name__ == "__main__":
    main()

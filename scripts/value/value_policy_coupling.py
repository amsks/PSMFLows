"""Value<->policy coupling probe — does the value function have causal grip on
behavior, or is success BC-driven (decoupled from the critic)?

On a trajectory's hindsight-goal instance (traj index = sorted episode file):
 (A) DATA-state critic ranking: mean critic value at the actor's action vs the
     data action vs random actions, over states along the play trajectory.
       Q(pi) >> Q(rand)  -> actor exploits the critic.
       Q(pi) ~= Q(data)  -> actor is BC-imitating (not climbing the critic).
 (B) GREEDY-rollout coupling: along a greedy rollout from the start, record the
     cube->goal distance d_t and the critic value Q(s_t, pi). If Q stays high
     while d stays high, the critic is BLIND to failure (value decoupled from
     outcome). spearman(Q, -d) summarizes whether value tracks real progress.

  # FB (.venv):
  MUJOCO_GL=glfw .venv/bin/python -m scripts.value.value_policy_coupling --method fb \
    --config <run>/.hydra/config.yaml --checkpoint <run>/checkpoints/final.pt --traj 236
  # GCIQL / CRL (.venv-jax-cpu):
  MUJOCO_GL=glfw PYTHONPATH=tools/wandb_mode_shim:third_party/ogbench/impls \
  .venv-jax-cpu/bin/python -m scripts.value.value_policy_coupling --method crl \
    --run-dir <crl_run> --step 1000000 --traj 236
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from evals._profile_core import _spearman

CUBE = slice(14, 17)
DOMAIN = "cube-single-play-v0"
TASK = "cube-single-play-singletask-task1-v0"


def _episode(traj):
    f = sorted(glob.glob(str(REPO_ROOT / "datasets" / DOMAIN / "buffer" / "episode_*.npz")))[traj]
    z = np.load(f)
    return (np.asarray(z["observation"], np.float32), np.asarray(z["action"], np.float32),
            np.asarray(z["physics"], np.float64))


def _fb(args, goal_obs):
    import torch
    from evals.analysis import build_env_and_agent, load_cfg, load_checkpoint
    from evals.representation_profile import q_values
    cfg = load_cfg(args.config, device="cpu"); cfg.data_path = "datasets"
    e0, agent, _, _ = build_env_and_agent(cfg)
    load_checkpoint(agent, args.checkpoint, map_location="cpu"); agent.model.eval()
    if hasattr(e0, "close"):
        e0.close()
    m = agent.model
    with torch.no_grad():
        z = m.project_z(m.backward_map(torch.as_tensor(goal_obs[None], dtype=torch.float32))).squeeze(0)

    def q_batch(obs, act):
        with torch.no_grad():
            return np.asarray(q_values(m, torch.as_tensor(np.atleast_2d(obs), dtype=torch.float32),
                                       torch.as_tensor(np.atleast_2d(act), dtype=torch.float32), z)).reshape(-1)

    def pi(obs):
        o = torch.as_tensor(np.atleast_2d(obs), dtype=torch.float32)
        with torch.no_grad():
            return m.act(o, z.reshape(1, -1).expand(len(o), -1), mean=True).cpu().numpy()
    return q_batch, pi


def _jax(args, goal_obs):
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
    agent = restore_agent(agent, str(args.run_dir), args.step); env.close()
    key = jax.random.PRNGKey(0)

    def q_batch(obs, act):
        o = jnp.asarray(np.atleast_2d(obs)); a = jnp.asarray(np.atleast_2d(act))
        gg = jnp.broadcast_to(jnp.asarray(goal_obs)[None], o.shape)
        out = jnp.asarray(agent.network.select("critic")(o, gg, a))
        q = out.min(0) if out.ndim >= 2 else out
        return np.asarray(q).reshape(-1)

    def pi(obs):
        o = jnp.asarray(np.atleast_2d(obs))
        gg = jnp.broadcast_to(jnp.asarray(goal_obs)[None], o.shape)
        return np.asarray(agent.sample_actions(o, gg, seed=key, temperature=0.0))
    return q_batch, pi


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--method", required=True,
                    choices=["fb", "gciql", "crl", "rldp"])
    ap.add_argument("--config"); ap.add_argument("--checkpoint")
    ap.add_argument("--run-dir"); ap.add_argument("--step", type=int, default=1000000)
    ap.add_argument("--traj", type=int, required=True)
    ap.add_argument("--goal-offset", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--thresh", type=float, default=0.04)
    ap.add_argument("--mujoco-gl", default=None)
    args = ap.parse_args()
    if args.mujoco_gl:
        import os; os.environ["MUJOCO_GL"] = args.mujoco_gl

    obs_ep, act_ep, phys = _episode(args.traj)
    T = len(obs_ep); tg = T - 1 - args.goal_offset
    goal_obs = obs_ep[tg].astype(np.float32); goal_cube = phys[tg, CUBE]
    # rldp is a PyTorch FB-family agent -> the FB critic/policy path.
    q_batch, pi = (_fb if args.method in ("fb", "rldp") else _jax)(args, goal_obs)

    # (A) DATA-state critic ranking: Q(pi) vs Q(data) vs Q(random)
    ts = np.arange(0, tg, 25)
    O = obs_ep[ts]
    a_data = act_ep[np.minimum(ts + 1, T - 1)]            # action taken at obs[t] (EXORL shift)
    a_rand = np.random.default_rng(0).uniform(-1, 1, a_data.shape).astype(np.float32)
    q_pi = q_batch(O, pi(O)); q_dat = q_batch(O, a_data); q_rnd = q_batch(O, a_rand)
    print(f"\n=== {args.method.upper()}  traj {args.traj} ===")
    print("(A) DATA states — critic value at each action set (mean over states):")
    print(f"    Q(pi)={q_pi.mean():+.3f}   Q(data)={q_dat.mean():+.3f}   Q(random)={q_rnd.mean():+.3f}")
    print(f"    actor>random: {q_pi.mean()-q_rnd.mean():+.3f}   actor>data: {q_pi.mean()-q_dat.mean():+.3f}  "
          f"({'climbs critic' if q_pi.mean()-q_rnd.mean()>0.05 else 'IGNORES critic'}; "
          f"{'>data (Q-max)' if q_pi.mean()-q_dat.mean()>0.05 else '~data (BC-imitate)'})")

    # (B) GREEDY-rollout coupling: d_t and Q(s_t, pi) along the rollout
    import mujoco
    from envs.ogbench import create_ogbench_env
    env, _ = create_ogbench_env(TASK, seed=0, obs_type="state"); u = env.unwrapped; nv = u.model.nv
    env.reset(seed=0); u.set_state(phys[0], np.zeros(nv)); mujoco.mj_forward(u.model, u.data)
    ob = np.asarray(u.compute_observation(), np.float32)
    ds, qs = [], []
    for _ in range(args.max_steps):
        a = pi(ob)[0]; qs.append(float(q_batch(ob, a)[0]))
        ob, _, term, trunc, _ = env.step(a); ob = np.asarray(ob, np.float32)
        d = float(np.linalg.norm(np.asarray(u.data.qpos[CUBE]) - goal_cube)); ds.append(d)
        if d < args.thresh or term or trunc:
            break
    env.close()
    ds, qs = np.asarray(ds), np.asarray(qs)
    qn = (qs - qs.min()) / (qs.max() - qs.min() + 1e-9)
    solved = ds.min() < args.thresh
    print("(B) GREEDY rollout — does the critic track actual progress?")
    print(f"    cube->goal d: start={ds[0]:.3f}  end={ds[-1]:.3f}  min={ds.min():.3f}  -> "
          f"{'SOLVED' if solved else 'FAILED'}")
    print(f"    Q(s,pi):      start={qs[0]:+.3f}  end={qs[-1]:+.3f}  (norm start={qn[0]:.2f} end={qn[-1]:.2f})")
    print(f"    spearman(Q, -d) along rollout = {_spearman(qs, -ds):+.3f}")
    if not solved and qn[-1] > 0.6:
        print("    => critic stays HIGH while the cube never arrives: CRITIC BLIND to failure.")
    elif not solved:
        print("    => critic drops as it fails: value 'knows', but the actor can't do better.")
    else:
        print("    => solved; critic value rises with progress." if _spearman(qs, -ds) > 0.3
              else "    => solved, but critic value does NOT track progress (success despite flat value).")


if __name__ == "__main__":
    main()

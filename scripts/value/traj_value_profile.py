"""Trajectory value profile — evaluate the goal-conditioned value ALONG real play
trajectories with a HINDSIGHT goal (the trajectory's own later cube position), so
states are coherent (no cube-xy marginalization). Run across the WHOLE training
dataset; dump per-(traj,step) value profile + a per-trajectory goal-peaking score
(Spearman(V, -d): high ⇒ value rises toward the goal). State regime.

FB uses z = project_z(B(goal_obs)) — so B's resolution enters directly.

  # FB (.venv):           --method fb  --config <run>/.hydra/config.yaml --checkpoint ...
  # GCIQL (.venv-jax-cpu):--method gciql --run-dir <gciql_run> --step 1000000
  .../python -m scripts.value.traj_value_profile --method fb ... \
     --data-path datasets --out analysis/value/traj_value/fb.parquet --n-traj 1000
Writes <out> (per-step) and <out stem>_summary.csv (per-trajectory).
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
from evals._profile_core import _spearman

CUBE_SLICE = slice(14, 17)
GRIP_IDX, LIFT_IDX = 6, 16


def _episodes(data_path, domain, n_traj, seed):
    files = sorted(glob.glob(str(Path(data_path) / domain / "buffer" / "episode_*.npz")))
    rng = np.random.default_rng(seed)
    if n_traj < len(files):
        files = [files[i] for i in sorted(rng.choice(len(files), n_traj, replace=False))]
    for f in files:
        z = np.load(f)
        yield (np.asarray(z["observation"], np.float32), np.asarray(z["physics"], np.float32))


def _rows_from(ti, obs, phys, ts, tg, V):
    cube = phys[ts][:, CUBE_SLICE].astype(np.float64)
    gx = phys[tg, CUBE_SLICE].astype(np.float64)
    d = np.linalg.norm(cube - gx, axis=1)
    gr = phys[ts][:, GRIP_IDX].astype(np.float64)
    lf = phys[ts][:, LIFT_IDX].astype(np.float64)
    return [dict(traj=ti, t=int(t), progress=t / tg, d=float(d[k]), V=float(V[k]),
                 cube_x=float(cube[k, 0]), cube_y=float(cube[k, 1]),
                 grip=float(gr[k]), lift=float(lf[k]))
            for k, t in enumerate(ts)]


def _fb(args):
    import torch
    from evals.analysis import build_env_and_agent, load_cfg, load_checkpoint
    from evals.representation_profile import v_values
    cfg = load_cfg(args.config, device=args.device)
    if args.data_path:
        cfg.data_path = args.data_path
    env, agent, _, _ = build_env_and_agent(cfg)
    load_checkpoint(agent, args.checkpoint, map_location=args.device)
    agent.model.eval()
    if hasattr(env, "close"):
        env.close()
    m = agent.model
    rows = []
    for ti, (obs, phys) in enumerate(_episodes(cfg.data_path, cfg.domain, args.n_traj, cfg.seed)):
        T = len(obs); tg = T - 1 - args.goal_offset
        if tg < args.stride * 3:
            continue
        goal_obs = torch.as_tensor(obs[tg:tg + 1], dtype=torch.float32, device=args.device)
        z = m.project_z(m.backward_map(goal_obs)).squeeze(0)
        ts = np.arange(0, tg, args.stride)
        V = np.asarray(v_values(m, agent, torch.as_tensor(obs[ts], dtype=torch.float32,
                                                          device=args.device), z)).reshape(-1)
        rows += _rows_from(ti, obs, phys, ts, tg, V)
    return pd.DataFrame(rows)


def _gciql(args):
    ogb = str(REPO_ROOT / "third_party" / "ogbench" / "impls")
    if ogb in sys.path:
        sys.path.remove(ogb)
    sys.path.insert(0, ogb)
    import ogbench  # noqa: F401
    import jax.numpy as jnp
    from utils.flax_utils import restore_agent
    flags = json.loads((Path(args.run_dir) / "flags.json").read_text())
    saved = flags.get("agent")
    name = (saved.get("agent_name") if isinstance(saved, dict) else None) or "gciql"
    if name == "gcivl":
        from agents.gcivl import GCIVLAgent as Cls, get_config
    else:
        from agents.gciql import GCIQLAgent as Cls, get_config
    config = get_config()
    if isinstance(saved, dict):
        for k, v in saved.items():
            if k in config:
                config[k] = v
    env_name = flags["env_name"]
    env = ogbench.make_env_and_datasets(env_name, env_only=True)
    ex_obs, _ = env.reset(options=dict(task_id=1))
    agent = Cls.create(int(flags["seed"]), np.asarray(ex_obs, np.float32)[None],
                       np.asarray(env.action_space.sample(), np.float32)[None], config)
    agent = restore_agent(agent, str(args.run_dir), args.step)
    env.close()

    def value(o, g):
        v = jnp.asarray(agent.network.select("value")(o, g))
        return np.asarray(v.mean(0) if v.ndim == 2 else v)

    rows = []
    for ti, (obs, phys) in enumerate(_episodes(args.data_path, env_name.replace("visual-", ""),
                                               args.n_traj, int(flags.get("seed", 0)))):
        T = len(obs); tg = T - 1 - args.goal_offset
        if tg < args.stride * 3:
            continue
        ts = np.arange(0, tg, args.stride)
        goal = np.broadcast_to(obs[tg][None], (len(ts),) + obs[tg].shape)
        V = value(jnp.asarray(obs[ts], jnp.float32), jnp.asarray(goal, jnp.float32)).reshape(-1)
        rows += _rows_from(ti, obs, phys, ts, tg, V)
    return pd.DataFrame(rows)


def _crl(args):
    """CRL+FlowBC (JAX): contrastive value V(s,g)=min(q1,q2)(s,g,pi(s,g)).
    Needs the wandb_mode_shim on path (crl_flowbc + ogbench_flow live there)."""
    ogb = str(REPO_ROOT / "third_party" / "ogbench" / "impls")
    shim = str(REPO_ROOT / "tools" / "wandb_mode_shim")
    for p in (ogb, shim):
        if p in sys.path:
            sys.path.remove(p)
    sys.path.insert(0, ogb)
    sys.path.insert(0, shim)
    import ogbench  # noqa: F401
    import jax
    import jax.numpy as jnp
    from utils.flax_utils import restore_agent
    from crl_flowbc import CRLFlowBCAgent, get_config
    flags = json.loads((Path(args.run_dir) / "flags.json").read_text())
    saved = flags.get("agent")
    config = get_config()
    if isinstance(saved, dict):
        for k, v in saved.items():
            if k in config:
                config[k] = v
    env_name = flags["env_name"]
    env = ogbench.make_env_and_datasets(env_name, env_only=True)
    ex_obs, _ = env.reset(options=dict(task_id=1))
    agent = CRLFlowBCAgent.create(int(flags["seed"]), np.asarray(ex_obs, np.float32)[None],
                                  np.asarray(env.action_space.sample(), np.float32)[None], config)
    agent = restore_agent(agent, str(args.run_dir), args.step)
    env.close()
    key = jax.random.PRNGKey(0)

    def value(o, g):
        a = agent.sample_actions(o, g, seed=key)
        q1, q2 = agent.network.select("critic")(o, g, a)
        return np.asarray(jnp.minimum(q1, q2)).reshape(-1)

    rows = []
    for ti, (obs, phys) in enumerate(_episodes(args.data_path, env_name.replace("visual-", ""),
                                               args.n_traj, int(flags.get("seed", 0)))):
        T = len(obs); tg = T - 1 - args.goal_offset
        if tg < args.stride * 3:
            continue
        ts = np.arange(0, tg, args.stride)
        goal = np.broadcast_to(obs[tg][None], (len(ts),) + obs[tg].shape)
        V = value(jnp.asarray(obs[ts], jnp.float32), jnp.asarray(goal, jnp.float32)).reshape(-1)
        rows += _rows_from(ti, obs, phys, ts, tg, V)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True,
                    choices=["fb", "gciql", "crl", "rldp"])
    ap.add_argument("--config"); ap.add_argument("--checkpoint")
    ap.add_argument("--run-dir"); ap.add_argument("--step", type=int)
    ap.add_argument("--data-path", default="datasets")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-traj", type=int, default=1000, help="cap trajectories (default ~whole dataset)")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--goal-offset", type=int, default=5)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    # rldp is a PyTorch FB-family flow-BC agent, so it uses the FB loader;
    # only the method label differs.
    df = {"fb": _fb, "rldp": _fb, "gciql": _gciql, "crl": _crl}[args.method](args)
    df["method"] = args.method
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    # per-trajectory goal-peaking score: Spearman(V, -d) along the trajectory.
    rows = []
    for tr, g in df.groupby("traj"):
        rows.append(dict(method=args.method, traj=int(tr), n=len(g),
                         goal_peak_rho=float(_spearman(g["V"].to_numpy(), -g["d"].to_numpy())),
                         V_near=float(g.loc[g["d"].idxmin(), "V"]),
                         V_far=float(g.loc[g["d"].idxmax(), "V"])))
    summ = pd.DataFrame(rows)
    summ.to_csv(out.with_name(out.stem + "_summary.csv"), index=False)
    print(f"[traj_value:{args.method}] {df['traj'].nunique()} trajs, {len(df)} pts; "
          f"goal_peak_rho mean={summ['goal_peak_rho'].mean():.3f} -> {out}")


if __name__ == "__main__":
    main()

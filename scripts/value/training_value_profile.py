"""scripts/value/training_value_profile.py — value vs cube-to-goal-distance on the
offline TRAINING states, per phase, for FB (torch) and GCIQL (jax).

--method fb    : samples aligned (obs, physics, action) from the buffer npz,
                 classifies phases, scores FB policy/data values per task, and
                 writes the shared training_states.npz (run under .venv).
--method gciql : scores GCIQL (or GCIVL) V(s,g) on the SAME shared states
                 (run under .venv-jax-cpu).

Usage (FB):
    .venv/bin/python scripts/value/training_value_profile.py --method fb \
        --config RESULTS/Factored-FB-cube-run/<run>/.hydra/config.yaml \
        --checkpoint RESULTS/Factored-FB-cube-run/<run>/checkpoints/final.pt \
        --data-path datasets --mujoco-gl glfw --n-states 20000 \
        --out analysis/value/training_value

Usage (GCIQL):
    .venv-jax-cpu/bin/python scripts/value/training_value_profile.py --method gciql \
        --run-dir RESULTS/gciql_.../sd003_... --step 1000000 \
        --out analysis/value/training_value --mujoco-gl glfw
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path
from typing import List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals.training_value import (region_labels, cube_to_goal_dist,
                                   horizon_reach_label)

CUBE_SLICE = slice(14, 17)
GRIP_QPOS_IDX = 6      # ur5e/robotiq/right_driver_joint id (verified)
TASK_TMPL = "cube-single-play-singletask-task{n}-v0"


def _episode_files(data_path: str, domain: str) -> List[str]:
    return sorted(glob.glob(str(Path(data_path) / domain / "buffer"
                                / "episode_*.npz")))


def _sample_states(files, n_states, seed=0):
    """Aligned (obs, physics, action) sampled across episodes, plus each
    sampled state's (ep_id, step_id) and the per-episode cube trajectories
    (for the outcome-label tail lookups)."""
    rng = np.random.default_rng(seed)
    obs_l, phys_l, act_l, ep_l, st_l, cube_by_ep = [], [], [], [], [], []
    for ei, f in enumerate(files):
        z = np.load(f)
        o = np.asarray(z["observation"], np.float32)
        ph = np.asarray(z["physics"], np.float32)
        obs_l.append(o)
        phys_l.append(ph)
        act_l.append(np.asarray(z["action"], np.float32))
        ep_l.append(np.full(len(o), ei, dtype=np.int64))
        st_l.append(np.arange(len(o), dtype=np.int64))
        cube_by_ep.append(ph[:, CUBE_SLICE].astype(np.float64))
    obs = np.concatenate(obs_l)
    phys = np.concatenate(phys_l)
    act = np.concatenate(act_l)
    ep_id = np.concatenate(ep_l)
    step_id = np.concatenate(st_l)
    if len(obs) > n_states:
        idx = rng.choice(len(obs), n_states, replace=False)
        obs, phys, act = obs[idx], phys[idx], act[idx]
        ep_id, step_id = ep_id[idx], step_id[idx]
    return obs, phys, act, ep_id, step_id, cube_by_ep


def _phase_composition(files, thr, table_z):
    """Fraction reach/grasp/transport over ALL transitions (trajectory pass)."""
    import pandas as pd
    grip_l, lift_l = [], []
    for f in files:
        z = np.load(f)
        phys = np.asarray(z["physics"], np.float32)
        grip_l.append(np.clip(phys[:, GRIP_QPOS_IDX] / 0.8, 0, 1))
        lift_l.append(phys[:, 16] - table_z)
    reg = region_labels(np.concatenate(grip_l), np.concatenate(lift_l), thr)
    vc = pd.Series(reg).value_counts(normalize=True)
    return pd.DataFrame({"region": vc.index, "fraction": vc.values})


def _outcome_labels(cube_by_ep, ep_id, step_id, goal_xyz, horizon, thresh):
    """[n_states] bool: success_bound per state for one goal — does the cube
    reach within `thresh` of `goal_xyz` within `horizon` future steps of its
    own episode."""
    g = np.asarray(goal_xyz, np.float64).reshape(3)
    out = np.zeros(len(ep_id), dtype=bool)
    for i in range(len(ep_id)):
        tail = cube_by_ep[ep_id[i]][step_id[i]: step_id[i] + horizon]
        d_future = np.linalg.norm(tail - g, axis=1)
        out[i] = horizon_reach_label(d_future, thresh, horizon)
    return out


def _fb(args) -> None:
    import torch
    import pandas as pd
    from evals.analysis import build_env_and_agent, load_cfg, load_checkpoint
    from evals.representation_profile import q_values, v_values
    from evals.ogbench import OGBenchEvaluator
    from envs.ogbench import ALL_TASKS, create_ogbench_env
    from evals.phase_probe import Thresholds
    from data.ogbench import load_ogbench_dataset

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_cfg(args.config, device=args.device)
    if args.data_path:
        cfg.data_path = args.data_path
    env, agent, _, _ = build_env_and_agent(cfg)
    load_checkpoint(agent, args.checkpoint, map_location=args.device)
    if hasattr(env, "close"):
        env.close()

    e0, _ = create_ogbench_env(TASK_TMPL.format(n=1), seed=cfg.seed,
                               obs_type=cfg.obs_type)
    tb = int(getattr(e0.unwrapped, "_target_block", 0) or 0)
    table_z = float(e0.unwrapped.cur_task_info["init_xyzs"][tb][2])
    e0.close()
    thr = Thresholds()

    files = _episode_files(cfg.data_path, cfg.domain)
    obs, phys, act, ep_id, step_id, cube_by_ep = _sample_states(
        files, args.n_states, seed=cfg.seed)
    cube = phys[:, CUBE_SLICE]
    grip = np.clip(phys[:, GRIP_QPOS_IDX] / 0.8, 0, 1)
    region = region_labels(grip, phys[:, 16] - table_z, thr)

    buffer = load_ogbench_dataset(domain=cfg.domain, data_path=cfg.data_path,
                                  load_n_episodes=cfg.load_n_episodes,
                                  device=args.device,
                                  n_transitions=cfg.n_transitions,
                                  obs_type=cfg.obs_type)
    evaluator = OGBenchEvaluator(domain=cfg.domain, agent=agent,
                                 offline_buffer=buffer,
                                 relabel_size=cfg.eval_relabel_size,
                                 n_episodes=1,
                                 shift_reward=cfg.eval_shift_reward,
                                 obs_type=cfg.obs_type, seed=cfg.seed,
                                 device=args.device, use_wandb=False)

    tasks = list(ALL_TASKS.get(cfg.domain, []))
    obs_t = torch.as_tensor(obs, dtype=torch.float32)
    act_t = torch.as_tensor(act, dtype=torch.float32)
    rows, goals, outcome_cols = [], {}, []
    for ti, task in enumerate(tasks, start=1):
        z, _ = evaluator._infer_z(task)
        e, _ = create_ogbench_env(task, seed=cfg.seed, obs_type=cfg.obs_type)
        tbk = int(getattr(e.unwrapped, "_target_block", 0) or 0)
        g = np.asarray(e.unwrapped.cur_task_info["goal_xyzs"][tbk], np.float64)
        e.close()
        goals[f"task{ti}"] = g
        d = cube_to_goal_dist(cube, g)
        oc = _outcome_labels(cube_by_ep, ep_id, step_id, g,
                             args.horizon, args.thresh)
        outcome_cols.append(oc)
        v_pol = v_values(agent.model, agent, obs_t, z)
        v_dat = q_values(agent.model, obs_t, act_t, z)
        for i in range(len(obs)):
            rows.append({"task": f"task{ti}", "region": str(region[i]),
                         "d": float(d[i]), "cube_x": float(cube[i, 0]),
                         "cube_y": float(cube[i, 1]),
                         "V_policy": float(v_pol[i]),
                         "V_data": float(v_dat[i]),
                         "outcome": "success_bound" if oc[i]
                         else "fail_bound"})
    pd.DataFrame(rows).to_parquet(out / "fb_values.parquet")
    _phase_composition(files, thr, table_z).to_parquet(
        out / "phase_composition.parquet")
    outcome_mat = np.stack(outcome_cols, axis=1)  # [n_states, n_tasks] bool
    np.savez(out / "training_states.npz", obs=obs, cube=cube,
             region=region.astype("U12"),
             goals=np.stack([goals[f"task{i}"] for i in range(1, 6)]),
             outcome=outcome_mat)
    print(f"[training_value:fb] {len(obs)} states x {len(tasks)} tasks -> {out}")


def _gciql(args) -> None:
    ogb_impls = REPO_ROOT / "third_party" / "ogbench" / "impls"
    if str(ogb_impls) not in sys.path:
        sys.path.insert(0, str(ogb_impls))
    import jax
    import jax.numpy as jnp
    import pandas as pd
    import ogbench  # vendored, registers envs
    from utils.flax_utils import restore_agent
    from scripts.profiles.gciql_profile import parse_flags

    out = Path(args.out)
    st = np.load(out / "training_states.npz", allow_pickle=True)
    obs = np.asarray(st["obs"], np.float32)
    cube = np.asarray(st["cube"], np.float64)
    region = [str(r) for r in st["region"]]
    if "outcome" not in st:
        raise SystemExit("training_states.npz has no 'outcome' — re-run the "
                         "FB pass (Task 2) before the GCIQL pass.")
    outcome = np.asarray(st["outcome"], dtype=bool)  # [n_states, n_tasks]

    flags = parse_flags(args.run_dir)
    saved = flags.get("agent")
    agent_name = (saved.get("agent_name") if isinstance(saved, dict)
                  else None) or "gciql"
    if agent_name == "gcivl":
        from agents.gcivl import GCIVLAgent as AgentCls, get_config
    else:
        from agents.gciql import GCIQLAgent as AgentCls, get_config
    config = get_config()
    if isinstance(saved, dict):
        for k, v in saved.items():
            if k in config:
                config[k] = v

    env = ogbench.make_env_and_datasets(flags["env_name"], env_only=True)
    ex_obs, _ = env.reset(options=dict(task_id=1))
    agent = AgentCls.create(flags["seed"],
                            np.asarray(ex_obs, np.float32)[None],
                            np.asarray(env.action_space.sample(),
                                       np.float32)[None], config)
    agent = restore_agent(agent, str(args.run_dir), args.step)

    @jax.jit
    def _value(o, g):
        # o,g are batched [N, obs_dim]; value is [N] (gciql) or [2, N]
        # (gcivl twin) -> average the leading twin axis if present.
        v = jnp.asarray(agent.network.select("value")(o, g))
        return v.mean(0) if v.ndim >= 2 else v

    rows = []
    obs_j = jnp.asarray(obs)
    for ti in range(1, 6):
        _, info = env.reset(options=dict(task_id=ti))
        goal = np.asarray(info["goal"], np.float32)
        tbk = int(getattr(env.unwrapped, "_target_block", 0) or 0)
        g_xyz = np.asarray(env.unwrapped.cur_task_info["goal_xyzs"][tbk],
                           np.float64)
        d = cube_to_goal_dist(cube, g_xyz)
        gj = jnp.broadcast_to(jnp.asarray(goal)[None], obs_j.shape)
        V = np.asarray(_value(obs_j, gj)).reshape(-1)
        oc = outcome[:, ti - 1]
        for i in range(len(obs)):
            rows.append({"task": f"task{ti}", "region": region[i],
                         "d": float(d[i]), "cube_x": float(cube[i, 0]),
                         "cube_y": float(cube[i, 1]), "V": float(V[i]),
                         "outcome": "success_bound" if oc[i]
                         else "fail_bound"})
    env.close()
    pd.DataFrame(rows).to_parquet(out / "gciql_values.parquet")
    print(f"[training_value:gciql] {len(obs)} states x 5 tasks -> {out}")


def _crl(args) -> None:
    """CRL+FlowBC (JAX) contrastive value V(s,g)=min(q1,q2)(s,g,pi(s,g)) on the
    SAME shared training_states.npz produced by the FB pass. The crl_flowbc agent
    + ogbench_flow nets live in the wandb_mode_shim, so it goes on path too."""
    ogb = str(REPO_ROOT / "third_party" / "ogbench" / "impls")
    shim = str(REPO_ROOT / "tools" / "wandb_mode_shim")
    for p in (ogb, shim):
        if p in sys.path:
            sys.path.remove(p)
    sys.path.insert(0, ogb)
    sys.path.insert(0, shim)
    import jax
    import jax.numpy as jnp
    import pandas as pd
    import ogbench  # vendored, registers envs
    from utils.flax_utils import restore_agent
    from crl_flowbc import CRLFlowBCAgent, get_config
    from scripts.profiles.gciql_profile import parse_flags

    out = Path(args.out)
    st = np.load(out / "training_states.npz", allow_pickle=True)
    obs = np.asarray(st["obs"], np.float32)
    cube = np.asarray(st["cube"], np.float64)
    region = [str(r) for r in st["region"]]
    if "outcome" not in st:
        raise SystemExit("training_states.npz has no 'outcome' — run the FB pass first.")
    outcome = np.asarray(st["outcome"], dtype=bool)  # [n_states, n_tasks]

    flags = parse_flags(args.run_dir)
    saved = flags.get("agent")
    config = get_config()
    if isinstance(saved, dict):
        for k, v in saved.items():
            if k in config:
                config[k] = v

    env = ogbench.make_env_and_datasets(flags["env_name"], env_only=True)
    ex_obs, _ = env.reset(options=dict(task_id=1))
    agent = CRLFlowBCAgent.create(flags["seed"], np.asarray(ex_obs, np.float32)[None],
                                  np.asarray(env.action_space.sample(), np.float32)[None], config)
    agent = restore_agent(agent, str(args.run_dir), args.step)
    key = jax.random.PRNGKey(0)

    def _value(o, g):
        a = agent.sample_actions(o, g, seed=key)
        q1, q2 = agent.network.select("critic")(o, g, a)
        return np.asarray(jnp.minimum(q1, q2)).reshape(-1)

    rows = []
    obs_j = jnp.asarray(obs)
    for ti in range(1, 6):
        _, info = env.reset(options=dict(task_id=ti))
        goal = np.asarray(info["goal"], np.float32)
        tbk = int(getattr(env.unwrapped, "_target_block", 0) or 0)
        g_xyz = np.asarray(env.unwrapped.cur_task_info["goal_xyzs"][tbk], np.float64)
        d = cube_to_goal_dist(cube, g_xyz)
        gj = jnp.broadcast_to(jnp.asarray(goal)[None], obs_j.shape)
        V = _value(obs_j, gj)
        oc = outcome[:, ti - 1]
        for i in range(len(obs)):
            rows.append({"task": f"task{ti}", "region": region[i],
                         "d": float(d[i]), "cube_x": float(cube[i, 0]),
                         "cube_y": float(cube[i, 1]), "V": float(V[i]),
                         "outcome": "success_bound" if oc[i]
                         else "fail_bound"})
    env.close()
    pd.DataFrame(rows).to_parquet(out / "crl_values.parquet")
    print(f"[training_value:crl] {len(obs)} states x 5 tasks -> {out}")


def _rldp(args) -> None:
    """RLDP+FlowBC (PyTorch, FB-family) policy value V = Q(s, pi(s, z)) on the
    SAME shared training_states.npz produced by the FB pass. RLDP reuses the FB
    torch scoring; only the source checkpoint and output filename differ. The
    extra `_predictor` head is dropped on load (evals.analysis.load_checkpoint)."""
    import torch
    import pandas as pd
    from evals.analysis import build_env_and_agent, load_cfg, load_checkpoint
    from evals.representation_profile import v_values
    from evals.ogbench import OGBenchEvaluator
    from envs.ogbench import ALL_TASKS
    from data.ogbench import load_ogbench_dataset

    out = Path(args.out)
    st = np.load(out / "training_states.npz", allow_pickle=True)
    obs = np.asarray(st["obs"], np.float32)
    cube = np.asarray(st["cube"], np.float64)
    region = [str(r) for r in st["region"]]
    if "outcome" not in st:
        raise SystemExit("training_states.npz has no 'outcome' — run the FB pass first.")
    outcome = np.asarray(st["outcome"], dtype=bool)  # [n_states, n_tasks]
    goals = np.asarray(st["goals"], np.float64)       # [5, 3]

    cfg = load_cfg(args.config, device=args.device)
    if args.data_path:
        cfg.data_path = args.data_path
    env, agent, _, _ = build_env_and_agent(cfg)
    load_checkpoint(agent, args.checkpoint, map_location=args.device)
    if hasattr(env, "close"):
        env.close()

    buffer = load_ogbench_dataset(domain=cfg.domain, data_path=cfg.data_path,
                                  load_n_episodes=cfg.load_n_episodes,
                                  device=args.device,
                                  n_transitions=cfg.n_transitions,
                                  obs_type=cfg.obs_type)
    evaluator = OGBenchEvaluator(domain=cfg.domain, agent=agent,
                                 offline_buffer=buffer,
                                 relabel_size=cfg.eval_relabel_size,
                                 n_episodes=1,
                                 shift_reward=cfg.eval_shift_reward,
                                 obs_type=cfg.obs_type, seed=cfg.seed,
                                 device=args.device, use_wandb=False)

    tasks = list(ALL_TASKS.get(cfg.domain, []))
    obs_t = torch.as_tensor(obs, dtype=torch.float32)
    rows = []
    for ti, task in enumerate(tasks, start=1):
        z, _ = evaluator._infer_z(task)
        g = goals[ti - 1]
        d = cube_to_goal_dist(cube, g)
        v_pol = v_values(agent.model, agent, obs_t, z)
        oc = outcome[:, ti - 1]
        for i in range(len(obs)):
            rows.append({"task": f"task{ti}", "region": region[i],
                         "d": float(d[i]), "cube_x": float(cube[i, 0]),
                         "cube_y": float(cube[i, 1]), "V": float(v_pol[i]),
                         "outcome": "success_bound" if oc[i]
                         else "fail_bound"})
    pd.DataFrame(rows).to_parquet(out / "rldp_values.parquet")
    print(f"[training_value:rldp] {len(obs)} states x {len(tasks)} tasks -> {out}")


def _tdmpc2(args) -> None:
    """TD-MPC2 (PyTorch, model-based MPC) policy value on the cube training
    states. The MPC planner is far too slow to evaluate per state, so we use the
    learned policy-prior value V = Q(z, pi_mean(z)) — the same quantity the MPPI
    value target bootstraps from — where z = encode(fold(obs, goal_xyz)). The
    goal context is the cube goal xyz directly (no _infer_z / backward map).

    Reuses the shared training_states.npz when present (cross-method comparison,
    like the RLDP pass); otherwise generates it (like the FB pass) so TD-MPC2 can
    bootstrap the shared state set on its own."""
    import torch
    import pandas as pd
    from evals.analysis import build_env_and_agent, load_cfg, load_checkpoint
    from evals.ogbench import OGBenchEvaluator  # noqa: F401  (parity w/ other passes)
    from envs.ogbench import ALL_TASKS, create_ogbench_env
    from evals.phase_probe import Thresholds
    from data.ogbench import load_ogbench_dataset

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_cfg(args.config, device=args.device)
    if args.data_path:
        cfg.data_path = args.data_path
    env, agent, _, _ = build_env_and_agent(cfg)
    load_checkpoint(agent, args.checkpoint, map_location=args.device)
    agent.eval() if hasattr(agent, "eval") else None
    if hasattr(env, "close"):
        env.close()

    thr = Thresholds()
    npz_path = out / "training_states.npz"
    if npz_path.exists():
        st = np.load(npz_path, allow_pickle=True)
        obs = np.asarray(st["obs"], np.float32)
        cube = np.asarray(st["cube"], np.float64)
        region = [str(r) for r in st["region"]]
        if "outcome" not in st:
            raise SystemExit("training_states.npz has no 'outcome' — run the FB pass first.")
        outcome = np.asarray(st["outcome"], dtype=bool)
        goals = np.asarray(st["goals"], np.float64)
        tasks = list(ALL_TASKS.get(cfg.domain, []))
    else:
        # Bootstrap the shared state set ourselves (mirrors the FB pass).
        e0, _ = create_ogbench_env(TASK_TMPL.format(n=1), seed=cfg.seed,
                                   obs_type=cfg.obs_type)
        tb = int(getattr(e0.unwrapped, "_target_block", 0) or 0)
        table_z = float(e0.unwrapped.cur_task_info["init_xyzs"][tb][2])
        e0.close()
        files = _episode_files(cfg.data_path, cfg.domain)
        obs, phys, act, ep_id, step_id, cube_by_ep = _sample_states(
            files, args.n_states, seed=cfg.seed)
        cube = phys[:, CUBE_SLICE]
        grip = np.clip(phys[:, GRIP_QPOS_IDX] / 0.8, 0, 1)
        region_arr = region_labels(grip, phys[:, 16] - table_z, thr)
        region = [str(r) for r in region_arr]
        tasks = list(ALL_TASKS.get(cfg.domain, []))
        goals_l, outcome_cols = [], []
        for ti, task in enumerate(tasks, start=1):
            e, _ = create_ogbench_env(task, seed=cfg.seed, obs_type=cfg.obs_type)
            tbk = int(getattr(e.unwrapped, "_target_block", 0) or 0)
            g = np.asarray(e.unwrapped.cur_task_info["goal_xyzs"][tbk], np.float64)
            e.close()
            goals_l.append(g)
            outcome_cols.append(_outcome_labels(cube_by_ep, ep_id, step_id, g,
                                                 args.horizon, args.thresh))
        goals = np.stack(goals_l)                       # [5, 3]
        outcome = np.stack(outcome_cols, axis=1)        # [n_states, n_tasks]
        _phase_composition(files, thr, table_z).to_parquet(
            out / "phase_composition.parquet")
        np.savez(npz_path, obs=obs, cube=cube,
                 region=region_arr.astype("U12"), goals=goals, outcome=outcome)

    @torch.no_grad()
    def tdmpc2_v(obs_np, goal_xyz, chunk=8192):
        dev = agent.device
        model = agent.core.model
        g = torch.as_tensor(goal_xyz, dtype=torch.float32, device=dev).reshape(3)
        vals = []
        for i in range(0, len(obs_np), chunk):
            ob = torch.as_tensor(obs_np[i:i + chunk], dtype=torch.float32,
                                 device=dev)
            folded = agent._fold(ob, g.expand(ob.shape[0], 3))
            z = model.encode(folded, None)
            _a, info = model.pi(z, None)
            v = model.Q(z, info["mean"], None, return_type="min")
            vals.append(v.reshape(-1).cpu().numpy())
        return np.concatenate(vals)

    rows = []
    for ti, task in enumerate(tasks, start=1):
        g = goals[ti - 1]
        d = cube_to_goal_dist(cube, g)
        v = tdmpc2_v(obs, g)
        oc = outcome[:, ti - 1]
        for i in range(len(obs)):
            rows.append({"task": f"task{ti}", "region": region[i],
                         "d": float(d[i]), "cube_x": float(cube[i, 0]),
                         "cube_y": float(cube[i, 1]), "V": float(v[i]),
                         "outcome": "success_bound" if oc[i] else "fail_bound"})
    pd.DataFrame(rows).to_parquet(out / "tdmpc2_values.parquet")
    print(f"[training_value:tdmpc2] {len(obs)} states x {len(tasks)} tasks -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--method", required=True,
                    choices=["fb", "gciql", "crl", "rldp", "tdmpc2"])
    ap.add_argument("--out", default="analysis/value/training_value")
    ap.add_argument("--n-states", type=int, default=20000)
    ap.add_argument("--mujoco-gl", default=None)
    # fb
    ap.add_argument("--config")
    ap.add_argument("--checkpoint")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--data-path", default="datasets")
    ap.add_argument("--horizon", type=int, default=200)
    ap.add_argument("--thresh", type=float, default=0.04)
    # gciql
    ap.add_argument("--run-dir")
    ap.add_argument("--step", type=int, default=1000000)
    args = ap.parse_args()
    if args.mujoco_gl:
        os.environ["MUJOCO_GL"] = args.mujoco_gl
    if args.method == "fb":
        assert args.config and args.checkpoint, "fb needs --config/--checkpoint"
        _fb(args)
    elif args.method == "rldp":
        assert args.config and args.checkpoint, "rldp needs --config/--checkpoint"
        _rldp(args)
    elif args.method == "tdmpc2":
        assert args.config and args.checkpoint, "tdmpc2 needs --config/--checkpoint"
        _tdmpc2(args)
    elif args.method == "crl":
        assert args.run_dir, "crl needs --run-dir"
        _crl(args)
    else:
        assert args.run_dir, "gciql needs --run-dir"
        _gciql(args)


if __name__ == "__main__":
    main()

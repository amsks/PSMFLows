#!/usr/bin/env python
"""scripts/value/repsep_extract_jax.py — grasp-state representations for GCIQL / CRL.

Mirror of scripts/value/repsep_extract_torch.py for the JAX agents. Emits the same
analysis/value/repsep/<method>.parquet schema {split, task, success, rep_*} so the
shared plotter (PCA + linear-probe AUC) is method-agnostic.

Representation per method (the learned state embedding its score reads):
  - GCIQL: penultimate features of the goal-conditioned value MLP V(s, g)
           (the post-LayerNorm activation feeding the scalar head), conditioned
           on the eval task's goal observation.
  - CRL:   the state-side contrastive embedding phi(s, .) from the bilinear
           critic (goal-independent; psi carries the goal). Actions are zeroed
           so it is a pure state representation, consistent train vs eval.

Grasp/carry states only (cube in hand): training grasps from the shared
training_states.npz (region in {grasp, transport}, label = per-task
success-bound); eval grasps from deterministic rollouts (in-hand steps via
transport_mask, label = episode success).

Run under .venv-jax-cpu (jax + vendored OGBench).

Usage:
    .venv-jax-cpu/bin/python scripts/value/repsep_extract_jax.py --method gciql \
        --run-dir results/.../<gciql seed dir> --step 1000000 \
        --out analysis/value/repsep/gciql.parquet
    .venv-jax-cpu/bin/python scripts/value/repsep_extract_jax.py --method crl \
        --checkpoint-dir results/factored-fb-crl-flowbc/<seed dir> \
        --step 1000000 --seed 0 --out analysis/value/repsep/crl.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CONTACT = {"grasp", "transport"}   # cube in hand (pickup + carry regimes)
N_TASKS = 5


# ---------------------------------------------------------------------------
# Representation hooks
# ---------------------------------------------------------------------------
def _gciql_phi_fn(agent, value_hidden_dims):
    """Return fn(obs[B,O], goal[O]) -> penultimate value features [B, H]."""
    import jax
    import jax.numpy as jnp

    h_pen = int(value_hidden_dims[-1])

    def _select_penultimate(intermediates):
        """Pick the last post-LayerNorm hidden activation of width h_pen."""
        flat = jax.tree_util.tree_flatten_with_path(intermediates)[0]
        # entries are (key_path, array); keep arrays whose last dim == h_pen
        cands = []
        for path, arr in flat:
            a = jnp.asarray(arr)
            if a.shape[-1] != h_pen:
                continue
            key = "/".join(str(getattr(k, "key", k)) for k in path)
            cands.append((key, a))
        # prefer LayerNorm outputs (post-activation hidden); else any match.
        ln = [c for c in cands if "LayerNorm" in c[0]]
        pool = ln if ln else cands
        # deterministic: last in flatten order (deepest/last layer)
        return pool[-1][1], pool[-1][0]

    @jax.jit
    def _phi(obs, goal):
        g = jnp.broadcast_to(goal[None], (obs.shape[0], goal.shape[-1]))
        _, mod = agent.network.model_def.apply(
            {"params": agent.network.params}, obs, g, name="value",
            capture_intermediates=True, mutable=["intermediates"])
        rep, _ = _select_penultimate(mod["intermediates"])
        if rep.ndim == 3:          # [ensemble, B, H] -> mean over ensemble
            rep = rep.mean(axis=0)
        return rep

    return _phi


def _crl_phi_fn(agent):
    """Return fn(obs[B,O]) -> state-side contrastive embedding phi [B, d]."""
    import jax

    @jax.jit
    def _phi(obs, act):
        _, phi, _ = agent.network.select("critic")(obs, obs, act, info=True)
        if phi.ndim == 3:          # [ensemble, B, d] -> mean over ensemble
            phi = phi.mean(axis=0)
        return phi

    return _phi


# ---------------------------------------------------------------------------
# Agent / env loading
# ---------------------------------------------------------------------------
def _load_gciql(run_dir: str, step: int):
    """Load a GCIQL agent + per-task goal obs (mirror gciql_profile)."""
    ogb = str(REPO_ROOT / "third_party" / "ogbench" / "impls")
    if ogb in sys.path:
        sys.path.remove(ogb)
    sys.path.insert(0, ogb)
    import ogbench  # vendored
    from utils.flax_utils import restore_agent
    from scripts.profiles.gciql_profile import parse_flags

    flags = parse_flags(run_dir)
    saved = flags.get("agent")
    name = (saved.get("agent_name") if isinstance(saved, dict) else None) or "gciql"
    if name == "gcivl":
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
    ex_act = env.action_space.sample()
    agent = AgentCls.create(flags["seed"],
                            np.asarray(ex_obs, np.float32)[None],
                            np.asarray(ex_act, np.float32)[None], config)
    agent = restore_agent(agent, str(run_dir), step)
    env.close()
    return agent, config, flags["env_name"]


def _gciql_task_env(env_name, task_id):
    import ogbench
    return ogbench.make_env_and_datasets(env_name, env_only=True)


# ---------------------------------------------------------------------------
# Eval rollouts -> in-hand obs + episode success
# ---------------------------------------------------------------------------
def _rollout_gciql(agent, env_name, task_ids, n_episodes, max_steps):
    import jax
    from evals.phase_probe import Thresholds
    from evals._profile_core import transport_mask
    thr = Thresholds()
    out: Dict[int, Dict[str, Any]] = {}
    eps_by_task: Dict[int, List[Dict[str, Any]]] = {}
    goals: Dict[int, np.ndarray] = {}
    for task_id in task_ids:
        env = _gciql_task_env(env_name, task_id)
        eps_by_task[task_id] = []
        rng = jax.random.PRNGKey(0)
        for ep_i in range(n_episodes):
            obs, info = env.reset(options=dict(task_id=task_id))
            goal = np.asarray(info["goal"], np.float32)
            goals.setdefault(task_id, goal)
            O, CUBE, GRIP, EFF = [], [], [], []
            u = env.unwrapped
            tb = int(getattr(u, "_target_block", 0) or 0)
            table_z = float(u.cur_task_info["init_xyzs"][tb][2])
            succ = False
            for _ in range(max_steps):
                rng, key = jax.random.split(rng)
                a = np.clip(np.asarray(agent.sample_actions(
                    observations=np.asarray(obs, np.float32),
                    goals=goal, seed=key, temperature=0.0)), -1.0, 1.0)
                obs, _, term, trunc, info = env.step(a)
                O.append(np.asarray(obs, np.float32))
                CUBE.append(np.asarray(info["privileged/block_0_pos"], np.float64))
                GRIP.append(float(np.asarray(info["proprio/gripper_opening"]).reshape(-1)[0]))
                EFF.append(np.asarray(info["proprio/effector_pos"], np.float64))
                succ = succ or bool(info.get("success", False))
                if term or trunc:
                    break
            sig = {"obs": np.asarray(O, np.float32), "cube": np.asarray(CUBE),
                   "grip": np.asarray(GRIP), "eff": np.asarray(EFF),
                   "table_z": table_z, "success": succ, "length": len(O)}
            sig["transport_mask"] = transport_mask(sig, thr)
            eps_by_task[task_id].append(sig)
        env.close()
    return eps_by_task, goals


def _rollout_crl(checkpoint_dir, step, seed, tasks, n_episodes, max_steps):
    from scripts.probes.phase_probe_crl import load_agent, make_env, EVAL_TASKS
    from scripts.coverage.crl_coverage import rollout_episode_obs
    from evals.phase_probe import Thresholds, ensure_manip_env
    from evals._profile_core import transport_mask
    thr = Thresholds()
    agent = load_agent(Path(checkpoint_dir).resolve(), step, seed)
    task_list = tasks or EVAL_TASKS
    eps_by_task: Dict[str, List[Dict[str, Any]]] = {}
    for ti, task in enumerate(task_list, start=1):
        env = make_env(task)
        ensure_manip_env(env)
        eps_by_task[ti] = []
        for ep in range(n_episodes):
            sig = rollout_episode_obs(agent, env, seed * 100_000 + ep, max_steps)
            sig["transport_mask"] = transport_mask(sig, thr)
            eps_by_task[ti].append(sig)
        env.close()
    return agent, eps_by_task


# ---------------------------------------------------------------------------
# Row builders
# ---------------------------------------------------------------------------
def _eval_rows(phi_fn, eps_by_task, method, goals=None):
    rows = []
    gep = 0                                  # global episode id (unique per rollout)
    for ti, eps in eps_by_task.items():
        for ep in eps:
            mask = np.asarray(ep["transport_mask"], bool)
            if mask.sum() == 0:
                gep += 1
                continue
            obs = np.asarray(ep["obs"], np.float32)[mask]
            success = bool(ep["success"])
            if method == "gciql":
                rep = np.asarray(phi_fn(obs, goals[ti]))
            else:
                rep = np.asarray(phi_fn(obs, np.zeros((obs.shape[0],
                                  _act_dim(method)), np.float32)))
            for r in rep:
                rows.append(("eval", ti, gep, success, r))
            gep += 1
    return rows


def _train_rows(phi_fn, states_npz, method, goals=None):
    st = np.load(states_npz, allow_pickle=True)
    obs = np.asarray(st["obs"], np.float32)
    region = np.array([str(r) for r in st["region"]])
    outcome = np.asarray(st["outcome"], bool)
    idx = np.where(np.isin(region, list(CONTACT)))[0]
    sub = obs[idx]
    rows = []
    for ti in range(N_TASKS):
        if method == "gciql":
            rep = np.asarray(phi_fn(sub, goals[ti + 1]))
        else:
            rep = np.asarray(phi_fn(sub, np.zeros((sub.shape[0],
                             _act_dim(method)), np.float32)))
        for j, i in enumerate(idx):
            rows.append(("train", ti + 1, -1, bool(outcome[i, ti]), rep[j]))
    return rows


_ACT_DIM = {}


def _act_dim(method):
    return _ACT_DIM.get(method, 5)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--method", required=True, choices=["gciql", "crl"])
    ap.add_argument("--run-dir", help="GCIQL seed dir (flags.json + params_*)")
    ap.add_argument("--checkpoint-dir", help="CRL seed dir")
    ap.add_argument("--step", type=int, default=1000000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--states",
                    default="analysis/value/training_value_multiseed/p0/training_states.npz")
    ap.add_argument("--tasks", default=None, help="CRL: comma task names")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="1 task / 1 episode; print shapes; no write")
    args = ap.parse_args()

    n_ep = 1 if args.smoke else args.episodes

    if args.method == "gciql":
        assert args.run_dir, "--run-dir required for gciql"
        agent, config, env_name = _load_gciql(args.run_dir, args.step)
        vhd = config.get("value_hidden_dims", (512, 512, 512))
        phi_fn = _gciql_phi_fn(agent, vhd)
        task_ids = [1] if args.smoke else [1, 2, 3, 4, 5]
        eps_by_task, goals = _rollout_gciql(agent, env_name, task_ids, n_ep,
                                            args.max_steps)
        # probe penultimate dim
        any_ep = next(iter(eps_by_task.values()))[0]
        m = np.asarray(any_ep["transport_mask"], bool)
        probe_obs = np.asarray(any_ep["obs"], np.float32)[m][:4] if m.sum() else \
            np.asarray(any_ep["obs"], np.float32)[:4]
        rep_dim = np.asarray(phi_fn(probe_obs, goals[task_ids[0]])).shape[-1]
        print(f"[repsep:gciql] value_hidden_dims={tuple(vhd)} rep_dim={rep_dim}")
        eval_rows = _eval_rows(phi_fn, eps_by_task, "gciql", goals)
        train_rows = [] if args.smoke else _train_rows(phi_fn, args.states,
                                                       "gciql", goals)
    else:
        assert args.checkpoint_dir, "--checkpoint-dir required for crl"
        tasks = args.tasks.split(",") if args.tasks else None
        agent, eps_by_task = _rollout_crl(args.checkpoint_dir, args.step,
                                          args.seed, tasks, n_ep, args.max_steps)
        # CRL action dim from env action space (cube-single = 5)
        import ogbench
        from scripts.probes.phase_probe_crl import EVAL_TASKS
        e = ogbench.make_env_and_datasets(EVAL_TASKS[0], env_only=True)
        _ACT_DIM["crl"] = int(np.asarray(e.action_space.sample()).reshape(-1).shape[0])
        e.close()
        phi_fn = _crl_phi_fn(agent)
        probe_obs = None
        for eps in eps_by_task.values():
            for ep in eps:
                m = np.asarray(ep["transport_mask"], bool)
                if m.sum():
                    probe_obs = np.asarray(ep["obs"], np.float32)[m][:4]
                    break
            if probe_obs is not None:
                break
        if probe_obs is not None:
            rep_dim = np.asarray(phi_fn(probe_obs, np.zeros((probe_obs.shape[0],
                                 _act_dim("crl")), np.float32))).shape[-1]
            print(f"[repsep:crl] act_dim={_act_dim('crl')} rep_dim={rep_dim}")
        eval_rows = _eval_rows(phi_fn, eps_by_task, "crl")
        train_rows = [] if args.smoke else _train_rows(phi_fn, args.states, "crl")

    rows = train_rows + eval_rows
    if args.smoke:
        ev = [r for r in eval_rows]
        import collections
        c = collections.Counter((s, y) for s, t, e, y, r in ev)
        print(f"[smoke] eval rows={len(ev)} by (split,success)={dict(c)}")
        if ev:
            print(f"[smoke] rep[0] dim={len(ev[0][4])} sample={ev[0][4][:5]}")
        return 0

    df = pd.DataFrame([{"split": s, "task": t, "episode": e, "success": y,
                        **{f"rep_{i}": float(v) for i, v in enumerate(r)}}
                       for s, t, e, y, r in rows])
    out = Path(args.out or f"analysis/value/repsep/{args.method}.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    k = sum(c.startswith("rep_") for c in df.columns)
    print(f"[repsep:{args.method}] {len(df)} rows (dim {k}); "
          f"{df.groupby('split')['success'].agg(['size', 'mean']).to_dict()} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

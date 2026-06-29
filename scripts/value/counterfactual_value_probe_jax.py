#!/usr/bin/env python
"""scripts/value/counterfactual_value_probe_jax.py — counterfactual value aliasing, GCIQL/CRL.

JAX companion to counterfactual_value_probe.py. For each in-hand grasp sampled
from the play data, reset the sim to it (set_state + hold gripper), roll out the
agent's own policy, and label by whether it reaches the goal; then measure
whether the value ranks grasps by that counterfactual success.

  GCIQL: V(s, g) = value(s, g)
  CRL:   Q(s, pi(s,g), g) = phi(s,a).psi(g)

Writes analysis/value/repsep/cf_value_<method>.json and cf_records_<method>.parquet.
Run under .venv-jax-cpu.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

N_TASKS = 5


def _auc(y, s):
    y = np.asarray(y).astype(bool); s = np.asarray(s, float)
    n1, n2 = int(y.sum()), int((~y).sum())
    if n1 < 5 or n2 < 5:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float); ranks[order] = np.arange(1, len(s) + 1)
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n2))


def _establish_grasp(u, qpos, mujoco):
    nv = u._model.nv
    u.set_state(np.asarray(qpos, np.float64), np.zeros(nv))
    u._data.ctrl[u._gripper_actuator_ids] = 0.0
    mujoco.mj_forward(u._model, u._data)
    return np.asarray(u.compute_observation(), np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--method", required=True, choices=["gciql", "crl"])
    ap.add_argument("--run-dir")
    ap.add_argument("--checkpoint-dir")
    ap.add_argument("--step", type=int, default=1000000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--domain", default="cube-single-play-v0")
    ap.add_argument("--data-path", default="datasets")
    ap.add_argument("--n-states", type=int, default=200)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--out-tag", default="",
                    help="suffix for output files, e.g. '_seed03' (avoids clobbering the primary run)")
    args = ap.parse_args()

    import mujoco
    import jax
    import jax.numpy as jnp
    import pandas as pd
    from scripts.value.counterfactual_value_probe import sample_grasps

    obs_g, phys_g = sample_grasps(args.data_path, args.domain, args.n_states)

    if args.method == "gciql":
        assert args.run_dir
        import ogbench
        from scripts.value.repsep_extract_jax import _load_gciql
        agent, _cfg, env_name = _load_gciql(args.run_dir, args.step)

        @jax.jit
        def value(O, G):
            v = jnp.asarray(agent.network.select("value")(O, G))
            return v.mean(0) if v.ndim > 1 else v

        def make_task_env(ti):
            e = ogbench.make_env_and_datasets(env_name, env_only=True)
            _o, info = e.reset(options=dict(task_id=ti))
            return e, np.asarray(info["goal"], np.float32)

        def reset_task(e, ti):
            e.reset(options=dict(task_id=ti))

    else:  # crl
        assert args.checkpoint_dir
        from scripts.probes.phase_probe_crl import EVAL_TASKS, load_agent, make_env
        agent = load_agent(Path(args.checkpoint_dir).resolve(), args.step, args.seed)

        @jax.jit
        def qcrit(O, G, A):
            v = jnp.asarray(agent.network.select("critic")(O, G, A))
            return v.mean(0) if v.ndim > 1 else v

        def make_task_env(ti):
            e = make_env(EVAL_TASKS[ti - 1])
            _o, info = e.reset(seed=args.seed)
            g = info.get("goal")
            if g is None:
                from evals.phase_probe import episode_goal
                g = episode_goal(e)
            return e, np.asarray(g, np.float32)

        def reset_task(e, ti):
            e.reset(seed=args.seed)

    print(f"[cf-jax:{args.method}] {len(obs_g)} grasps x {N_TASKS} goals "
          f"x {args.max_steps} steps")
    per_task_auc, per_task_sr, records = [], [], []
    rng = jax.random.PRNGKey(args.seed)
    for ti in range(1, N_TASKS + 1):
        e, goal = make_task_env(ti)
        u = e.unwrapped
        G = np.broadcast_to(goal[None], obs_g.shape).astype(np.float32)
        if args.method == "gciql":
            val = np.asarray(value(obs_g, G))
        else:
            rng, k = jax.random.split(rng)
            A = np.clip(np.asarray(agent.sample_actions(
                observations=obs_g, goals=G, seed=k, temperature=0.0)), -1, 1).astype(np.float32)
            val = np.asarray(qcrit(obs_g, G, A))
        succ = np.zeros(len(phys_g), bool)
        for j, qpos in enumerate(phys_g):
            reset_task(e, ti)
            ob = _establish_grasp(u, qpos, mujoco)
            s = False
            for _ in range(args.max_steps):
                rng, k = jax.random.split(rng)
                a = np.clip(np.asarray(agent.sample_actions(
                    observations=ob, goals=goal, seed=k, temperature=0.0)), -1, 1)
                ob_, _, term, trunc, info = e.step(a)
                ob = np.asarray(ob_, np.float32)
                s = s or bool(info.get("success", False))
                if term or trunc:
                    break
            succ[j] = s
        if hasattr(e, "close"):
            e.close()
        auc = _auc(succ, val)
        per_task_auc.append(auc); per_task_sr.append(float(succ.mean()))
        for j in range(len(obs_g)):
            records.append((ti, bool(succ[j]), float(val[j]), obs_g[j]))
        print(f"  task{ti}: cf success {succ.mean():.2f}  value-AUC {auc:.3f}")

    pd.DataFrame([{"task": t, "success": s, "value": v,
                   **{f"obs_{i}": float(o[i]) for i in range(o.shape[0])}}
                  for t, s, v, o in records]).to_parquet(
        REPO / f"analysis/value/repsep/cf_records_{args.method}{args.out_tag}.parquet")
    per_task_auc = np.array(per_task_auc, float)
    out = REPO / f"analysis/value/repsep/cf_value_{args.method}{args.out_tag}.json"
    out.write_text(json.dumps({
        "method": args.method, "n_states": int(len(obs_g)),
        "cf_value_auc_mean": float(np.nanmean(per_task_auc)),
        "cf_value_auc_per_task": per_task_auc.tolist(),
        "cf_success_rate": per_task_sr}, indent=2))
    print(f"[cf-jax:{args.method}] value-AUC mean {np.nanmean(per_task_auc):.3f} "
          f"(cf success {np.mean(per_task_sr):.2f}) -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

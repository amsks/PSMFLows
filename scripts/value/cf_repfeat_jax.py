#!/usr/bin/env python
"""scripts/value/cf_repfeat_jax.py — dump GCIQL/CRL representations on the cf grasps.

Reads the counterfactual grasp records (obs + agent's own success + value) and
emits, per grasp, the method's learned state representation so the linear-probe
ladder can be computed under .venv (sklearn). GCIQL rep = penultimate features of
the goal-conditioned value net; CRL rep = the contrastive state embedding
phi(s, pi(s,g)). Writes analysis/value/repsep/cf_repfeat_<method>.parquet
{task, success, value, obs_*, rep_*}. Run under .venv-jax-cpu.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

N_TASKS = 5


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--method", required=True, choices=["gciql", "crl"])
    ap.add_argument("--run-dir")
    ap.add_argument("--checkpoint-dir")
    ap.add_argument("--step", type=int, default=1000000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import jax
    rec = pd.read_parquet(REPO / f"analysis/value/repsep/cf_records_{args.method}.parquet")
    obs_cols = [c for c in rec.columns if c.startswith("obs_")]

    if args.method == "gciql":
        assert args.run_dir
        import ogbench
        from scripts.value.repsep_extract_jax import _load_gciql, _gciql_phi_fn
        agent, config, env_name = _load_gciql(args.run_dir, args.step)
        phi_fn = _gciql_phi_fn(agent, config.get("value_hidden_dims", (512, 512, 512)))
        goals = {}
        for ti in range(1, N_TASKS + 1):
            e = ogbench.make_env_and_datasets(env_name, env_only=True)
            _o, info = e.reset(options=dict(task_id=ti)); goals[ti] = np.asarray(info["goal"], np.float32); e.close()

        def rep_of(obs, ti):
            return np.asarray(phi_fn(obs, goals[ti]))
    else:
        assert args.checkpoint_dir
        from scripts.probes.phase_probe_crl import EVAL_TASKS, load_agent, make_env
        from scripts.value.repsep_extract_jax import _crl_phi_fn
        agent = load_agent(Path(args.checkpoint_dir).resolve(), args.step, args.seed)
        phi_fn = _crl_phi_fn(agent)
        goals = {}
        for ti, task in enumerate(EVAL_TASKS, start=1):
            e = make_env(task); _o, info = e.reset(seed=args.seed)
            g = info.get("goal")
            if g is None:
                from evals.phase_probe import episode_goal
                g = episode_goal(e)
            goals[ti] = np.asarray(g, np.float32)
            if hasattr(e, "close"):
                e.close()

        rng = jax.random.PRNGKey(args.seed)

        def rep_of(obs, ti):
            nonlocal rng
            G = np.broadcast_to(goals[ti][None], obs.shape).astype(np.float32)
            rng, k = jax.random.split(rng)
            a = np.clip(np.asarray(agent.sample_actions(observations=obs, goals=G,
                        seed=k, temperature=0.0)), -1, 1).astype(np.float32)
            return np.asarray(phi_fn(obs, a))

    out_rows = []
    for ti in range(1, N_TASKS + 1):
        d = rec[rec["task"] == ti]
        obs = d[obs_cols].to_numpy(np.float32)
        rep = rep_of(obs, ti)
        for k in range(len(d)):
            out_rows.append((ti, bool(d["success"].iloc[k]), float(d["value"].iloc[k]),
                             obs[k], rep[k]))
    df = pd.DataFrame([{"task": t, "success": s, "value": v,
                        **{f"obs_{i}": float(o[i]) for i in range(o.shape[0])},
                        **{f"rep_{i}": float(r[i]) for i in range(r.shape[0])}}
                       for t, s, v, o, r in out_rows])
    out = REPO / f"analysis/value/repsep/cf_repfeat_{args.method}.parquet"
    df.to_parquet(out)
    print(f"[cf_repfeat:{args.method}] {len(df)} rows, rep dim "
          f"{sum(c.startswith('rep_') for c in df.columns)} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

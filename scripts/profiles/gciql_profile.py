"""scripts/profiles/gciql_profile.py — per-checkpoint GCIQL profiler.

Runs ONLY in the isolated JAX venv (it imports jax/flax/vendored
OGBench inside main()/_jax_*). The pure seams below (flags parsing,
episode-records -> frames) are JAX-free and unit-tested.

T1 (Spearman rho(V(s,g), -d)), T4 (off-support coverage) and the phase
funnel are computed with evals._profile_core / evals.phase_probe so the
FB and GCIQL numbers are byte-identical.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals._profile_core import (
    _region, _spearman, probe_coverage, transport_mask)
from evals.phase_probe import Thresholds, classify_phases


def parse_flags(run_dir) -> Dict[str, Any]:
    """Read OGBench flags.json from a GCIQL seed dir."""
    data = json.loads((Path(run_dir) / "flags.json").read_text())
    return {"env_name": data.get("env_name", "cube-single-play-v0"),
            "seed": int(data.get("seed", 0)),
            "agent": data.get("agent", "agents/gciql.py"),
            "obs_type": data.get("obs_type", "state")}


def episodes_to_frames(episodes: List[Dict[str, Any]], thr: Thresholds,
                       ref_obs: np.ndarray, feature: str = "obs"):
    """Episode records -> (value_landscape, coverage, funnel) frames.

    Each record: obs[T,O], d[T], V[T], grip[T], cube[T,3], table_z,
    success(bool), task(str), episode(int). Pure; no jax.
    """
    vl_rows, fun_rows, cov_eps, vs_rows = [], [], [], []
    for ep in episodes:
        d = np.asarray(ep["d"], np.float64)
        V = np.asarray(ep["V"], np.float64)
        tm = transport_mask(ep, thr)
        sig = {
            "eff": np.asarray(ep["eff"], np.float64).reshape(-1, 3),
            "cube": np.asarray(ep["cube"], np.float64).reshape(-1, 3),
            "grip": np.asarray(ep["grip"], np.float64).reshape(-1),
            "goal": np.asarray(ep["goal"], np.float64).reshape(3),
            "success": bool(ep["success"]),
            "table_z": float(ep["table_z"]),
            "length": len(d),
        }
        cls = classify_phases(sig, thr)
        if cls["success"]:
            outcome = "success"
        elif cls["fail_phase"] == "transport":
            outcome = ("maintain_fail"
                       if cls["final_cube_lift"] < thr.delta_lift
                       else "transport_fail")
        else:
            outcome = "other"
        n_t = int(tm.sum())
        if n_t >= 5 and np.ptp(d[tm]) > 1e-9 and np.ptp(V[tm]) > 1e-9:
            rho = _spearman(V[tm], -d[tm])
            v_secure, v_end = float(V[tm][0]), float(V[-1])
        else:
            rho, v_secure, v_end = float("nan"), float("nan"), float("nan")
        vl_rows.append({"task": ep["task"], "episode": ep["episode"],
                        "outcome": outcome, "rho_V_negd": rho,
                        "V_at_secure": v_secure, "V_at_end": v_end,
                        "n_transport_steps": n_t})
        fun_rows.append({"task": ep["task"], "episode": ep["episode"],
                         "furthest_phase": cls["fail_phase"]
                         if not cls["success"] else "success",
                         "outcome": outcome,
                         "final_cube_lift": cls["final_cube_lift"],
                         "final_grip": cls["final_grip"],
                         "final_cube_goal_dist": cls["final_cube_goal_dist"]})
        cov_eps.append({"obs": ep["obs"], "outcome": outcome,
                        "grip": ep["grip"], "cube": ep["cube"],
                        "table_z": ep["table_z"],
                        "transport_mask": tm})
        cube_xy = np.asarray(ep["cube"], np.float64).reshape(-1, 3)[:, :2]
        eff_xy = np.asarray(ep["eff"], np.float64).reshape(-1, 3)[:, :2]
        ep_for_region = dict(ep)
        ep_for_region["transport_mask"] = tm
        region = _region(ep_for_region, thr)
        for ti, (dd, vv, mm, cxy, exy, rg) in enumerate(
                zip(d, V, tm, cube_xy, eff_xy, region)):
            vs_rows.append({"task": ep["task"], "episode": ep["episode"],
                            "t": int(ti), "outcome": outcome,
                            "region": str(rg), "d": float(dd),
                            "V": float(vv), "transport": bool(mm),
                            "cube_x": float(cxy[0]),
                            "cube_y": float(cxy[1]),
                            "eef_x": float(exy[0]),
                            "eef_y": float(exy[1])})
    cov = probe_coverage(cov_eps, ref_obs, thr, feature=feature)
    cov.insert(0, "task", episodes[0]["task"] if episodes else "")
    vs = pd.DataFrame(vs_rows, columns=["task", "episode", "t", "outcome",
                                        "region", "d", "V", "transport",
                                        "cube_x", "cube_y",
                                        "eef_x", "eef_y"])
    return (pd.DataFrame(vl_rows), cov, pd.DataFrame(fun_rows), vs)


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True,
                    help="a GCIQL seed dir (contains flags.json, "
                         "params_<step>.pkl)")
    ap.add_argument("--step", type=int, default=1000000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", default="1,2,3,4,5")
    ap.add_argument("--n-episodes", type=int, default=10)
    ap.add_argument("--scenario", default="S0", choices=["S0"])
    ap.add_argument("--dataset-path",
                    default="datasets/cube-single-play-v0")
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--obs-type", default=None,
                    help="state|pixels; default reads flags.json")
    return ap


def _load_ref_obs(dataset_path: str, max_n: int = 5000,
                  feature: str = "obs") -> np.ndarray:
    """Reference array for the coverage probe. feature='obs' stacks
    next/observation (state path); feature='cube' stacks cube xyz
    (physics[:,14:17]) for the pixel path."""
    buf = Path(dataset_path) / "buffer"
    files = sorted(buf.glob("episode_*.npz"))[:200]
    obs = []
    for f in files:
        with np.load(f) as z:
            if feature == "cube":
                arr_i = np.asarray(z["physics"], np.float32)[:, 14:17]
            else:
                key = ("observation" if "observation" in z
                       else "observations" if "observations" in z
                       else "obs")
                arr_i = np.asarray(z[key], np.float32)
        obs.append(arr_i)
    arr = np.concatenate(obs, 0) if obs else np.zeros((1, 1), np.float32)
    if len(arr) > max_n:
        rng = np.random.default_rng(0)
        arr = arr[rng.choice(len(arr), max_n, replace=False)]
    return arr


def _jax_rollout(run_dir: str, step: int, task_ids: List[int],
                 n_episodes: int, max_steps: int):
    """Load the GCIQL checkpoint and roll out the goal-conditioned
    policy. Imports jax/flax/vendored OGBench — JAX venv only."""
    ogb_impls = str(REPO_ROOT / "third_party" / "ogbench" / "impls")
    # Always front-place so the vendored OGBench `agents`/`utils` win over the
    # repo-root FB `agents` — even when ogb_impls is pre-injected via PYTHONPATH
    # (needed to register the DrQ encoder in sitecustomize at startup).
    if ogb_impls in sys.path:
        sys.path.remove(ogb_impls)
    sys.path.insert(0, ogb_impls)
    import ogbench  # vendored
    from utils.flax_utils import restore_agent
    import jax
    import jax.numpy as jnp

    flags = parse_flags(run_dir)
    saved_agent = flags.get("agent")
    # Select the agent class by name. gciql and gcivl share the
    # create / sample_actions / value(s,g) API used below (gcivl's value
    # returns a twin tuple, handled in _value).
    agent_name = (saved_agent.get("agent_name")
                  if isinstance(saved_agent, dict) else None) or "gciql"
    if agent_name == "gcivl":
        from agents.gcivl import GCIVLAgent as AgentCls, get_config
    else:
        from agents.gciql import GCIQLAgent as AgentCls, get_config
    # Build the agent config from get_config() defaults, then overlay the
    # saved run's agent config (flags.json stores the full agent dict for
    # pixel runs — crucially `encoder`, e.g. 'impala_small' — without which
    # restore_agent would mismatch the checkpoint's network shapes). State
    # runs store `agent` as a path string; the overlay is then skipped.
    config = get_config()
    if isinstance(saved_agent, dict):
        for k, v in saved_agent.items():
            if k in config:
                config[k] = v
    episodes: List[Dict[str, Any]] = []
    fs = config.get("frame_stack")
    for task_id in task_ids:
        env = ogbench.make_env_and_datasets(
            flags["env_name"], env_only=True)
        # Pixel DrQ checkpoints train with a frame-stack wrapper (e.g. 3);
        # match it so the encoder's input channels line up on restore/rollout.
        if fs:
            from utils.env_utils import FrameStackWrapper
            env = FrameStackWrapper(env, fs)
        ex_obs, info = env.reset(options=dict(task_id=task_id))
        ex_act = env.action_space.sample()
        agent = AgentCls.create(
            flags["seed"],
            np.asarray(ex_obs, np.float32)[None],
            np.asarray(ex_act, np.float32)[None],
            config,
        )
        agent = restore_agent(agent, str(run_dir), step)

        @jax.jit
        def _value(o, g):
            # Single (o,g) -> scalar value. Averages over any ensemble/twin
            # axis (gcivl's value returns two heads) and the batch-of-1.
            v = agent.network.select("value")(o[None], g[None])
            return jnp.asarray(v).mean()

        rng = jax.random.PRNGKey(0)
        for ep_i in range(n_episodes):
            obs, info = env.reset(options=dict(task_id=task_id))
            goal = info["goal"]
            u = env.unwrapped
            tb = int(getattr(u, "_target_block", 0) or 0)
            gxyz = np.asarray(u.cur_task_info["goal_xyzs"][tb], np.float64)
            table_z = float(u.cur_task_info["init_xyzs"][tb][2])
            O, EFF, CUBE, GRIP, VV, D = [], [], [], [], [], []
            succ = False
            for _ in range(max_steps):
                rng, key = jax.random.split(rng)
                a = np.asarray(agent.sample_actions(
                    observations=np.asarray(obs, np.float32),
                    goals=np.asarray(goal, np.float32),
                    seed=key, temperature=0.0))
                a = np.clip(a, -1.0, 1.0)
                v = float(np.asarray(_value(
                    np.asarray(obs, np.float32),
                    np.asarray(goal, np.float32))))
                obs, _, term, trunc, info = env.step(a)
                cube = np.asarray(info["privileged/block_0_pos"],
                                  np.float64)
                grip = float(np.asarray(
                    info["proprio/gripper_opening"]).reshape(-1)[0])
                O.append(np.asarray(obs, np.float32))
                EFF.append(np.asarray(info["proprio/effector_pos"],
                                      np.float64))
                CUBE.append(cube)
                GRIP.append(grip)
                VV.append(v)
                D.append(float(np.linalg.norm(cube - gxyz)))
                succ = succ or bool(info.get("success", False))
                if term or trunc:
                    break
            episodes.append({
                "obs": np.asarray(O, np.float32),
                "d": np.asarray(D), "V": np.asarray(VV),
                "grip": np.asarray(GRIP),
                "cube": np.asarray(CUBE),
                "eff": np.asarray(EFF),
                "goal": gxyz,
                "table_z": table_z,
                "success": succ,
                "task": f"task{task_id}", "episode": ep_i})
        env.close()
    return episodes, flags


def main() -> None:
    args = build_argparser().parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    task_ids = [int(t) for t in args.tasks.split(",") if t.strip()]
    episodes, flags = _jax_rollout(
        args.run_dir, args.step, task_ids, args.n_episodes,
        args.max_steps)
    obs_type = args.obs_type or flags.get("obs_type", "state")
    feature = "cube" if obs_type == "pixels" else "obs"
    ref = _load_ref_obs(args.dataset_path, feature=feature)
    vl, cov, fun, vs = episodes_to_frames(episodes, Thresholds(), ref,
                                          feature=feature)
    vl.to_parquet(out / "value_landscape.parquet")
    cov.to_parquet(out / "coverage.parquet")
    fun.to_parquet(out / "phase_funnel.parquet")
    vs.to_parquet(out / "value_steps.parquet")
    (out / "metrics.json").write_text(json.dumps({
        "run_dir": str(args.run_dir), "step": args.step,
        "tasks": task_ids, "n_episodes": args.n_episodes,
        "env_name": flags["env_name"],
        "value_fn": "GCValue V(s,g)",
    }, indent=2))
    print(f"[gciql_profile] done -> {out}")


if __name__ == "__main__":
    main()

"""evals/phase_probe.py — test where FB fails in a cube episode.

M1: classify normal rollouts into reach/secure/success phases.
M2: counterfactual start states (S0 baseline / S1 skip-reach /
S2 pre-grasped) to isolate the grasp bottleneck.

See docs/superpowers/specs/2026-05-18-cube-phase-failure-probe-design.md.
Pure logic here is unit-tested; MuJoCo physics is validated by a
documented Linux acceptance run. No training/eval code is modified.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# mujoco is imported lazily by _ensure_mujoco(): importing it at module load
# triggers a GL-context check that fails when MUJOCO_GL is set to an
# unsupported value (e.g. `egl` on macOS, which train.py sets). The pure
# phase-classification logic and unit tests must import this module with no
# valid GL backend (spec §8); only apply_scenario needs real MuJoCo. Tests
# monkeypatch `phase_probe.mujoco` with a fake, which _ensure_mujoco honours.
mujoco = None

PHASE_ORDER = ["none", "reached", "secured", "success"]
N_SETTLE = 5  # MuJoCo steps to clamp the cube in S2 (module constant; YAGNI)


@dataclass
class Thresholds:
    eps_reach: float = 0.06   # m, hand-to-cube distance for "reached"
    delta_lift: float = 0.03  # m, cube height above table for "lifted"
    k_steps: int = 5          # consecutive steps the lift must hold
    tau_grip: float = 0.5     # gripper_opening > tau => "closed"


def _max_true_run(mask: np.ndarray) -> int:
    """Length of the longest run of True in a 1-D boolean array."""
    best = run = 0
    for v in mask:
        run = run + 1 if v else 0
        best = max(best, run)
    return best


def classify_phases(signals: Dict[str, Any], thr: Thresholds) -> Dict[str, Any]:
    """Classify ONE episode's per-step signals into phase milestones.

    signals: eff[T,3], cube[T,3], grip[T], goal[3], success(bool),
             table_z(float).
    Returns the milestones, monotone furthest_phase/fail_phase, and
    diagnostics. Handles T == 0.
    """
    eff = np.asarray(signals["eff"], dtype=np.float64).reshape(-1, 3)
    cube = np.asarray(signals["cube"], dtype=np.float64).reshape(-1, 3)
    grip = np.asarray(signals["grip"], dtype=np.float64).reshape(-1)
    goal = np.asarray(signals["goal"], dtype=np.float64).reshape(3)
    table_z = float(signals["table_z"])
    success = bool(signals["success"])
    T = eff.shape[0]

    if T == 0:
        return {
            "reached": success, "secured": success, "success": success,
            "reached_raw": False, "secured_raw": False,
            "furthest_phase": "success" if success else "none",
            "fail_phase": "none" if success else "reach",
            "min_eff_cube_dist": float("nan"),
            "max_cube_lift": float("nan"),
            "final_cube_lift": float("nan"),
            "final_grip": float("nan"),
            "final_cube_goal_dist": float("nan"),
            "length": 0,
        }

    eff_cube_dist = np.linalg.norm(eff - cube, axis=1)
    eff_cube_xy = np.linalg.norm(eff[:, :2] - cube[:, :2], axis=1)
    lift = cube[:, 2] - table_z

    reached_raw = bool(eff_cube_dist.min() < thr.eps_reach)
    secured_mask = (
        (lift > thr.delta_lift)
        & (grip > thr.tau_grip)
        & (eff_cube_xy < thr.eps_reach)
    )
    secured_raw = bool(_max_true_run(secured_mask) >= thr.k_steps)

    # Monotone ladder: success => secured => reached.
    secured = secured_raw or success
    reached = reached_raw or secured

    if success:
        furthest, fail = "success", "none"
    elif secured:
        furthest, fail = "secured", "transport"
    elif reached:
        furthest, fail = "reached", "grasp"
    else:
        furthest, fail = "none", "reach"

    return {
        "reached": reached, "secured": secured, "success": success,
        "reached_raw": reached_raw, "secured_raw": secured_raw,
        "furthest_phase": furthest, "fail_phase": fail,
        "min_eff_cube_dist": float(eff_cube_dist.min()),
        "max_cube_lift": float(lift.max()),
        "final_cube_lift": float(lift[-1]),
        "final_grip": float(grip[-1]),
        "final_cube_goal_dist": float(np.linalg.norm(cube[-1] - goal)),
        "length": int(T),
    }


def step_signals(info: Dict[str, Any]) -> Dict[str, Any]:
    """Per-step signals from an OGBench manipspace `info` dict."""
    return {
        "eff": np.asarray(info["proprio/effector_pos"], dtype=np.float64),
        "cube": np.asarray(info["privileged/block_0_pos"], dtype=np.float64),
        "grip": float(np.asarray(info["proprio/gripper_opening"]).reshape(-1)[0]),
        "success": bool(info.get("success", False)),
    }


def _target_block(env) -> int:
    return int(getattr(env.unwrapped, "_target_block", 0) or 0)


def episode_goal(env) -> np.ndarray:
    tb = _target_block(env)
    return np.asarray(
        env.unwrapped.cur_task_info["goal_xyzs"][tb], dtype=np.float64
    )


def episode_table_z(env) -> float:
    tb = _target_block(env)
    return float(env.unwrapped.cur_task_info["init_xyzs"][tb][2])


def ensure_manip_env(env) -> None:
    """Raise a clear error if this is not an OGBench cube manip env."""
    u = getattr(env, "unwrapped", None)
    if u is None or not all(
        hasattr(u, a) for a in ("_data", "_pinch_site_id", "cur_task_info")
    ):
        raise RuntimeError(
            "phase_probe is cube-only: env.unwrapped lacks manipspace "
            "handles (_data/_pinch_site_id/cur_task_info)."
        )


SCENARIOS = ["S0", "S1", "S2"]


def _ensure_mujoco():
    """Return the mujoco module, importing it on first real use.

    If a test (or caller) has set the module-level ``mujoco`` attribute
    (e.g. a fake), that value is used and no real import happens.
    """
    global mujoco
    if mujoco is None:
        import mujoco as _m

        mujoco = _m
    return mujoco


def apply_scenario(env, scenario: str) -> None:
    """Mutate post-reset MuJoCo state for a counterfactual start.

    S0: no-op. S1: cube under the hand at table height, gripper open.
    S2: cube at the effector, gripper closed, settle N_SETTLE steps so
    the fingers physically clamp it (no weld). The task goal/target
    mocap is never touched.
    """
    if scenario == "S0":
        return

    mj = _ensure_mujoco()
    u = env.unwrapped
    model, data = u._model, u._data
    eff = np.asarray(data.site_xpos[u._pinch_site_id], dtype=np.float64).copy()
    joint = data.joint("object_joint_0")

    if scenario == "S1":
        table_z = episode_table_z(env)
        joint.qpos[0] = eff[0]
        joint.qpos[1] = eff[1]
        joint.qpos[2] = table_z
        mj.mj_forward(model, data)
        return

    if scenario == "S2":
        joint.qpos[:3] = eff
        mj.mj_forward(model, data)
        data.ctrl[u._gripper_actuator_ids] = 255.0
        for _ in range(N_SETTLE):
            mj.mj_step(model, data)
        mj.mj_forward(model, data)
        return

    raise ValueError(f"unknown scenario {scenario!r} (expected one of {SCENARIOS})")


def _initial_obs(env, reset_obs, scenario: str):
    """Initial observation for the post-reset (post-scenario) state.

    State envs: ``compute_observation()`` on the unwrapped env reflects any
    ``apply_scenario`` physics mutation without a re-randomizing reset.

    Pixel envs (PixelWrapper / FrameStackObservation, obs ndim >= 3): the
    wrapper only emits correctly stacked CHW frames via reset()/step(), so we
    use the reset observation. That is valid only for S0 (no mutation); the
    wrapper cannot re-render a frame stack after an S1/S2 physics mutation, so
    those are rejected explicitly rather than fed a stale/raw frame.
    """
    obs0 = np.asarray(reset_obs, dtype=np.float32)
    if obs0.ndim >= 3:
        if scenario != "S0":
            raise NotImplementedError(
                "pixel counterfactual scenarios (S1/S2) are not supported "
                "by phase_probe; run with --scenarios S0")
        return obs0
    u = env.unwrapped
    if hasattr(u, "compute_observation"):
        return np.asarray(u.compute_observation(), dtype=np.float32)
    return obs0


def rollout_with_phase_signals(
    env, agent, z, n_episodes: int, thr: Thresholds, scenario: str = "S0",
    record_obs: bool = False,
) -> List[Dict[str, Any]]:
    """Roll out n_episodes under `scenario`, recording per-step signals.

    Each episode: one env.reset() -> apply_scenario -> run policy.
    `success` is OR-accumulated from per-step info (same convention as
    evals/ogbench.py). Returns one signals dict per episode suitable for
    classify_phases.
    """
    import torch

    ensure_manip_env(env)
    episodes: List[Dict[str, Any]] = []
    for _ in range(n_episodes):
        # Clear per-episode planner state. No-op for FB/RLDP/CRL (BaseAgent
        # default); stateful planners (TD-MPC2) reset t0 + MPPI warm-start so
        # every episode plans fresh rather than carrying the previous mean.
        if hasattr(agent, "reset"):
            agent.reset()
        reset_obs, info = env.reset()
        apply_scenario(env, scenario)
        goal = episode_goal(env)
        table_z = episode_table_z(env)

        effs, cubes, grips = [], [], []
        s0 = step_signals(info)
        effs.append(s0["eff"]); cubes.append(s0["cube"]); grips.append(s0["grip"])
        ep_success = bool(s0["success"])

        obs_list, act_list = [], []
        observation = _initial_obs(env, reset_obs, scenario)
        if record_obs:
            obs_list.append(np.asarray(observation, dtype=np.float32))
        while True:
            a = agent.act(
                obs=torch.tensor(np.asarray(observation), device=agent.device,
                                 dtype=torch.float32)[None],
                z=z,
            ).cpu().numpy()[0]
            if record_obs:
                act_list.append(np.asarray(a, dtype=np.float32))
            observation, _, terminated, truncated, info = env.step(a)
            s = step_signals(info)
            effs.append(s["eff"]); cubes.append(s["cube"]); grips.append(s["grip"])
            if record_obs:
                obs_list.append(np.asarray(observation, dtype=np.float32))
            ep_success = ep_success or s["success"]
            if terminated or truncated:
                break

        episodes.append({
            "eff": np.asarray(effs, dtype=np.float64),
            "cube": np.asarray(cubes, dtype=np.float64),
            "grip": np.asarray(grips, dtype=np.float64),
            "goal": goal,
            "success": bool(ep_success),
            "table_z": table_z,
            "length": len(effs),
            "obs": (np.asarray(obs_list, dtype=np.float32)
                    if record_obs else None),
            "action": (np.asarray(act_list, dtype=np.float32)
                       if record_obs else None),
        })
    return episodes


def run_phase_probe(
    *, agent, infer_z, make_env, tasks: List[str], scenarios: List[str],
    n_episodes: int, thr: Thresholds,
):
    """Run M1+M2 over tasks x scenarios. Returns (per_ep_df, summary_df,
    hist) where hist[task] -> {furthest_phase: count} for scenario S0."""
    import pandas as pd

    rows: List[Dict[str, Any]] = []
    for task in tasks:
        z = infer_z(task)
        for scenario in scenarios:
            env = make_env(task)
            try:
                eps = rollout_with_phase_signals(
                    env, agent, z, n_episodes, thr, scenario=scenario)
            finally:
                if hasattr(env, "close"):
                    env.close()
            for i, sig in enumerate(eps):
                c = classify_phases(sig, thr)
                rows.append({
                    "task": task, "scenario": scenario, "episode": i,
                    "reached": c["reached"], "secured": c["secured"],
                    "success": c["success"], "reached_raw": c["reached_raw"],
                    "secured_raw": c["secured_raw"],
                    "furthest_phase": c["furthest_phase"],
                    "fail_phase": c["fail_phase"],
                    "min_eff_cube_dist": c["min_eff_cube_dist"],
                    "max_cube_lift": c["max_cube_lift"],
                    "final_cube_lift": c["final_cube_lift"],
                    "final_grip": c["final_grip"],
                    "final_cube_goal_dist": c["final_cube_goal_dist"],
                    "length": c["length"],
                })

    per_ep = pd.DataFrame(rows)
    summary = (
        per_ep.groupby(["task", "scenario"])
        .agg(reached_rate=("reached", "mean"),
             secured_rate=("secured", "mean"),
             success_rate=("success", "mean"),
             n=("episode", "count"))
        .reset_index()
    )
    hist: Dict[str, Dict[str, int]] = {}
    s0 = per_ep[per_ep.scenario == "S0"]
    for task, g in s0.groupby("task"):
        hist[task] = g["furthest_phase"].value_counts().to_dict()
    return per_ep, summary, hist


def plot_phase_histogram(per_ep, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s0 = per_ep[per_ep.scenario == "S0"] if "scenario" in per_ep else per_ep
    tasks = sorted(s0["task"].unique())
    fig, ax = plt.subplots(figsize=(max(5, 1.6 * len(tasks)), 4))
    bottom = np.zeros(len(tasks))
    for phase in PHASE_ORDER:
        counts = [
            int((s0[s0.task == t]["furthest_phase"] == phase).sum())
            for t in tasks
        ]
        ax.bar(tasks, counts, bottom=bottom, label=phase)
        bottom += np.array(counts)
    ax.set_ylabel("episodes")
    ax.set_title("M1: furthest phase per task (S0)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_scenario_success(summary, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tasks = sorted(summary["task"].unique())
    scenarios = sorted(summary["scenario"].unique())
    x = np.arange(len(tasks))
    w = 0.8 / max(1, len(scenarios))
    fig, ax = plt.subplots(figsize=(max(5, 1.6 * len(tasks)), 4))
    for j, sc in enumerate(scenarios):
        vals = [
            float(summary[(summary.task == t) & (summary.scenario == sc)]
                   ["success_rate"].mean())
            for t in tasks
        ]
        ax.bar(x + j * w, vals, w, label=sc)
    ax.set_xticks(x + w * (len(scenarios) - 1) / 2)
    ax.set_xticklabels(tasks, rotation=20, ha="right", fontsize=7)
    ax.set_ylabel("success rate")
    ax.set_title("M2: success rate by scenario")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_summary_md(summary, path) -> None:
    lines = ["# Cube phase-failure probe — summary", ""]
    for _, r in summary.sort_values(["task", "scenario"]).iterrows():
        lines.append(
            f"- **{r['task']} / {r['scenario']}**: reached "
            f"{r['reached_rate']*100:.1f}%  secured "
            f"{r['secured_rate']*100:.1f}%  success "
            f"{r['success_rate']*100:.1f}%  (n={int(r['n'])})")
    lines += ["", "## Hypothesis readout", ""]
    lines.append(
        "Prediction: success(S0) ≈ success(S1) (grasp still required) and "
        "success(S2) ≫ success(S0) (transport easy once held); baseline "
        "reached-rate ≫ secured-rate ⇒ grasp/pick-up is the bottleneck.")
    for task in sorted(summary["task"].unique()):
        sub = summary[summary.task == task].set_index("scenario")

        def g(sc, col):
            return float(sub.loc[sc, col]) if sc in sub.index else float("nan")

        lines.append(
            f"- {task}: S0 reached={g('S0','reached_rate')*100:.0f}% "
            f"secured={g('S0','secured_rate')*100:.0f}% "
            f"success S0={g('S0','success_rate')*100:.0f}% "
            f"S1={g('S1','success_rate')*100:.0f}% "
            f"S2={g('S2','success_rate')*100:.0f}%")
    Path(path).write_text("\n".join(lines))


# ──────────────────────────────────────────────────────────────────────────────
# TD-MPC2 latent phase-transition-prediction probe (rlbrew MBRL diagnostic)
# ──────────────────────────────────────────────────────────────────────────────
def tdmpc2_latent_phase_probe(agent, batch, goal, *, rollout_len=3, ridge=1e-3):
    """Fit a linear probe z -> cube_xyz on encoded states, then measure how
    cube-xyz prediction error grows along an imagined rollout (model.next).

    Decoder-free TD-MPC2 cannot reconstruct observations, so we probe the latent.
    Returns dict with single-step probe MSE and open-loop MSE per imagined step --
    the "does the model predict lift/slip/drop" signal for the paper's MBRL row.
    """
    import torch

    dev = agent.device
    obs = batch["observation"].to(dev).float()             # [B,H,obs]
    next_phys = batch["next"]["physics"].to(dev).float()
    cube = next_phys[..., 14:17]                           # [B,H,3] targets
    g = goal.to(dev).float()

    with torch.no_grad():
        folded = agent._fold(obs[:, 0, :], g.expand(obs.shape[0], 3))  # [B,obs+3]
        z0 = agent.core.model.encode(folded, None)         # [B,latent]
    # closed-form ridge probe z0 -> cube[:,0]
    Z = z0
    Y = cube[:, 0, :]
    W = torch.linalg.solve(Z.T @ Z + ridge * torch.eye(Z.shape[1], device=dev), Z.T @ Y)
    pred0 = Z @ W
    cube_xyz_mse = torch.mean((pred0 - Y) ** 2).item()

    # open-loop imagined rollout error
    open_loop = []
    with torch.no_grad():
        z = z0
        for t in range(min(rollout_len, cube.shape[1])):
            pred = z @ W
            open_loop.append(torch.mean((pred - cube[:, t, :]) ** 2).item())
            a_t = batch["action"][:, t, :].to(dev).float() if "action" in batch \
                else torch.zeros(z.shape[0], agent.action_dim, device=dev)
            z = agent.core.model.next(z, a_t, None)
    return {"cube_xyz_mse": cube_xyz_mse, "open_loop_cube_mse_by_step": open_loop}

"""evals/analysis.py — importable helpers for checkpoint re-run analysis.

Reuses train.make_agent + envs/evals; never modifies training paths.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from omegaconf import OmegaConf


def load_cfg(config_path: str | Path, device: str = "cpu"):
    """Load a run's .hydra/config.yaml and force the device."""
    cfg = OmegaConf.load(str(config_path))
    cfg.device = device
    return cfg


def build_env_and_agent(cfg):
    """Build (env, agent, obs_space, action_dim) from a loaded cfg."""
    from envs.ogbench import create_ogbench_env
    from train import make_agent

    env, _ = create_ogbench_env(
        cfg.domain, obs_type=cfg.obs_type, seed=cfg.seed,
        frame_stack=int(getattr(cfg, "frame_stack", 1)))
    obs_space = env.observation_space
    action_dim = env.action_space.shape[0]
    agent = make_agent(cfg, obs_space, action_dim)
    return env, agent, obs_space, action_dim


def load_checkpoint(agent, ckpt_path: str | Path, map_location: str = "cpu") -> None:
    """Load a torch state_dict checkpoint into agent with a clear error
    message when the checkpoint does not match the agent variant.

    A strict load is tried first. If it fails *only* because the checkpoint
    carries extra modules the local agent lacks (e.g. rldp_flowbc's
    `_predictor` head), those keys are ignored and the load proceeds — but
    every parameter the agent needs must still be present and shape-matched,
    so a genuine architecture mismatch still raises.
    """
    state = torch.load(str(ckpt_path), map_location=map_location)
    try:
        agent.load_state_dict(state)
        return
    except (KeyError, RuntimeError):
        pass
    # Fallback for re-run analysis: load only the model weights (optimizers
    # are irrelevant for inference). Tolerates a checkpoint saved by a
    # different agent variant — e.g. rldp_flowbc, which carries an extra
    # `_predictor` head and omits the actor-vf optimizer. Extra checkpoint
    # modules are ignored, but every parameter the local model needs must be
    # present and shape-matched, so a genuine architecture mismatch raises.
    missing, unexpected = agent.model.load_state_dict(state["model"],
                                                      strict=False)
    if missing:
        raise RuntimeError(
            f"checkpoint does not match agent (cfg.agent build): "
            f"{len(missing)} model parameter(s) the agent needs are absent "
            f"from {ckpt_path}, e.g. {missing[:5]}")
    print(f"[load_checkpoint] loaded model weights only; ignoring "
          f"{len(unexpected)} extra checkpoint module(s), e.g. "
          f"{unexpected[:3]}")


def rollout_with_trajectory(
    env: Any,
    agent: Any,
    num_episodes: int,
    z: torch.Tensor,
    record: bool = False,
) -> Dict[str, Any]:
    """Roll out `num_episodes` episodes, collecting per-step arrays.

    Returns padded arrays [n_episodes, T_max, ...] plus lengths[n_episodes].
    Mirrors the env protocol used by envs/rollout.py.
    """
    episodes = []
    frames_all = [] if record else None
    for _ in range(num_episodes):
        obs, info = env.reset()
        ep = {"observations": [obs], "actions": [], "rewards": [],
              "terminated": [], "truncated": [], "success": False}
        ep_frames = [env.render()] if record else None
        while True:
            a = agent.act(
                obs=torch.tensor(np.asarray(obs), device=agent.device,
                                 dtype=torch.float32)[None],
                z=z,
            ).cpu().numpy()[0]
            obs, reward, terminated, truncated, info = env.step(a)
            ep["actions"].append(a)
            ep["rewards"].append(float(reward))
            ep["terminated"].append(bool(terminated))
            ep["truncated"].append(bool(truncated))
            ep["observations"].append(obs)
            if info.get("success", False):
                ep["success"] = True
            if record:
                ep_frames.append(env.render())
            if terminated or truncated:
                break
        episodes.append(ep)
        if record:
            frames_all.append(ep_frames)

    lengths = np.array([len(e["observations"]) for e in episodes], dtype=np.int64)
    t_max = int(lengths.max())
    obs_dim = np.asarray(episodes[0]["observations"][0]).shape[-1]
    act_dim = np.asarray(episodes[0]["actions"][0]).shape[-1]

    n = len(episodes)
    observations = np.zeros((n, t_max, obs_dim), dtype=np.float32)
    actions = np.zeros((n, t_max, act_dim), dtype=np.float32)
    rewards = np.zeros((n, t_max), dtype=np.float32)
    terminated = np.zeros((n, t_max), dtype=bool)
    truncated = np.zeros((n, t_max), dtype=bool)
    for i, e in enumerate(episodes):
        L = len(e["observations"])
        observations[i, :L] = np.asarray(e["observations"], dtype=np.float32)
        na = len(e["actions"])
        actions[i, :na] = np.asarray(e["actions"], dtype=np.float32)
        rewards[i, :na] = np.asarray(e["rewards"], dtype=np.float32)
        terminated[i, :na] = np.asarray(e["terminated"], dtype=bool)
        truncated[i, :na] = np.asarray(e["truncated"], dtype=bool)

    return {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "terminated": terminated,
        "truncated": truncated,
        "lengths": lengths,
        "success": np.array([e["success"] for e in episodes], dtype=bool),
        "z": z.detach().cpu().numpy(),
        "frames": frames_all,
    }


def save_trajectories(result: Dict[str, Any], task: str,
                      out_dir: str | Path) -> None:
    """Write trajectories/<task>.npz and append trajectory_summary.parquet."""
    import pandas as pd

    traj_dir = Path(out_dir) / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        traj_dir / f"{task}.npz",
        observations=result["observations"],
        actions=result["actions"],
        rewards=result["rewards"],
        terminated=result["terminated"],
        truncated=result["truncated"],
        lengths=result["lengths"],
        success=result["success"],
        z=result["z"],
    )
    rows = []
    for i in range(len(result["lengths"])):
        L = int(result["lengths"][i])
        rows.append({
            "task": task,
            "episode": i,
            "return": float(result["rewards"][i, :max(L - 1, 0)].sum()),
            "length": L,
            "success": bool(result["success"][i]),
        })
    new = pd.DataFrame(rows)
    summ_path = traj_dir / "trajectory_summary.parquet"
    if summ_path.exists():
        new = pd.concat([pd.read_parquet(summ_path), new], ignore_index=True)
    new.to_parquet(summ_path)


def _episode_stats(env, agent, z, n_episodes):
    res = rollout_with_trajectory(env, agent, n_episodes, z, record=False)
    success = float(np.mean(res["success"]))
    reward = float(np.mean([
        res["rewards"][i, : max(int(res["lengths"][i]) - 1, 0)].sum()
        for i in range(len(res["lengths"]))
    ]))
    return success, reward


def z_probe_cross_task(tasks, agent, infer_z, make_env, n_episodes: int = 10):
    """Roll out in env_task conditioned on z_task's z for all pairs.

    `infer_z(task) -> Tensor`, `make_env(task) -> env`. Returns a tidy
    DataFrame (env_task, z_task, success, reward) — a |tasks|^2 matrix.
    """
    import pandas as pd

    zs = {t: infer_z(t) for t in tasks}
    rows = []
    for env_task in tasks:
        env = make_env(env_task)
        try:
            for z_task in tasks:
                s, r = _episode_stats(env, agent, zs[z_task], n_episodes)
                rows.append({"env_task": env_task, "z_task": z_task,
                             "success": s, "reward": r})
        finally:
            if hasattr(env, "close"):
                env.close()
    return pd.DataFrame(rows)


def z_interp(z_a, z_b, agent, env, n_alpha: int = 11, n_episodes: int = 10):
    """Sweep z = (1-α)·z_a + α·z_b for α in linspace(0,1,n_alpha) in `env`."""
    import pandas as pd

    rows = []
    for alpha in np.linspace(0.0, 1.0, n_alpha):
        z = (1.0 - float(alpha)) * z_a + float(alpha) * z_b
        s, r = _episode_stats(env, agent, z, n_episodes)
        rows.append({"alpha": round(float(alpha), 10), "success": s,
                     "reward": r})
    return pd.DataFrame(rows)

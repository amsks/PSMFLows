"""evals/ogbench.py — OGBench evaluator.

For each of the 5 tasks in a domain:
  1. Sample a batch from the offline buffer.
  2. Relabel with `get_relabel_fn(domain, task)(next.physics, action)`.
  3. Shift rewards by `shift_reward` (default 1.0).
  4. Infer z = project(sum r_i * B(next_obs_i)).
  5. Roll out `n_episodes` episodes via envs/rollout.py.
  6. Report mean/std success and reward per task, plus relabel-inference
     debug metrics (matches td_jepa's OGBenchRewardEvaluation.run output).
  7. Aggregate across tasks: eval/reward, eval/reward#std, eval/success.
  8. (Optional) Save first episode per task as <video_dir>/step_<N>/<task>.gif
     and upload to wandb as wandb.Video when save_videos=True.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from envs.ogbench import ALL_TASKS, create_ogbench_env, get_relabel_fn
from envs.rollout import rollout


class OGBenchEvaluator:
    def __init__(
        self,
        domain: str,
        agent,
        offline_buffer,
        relabel_size: int = 10_000,
        n_episodes: int = 10,
        shift_reward: float = 1.0,
        obs_type: str = "state",
        frame_stack: int = 1,
        seed: int = 0,
        device: str = "cpu",
        save_videos: bool = False,
        video_dir: Optional[Path | str] = None,
        use_wandb: bool = False,
    ):
        # Match td_jepa's OGBenchRewardEvaluation: do NOT build envs upfront.
        # Building all 5 eval envs + 5 relabel-fn envs at startup would add
        # ~50s of mujoco init even on runs that never call run(). Build them
        # lazily inside run() / _infer_z() and tear them down after each task.
        self.domain = domain
        self.agent = agent
        self.offline_buffer = offline_buffer
        self.relabel_size = relabel_size
        self.n_episodes = n_episodes
        self.shift_reward = shift_reward
        self.obs_type = obs_type
        self.frame_stack = frame_stack
        self.seed = seed
        self.device = device
        self.tasks = ALL_TASKS.get(domain, [])
        self.save_videos = save_videos
        self.video_dir = Path(video_dir) if video_dir is not None else None
        self.use_wandb = use_wandb
        if self.save_videos and self.video_dir is None:
            raise ValueError("save_videos=True requires video_dir to be set")

    def run(self, step: int = 0) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        all_rewards: list[np.ndarray] = []
        all_successes: list[np.ndarray] = []
        for task in self.tasks:
            env, _ = create_ogbench_env(
                task,
                seed=self.seed,
                obs_type=self.obs_type,
                frame_stack=self.frame_stack,
            )
            try:
                if getattr(self.agent, "_use_eval_context", False):
                    goal_obs = self._goal_observation(task) if getattr(self.agent, "goal_cond", False) else None
                    z, relabel_metrics = self.agent.eval_context(env=env, domain=self.domain, task=task, goal_obs=goal_obs)
                else:
                    z, relabel_metrics = self._infer_z(task)
                stats, infos, frames = rollout(
                    env, self.agent, self.n_episodes, ctx=z, record=self.save_videos
                )
            finally:
                env.close()

            per_ep_success = np.array(
                [any(step_info.get("success", False) for step_info in ep_info) for ep_info in infos],
                dtype=np.float32,
            )
            per_ep_reward = np.array(stats["reward"], dtype=np.float32)

            metrics[f"{task}/success"] = float(np.mean(per_ep_success))
            metrics[f"{task}/reward"] = float(np.mean(per_ep_reward))
            metrics[f"{task}/reward#std"] = float(np.std(per_ep_reward))
            for k, v in relabel_metrics.items():
                metrics[f"{task}/{k}"] = float(v)

            all_rewards.append(per_ep_reward)
            all_successes.append(per_ep_success)

            if self.save_videos and frames is not None and len(frames) > 0:
                self._save_gif(task=task, step=step, episode_frames=frames[0])

        if all_rewards:
            rewards_all = np.concatenate(all_rewards)
            successes_all = np.concatenate(all_successes)
            metrics["eval/reward"] = float(np.mean(rewards_all))
            metrics["eval/reward#std"] = float(np.std(rewards_all))
            metrics["eval/success"] = float(np.mean(successes_all))

        return metrics

    def _save_gif(self, *, task: str, step: int, episode_frames: list) -> None:
        import imageio

        ep_arr = np.stack(episode_frames)  # [T, H, W, 3] uint8
        step_dir = self.video_dir / f"step_{step}"
        step_dir.mkdir(parents=True, exist_ok=True)
        gif_path = step_dir / f"{task}.gif"
        imageio.mimsave(gif_path, ep_arr, format="GIF", fps=30, loop=0)
        print(f"[eval] wrote {gif_path}")

        if self.use_wandb:
            import wandb
            wandb.log(
                {f"eval_video/{task}": wandb.Video(str(gif_path), fps=30, format="gif")},
                step=step,
            )

    def _relabel_subsample(self, task: str):
        """Shared sample+relabel used by _infer_z (baseline) and _goal_observation (V1).

        Returns (batch, rewards, relabel_metrics). When the buffer holds fewer than 8
        samples, returns (None, None, {}) so callers can apply their own small-n fallback.
        rewards is a [N,1] float32 Tensor on self.device (already shifted by shift_reward).
        """
        relabel_fn = get_relabel_fn(self.domain, task)  # builds + closes a temp env internally
        n = min(self.relabel_size, len(self.offline_buffer))
        if n < 8:
            return None, None, {}

        batch = self.offline_buffer.sample(n)

        if "physics" not in batch["next"]:
            raise KeyError(
                "Buffer batch is missing batch['next']['physics']. "
                "Re-load the dataset with data.ogbench.load_ogbench_dataset "
                "(updated to place physics under next/)."
            )

        phys_np = batch["next"]["physics"].cpu().numpy()
        act_np = batch["action"].cpu().numpy()

        rewards_np = relabel_fn(phys_np, act_np)               # [N, 1]
        rewards_np = rewards_np + self.shift_reward
        rewards = torch.as_tensor(rewards_np, dtype=torch.float32, device=self.device)

        non_zero = int(np.count_nonzero(rewards_np.ravel()))
        relabel_metrics = {
            "relabel_reward#mean": float(np.mean(rewards_np)),
            "relabel_reward#nonzero": float(non_zero),
            "relabel_reward#zero": float(rewards_np.size - non_zero),
            "relabel_reward#num_samples": float(rewards_np.size),
        }
        return batch, rewards, relabel_metrics

    def _infer_z(self, task: str) -> Tuple[Tensor, Dict[str, float]]:
        batch, rewards, relabel_metrics = self._relabel_subsample(task)
        if batch is None:
            return self.agent.model.sample_z(1).squeeze(0), {}

        z = self.agent.model.reward_inference(
            next_obs=batch["next"]["observation"].to(self.device),
            reward=rewards,
        )
        return z, relabel_metrics

    def _goal_observation(self, task: str) -> Tensor:
        """Source g* for V1 (goal_cond): the highest-reward next-obs from the relabel
        subsample, returned as a [1, obs_dim] raw (unnormalized) Tensor on self.device.

        NOTE: not unit-tested here — it needs a real buffer + mujoco relabel env. It is
        exercised by the Task 4.2 2k-step GPU sanity run.
        """
        batch, rewards, _ = self._relabel_subsample(task)
        next_obs = (batch["next"]["observation"] if batch is not None
                    else self.offline_buffer.sample(1)["next"]["observation"])
        next_obs = next_obs.to(self.device).float()
        if rewards is None:
            idx = 0                                   # small-n fallback: any valid obs
        else:
            idx = int(torch.argmax(rewards.reshape(-1)).item())
        return next_obs[idx].unsqueeze(0)             # [1, obs_dim]

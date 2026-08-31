import collections
import re
import time

import gymnasium
import numpy as np
import ogbench
from gymnasium.spaces import Box

from utils.datasets import Dataset


class EpisodeMonitor(gymnasium.Wrapper):
    """Environment wrapper to monitor episode statistics."""

    def __init__(self, env, filter_regexes=None):
        super().__init__(env)
        self._reset_stats()
        self.total_timesteps = 0
        self.filter_regexes = filter_regexes if filter_regexes is not None else []

    def _reset_stats(self):
        self.reward_sum = 0.0
        self.episode_length = 0
        self.start_time = time.time()

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        # Remove keys that are not needed for logging.
        for filter_regex in self.filter_regexes:
            for key in list(info.keys()):
                if re.match(filter_regex, key) is not None:
                    del info[key]

        self.reward_sum += reward
        self.episode_length += 1
        self.total_timesteps += 1
        info['total'] = {'timesteps': self.total_timesteps}

        if terminated or truncated:
            info['episode'] = {}
            info['episode']['final_reward'] = reward
            info['episode']['return'] = self.reward_sum
            info['episode']['length'] = self.episode_length
            info['episode']['duration'] = time.time() - self.start_time

            if hasattr(self.unwrapped, 'get_normalized_score'):
                info['episode']['normalized_return'] = (
                    self.unwrapped.get_normalized_score(info['episode']['return']) * 100.0
                )

        return observation, reward, terminated, truncated, info

    def reset(self, *args, **kwargs):
        self._reset_stats()
        return self.env.reset(*args, **kwargs)


class FrameStackWrapper(gymnasium.Wrapper):
    """Environment wrapper to stack observations."""

    def __init__(self, env, num_stack):
        super().__init__(env)

        self.num_stack = num_stack
        self.frames = collections.deque(maxlen=num_stack)

        low = np.concatenate([self.observation_space.low] * num_stack, axis=-1)
        high = np.concatenate([self.observation_space.high] * num_stack, axis=-1)
        self.observation_space = Box(low=low, high=high, dtype=self.observation_space.dtype)

    def get_observation(self):
        assert len(self.frames) == self.num_stack
        return np.concatenate(list(self.frames), axis=-1)

    def reset(self, **kwargs):
        ob, info = self.env.reset(**kwargs)
        for _ in range(self.num_stack):
            self.frames.append(ob)
        if 'goal' in info:
            info['goal'] = np.concatenate([info['goal']] * self.num_stack, axis=-1)
        return self.get_observation(), info

    def step(self, action):
        ob, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(ob)
        return self.get_observation(), reward, terminated, truncated, info


def subsample_episodes(dataset_dict, fraction, seed=0):
    """Keep a random `fraction` of episodes (whole episodes, boundaries from `terminals`).

    Whole-episode removal keeps the idx+1 successor pairing intact inside every surviving
    episode; the seam between two formerly non-adjacent episodes is an episode boundary
    either way, which `terminals` already marks. Never drops individual rows (that would
    silently re-pair transitions across the gap — see the 07-29 preimage-repair note).
    """
    terminals = np.asarray(dataset_dict['terminals'])
    ends = np.nonzero(terminals > 0.5)[0]
    starts = np.concatenate([[0], ends[:-1] + 1])
    n_ep = len(ends)
    n_keep = max(1, int(round(n_ep * fraction)))
    keep = np.sort(np.random.default_rng(seed).choice(n_ep, size=n_keep, replace=False))
    row_mask = np.zeros(terminals.shape[0], dtype=bool)
    for i in keep:
        row_mask[starts[i]:ends[i] + 1] = True
    out = {k: np.asarray(v)[row_mask] for k, v in dataset_dict.items()}
    print(f'subsample_episodes: kept {n_keep}/{n_ep} episodes '
          f'({row_mask.sum()}/{terminals.shape[0]} rows) at fraction={fraction} seed={seed}')
    return out


def make_env_and_datasets(env_name, frame_stack=None, action_clip_eps=1e-5,
                          dataset_fraction=1.0, dataset_fraction_seed=0,
                          add_info=False):
    """Make offline RL environment and datasets.

    Args:
        env_name: Name of the environment or dataset.
        frame_stack: Number of frames to stack.
        action_clip_eps: Epsilon for action clipping.
        dataset_fraction: Keep this fraction of training episodes (episode-level, seeded).
        dataset_fraction_seed: RNG seed for the episode subsample.
        add_info: Keep OGBench's simulator-state keys ('qpos', 'qvel', 'button_states') in
            the datasets. They are row-aligned with `observations` and are what
            `set_state` needs to put the simulator back at a recorded transition.
            OGBench drops them by default.

    Returns:
        A tuple of the environment, evaluation environment, training dataset, and validation dataset.
    """

    assert not (add_info and 'singletask' not in env_name), (
        f'add_info is an OGBench-only option; {env_name!r} carries no qpos/qvel')

    if 'singletask' in env_name:
        # OGBench.
        env, train_dataset, val_dataset = ogbench.make_env_and_datasets(env_name, add_info=add_info)
        eval_env = ogbench.make_env_and_datasets(env_name, env_only=True)
        env = EpisodeMonitor(env, filter_regexes=['.*privileged.*', '.*proprio.*'])
        eval_env = EpisodeMonitor(eval_env, filter_regexes=['.*privileged.*', '.*proprio.*'])
        if dataset_fraction < 1.0:
            train_dataset = subsample_episodes(dict(train_dataset), dataset_fraction,
                                               dataset_fraction_seed)
        train_dataset = Dataset.create(**train_dataset)
        val_dataset = Dataset.create(**val_dataset)
    elif 'antmaze' in env_name and ('diverse' in env_name or 'play' in env_name or 'umaze' in env_name):
        # D4RL AntMaze.
        from envs import d4rl_utils

        env = d4rl_utils.make_env(env_name)
        eval_env = d4rl_utils.make_env(env_name)
        dataset = d4rl_utils.get_dataset(env, env_name)
        train_dataset, val_dataset = dataset, None
    elif 'pen' in env_name or 'hammer' in env_name or 'relocate' in env_name or 'door' in env_name:
        # D4RL Adroit.
        import d4rl.hand_manipulation_suite  # noqa
        from envs import d4rl_utils

        env = d4rl_utils.make_env(env_name)
        eval_env = d4rl_utils.make_env(env_name)
        dataset = d4rl_utils.get_dataset(env, env_name)
        train_dataset, val_dataset = dataset, None
    else:
        raise ValueError(f'Unsupported environment: {env_name}')

    if frame_stack is not None:
        env = FrameStackWrapper(env, frame_stack)
        eval_env = FrameStackWrapper(eval_env, frame_stack)

    env.reset()
    eval_env.reset()

    # Clip dataset actions.
    if action_clip_eps is not None:
        train_dataset = train_dataset.copy(
            add_or_replace=dict(actions=np.clip(train_dataset['actions'], -1 + action_clip_eps, 1 - action_clip_eps))
        )
        if val_dataset is not None:
            val_dataset = val_dataset.copy(
                add_or_replace=dict(actions=np.clip(val_dataset['actions'], -1 + action_clip_eps, 1 - action_clip_eps))
            )

    return env, eval_env, train_dataset, val_dataset

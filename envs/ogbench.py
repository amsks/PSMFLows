# Copyright (c) Meta Platforms, Inc. and affiliates.
# CC BY-NC 4.0 license

import typing as tp
from functools import partial

import gymnasium
import numpy as np
from ogbench.utils import make_env_and_datasets

from envs.wrappers import PixelWrapper

CUBE_DOMAINS = ["cube-single-play-v0", "cube-double-play-v0"]
PUZZLE_DOMAINS = ["puzzle-3x3-play-v0"]
SCENE_DOMAINS = ["scene-play-v0"]
ANT_DOMAINS = []
for size in ["medium", "large", "giant"]:
    for data_type in ["navigate", "stitch", "explore"]:
        ANT_DOMAINS += [f"antmaze-{size}-{data_type}-v0"]
ALL_DOMAINS = CUBE_DOMAINS + PUZZLE_DOMAINS + SCENE_DOMAINS + ANT_DOMAINS
ALL_TASKS = {}
for d in ALL_DOMAINS:
    ALL_TASKS[d] = [d[:-2] + f"singletask-task{i + 1}-" + d[-2:] for i in range(5)]


def cube_reward_fn(qpos: np.ndarray, action: np.ndarray, *, target_position: np.ndarray, threshold: float = 0.04) -> np.ndarray:
    num_cubes = target_position.shape[0]
    cube_positions = [qpos[..., 14:17], qpos[..., 21:24], qpos[..., 28:31], qpos[..., 35:38]][:num_cubes]
    distances = [np.linalg.norm(cpos - tpos, axis=-1) for cpos, tpos in zip(cube_positions, target_position)]
    successes = sum([(d < threshold).astype(float) for d in distances])
    return (successes - num_cubes).reshape(-1, 1)


def puzzle_reward_fn(qpos: np.ndarray, action: np.ndarray, *, target_position: np.ndarray) -> np.ndarray:
    return (qpos[:, -len(target_position):] == target_position).sum(-1, keepdims=True).astype(float) - len(target_position)


def scene_reward_fn(qpos: np.ndarray, action: np.ndarray, *, target_position: np.ndarray, threshold: float = 0.04) -> np.ndarray:
    cube = (np.linalg.norm(qpos[..., 14:17] - target_position[:3], axis=-1) <= threshold).astype(float)
    button1 = (qpos[..., -2] == target_position[3]).astype(float)
    button2 = (qpos[..., -1] == target_position[4]).astype(float)
    drawer = (np.abs(qpos[..., -4] - target_position[5]) <= threshold).astype(float)
    window = (np.abs(qpos[..., -3] - target_position[6]) <= threshold).astype(float)
    return (cube + button1 + button2 + drawer + window - 5).reshape(-1, 1)


def ant_reward_fn(qpos: np.ndarray, action: np.ndarray, *, target_position: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (np.linalg.norm(qpos[..., :2] - target_position, axis=-1).reshape(-1, 1) <= threshold) - 1.0


def create_ogbench_env(
    task: str,
    wrappers: tp.List = None,
    seed: tp.Optional[int] = None,
    render_height: int = 64,
    render_width: int = 64,
    obs_type: str = "state",
    frame_stack: int = 1,
):
    if wrappers is None:
        wrappers = []
    match obs_type:
        case "state":
            pass
        case "pixels":
            task = "visual-" + task
            wrappers = wrappers + [lambda e: PixelWrapper(e, frame_stack)]
        case _:
            raise ValueError(f"Unsupported observation type {obs_type}")
    env_kwargs = {"height": render_height, "width": render_width}
    env = make_env_and_datasets(task, env_only=True, **env_kwargs)
    for wrapper in wrappers:
        env = wrapper(env)
    env.reset(seed=seed)
    return env, {}


def get_relabel_fn(domain: str, task: str):
    """Return a reward function (qpos, action) -> reward for a specific task."""
    env = make_env_and_datasets(task, env_only=True)
    env.reset()  # required for antmaze to initialize goal
    if domain in CUBE_DOMAINS:
        return partial(cube_reward_fn, target_position=env.unwrapped.cur_task_info["goal_xyzs"])
    if domain in PUZZLE_DOMAINS:
        return partial(puzzle_reward_fn, target_position=env.unwrapped.cur_task_info["goal_button_states"])
    if domain in SCENE_DOMAINS:
        return partial(
            scene_reward_fn,
            target_position=np.concatenate([np.array([v]).ravel() for v in env.unwrapped.cur_task_info["goal"].values()]),
        )
    if domain in ANT_DOMAINS:
        return partial(ant_reward_fn, target_position=env.unwrapped.get_oracle_rep())
    raise NotImplementedError(f"Unknown domain: {domain}")

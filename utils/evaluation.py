from collections import defaultdict

import jax
import numpy as np
from tqdm import trange


def extract_goal(env):
    """Return the goal observation for a goal-conditioned OGBench env, or None.
    OGBench exposes info['goal'] on reset (see envs/env_utils.py frame-stack wrapper)."""
    _, info = env.reset()
    return info.get('goal', None)


def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """Helper function to split the random number generator key before each call to the function."""

    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)

    return wrapped


def flatten(d, parent_key='', sep='.'):
    """Flatten a dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, 'items'):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    """Append values to the corresponding lists in the dictionary."""
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def evaluate(
    agent,
    env,
    config=None,
    num_eval_episodes=50,
    num_video_episodes=0,
    video_frame_skip=3,
    eval_temperature=0,
    seed=None,
):
    """Evaluate the agent in the environment.

    Args:
        agent: Agent.
        env: Environment.
        config: Configuration dictionary.
        num_eval_episodes: Number of episodes to evaluate the agent.
        num_video_episodes: Number of episodes to render. These episodes are not included in the statistics.
        video_frame_skip: Number of frames to skip between renders.
        eval_temperature: Action sampling temperature.
        seed: If not None, pin BOTH sources of eval randomness to this value at the start
            of every eval -- the environment's episode-init RNG (matching the reference
            `evals/ogbench.py`, which re-creates and re-seeds the eval env with `cfg.seed`
            on each eval) and the agent's action-sampling key stream -- so success is a
            reproducible function of the weights. When None, the env keeps its own
            (entropy-seeded) RNG, the action key is drawn from OS entropy, and eval is not
            reproducible run-to-run.

    Returns:
        A tuple containing the statistics, trajectories, and rendered videos.
    """
    # Action noise was seeded from OS entropy regardless of `seed`, so re-evaluating the
    # same weights drew a different action stream every time: pinning the env's episode
    # inits alone did not make eval reproducible. Stochastic actors (flow decode, the
    # residual path) move by more than rounding under a different key.
    actor_key = seed if seed is not None else np.random.randint(0, 2**32)
    actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(actor_key))
    trajs = []
    stats = defaultdict(list)

    # Pin the env's episode-init RNG to `seed` once per eval. The per-episode
    # resets below stay unseeded so they advance this seeded generator (matches
    # the reference: create_ogbench_env resets with seed, then rollout resets
    # unseeded per episode).
    if seed is not None:
        env.reset(seed=seed)

    renders = []
    for i in trange(num_eval_episodes + num_video_episodes):
        traj = defaultdict(list)
        should_render = i >= num_eval_episodes

        observation, info = env.reset()
        done = False
        step = 0
        render = []
        while not done:
            action = actor_fn(observations=observation, temperature=eval_temperature)
            action = np.array(action)
            action = np.clip(action, -1, 1)

            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1

            if should_render and (step % video_frame_skip == 0 or done):
                frame = env.render().copy()
                render.append(frame)

            transition = dict(
                observation=observation,
                next_observation=next_observation,
                action=action,
                reward=reward,
                done=done,
                info=info,
            )
            add_to(traj, transition)
            observation = next_observation
        if i < num_eval_episodes:
            add_to(stats, flatten(info))
            trajs.append(traj)
        else:
            renders.append(np.array(render))

    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats, trajs, renders

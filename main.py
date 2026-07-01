import os

import json
import random
import time

import hydra
import jax
import ml_collections
import numpy as np
import tqdm
import wandb
from omegaconf import DictConfig, OmegaConf

from agents import agents
from envs.env_utils import make_env_and_datasets
from utils.datasets import Dataset, ReplayBuffer
from utils.evaluation import evaluate, flatten
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import CsvLogger, get_exp_name, get_wandb_video, setup_wandb


def _lists_to_tuples(x):
    """Recursively convert lists to tuples (e.g. *_hidden_dims) so the agent
    config matches the original ml_collections shape and stays hashable for jax."""
    if isinstance(x, list):
        return tuple(_lists_to_tuples(v) for v in x)
    if isinstance(x, dict):
        return {k: _lists_to_tuples(v) for k, v in x.items()}
    return x


@hydra.main(version_base='1.3', config_path='configs', config_name='config')
def main(cfg: DictConfig):
    # Set up logger.
    exp_name = get_exp_name(cfg.seed)
    setup_wandb(
        project='fql', group=cfg.run_group, name=exp_name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    save_dir = os.path.join(cfg.save_dir, wandb.run.project, cfg.run_group, exp_name)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'flags.json'), 'w') as f:
        json.dump(OmegaConf.to_container(cfg, resolve=True), f, default=str)

    # Build the agent config (ml_collections.ConfigDict) from the Hydra agent group.
    config = ml_collections.ConfigDict(_lists_to_tuples(OmegaConf.to_container(cfg.agent, resolve=True)))

    # Make environment and datasets.
    env, eval_env, train_dataset, val_dataset = make_env_and_datasets(cfg.env_name, frame_stack=cfg.frame_stack)
    if cfg.video_episodes > 0:
        assert 'singletask' in cfg.env_name, 'Rendering is currently only supported for OGBench environments.'
    if cfg.online_steps > 0:
        assert 'visual' not in cfg.env_name, 'Online fine-tuning is currently not supported for visual environments.'

    # Initialize agent.
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Set up datasets.
    train_dataset = Dataset.create(**train_dataset)
    if cfg.balanced_sampling:
        # Create a separate replay buffer so that we can sample from both the training dataset and the replay buffer.
        example_transition = {k: v[0] for k, v in train_dataset.items()}
        replay_buffer = ReplayBuffer.create(example_transition, size=cfg.buffer_size)
    else:
        # Use the training dataset as the replay buffer.
        train_dataset = ReplayBuffer.create_from_initial_dataset(
            dict(train_dataset), size=max(cfg.buffer_size, train_dataset.size + 1)
        )
        replay_buffer = train_dataset
    # Set p_aug and frame_stack.
    for dataset in [train_dataset, val_dataset, replay_buffer]:
        if dataset is not None:
            dataset.p_aug = cfg.p_aug
            dataset.frame_stack = cfg.frame_stack
            if config['agent_name'] == 'rebrac':
                dataset.return_next_actions = True

    # Create agent.
    example_batch = train_dataset.sample(1)

    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        cfg.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
    )

    # Restore agent.
    if cfg.restore_path is not None:
        agent = restore_agent(agent, cfg.restore_path, cfg.restore_epoch)

    # Train agent.
    train_logger = CsvLogger(os.path.join(save_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(save_dir, 'eval.csv'))
    first_time = time.time()
    last_time = time.time()

    step = 0
    done = True
    expl_metrics = dict()
    online_rng = jax.random.PRNGKey(cfg.seed)
    for i in tqdm.tqdm(range(1, cfg.offline_steps + cfg.online_steps + 1), smoothing=0.1, dynamic_ncols=True):
        if i <= cfg.offline_steps:
            # Offline RL.
            batch = train_dataset.sample(config['batch_size'])

            if config['agent_name'] == 'rebrac':
                agent, update_info = agent.update(batch, full_update=(i % config['actor_freq'] == 0))
            else:
                agent, update_info = agent.update(batch)
        else:
            # Online fine-tuning.
            online_rng, key = jax.random.split(online_rng)

            if done:
                step = 0
                ob, _ = env.reset()

            action = agent.sample_actions(observations=ob, temperature=1, seed=key)
            action = np.array(action)

            next_ob, reward, terminated, truncated, info = env.step(action.copy())
            done = terminated or truncated

            if 'antmaze' in cfg.env_name and (
                'diverse' in cfg.env_name or 'play' in cfg.env_name or 'umaze' in cfg.env_name
            ):
                # Adjust reward for D4RL antmaze.
                reward = reward - 1.0

            replay_buffer.add_transition(
                dict(
                    observations=ob,
                    actions=action,
                    rewards=reward,
                    terminals=float(done),
                    masks=1.0 - terminated,
                    next_observations=next_ob,
                )
            )
            ob = next_ob

            if done:
                expl_metrics = {f'exploration/{k}': np.mean(v) for k, v in flatten(info).items()}

            step += 1

            # Update agent.
            if cfg.balanced_sampling:
                # Half-and-half sampling from the training dataset and the replay buffer.
                dataset_batch = train_dataset.sample(config['batch_size'] // 2)
                replay_batch = replay_buffer.sample(config['batch_size'] // 2)
                batch = {k: np.concatenate([dataset_batch[k], replay_batch[k]], axis=0) for k in dataset_batch}
            else:
                batch = replay_buffer.sample(config['batch_size'])

            if config['agent_name'] == 'rebrac':
                agent, update_info = agent.update(batch, full_update=(i % config['actor_freq'] == 0))
            else:
                agent, update_info = agent.update(batch)

        # Log metrics.
        if i % cfg.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            if val_dataset is not None:
                val_batch = val_dataset.sample(config['batch_size'])
                _, val_info = agent.total_loss(val_batch, grad_params=None)
                train_metrics.update({f'validation/{k}': v for k, v in val_info.items()})
            train_metrics['time/epoch_time'] = (time.time() - last_time) / cfg.log_interval
            train_metrics['time/total_time'] = time.time() - first_time
            train_metrics.update(expl_metrics)
            last_time = time.time()
            wandb.log(train_metrics, step=i)
            train_logger.log(train_metrics, step=i)

        # Evaluate agent.
        if cfg.eval_interval != 0 and (i == 1 or i % cfg.eval_interval == 0):
            renders = []
            eval_metrics = {}
            # PSM acts on a reward-inferred task latent; infer it from the (trained)
            # agent over a dataset sample so eval is goal-directed. No-op for agents
            # without infer_eval_z (they act directly on observations).
            eval_agent = agent
            if hasattr(agent, 'infer_eval_z'):
                z_batch = train_dataset.sample(min(train_dataset.size, 100000))
                eval_agent = agent.infer_eval_z(z_batch['next_observations'], z_batch['rewards'])
            eval_info, trajs, cur_renders = evaluate(
                agent=eval_agent,
                env=eval_env,
                config=config,
                num_eval_episodes=cfg.eval_episodes,
                num_video_episodes=cfg.video_episodes,
                video_frame_skip=cfg.video_frame_skip,
            )
            renders.extend(cur_renders)
            for k, v in eval_info.items():
                eval_metrics[f'evaluation/{k}'] = v

            if cfg.video_episodes > 0:
                video = get_wandb_video(renders=renders)
                eval_metrics['video'] = video

            wandb.log(eval_metrics, step=i)
            eval_logger.log(eval_metrics, step=i)

        # Save agent.
        if i % cfg.save_interval == 0:
            save_agent(agent, save_dir, i)

    train_logger.close()
    eval_logger.close()


if __name__ == '__main__':
    main()

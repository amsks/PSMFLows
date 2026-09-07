"""Train an ARCHIVED agent -- the baseline entry point, `main.py` for `archive/agents/`.

`main.py` builds only the two live agents (`fql`, `psmflow`) and its eval block knows only
`infer_eval_z`. Reviving an archived agent inside it would mean re-adding branches to the
live entry point for agents the repo has deliberately frozen. This script is that entry
point instead: same hydra config tree, same offline loop, same checkpoint layout, plus the
two seams the archived successor-measure agents need and `main.py` no longer carries:

  * `dataset.return_index = True` -- the PSM proto (codebook) sampler keys pi_z on the
    GLOBAL buffer row index; without it, it keys on batch POSITION and the TD target is
    re-randomised every resample (the bug fixed in `f19a2ba`).
  * goal-conditioned eval -- `affine_psm` infers a task coordinate `w_inf` for the env's
    goal (`infer_eval` -> LP or closed form), where `psm` infers a task vector from
    relabelled rewards (`infer_eval_z`), exactly as `main.py` does for `psmflow`.

Nothing under `archive/` is moved or edited; see `scripts/baselines/_archive.py`.

Run (cube, affine PSM in RAW action space with the BC anchor OFF):
    MUJOCO_GL=egl .venv/bin/python scripts/baselines/run_archived.py \
        agent=affine_psm agent.actor.bc_coeff=0.0 \
        env_name=cube-single-play-singletask-v0 offline_steps=500000 ...
"""
import json
import os
import random
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))  # repo root
sys.path.insert(0, _HERE)                                    # this directory, for _archive

# isort: off  -- utils.xla_guard MUST be the first import that reaches jax: XLA:GPU's
# autotuner miscompiles the unrolled flow ODE. Do not sort this block.
import utils.xla_guard  # noqa: F401
# isort: on

import hydra
import ml_collections
import numpy as np
import tqdm
import wandb
from _archive import all_agents, register_archive_configs
from omegaconf import DictConfig, OmegaConf

from envs.env_utils import make_env_and_datasets
from main import _lists_to_tuples
from utils.datasets import Dataset, ReplayBuffer
from utils.evaluation import evaluate, extract_goal
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import CsvLogger, get_exp_name, setup_wandb

register_archive_configs()

# Agents whose proto/codebook policy keys on the global buffer row index.
_NEEDS_INDEX = ('psm', 'affine_psm', 'latent_affine_psm')


def infer_for_eval(agent, cfg, dataset, eval_env):
    """The archived agents' two task-inference protocols, verbatim from the pre-archive
    `main.py` (commit db96e48). Returns the agent to roll out with."""
    if hasattr(agent, 'infer_w_goal'):
        # affine_psm: goal-conditioned. Solve w_inf for the env goal (LP `full`, or the
        # closed form under inference.mode=zero_shot); the w-conditioned actor acts greedily.
        goal = extract_goal(eval_env)
        assert goal is not None, 'this agent needs a goal-conditioned env (info["goal"]).'
        return agent.infer_eval(dataset, goal)
    if hasattr(agent, 'infer_eval_z'):
        # psm/fb: task vector from relabelled rewards, shifted so {-1,0} -> {0,1}.
        n_relabel = min(dataset.size, int(cfg.get('eval_relabel_size', 10000)))
        zb = dataset.sample(n_relabel)
        return agent.infer_eval_z(zb['next_observations'],
                                  zb['rewards'] + float(cfg.get('eval_reward_shift', 1.0)))
    return agent


@hydra.main(version_base='1.3', config_path='../../configs', config_name='config')
def main(cfg: DictConfig):
    exp_name = get_exp_name(cfg.seed)
    setup_wandb(entity=cfg.wandb_entity, project=cfg.wandb_project, group=cfg.run_group,
                name=exp_name, config=OmegaConf.to_container(cfg, resolve=True))
    save_dir = os.path.join(cfg.save_dir, wandb.run.project, cfg.run_group, exp_name)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'flags.json'), 'w') as f:
        json.dump(OmegaConf.to_container(cfg, resolve=True), f, default=str)

    config = ml_collections.ConfigDict(_lists_to_tuples(OmegaConf.to_container(cfg.agent, resolve=True)))
    name = config['agent_name']

    _env, eval_env, train_dataset, val_dataset = make_env_and_datasets(
        cfg.env_name, frame_stack=cfg.frame_stack,
        dataset_fraction=cfg.get('dataset_fraction', 1.0),
        dataset_fraction_seed=cfg.get('dataset_fraction_seed', 0))
    assert cfg.online_steps == 0, 'baseline runs are offline only.'

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    train_dataset = Dataset.create(**train_dataset)
    train_dataset = ReplayBuffer.create_from_initial_dataset(
        dict(train_dataset), size=max(cfg.buffer_size, train_dataset.size + 1))
    for dataset in [train_dataset, val_dataset]:
        if dataset is None:
            continue
        dataset.p_aug = cfg.p_aug
        dataset.frame_stack = cfg.frame_stack
        if name in _NEEDS_INDEX:
            dataset.return_index = True

    example_batch = train_dataset.sample(1)
    agent_class = all_agents()[name]
    agent = agent_class.create(cfg.seed, example_batch['observations'], example_batch['actions'], config)
    if cfg.restore_path is not None:
        agent = restore_agent(agent, cfg.restore_path, cfg.restore_epoch)

    train_logger = CsvLogger(os.path.join(save_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(save_dir, 'eval.csv'))
    first_time = last_time = time.time()

    for i in tqdm.tqdm(range(1, cfg.offline_steps + 1), smoothing=0.1, dynamic_ncols=True):
        batch = train_dataset.sample(config['batch_size'])
        agent, update_info = agent.update(batch)

        if i % cfg.log_interval == 0:
            metrics = {f'training/{k}': v for k, v in update_info.items()}
            if val_dataset is not None:
                _, val_info = agent.total_loss(val_dataset.sample(config['batch_size']))
                metrics.update({f'validation/{k}': v for k, v in val_info.items()})
            metrics['time/epoch_time'] = (time.time() - last_time) / cfg.log_interval
            metrics['time/total_time'] = time.time() - first_time
            last_time = time.time()
            wandb.log(metrics, step=i)
            train_logger.log(metrics, step=i)

        if cfg.eval_interval != 0 and (i == 1 or i % cfg.eval_interval == 0):
            eval_agent = infer_for_eval(agent, cfg, train_dataset, eval_env)
            eval_info, _, _ = evaluate(agent=eval_agent, env=eval_env, config=config,
                                       num_eval_episodes=cfg.eval_episodes, num_video_episodes=0,
                                       seed=cfg.seed)
            eval_metrics = {f'evaluation/{k}': v for k, v in eval_info.items()}
            wandb.log(eval_metrics, step=i)
            eval_logger.log(eval_metrics, step=i)

        if i % cfg.save_interval == 0:
            save_agent(agent, save_dir, i)

    train_logger.close()
    eval_logger.close()


if __name__ == '__main__':
    main()

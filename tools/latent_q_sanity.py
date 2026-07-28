"""D4: oracle-reward flow-GPI sanity — the representation + GPI pipeline, isolated.

Loads a TRAINED psmflow ckpt, infers w from the dataset's TRUE task rewards (with the
standard +1 eval shift), runs the seeded eval, prints success. This removes reward
inference quality as a variable: if success is near zero even with oracle rewards, the
problem is the representation or GPI, not the inference.

Compare against the FQL benchmark number for the same env (docs/reference_benchmarks.md).
Gate (spec §1): >= 50% of FQL on pointmaze-medium before scaling to cube.

Run: JAX_PLATFORMS='' .venv/bin/python tools/latent_q_sanity.py \
    agent=psmflow agent.preimage_path=<npz> agent.flow_ckpt_path=<flow_dir> \
    env_name=pointmaze-medium-navigate-singletask-task1-v0 \
    restore_path='/var/local/amsks/exp/<psmflow_run_dir>' restore_epoch=500000

On midi-01 add MUJOCO_GL=egl (no DISPLAY) and OGBENCH_DATASET_DIR=/var/local/amsks/ogbench.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils.xla_guard  # noqa: F401  -- MUST precede jax (see module docstring)

import hydra
import ml_collections
import numpy as np
from omegaconf import OmegaConf

from agents import agents
from envs.env_utils import make_env_and_datasets
from main import _lists_to_tuples
from utils.datasets import Dataset
from utils.evaluation import evaluate
from utils.flax_utils import restore_agent


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    env, eval_env, train_dataset, _ = make_env_and_datasets(cfg.env_name, frame_stack=cfg.frame_stack)
    ds = Dataset.create(**train_dataset)
    config = ml_collections.ConfigDict(_lists_to_tuples(OmegaConf.to_container(cfg.agent, resolve=True)))
    assert config['agent_name'] == 'psmflow'
    ex = ds.sample(1)
    agent = agents['psmflow'].create(cfg.seed, ex['observations'], ex['actions'], config)
    assert cfg.restore_path is not None, 'D4 evaluates a TRAINED psmflow checkpoint'
    agent = restore_agent(agent, cfg.restore_path, cfg.restore_epoch)

    n = min(ds.size, int(cfg.get('eval_relabel_size', 10000)))
    zb = ds.sample(n)
    # The +1 shift turns cube-single's {-1, 0} task reward into {0, 1} so w points at
    # goal-reaching states; without it z is inverted (see main.py eval hook).
    rew = zb['rewards'] + float(cfg.get('eval_reward_shift', 1.0))
    agent = agent.infer_eval_z(zb['next_observations'], rew)
    info, _, _ = evaluate(agent=agent, env=eval_env, config=config,
                          num_eval_episodes=50, num_video_episodes=0,
                          video_frame_skip=3, seed=cfg.seed)
    print(json.dumps({k: float(v) for k, v in info.items()}, indent=2))


if __name__ == "__main__":
    main()

"""Invert the BC flow over a whole dataset and persist the augmented dataset.

WP1 (occupancy-learning data pipeline): precompute, per transition, the EM Gaussian
mixture over the noise preimage of each action, then save it so PSMFlows training just
loads the augmented dataset instead of re-running the expensive inversion.

Run (plumbing smoke, untrained flow):
  JAX_PLATFORMS=cpu .venv-jax/bin/python tools/precompute_preimages.py \
      env_name=pointmaze-medium-navigate-singletask-task1-v0 \
      inversion.num_samples=20 inversion.n_steps=2 inversion.n_initial_steps=10

NOTE: meaningful preimages require a TRAINED flow (load a checkpoint via cfg.restore_path).
A freshly-created flow is acceptable ONLY for shape/pipeline smoke — its preimages carry
no behavior information. See plan Task 2 Step 6.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra

from agents.fql import FQLAgent, get_config
from envs.env_utils import make_env_and_datasets
from utils.datasets import Dataset
from utils.flow_inversion import (
    augment_dataset_with_preimage_distribution,
    save_augmented_dataset,
)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    _, _, train_dataset, _ = make_env_and_datasets(cfg.env_name, frame_stack=cfg.frame_stack)
    ds = dict(Dataset.create(**train_dataset))

    # Optional slice for plumbing smokes (the full OGBench dataset is ~1M transitions).
    limit = cfg.get('preimage_limit', None)
    if limit is not None:
        ds = {k: v[:limit] for k, v in ds.items()}

    agent_cfg = get_config()
    agent_cfg['flow_steps'] = 100
    agent = FQLAgent.create(cfg.seed, ds['observations'][:1], ds['actions'][:1], agent_cfg)

    out = augment_dataset_with_preimage_distribution(agent, ds, dict(cfg.inversion))
    out_path = cfg.get('preimage_out', 'preimages.npz')
    save_augmented_dataset(out_path, out)
    print(f"Wrote augmented dataset -> {out_path} "
          f"(num_clusters={cfg.inversion.get('num_clusters', 1)}, n={out['actions'].shape[0]})")


if __name__ == "__main__":
    main()

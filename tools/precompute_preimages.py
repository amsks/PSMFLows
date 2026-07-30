"""Invert the BC flow over a whole dataset and persist the augmented dataset.

Stage B of the PSMFlow v1 pipeline (plan Task 2). Precomputes, per transition:
  - the EM Gaussian mixture over the noise preimage of the action (WP1),
  - the exact backward-ODE point preimage (point-vs-mixture ablation),
  - the `preimage_roundtrip` / `preimage_ess` health scalars,
so PSMFlows training just loads the augmented dataset instead of re-running the
expensive inversion.

Run against a TRAINED Stage-A flow (scripts/pretrain_behavior_flow.sh):

  .venv/bin/python tools/precompute_preimages.py agent=fql \
      env_name=cube-single-play-singletask-v0 \
      restore_path='/var/local/amsks/exp/PSMFLows/bcflow_*/sd000_*' restore_epoch=500000 \
      agent.flow_steps=100 preimage_out=/var/local/amsks/exp/PSMFLows/preimages_cube.npz

`agent.flow_steps` and `inversion.n_initial_steps` MUST be equal (the forward and inverse
maps have to share a discretization, else preimage_roundtrip measures the mismatch) and
MUST be >= 100. The implicit-Euler inversion diverges outright at the training default of
10 — diagnostic D3 on cube-single reports a NaN round-trip at 10 steps, KS 0.217 at 30,
and 1.2e-4 at 100. Both are asserted below.

Preimages from an UNTRAINED flow carry no behavior information, so a checkpoint is
required. For a shape/pipeline smoke only, override the guard:

  MUJOCO_GL=egl JAX_PLATFORMS=cpu .venv/bin/python tools/precompute_preimages.py agent=fql \
      env_name=pointmaze-medium-navigate-singletask-task1-v0 \
      inversion.allow_untrained=true inversion.num_samples=6 inversion.n_steps=1 \
      inversion.n_initial_steps=2 agent.flow_steps=2 preimage_limit=64 \
      preimage_out=/tmp/pre_smoke.npz

(MUJOCO_GL=egl is needed on midi-01: the pointmaze env builds a renderer and there is no
DISPLAY. Set OGBENCH_DATASET_DIR=/var/local/amsks/ogbench to keep datasets off home NFS.)
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils.xla_guard  # noqa: F401  -- MUST precede jax (see module docstring)

import hydra
import ml_collections
from omegaconf import OmegaConf

from agents.fql import FQLAgent
from envs.env_utils import make_env_and_datasets
from main import _lists_to_tuples
from utils.datasets import Dataset
from utils.flax_utils import restore_agent
from utils.flow_inversion import (
    augment_dataset_with_point_preimage,
    augment_dataset_with_preimage_distribution,
    repair_invalid_preimages,
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

    # Build the agent from the Hydra `agent` group, NOT fql.get_config(): the net shapes
    # must match the Stage-A run whose checkpoint we are about to restore.
    assert cfg.agent.agent_name == 'fql', 'run with agent=fql (Stage-A flow shapes)'
    agent_cfg = ml_collections.ConfigDict(
        _lists_to_tuples(OmegaConf.to_container(cfg.agent, resolve=True)))
    agent = FQLAgent.create(cfg.seed, ds['observations'][:1], ds['actions'][:1], agent_cfg)

    # The point preimage is inverted at inversion.n_initial_steps but decoded at the
    # agent's flow_steps. Round-trip consistency only holds when the forward and inverse
    # maps share a discretization; if these disagree, every preimage_roundtrip written
    # into the npz measures the mismatch instead of inversion quality — a silent, and very
    # believable, false alarm baked into ~1M rows.
    assert int(agent_cfg['flow_steps']) == int(cfg.inversion.n_initial_steps), (
        f"agent.flow_steps ({agent_cfg['flow_steps']}) must equal "
        f"inversion.n_initial_steps ({cfg.inversion.n_initial_steps}); "
        'the inverse and forward maps must share a discretization')

    if cfg.restore_path is not None:
        agent = restore_agent(agent, cfg.restore_path, cfg.restore_epoch)
        # Only meaningful for a real run: the smoke path below deliberately uses tiny
        # step counts. At 10 the implicit-Euler fixed point diverges to NaN outright and
        # at 30 typicality is strongly rejected (D3, cube-single).
        assert int(cfg.inversion.n_initial_steps) >= 100, (
            f'inversion.n_initial_steps ({cfg.inversion.n_initial_steps}) must be >= 100; '
            'the inverter does not converge at the flow_steps=10 training default')
    else:
        assert cfg.inversion.get('allow_untrained', False), (
            'preimages from an UNTRAINED flow carry no behavior information; '
            'set restore_path=<stage-A ckpt dir> '
            '(or inversion.allow_untrained=true for plumbing smokes)')

    out = augment_dataset_with_preimage_distribution(agent, ds, dict(cfg.inversion))
    out = augment_dataset_with_point_preimage(agent, out, dict(cfg.inversion))

    # The flow inverse can diverge on individual transitions and NaN spreads from there
    # through every product built on it. Record which rows are trustworthy BEFORE writing,
    # so a training run never has to rediscover it, and abort outright if the count says the
    # inversion is broken rather than merely tailed.
    out, valid = repair_invalid_preimages(out)
    n_bad = int((valid < 0.5).sum())

    out_path = cfg.get('preimage_out', 'preimages.npz')
    save_augmented_dataset(out_path, out)
    # Sidecar: which flow produced these latents. Without it an npz is unusable — the
    # preimages are only meaningful for the exact checkpoint and discretization below.
    with open(out_path + '.meta.json', 'w') as f:
        json.dump(dict(env_name=cfg.env_name, restore_path=str(cfg.restore_path),
                       restore_epoch=cfg.restore_epoch, flow_steps=int(agent_cfg['flow_steps']),
                       inversion=OmegaConf.to_container(cfg.inversion, resolve=True),
                       num_transitions=int(out['actions'].shape[0]),
                       num_invalid_preimages=n_bad), f, indent=2)
    print(f"Wrote {out_path} (+ .meta.json), n={out['actions'].shape[0]}, "
          f"invalid={n_bad}")


if __name__ == "__main__":
    main()

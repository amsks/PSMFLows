"""Hypersweeper/SMAC target for the preimage-inversion trade-off.

One call = one trial: invert the fixed sampled batch under the configured `inversion`
settings (the sweeper overrides `inversion.alpha` / `inversion.prior_scale` /
`inversion.n_steps` per trial) and return the scalar the optimizer minimizes:

    cost = -coverage@k  +  penalty * max(0, decode_mix - budget)

i.e. maximize coverage subject to a decode-error budget, scalarized with a hinge --
the decision rule from docs/TUNING_INVERSION.md ("widest setting inside a decode-error
budget") stated as an objective. `coverage@k` is the full-k end of the `covered_at_k`
curve (any of the row's `cov_k` nearest states may supply the latent).

The batch, its neighbourhood sampling, and every scoring detail are exactly
tools/tune_preimage_inversion.py's (`_score` is imported from it), and the batch seed is
fixed, so all trials score the same rows and the same prior draws: differences between
trials are the setting, nothing else. The dataset, batch, and restored agent are cached
at module level -- hydra's basic launcher runs every trial in this process, so the
per-trial cost is the inversion itself, not the setup.

Run (sweep):
  XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python tools/hpo_preimage_inversion.py -m \
      --config-name=hpo_preimage agent=fql env_name=cube-single-play-singletask-v0 \
      restore_path=<ckpt dir> restore_epoch=500000 agent.flow_steps=100 \
      +n_rows=1280 +sample_k=128 +cov_k=64

Single trial (no -m) evaluates the configured baseline once and prints the cost.
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

from agents.fql import FQLAgent
from envs.env_utils import make_env_and_datasets
from main import _lists_to_tuples
from tools.tune_preimage_inversion import _score
from utils.datasets import Dataset
from utils.flax_utils import restore_agent
from utils.nn_search import neighborhood_sample

# One entry per (env, ckpt, batch spec): hydra's basic launcher runs all trials of a
# sweep in this process, so the dataset/batch/agent survive across trials.
_CACHE = {}


def _setup(cfg):
    key = (cfg.env_name, str(cfg.restore_path), str(cfg.restore_epoch), int(cfg.seed),
           int(cfg.get('n_rows', 2000)), str(cfg.get('sample_mode', 'neighborhoods')),
           int(cfg.get('sample_k', 20)), int(cfg.agent.flow_steps))
    if key in _CACHE:
        return _CACHE[key]

    _, _, train_dataset, _ = make_env_and_datasets(cfg.env_name, frame_stack=cfg.frame_stack)
    ds = dict(Dataset.create(**train_dataset))
    n_rows = int(cfg.get('n_rows', 2000))
    sample_mode = str(cfg.get('sample_mode', 'neighborhoods'))
    if sample_mode == 'neighborhoods':
        obs_all = np.asarray(ds['observations'], np.float32)
        rows = neighborhood_sample(obs_all, n_rows, int(cfg.get('sample_k', 20)),
                                   seed=int(cfg.seed), scale=np.maximum(obs_all.std(0), 1e-6))
    elif sample_mode == 'uniform':
        size = len(ds['observations'])
        rows = np.sort(np.random.default_rng(int(cfg.seed)).choice(size, n_rows, replace=False))
    else:
        raise ValueError(f'+sample_mode must be neighborhoods|uniform, got {sample_mode!r}')
    batch = {k: v[rows] for k, v in ds.items() if k in ('observations', 'actions')}

    agent_cfg = ml_collections.ConfigDict(
        _lists_to_tuples(OmegaConf.to_container(cfg.agent, resolve=True)))
    agent = FQLAgent.create(int(cfg.seed), batch['observations'][:1], batch['actions'][:1], agent_cfg)
    assert cfg.restore_path is not None, 'HPO against an untrained flow measures nothing'
    agent = restore_agent(agent, cfg.restore_path, cfg.restore_epoch)

    _CACHE[key] = (agent, batch)
    return _CACHE[key]


@hydra.main(version_base=None, config_path='../configs', config_name='hpo_preimage')
def main(cfg):
    assert cfg.agent.agent_name == 'fql', 'run with agent=fql (Stage-A flow shapes)'
    assert int(cfg.agent.flow_steps) == int(cfg.inversion.n_initial_steps), (
        f'agent.flow_steps ({cfg.agent.flow_steps}) must equal inversion.n_initial_steps '
        f'({cfg.inversion.n_initial_steps})')
    agent, batch = _setup(cfg)

    inv = OmegaConf.to_container(cfg.inversion, resolve=True)
    record = _score(agent, batch, inv, int(inv.get('batch_size', 256)),
                    cov_k=int(cfg.get('cov_k', 16)), cov_draws=int(cfg.get('cov_draws', 8)))

    curve = record['coverage']['covered_at_k']
    coverage = curve[str(record['coverage']['k_max'])]
    decode = record['decode_mix_mean']
    budget = float(cfg.hpo_decode_budget)
    penalty = float(cfg.hpo_penalty)
    if decode is None:  # every mixture draw decoded to NaN: worst possible setting
        cost = penalty
    else:
        cost = -coverage + penalty * max(0.0, decode - budget)

    trial = {'inversion': {k: inv[k] for k in ('alpha', 'prior_scale', 'n_steps',
                                               'num_samples', 'num_clusters')},
             'cost': round(float(cost), 6), 'coverage_at_kmax': coverage, **record}
    print(json.dumps(trial))
    with open('hpo_trial.json', 'w') as f:  # hydra runs each trial in its own run dir
        json.dump(trial, f, indent=2)
    return float(cost)


if __name__ == '__main__':
    main()

"""500-episode evaluation of an ARCHIVED agent's checkpoint.

`tools/eval_checkpoint.py` is the evaluator of record, but it builds agents from
`agents.agents` (live registry only) and its task inference is `infer_eval_z` only. This
is the same tool for the frozen agents: it REUSES `eval_checkpoint`'s config-provenance
and interval machinery unchanged (`merge_run_config`, `_cli_agent_keys`, `wilson`) and
differs only in (a) the registry it looks the agent up in and (b) the goal-conditioned
inference path `affine_psm` needs. Report JSON fields match `eval_checkpoint`'s so
`tools/make_tables.py` and the existing eval500 JSONs stay comparable.

Run:
    MUJOCO_GL=egl .venv/bin/python scripts/baselines/eval_archived.py \
        agent=affine_psm env_name=cube-single-play-singletask-v0 \
        restore_path=<run_dir> restore_epoch=500000 eval_episodes=500 \
        report_out=<out.json>
"""
import os
import sys

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
from _archive import all_agents, register_archive_configs
from omegaconf import OmegaConf
from run_archived import infer_for_eval

from envs.env_utils import make_env_and_datasets
from main import _lists_to_tuples
from tools.eval_checkpoint import _cli_agent_keys, merge_run_config, wilson
from utils.datasets import Dataset
from utils.evaluation import evaluate
from utils.flax_utils import restore_agent
from utils.log_utils import write_report

register_archive_configs()


@hydra.main(version_base=None, config_path='../../configs', config_name='config')
def main(cfg):
    np.random.seed(int(cfg.seed))
    _env, eval_env, train_dataset, _ = make_env_and_datasets(cfg.env_name, frame_stack=cfg.frame_stack)
    ds = Dataset.create(**train_dataset)

    cli_agent = OmegaConf.to_container(cfg.agent, resolve=True)
    merged, prov = merge_run_config(cli_agent, cfg.restore_path, _cli_agent_keys())
    if prov['flags_json']:
        print(f"agent config defaults from {prov['flags_json']}")
        for k, d in sorted(prov['inherited'].items()):
            print(f"  {k}: config {d['config']!r} -> run {d['run']!r}")
        if prov['cli_overrides']:
            print(f"  CLI overrides kept: {', '.join(prov['cli_overrides'])}")
    else:
        print(f'NOTE: no flags.json under restore_path={cfg.restore_path!r}')
    config = ml_collections.ConfigDict(_lists_to_tuples(merged))
    name = config['agent_name']

    # The proto sampler reads batch['index'] during eval-time inference batches too.
    ds.return_index = True

    ex = ds.sample(1)
    agent = all_agents()[name].create(cfg.seed, ex['observations'], ex['actions'], config)
    assert cfg.restore_path is not None, 'needs a trained checkpoint (restore_path)'
    agent = restore_agent(agent, cfg.restore_path, cfg.restore_epoch)
    agent = infer_for_eval(agent, cfg, ds, eval_env)

    n_ep = int(cfg.eval_episodes)
    info, trajs, _ = evaluate(agent=agent, env=eval_env, config=config, num_eval_episodes=n_ep,
                              num_video_episodes=0, seed=int(cfg.seed))
    per_ep = [float(np.max(np.asarray(t['info'][-1].get('success', 0.0)))) for t in trajs if 'info' in t]
    if per_ep:
        k, n = int(sum(p > 0.5 for p in per_ep)), len(per_ep)
    else:
        k, n = int(round(float(info['success']) * n_ep)), n_ep
    lo, hi = wilson(k, n)

    ac = config['actor']
    report = {
        'env': cfg.env_name,
        'agent': name,
        'archived': True,
        'action_space': 'raw',
        'acting_mode': (f"amortized {ac.get('type', 'ddpgbc')} actor over RAW actions, "
                        f"bc_coeff={ac.get('bc_coeff')}, "
                        f"inference={config.get('inference', {}).get('mode', 'reward-z')}"),
        'bc_coeff': ac.get('bc_coeff'),
        'restore_path': str(cfg.restore_path),
        'restore_epoch': int(cfg.restore_epoch) if cfg.restore_epoch is not None else None,
        'eval_episodes': n_ep,
        'seed': int(cfg.seed),
        'success': round(k / n, 4) if n else None,
        'successes': k,
        'episodes': n,
        'wilson95': [lo, hi],
        'per_episode_success': per_ep,
        'agent_config_source': prov,
        'agent_config': merged,
    }
    for key in ('success', 'episode_length', 'total.timesteps'):
        if key in info and key != 'success':
            report[key] = float(info[key])
    print(f"{name} {cfg.env_name} @ {cfg.restore_epoch}: success {k}/{n} = "
          f"{k / n:.4f}  Wilson95 [{lo}, {hi}]")
    write_report(report, cfg, default_name='eval_archived.json')


if __name__ == '__main__':
    main()

"""Strict final-endpoint aggregation across three independent training seeds."""

import argparse
import json
from pathlib import Path

import numpy as np

from tools.walker_psm.data import sha256, verify_run_sidecars

TASKS = ('stand', 'walk', 'run', 'flip')


def seed_statistics(task_returns):
    values = np.asarray(task_returns, np.float64)
    if values.ndim != 2 or values.shape[0] != 3 or not np.isfinite(values).all():
        raise ValueError('Require finite task matrix with three training seeds')
    means = values.mean(axis=1)
    halfwidth = float(4.302652729911275 * means.std(ddof=1) / np.sqrt(3))
    return {
        'seed_task_means': means.tolist(),
        'mean': float(means.mean()),
        'ci95_halfwidth': halfwidth,
        'interval': 'Student-t across training seeds, df=2',
    }


def aggregate(run_dirs, step=2000000, episodes=500):
    if len(run_dirs) != 3 or len({str(Path(p).resolve()) for p in run_dirs}) != 3:
        raise ValueError('Require three distinct training run directories')
    matrix, seeds, provenance = [], [], []
    common = None
    for directory in run_dirs:
        run = Path(directory)
        flags = json.loads((run / 'flags.json').read_text())
        verify_run_sidecars(run, flags)
        report = json.loads((run / f'eval_{step}.json').read_text())
        if report['step'] != step or report['seed'] != flags['seed']:
            raise ValueError('Checkpoint/seed mismatch')
        if not report.get('evaluated_restored_checkpoint') or not (run / f'checkpoint_{step}.pkl').is_file():
            raise ValueError('Final report must evaluate the saved checkpoint')
        if report['flags_sha256'] != sha256(run / 'flags.json'):
            raise ValueError('Saved evaluation does not match run flags')
        signature = {
            'agent': flags['agent'],
            'proto_table_sha256': flags['proto_table_sha256'],
            'dataset': sha256(run / 'dataset_manifest.json'),
            'source': sha256(run / 'source_manifest.json'),
            'inference_samples': report['inference_samples'],
            'eval_seed': report['eval_seed'],
        }
        if common is not None and signature != common:
            raise ValueError('Training/data/source/evaluation settings differ across seeds')
        common = signature
        if set(report['tasks']) != set(TASKS):
            raise ValueError('Incomplete task evaluation')
        row = []
        for task in TASKS:
            cell = report['tasks'][task]
            returns = np.asarray(cell['episode_returns'])
            if cell['episodes'] != episodes or len(returns) != episodes:
                raise ValueError('Evaluation episode count mismatch')
            if len(cell['episode_lengths']) != episodes or set(cell['episode_lengths']) != {1000}:
                raise ValueError('Unexpected evaluation horizon')
            if not np.isfinite(returns).all() or not np.isclose(returns.mean(), cell['mean_return']):
                raise ValueError('Invalid reported mean')
            row.append(float(returns.mean()))
        matrix.append(row)
        seeds.append(flags['seed'])
        provenance.append({'run_dir': str(run.resolve()), 'eval_sha256': sha256(run / f'eval_{step}.json')})
    if sorted(seeds) != [0, 1, 2]:
        raise ValueError('Require distinct training seeds0,1,2')
    return dict(
        **seed_statistics(matrix),
        seed_order=seeds,
        tasks=list(TASKS),
        task_return_matrix=matrix,
        step=step,
        episodes_per_task=episodes,
        provenance=provenance,
        common=common,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dirs', nargs=3, required=True)
    parser.add_argument('--step', type=int, default=2000000)
    parser.add_argument('--episodes', type=int, default=500)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    result = aggregate(args.run_dirs, args.step, args.episodes)
    with Path(args.out).open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps(result, indent=2))

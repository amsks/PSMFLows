"""Strict fixed-endpoint aggregation: tasks first, then three training seeds."""

import argparse
import json
from pathlib import Path

import numpy as np

from tools.affine_stitch.data import sha256, verify_run_sidecars


def seed_statistics(matrix):
    values = np.asarray(matrix, np.float64)
    if values.shape != (3, 5) or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError('Require finite3x5 success matrix within[0,1]')
    means = values.mean(axis=1)
    return {'seed_task_means': means.tolist(), 'mean': float(means.mean()),
            'ci95_halfwidth': float(4.302652729911275 * means.std(ddof=1) / np.sqrt(3)),
            'interval': 'Student-t across training seeds, df=2; tasks averaged within each seed'}


def aggregate(run_dirs, step=500000, episodes=500):
    if len(run_dirs) != 3 or len({str(Path(p).resolve()) for p in run_dirs}) != 3:
        raise ValueError('Require three distinct run directories')
    rows, seeds, provenance = [], [], []
    common = None
    for directory in run_dirs:
        run = Path(directory).resolve()
        flags = json.loads((run / 'flags.json').read_text())
        verify_run_sidecars(run, flags)
        report_path = run / f'eval_{step}.json'
        report = json.loads(report_path.read_text())
        if report['step'] != step or flags['run']['steps'] != step or report['seed'] != flags['seed']:
            raise ValueError('Endpoint or seed mismatch')
        if not report['evaluated_restored_checkpoint'] or report['flags_sha256'] != sha256(run / 'flags.json'):
            raise ValueError('Evaluation must restore checkpoint-bound run flags')
        if report['checkpoint_sha256'] != sha256(run / f'checkpoint_{step}.pkl'):
            raise ValueError('Evaluated checkpoint hash changed')
        signature = {
            'agent': flags['agent'], 'proto_mode': flags['proto_mode'],
            'source': flags['source_manifest_sha256'], 'dataset': flags['dataset_manifest_sha256'],
            'eval_seed': report['eval_seed'], 'workers': report['workers'], 'worker_mode': report['worker_mode'],
            'goal_hashes': {k: value['goal_sha256'] for k, value in report['tasks'].items()},
        }
        if common is not None and signature != common:
            raise ValueError('Source/data/training/evaluation settings differ across seeds')
        common = signature
        if set(report['tasks']) != {'1', '2', '3', '4', '5'}:
            raise ValueError('Incomplete five-task matrix')
        row = []
        for task in map(str, range(1, 6)):
            cell = report['tasks'][task]
            success = np.asarray(cell['episode_success'])
            lengths = np.asarray(cell['episode_lengths'])
            inference = cell['inference']
            if cell['episodes'] != episodes or len(success) != episodes or len(lengths) != episodes:
                raise ValueError('Episode count mismatch')
            if not np.isin(success, [0, 1]).all() or not np.all((lengths > 0) & (lengths <= 1000)):
                raise ValueError('Invalid episode outcome/horizon')
            if not np.isclose(success.mean(), cell['success']) or success.sum() != cell['successes']:
                raise ValueError('Reported success differs from episodes')
            if (not inference['representation_unchanged'] or not inference['training_rng_unchanged']
                    or inference['coordinate_updates'] != 5120 or inference['actor_updates'] != 512):
                raise ValueError('Full-inference protocol mismatch')
            row.append(float(success.mean()))
        rows.append(row)
        seeds.append(flags['seed'])
        provenance.append({'run_dir': str(run), 'report_sha256': sha256(report_path)})
    if sorted(seeds) != [0, 1, 2]:
        raise ValueError('Require training seeds0,1,2')
    return dict(**seed_statistics(rows), seed_order=seeds, tasks=[1, 2, 3, 4, 5],
                task_success_matrix=rows, step=step, episodes_per_task=episodes,
                common=common, provenance=provenance)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dirs', nargs=3, required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    result = aggregate(args.run_dirs)
    with Path(args.out).open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result, indent=2))

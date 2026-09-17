"""Evaluation-only fresh-GPI adapter for immutable Walker identity checkpoints."""
# ruff: noqa: I001 -- xla_guard must precede JAX.

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

from utils import xla_guard  # before JAX
import jax
import numpy as np


RNG_MODE = 'fresh_per_call_eval_seed_stream_reset_per_task_v1'
TASKS = ('stand', 'walk', 'run', 'flip')


class FreshDrawsAgent:
    """Supply explicit evaluation keys without modifying the checkpoint agent.

    Each task starts the same evaluation-seed stream (common random numbers).
    The stream advances across steps AND episode batches, independently of training.
    """

    def __init__(self, agent, eval_seed=0, traces=None):
        self.agent = agent
        self.eval_seed = eval_seed
        self.rng = jax.random.PRNGKey(eval_seed)
        self.traces = [] if traces is None else traces
        self.calls = 0
        self.first_keys = []
        self.key_hash = hashlib.sha256()

    @property
    def task_z(self):
        return self.agent.task_z

    def infer_eval_z(self, observations, rewards):
        task = FreshDrawsAgent(self.agent.infer_eval_z(observations, rewards), self.eval_seed, self.traces)
        self.traces.append(task)
        return task

    def sample_actions(self, observations):
        self.rng, key = jax.random.split(self.rng)
        raw_key = np.asarray(key)
        self.key_hash.update(raw_key.tobytes())
        self.calls += 1
        if len(self.first_keys) < 8:
            self.first_keys.append(raw_key.tolist())
        if self.calls % 5000 == 0:
            print(json.dumps({'event': 'gpi_progress', 'task_ordinal': len(self.traces), 'calls': self.calls}), flush=True)
        return self.agent.sample_actions(observations, seed=key)

    def rng_audit(self):
        return [
            {'calls': task.calls, 'first_keys': task.first_keys, 'key_sha256': task.key_hash.hexdigest()}
            for task in self.traces
        ]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def state_sha256(agent):
    """Fingerprint all dynamic checkpoint leaves, including optimizer/target/RNG state."""
    digest = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(agent):
        array = np.asarray(leaf)
        digest.update(str((array.dtype.str, array.shape)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def validate_report(report, manifest, *, seed, episodes):
    """Reject stale, fixed-panel, incomplete, or checkpoint-mismatched results."""
    protocol = manifest['protocol']
    expected = next((item for item in manifest['verified_checkpoint_provenance'] if item['seed'] == seed), None)
    if expected is None:
        raise ValueError(f'Unregistered training seed: {seed}')
    checks = {
        'step': protocol['checkpoint_step'], 'seed': seed, 'run_dir': expected['run_dir'],
        'flags_sha256': expected['flags_sha256'], 'input_provenance': expected,
        'evaluator_sha256': manifest['evaluator_sha256'], 'agent_config': manifest['saved_agent_config'],
        'evaluated_restored_checkpoint': True, 'rng_mode': RNG_MODE,
        **{key: protocol[key] for key in ('eval_seed', 'eval_workers', 'inference_samples')},
    }
    for key, expected_value in checks.items():
        if report.get(key) != expected_value:
            raise ValueError(f'Evaluation provenance/protocol mismatch: {key}')
    if (not report.get('state_sha256_before')
            or report.get('state_sha256_before') != report.get('state_sha256_after')):
        raise ValueError('Evaluation changed checkpoint state')
    if set(report.get('tasks', {})) != set(TASKS):
        raise ValueError('Incomplete task evaluation')
    rows = []
    for name in TASKS:
        task = report['tasks'][name]
        returns = np.asarray(task['episode_returns'], dtype=np.float64)
        if task['episodes'] != episodes or returns.shape != (episodes,) or not np.isfinite(returns).all():
            raise ValueError('Invalid evaluation episode count/returns')
        if task['episode_lengths'] != [1000] * episodes:
            raise ValueError('Unexpected evaluation horizon')
        if not np.isclose(returns.mean(), task['mean_return'], rtol=1e-12, atol=1e-9):
            raise ValueError('Invalid task mean')
        if not np.isfinite(task['task_z']).all():
            raise ValueError('Nonfinite inferred task vector')
        rows.append(float(returns.mean()))
    if not np.isclose(np.mean(rows), report['task_mean'], rtol=1e-12, atol=1e-9):
        raise ValueError('Invalid task-average return')
    traces = report.get('rng_audit', [])
    if len(traces) != len(TASKS):
        raise ValueError('Missing per-task RNG audit')
    expected_calls = math.ceil(episodes / protocol['eval_workers']) * 1000
    for trace in traces:
        keys = [tuple(key) for key in trace['first_keys']]
        if trace['calls'] != expected_calls or len(keys) < 2 or len(set(keys)) != len(keys):
            raise ValueError('Invalid fresh-key call count/trace')
        if not trace.get('key_sha256') or trace['key_sha256'] != traces[0]['key_sha256']:
            raise ValueError('Task RNG streams should be paired and reproducible')
    return rows


def verify_inputs(manifest, seed):
    expected = next(item for item in manifest['verified_checkpoint_provenance'] if item['seed'] == seed)
    run = Path(expected['run_dir'])
    source = Path(manifest['source_directory'])
    step = manifest['protocol']['checkpoint_step']
    paths = {
        'flags_sha256': run / 'flags.json',
        'checkpoint_sha256': run / f'checkpoint_{step}.pkl',
        'saved_source_sha256': run / 'source_manifest.json',
        'dataset_sha256': run / 'dataset_manifest.json',
        'source_file_sha256': source / 'source_manifest.json',
    }
    for key, path in paths.items():
        if sha256(path) != expected[key]:
            raise ValueError(f'Input hash changed: {path}')
    flags = json.loads((run / 'flags.json').read_text())
    if flags['seed'] != seed or flags['agent'] != manifest['saved_agent_config']:
        raise ValueError('Saved agent settings differ from manifest')
    if {package: importlib.metadata.version(package) for package in expected['versions']} != expected['versions']:
        raise ValueError('Runtime package versions changed from training')
    return expected, flags


def evaluate_checkpoint(args, manifest):
    out = Path(args.report)
    if out.exists():
        raise FileExistsError(out)
    if args.episodes not in (manifest['smoke_episodes_per_task'], manifest['protocol']['episodes_per_task']):
        raise ValueError('Episode count must match declared smoke or production protocol')
    if args.episodes == manifest['protocol']['episodes_per_task']:
        if args.smoke_report is None:
            raise ValueError('Production requires a completed corrected-evaluation smoke')
        smoke = json.loads(Path(args.smoke_report).read_text())
        validate_report(smoke, manifest, seed=0, episodes=manifest['smoke_episodes_per_task'])
        receipt = json.loads(Path(args.smoke_report).with_suffix('.validated.json').read_text())
        if receipt['report_sha256'] != sha256(args.smoke_report):
            raise ValueError('Smoke validation receipt hash mismatch')
    expected, flags = verify_inputs(manifest, args.seed)
    source = Path(manifest['source_directory']).resolve()
    # Run this script with PYTHONPATH pointing to the immutable training snapshot.
    # Refuse accidental imports from today's checkout instead of changing its guards.
    if not Path(xla_guard.__file__).resolve().is_relative_to(source):
        raise ValueError('PYTHONPATH must select the original training snapshot')
    sys.path.insert(0, str(source))
    spec = importlib.util.spec_from_file_location('walker_identity_frozen_eval_runner', source / 'tools/walker_psm/run.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    original_evaluate = runner.evaluate
    protocol = manifest['protocol']
    started = time.monotonic()

    def corrected_evaluate(agent, dataset, **kwargs):
        before = state_sha256(agent)
        wrapper = FreshDrawsAgent(agent, eval_seed=kwargs['eval_seed'])
        result = original_evaluate(wrapper, dataset, **kwargs)
        after = state_sha256(agent)
        if before != after:
            raise ValueError('Evaluation mutated checkpoint state')
        return dict(
            **result, rng_mode=RNG_MODE, rng_audit=wrapper.rng_audit(),
            state_sha256_before=before, state_sha256_after=after,
            evaluator_sha256=manifest['evaluator_sha256'], input_provenance=expected,
            agent_config=flags['agent'], evaluated_restored_checkpoint=True,
            eval_workers=kwargs['workers'], eval_seconds=time.monotonic() - started,
            slurm_job_id=os.environ.get('SLURM_JOB_ID'),
        )

    # Only replace the evaluation boundary in memory; all snapshot files, loading,
    # restoration, reward inference, environments, and action scoring remain original.
    runner.evaluate = corrected_evaluate
    original_argv = sys.argv
    sys.argv = [
        str(spec.origin), 'eval', '--data-dir', flags['run']['data_dir'],
        '--psi-form', flags['agent']['psi_form'], '--run-dir', expected['run_dir'],
        '--seed', str(args.seed), '--max-episodes', str(flags['run']['max_episodes']),
        '--checkpoint-step', str(protocol['checkpoint_step']), '--report', str(out),
        '--eval-episodes', str(args.episodes), '--inference-samples', str(protocol['inference_samples']),
        '--eval-seed', str(protocol['eval_seed']), '--eval-workers', str(protocol['eval_workers']),
    ]
    startup = {
        'manifest_sha256': sha256(args.manifest), 'argv': sys.argv, 'input_provenance': expected,
        'agent_config': flags['agent'], 'protocol': protocol, 'episodes': args.episodes,
        'evaluator_sha256': manifest['evaluator_sha256'], 'rng_mode': RNG_MODE,
    }
    write_json(out.with_suffix('.started.json'), startup)
    print(json.dumps({'event': 'verified_startup', **startup}), flush=True)
    try:
        runner.main()
    finally:
        sys.argv = original_argv
    verify_inputs(manifest, args.seed)
    report = json.loads(out.read_text())
    validate_report(report, manifest, seed=args.seed, episodes=args.episodes)
    write_json(out.with_suffix('.validated.json'), {
        'report_sha256': sha256(out), 'seed': args.seed, 'episodes_per_task': args.episodes,
        'all_checks_passed': True, 'manifest_sha256': sha256(args.manifest),
    })
    print(json.dumps({'event': 'evaluation_validated', 'report': str(out), 'task_mean': report['task_mean']}), flush=True)


def aggregate_reports(manifest, report_paths):
    if len(report_paths) != 3 or len({str(Path(p).resolve()) for p in report_paths}) != 3:
        raise ValueError('Require three distinct corrected evaluation reports')
    matrix, seeds, provenance = [], [], []
    common_rng = None
    for path in report_paths:
        report = json.loads(Path(path).read_text())
        seed = report['seed']
        matrix.append(validate_report(report, manifest, seed=seed, episodes=manifest['protocol']['episodes_per_task']))
        signature = (report['inference_index_sha256'], report['rng_audit'][0]['key_sha256'])
        if common_rng is not None and signature != common_rng:
            raise ValueError('Evaluation RNG/inference rows differ across seeds')
        common_rng = signature
        receipt = json.loads(Path(path).with_suffix('.validated.json').read_text())
        if receipt['report_sha256'] != sha256(path) or receipt.get('all_checks_passed') is not True:
            raise ValueError('Evaluation receipt mismatch')
        seeds.append(seed)
        provenance.append({'seed': seed, 'report': str(Path(path).resolve()), 'report_sha256': sha256(path)})
    if sorted(seeds) != [0, 1, 2]:
        raise ValueError('Require distinct training seeds 0, 1, 2')
    means = np.asarray(matrix, dtype=np.float64).mean(axis=1)
    return {
        'seed_order': seeds, 'tasks': list(TASKS), 'task_return_matrix': matrix, 'seed_task_means': means.tolist(),
        'mean': float(means.mean()), 'ci95_halfwidth': float(4.302652729911275 * means.std(ddof=1) / np.sqrt(3)),
        'interval': 'Student-t across training seeds, df=2; tasks averaged within each seed',
        'step': manifest['protocol']['checkpoint_step'],
        'episodes_per_task': manifest['protocol']['episodes_per_task'],
        'protocol': manifest['protocol'], 'rng_mode': RNG_MODE, 'evaluator_sha256': manifest['evaluator_sha256'],
        'agent_config': manifest['saved_agent_config'], 'provenance': provenance,
        'method': 'Affine identity-decoder PSM, no flow/actor; corrected fresh-GPI evaluation; no training at evaluation',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['evaluate', 'aggregate'])
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--seed', type=int, choices=[0, 1, 2])
    parser.add_argument('--episodes', type=int, default=500)
    parser.add_argument('--report', required=True)
    parser.add_argument('--smoke-report')
    parser.add_argument('--reports', nargs=3)
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    if sha256(__file__) != manifest['evaluator_sha256']:
        raise ValueError('Evaluator source hash mismatch')
    if args.mode == 'evaluate':
        if args.seed is None:
            parser.error('evaluate requires --seed')
        evaluate_checkpoint(args, manifest)
    else:
        if args.reports is None:
            parser.error('aggregate requires --reports')
        result = aggregate_reports(manifest, args.reports)
        write_json(args.report, result)
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

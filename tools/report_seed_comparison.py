"""Validate and summarize a three-seed, multi-task eval500 manifest."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

T_CRIT_95_DF2 = 4.302652729911275
ALLOWED_OVERRIDES = {'flow_ckpt_path', 'flow_ckpt_epoch', 'preimage_path', 'use_point_preimage'}
RELOCATION_KEYS = {'flow_ckpt_path', 'preimage_path'}
NON_CONFIG_KEYS = {'seed', 'env_name', 'save_dir', 'restore_path', 'restore_epoch', 'run_group'}


def _canonical(value: Any, key: str = '') -> Any:
    if isinstance(value, dict):
        return {k: _canonical(v, k) for k, v in sorted(value.items())
                if k not in NON_CONFIG_KEYS and k not in RELOCATION_KEYS}
    if isinstance(value, list):
        return [_canonical(v, key) for v in value]
    return value


def _flags_for(report: dict[str, Any], report_path: Path) -> tuple[dict[str, Any], Path]:
    source = report.get('agent_config_source', {})
    flags_path = source.get('flags_json')
    if flags_path:
        path = Path(flags_path)
    else:
        restore = report.get('restore_path')
        path = Path(restore) / 'flags.json' if restore else report_path.parent / 'flags.json'
    try:
        flags = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f'cannot read flags.json for {report_path}: {exc}') from exc
    restore = report.get('restore_path')
    if not restore or path.parent.resolve() != Path(restore).resolve():
        raise ValueError(f'flags.json is outside report restore_path for {report_path}')
    return flags, path


def _has_real_reward_dsrl(value: Any) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {'dsrl_na', 'dsrlna'} and isinstance(child, dict) and child.get('enabled'):
                source = child.get('reward_source', child.get('reward_ref')) or 'real'
                if source == 'real' or str(source).startswith('phi_readout'):
                    return True
            if _has_real_reward_dsrl(child):
                return True
    elif isinstance(value, list):
        return any(_has_real_reward_dsrl(x) for x in value)
    return False


def build_report(manifest: dict[str, Any]) -> dict[str, Any]:
    expected = manifest.get('expected_seeds')
    if expected != [0, 1, 2] or len(set(expected)) != 3:
        raise ValueError('expected_seeds must be exactly [0, 1, 2]')
    tasks = {str(k): v for k, v in manifest.get('tasks', {}).items()}
    if set(tasks) != {'1', '2', '3', '4', '5'}:
        raise ValueError("tasks must contain exactly IDs '1'..'5'")
    env_prefix = None
    for task, env in tasks.items():
        match = re.fullmatch(r'(.+)-singletask-task([1-5])-v0', str(env))
        if not match or match.group(2) != task:
            raise ValueError(f'task {task} must name its matching singletask environment')
        if env_prefix is None:
            env_prefix = match.group(1)
        elif match.group(1) != env_prefix:
            raise ValueError('all tasks must share one environment domain prefix')
    target_epoch = manifest.get('restore_epoch')
    target_episodes = manifest.get('num_episodes')
    if target_epoch != 500000 or target_episodes != 500:
        raise ValueError('manifest must request restore_epoch=500000 and num_episodes=500')

    entries = manifest.get('reports', [])
    matrix: dict[tuple[int, str], dict[str, Any]] = {}
    paths: set[str] = set()
    configs: list[Any] = []
    dataset_meta: list[tuple[Any, Any]] = []
    run_by_seed: dict[int, str] = {}
    zero_shot = True
    task_scores: dict[str, list[float]] = {task: [] for task in tasks}
    for entry in entries:
        seed, task = entry.get('seed'), str(entry.get('task'))
        if seed not in expected or task not in tasks:
            raise ValueError(f'unknown seed/task in manifest entry: {seed}/{task}')
        key = (seed, task)
        if key in matrix:
            raise ValueError(f'duplicate seed/task matrix entry: {seed}/{task}')
        report_path = Path(entry['path'])
        path_key = str(report_path.resolve())
        if path_key in paths:
            raise ValueError('duplicate report path; matrix entries must be unique and each seed/task needs its own report')
        paths.add(path_key)
        try:
            report = json.loads(report_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f'cannot read report {report_path}: {exc}') from exc
        if report.get('env') != tasks[task]:
            raise ValueError(f'env mismatch for seed/task {seed}/{task}')
        if report.get('num_episodes') != target_episodes:
            raise ValueError(f'num_episodes mismatch for {report_path}')
        if report.get('restore_epoch') != target_epoch:
            raise ValueError(f'restore_epoch mismatch for {report_path}')
        count, success = report.get('num_success'), report.get('success')
        if not isinstance(count, int) or not 0 <= count <= target_episodes:
            raise ValueError(f'invalid num_success for {report_path}')
        if not isinstance(success, (int, float)) or not math.isclose(success, count / target_episodes):
            raise ValueError(f'success count disagreement for {report_path}')
        flags, flags_path = _flags_for(report, report_path)
        if flags.get('seed') != seed:
            raise ValueError(f'training seed mismatch for {report_path}: flags seed is {flags.get("seed")}')
        if flags.get('offline_steps') != target_epoch or flags.get('restore_path') is not None:
            raise ValueError(f'continuation/non-final flags in {flags_path}')
        if flags.get('online_steps', 0) > 0:
            raise ValueError(f'online continuation is not allowed in {flags_path}')
        overrides = set(report.get('agent_config_source', {}).get('cli_overrides', []))
        if not overrides <= ALLOWED_OVERRIDES:
            raise ValueError(f'policy-changing CLI override in {report_path}: {sorted(overrides - ALLOWED_OVERRIDES)}')
        run = str(Path(report.get('restore_path', '')).resolve())
        if not run or run == str(Path.cwd()):
            raise ValueError(f'missing restore_path in {report_path}')
        if seed in run_by_seed and run_by_seed[seed] != run:
            raise ValueError(f'seed {seed} does not use one run across tasks')
        run_by_seed[seed] = run
        configs.append(_canonical(flags.get('agent', {})))
        dataset_meta.append((flags.get('dataset_fraction', 1.0), flags.get('dataset_fraction_seed', 0)))
        zero_shot &= not _has_real_reward_dsrl(flags)
        matrix[key] = report
        report['_manifest_path'] = str(report_path)
        task_scores[task].append(count / target_episodes)

    expected_keys = {(s, t) for s in expected for t in tasks}
    if set(matrix) != expected_keys:
        raise ValueError('reports must contain a complete unique seed/task matrix')
    if len(set(run_by_seed.values())) != 3:
        raise ValueError('training seeds must use distinct runs')
    if configs and any(config != configs[0] for config in configs[1:]):
        raise ValueError('relevant agent config differs across reports')
    if dataset_meta and any(meta != dataset_meta[0] for meta in dataset_meta[1:]):
        raise ValueError('dataset_fraction or dataset_fraction_seed differs across reports')

    seed_scores = [sum(matrix[s, task]['num_success'] / target_episodes for task in tasks) / len(tasks)
                   for s in expected]
    mean = sum(seed_scores) / 3
    variance = sum((x - mean) ** 2 for x in seed_scores) / 2
    ci95 = T_CRIT_95_DF2 * math.sqrt(variance / 3)
    return {
        'label': manifest.get('label', ''), 'restore_epoch': target_epoch,
        'num_episodes': target_episodes, 'n_seeds': 3, 'n_tasks': len(tasks),
        'training_seeds': expected, 'seed_scores': seed_scores,
        'task_scores': {task: [matrix[s, task]['success'] for s in expected] for task in tasks},
        'mean': mean, 'ci95': ci95, 'mean_ci95': [mean - ci95, mean + ci95],
        'zero_shot': zero_shot,
        'paths': {f'{s}:{task}': matrix[s, task]['_manifest_path'] for s in expected for task in tasks},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--report-out', required=True)
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    result = build_report(manifest)
    Path(args.report_out).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Train matched control/frozen-phi continuations from one Stage-C checkpoint.

The two agents live in one process.  Each update receives the same dataset batch and
starts from the same restored agent RNG; their only configuration difference is
``train_phi``.  Checkpoint names use the absolute training update count, so the default
50k + 450k continuation ends at ``params_500000.pkl``.
"""

import argparse
import copy
import glob
import hashlib
import json
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import utils.xla_guard  # noqa: F401  -- MUST precede jax

DRIVER_NAME = 'train_freeze_phi_pair'
SCHEMA_VERSION = 1
EXPECTED_ENV = 'antmaze-medium-navigate-singletask-v0'


@dataclass(frozen=True)
class ContinuationBudget:
    source_updates: int
    additional_steps: int
    total_updates: int
    save_steps: tuple[int, ...]


def build_budget(source_updates, additional_steps, save_interval):
    """Return the absolute update/checkpoint schedule for a continuation."""
    source_updates = int(source_updates)
    additional_steps = int(additional_steps)
    save_interval = int(save_interval)
    if source_updates < 1:
        raise ValueError(f'source_updates must be >= 1, got {source_updates}')
    if additional_steps < 1:
        raise ValueError(f'additional_steps must be >= 1, got {additional_steps}')
    if save_interval < 1:
        raise ValueError(f'save_interval must be >= 1, got {save_interval}')
    total = source_updates + additional_steps
    scheduled = [source_updates]
    scheduled.extend(source_updates + offset
                     for offset in range(save_interval, additional_steps + 1, save_interval))
    if scheduled[-1] != total:
        scheduled.append(total)
    return ContinuationBudget(source_updates, additional_steps, total, tuple(scheduled))


def _nested(config, *keys, default=None):
    node = config
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def validate_source_flags(flags):
    """Reject source runs that do not describe the predeclared AntMaze pilot."""
    if int(flags.get('seed', -1)) != 0:
        raise ValueError('freeze-phi pilot requires the predeclared source seed=0')
    if flags.get('env_name') != EXPECTED_ENV:
        raise ValueError(f'freeze-phi pilot requires AntMaze source {EXPECTED_ENV!r}')
    if int(flags.get('online_steps', 0)) != 0:
        raise ValueError('source must be offline-only (online_steps=0)')
    if bool(flags.get('balanced_sampling', False)):
        raise ValueError('source balanced_sampling must be disabled')
    agent = flags.get('agent')
    if not isinstance(agent, dict) or agent.get('agent_name') != 'psmflow':
        raise ValueError('source must be a psmflow run')
    if agent.get('measure_action_input') != 'action':
        raise ValueError('source must be the action-conditioned measure arm')
    if agent.get('psi_form') != 'affine':
        raise ValueError('source must use affine psi')
    if agent.get('policy_index') != 'latent':
        raise ValueError('source must use the latent policy index')
    if agent.get('acting') != 'gpi':
        raise ValueError('source must use GPI acting')
    if agent.get('index_agg') != 'max' or agent.get('gpi_select', 'argmax') != 'argmax':
        raise ValueError('source must use the original max/argmax GPI rule')
    if float(agent.get('discount', 0.0)) != 0.99:
        raise ValueError('source discount must be 0.99')
    if bool(agent.get('train_actor')):
        raise ValueError('source must be actor-free (train_actor=false)')
    if bool(_nested(agent, 'dsrl_na', 'enabled', default=False)):
        raise ValueError('source must not enable DSRL')
    if bool(_nested(agent, 'action_critic', 'enabled', default=False)):
        raise ValueError('source must not enable the action critic')
    if agent.get('train_phi', True) is not True:
        raise ValueError('source checkpoint must have trained phi before the fork')
    if not agent.get('flow_ckpt_path') or not agent.get('preimage_path'):
        raise ValueError('source flags must record flow_ckpt_path and preimage_path')
    if not bool(agent.get('use_point_preimage')):
        raise ValueError('source must use the point preimage arm')
    if int(agent.get('measure_u_samples', 1)) != 1:
        raise ValueError('source must use exactly one measure input per transition')
    return agent


def build_pair_configs(source_agent_config):
    """Deep-copy the source config into two arms differing only in ``train_phi``."""
    control = copy.deepcopy(dict(source_agent_config))
    frozen = copy.deepcopy(dict(source_agent_config))
    control.pop('train_phi', None)
    frozen.pop('train_phi', None)
    control['train_phi'] = True
    frozen['train_phi'] = False
    return control, frozen


def _realpath_matches(left, right):
    left_matches = {os.path.realpath(path) for path in glob.glob(str(left))}
    right_matches = {os.path.realpath(path) for path in glob.glob(str(right))}
    return bool(left_matches & right_matches)


def validate_preimage_pairing(
    meta,
    *,
    env_name,
    flow_ckpt_path,
    flow_ckpt_epoch,
    dataset_fraction,
    dataset_fraction_seed,
    augmented_observations,
    dataset_observations,
):
    """Apply the strict metadata and transition-order guards used by ``main.py``."""
    if not isinstance(meta, dict):
        raise TypeError('preimage .meta.json sidecar is required for exact provenance')
    if meta.get('sampled_batch'):
        raise ValueError('preimage is a sampled tuning batch, not a training artifact')
    if meta.get('env_name') != env_name:
        raise ValueError(f'preimage env {meta.get("env_name")!r} does not match {env_name!r}')
    if not meta.get('restore_path') or not _realpath_matches(meta['restore_path'], flow_ckpt_path):
        raise ValueError('preimage restore_path does not resolve to the source flow checkpoint')
    if int(meta.get('restore_epoch') or 0) != int(flow_ckpt_epoch):
        raise ValueError('preimage restore_epoch does not match flow_ckpt_epoch')
    if 'dataset_fraction' in meta:
        if float(meta['dataset_fraction']) != float(dataset_fraction):
            raise ValueError('preimage dataset_fraction does not match the source run')
        if int(meta['dataset_fraction_seed']) != int(dataset_fraction_seed):
            raise ValueError('preimage dataset_fraction_seed does not match the source run')
    elif float(dataset_fraction) != 1.0:
        raise ValueError(
            'preimage metadata predates dataset_fraction but the source run uses a subset')

    augmented = np.asarray(augmented_observations)
    dataset = np.asarray(dataset_observations)
    if augmented.shape[0] != dataset.shape[0]:
        raise ValueError('preimage row count does not match the environment dataset')
    n_rows = augmented.shape[0]
    n_check = min(1000, n_rows)
    for lo, hi, label in ((0, n_check, 'first'), (n_rows - n_check, n_rows, 'last')):
        if not np.array_equal(augmented[lo:hi], dataset[lo:hi]):
            raise ValueError(f'preimage observations differ in the {label} {n_check} dataset rows')


def create_output_tree(output_root):
    """Atomically claim a new output root and create the two eval-compatible run dirs."""
    root = Path(output_root)
    try:
        root.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise FileExistsError(f'paired continuation requires a new output root: {root}') from exc
    paths = {'control': root / 'control', 'frozen': root / 'frozen'}
    for path in paths.values():
        path.mkdir()
    return paths


@contextmanager
def preserve_numpy_rng(seed):
    """Run evaluation on a separate NumPy stream, restoring training state afterward."""
    state = np.random.get_state()
    try:
        np.random.seed(int(seed))
        yield
    finally:
        np.random.set_state(state)


def _trees_identical(left, right):
    import jax

    left_leaves = jax.tree_util.tree_leaves(left)
    right_leaves = jax.tree_util.tree_leaves(right)
    if len(left_leaves) != len(right_leaves):
        return False
    return all(np.array_equal(np.asarray(a), np.asarray(b))
               for a, b in zip(left_leaves, right_leaves))


def tree_digest(tree):
    """Stable SHA-256 over pytree leaves, including shape and dtype."""
    import jax

    digest = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(tree):
        array = np.asarray(leaf)
        digest.update(str(array.dtype).encode())
        digest.update(repr(array.shape).encode())
        digest.update(array.tobytes(order='C'))
    return digest.hexdigest()


def _tree_finite(tree):
    import jax

    return all(np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree_util.tree_leaves(tree))


def update_pair(control, frozen, batch):
    """Advance both arms on the same batch while enforcing their common RNG stream."""
    if not _trees_identical(control.rng, frozen.rng):
        raise AssertionError('paired agent RNG streams diverged before update')
    control, control_info = control.update(batch)
    frozen, frozen_info = frozen.update(batch)
    if not _trees_identical(control.rng, frozen.rng):
        raise AssertionError('paired agent RNG streams diverged after update')
    return control, frozen, {'control': control_info, 'frozen': frozen_info}


def _json_dump(path, payload):
    with open(path, 'w') as file:
        json.dump(payload, file, indent=2, default=str)


def _scalar_metrics(info, additional_step, elapsed):
    metrics = {'additional_step': int(additional_step), 'time/total_seconds': round(elapsed, 3)}
    for key, value in info.items():
        array = np.asarray(value)
        if array.size == 1:
            metrics[f'training/{key}'] = float(array.reshape(()))
    return metrics


def _should_run(additional_step, interval, final_step):
    return additional_step == final_step or (interval > 0 and additional_step % interval == 0)


def _load_source(source_run, source_step):
    if int(source_step) != 50_000:
        raise ValueError(f'freeze-phi pilot source step must be 50000, got {source_step}')
    source_run = Path(source_run).expanduser().resolve()
    if not source_run.is_dir():
        raise FileNotFoundError(f'source run directory does not exist: {source_run}')
    flags_path = source_run / 'flags.json'
    checkpoint_path = source_run / f'params_{source_step}.pkl'
    if not flags_path.is_file():
        raise FileNotFoundError(f'source flags do not exist: {flags_path}')
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f'source checkpoint does not exist: {checkpoint_path}')
    with open(flags_path) as file:
        flags = json.load(file)
    validate_source_flags(flags)
    return source_run, checkpoint_path, flags


def _load_dataset_and_env(source_flags):
    from envs.env_utils import make_env_and_datasets
    from utils.datasets import Dataset
    from utils.flow_inversion import load_augmented_dataset, repair_invalid_preimages

    agent_config = source_flags['agent']
    _, eval_env, env_dataset, _ = make_env_and_datasets(
        source_flags['env_name'],
        frame_stack=source_flags.get('frame_stack'),
        dataset_fraction=source_flags.get('dataset_fraction', 1.0),
        dataset_fraction_seed=source_flags.get('dataset_fraction_seed', 0),
    )
    preimage_path = Path(agent_config['preimage_path']).expanduser()
    if not preimage_path.is_file():
        raise FileNotFoundError(f'preimage artifact does not exist: {preimage_path}')
    meta_path = Path(str(preimage_path) + '.meta.json')
    if not meta_path.is_file():
        raise FileNotFoundError(f'preimage provenance sidecar does not exist: {meta_path}')
    with open(meta_path) as file:
        meta = json.load(file)
    augmented = load_augmented_dataset(preimage_path)
    validate_preimage_pairing(
        meta,
        env_name=source_flags['env_name'],
        flow_ckpt_path=agent_config['flow_ckpt_path'],
        flow_ckpt_epoch=agent_config['flow_ckpt_epoch'],
        dataset_fraction=source_flags.get('dataset_fraction', 1.0),
        dataset_fraction_seed=source_flags.get('dataset_fraction_seed', 0),
        augmented_observations=augmented['observations'],
        dataset_observations=env_dataset['observations'],
    )
    augmented, _ = repair_invalid_preimages(augmented)
    dataset = Dataset.create(**augmented)
    dataset.p_aug = source_flags.get('p_aug')
    dataset.frame_stack = source_flags.get('frame_stack')
    dataset.return_preimage_noise = True
    dataset.preimage_point_mode = True
    return eval_env, dataset


def _fork_agents(source_agent):
    from flax.core import FrozenDict, unfreeze

    synchronized = source_agent.replace(target_phi=source_agent.phi.params)
    if not _trees_identical(synchronized.phi.params, synchronized.target_phi):
        raise AssertionError('failed to synchronize target_phi from online phi')
    control_config, frozen_config = build_pair_configs(unfreeze(synchronized.config))
    control = synchronized.replace(config=FrozenDict(control_config))
    frozen = synchronized.replace(config=FrozenDict(frozen_config))
    if not _trees_identical(control.rng, frozen.rng):
        raise AssertionError('restored RNG differs across forked arms')
    if not _trees_identical(control, frozen):
        raise AssertionError('forked arms differ in restored trainable or target state')
    return control, frozen


def _evaluate_pair(agents, dataset, eval_env, *, episodes, relabel_size, reward_shift, seed):
    from utils.evaluation import evaluate

    results = {}
    with preserve_numpy_rng(seed):
        z_batch = dataset.sample(min(dataset.size, int(relabel_size)))
        rewards = z_batch['rewards'] + float(reward_shift)
        for arm, agent in agents.items():
            eval_agent = agent.infer_eval_z(z_batch['next_observations'], rewards)
            info, _, _ = evaluate(
                agent=eval_agent,
                env=eval_env,
                config=agent.config,
                num_eval_episodes=int(episodes),
                num_video_episodes=0,
                seed=int(seed),
            )
            results[arm] = {key: float(np.asarray(value)) for key, value in info.items()}
    return results


def _arm_flags(
    source_flags,
    agent_config,
    arm,
    source_run,
    source_step,
    budget,
    output_root,
    *,
    save_interval,
    log_interval,
    eval_interval,
    eval_episodes,
    eval_relabel_size,
):
    flags = copy.deepcopy(source_flags)
    flags.update({
        'restore_path': str(source_run),
        'restore_epoch': int(source_step),
        'offline_steps': int(budget.additional_steps),
        'online_steps': 0,
        'save_dir': str(output_root),
        'run_group': f'{output_root.name}_{arm}',
        'save_interval': int(save_interval),
        'log_interval': int(log_interval),
        'eval_interval': int(eval_interval),
        'eval_episodes': int(eval_episodes),
        'eval_relabel_size': int(eval_relabel_size),
        'agent': copy.deepcopy(agent_config),
        'driver': {'name': DRIVER_NAME, 'schema_version': SCHEMA_VERSION},
        'continuation': {
            'paired': True,
            'arm': arm,
            'source_run': str(source_run),
            'source_checkpoint': str(source_run / f'params_{source_step}.pkl'),
            'source_epoch': int(source_step),
            'source_updates': int(budget.source_updates),
            'additional_steps': int(budget.additional_steps),
            'total_updates': int(budget.total_updates),
            'absolute_step_checkpoints': list(budget.save_steps),
            'target_phi_synchronized_before_fork': True,
            'shared_dataset_batch_per_update': True,
            'shared_restored_agent_rng': True,
        },
    })
    return flags


def run(args):
    import ml_collections
    from flax.core import unfreeze

    from agents import agents as agent_registry
    from main import _lists_to_tuples
    from utils.flax_utils import restore_agent, save_agent
    from utils.log_utils import CsvLogger

    source_run, source_checkpoint, source_flags = _load_source(args.source_run, args.source_step)
    if int(args.seed) != 0 or int(args.seed) != int(source_flags['seed']):
        raise ValueError('--seed must be 0 and match the predeclared source training seed')
    budget = build_budget(args.source_step, args.additional_steps, args.save_interval)
    output_root = Path(args.output_dir).expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f'paired continuation requires a new output root: {output_root}')
    if output_root == source_run or source_run in output_root.parents:
        raise ValueError('output directory must not be inside the source run')

    print(json.dumps({
        'driver': DRIVER_NAME,
        'schema_version': SCHEMA_VERSION,
        'source_run': str(source_run),
        'source_checkpoint': str(source_checkpoint),
        'source_updates': budget.source_updates,
        'additional_steps': budget.additional_steps,
        'total_updates': budget.total_updates,
        'save_steps': budget.save_steps,
        'eval_interval': int(args.eval_interval),
        'eval_episodes': int(args.eval_episodes),
        'seed': int(args.seed),
    }, indent=2))

    eval_env, dataset = _load_dataset_and_env(source_flags)
    np.random.seed(int(args.seed))
    example_batch = dataset.sample(1)
    source_config = copy.deepcopy(source_flags['agent'])
    source_config['train_phi'] = True
    config = ml_collections.ConfigDict(_lists_to_tuples(source_config))
    agent_class = agent_registry[config['agent_name']]
    source_agent = agent_class.create(
        int(args.seed), example_batch['observations'], example_batch['actions'], config)
    source_agent = restore_agent(source_agent, str(source_run), int(args.source_step))
    completed = int(np.asarray(source_agent.psi.step)) - 1
    if completed != budget.source_updates:
        raise ValueError(
            f'source psi.step proves {completed} completed updates, requested source step '
            f'is {budget.source_updates}')

    control, frozen = _fork_agents(source_agent)
    initial = {
        'source_psi_step': int(np.asarray(source_agent.psi.step)),
        'source_phi_step': int(np.asarray(source_agent.phi.step)),
        'phi_state_sha256': tree_digest(frozen.phi),
        'phi_params_sha256': tree_digest(frozen.phi.params),
        'psi_params_sha256': tree_digest(frozen.psi.params),
        'flow_vf_sha256': tree_digest(frozen.flow_vf),
        'flow_onestep_sha256': tree_digest(frozen.flow_onestep),
        'target_phi_before_sync_sha256': tree_digest(source_agent.target_phi),
        'target_phi_after_sync_sha256': tree_digest(frozen.target_phi),
    }

    paths = create_output_tree(output_root)
    runtime_configs = {
        'control': unfreeze(control.config),
        'frozen': unfreeze(frozen.config),
    }
    log_interval = int(args.log_interval or source_flags.get('log_interval', 5000))
    relabel_size = int(args.eval_relabel_size or source_flags.get('eval_relabel_size', 10000))
    reward_shift = float(source_flags.get('eval_reward_shift', 1.0))
    manifest = {
        'driver': {'name': DRIVER_NAME, 'schema_version': SCHEMA_VERSION},
        'status': 'running',
        'source_run': str(source_run),
        'source_checkpoint': str(source_checkpoint),
        'source_epoch': int(args.source_step),
        'source_updates': budget.source_updates,
        'additional_steps': budget.additional_steps,
        'total_updates': budget.total_updates,
        'save_interval': int(args.save_interval),
        'log_interval': log_interval,
        'eval_interval': int(args.eval_interval),
        'eval_episodes': int(args.eval_episodes),
        'eval_relabel_size': relabel_size,
        'seed': int(args.seed),
        'arms': {arm: str(path) for arm, path in paths.items()},
        'initial_hashes': initial,
    }
    _json_dump(output_root / 'pair_manifest.json', manifest)
    for arm, path in paths.items():
        _json_dump(path / 'flags.json', _arm_flags(
            source_flags, runtime_configs[arm], arm, source_run, args.source_step,
            budget, output_root,
            save_interval=args.save_interval,
            log_interval=log_interval,
            eval_interval=args.eval_interval,
            eval_episodes=args.eval_episodes,
            eval_relabel_size=relabel_size,
        ))

    agents = {'control': control, 'frozen': frozen}
    for arm, agent in agents.items():
        save_agent(agent, str(paths[arm]), budget.source_updates)

    train_logs = {arm: CsvLogger(str(path / 'train.csv')) for arm, path in paths.items()}
    eval_logs = {arm: CsvLogger(str(path / 'eval.csv')) for arm, path in paths.items()}
    start_time = time.time()

    # Reset after agent construction and the example batch: this is the recorded training
    # batch stream, and evaluation restores it exactly around every rollout.
    np.random.seed(int(args.seed))
    try:
        for additional_step in range(1, budget.additional_steps + 1):
            absolute_step = budget.source_updates + additional_step
            batch = dataset.sample(int(control.config['batch_size']))
            control, frozen, infos = update_pair(control, frozen, batch)
            agents = {'control': control, 'frozen': frozen}

            if _should_run(additional_step, log_interval, budget.additional_steps):
                elapsed = time.time() - start_time
                for arm in agents:
                    train_logs[arm].log(
                        _scalar_metrics(infos[arm], additional_step, elapsed),
                        step=absolute_step,
                    )

            if int(args.eval_interval) > 0 and _should_run(
                    additional_step, int(args.eval_interval), budget.additional_steps):
                eval_results = _evaluate_pair(
                    agents,
                    dataset,
                    eval_env,
                    episodes=args.eval_episodes,
                    relabel_size=relabel_size,
                    reward_shift=reward_shift,
                    seed=args.seed,
                )
                for arm, metrics in eval_results.items():
                    eval_logs[arm].log(
                        {f'evaluation/{key}': value for key, value in metrics.items()},
                        step=absolute_step,
                    )

            if absolute_step in budget.save_steps:
                if tree_digest(frozen.phi) != initial['phi_state_sha256']:
                    raise AssertionError(f'frozen phi or its optimizer changed by step {absolute_step}')
                if not _trees_identical(frozen.target_phi, frozen.phi.params):
                    raise AssertionError(f'frozen target_phi differs from phi at step {absolute_step}')
                expected_control_phi_step = initial['source_phi_step'] + additional_step
                if int(np.asarray(control.phi.step)) != expected_control_phi_step:
                    raise AssertionError(
                        f'control phi.step={control.phi.step}, expected {expected_control_phi_step}')
                if int(np.asarray(frozen.phi.step)) != initial['source_phi_step']:
                    raise AssertionError('frozen phi.step advanced')
                for arm, agent in agents.items():
                    expected_step = absolute_step + 1
                    if int(np.asarray(agent.psi.step)) != expected_step:
                        raise AssertionError(
                            f'{arm} psi.step={agent.psi.step}, expected {expected_step}')
                    if not _tree_finite(agent.psi):
                        raise FloatingPointError(f'{arm} psi state is non-finite at step {absolute_step}')
                    save_agent(agent, str(paths[arm]), absolute_step)
    finally:
        for logger in (*train_logs.values(), *eval_logs.values()):
            logger.close()

    final_hashes = {
        'control_phi_state_sha256': tree_digest(control.phi),
        'frozen_phi_state_sha256': tree_digest(frozen.phi),
        'frozen_phi_params_sha256': tree_digest(frozen.phi.params),
        'frozen_target_phi_sha256': tree_digest(frozen.target_phi),
        'control_psi_params_sha256': tree_digest(control.psi.params),
        'frozen_psi_params_sha256': tree_digest(frozen.psi.params),
        'control_flow_vf_sha256': tree_digest(control.flow_vf),
        'frozen_flow_vf_sha256': tree_digest(frozen.flow_vf),
        'control_flow_onestep_sha256': tree_digest(control.flow_onestep),
        'frozen_flow_onestep_sha256': tree_digest(frozen.flow_onestep),
    }
    if final_hashes['frozen_phi_state_sha256'] != initial['phi_state_sha256']:
        raise AssertionError('frozen phi TrainState changed over the continuation')
    if final_hashes['control_phi_state_sha256'] == initial['phi_state_sha256']:
        raise AssertionError('control phi TrainState did not change')
    if final_hashes['frozen_phi_params_sha256'] != final_hashes['frozen_target_phi_sha256']:
        raise AssertionError('frozen target_phi does not equal online phi at completion')
    for arm in ('control', 'frozen'):
        if final_hashes[f'{arm}_flow_vf_sha256'] != initial['flow_vf_sha256']:
            raise AssertionError(f'{arm} flow_vf changed over the continuation')
        if final_hashes[f'{arm}_flow_onestep_sha256'] != initial['flow_onestep_sha256']:
            raise AssertionError(f'{arm} flow_onestep changed over the continuation')
    if final_hashes['frozen_psi_params_sha256'] == initial['psi_params_sha256']:
        raise AssertionError('frozen-arm psi parameters did not change')

    completion = {
        **manifest,
        'status': 'complete',
        'final_absolute_step': budget.total_updates,
        'control_psi_step': int(np.asarray(control.psi.step)),
        'frozen_psi_step': int(np.asarray(frozen.psi.step)),
        'elapsed_seconds': round(time.time() - start_time, 3),
        'final_hashes': final_hashes,
        'checks': {
            'frozen_phi_and_optimizer_exact': True,
            'control_phi_advanced': True,
            'frozen_target_phi_equals_phi': True,
            'psi_steps_advanced': True,
            'psi_states_finite': True,
            'flow_unchanged': True,
            'paired_rng_streams_equal': True,
        },
        'final_evaluation_note': (
            'In-loop evaluation is provisional. Evaluate each final checkpoint with '
            'tools/eval_checkpoint.py using its arm directory and the absolute endpoint.'),
    }
    _json_dump(output_root / 'completion.json', completion)
    manifest['status'] = 'complete'
    manifest['completion'] = str(output_root / 'completion.json')
    _json_dump(output_root / 'pair_manifest.json', manifest)
    print(f'paired continuation complete at absolute step {budget.total_updates}: {output_root}')
    return completion


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', required=True)
    parser.add_argument('--source-step', type=int, default=50_000)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--additional-steps', type=int, default=450_000)
    parser.add_argument('--save-interval', type=int, default=50_000)
    parser.add_argument('--log-interval', type=int, default=None)
    parser.add_argument('--eval-interval', type=int, default=50_000)
    parser.add_argument('--eval-episodes', type=int, default=50)
    parser.add_argument('--eval-relabel-size', type=int, default=None)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args(argv)
    if args.eval_interval < 0:
        parser.error('--eval-interval must be >= 0')
    if args.eval_episodes < 1:
        parser.error('--eval-episodes must be >= 1')
    if args.log_interval is not None and args.log_interval < 1:
        parser.error('--log-interval must be >= 1')
    if args.eval_relabel_size is not None and args.eval_relabel_size < 1:
        parser.error('--eval-relabel-size must be >= 1')
    return args


if __name__ == '__main__':
    run(parse_args())

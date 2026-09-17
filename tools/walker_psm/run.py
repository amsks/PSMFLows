"""Train/evaluate archived raw-action PSM only from a verified source snapshot."""
# ruff: noqa: I001 -- xla_guard must precede JAX imports.

import argparse
import csv
import importlib.metadata
import json
from pathlib import Path
import pickle
import time

from utils import xla_guard  # noqa: F401 -- before JAX
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np

from tools.walker_psm.data import load_episodes, sample_batch, sha256, verify_run_sidecars

TASKS = ('stand', 'walk', 'run', 'flip')
EXPECTED_PROTO_SHA256 = {
    'released': '2d6986380bdf81eae9015a01f659dca3a676b08ba8dcea56c7497f0b766da7e8',
    'bounded': '730718b792a05097adbe1d22af8ffa3375380f5b69cd99481232d0fd2494130e',
}


def validate_proto_table(path, table, mode):
    low, high = (-1, 1) if mode == 'bounded' else (-2, 0)
    if table.shape != (85536, 6) or not np.isfinite(table).all() or table.min() < low or table.max() >= high:
        raise ValueError(f'{mode} proto table must have shape[85536,6] and range[{low},{high})')
    if sha256(path) != EXPECTED_PROTO_SHA256[mode]:
        raise ValueError('Proto table differs from verified per-row Torch MT19937 artifact')


def reference_config(proto_table):
    return {
        'agent_name': 'psm',
        'batch_size': 1024,
        'z_dim': 128,
        'max_log_seed': 16,
        'num_parallel': 2,
        'discount': 0.98,
        'tau': 0.01,
        'ortho_coef': 1.0,
        'mix_ratio': 0.5,
        'pessimism_penalty': 0.0,
        'actor_pessimism_penalty': 0.5,
        'actor_std': 0.2,
        'stddev_clip': 0.3,
        'norm_z': True,
        'phi_input': 's',
        'lr_phi': 1e-4,
        'lr_sf': 1e-4,
        'lr_actor': 1e-4,
        'phi': {'hidden_dim': 256, 'hidden_layers': 2},
        'sf': {'hidden_dim': 1024, 'hidden_layers': 1, 'embedding_layers': 2},
        'actor': {'type': 'ddpgbc', 'hidden_dim': 1024, 'hidden_layers': 1, 'embedding_layers': 2, 'bc_coeff': 0.0},
        'proto_table_path': str(Path(proto_table).resolve()),
        'encoder': None,
    }


def verify_snapshot(root):
    manifest = json.loads((root / 'source_manifest.json').read_text())
    for name, expected in manifest['files'].items():
        if sha256(root / name) != expected:
            raise ValueError(f'Source snapshot hash mismatch: {name}')
    return manifest


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def make_env(task, seed):
    if task == 'flip':
        from walker_tasks import walker

        return walker.make('flip', task_kwargs={'random': seed})
    from dm_control import suite

    return suite.load('walker', task, task_kwargs={'random': seed})


def flatten(observation):
    return np.concatenate([np.asarray(v).reshape(-1) for v in observation.values()]).astype(np.float32)


def task_rewards(task, physics):
    env = make_env(task, 0)
    env.reset()
    rewards = np.empty(len(physics), np.float32)
    for i, state in enumerate(physics):
        with env.physics.reset_context():
            env.physics.set_state(state)
        rewards[i] = env.task.get_reward(env.physics)
    env.close()
    return rewards


def evaluate(agent, dataset, *, episodes, inference_samples, eval_seed, workers=4):
    # Separate deterministic RNG: evaluation cannot change the training stream.
    indices = np.random.default_rng(eval_seed).integers(len(dataset['actions']), size=inference_samples)
    outputs = {}
    for task in TASKS:
        rewards = task_rewards(task, dataset['next_physics'][indices])
        task_agent = agent.infer_eval_z(dataset['next_observations'][indices], rewards)
        returns, lengths = [], []
        envs = [make_env(task, eval_seed) for _ in range(min(workers, episodes))]
        try:
            for start in range(0, episodes, len(envs)):
                active = envs[: min(len(envs), episodes - start)]
                timesteps = []
                for offset, env in enumerate(active):
                    env.task._random.seed(eval_seed + start + offset)
                    timesteps.append(env.reset())
                totals = np.zeros(len(active), np.float64)
                count = np.zeros(len(active), np.int32)
                done = np.zeros(len(active), bool)
                for _ in range(1000):
                    observations = np.stack([flatten(t.observation) for t in timesteps])
                    actions = np.asarray(task_agent.sample_actions(observations))
                    if not np.isfinite(actions).all() or (np.abs(actions) > 1.000001).any():
                        raise ValueError('Invalid evaluated actor action')
                    for i, env in enumerate(active):
                        if not done[i]:
                            timesteps[i] = env.step(actions[i])
                            totals[i] += float(timesteps[i].reward)
                            count[i] += 1
                            done[i] = timesteps[i].last()
                    if done.all():
                        break
                if not done.all() or not np.all(count == 1000):
                    raise ValueError(f'Unexpected Walker episode length: {count}')
                returns.extend(totals.tolist())
                lengths.extend(count.tolist())
        finally:
            for env in envs:
                env.close()
        outputs[task] = {
            'mean_return': float(np.mean(returns)),
            'episode_returns': returns,
            'episode_lengths': lengths,
            'episodes': episodes,
            'task_z': np.asarray(task_agent.task_z).tolist(),
            'inference_reward_mean': float(rewards.mean()),
        }
    return {
        'tasks': outputs,
        'task_mean': float(np.mean([v['mean_return'] for v in outputs.values()])),
        'inference_samples': inference_samples,
        'eval_seed': eval_seed,
        'inference_index_sha256': __import__('hashlib').sha256(indices.tobytes()).hexdigest(),
    }


def save_checkpoint(path, agent, step, rng, flags_sha256):
    # Exclusive output prevents replacing an earlier experiment/checkpoint.
    state = {
        'agent': flax.serialization.to_state_dict(jax.device_get(agent)),
        'step': step,
        'numpy_rng': rng.bit_generator.state,
        'flags_sha256': flags_sha256,
    }
    with Path(path).open('xb') as f:
        pickle.dump(state, f, protocol=5)


def restore_checkpoint(path, template, flags_sha256):
    with Path(path).open('rb') as f:
        state = pickle.load(f)  # trusted, locally produced checkpoint only
    if state['flags_sha256'] != flags_sha256:
        raise ValueError('Checkpoint and run flags do not match')
    agent = flax.serialization.from_state_dict(template, state['agent'])
    rng = np.random.default_rng()
    rng.bit_generator.state = state['numpy_rng']
    return agent, int(state['step']), rng


def parser():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['train', 'eval', 'config'])
    p.add_argument('--data-dir', required=True)
    p.add_argument('--proto-table', required=True)
    p.add_argument('--proto-mode', choices=['bounded', 'released'], default='bounded')
    p.add_argument('--run-dir', required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--steps', type=int, default=2000000)
    p.add_argument('--max-episodes', type=int, default=5000)
    p.add_argument('--eval-interval', type=int, default=100000)
    p.add_argument('--save-interval', type=int, default=100000)
    p.add_argument('--log-interval', type=int, default=1000)
    p.add_argument('--eval-episodes', type=int, default=10)
    p.add_argument('--final-eval-episodes', type=int, default=500)
    p.add_argument('--inference-samples', type=int, default=10000)
    p.add_argument('--eval-seed', type=int, default=0)
    p.add_argument('--eval-workers', type=int, default=4)
    p.add_argument('--checkpoint-step', type=int)
    p.add_argument('--report', help='Exclusive JSON output for mode=eval')
    return p


def main():
    args = parser().parse_args()
    for name in (
        'steps',
        'max_episodes',
        'eval_interval',
        'save_interval',
        'log_interval',
        'eval_episodes',
        'final_eval_episodes',
        'inference_samples',
        'eval_workers',
    ):
        if getattr(args, name) < 1:
            raise ValueError(f'{name} must be positive')
    cfg = reference_config(args.proto_table)
    if args.mode == 'config':
        print(json.dumps({'agent': cfg, 'run': vars(args)}, indent=2))
        return
    root = Path(__file__).resolve().parents[2]
    source = verify_snapshot(root)
    from psm_agent import PSMAgent

    table = np.load(args.proto_table, allow_pickle=False)
    validate_proto_table(args.proto_table, table, args.proto_mode)
    dataset, data_manifest = load_episodes(args.data_dir, args.max_episodes)
    if dataset['observations'].shape[1:] != (24,) or dataset['actions'].shape[1:] != (6,):
        raise ValueError('Require Walker observation24/action6')
    run = Path(args.run_dir).resolve()
    agent = PSMAgent.create(args.seed, jnp.zeros((1, 24)), jnp.zeros((1, 6)), cfg)
    if args.mode == 'eval':
        flags = json.loads((run / 'flags.json').read_text())
        verify_run_sidecars(run, flags)
        if (
            flags['seed'] != args.seed
            or flags['agent'] != json.loads(json.dumps(cfg))
            or flags['proto_mode'] != args.proto_mode
        ):
            raise ValueError('Evaluation arguments differ from saved training config')
        if data_manifest != json.loads((run / 'dataset_manifest.json').read_text()):
            raise ValueError('Evaluation buffer differs from training buffer')
        if source != json.loads((run / 'source_manifest.json').read_text()):
            raise ValueError('Evaluation source differs from training source')
        if sha256(args.proto_table) != flags['proto_table_sha256']:
            raise ValueError('Evaluation proto table changed')
        if args.checkpoint_step is None or args.report is None:
            raise ValueError('Evaluation requires --checkpoint-step and --report')
        agent, step, _ = restore_checkpoint(
            run / f'checkpoint_{args.checkpoint_step}.pkl', agent, sha256(run / 'flags.json')
        )
        report = evaluate(
            agent,
            dataset,
            episodes=args.eval_episodes,
            inference_samples=args.inference_samples,
            eval_seed=args.eval_seed,
            workers=args.eval_workers,
        )
        write_json(
            args.report,
            dict(**report, step=step, seed=args.seed, run_dir=str(run), flags_sha256=sha256(run / 'flags.json')),
        )
        return
    run.mkdir(parents=True, exist_ok=False)
    write_json(run / 'source_manifest.json', source)
    write_json(run / 'dataset_manifest.json', data_manifest)
    flags = {
        'seed': args.seed,
        'agent': cfg,
        'run': vars(args),
        'method': 'raw-action bilinear PSM; no behavior flow; TD3 actor; no reward training',
        'protocol': 'Standardized raw-action PSM; paper5M data/2M budget; released optimizer knobs; 10k inference',
        'proto_mode': args.proto_mode,
        'deliberate_deviations': [
            'JAX initialization/minibatch RNG',
            'Task mixture reads phi before proto step (archived verified port)',
            'Full5M buffer, not released first10% truncation',
            '2M endpoint, not released3M default',
            '10k inference, not released5000',
            (
                'Bounded proto2*rand-1 corrects released out-of-bounds action sampler; NOT exact replication'
                if args.proto_mode == 'bounded'
                else 'Exact Torch proto table retains released [-2,0) actions'
            ),
        ],
        'proto_table_sha256': sha256(args.proto_table),
        'source_manifest_sha256': sha256(run / 'source_manifest.json'),
        'dataset_manifest_sha256': sha256(run / 'dataset_manifest.json'),
        'proto_table_stats': {
            'min': float(table.min()),
            'max': float(table.max()),
            'fraction_rows_outside_env_action_bounds': float(np.mean(np.any(np.abs(table) > 1, axis=1))),
        },
        'versions': {p: importlib.metadata.version(p) for p in ['jax', 'flax', 'optax', 'dm-control', 'mujoco']},
        'devices': [str(d) for d in jax.devices()],
    }
    write_json(run / 'flags.json', flags)
    rng = np.random.default_rng(args.seed)
    fields = [
        'step',
        'elapsed_seconds',
        'psm_loss',
        'psm_diag',
        'psm_offdiag',
        'orth_loss',
        'orth_diag',
        'orth_offdiag',
        'sf_loss',
        'sf_diag',
        'sf_offdiag',
        'actor_loss',
        'q',
    ]
    started = time.monotonic()
    with (run / 'train.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for step in range(1, args.steps + 1):
            indices = rng.integers(len(dataset['actions']), size=cfg['batch_size'])
            agent, info = agent.update(sample_batch(dataset, indices))
            if step == 1 or step % args.log_interval == 0 or step == args.steps:
                metrics = {k: float(v) for k, v in jax.device_get(info).items()}
                if not np.isfinite(list(metrics.values())).all():
                    raise FloatingPointError(f'Nonfinite losses at step{step}: {metrics}')
                writer.writerow(dict(step=step, elapsed_seconds=time.monotonic() - started, **metrics))
                f.flush()
                print(json.dumps(dict(step=step, elapsed_seconds=time.monotonic() - started, **metrics)), flush=True)
            if step % args.save_interval == 0 or step == args.steps:
                save_checkpoint(run / f'checkpoint_{step}.pkl', agent, step, rng, sha256(run / 'flags.json'))
            if step % args.eval_interval == 0 or step == args.steps:
                n = args.final_eval_episodes if step == args.steps else args.eval_episodes
                eval_agent = agent
                if step == args.steps:
                    verify_run_sidecars(run, json.loads((run / 'flags.json').read_text()))
                    eval_agent, restored_step, _ = restore_checkpoint(
                        run / f'checkpoint_{step}.pkl', agent, sha256(run / 'flags.json')
                    )
                    if restored_step != step:
                        raise ValueError('Final checkpoint step mismatch')
                report = evaluate(
                    eval_agent,
                    dataset,
                    episodes=n,
                    inference_samples=args.inference_samples,
                    eval_seed=args.eval_seed,
                    workers=args.eval_workers,
                )
                write_json(
                    run / f'eval_{step}.json',
                    dict(
                        **report,
                        step=step,
                        seed=args.seed,
                        run_dir=str(run),
                        evaluated_restored_checkpoint=step == args.steps,
                        flags_sha256=sha256(run / 'flags.json'),
                    ),
                )
                print(json.dumps({'step': step, 'task_mean': report['task_mean'], 'eval_episodes': n}), flush=True)


if __name__ == '__main__':
    main()

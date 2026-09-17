"""Train and evaluate the archived full-affine agent from an immutable source snapshot."""
# ruff: noqa: I001 -- xla_guard must precede JAX.

import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import pickle
import time

from utils import xla_guard  # noqa: F401
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np

from tools.affine_stitch.data import Replay, load_data, sha256, verify_run_sidecars

TASKS = (1, 2, 3, 4, 5)


def reference_config():
    return {
        'agent_name': 'affine_psm', 'batch_size': 1024, 'd_dim': 128, 'z_dim': 128,
        'max_log_seed': 16, 'proto_table_path': None, 'discount': .99, 'tau': .01,
        'lr': 1e-4, 'lr_w': 1e-4, 'lr_actor': 1e-4, 'ortho_coef': 1000.,
        'measure': {'factored': True, 'k_dim': 32, 'hidden_dim': 1024, 'hidden_layers': 3, 'b_scale': 10.},
        'actor': {'type': 'ddpgbc', 'hidden_dim': 1024, 'hidden_layers': 1,
                  'embedding_layers': 2, 'bc_coeff': .3},
        'inference': {'mode': 'full', 'use_dgd': True, 'inf_coeff': 5.,
                      'num_inference_steps': 5120, 'lagrange_hidden_dim': 256,
                      'lagrange_hidden_layers': 2, 'norm_w': True, 'num_actor_inference_steps': 512},
        'encoder': None, 'ob_dims': None, 'action_dim': None,
    }


def correct_proto_table(released, mode):
    table = np.asarray(released)
    if not np.isfinite(table).all() or table.min() < -2 or table.max() >= 0:
        raise ValueError('Archived proto table must have range[-2,0)')
    if mode == 'released':
        return table.copy()
    if mode != 'bounded':
        raise ValueError('Unknown proto mode')
    return table + np.float32(1)


def verify_snapshot(root):
    root = Path(root).resolve()
    manifest = json.loads((root / 'source_manifest.json').read_text())
    for name, digest in manifest['files'].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or sha256(path) != digest:
            raise ValueError(f'Source snapshot hash mismatch: {name}')
    return manifest


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def tree_digest(tree):
    digest = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(jax.device_get(tree)):
        value = np.asarray(leaf)
        digest.update(str((value.shape, value.dtype)).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def representation_digest(agent):
    return tree_digest((agent.measure, agent.w, agent.target_measure, agent.target_w, agent.proto))


def make_env(task):
    import gymnasium
    import ogbench  # noqa: F401 -- register environments

    env = gymnasium.make(f'antmaze-medium-singletask-task{task}-v0')
    if env.spec.max_episode_steps != 1000 or env.observation_space.shape != (29,) or env.action_space.shape != (8,):
        env.close()
        raise ValueError('AntMaze environment schema/horizon mismatch')
    return env


def seeded_reset(env, seed):
    # OGBench add_noise draws global NumPy before super.reset(seed). Preserve that
    # ambient stream while explicitly pinning all three reset RNG sources.
    previous = np.random.get_state()
    try:
        np.random.seed(seed)
        env.action_space.seed(seed)
        return env.reset(seed=seed)
    finally:
        np.random.set_state(previous)


def task_goal(task, seed):
    env = make_env(task)
    try:
        _, info = seeded_reset(env, seed)
        goal = np.asarray(info['goal'], np.float32)
        if goal.shape != (29,) or not np.isfinite(goal).all():
            raise ValueError('Expected finite full29D goal')
        np.testing.assert_allclose(goal[:2], env.unwrapped.cur_goal_xy, atol=1e-6)
        return goal
    finally:
        env.close()


def infer_task(agent, replay, goal, seed):
    before = representation_digest(agent)
    original_rng = json.dumps(replay.rng.bit_generator.state, sort_keys=True)
    # The archived implementation calls distill_actor once after coordinate inference.
    # Pin its actor RNG as well as dataset/dual/permutation RNGs independently of training.
    base = agent.replace(rng=jax.random.PRNGKey(seed))
    inferred = base.infer_w_goal(replay.fork(seed), goal, seed=seed)
    after = representation_digest(inferred)
    actor_steps = int(inferred.actor.step) - int(agent.actor.step)
    expected = int(agent.config['inference']['num_actor_inference_steps'])
    if before != after or actor_steps != expected:
        raise ValueError('Inference changed representation or adapted actor an incorrect number of times')
    if original_rng != json.dumps(replay.rng.bit_generator.state, sort_keys=True):
        raise ValueError('Inference advanced training sampling RNG')
    if not np.isfinite(np.asarray(inferred.w_inf)).all():
        raise FloatingPointError('Nonfinite inferred coordinate')
    return inferred, {'representation_sha256': before, 'representation_unchanged': True,
                      'actor_updates': actor_steps,
                      'coordinate_updates': int(agent.config['inference']['num_inference_steps']),
                      'sampling_seed': seed, 'training_rng_unchanged': True}


def evaluate(agent, replay, *, episodes, eval_seed, workers):
    outputs = {}
    started = time.monotonic()
    for task in TASKS:
        seed = eval_seed + task - 1
        goal = task_goal(task, seed)
        inference_start = time.monotonic()
        task_agent, diagnostic = infer_task(agent, replay, goal, seed)
        diagnostic['seconds'] = time.monotonic() - inference_start
        envs = [make_env(task) for _ in range(min(workers, episodes))]
        successes, lengths, episode_seeds, initial_hashes = [], [], [], []
        rollout_start = time.monotonic()
        try:
            for start in range(0, episodes, len(envs)):
                active = envs[:min(len(envs), episodes - start)]
                observations = []
                for offset, env in enumerate(active):
                    episode_seed = eval_seed + start + offset
                    observation, info = seeded_reset(env, episode_seed)
                    np.testing.assert_allclose(np.asarray(info['goal'])[:2], goal[:2], atol=1e-6)
                    np.testing.assert_allclose(env.unwrapped.cur_goal_xy, goal[:2], atol=1e-6)
                    observations.append(observation)
                    episode_seeds.append(episode_seed)
                    initial_hashes.append(hashlib.sha256(np.asarray(observation, np.float32).tobytes()).hexdigest())
                done = np.zeros(len(active), bool)
                success = np.zeros(len(active), bool)
                counts = np.zeros(len(active), np.int32)
                for _ in range(1000):
                    actions = np.asarray(task_agent.sample_actions(np.asarray(observations, np.float32)))
                    if not np.isfinite(actions).all() or (np.abs(actions) > 1.000001).any():
                        raise FloatingPointError('Invalid actor action')
                    for i, env in enumerate(active):
                        if not done[i]:
                            observations[i], _, terminated, truncated, info = env.step(actions[i])
                            success[i] |= bool(info['success'])
                            counts[i] += 1
                            done[i] = terminated or truncated
                    if done.all():
                        break
                if not done.all():
                    raise ValueError('AntMaze failed to terminate within1000 steps')
                successes.extend(success.astype(int).tolist())
                lengths.extend(counts.tolist())
        finally:
            for env in envs:
                env.close()
        outputs[str(task)] = {
            'env': f'antmaze-medium-singletask-task{task}-v0', 'goal': goal.tolist(),
            'goal_xy': goal[:2].tolist(), 'goal_sha256': hashlib.sha256(goal.tobytes()).hexdigest(),
            'success': float(np.mean(successes)), 'successes': sum(successes), 'episodes': episodes,
            'episode_success': successes, 'episode_lengths': lengths,
            'episode_seeds': episode_seeds, 'initial_observation_sha256': initial_hashes,
            'w_inf': np.asarray(task_agent.w_inf).tolist(), 'inference': diagnostic,
            'rollout_seconds': time.monotonic() - rollout_start,
        }
        print(json.dumps({'task': task, 'success': outputs[str(task)]['success'],
                          'inference_seconds': diagnostic['seconds']}), flush=True)
    return {'tasks': outputs, 'task_mean': float(np.mean([v['success'] for v in outputs.values()])),
            'eval_seed': eval_seed, 'workers': workers, 'worker_mode': 'batched actor over independent environments',
            'eval_seconds': time.monotonic() - started}


def save_checkpoint(path, agent, step, rng, flags_sha256):
    state = {'agent': flax.serialization.to_state_dict(jax.device_get(agent)), 'step': step,
             'numpy_rng': rng.bit_generator.state, 'flags_sha256': flags_sha256}
    with Path(path).open('xb') as stream:
        pickle.dump(state, stream, protocol=5)


def restore_checkpoint(path, template, flags_sha256):
    with Path(path).open('rb') as stream:
        state = pickle.load(stream)  # trusted local checkpoints only
    if state['flags_sha256'] != flags_sha256:
        raise ValueError('Checkpoint and run flags differ')
    agent = flax.serialization.from_state_dict(template, state['agent'])
    rng = np.random.default_rng()
    rng.bit_generator.state = state['numpy_rng']
    return agent, int(state['step']), rng


def parser():
    result = argparse.ArgumentParser()
    result.add_argument('mode', choices=['train', 'preflight', 'config'])
    result.add_argument('--data-dir', required=True)
    result.add_argument('--run-dir', required=True)
    result.add_argument('--seed', type=int, default=0)
    result.add_argument('--steps', type=int, default=500000)
    result.add_argument('--proto-mode', choices=['bounded', 'released'], default='bounded')
    result.add_argument('--eval-interval', type=int, default=100000)
    result.add_argument('--save-interval', type=int, default=100000)
    result.add_argument('--log-interval', type=int, default=1000)
    result.add_argument('--eval-episodes', type=int, default=10)
    result.add_argument('--final-eval-episodes', type=int, default=500)
    result.add_argument('--eval-seed', type=int, default=0)
    result.add_argument('--eval-workers', type=int, default=4)
    return result


def main():
    args = parser().parse_args()
    for name in ('steps', 'eval_interval', 'save_interval', 'log_interval', 'eval_episodes',
                 'final_eval_episodes', 'eval_workers'):
        if getattr(args, name) < 1:
            raise ValueError(f'{name} must be positive')
    cfg = reference_config()
    if args.mode == 'config':
        print(json.dumps({'agent': cfg, 'run': vars(args)}, indent=2))
        return
    root = Path(__file__).resolve().parents[2]
    source = verify_snapshot(root)
    from affine_agent import AffinePSMAgent

    datasets, data_manifest = load_data(args.data_dir)
    replay = Replay(datasets['train'], args.seed)
    agent = AffinePSMAgent.create(args.seed, jnp.zeros((1, 29)), jnp.zeros((1, 8)), cfg)
    original, powers = agent.proto
    table = correct_proto_table(original, args.proto_mode)
    agent = agent.replace(proto=(jnp.asarray(table), powers))
    run = Path(args.run_dir).resolve()
    run.mkdir(parents=True, exist_ok=False)
    write_json(run / 'source_manifest.json', source)
    write_json(run / 'dataset_manifest.json', data_manifest)
    flags = {
        'seed': args.seed, 'agent': cfg, 'resolved_agent': flax.core.unfreeze(agent.config),
        'run': vars(args), 'proto_mode': args.proto_mode,
        'method': 'Archived full-affine PSM, raw actions, reward-free training; full goal inference',
        'protocol': 'Untuned AntMaze transfer; not reference parity',
        'deviations': ['AntMaze ddpgbc actor with BC0.3', 'AntMaze discount0.99',
                       'Proto table +1 for bounded mode; otherwise archived released range'],
        'proto_table_sha256': hashlib.sha256(table.tobytes()).hexdigest(),
        'proto_table_stats': {'min': float(table.min()), 'max': float(table.max()),
                              'fraction_out_of_bounds': float(np.mean(np.any(np.abs(table) > 1, axis=1)))},
        'source_manifest_sha256': sha256(run / 'source_manifest.json'),
        'dataset_manifest_sha256': sha256(run / 'dataset_manifest.json'),
        'versions': {p: importlib.metadata.version(p) for p in ('jax', 'flax', 'optax', 'ogbench', 'gymnasium', 'mujoco')},
        'devices': [str(device) for device in jax.devices()],
    }
    write_json(run / 'flags.json', flags)
    flags_hash = sha256(run / 'flags.json')
    if args.mode == 'preflight':
        save_checkpoint(run / 'checkpoint_0.pkl', agent, 0, replay.rng, flags_hash)
        restored, step, _ = restore_checkpoint(run / 'checkpoint_0.pkl', agent, flags_hash)
        if step != 0 or tree_digest(agent) != tree_digest(restored):
            raise ValueError('Preflight checkpoint restoration mismatch')
        goals = {str(task): task_goal(task, args.eval_seed + task - 1).tolist() for task in TASKS}
        write_json(run / 'preflight.json', {'checkpoint_roundtrip': True, 'goals': goals,
                                           'dataset': data_manifest, 'flags_sha256': flags_hash})
        return
    started = time.monotonic()
    fields = ['step', 'elapsed_seconds', 'update_seconds', 'psm_loss', 'psm_diag', 'psm_offdiag',
              'orth_loss', 'orth_diag', 'orth_offdiag', 'actor_loss', 'actor_q', 'actor_bc']
    update_seconds = 0.
    with (run / 'train.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for step in range(1, args.steps + 1):
            begin = time.monotonic()
            agent, info = agent.update(replay.sample(cfg['batch_size']))
            jax.block_until_ready(info)
            update_seconds += time.monotonic() - begin
            if step == 1 or step % args.log_interval == 0 or step == args.steps:
                metrics = {k: float(v) for k, v in jax.device_get(info).items()}
                if not np.isfinite(list(metrics.values())).all():
                    raise FloatingPointError(f'Nonfinite losses at step{step}: {metrics}')
                row = dict(step=step, elapsed_seconds=time.monotonic() - started,
                           update_seconds=update_seconds, **metrics)
                writer.writerow(row)
                stream.flush()
                print(json.dumps(row), flush=True)
            if step % args.save_interval == 0 or step == args.steps:
                save_checkpoint(run / f'checkpoint_{step}.pkl', agent, step, replay.rng, flags_hash)
            if step % args.eval_interval == 0 or step == args.steps:
                final = step == args.steps
                evaluation_agent = agent
                if final:
                    verify_run_sidecars(run, json.loads((run / 'flags.json').read_text()))
                    evaluation_agent, restored_step, _ = restore_checkpoint(
                        run / f'checkpoint_{step}.pkl', agent, sha256(run / 'flags.json'))
                    if restored_step != step or tree_digest(evaluation_agent) != tree_digest(agent):
                        raise ValueError('Final checkpoint restoration mismatch')
                report = evaluate(evaluation_agent, replay,
                                  episodes=args.final_eval_episodes if final else args.eval_episodes,
                                  eval_seed=args.eval_seed, workers=args.eval_workers)
                write_json(run / f'eval_{step}.json', dict(
                    **report, step=step, seed=args.seed, run_dir=str(run), flags_sha256=flags_hash,
                    evaluated_restored_checkpoint=final,
                    checkpoint_sha256=sha256(run / f'checkpoint_{step}.pkl') if final else None))


if __name__ == '__main__':
    main()

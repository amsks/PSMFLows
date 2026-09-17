"""Explicit ExORL transition pairing; collection rewards never enter training."""

import hashlib
import re
from pathlib import Path

import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify_run_sidecars(directory, flags):
    """Bind source/data sidecars to the flags hash recorded inside checkpoints."""
    for name in ('source_manifest', 'dataset_manifest'):
        expected = flags.get(f'{name}_sha256')
        if expected is None or sha256(Path(directory) / f'{name}.json') != expected:
            raise ValueError(f'{name} hash does not match checkpoint-bound run flags')


def episode_transitions(ep):
    obs, act = np.asarray(ep['observation']), np.asarray(ep['action'])
    physics = np.asarray(ep['physics'])
    disc = np.asarray(ep['discount']).reshape(-1)
    if len(obs) < 2 or len({len(obs), len(act), len(physics), len(disc)}) != 1:
        raise ValueError('Episode fields must share T+1 rows, including initial dummy row')
    if not np.all(disc[1:] == 1):
        raise ValueError('Nonunit environment discount: archived PSM uses scalar gamma; refusing invalid backup')
    for name, value in [('observation', obs), ('action', act), ('physics', physics)]:
        if not np.isfinite(value).all():
            raise ValueError(f'Nonfinite {name}')
    if np.any(np.abs(act[1:]) > 1.000001):
        raise ValueError('Recorded Walker action outside [-1,1]')
    return {
        'observations': obs[:-1].astype(np.float32),
        'actions': act[1:].astype(np.float32),
        'next_observations': obs[1:].astype(np.float32),
        'next_physics': physics[1:].copy(),
    }


def load_episodes(directory, max_episodes=5000, expected_length=1000):
    paths = list(Path(directory).glob('episode_*_*.npz'))

    def key(path):
        match = re.search(r'episode_(\d+)_(\d+)\.npz$', path.name)
        if match is None:
            raise ValueError(f'Unrecognized episode name: {path}')
        return int(match[1])

    paths.sort(key=key)
    if len(paths) < max_episodes or max_episodes < 1:
        raise ValueError(f'Requested {max_episodes} episodes, found {len(paths)}')
    paths = paths[:max_episodes]
    if len({key(p) for p in paths}) != len(paths):
        raise ValueError('Duplicate numeric episode IDs')
    arrays = None
    records = []
    for i, path in enumerate(paths):
        with np.load(path, allow_pickle=False) as ep:
            part = episode_transitions(ep)
        if len(part['actions']) != expected_length:
            raise ValueError(f'{path}: expected {expected_length} transitions')
        if arrays is None:
            arrays = {k: np.empty((max_episodes * expected_length, *v.shape[1:]), v.dtype) for k, v in part.items()}
        for k, value in part.items():
            arrays[k][i * expected_length : (i + 1) * expected_length] = value
        records.append({'name': path.name, 'sha256': sha256(path), 'transitions': expected_length})
    return arrays, {
        'episodes': records,
        'transitions': max_episodes * expected_length,
        'ordering': 'numeric episode ID, then original transition order',
        'collection_reward_used': False,
        'environment_discounts_all_one': True,
    }


def sample_batch(dataset, indices):
    return {
        **{k: dataset[k][indices] for k in ('observations', 'actions', 'next_observations')},
        'index': np.asarray(indices, np.int32),
    }

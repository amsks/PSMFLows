"""Native OGBench row pairing with independent, reward-free sampling streams."""

import hashlib
from pathlib import Path

import numpy as np

FILES = {
    'train': ('antmaze-medium-stitch-v0.npz', 'b3fa34eef8d24dc812d734abaf74db279a5f39cd63debf23961d62c314d71e99', 1000000),
    'validation': ('antmaze-medium-stitch-v0-val.npz', '3015f74bb9d0f0fef0037d310c2be157b78af2c613f8cb0579f1de018e7c0af0', 100000),
}
FIELDS = ('observations', 'actions', 'next_observations')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_run_sidecars(directory, flags):
    for name in ('source_manifest', 'dataset_manifest'):
        if sha256(Path(directory) / f'{name}.json') != flags.get(f'{name}_sha256'):
            raise ValueError(f'{name} hash differs from checkpoint-bound flags')


def load_file(path, expected_sha256, expected_transitions, episode_rows=201):
    from ogbench.utils import load_dataset

    path = Path(path).resolve()
    if sha256(path) != expected_sha256:
        raise ValueError(f'Dataset checksum mismatch: {path}')
    with np.load(path, allow_pickle=False) as raw:
        terminal = raw['terminals']
        expected = np.arange(episode_rows - 1, len(terminal), episode_rows)
        if (terminal.ndim != 1 or len(terminal) % episode_rows != 0
                or not np.array_equal(np.flatnonzero(terminal), expected)):
            raise ValueError('Unexpected terminal sentinel layout')
        raw_rows = len(terminal)
    # Native loader removes terminal sentinel rows and pairs with original next rows.
    parsed = load_dataset(str(path), compact_dataset=False)
    arrays = {k: np.asarray(parsed[k], np.float32) for k in FIELDS}
    if any(len(value) != expected_transitions for value in arrays.values()):
        raise ValueError('Unexpected valid transition count')
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError('Nonfinite dataset values')
    if np.any(np.abs(arrays['actions']) > 1.000001):
        raise ValueError('Dataset action outside environment bounds')
    for value in arrays.values():
        value.setflags(write=False)
    return arrays, {'path': str(path), 'sha256': expected_sha256, 'raw_rows': raw_rows,
                    'transitions': expected_transitions, 'episode_rows': episode_rows,
                    'shapes': {k: list(v.shape) for k, v in arrays.items()}}


def load_data(directory):
    datasets, records = {}, {}
    for split, (name, digest, size) in FILES.items():
        datasets[split], records[split] = load_file(Path(directory) / name, digest, size)
        if (datasets[split]['observations'].shape != (size, 29)
                or datasets[split]['actions'].shape != (size, 8)):
            raise ValueError('Require AntMaze observation29/action8')
    return datasets, {'splits': records, 'reward_fields_used': False,
                      'pairing': 'ogbench.utils.load_dataset compact_dataset=False',
                      'training_split': 'train', 'inference_split': 'train'}


class Replay:
    def __init__(self, arrays, seed):
        self.arrays = arrays
        self.size = len(arrays['actions'])
        self.rng = np.random.default_rng(seed)

    def sample(self, count):
        indices = self.rng.integers(self.size, size=count, dtype=np.int32)
        return {**{k: self.arrays[k][indices] for k in FIELDS}, 'index': indices}

    def fork(self, seed):
        return Replay(self.arrays, seed)

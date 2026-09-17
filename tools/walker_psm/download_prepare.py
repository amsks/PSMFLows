"""Validate the official ExORL archive, extract safely, and record provenance."""

import argparse
import hashlib
import json
import shutil
import stat
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(root):
    archive = root / 'walker/rnd.zip.part'
    complete = root / 'walker/rnd.zip'
    if not archive.exists():
        archive = complete
    assert archive.stat().st_size == 2585461986, 'Unexpected official archive byte count'
    archive_sha = sha256(archive)
    destination = (root / 'walker/rnd').resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        members = source.infolist()
        for member in members:
            name = PurePosixPath(member.filename)
            assert not name.is_absolute() and '..' not in name.parts, member.filename
            assert not stat.S_ISLNK(member.external_attr >> 16), member.filename
            target = destination.joinpath(*name.parts)
            assert target.resolve().is_relative_to(destination), member.filename
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            assert not target.exists(), f'Refusing to overwrite {target}'
            target.parent.mkdir(parents=True, exist_ok=True)
            # Reading each member through EOF validates the ZIP CRC.
            with source.open(member) as incoming, target.open('xb') as outgoing:
                shutil.copyfileobj(incoming, outgoing)
        extracted_bytes = sum(member.file_size for member in members)
    if archive != complete:
        assert not complete.exists()
        archive.rename(complete)
    print('Archive CRC verification and safe extraction complete', flush=True)
    episodes = sorted((destination / 'buffer').glob('*.npz'), key=lambda path: int(path.stem.split('_')[1]))
    records = []
    schemas = {}
    min_action, max_action = float('inf'), float('-inf')
    discounts = set()
    for index, path in enumerate(episodes):
        with np.load(path, allow_pickle=False) as episode:
            fields = {
                key: {'shape': list(episode[key].shape), 'dtype': str(episode[key].dtype)} for key in episode.files
            }
            schema_key = json.dumps(fields, sort_keys=True)
            schemas[schema_key] = schemas.get(schema_key, 0) + 1
            length = episode['observation'].shape[0]
            assert all(episode[key].shape[0] == length for key in episode.files), path
            assert length - 1 == int(path.stem.split('_')[2]), path
            for key in episode.files:
                assert np.isfinite(episode[key]).all(), (path, key)
            min_action = min(min_action, float(episode['action'].min()))
            max_action = max(max_action, float(episode['action'].max()))
            discounts.update(float(value) for value in np.unique(episode['discount']))
            records.append({'filename': path.name, 'transitions': length - 1, 'sha256': sha256(path)})
        if (index + 1) % 1000 == 0:
            print(f'Validated {index + 1}/{len(episodes)} episode NPZs', flush=True)
    assert len(episodes) == 10000
    provenance = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'official_project': 'https://sites.google.com/view/exorl',
        'official_repository': 'https://github.com/denisyarats/exorl',
        'official_download_script': 'https://raw.githubusercontent.com/denisyarats/exorl/main/download.sh',
        'source_url': 'https://dl.fbaipublicfiles.com/exorl/walker/rnd.zip',
        'archive': str(complete.resolve()),
        'expected_archive_bytes': 2585461986,
        'actual_archive_bytes': complete.stat().st_size,
        'archive_sha256': archive_sha,
        'publisher_checksum_available': False,
        'http_etag': '5fcbe90c646cb714d080a6b4c8586186-309',
        'http_last_modified': 'Tue, 01 Feb 2022 04:02:44 GMT',
        'http_s3_version_id': 'jckeUX2phBkhmnJC28AX0DQkX6apa3jm',
        'zip_crc_verified_all_members': True,
        'extracted_bytes': extracted_bytes,
        'safe_extraction': 'All member names and symlink modes checked; existing files never overwritten.',
        'episode_directory': str((destination / 'buffer').resolve()),
        'episodes': len(records),
        'transitions': sum(record['transitions'] for record in records),
        'reference_subset': {
            'selection': 'First 5000 episodes sorted by numeric episode index',
            'episodes': 5000,
            'transitions': sum(record['transitions'] for record in records[:5000]),
        },
        'alignment': 'For t=1..T: observation[t-1], action[t], observation[t], discount[t]. Row 0 is reset/dummy.',
        'action_min': min_action,
        'action_max': max_action,
        'discount_values': sorted(discounts),
        'schemas': [{'fields': json.loads(key), 'episodes': count} for key, count in schemas.items()],
        'episode_checksums': records,
    }
    with (root / 'provenance.json').open('x') as stream:
        json.dump(provenance, stream, indent=2)
    print(json.dumps({key: value for key, value in provenance.items() if key != 'episode_checksums'}, indent=2))


def proto_table(root, mode='released'):
    import torch

    result = np.empty((2**16 + 20000, 6), dtype=np.float32)
    generator = torch.Generator(device='cpu')
    for index in range(len(result)):
        generator.manual_seed(index)
        draw = torch.rand(6, generator=generator)
        result[index] = (2 * draw - 1 if mode == 'bounded' else (draw - 1) * 2).numpy()
    # Independent comparison against the original global-generator spelling.
    for index in (0, 1, 127, 65535, 65536, 85535):
        torch.random.manual_seed(index)
        draw = torch.rand(6)
        expected = (2 * draw - 1 if mode == 'bounded' else (draw - 1) * 2).numpy()
        assert np.array_equal(result[index], expected)
    legacy_comparison = None
    if mode == 'bounded':
        legacy_path = root / 'proto_table.npy'
        legacy = np.load(legacy_path, allow_pickle=False)
        assert legacy.shape == result.shape and legacy.dtype == result.dtype
        assert np.array_equal(result, legacy + np.float32(1))
        legacy_comparison = {
            'path': str(legacy_path.resolve()),
            'sha256': sha256(legacy_path),
            'bounded_equals_legacy_plus_one_exactly': True,
            'legacy_fraction_rows_outside_action_bounds': float(np.any((legacy < -1) | (legacy > 1), axis=1).mean()),
        }
    stem = 'proto_table_bounded' if mode == 'bounded' else 'proto_table'
    path = root / f'{stem}.npy'
    with path.open('xb') as stream:
        np.save(stream, result)
    manifest = {
        'path': str(path.resolve()),
        'sha256': sha256(path),
        'shape': list(result.shape),
        'dtype': str(result.dtype),
        'torch_version': torch.__version__,
        'proto_mode': mode,
        'formula': 'For i: torch.Generator(device="cpu").manual_seed(i); '
        + ('2 * torch.rand(6, generator=g) - 1' if mode == 'bounded' else '(torch.rand(6, generator=g) - 1) * 2'),
        'global_generator_equivalence_checked_rows': [0, 1, 127, 65535, 65536, 85535],
        'minimum': float(result.min()),
        'maximum': float(result.max()),
        'fraction_rows_outside_action_bounds': float(np.any((result < -1) | (result > 1), axis=1).mean()),
        'legacy_comparison': legacy_comparison,
    }
    with (root / f'{stem}.json').open('x') as stream:
        json.dump(manifest, stream, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'proto-table'])
    parser.add_argument('--root', type=Path, default=Path('outputs/walker_psm_20260913/data'))
    parser.add_argument('--proto-mode', choices=['bounded', 'released'], default='released')
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.root)
    else:
        proto_table(args.root, mode=args.proto_mode)

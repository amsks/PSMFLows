"""Create an exclusive, complete execution snapshot without restoring live agents."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ARCHIVE_COMMIT = 'ee0e1746b36f48300eee06ad02e66b16f46851ff'


def materialize(destination):
    root = Path(__file__).resolve().parents[2]
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError('Snapshot exists; never overwrite experiment source')
    mapping = {'affine_agent.py': 'archive/agents/affine_psm.py',
               'affine_psm.yaml': 'archive/configs/agent/affine_psm.yaml'}
    for name in ('__init__.py', 'flax_utils.py', 'networks.py', 'psm_common.py', 'psm_networks.py', 'xla_guard.py'):
        mapping[f'utils/{name}'] = f'utils/{name}'
    files = {name: subprocess.check_output(['git', 'show', f'{ARCHIVE_COMMIT}:{source}'], cwd=root)
             for name, source in mapping.items()}
    files['tools/__init__.py'] = b''
    for path in (root / 'tools' / 'affine_stitch').glob('*.py'):
        files[f'tools/affine_stitch/{path.name}'] = path.read_bytes()
    destination.mkdir(parents=True, exist_ok=False)
    for name, content in files.items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(content)
    manifest = {'archive_commit': ARCHIVE_COMMIT, 'archive_paths': mapping,
                'files': {name: hashlib.sha256(content).hexdigest() for name, content in sorted(files.items())}}
    with (destination / 'source_manifest.json').open('x') as stream:
        json.dump(manifest, stream, indent=2)
        stream.write('\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--destination', required=True)
    print(json.dumps(materialize(parser.parse_args().destination), indent=2))

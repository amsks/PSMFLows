"""Generate a self-contained, hashed execution snapshot without restoring live agents."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ARCHIVE_COMMIT = 'ee0e1746b36f48300eee06ad02e66b16f46851ff'
SIBLING_COMMIT = 'b62dc9d5e73f282924c29d0ab64d1f889b43532e'


def materialize(destination):
    root = Path(__file__).resolve().parents[2]
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f'Snapshot already exists: {destination}; never overwrite a run source')
    files = {}
    files['psm_agent.py'] = subprocess.check_output(
        ['git', 'show', f'{ARCHIVE_COMMIT}:archive/agents/psm.py'], cwd=root
    )
    for name in ['__init__.py', 'flax_utils.py', 'networks.py', 'psm_common.py', 'psm_networks.py', 'xla_guard.py']:
        files[f'utils/{name}'] = subprocess.check_output(['git', 'show', f'{ARCHIVE_COMMIT}:utils/{name}'], cwd=root)
    for path in (root / 'tools' / 'walker_psm').glob('*.py'):
        files[f'tools/walker_psm/{path.name}'] = path.read_bytes()
    sibling = root.parent / 'Factored-FB'
    for name in ['walker.py', 'walker.xml', 'LICENSE', 'PROVENANCE.md']:
        files[f'walker_tasks/{name}'] = subprocess.check_output(
            ['git', 'show', f'{SIBLING_COMMIT}:env/exorl_dmc_tasks/{name}'], cwd=sibling
        )
    files['walker_tasks/__init__.py'] = b''
    for name, content in files.items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    manifest = {
        'archive_commit': ARCHIVE_COMMIT,
        'sibling_commit': SIBLING_COMMIT,
        'files': {k: hashlib.sha256(v).hexdigest() for k, v in sorted(files.items())},
    }
    (destination / 'source_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--destination', required=True)
    args = parser.parse_args()
    print(json.dumps(materialize(args.destination), indent=2))

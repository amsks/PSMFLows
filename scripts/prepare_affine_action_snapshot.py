"""Copy the current source (including uncommitted changes) for a fixed experiment."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign-dir', required=True)
    parser.add_argument('--skip-evaluation-manifests', action='store_true',
                        help='Copy source only when the campaign uses a different evaluation protocol.')
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1]
    campaign = Path(args.campaign_dir).resolve()
    destination = campaign / 'code'
    destination.mkdir(parents=True, exist_ok=False)
    files = [source / name for name in ('main.py', 'pyproject.toml', 'requirements.txt', 'CLAUDE.md', 'AGENTS.md')]
    for folder in ('agents', 'configs', 'envs', 'utils', 'tools', 'scripts', 'tests'):
        files.extend(p for p in (source / folder).rglob('*')
                     if p.is_file() and p.suffix in ('.py', '.yaml', '.yml', '.sh', '.sbatch', '.json'))
    hashes = {}
    for path in sorted(files):
        relative = path.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[str(relative)] = hashlib.sha256(target.read_bytes()).hexdigest()
    (destination / '.venv').symlink_to(source / '.venv', target_is_directory=True)
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip()
    manifest = {'source': str(source), 'git_head': head,
                'note': 'Includes current working-tree content; hashes, not HEAD alone, identify the code.',
                'sha256': hashes}
    (destination / 'source_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (campaign / 'logs').mkdir(exist_ok=True)
    (campaign / 'reports').mkdir(exist_ok=True)
    domains = () if args.skip_evaluation_manifests else (
        ('cube', 'cube-single-play'), ('antmaze', 'antmaze-medium-navigate'))
    for key, domain in domains:
        tasks = {str(t): f'{domain}-singletask-task{t}-v0' for t in range(1, 6)}
        evaluation = {'label': f'affine_action_{key}', 'expected_seeds': [0, 1, 2],
                      'restore_epoch': 500000, 'num_episodes': 500, 'tasks': tasks,
                      'reports': [{'seed': seed, 'task': task,
                                   'path': str(campaign / 'reports' / f'{key}_sd{seed}_task{task}_500000.json')}
                                  for seed in (0, 1, 2) for task in range(1, 6)]}
        (campaign / f'{key}_evaluation_manifest.json').write_text(json.dumps(evaluation, indent=2) + '\n')
    print(f'Snapshot: {destination}; {len(hashes)} source files hashed; dependencies use {source / ".venv"}')


if __name__ == '__main__':
    main()

"""Archive finished experiment checkpoint dirs to a Hugging Face dataset repo.

Why: /var/local is a 24 GB node-local partition. On 2026-08-31 it hit 100% and killed six
Stage-C runs mid-training with `OSError: [Errno 28] No space left on device`. The runs that
filled it are finished July experiments whose numbers are already in docs/tables/results.md
-- worth keeping, not worth keeping on that partition.

Upload only. Deletion is a separate, deliberate step: run with --verify after uploading and
delete locally only once it reports OK for a directory.

  export HF_TOKEN=hf_...        # or: hf auth login
  .venv/bin/python scripts/archive_checkpoints_hf.py --repo-id amsks/psmflows-checkpoint-archive
  .venv/bin/python scripts/archive_checkpoints_hf.py --repo-id ... --verify

Layout in the repo: exp/<run_group>/<seed_dir>/... mirroring the local tree, plus
MANIFEST.json recording every file's size so --verify is a real check and not a file count.
"""
import argparse
import json
import os
import sys

SRC = '/var/local/amsks/exp/PSMFLows'

#: Finished July/August experiment groups. NOT the bcflow_* Stage-A checkpoints (still
#: loaded by every Stage-C run and published separately with the preimages), NOT the
#: preimage npz files, NOT anything still training.
ARCHIVE = [
    'psm_cube_match_ortho1000_20260706_205705',
    'psm_multiseed1M_flow_ortho1000_20260708_175041',
    'fb_recover500k_flow_ortho1000_20260715_133600',
    'fb_recover500k_flow_20260715_115738',
    'affine_fixed_zeroshot_20260726_183729',
    'affine_ref1024_20260726_133733',
    'psm_recover500k_flow_ortho1000_20260713_160709',
    'psm_protoxplant_flow_ortho1000_20260707_184559',
    'psm_recover1M_flow_ortho1000_s2_20260713_174415',
]


def local_manifest(names):
    """{repo_path: size} for every file under each named directory."""
    out = {}
    for name in names:
        root = os.path.join(SRC, name)
        if not os.path.isdir(root):
            print(f'  SKIP (absent locally): {name}')
            continue
        for dirpath, _, files in os.walk(root):
            for f in files:
                p = os.path.join(dirpath, f)
                out[f'exp/{os.path.relpath(p, SRC)}'] = os.path.getsize(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo-id', required=True)
    ap.add_argument('--only', nargs='*', default=None, help='subset of ARCHIVE names')
    ap.add_argument('--verify', action='store_true', help='compare remote against local, no upload')
    ap.add_argument('--public', action='store_true', help='default is private, like the preimages repo')
    args = ap.parse_args()

    from huggingface_hub import HfApi
    api = HfApi()
    names = args.only or ARCHIVE
    man = local_manifest(names)
    total = sum(man.values())
    print(f'{len(man)} files, {total / 2**30:.2f} GiB, across {len(names)} directories')

    if args.verify:
        remote = {s.path: s.size for s in api.list_repo_tree(
            args.repo_id, repo_type='dataset', recursive=True) if getattr(s, 'size', None) is not None}
        bad = 0
        for name in names:
            sub = {k: v for k, v in man.items() if k.startswith(f'exp/{name}/')}
            missing = [k for k in sub if k not in remote]
            wrong = [k for k in sub if k in remote and remote[k] != sub[k]]
            if missing or wrong:
                bad += 1
                print(f'  FAIL {name}: {len(missing)} missing, {len(wrong)} size mismatch')
                for k in (missing + wrong)[:5]:
                    print(f'        {k}')
            else:
                print(f'  OK   {name}: {len(sub)} files, {sum(sub.values()) / 2**30:.2f} GiB '
                      f'-- safe to delete locally')
        sys.exit(1 if bad else 0)

    api.create_repo(args.repo_id, repo_type='dataset', private=not args.public, exist_ok=True)
    with open('/tmp/_archive_manifest.json', 'w') as f:
        json.dump({'source_host_path': SRC, 'files': man}, f, indent=1)
    api.upload_file(path_or_fileobj='/tmp/_archive_manifest.json', path_in_repo='MANIFEST.json',
                    repo_id=args.repo_id, repo_type='dataset')
    for name in names:
        root = os.path.join(SRC, name)
        if not os.path.isdir(root):
            continue
        n = len([k for k in man if k.startswith(f'exp/{name}/')])
        print(f'uploading {name} ({n} files) ...', flush=True)
        api.upload_folder(folder_path=root, path_in_repo=f'exp/{name}',
                          repo_id=args.repo_id, repo_type='dataset')
    print(f'done -> https://huggingface.co/datasets/{args.repo_id}')
    print('now re-run with --verify before deleting anything locally')


if __name__ == '__main__':
    main()

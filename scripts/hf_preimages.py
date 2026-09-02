"""Move preimage artifacts between machines through a Hugging Face dataset repo.

`scripts/upload_preimages_hf.py` publishes the three canonical environments from
hardcoded local paths. This one is the transport for everything else: an arbitrary npz
under an arbitrary name, so preimages generated on a SLURM cluster can be pulled here and
vice versa. Same repo, same layout, so the two coexist:

    preimages/<name>.npz            the preimage-augmented dataset
    preimages/<name>.npz.meta.json  provenance sidecar
    flow/<flow-name>/params_*.pkl   the frozen Stage-A flow the latents invert
    flow/<flow-name>/flags.json     that flow's training config

Names are free-form. Use the inversion settings when a variant is not the canonical one,
e.g. `cube-single-play-a20p6-ps0p69-ns12-N200`, so two files that differ only in alpha
cannot be confused after download.

    export HF_TOKEN=hf_...                     # or: hf auth login
    python scripts/hf_preimages.py list
    python scripts/hf_preimages.py push --npz preimages_x.npz --name cube-single-play-x \
        --flow-dir /path/to/bcflow/sd000 --flow-epoch 500000 --flow-name cube-single-play
    python scripts/hf_preimages.py pull --name cube-single-play-x --dest $PSM_DATA --with-flow

WHY `pull` REWRITES THE SIDECAR. Each npz records the ABSOLUTE path of the checkpoint it
was inverted from, on the machine that produced it, and `main.py` refuses to pair an npz
with a checkpoint that resolves elsewhere. `pull --with-flow` downloads the flow and
repoints `restore_path` at the local copy; that repair is what makes a downloaded artifact
usable, not a convenience.

The flow checkpoint is not optional baggage either: latents are only meaningful for the
exact flow that produced them.
"""
import argparse
import json
import os
import sys

DEFAULT_REPO = os.environ.get('HF_REPO', 'amsks/psmflows-preimages')


def _api():
    from huggingface_hub import HfApi
    return HfApi()


def cmd_list(args):
    api = _api()
    rows = [(f.path, getattr(f, 'size', None))
            for f in api.list_repo_tree(args.repo, repo_type='dataset', recursive=True)]
    npz = sorted(p for p, _ in rows if p.startswith('preimages/') and p.endswith('.npz'))
    flows = sorted({p.split('/')[1] for p, _ in rows if p.startswith('flow/')})
    size = {p: s for p, s in rows}
    print(f'{args.repo}\n\npreimages:')
    for p in npz:
        print(f'  {size.get(p, 0) / 2**20:8.1f} MiB  {p[len("preimages/"):-len(".npz")]}')
    print('flows:')
    for f in flows:
        print(f'  {f}')


def cmd_push(args):
    api = _api()
    npz = os.path.abspath(args.npz)
    meta_path = npz + '.meta.json'
    for p in (npz, meta_path):
        if not os.path.exists(p):
            sys.exit(f'missing {p}')
    # The sidecar is what makes a downloaded npz traceable; refuse to publish a file whose
    # provenance we cannot state.
    with open(meta_path) as f:
        meta = json.load(f)
    for k in ('env_name', 'restore_path', 'restore_epoch', 'inversion'):
        if k not in meta:
            sys.exit(f'sidecar {meta_path} has no `{k}` -- not publishable')

    flow_name = args.flow_name or args.name
    api.create_repo(args.repo, repo_type='dataset', private=True, exist_ok=True)

    uploads = [(npz, f'preimages/{args.name}.npz'),
               (meta_path, f'preimages/{args.name}.npz.meta.json')]
    if args.flow_dir:
        epoch = args.flow_epoch
        for src, dst in ((f'{args.flow_dir}/params_{epoch}.pkl',
                          f'flow/{flow_name}/params_{epoch}.pkl'),
                         (f'{args.flow_dir}/flags.json', f'flow/{flow_name}/flags.json')):
            if not os.path.exists(src):
                sys.exit(f'missing {src}')
            uploads.append((src, dst))

    total = sum(os.path.getsize(s) for s, _ in uploads)
    print(f'uploading {len(uploads)} files, {total / 2**30:.2f} GiB -> {args.repo}')
    print(f'  env {meta["env_name"]}  inversion {meta["inversion"]}')
    if args.dry_run:
        for s, d in uploads:
            print(f'  DRY {os.path.getsize(s) / 2**20:8.1f} MiB  {s} -> {d}')
        return
    for s, d in uploads:
        print(f'  {d} ...', flush=True)
        api.upload_file(path_or_fileobj=s, path_in_repo=d,
                        repo_id=args.repo, repo_type='dataset')
    print(f'done -> https://huggingface.co/datasets/{args.repo}')
    print(f'pull it with: python scripts/hf_preimages.py pull --name {args.name} '
          f'--dest $PSM_DATA --with-flow')


def cmd_pull(args):
    from huggingface_hub import hf_hub_download
    dest = os.path.abspath(args.dest)
    os.makedirs(os.path.join(dest, 'preimages'), exist_ok=True)

    def grab(path_in_repo):
        return hf_hub_download(args.repo, path_in_repo, repo_type='dataset',
                               local_dir=dest)

    npz = grab(f'preimages/{args.name}.npz')
    meta_path = grab(f'preimages/{args.name}.npz.meta.json')
    with open(meta_path) as f:
        meta = json.load(f)
    print(f'{args.name}: env={meta["env_name"]} epoch={meta["restore_epoch"]}')
    print(f'  inversion {meta["inversion"]}')

    if args.with_flow:
        flow_name = args.flow_name or args.name
        epoch = meta['restore_epoch']
        grab(f'flow/{flow_name}/flags.json')
        ckpt = grab(f'flow/{flow_name}/params_{epoch}.pkl')
        flow_dir = os.path.dirname(ckpt)
        # THE PAIRING REPAIR. The sidecar names the producing machine's absolute path;
        # main.py compares realpaths and refuses a mismatch. Point it at this copy.
        meta['restore_path'] = flow_dir
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        print(f'  sidecar restore_path -> {flow_dir}')
        print(f'\ntrain with:\n  agent.preimage_path={npz} \\\n'
              f'  agent.flow_ckpt_path={flow_dir} agent.flow_ckpt_epoch={epoch}')
    else:
        print(f'\nnpz at {npz}\n  NOTE: sidecar still points at the producing machine '
              f'({meta["restore_path"]}). Re-run with --with-flow, or fix restore_path '
              f'by hand, or main.py will refuse the pairing.')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--repo', default=DEFAULT_REPO)
    sub = ap.add_subparsers(dest='cmd', required=True)

    sub.add_parser('list').set_defaults(func=cmd_list)

    p = sub.add_parser('push')
    p.add_argument('--npz', required=True, help='local npz; its .meta.json must sit beside it')
    p.add_argument('--name', required=True, help='name in the repo, e.g. cube-single-play-a20p6')
    p.add_argument('--flow-dir', help='Stage-A run dir to publish alongside (recommended)')
    p.add_argument('--flow-epoch', type=int, default=500000)
    p.add_argument('--flow-name', help='flow/<name>/ in the repo; defaults to --name')
    p.add_argument('--dry-run', action='store_true')
    p.set_defaults(func=cmd_push)

    p = sub.add_parser('pull')
    p.add_argument('--name', required=True)
    p.add_argument('--dest', required=True, help='e.g. $PSM_DATA')
    p.add_argument('--with-flow', action='store_true',
                   help='also fetch the flow and repair the sidecar (needed to train)')
    p.add_argument('--flow-name')
    p.set_defaults(func=cmd_pull)

    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()

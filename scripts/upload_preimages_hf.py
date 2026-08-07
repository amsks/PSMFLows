"""Upload the precomputed noise preimages (Stage B) to a Hugging Face dataset repo.

Uploads, per environment, the artifacts needed to run Stage C without re-inverting the
flow (~19 h on antmaze):

  preimages/<env>.npz            preimage-augmented dataset (Stage C trains on this)
  preimages/<env>.npz.meta.json  provenance sidecar (env, Stage-A ckpt, inversion config)
  flow/<env>/params_500000.pkl   the FROZEN Stage-A behaviour flow the latents invert
  flow/<env>/flags.json          the Stage-A run config

The flow checkpoint is not optional: latents are only meaningful for the exact flow that
produced them, and main.py refuses to pair an npz with a different checkpoint.

Usage:
  export HF_TOKEN=hf_...            # or: hf auth login
  .venv/bin/python scripts/upload_preimages_hf.py --repo-id <user>/<name> --private
  .venv/bin/python scripts/upload_preimages_hf.py --repo-id <user>/<name> --dry-run

Re-running is safe: files already present with the same hash are skipped by the hub.
"""
import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = '/data-local/amsks/PSMFLows'
EXP = '/var/local/amsks/exp/PSMFLows'

#: name -> (npz path, Stage-A run dir, ckpt epoch)
ARTIFACTS = {
    'pointmaze-medium-navigate': (
        f'{DATA}/preimages_pointmaze_medium_a20_n200.npz',
        f'{EXP}/bcflow_pointmaze-medium-navigate_20260729_142219/sd000_20260729_142225',
        500000,
    ),
    'cube-single-play': (
        f'{DATA}/preimages_cube_single_a20_n200.npz',
        f'{EXP}/bcflow_cube_single_20260726_135032/sd000_20260726_135037',
        500000,
    ),
    'antmaze-medium-navigate': (
        f'{DATA}/preimages_antmaze_medium_a20_n200.npz',
        f'{EXP}/bcflow_antmaze-medium-navigate_20260805_014546/sd000_20260805_014548',
        500000,
    ),
}


def plan():
    """(local_path, path_in_repo) pairs, verified to exist."""
    items, missing = [], []
    for name, (npz, ckpt_dir, epoch) in ARTIFACTS.items():
        pairs = [
            (npz, f'preimages/{name}.npz'),
            (npz + '.meta.json', f'preimages/{name}.npz.meta.json'),
            (f'{ckpt_dir}/params_{epoch}.pkl', f'flow/{name}/params_{epoch}.pkl'),
            (f'{ckpt_dir}/flags.json', f'flow/{name}/flags.json'),
        ]
        for src, dst in pairs:
            (items if os.path.exists(src) else missing).append((src, dst))
    return items, missing


def card(repo_id):
    """Dataset card. Numbers are read off the npz files at upload time, not hardcoded."""
    import numpy as np

    rows = []
    for name, (npz, _, _) in ARTIFACTS.items():
        if not os.path.exists(npz):
            continue
        with np.load(npz) as z:
            u = z['noise_preimage_point']
            ess = z['preimage_ess']
            valid = z['preimage_valid']
            rt = z['preimage_roundtrip']
        d_a = u.shape[1]
        rows.append(
            f'| `{name}` | {u.shape[0]:,} | {d_a} | {float((u ** 2).sum(1).mean()):.2f} '
            f'({d_a}) | {float(ess.mean()):.1f} | {float((ess > 20).mean()):.2f} | '
            f'{float(np.median(rt)):.1e} | {int((valid < 0.5).sum())} |'
        )

    meta = {}
    for name, (npz, _, _) in ARTIFACTS.items():
        p = npz + '.meta.json'
        if os.path.exists(p):
            with open(p) as f:
                meta[name] = json.load(f)['inversion']

    return f'''---
license: mit
tags:
- reinforcement-learning
- offline-rl
- flow-matching
---

# PSMFlows noise preimages

Precomputed latents for the PSMFlows pipeline
([code](https://github.com/amsks/PSMFLows)). A behaviour-cloned conditional flow
`G(s, u)` maps Gaussian noise to dataset actions; these files store, for every transition
of an [OGBench](https://github.com/seohongpark/ogbench) dataset, the noise `u` that
decodes to the recorded action. Computing them takes 4-19 h on a GPU per environment;
downloading them takes minutes.

## Contents

```
preimages/<env>.npz              preimage-augmented dataset (training input)
preimages/<env>.npz.meta.json    provenance: env, Stage-A checkpoint, inversion config
flow/<env>/params_500000.pkl     the frozen behaviour flow the latents invert
flow/<env>/flags.json            the behaviour-flow training config
```

Each `.npz` holds the OGBench transitions (`observations`, `actions`, `rewards`,
`terminals`, `masks`, `next_observations`) plus:

| key | shape | what it is |
|---|---|---|
| `noise_preimage_point` | `(N, d_a)` | backward-ODE point preimage `u*`, `G(s, u*) ~ a` |
| `noise_preimage_mean` | `(N, K, d_a)` | Gaussian mixture posterior over `u`, means |
| `noise_preimage_cov` | `(N, K, d_a, d_a)` | mixture covariances |
| `noise_preimage_weights` | `(N, K)` | mixture weights |
| `preimage_ess` | `(N,)` | effective sample size of the final EM iterate (of 200) |
| `preimage_roundtrip` | `(N,)` | `\\|G(s, u*) - a\\|`, the inversion residual |
| `preimage_valid` | `(N,)` | 1.0 if every preimage product for the row is finite |

## Quality

| env | rows | `d_a` | mean `\\|u*\\|^2` (expected) | mean ESS | frac ESS>20 | median round-trip | invalid rows |
|---|---:|---:|---|---:|---:|---:|---:|
{chr(10).join(rows)}

The point preimages are healthy in every environment: `\\|u*\\|^2` sits near its expected
value `d_a` under the standard normal prior, and the round-trip residual is ~1e-4.

**The antmaze mixture posterior is not usable.** Its mean ESS is far below the rest and
only a few percent of rows clear ESS > 20; `alpha=20` was tuned at `d_a=5` and does not
transfer to `d_a=8`. Use `agent.use_point_preimage=true` there (which is what our own
antmaze runs use), or re-run the inversion with a re-tuned `alpha`.

## Inversion settings

```json
{json.dumps(meta, indent=2)}
```

`n_initial_steps` must equal the flow's `flow_steps` and both must be >= 100: the implicit
Euler inverse diverges at the training default of 10.

## Use

```python
from huggingface_hub import hf_hub_download

npz = hf_hub_download('{repo_id}', 'preimages/cube-single-play.npz', repo_type='dataset')
```

Then train against the matching flow checkpoint; see
[`docs/PREIMAGES.md`](https://github.com/amsks/PSMFLows/blob/main/docs/PREIMAGES.md).

## Provenance

The transition arrays are copied from OGBench datasets and are redistributed here only so
that a `.npz` is a drop-in training input; OGBench is the original source. The preimage
arrays and flow checkpoints are ours.
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo-id', required=True, help='e.g. amsks/psmflows-preimages')
    vis = ap.add_mutually_exclusive_group()
    vis.add_argument('--private', dest='private', action='store_true', default=True)
    vis.add_argument('--public', dest='private', action='store_false')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    items, missing = plan()
    total = sum(os.path.getsize(s) for s, _ in items)
    for src, dst in items:
        print(f'  {os.path.getsize(src) / 1e6:9.1f} MB  {dst}')
    for src, dst in missing:
        print(f'  MISSING              {dst}  <- {src}')
    print(f'\n{len(items)} files, {total / 1e9:.2f} GB -> {args.repo_id} '
          f'({"private" if args.private else "PUBLIC"})')
    if missing:
        sys.exit('refusing to upload a partial set; fix the missing paths above')
    if args.dry_run:
        return

    from huggingface_hub import HfApi
    api = HfApi()
    who = api.whoami()
    print(f'authenticated as {who["name"]}')
    api.create_repo(args.repo_id, repo_type='dataset', private=args.private, exist_ok=True)

    readme = os.path.join(REPO, 'scratch_dataset_card.md')
    with open(readme, 'w') as f:
        f.write(card(args.repo_id))
    api.upload_file(path_or_fileobj=readme, path_in_repo='README.md',
                    repo_id=args.repo_id, repo_type='dataset')
    os.remove(readme)

    for i, (src, dst) in enumerate(items, 1):
        print(f'[{i}/{len(items)}] {dst} ({os.path.getsize(src) / 1e6:.0f} MB)', flush=True)
        api.upload_file(path_or_fileobj=src, path_in_repo=dst,
                        repo_id=args.repo_id, repo_type='dataset')
    print(f'done: https://huggingface.co/datasets/{args.repo_id}')


if __name__ == '__main__':
    main()

# Archived experiment checkpoints (Hugging Face)

**Repo:** https://huggingface.co/datasets/amsks/psmflows-checkpoint-archive (private)

Fetch one back:

```bash
hf auth login   # or export HF_TOKEN=hf_...
hf download amsks/psmflows-checkpoint-archive --repo-type dataset \
    --include 'exp/<run_group>/*' --local-dir /data-local/amsks/PSMFLows/restored
```

## Why these moved

`/var/local` is a 24 GB node-local partition. On 2026-08-31 it reached 100% and killed six
Stage-C runs mid-training with `OSError: [Errno 28] No space left on device` (Arm A and
Arm B of the interface-fork plan, at 100k-400k of 500k steps). The 11.8 GB below is
finished July experiments whose numbers already live in `docs/tables/results.md`. They were
uploaded, verified file-by-file against `MANIFEST.json` (path and byte size), and only then
deleted locally.

Stage-C runs now write to `/data-local` (3.5 TB): `scripts/launch_psmflow.sh` takes
`STORE=/data-local/amsks/PSMFLows`, and its default stays `/var/local` so older launch
lines still reproduce.

## What is archived

| run group | size |
|---|---|
| `psm_cube_match_ortho1000_20260706_205705` | 4.8 G |
| `psm_multiseed1M_flow_ortho1000_20260708_175041` | 2.0 G |
| `fb_recover500k_flow_ortho1000_20260715_133600` | 1.1 G |
| `fb_recover500k_flow_20260715_115738` | 1.1 G |
| `affine_fixed_zeroshot_20260726_183729` | 946 M |
| `affine_ref1024_20260726_133733` | 776 M |
| `psm_recover500k_flow_ortho1000_20260713_160709` | 503 M |
| `psm_protoxplant_flow_ortho1000_20260707_184559` | 503 M |
| `psm_recover1M_flow_ortho1000_s2_20260713_174415` | 336 M |

179 files, 11.82 GiB.

## What deliberately did NOT move

- `bcflow_*` — the frozen Stage-A behaviour flows. Every Stage-C run loads one at startup,
  and they are already published alongside the latents in `amsks/psmflows-preimages`.
- `preimages_*.npz` — the Stage B latents, published in that same repo.
- `psmflow_paperfaith_arm{A,B}_20260831` — the 250k checkpoints salvaged from the runs the
  disk-full killed. Small, and kept until the relaunched runs finish.

Tooling: `scripts/archive_checkpoints_hf.py` (upload, then `--verify` before deleting).

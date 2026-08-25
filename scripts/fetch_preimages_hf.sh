#!/usr/bin/env bash
# Download the published Stage-A flow checkpoints and preimage npz files from Hugging Face
# into $PSM_DATA, and repair their provenance sidecars for this machine.
#
# Usage:
#   PSM_DATA=/path/for/big/files bash scripts/fetch_preimages_hf.sh [cube|antmaze|pointmaze|all]
#
# The repo is private, so authenticate first (`hf auth login`, or export HF_TOKEN=hf_...).
# Re-running is cheap: the hub skips files already present with the same hash.
#
# The sidecar repair is not optional. Each npz records the ABSOLUTE path of the checkpoint
# it was inverted from, on the machine that produced it, and main.py (and both recovery
# tools) refuse to pair an npz with a checkpoint that resolves elsewhere. Rewriting it to
# $PSM_DATA/flow/<name> is what makes the published artifacts usable here.
set -euo pipefail

WHICH="${1:-all}"
: "${PSM_DATA:?PSM_DATA=<dir for the ~1.1 GB of artifacts> required}"
# Exported, not just set: both are read by the python heredocs below.
export PSM_DATA
export HF_REPO="${HF_REPO:-amsks/psmflows-preimages}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-$REPO/.venv/bin/python}"
[ -x "$PY" ] || PY=python

case "$WHICH" in
  cube)      NAMES=(cube-single-play) ;;
  antmaze)   NAMES=(antmaze-medium-navigate) ;;
  pointmaze) NAMES=(pointmaze-medium-navigate) ;;
  all)       NAMES=(cube-single-play antmaze-medium-navigate pointmaze-medium-navigate) ;;
  *) echo "unknown target: $WHICH (cube|antmaze|pointmaze|all)" >&2; exit 1 ;;
esac

mkdir -p "$PSM_DATA"
echo "fetching ${NAMES[*]} from $HF_REPO -> $PSM_DATA"

PSM_NAMES="${NAMES[*]}" "$PY" - <<'EOF'
import os
from huggingface_hub import snapshot_download

names = os.environ['PSM_NAMES'].split()
snapshot_download(
    os.environ['HF_REPO'],
    repo_type='dataset',
    local_dir=os.environ['PSM_DATA'],
    allow_patterns=[f'*{name}*' for name in names],
)
EOF

# Point every sidecar at this machine's copy of the checkpoint, and report what the
# artifacts claim about themselves so a mismatch is visible before a tool asserts on it.
PSM_NAMES="${NAMES[*]}" "$PY" - <<'EOF'
import json, os

data = os.environ['PSM_DATA']
for name in os.environ['PSM_NAMES'].split():
    meta_path = f'{data}/preimages/{name}.npz.meta.json'
    if not os.path.exists(meta_path):
        raise SystemExit(f'missing {meta_path} -- did the download cover {name}?')
    with open(meta_path) as f:
        meta = json.load(f)
    meta['restore_path'] = f'{data}/flow/{name}'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    ckpt = f"{data}/flow/{name}/params_{meta['restore_epoch']}.pkl"
    if not os.path.exists(ckpt):
        raise SystemExit(f'sidecar names epoch {meta["restore_epoch"]} but {ckpt} is missing')
    print(f'{name}: env={meta["env_name"]} epoch={meta["restore_epoch"]} '
          f'flow_steps={meta["flow_steps"]} rows={meta.get("num_transitions")} '
          f'invalid={meta.get("num_invalid_preimages")} -> ok')
EOF

echo "done. artifacts under $PSM_DATA/{preimages,flow}"

#!/usr/bin/env bash
# scripts/dev/smoke_resume.sh — end-to-end crash-recovery resume smoke for train.py.
# Runs a tiny job to its first checkpoint, then resumes and asserts it continues
# to completion at the right step. Covers state by default; pass DOMAIN/AGENT for
# the pixel path. Needs the offline dataset at DATA_PATH and a GPU.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-$REPO/.venv/bin/python}"
DATA_PATH="${DATA_PATH:-/dev/shm/factored-fb/datasets}"
DOMAIN="${DOMAIN:-cube_single}"
AGENT="${AGENT:-fb_flowbc}"
NAME="smoke_resume_$$"
SAVE_DIR="$REPO/outputs/$NAME/checkpoints"

common=( "$PY" train.py agent="$AGENT" domain="$DOMAIN" data_path="$DATA_PATH"
         use_wandb=false eval_every=0 log_every=20 save_every=40
         load_n_episodes=20 wandb_run_name="$NAME" seed=0 )

echo "[smoke] phase 1: fresh run to 60 steps (checkpoints at 40)"
"${common[@]}" num_train_steps=60
test -f "$SAVE_DIR/step_40.pt" || { echo "FAIL: no step_40.pt"; exit 1; }
test -f "$SAVE_DIR/train_state.json" || { echo "FAIL: no sidecar"; exit 1; }
"$PY" - "$SAVE_DIR" <<'PY'
import sys
from resume import read_train_state
ts = read_train_state(sys.argv[1]); assert ts and ts["step"] == 40, ts
print("[smoke] sidecar OK:", ts)
PY

echo "[smoke] phase 2: resume to 120 steps"
"${common[@]}" num_train_steps=120 resume=true
test -f "$SAVE_DIR/step_120.pt" || { echo "FAIL: resume did not reach step_120.pt"; exit 1; }
test -f "$SAVE_DIR/final.pt" || { echo "FAIL: no final.pt after resume"; exit 1; }
echo "[smoke] PASS ($AGENT / $DOMAIN)  save_dir=$SAVE_DIR"

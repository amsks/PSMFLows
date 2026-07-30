#!/usr/bin/env bash
# Stage C. PSMFlow representation training: learns phi / psi against the FROZEN Stage-A
# behaviour flow, using precomputed noise preimages as the behavioural index measure
# (psmflow plan Tasks 3-6).
#
# Requires BOTH Stage-A outputs to exist:
#   FLOW_CKPT   Stage-A run dir (restore_agent glob convention), from
#               scripts/pretrain_behavior_flow.sh
#   FLOW_EPOCH  the epoch of the Stage-A ckpt to load, e.g. 500000
#   PREIMAGES   the npz from tools/precompute_preimages.py, computed from THAT ckpt
#               (main.py asserts the npz row count matches the dataset, which catches
#               the wrong-env / stale-file case)
#
# Usage: FLOW_CKPT=<dir> FLOW_EPOCH=<n> PREIMAGES=<npz> \
#          bash scripts/launch_psmflow.sh ENV [GPU] [STEPS] [WANDB_MODE]
#   SEEDS="0 1 2"  space-separated seeds; EXTRA= extra hydra overrides
#
# NOTE (one seed per GPU): invoke once per GPU with SEEDS="0", SEEDS="1", … rather than
# letting several seeds share one device (HANDOFF infra lesson).
set -euo pipefail

ENV="${1:?env_name required, e.g. cube-single-play-singletask-v0}"
GPU="${2:-1}"
STEPS="${3:-500000}"
WMODE="${4:-online}"
: "${FLOW_CKPT:?FLOW_CKPT=<stage-A run dir glob> required}"
: "${FLOW_EPOCH:?FLOW_EPOCH=<stage-A ckpt epoch> required}"
: "${PREIMAGES:?PREIMAGES=<preimages.npz> required}"
read -r -a SEEDS <<< "${SEEDS:-0 1 2}"
EVAL_INT="${EVAL_INT:-50000}"
SAVE_INT="${SAVE_INT:-250000}"
MEM_FRAC="${MEM_FRAC:-0.30}"
EXTRA="${EXTRA:-}"
GROUP="${GROUP:-psmflow_${ENV%%-singletask*}_$(date +%Y%m%d_%H%M%S)}"

REPO=/u/amsks/git/PSMFLows
LOCAL=/var/local/amsks
PY="$REPO/.venv/bin/python"     # system `python` is 2.7 on midi-01
mkdir -p "$LOCAL/exp" "$LOCAL/wandb" "$LOCAL/logs/$GROUP"

# Fail here rather than 40 seconds into JIT if the preimage file is missing/mistyped.
[ -f "$PREIMAGES" ] || { echo "no such preimage npz: $PREIMAGES" >&2; exit 1; }

cd "$REPO"
export CUDA_VISIBLE_DEVICES="$GPU"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="$MEM_FRAC"
export WANDB_MODE="$WMODE"
export WANDB_DIR="$LOCAL/wandb"
export WANDB_ENTITY=amsks
export OGBENCH_DATASET_DIR="$LOCAL/ogbench"
export MUJOCO_GL=egl            # midi-01 has no DISPLAY; pointmaze builds a renderer

for seed in "${SEEDS[@]}"; do
  log="$LOCAL/logs/$GROUP/sd${seed}.log"
  echo "launching psmflow sd${seed} on $ENV -> $log"
  nohup "$PY" main.py \
    agent=psmflow \
    agent.flow_ckpt_path="$FLOW_CKPT" agent.flow_ckpt_epoch="$FLOW_EPOCH" \
    agent.preimage_path="$PREIMAGES" $EXTRA \
    env_name="$ENV" \
    offline_steps="$STEPS" online_steps=0 \
    log_interval=5000 eval_interval="$EVAL_INT" eval_episodes=50 \
    save_interval="$SAVE_INT" save_dir="$LOCAL/exp" \
    run_group="$GROUP" seed="$seed" \
    > "$log" 2>&1 &
  sleep 8   # stagger JIT compiles so they don't collide on startup
done

echo "wandb group: $GROUP"
echo "logs: $LOCAL/logs/$GROUP/"
wait

#!/usr/bin/env bash
# Pretrain the reward-free BEHAVIOUR FLOW G_theta(s,u) that PSMFlows freezes and indexes
# policies by (psmflow plan Task 1). This is FQL with bc_only=true: no critic, no Q term,
# rewards/masks never read — what remains is the conditional-flow-matching objective plus
# its one-step distillation.
#
# The checkpoint this writes is the input to the preimage pipeline
# (tools/precompute_preimages.py) and to diagnostics D1/D3
# (tools/validate_flow_inversion.py).
#
# Usage: bash scripts/pretrain_behavior_flow.sh [GPU] [STEPS] [ENV]
set -euo pipefail

GPU="${1:-1}"
STEPS="${2:-500000}"
ENV="${3:-cube-single-play-singletask-v0}"
SEED="${SEED:-0}"
EXTRA="${EXTRA:-}"
GROUP="${GROUP:-bcflow_$(echo "$ENV" | cut -d- -f1-3)_$(date +%Y%m%d_%H%M%S)}"

REPO=/u/amsks/git/PSMFLows
LOCAL=/var/local/amsks
PY="$REPO/.venv/bin/python"     # system `python` is 2.7 on midi-01
mkdir -p "$LOCAL/exp" "$LOCAL/wandb" "$LOCAL/logs/$GROUP"

cd "$REPO"
export CUDA_VISIBLE_DEVICES="$GPU"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.25
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_DIR="$LOCAL/wandb"
export WANDB_ENTITY=amsks
export OGBENCH_DATASET_DIR="$LOCAL/ogbench"

log="$LOCAL/logs/$GROUP/bcflow_sd${SEED}.log"
echo "pretraining behaviour flow on $ENV, GPU $GPU, $STEPS steps -> $log"

# eval_interval=0 disables eval: task success is meaningless for a reward-free BC flow.
# save_interval == STEPS so exactly one checkpoint lands, at the end.
nohup "$PY" main.py \
  agent=fql agent.bc_only=true $EXTRA \
  env_name="$ENV" \
  offline_steps="$STEPS" online_steps=0 \
  log_interval=5000 eval_interval=0 \
  save_interval="$STEPS" save_dir="$LOCAL/exp" \
  run_group="$GROUP" seed="$SEED" \
  > "$log" 2>&1 &

echo "wandb group: $GROUP"
echo "checkpoint will be: $LOCAL/exp/PSMFLows/$GROUP/sd$(printf '%03d' "$SEED")_*/params_${STEPS}.pkl"
wait

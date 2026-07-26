#!/usr/bin/env bash
# Launch affine-PSM cube-single runs, ONE SEED PER GPU (the HANDOFF infra lesson: seeds
# sharing a GPU run ~2x slower). Data/exp/wandb go to node-local /var/local to stay off
# the NFS home quota.
#
# Usage: bash scripts/launch_affine_psm_cube.sh [GPUS] [STEPS] [WANDB_MODE]
#   GPUS   comma-separated, one per seed, e.g. "0,1,3"
# Env:
#   SEEDS="0 1 2"   ACTOR=flow|ddpgbc   EXTRA="agent.ortho_coef=1000 ..."
set -euo pipefail

IFS=',' read -r -a GPUS <<< "${1:-0,1,3}"
STEPS="${2:-500000}"
WMODE="${3:-online}"           # online | offline
read -r -a SEEDS <<< "${SEEDS:-0 1 2}"
ACTOR="${ACTOR:-flow}"         # cube/manipulation default; see configs/agent/affine_psm.yaml
EXTRA="${EXTRA:-}"
EVAL_INT="${EVAL_INT:-50000}"
SAVE_INT="${SAVE_INT:-250000}"
ENV=cube-single-play-singletask-v0
GROUP="${GROUP:-affine_psm_${ACTOR}_$(date +%Y%m%d_%H%M%S)}"

REPO=/u/amsks/git/PSMFLows
LOCAL=/var/local/amsks
PY="$REPO/.venv/bin/python"    # system `python` is 2.7 on midi-01 — never use it
mkdir -p "$LOCAL/exp" "$LOCAL/wandb" "$LOCAL/logs/$GROUP"

cd "$REPO"
export XLA_PYTHON_CLIENT_PREALLOCATE=false   # share the GPU politely (grow-as-needed)
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.30   # hard per-process ceiling
export WANDB_MODE="$WMODE"
export WANDB_DIR="$LOCAL/wandb"
export WANDB_ENTITY=amsks
export OGBENCH_DATASET_DIR="$LOCAL/ogbench"

if [ "${#SEEDS[@]}" -gt "${#GPUS[@]}" ]; then
  echo "error: ${#SEEDS[@]} seeds but only ${#GPUS[@]} GPUs; one seed per GPU." >&2
  exit 1
fi

for i in "${!SEEDS[@]}"; do
  seed="${SEEDS[$i]}"
  gpu="${GPUS[$i]}"
  name="${ACTOR}_sd${seed}"
  log="$LOCAL/logs/$GROUP/${name}.log"
  echo "launching $name on GPU $gpu -> $log"
  CUDA_VISIBLE_DEVICES="$gpu" nohup "$PY" main.py \
    agent=affine_psm agent.actor.type="$ACTOR" $EXTRA \
    env_name="$ENV" \
    offline_steps="$STEPS" online_steps=0 \
    log_interval=5000 eval_interval="$EVAL_INT" eval_episodes=50 \
    save_interval="$SAVE_INT" save_dir="$LOCAL/exp" \
    run_group="$GROUP" seed="$seed" \
    > "$log" 2>&1 &
  sleep 8   # stagger JIT compiles so they don't collide on startup
done

echo "all launched under wandb group: $GROUP"
echo "logs: $LOCAL/logs/$GROUP/"
wait

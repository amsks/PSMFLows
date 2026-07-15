#!/usr/bin/env bash
# Launch PSM cube-single runs: N seeds x {flow, ddpgbc}, all on one GPU (shared box).
# Data/exp/wandb go to node-local /var/local to stay off the NFS home quota.
# Usage: bash scripts/launch_psm_cube.sh [GPU] [STEPS] [WANDB_MODE]
set -euo pipefail

GPU="${1:-1}"
STEPS="${2:-500000}"
WMODE="${3:-online}"           # online | offline
# space-separated seeds via SEEDS env; extra hydra overrides via EXTRA env.
read -r -a SEEDS <<< "${SEEDS:-0 1 2}"
read -r -a ACTORS <<< "${ACTORS:-flow ddpgbc}"
EXTRA="${EXTRA:-}"             # e.g. "agent.ortho_coef=1000 agent.z_dim=50"
EVAL_INT="${EVAL_INT:-100000}"
SAVE_INT="${SAVE_INT:-500000}"
ENV=cube-single-play-singletask-v0
GROUP="${GROUP:-psm_cube_single_$(date +%Y%m%d_%H%M%S)}"

REPO=/u/amsks/git/PSMFLows
LOCAL=/var/local/amsks
mkdir -p "$LOCAL/exp" "$LOCAL/wandb" "$LOCAL/logs/$GROUP"

cd "$REPO"
export CUDA_VISIBLE_DEVICES="$GPU"
export XLA_PYTHON_CLIENT_PREALLOCATE=false   # share the GPU politely (grow-as-needed)
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.30   # hard per-process ceiling
export WANDB_MODE="$WMODE"
export WANDB_DIR="$LOCAL/wandb"
export WANDB_ENTITY=amsks
export OGBENCH_DATASET_DIR="$LOCAL/ogbench"

for actor in "${ACTORS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    name="${actor}_sd${seed}"
    log="$LOCAL/logs/$GROUP/${name}.log"
    echo "launching $name -> $log"
    nohup python main.py \
      agent=psm agent.actor.type="$actor" $EXTRA \
      env_name="$ENV" \
      offline_steps="$STEPS" online_steps=0 \
      log_interval=5000 eval_interval="$EVAL_INT" eval_episodes=50 \
      save_interval="$SAVE_INT" save_dir="$LOCAL/exp" \
      run_group="$GROUP" seed="$seed" \
      > "$log" 2>&1 &
    sleep 8   # stagger JIT compiles so they don't collide on startup
  done
done

echo "all launched under wandb group: $GROUP"
echo "logs: $LOCAL/logs/$GROUP/"
wait

#!/usr/bin/env bash
# Launch FB cube-single runs: N seeds x {flow, td3}, one seed per GPU (shared box).
# Data/exp/wandb go to node-local /var/local to stay off the NFS home quota.
# Usage: bash scripts/launch_fb_cube.sh [GPU] [STEPS] [WANDB_MODE]
#   SEEDS="0 1 2" ACTORS="flow" EXTRA="ortho_coef=1.0" bash scripts/launch_fb_cube.sh 1 1000000
set -euo pipefail

GPU="${1:-1}"
STEPS="${2:-1000000}"
WMODE="${3:-online}"           # online | offline
read -r -a SEEDS <<< "${SEEDS:-0 1 2}"
read -r -a ACTORS <<< "${ACTORS:-flow}"   # flow (fb_flowbc, parity path) | td3
EXTRA="${EXTRA:-}"             # extra hydra overrides, e.g. "agent.lr_b=1e-5"
EVAL_INT="${EVAL_INT:-100000}"
SAVE_INT="${SAVE_INT:-100000}"            # save every 100k so we can transplant/re-eval
ENV=cube-single-play-singletask-v0
GROUP="${GROUP:-fb_cube_single_$(date +%Y%m%d_%H%M%S)}"

REPO=/u/amsks/git/PSMFLows
LOCAL=/var/local/amsks
mkdir -p "$LOCAL/exp" "$LOCAL/wandb" "$LOCAL/logs/$GROUP"

cd "$REPO"
export CUDA_VISIBLE_DEVICES="$GPU"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.30   # hard per-process ceiling (one seed per GPU)
export MUJOCO_GL=egl
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
      agent=fb agent.actor.type="$actor" $EXTRA \
      env_name="$ENV" \
      offline_steps="$STEPS" online_steps=0 \
      log_interval=5000 eval_interval="$EVAL_INT" eval_episodes=10 \
      save_interval="$SAVE_INT" save_dir="$LOCAL/exp" \
      run_group="$GROUP" seed="$seed" \
      > "$log" 2>&1 &
    sleep 8   # stagger JIT compiles
  done
done

echo "all launched under wandb group: $GROUP"
echo "logs: $LOCAL/logs/$GROUP/"
wait

#!/usr/bin/env bash
# Unconditional runs of the critic-diagnosis plan (docs/plans/2026-08-10-critic-diagnosis-and-baselines.md §D).
#
#   R1  antmaze LatentFlowPSM seeds 1,2   -> turns the 1-seed 0.222 into a 3-seed mean
#   R2  FB zero-shot on cube, 3 seeds     -> peer baseline
#   R3  PSM (action-space) on cube, 3 seeds -> the honest peer; LatentFlowPSM may not beat it
#
# Three sequential per-GPU queues rather than eight concurrent jobs: two training seeds
# sharing a device run about twice as slow each, so queueing finishes the batch sooner and
# keeps every run's wall-clock comparable.
#
# Usage: bash scripts/launch_baselines.sh          (launches all three queues)
#        QUEUE=0 bash scripts/launch_baselines.sh  (just one queue)
set -euo pipefail

REPO=/u/amsks/git/PSMFLows
PY="$REPO/.venv/bin/python"          # bare `python` is 2.7 on midi-01
DATA=/data-local/amsks/PSMFLows
EXPF=/var/local/amsks/exp/PSMFLows
LOGS=$DATA/logs
STEPS="${STEPS:-500000}"
STAMP=20260810

mkdir -p "$LOGS"
cd "$REPO"

env_for() {
  echo "CUDA_VISIBLE_DEVICES=$1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.32 MUJOCO_GL=egl \
OGBENCH_DATASET_DIR=/var/local/amsks/ogbench WANDB_MODE=offline"
}

# ---- R1: antmaze LatentFlowPSM, same recipe as sd0 (its flags.json is ground truth) ----
antmaze_run() {
  local gpu=$1 seed=$2
  echo "$(env_for "$gpu") $PY main.py agent=psmflow \
env_name=antmaze-medium-navigate-singletask-v0 \
agent.flow_ckpt_path=$EXPF/bcflow_antmaze-medium-navigate_20260805_014546/sd000_20260805_014548 \
agent.flow_ckpt_epoch=500000 \
agent.preimage_path=$DATA/preimages_antmaze_medium_a20_n200.npz \
agent.use_point_preimage=true \
offline_steps=$STEPS eval_interval=50000 eval_episodes=50 save_interval=50000 \
save_dir=$DATA/exp run_group=psmflow_latentpsm_antmaze seed=$seed \
2>&1 | tee $LOGS/latentpsm_antmaze_sd$seed.log; "
}

# ---- R2: FB. ortho_coef=1000 is the reference cube value; the repo default is 1.0. ----
fb_run() {
  local gpu=$1 seed=$2
  echo "$(env_for "$gpu") $PY main.py agent=fb agent.actor.type=flow agent.ortho_coef=1000 \
env_name=cube-single-play-singletask-v0 \
offline_steps=$STEPS online_steps=0 log_interval=5000 eval_interval=50000 eval_episodes=50 \
save_interval=$STEPS save_dir=$DATA/exp run_group=fb_cube_ortho1000_$STAMP seed=$seed \
2>&1 | tee $LOGS/fb_cube_sd$seed.log; "
}

# ---- R3: PSM, the audited recipe (z_dim=128, ortho_coef=1000, lr_phi=1e-5 are config
# defaults already); the flow actor is the parity path. ----
psm_run() {
  local gpu=$1 seed=$2
  echo "$(env_for "$gpu") $PY main.py agent=psm agent.actor.type=flow \
env_name=cube-single-play-singletask-v0 \
offline_steps=$STEPS online_steps=0 log_interval=5000 eval_interval=50000 eval_episodes=50 \
save_interval=$STEPS save_dir=$DATA/exp run_group=psm_cube_flow_$STAMP seed=$seed \
2>&1 | tee $LOGS/psm_cube_sd$seed.log; "
}

start_queue() {
  local name=$1 cmd=$2
  tmux new-session -d -s "$name" "$cmd echo '=== QUEUE $name DONE ==='"
  echo "launched queue $name"
}

Q="${QUEUE:-all}"

if [ "$Q" = "all" ] || [ "$Q" = "0" ]; then
  start_queue base_q0 "$(psm_run 0 0)$(psm_run 0 1)$(psm_run 0 2)"
fi
if [ "$Q" = "all" ] || [ "$Q" = "1" ]; then
  start_queue base_q1 "$(antmaze_run 1 1)$(fb_run 1 0)$(fb_run 1 1)"
fi
if [ "$Q" = "all" ] || [ "$Q" = "3" ]; then
  start_queue base_q3 "$(antmaze_run 3 2)$(fb_run 3 2)"
fi

sleep 5
tmux ls | grep base_q || true

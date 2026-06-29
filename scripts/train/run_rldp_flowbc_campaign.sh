#!/usr/bin/env bash
# scripts/train/run_rldp_flowbc_campaign.sh — RLDP+FlowBC large-run campaign (state).
#
# Launches 10 seeds of RLDP+FlowBC on cube-single state across 8 GPUs
# (seed i -> GPU i % nGPU; 10 seeds / 8 GPUs => 1-2 jobs/GPU). Full TRAIN_STEPS
# (default 1M), wandb online, checkpoints to /dev/shm. Thin wrapper around
# scripts/agents/rldp_flowbc/state.sh — sets 10-seed defaults and forwards.
#
# Run inside tmux so it survives SSH disconnect:
#   tmux new -s rldp; bash scripts/train/run_rldp_flowbc_campaign.sh 2>&1 | tee /dev/shm/rldp_flowbc_campaign.log
#
#   DRY_RUN=1 bash scripts/train/run_rldp_flowbc_campaign.sh   # print plan, launch nothing
#   SEEDS="0 1 2" GPUS="0 1" bash scripts/train/run_rldp_flowbc_campaign.sh   # override
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
export GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
export TRAIN_STEPS="${TRAIN_STEPS:-1000000}"
export RUN_GROUP="${RUN_GROUP:-rldp_flowbc_state_10seeds}"
export DATA_PATH="${DATA_PATH:-/dev/shm/factored-fb/datasets}"
export DRY_RUN="${DRY_RUN:-0}"

echo "[rldp-campaign] agent=rldp_flowbc  steps=$TRAIN_STEPS  seeds=[$SEEDS]  gpus=[$GPUS]"
echo "[rldp-campaign] group=$RUN_GROUP  data=$DATA_PATH"
echo "[rldp-campaign] wandb -> amsks/factored-fb (group=$RUN_GROUP)"
echo ""

exec bash "$REPO/scripts/agents/rldp_flowbc/state.sh"

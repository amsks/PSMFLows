#!/usr/bin/env bash
# scripts/train/run_grid_sweep.sh — generic 2-knob grid sweep for ANY train.py agent.
# Generalizes run_sweep.sh (which hardcodes FB's ortho_coef/lr_b) to arbitrary
# hydra knob names, so PSM (ortho_coef x lr_phi) and RLDP (ortho_coef x lr_b)
# reuse one scheduler. Round-robins jobs across GPUs; run inside tmux.
#
# Env knobs:
#   AGENT=psm_flowbc | rldp_flowbc | fb_flowbc   (hydra `agent=`)
#   DOMAIN=cube_single
#   KNOB_A_NAME=ortho_coef   KNOB_A_VALS="0.1 1 10"
#   KNOB_B_NAME=lr_phi|lr_b  KNOB_B_VALS="1e-4 1e-5"
#   SEEDS="0 1 2"            TRAIN_STEPS=500000
#   N_GPUS=8  JOBS_PER_GPU=3
#   RUN_GROUP_BASE=psm_state_sweep
#   DATA_PATH=/dev/shm/factored-fb/datasets
#   WANDB_ENTITY=amsks
#   DRY_RUN=1                (print commands, launch nothing)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$REPO/logs/grid_$TS"
# Absolute venv interpreter (mirrors scripts/agents/_common.sh): a tmux/cron
# non-login shell has no activated env, so bare `python` is not on PATH.
PY="${PYTHON:-$REPO/.venv/bin/python}"

AGENT="${AGENT:?set AGENT}"
DOMAIN="${DOMAIN:-cube_single}"
KNOB_A_NAME="${KNOB_A_NAME:?set KNOB_A_NAME}"; KNOB_A_VALS="${KNOB_A_VALS:?set KNOB_A_VALS}"
KNOB_B_NAME="${KNOB_B_NAME:?set KNOB_B_NAME}"; KNOB_B_VALS="${KNOB_B_VALS:?set KNOB_B_VALS}"
SEEDS="${SEEDS:-0 1 2}"
TRAIN_STEPS="${TRAIN_STEPS:-500000}"
N_GPUS="${N_GPUS:-8}"; JOBS_PER_GPU="${JOBS_PER_GPU:-3}"
N_PAR=$(( N_GPUS * JOBS_PER_GPU ))
RUN_GROUP_BASE="${RUN_GROUP_BASE:-${AGENT}_${DOMAIN}_sweep}"
DATA_PATH="${DATA_PATH:-/dev/shm/factored-fb/datasets}"
WANDB_ENTITY="${WANDB_ENTITY:-amsks}"
DRY_RUN="${DRY_RUN:-0}"
RESUME="${RESUME:-0}"            # 1 => append resume=true to every job (relaunch-continue)

[[ "$DRY_RUN" == "1" ]] || mkdir -p "$LOG_DIR"
_tag() { echo "${1//./p}"; }   # 1e-4 -> 1e-4 ; 0.1 -> 0p1 (filesystem-safe)

declare -a CMDS LOGS NAMES
for a in $KNOB_A_VALS; do
  for b in $KNOB_B_VALS; do
    for s in $SEEDS; do
      grp="${RUN_GROUP_BASE}__${KNOB_A_NAME}$(_tag "$a")__${KNOB_B_NAME}$(_tag "$b")"
      name="${grp}__s${s}"
      log="$LOG_DIR/${name}.log"
      cmd="$PY train.py agent=$AGENT domain=$DOMAIN \
$KNOB_A_NAME=$a $KNOB_B_NAME=$b seed=$s \
num_train_steps=$TRAIN_STEPS data_path=$DATA_PATH \
use_wandb=true wandb_entity=$WANDB_ENTITY wandb_group=$grp wandb_run_name=$name \
wandb_tags=[grid-$TS] save_eval_videos=false"
      [[ "$RESUME" == "1" ]] && cmd="$cmd resume=true"
      CMDS+=("$cmd"); LOGS+=("$log"); NAMES+=("$name")
    done
  done
done

N=${#CMDS[@]}
echo "[grid] AGENT=$AGENT  $N jobs  grid ${KNOB_A_NAME}{$KNOB_A_VALS} x ${KNOB_B_NAME}{$KNOB_B_VALS} x seeds{$SEEDS}"
echo "[grid] $N_GPUS GPUs x $JOBS_PER_GPU/GPU = $N_PAR parallel  steps=$TRAIN_STEPS  logs -> $LOG_DIR"

if [[ "$DRY_RUN" == "1" ]]; then
  for i in "${!CMDS[@]}"; do
    gpu=$(( i % N_GPUS ))
    echo "GPU$gpu  ${NAMES[$i]}"
    echo "   CUDA_VISIBLE_DEVICES=$gpu ${CMDS[$i]}"
  done
  exit 0
fi

declare -a SLOT_PIDS=(); for ((s=0;s<N_PAR;s++)); do SLOT_PIDS+=(""); done
declare -a ACTIVE_PIDS=()
find_free_slot() {
  while true; do
    for ((s=0;s<N_PAR;s++)); do
      pid="${SLOT_PIDS[$s]}"
      if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then echo "$s"; return; fi
    done
    sleep 3
  done
}
for i in "${!CMDS[@]}"; do
  slot=$(find_free_slot); gpu=$(( slot % N_GPUS ))
  echo "[grid] $(date +%T) launch ${NAMES[$i]} on GPU$gpu (slot $slot)"
  ( cd "$REPO"; CUDA_VISIBLE_DEVICES="$gpu" ${CMDS[$i]} > "${LOGS[$i]}" 2>&1 && rc=0 || rc=$?; \
    echo "[grid] $(date +%T) DONE ${NAMES[$i]} rc=$rc"; exit $rc ) &
  SLOT_PIDS[$slot]="$!"
  ACTIVE_PIDS+=("$!")
done
fail=0
for pid in "${ACTIVE_PIDS[@]}"; do
  wait "$pid" && st=0 || st=$?
  [[ $st -ne 0 ]] && { echo "[grid] WARN pid=$pid exited $st"; fail=$((fail+1)); }
done
echo "[grid] all done -> $LOG_DIR ($fail job(s) exited nonzero)"

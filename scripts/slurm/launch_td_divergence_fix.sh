#!/usr/bin/env bash
# 2026-09-07 TD-divergence fix: three arms x three seeds on antmaze at discount 0.995,
# the fastest-failing setting (docs/design/2026-09-07-td-divergence-fix.md).
#
#   F1 orel    agent.ortho_mode=relative   -- the geometry regulariser tracks the TD term
#   F2 pbound  agent.psi_bound=tanh        -- bounded reparameterisation of the psi head
#   F3 both    both switches
#
# One GPU per seed, 500k steps, checkpoints every 50k, 6 h wall (the unfixed g995 runs
# took 4.0 h). SMOKE=1 runs the 200-step wiring check instead.
#
#   PSM_REPO=... PSM_DATA=... bash scripts/slurm/launch_td_divergence_fix.sh [arm ...]
set -euo pipefail

PSM_REPO="${PSM_REPO:-/mnt/home/amohan/git/Austin/PSMFLows}"
PSM_DATA="${PSM_DATA:-/mnt/home/amohan/psm-data}"
PREIMAGES="${PREIMAGES:-$PSM_DATA/preimages/antmaze-medium-navigate.npz}"
SEEDS="${SEEDS:-0 1 2}"
PART="${PART:-kisski-inference}"
ACCT="${ACCT:-general}"
WALL="${WALL:-06:00:00}"
LOGDIR="$PSM_DATA/slurm_logs/tdfix"

F1="agent.ortho_mode=relative agent.ortho_rel_coef=1.0"
F2="agent.psi_bound=tanh agent.psi_bound_scale=200.0"
F3="$F1 $F2"

if [ "${SMOKE:-0}" = "1" ]; then
  STEPS=200; EVAL_INT=0; LOG_INT=50; SAVE_INT=200; WALL=00:40:00; TAG=smoke_tdfix
else
  STEPS=500000; EVAL_INT=50000; LOG_INT=5000; SAVE_INT=50000; TAG=affine_strict_antmaze_g995
fi

mkdir -p "$LOGDIR"
ARMS=("${@:-orel pbound both}")
for arm in ${ARMS[@]}; do
  case "$arm" in
    orel)   EXTRA="$F1" ;;
    pbound) EXTRA="$F2" ;;
    both)   EXTRA="$F3" ;;
    *) echo "unknown arm: $arm (orel|pbound|both)" >&2; exit 1 ;;
  esac
  if [ "${SMOKE:-0}" = "1" ]; then GROUP="${TAG}_${arm}"; else GROUP="${TAG}_${arm}"; fi
  for s in $SEEDS; do
    sbatch --partition="$PART" --account="$ACCT" --gres=gpu:1 \
      --job-name="tdfix-$arm-s$s" --time="$WALL" \
      --output="$LOGDIR/${GROUP}_sd${s}-%j.out" \
      --error="$LOGDIR/${GROUP}_sd${s}-%j.out" \
      --export=ALL,ENVKEY=antmaze,PREIMAGES="$PREIMAGES",SEED="$s",GROUP="$GROUP",DISCOUNT=0.995,STEPS="$STEPS",EVAL_INT="$EVAL_INT",LOG_INT="$LOG_INT",SAVE_INT="$SAVE_INT",EXTRA="$EXTRA" \
      "$PSM_REPO/scripts/slurm/train_psmflow.sbatch"
  done
done

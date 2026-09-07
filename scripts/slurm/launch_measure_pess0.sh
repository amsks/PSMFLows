#!/usr/bin/env bash
# 2026-09-07 measure-target pessimism off, Q pessimism kept: the arrangement PSM, Meta Motivo
# and td_jepa ship (docs/design/2026-09-08-measure-loss-audit.md Sec. 8 answer 3/4).
#
#   agent.pessimism_penalty=0.0        target_M = ensemble MEAN (was exact min at 0.5)
#   agent.actor_pessimism_penalty=0.5  unchanged: GPI acting Q = mean - |Q1-Q2|/2 = min
#
# antmaze at the launcher default 0.99 and cube at 0.98, three seeds, 500k, checkpoints
# every 50k for the ladder. A 200-step smoke per env is submitted first and the real jobs
# are chained on it with afterok, so a wiring failure cancels them.
#
#   PSM_REPO=... PSM_DATA=... bash scripts/slurm/launch_measure_pess0.sh [cube antmaze]
set -euo pipefail

PSM_REPO="${PSM_REPO:-/mnt/home/amohan/git/Austin/PSMFLows}"
PSM_DATA="${PSM_DATA:-/mnt/home/amohan/psm-data}"
SEEDS="${SEEDS:-0 1 2}"
PART="${PART:-kisski-inference}"
ACCT="${ACCT:-general}"
WALL="${WALL:-06:00:00}"
LOGDIR="$PSM_DATA/slurm_logs/mpess0"
EXTRA="agent.pessimism_penalty=0.0"
COMMON="PSM_REPO=$PSM_REPO,PSM_DATA=$PSM_DATA,OGBENCH_DATASET_DIR=${OGBENCH_DATASET_DIR:-/mnt/home/amohan/.ogbench/data}"

mkdir -p "$LOGDIR"
ENVS=("$@"); [ ${#ENVS[@]} -eq 0 ] && ENVS=(cube antmaze)
for env in "${ENVS[@]}"; do
  case "$env" in
    cube)    PRE="$PSM_DATA/preimages/cube-single-play.npz";        GROUP=affine_strict_cube_mpess0 ;;
    antmaze) PRE="$PSM_DATA/preimages/antmaze-medium-navigate.npz"; GROUP=affine_strict_antmaze_g99_mpess0 ;;
    *) echo "unknown env: $env" >&2; exit 1 ;;
  esac
  SMOKE=$(sbatch --parsable --partition="$PART" --account="$ACCT" --gres=gpu:1 \
    --job-name="mpess0_smoke_$env" --time=00:40:00 \
    --output="$LOGDIR/smoke_${env}-%j.out" --error="$LOGDIR/smoke_${env}-%j.out" \
    --export=ALL,$COMMON,ENVKEY=$env,PREIMAGES="$PRE",SEED=0,GROUP="smoke_${GROUP}",STEPS=200,EVAL_INT=0,LOG_INT=50,SAVE_INT=200,EXTRA="$EXTRA" \
    "$PSM_REPO/scripts/slurm/train_psmflow.sbatch")
  echo "smoke $env -> $SMOKE"
  for s in $SEEDS; do
    J=$(sbatch --parsable --partition="$PART" --account="$ACCT" --gres=gpu:1 \
      --dependency=afterok:$SMOKE --kill-on-invalid-dep=yes \
      --job-name="mpess0_${env}_s$s" --time="$WALL" \
      --output="$LOGDIR/${GROUP}_sd${s}-%j.out" --error="$LOGDIR/${GROUP}_sd${s}-%j.out" \
      --export=ALL,$COMMON,ENVKEY=$env,PREIMAGES="$PRE",SEED=$s,GROUP="$GROUP",STEPS=500000,EVAL_INT=50000,EVAL_EPS=50,LOG_INT=5000,SAVE_INT=50000,EXTRA="$EXTRA" \
      "$PSM_REPO/scripts/slurm/train_psmflow.sbatch")
    echo "  $env seed $s -> $J (afterok:$SMOKE)"
  done
done

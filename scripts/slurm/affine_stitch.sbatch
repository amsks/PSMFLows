#!/usr/bin/env bash
# sbatch [resources] affine_stitch.sbatch smoke|train SEED CAMPAIGN CODE [RUN_DIR]
#SBATCH --job-name=affine-stitch
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
set -euo pipefail
PHASE="${1:?smoke|train}"
SEED="${2:?seed}"
CAMPAIGN="${3:?absolute campaign path}"
CODE="${4:?absolute immutable snapshot path}"
RUN_DIR="${5:-$CAMPAIGN/$PHASE/seed$SEED}"
PY="${PSM_PYTHON:-/mnt/home/amohan/git/Austin/PSMFLows/.venv/bin/python}"
case "$PHASE" in
  smoke) STEPS=200; EVAL_INTERVAL=200; SAVE_INTERVAL=200; LOG_INTERVAL=50; FINAL_EPISODES=2 ;;
  train) STEPS=500000; EVAL_INTERVAL=100000; SAVE_INTERVAL=100000; LOG_INTERVAL=1000; FINAL_EPISODES=500 ;;
  *) echo 'phase must be smoke or train' >&2; exit 2 ;;
esac
case "$SEED" in 0|1|2) ;; *) echo 'seed must be0,1,2' >&2; exit 2 ;; esac
export PYTHONPATH="$CODE"
export PYTHONDONTWRITEBYTECODE=1
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=8
export WANDB_MODE=disabled
cd "$CODE"
srun "$PY" -m tools.affine_stitch.run train \
  --data-dir "$CAMPAIGN/data" --run-dir "$RUN_DIR" --seed "$SEED" \
  --proto-mode bounded --steps "$STEPS" \
  --eval-interval "$EVAL_INTERVAL" --save-interval "$SAVE_INTERVAL" \
  --log-interval "$LOG_INTERVAL" --eval-episodes 10 \
  --final-eval-episodes "$FINAL_EPISODES" --eval-seed 0 --eval-workers 4

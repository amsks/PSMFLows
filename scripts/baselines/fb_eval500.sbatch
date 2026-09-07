#!/usr/bin/env bash
#SBATCH --partition=kisski-inference
#SBATCH --account=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/home/amohan/psm-data/logs/slurm/%x_%j.out
#SBATCH --error=/mnt/home/amohan/psm-data/logs/slurm/%x_%j.err
#
# 500-episode ladder over one Factored-FB run's checkpoints, in PSMFlows' report format.
# Knobs: RUN_DIR SEED EPOCHS OUT_PREFIX CRITIC ACTOR [EXTRA_ARGS]
set -uo pipefail

FB_REPO=${FB_REPO:-/mnt/home/amohan/git/Austin/Factored-FB}
PSM_REPO=${PSM_REPO:-/mnt/home/amohan/git/Austin/PSMFLows}
PY=$FB_REPO/.venv/bin/python

export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OGBENCH_DATASET_DIR=/mnt/home/amohan/.ogbench/data
export WANDB_MODE=disabled
export PYTHONPATH=$FB_REPO
export PYTHONUNBUFFERED=1
export FB_REPO

RUN_DIR=${RUN_DIR:?}; SEED=${SEED:?}; OUT_PREFIX=${OUT_PREFIX:?}
EPOCHS=${EPOCHS:-"50000 100000 150000 200000 250000 300000 350000 400000 450000 500000"}
CRITIC=${CRITIC:-fb}; ACTOR=${ACTOR:-ddpgbc}; EXTRA_ARGS=${EXTRA_ARGS:-}

cd "$FB_REPO"
fail=0
for E in $EPOCHS; do
  [ -f "$RUN_DIR/params_$E.pkl" ] || { echo "MISSING $RUN_DIR/params_$E.pkl -- skipped"; continue; }
  K=$((E / 1000))
  OUT="${OUT_PREFIX}_${K}k_sd${SEED}.json"
  echo "=== epoch $E -> $OUT ==="
  "$PY" "$PSM_REPO/scripts/baselines/fb_eval500.py" \
      --critic "$CRITIC" --actor "$ACTOR" --domain cube_single --seed "$SEED" \
      --restore_path "$RUN_DIR" --restore_epoch "$E" \
      --task_id 1 --eval_episodes 500 --out "$OUT" $EXTRA_ARGS || { echo "FAILED epoch $E"; fail=1; }
done
exit $fail

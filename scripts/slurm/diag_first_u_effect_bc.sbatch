#!/usr/bin/env bash
#SBATCH --partition=kisski-inference
#SBATCH --account=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=first_u_bc
#SBATCH --output=/mnt/home/amohan/psm-data/logs/slurm/%x-%j.out
#SBATCH --error=/mnt/home/amohan/psm-data/logs/slurm/%x-%j.err
# tools/diag_first_u_effect_bc.py for one task: K candidate first latents x R rollouts
# under BC continuation, from anchors reached by a BC prefix of each length in PREFIX_LENS.
# Needs TASK (1..5). Optional: ENVKEY (cube | antmaze, default cube), PREFIX_LENS (default per
# env: cube 0,50,100; antmaze 0,250,500), NUM_U, NUM_ROLLOUTS, SEED, SMOKE=1, OUT.
#
#   for t in 1 2 3 4 5; do
#     sbatch --job-name=first_u_bc_t$t --export=ALL,TASK=$t scripts/slurm/diag_first_u_effect_bc.sbatch
#   done
# Pool afterwards:
#   .venv/bin/python tools/diag_first_u_effect_bc.py \
#     --aggregate $PSM_DATA/logs/diag_first_u_effect_bc_task{1,2,3,4,5}.json
#
# antmaze (limit 1000 steps, so pass a longer --time than the 2 h default):
#   for t in 1 2 3 4 5; do
#     sbatch --time=04:00:00 --job-name=first_u_bc_antmaze_t$t --export=ALL,ENVKEY=antmaze,TASK=$t \
#       scripts/slurm/diag_first_u_effect_bc.sbatch
#   done
#   .venv/bin/python tools/diag_first_u_effect_bc.py --env antmaze \
#     --aggregate $PSM_DATA/logs/diag_first_u_effect_bc_antmaze_task{1,2,3,4,5}.json
set -euo pipefail
export PSM_DATA=/mnt/home/amohan/psm-data
export PSM_REPO=/mnt/home/amohan/git/Austin/PSMFLows
export OGBENCH_DATASET_DIR=/mnt/home/amohan/.ogbench/data
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
cd "$PSM_REPO"

: "${TASK:?TASK=1..5}"
ENVKEY="${ENVKEY:-cube}"
case "$ENVKEY" in
  cube)    FLOW=cube-single-play;        PREFIX_LENS="${PREFIX_LENS:-0,50,100}" ;;
  antmaze) FLOW=antmaze-medium-navigate; PREFIX_LENS="${PREFIX_LENS:-0,250,500}" ;;
  *) echo "unknown ENVKEY: $ENVKEY" >&2; exit 1 ;;
esac
NUM_U="${NUM_U:-8}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-100}"
SEED="${SEED:-0}"
ARGS=(--env "$ENVKEY" --task "$TASK" --prefix_lens "$PREFIX_LENS" --num_u "$NUM_U" --num_rollouts "$NUM_ROLLOUTS"
      --seed "$SEED" --flow "$PSM_DATA/flow/$FLOW" --flow_epoch 500000)
[ "${SMOKE:-0}" = "1" ] && ARGS+=(--smoke)
[ -n "${OUT:-}" ] && ARGS+=(--out "$OUT")

echo "=== first-u effect under BC: env=$ENVKEY task=$TASK prefix_lens=$PREFIX_LENS K=$NUM_U R=$NUM_ROLLOUTS seed=$SEED smoke=${SMOKE:-0}"
.venv/bin/python tools/diag_first_u_effect_bc.py "${ARGS[@]}"

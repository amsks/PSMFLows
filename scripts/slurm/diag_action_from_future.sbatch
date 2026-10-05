#!/usr/bin/env bash
#SBATCH --partition=kisski-inference
#SBATCH --account=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --job-name=act_from_future
#SBATCH --output=/mnt/home/amohan/psm-data/logs/slurm/%x-%j.out
#SBATCH --error=/mnt/home/amohan/psm-data/logs/slurm/%x-%j.err
# tools/diag_action_from_future.py for one env: how well a state k steps ahead identifies the
# data latent at s among 64 candidates, from the dataset alone (no flow checkpoint, no simulator).
# Needs ENVKEY (cube | antmaze). Optional: STEPS (default 100000), SEED, SMOKE=1, OUT.
#
#   for e in cube antmaze; do
#     sbatch --job-name=act_from_future_$e --export=ALL,ENVKEY=$e scripts/slurm/diag_action_from_future.sbatch
#   done
# Figure afterwards (CPU):
#   .venv/bin/python tools/diag_action_from_future.py --figure \
#     $PSM_DATA/logs/diag_action_from_future_cube-single-play.json \
#     $PSM_DATA/logs/diag_action_from_future_antmaze-medium-navigate.json
set -euo pipefail
export PSM_DATA=/mnt/home/amohan/psm-data
export PSM_REPO=/mnt/home/amohan/git/Austin/PSMFLows
export OGBENCH_DATASET_DIR=/mnt/home/amohan/.ogbench/data
export XLA_PYTHON_CLIENT_PREALLOCATE=false
cd "$PSM_REPO"

: "${ENVKEY:?ENVKEY=cube|antmaze}"
ARGS=(--env "$ENVKEY" --steps "${STEPS:-100000}" --seed "${SEED:-0}")
[ "${SMOKE:-0}" = "1" ] && ARGS+=(--smoke)
[ -n "${OUT:-}" ] && ARGS+=(--report_out "$OUT")

echo "=== action from future: env=$ENVKEY steps=${STEPS:-100000} seed=${SEED:-0} smoke=${SMOKE:-0}"
.venv/bin/python tools/diag_action_from_future.py "${ARGS[@]}"

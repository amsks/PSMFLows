#!/usr/bin/env bash
# The commitment-horizon sweep (STEP B of the 2026-09-09 ranking post-mortem).
# tools/diag_commitment_horizon.py; docs/design/2026-09-08-critic-signal-and-dsrl-na.md 2.
#
# Aim at a frozen expert for m steps, then hand to behaviour cloning, for m in {1,3,5,10,20,50},
# against a random in-support pick over the same m. Finds the horizon a selection rule has to
# be right over before it beats not selecting at all. Checkpoint-independent: it uses only the
# frozen flow, the frozen expert, the dataset and the simulator.
#
# Submit:
#   sbatch --partition=kisski-inference --account=general \
#     --export=ALL,PSM_REPO=$PWD,PSM_DATA=$PSM_DATA,ENVKEY=cube \
#     scripts/slurm/diag_commitment_horizon.sbatch
#SBATCH --job-name=commithz
#SBATCH --output=/mnt/home/amohan/psm-data/logs/slurm/%x-%j.out
#SBATCH --error=/mnt/home/amohan/psm-data/logs/slurm/%x-%j.err
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
set -euo pipefail

: "${PSM_REPO:?}"; : "${PSM_DATA:?}"
ENVKEY="${ENVKEY:-cube}"
MS="${MS:-1,3,5,10,20,50}"
REPEATS="${REPEATS:-8}"
N_STATES="${N_STATES:-0}"

case "$ENVKEY" in
  cube)      ENV_NAME=cube-single-play-singletask-v0;                FLOW=cube-single-play;        ORACLE_GLOB="fqlexpert_cube_a300/sd000*" ;;
  antmaze)   ENV_NAME=antmaze-medium-navigate-singletask-v0;         FLOW=antmaze-medium-navigate; ORACLE_GLOB="fqlexpert_ant_a10/sd000*" ;;
  *) echo "unknown ENVKEY: $ENVKEY" >&2; exit 1 ;;
esac

REPORT="${REPORT:-$PSM_DATA/logs/diag_gpi_selection_cube_sd000_n256_250000.json}"
ORACLE=$(ls -d "$PSM_DATA/exp/PSMFLows/"$ORACLE_GLOB 2>/dev/null | head -1)
[ -n "$ORACLE" ] || { echo "no frozen expert matching $ORACLE_GLOB" >&2; exit 1; }
[ -f "$REPORT" ]  || { echo "no such report: $REPORT" >&2; exit 1; }
OUT="$PSM_DATA/logs/diag_commitment_horizon_${ENVKEY}.json"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${MEM_FRAC:-0.90}"
export MUJOCO_GL=egl
export OGBENCH_DATASET_DIR="${OGBENCH_DATASET_DIR:-$HOME/.ogbench/data}"

echo "=== commitment horizon: env=$ENVKEY ms=$MS repeats=$REPEATS states=${N_STATES:-all}"
echo "    report=$REPORT"
echo "    oracle=$ORACLE"
echo "    out=$OUT"

cd "$PSM_REPO"
srun .venv/bin/python tools/diag_commitment_horizon.py \
  --report "$REPORT" --oracle "$ORACLE" \
  --flow "$PSM_DATA/flow/$FLOW" --flow_epoch 500000 \
  --env "$ENV_NAME" --ms "$MS" --repeats "$REPEATS" --n_states "$N_STATES" \
  --out "$OUT"

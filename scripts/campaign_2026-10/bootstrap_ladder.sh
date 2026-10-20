#!/usr/bin/env bash
# Submit tools/diag_bootstrap_ladder.py for one run directory over a ladder of checkpoint
# epochs: one SLURM job per epoch (scripts/slurm/diag_bootstrap_ladder.sbatch), skipping epochs
# whose report already exists and epochs whose params_<epoch>.pkl is not on disk yet.
# Reports land at $PSM_DATA/logs/actor_ladder/<TAG>_<epoch>.json; plot them with
#   .venv/bin/python tools/fig_bootstrap_ladder.py --tags <TAG_A> <TAG_B>
#
# Usage:
#   PSM_DATA=/mnt/home/amohan/psm-data bash scripts/campaign_2026-10/bootstrap_ladder.sh <TAG> <RUN_DIR> [EPOCHS...]
#
#   TAG       report prefix (A_sd0, B_sd0, ...)
#   RUN_DIR   the run directory holding flags.json and params_<epoch>.pkl (a glob is accepted as long
#             as it matches exactly one directory)
#   EPOCHS    the checkpoint epochs; default every 25k from 25000 to 500000
#   env: PSM_DATA (required), PARTITION (kisski-inference), ACCOUNT (general), NICE (0),
#        N_STATES / N_DRAWS / SEED / EXTRA passed through to the sbatch, DRY=1 prints only.
set -euo pipefail

TAG="${1:?usage: bootstrap_ladder.sh <TAG> <RUN_DIR> [EPOCHS...]}"
RUN_GLOB="${2:?usage: bootstrap_ladder.sh <TAG> <RUN_DIR> [EPOCHS...]}"
shift 2
: "${PSM_DATA:?PSM_DATA required}"
PSM_REPO="${PSM_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PARTITION="${PARTITION:-kisski-inference}"
ACCOUNT="${ACCOUNT:-general}"
NICE="${NICE:-0}"
OUTD="$PSM_DATA/logs/actor_ladder"
mkdir -p "$OUTD" "$PSM_DATA/logs/slurm"

# shellcheck disable=SC2206  -- the glob is deliberate
RUN_DIRS=($RUN_GLOB)
[ "${#RUN_DIRS[@]}" -eq 1 ] && [ -d "${RUN_DIRS[0]}" ] || { echo "RUN_DIR must match exactly one directory: $RUN_GLOB" >&2; exit 1; }
RUN_DIR="${RUN_DIRS[0]}"
[ -f "$RUN_DIR/flags.json" ] || { echo "no flags.json in $RUN_DIR" >&2; exit 1; }

if [ "$#" -gt 0 ]; then
  EPOCHS=("$@")
else
  EPOCHS=()
  for ((e = 25000; e <= 500000; e += 25000)); do EPOCHS+=("$e"); done
fi

echo "=== bootstrap ladder: tag=$TAG run=$RUN_DIR epochs=${EPOCHS[*]}"
submitted=0
for e in "${EPOCHS[@]}"; do
  out="$OUTD/${TAG}_${e}.json"
  if [ -f "$out" ]; then echo "--- $e: $out exists, skip"; continue; fi
  if [ ! -f "$RUN_DIR/params_${e}.pkl" ]; then echo "--- $e: no params_${e}.pkl yet, skip"; continue; fi
  cmd=(sbatch --partition="$PARTITION" --account="$ACCOUNT" --nice="$NICE" --time=01:00:00
       --job-name="bl_${TAG}_${e}" --output="$PSM_DATA/logs/slurm/%x-%j.out" --error="$PSM_DATA/logs/slurm/%x-%j.err"
       --export=ALL,PSM_REPO="$PSM_REPO",PSM_DATA="$PSM_DATA",RUN_DIR="$RUN_DIR",EPOCH="$e",TAG="$TAG"
       "$PSM_REPO/scripts/slurm/diag_bootstrap_ladder.sbatch")
  if [ "${DRY:-0}" = "1" ]; then echo "--- $e: ${cmd[*]}"; continue; fi
  echo "--- $e: $("${cmd[@]}")"
  submitted=$((submitted + 1))
done
echo "=== submitted $submitted job(s); reports -> $OUTD/${TAG}_<epoch>.json"

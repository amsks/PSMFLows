#!/usr/bin/env bash
# One measure-vs-scalar-Q probe per SLURM job: the affine measure's readout psi^T w
# against a DSRL-NA scalar critic trained on the scaled r_hat from the SAME w
# (tools/diag_measure_vs_scalar_q.py). Eval only, ~11k dataset rows x 64 latents.
#
# Submit:
#   sbatch --partition=kisski-inference --account=general --time=01:30:00 \
#     --job-name=mvsq_cube_sd001 --output=$PSM_DATA/logs/%x-%j.out \
#     --export=ALL,PSM_REPO=<checkout>,ENVKEY=cube,RUN_DIR=<measure run dir>,EPOCH=500000,\
#              NA_DIR=<dsrl-na run dir>,NA_EPOCH=500000,\
#              REWARD_RAW=$PSM_DATA/rewards/<raw>.npz,REWARD_SCALED=$PSM_DATA/rewards/<scaled>.npz,\
#              TAG=cube_sd001 \
#     scripts/slurm/diag_measure_vs_scalar_q.sbatch
#
# Writes $PSM_DATA/logs/diag_measure_vs_scalar_q_<TAG>.json plus a .npz sidecar with the
# per-(state, latent) value tensors.
#SBATCH --job-name=mvsq
# SLURM does NOT expand environment variables in #SBATCH directives, so this is the
# literal $PSM_DATA/logs/slurm on KISSKI rather than the variable.
#SBATCH --output=/mnt/home/amohan/psm-data/logs/slurm/%x-%j.out
#SBATCH --error=/mnt/home/amohan/psm-data/logs/slurm/%x-%j.err
#SBATCH --time=01:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
set -euo pipefail

: "${PSM_REPO:?PSM_REPO=<path to the PSMFLows checkout> required}"
: "${PSM_DATA:?PSM_DATA=<scratch dir with flow/ and preimages/> required}"
: "${ENVKEY:?ENVKEY=cube|antmaze|pointmaze|scene required}"
: "${RUN_DIR:?RUN_DIR=<measure run dir with flags.json and params_*.pkl> required}"
: "${EPOCH:?EPOCH=<measure checkpoint epoch> required}"
: "${NA_DIR:?NA_DIR=<DSRL-NA run dir with flags.json and params_*.pkl> required}"
: "${REWARD_RAW:?REWARD_RAW=<raw r_hat npz with .meta.json> required}"
: "${REWARD_SCALED:?REWARD_SCALED=<scaled r_hat npz with .meta.json> required}"
NA_EPOCH="${NA_EPOCH:-500000}"
TAG="${TAG:-$ENVKEY}"
N_ROWS="${N_ROWS:-10000}"
N_SUCCESS_MIN="${N_SUCCESS_MIN:-1000}"
N_PANEL="${N_PANEL:-64}"
EXTRA="${EXTRA:-}"

case "$ENVKEY" in
  cube)      ENV_NAME=cube-single-play-singletask-v0;                FLOW=cube-single-play ;;
  antmaze)   ENV_NAME=antmaze-medium-navigate-singletask-v0;         FLOW=antmaze-medium-navigate ;;
  pointmaze) ENV_NAME=pointmaze-medium-navigate-singletask-task1-v0; FLOW=pointmaze-medium-navigate ;;
  scene)     ENV_NAME=scene-play-singletask-v0;                     FLOW=scene-play ;;
  *) echo "unknown ENVKEY: $ENVKEY" >&2; exit 1 ;;
esac

FLOW_DIR="$PSM_DATA/flow/$FLOW"
PRE_NPZ="$PSM_DATA/preimages/$FLOW.npz"
LOGS="$PSM_DATA/logs"
OUT="$LOGS/diag_measure_vs_scalar_q_${TAG}.json"
[ -d "$FLOW_DIR" ] || { echo "no such flow dir: $FLOW_DIR" >&2; exit 1; }
[ -f "$PRE_NPZ" ]  || { echo "no such preimage npz: $PRE_NPZ" >&2; exit 1; }
[ -f "$RUN_DIR/params_${EPOCH}.pkl" ] || { echo "no params_${EPOCH}.pkl in $RUN_DIR" >&2; exit 1; }
[ -f "$NA_DIR/params_${NA_EPOCH}.pkl" ] || { echo "no params_${NA_EPOCH}.pkl in $NA_DIR" >&2; exit 1; }
[ -f "$REWARD_RAW.meta.json" ] || { echo "no sidecar for $REWARD_RAW" >&2; exit 1; }
[ -f "$REWARD_SCALED.meta.json" ] || { echo "no sidecar for $REWARD_SCALED" >&2; exit 1; }
mkdir -p "$LOGS"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${MEM_FRAC:-0.90}"
export MUJOCO_GL=egl
export OGBENCH_DATASET_DIR="${OGBENCH_DATASET_DIR:-$HOME/.ogbench/data}"

echo "=== diag_measure_vs_scalar_q: env=$ENVKEY tag=$TAG"
echo "    measure=$RUN_DIR @ $EPOCH"
echo "    scalar=$NA_DIR @ $NA_EPOCH"
echo "    rewards raw=$REWARD_RAW scaled=$REWARD_SCALED"
echo "    n_rows=$N_ROWS n_success_min=$N_SUCCESS_MIN n_panel=$N_PANEL"
echo "    out=$OUT"

cd "$PSM_REPO"
# shellcheck disable=SC2086  -- EXTRA is a deliberate word-split list of hydra overrides
srun .venv/bin/python tools/diag_measure_vs_scalar_q.py agent=psmflow \
  env_name="$ENV_NAME" \
  agent.flow_ckpt_path="$FLOW_DIR" agent.flow_ckpt_epoch=500000 \
  agent.preimage_path="$PRE_NPZ" agent.use_point_preimage=true \
  restore_path="$RUN_DIR" restore_epoch="$EPOCH" \
  +na_path="$NA_DIR" +na_epoch="$NA_EPOCH" \
  +reward_raw_path="$REWARD_RAW" +reward_scaled_path="$REWARD_SCALED" \
  +n_rows="$N_ROWS" +n_success_min="$N_SUCCESS_MIN" +n_panel="$N_PANEL" \
  report_out="$OUT" $EXTRA

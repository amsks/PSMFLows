#!/usr/bin/env bash
# Policy-free basis phi by latent dynamics prediction (RLDP) on a SLURM cluster -- the
# scheduler wrapper around tools/pretrain_rldp_phi.py. One job = one (horizon, seed) run.
#
# The checkpoint lands under $PSM_DATA/exp/PSMFLows/rldp_phi_<ENVKEY>/H<HORIZON>_sd<SEED>/
# as params_<step>.pkl + flags.json + rldp_curve.json, and is consumed by a stage-C run as
#   EXTRA="agent.phi_restore_path=<that dir> agent.phi_restore_epoch=<step> agent.train_phi=false"
#
# Submit:
#   sbatch --partition=kisski-inference --account=general \
#     --export=ALL,ENVKEY=cube,HORIZON=5,SEED=0 scripts/slurm/pretrain_rldp_phi.sbatch
#
# Required in the environment:
#   PSM_REPO    path to this checkout on the cluster
#   PSM_DATA    scratch dir holding preimages/<name>.npz (the training rows) and exp/
#
#SBATCH --job-name=psm-rldp-phi
# SLURM does NOT expand environment variables in #SBATCH directives, so this is the
# literal $PSM_DATA/logs/slurm on KISSKI rather than the variable.
#SBATCH --output=/mnt/home/amohan/psm-data/logs/slurm/%x-%j.out
#SBATCH --error=/mnt/home/amohan/psm-data/logs/slurm/%x-%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
set -euo pipefail

: "${PSM_REPO:?PSM_REPO=<path to the PSMFLows checkout> required}"
: "${PSM_DATA:?PSM_DATA=<scratch dir with preimages/ and exp/> required}"
: "${ENVKEY:?ENVKEY=cube|antmaze|pointmaze|scene required}"
HORIZON="${HORIZON:-5}"
SEED="${SEED:-0}"
STEPS="${STEPS:-1000000}"
BATCH="${BATCH:-1024}"
LAM="${LAM:-1.0}"
SAVE_INT="${SAVE_INT:-250000}"
LOG_INT="${LOG_INT:-5000}"
EXTRA="${EXTRA:-}"

case "$ENVKEY" in
  cube)      NAME=cube-single-play ;;
  antmaze)   NAME=antmaze-medium-navigate ;;
  pointmaze) NAME=pointmaze-medium-navigate ;;
  scene)     NAME=scene-play ;;
  *) echo "unknown ENVKEY: $ENVKEY" >&2; exit 1 ;;
esac

NPZ="$PSM_DATA/preimages/$NAME.npz"
OUT_DIR="${OUT_DIR:-$PSM_DATA/exp/PSMFLows/rldp_phi_${ENVKEY}/H${HORIZON}_sd${SEED}}"
[ -f "$NPZ" ] || { echo "no such preimage npz: $NPZ" >&2; exit 1; }

cd "$PSM_REPO"
PY="${PY:-$PSM_REPO/.venv/bin/python}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${MEM_FRAC:-0.90}"
export MUJOCO_GL=egl
export OGBENCH_DATASET_DIR="${OGBENCH_DATASET_DIR:-$HOME/.ogbench/data}"
mkdir -p "$OUT_DIR"

echo "=== RLDP phi: env=$ENVKEY npz=$NPZ horizon=$HORIZON seed=$SEED steps=$STEPS batch=$BATCH lam=$LAM"
echo "    out=$OUT_DIR extra='$EXTRA'"

# shellcheck disable=SC2086  -- EXTRA is a deliberate word-split list of CLI flags
srun "$PY" tools/pretrain_rldp_phi.py \
  --dataset_npz "$NPZ" --out_dir "$OUT_DIR" \
  --steps "$STEPS" --batch_size "$BATCH" --horizon "$HORIZON" --lam "$LAM" \
  --save_interval "$SAVE_INT" --log_interval "$LOG_INT" --seed "$SEED" $EXTRA

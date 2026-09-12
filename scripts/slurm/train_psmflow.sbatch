#!/usr/bin/env bash
# Stage C on a SLURM cluster. One seed per job, one GPU per job -- the same "never share a
# device between seeds" rule as scripts/launch_psmflow.sh, expressed as separate jobs
# instead of separate CUDA_VISIBLE_DEVICES.
#
# scripts/launch_psmflow.sh is the midi-01 launcher: it hardcodes that machine's REPO and
# /var/local paths and backgrounds seeds with nohup. Neither survives a scheduler, hence
# this file. Same hydra line, same defaults.
#
# Submit (one sbatch per seed):
#   for s in 0 1 2; do
#     sbatch --partition=<part> --account=<acct> \
#       --export=ALL,ENVKEY=antmaze,PREIMAGES=$PSM_DATA/preimages_<name>.npz,SEED=$s,GROUP=<name> \
#       scripts/slurm/train_psmflow.sbatch
#   done
#
# Required in the environment:
#   PSM_REPO    path to this checkout on the cluster
#   PSM_DATA    scratch dir holding flow/<name>/ and the preimage npz
#   PREIMAGES   the npz from tools/precompute_preimages.py, computed from THAT flow ckpt
#
#SBATCH --job-name=psm-stagec
# SLURM does NOT expand environment variables in #SBATCH directives, so this is the
# literal $PSM_DATA/logs/slurm on KISSKI rather than the variable. A relative path here
# lands the log in the submit CWD -- i.e. the repo root. Launchers override both flags.
#SBATCH --output=/mnt/home/amohan/psm-data/logs/slurm/%x-%j.out
#SBATCH --error=/mnt/home/amohan/psm-data/logs/slurm/%x-%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
# Cluster-specific; set on the command line rather than editing.
set -euo pipefail

: "${PSM_REPO:?PSM_REPO=<path to the PSMFLows checkout> required}"
: "${PSM_DATA:?PSM_DATA=<scratch dir with flow/ and the npz> required}"
: "${ENVKEY:?ENVKEY=cube|antmaze|pointmaze|scene required}"
: "${PREIMAGES:?PREIMAGES=<preimage npz> required}"
SEED="${SEED:-0}"
STEPS="${STEPS:-500000}"
EPOCH="${EPOCH:-500000}"
EVAL_INT="${EVAL_INT:-50000}"
EVAL_EPS="${EVAL_EPS:-50}"
LOG_INT="${LOG_INT:-5000}"
SAVE_INT="${SAVE_INT:-250000}"
EXTRA="${EXTRA:-}"
# Which agent to train. `psmflow` is the method (defaults = the paper-strict affine
# LatentFlowPSM: psi_form=affine policy_index=latent train_actor=false acting=gpi, so a
# plain submit runs the primary algorithm and EXTRA is only for ablations). `fql` with
# agent.bc_only=true is Stage A / the BC control. Everything else moved to archive/.
AGENT="${AGENT:-psmflow}"

# Remember whether DISCOUNT came in from the caller's environment (even "") before the
# per-env case block below fills in a default -- ${VAR+set} is unset-safe under `set -u`.
DISCOUNT_USER_SET="${DISCOUNT+set}"

case "$ENVKEY" in
  cube)      ENV_NAME=cube-single-play-singletask-v0;               FLOW=cube-single-play;        DISCOUNT_DEFAULT=0.98 ;;
  # docs/design/2026-09-06-antmaze-failure-tests.md H2: gamma=0.98's 50-step horizon is
  # shorter than antmaze-medium's 200-400 step goal distance and the arm sits at the BC
  # floor; the 51-cell, 3-seed ladder found gamma=0.99 (0.294 +/- 0.070) clearly beats
  # 0.98 (0.076) and beats 0.995, which collapses to ~0.01 past 300k. configs/agent/
  # psmflow.yaml keeps discount=0.98 so existing checkpoints still restore -- this is a
  # launcher-level default, not a config-wide one.
  antmaze)   ENV_NAME=antmaze-medium-navigate-singletask-v0;        FLOW=antmaze-medium-navigate; DISCOUNT_DEFAULT=0.99 ;;
  pointmaze) ENV_NAME=pointmaze-medium-navigate-singletask-task1-v0; FLOW=pointmaze-medium-navigate; DISCOUNT_DEFAULT=0.98 ;;
  scene)     ENV_NAME=scene-play-singletask-v0;                     FLOW=scene-play;              DISCOUNT_DEFAULT=0.98 ;;
  *) echo "unknown ENVKEY: $ENVKEY" >&2; exit 1 ;;
esac

# DISCOUNT (if the caller set it) wins over the per-env default above.
DISCOUNT="${DISCOUNT:-$DISCOUNT_DEFAULT}"
# Only append a hydra override when it would actually change the yaml default (0.98) or
# was explicitly requested -- keeps the printed/launched command a no-op diff from the
# config for cube/pointmaze/scene at their current defaults. Skip it if EXTRA is already
# setting agent.discount itself, to avoid a duplicate-key hydra error.
DISCOUNT_OVERRIDE=""
if [ -n "$DISCOUNT_USER_SET" ] || [ "$DISCOUNT" != "0.98" ]; then
  case "$EXTRA" in
    *agent.discount=*) : ;;  # caller already overrides it via EXTRA; don't double-set
    *) DISCOUNT_OVERRIDE="agent.discount=$DISCOUNT" ;;
  esac
fi

FLOW_DIR="$PSM_DATA/flow/$FLOW"
GROUP="${GROUP:-psmflow_${ENVKEY}_$(date +%Y%m%d_%H%M%S)}"

# Fail here rather than 40 s into JIT.
[ -f "$PREIMAGES" ] || { echo "no such preimage npz: $PREIMAGES" >&2; exit 1; }
[ -d "$FLOW_DIR" ]  || { echo "no such flow dir: $FLOW_DIR" >&2; exit 1; }

cd "$PSM_REPO"
PY="${PY:-$PSM_REPO/.venv/bin/python}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${MEM_FRAC:-0.90}"
export MUJOCO_GL=egl                    # no DISPLAY on a compute node
export OGBENCH_DATASET_DIR="${OGBENCH_DATASET_DIR:-$PSM_DATA/ogbench}"
export WANDB_MODE="${WANDB_MODE:-offline}"   # compute nodes have no outbound wandb auth
export WANDB_DIR="$PSM_DATA/wandb"
mkdir -p "$PSM_DATA/exp" "$PSM_DATA/wandb"

echo "=== stage C: agent=$AGENT $ENV_NAME seed=$SEED group=$GROUP ==="
echo "    flow=$FLOW_DIR@$EPOCH"
echo "    preimages=$PREIMAGES"
echo "    discount=$DISCOUNT (envkey=$ENVKEY default=$DISCOUNT_DEFAULT override='${DISCOUNT_OVERRIDE:-none}')"
echo "    steps=$STEPS extra='$EXTRA'"

# Continuing a finished run: restore its weights, optimiser state and targets, then train
# STEPS more. RESTORE_PATH must be one run directory (main.py globs it and asserts exactly
# one match). Read the continuation's checkpoints as RESTORE_EPOCH + i -- its own counter
# restarts at 1 -- and give it a GROUP of its own, or it overwrites the original run's
# params_*.pkl. docs/PREIMAGES.md, "Continuing a run past its original budget".
RESTORE_ARG=""
if [ -n "${RESTORE_PATH:-}" ]; then
  [ -d "$RESTORE_PATH" ] || { echo "no such restore dir: $RESTORE_PATH" >&2; exit 1; }
  : "${RESTORE_EPOCH:?RESTORE_EPOCH required when RESTORE_PATH is set}"
  [ -f "$RESTORE_PATH/params_${RESTORE_EPOCH}.pkl" ] || {
    echo "no params_${RESTORE_EPOCH}.pkl in $RESTORE_PATH" >&2; exit 1; }
  RESTORE_ARG="restore_path=$RESTORE_PATH restore_epoch=$RESTORE_EPOCH"
  echo "    CONTINUATION from $RESTORE_PATH @$RESTORE_EPOCH (+$STEPS steps)"
fi

srun "$PY" main.py \
    agent="$AGENT" \
    agent.flow_ckpt_path="$FLOW_DIR" agent.flow_ckpt_epoch="$EPOCH" \
    agent.preimage_path="$PREIMAGES" $DISCOUNT_OVERRIDE $EXTRA \
    env_name="$ENV_NAME" \
    offline_steps="$STEPS" online_steps=0 \
    log_interval="$LOG_INT" eval_interval="$EVAL_INT" eval_episodes="$EVAL_EPS" \
    save_interval="$SAVE_INT" save_dir="$PSM_DATA/exp" \
    run_group="$GROUP" seed="$SEED" $RESTORE_ARG

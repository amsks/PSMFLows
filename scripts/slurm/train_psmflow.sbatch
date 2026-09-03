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
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
# Cluster-specific; set on the command line rather than editing.
set -euo pipefail

: "${PSM_REPO:?PSM_REPO=<path to the PSMFLows checkout> required}"
: "${PSM_DATA:?PSM_DATA=<scratch dir with flow/ and the npz> required}"
: "${ENVKEY:?ENVKEY=cube|antmaze|pointmaze required}"
: "${PREIMAGES:?PREIMAGES=<preimage npz> required}"
SEED="${SEED:-0}"
STEPS="${STEPS:-500000}"
EPOCH="${EPOCH:-500000}"
EVAL_INT="${EVAL_INT:-50000}"
EVAL_EPS="${EVAL_EPS:-50}"
LOG_INT="${LOG_INT:-5000}"
SAVE_INT="${SAVE_INT:-250000}"
EXTRA="${EXTRA:-}"
# Which agent to train. psmflow is the zero-shot method; latentrl is the per-task probe
# (and, at agent.critic_input=latent, the offline DSRL-SAC arm). Both read the same
# preimage-augmented dataset through the same flow, so only the hydra group changes.
AGENT="${AGENT:-psmflow}"

case "$ENVKEY" in
  cube)      ENV_NAME=cube-single-play-singletask-v0;               FLOW=cube-single-play ;;
  antmaze)   ENV_NAME=antmaze-medium-navigate-singletask-v0;        FLOW=antmaze-medium-navigate ;;
  pointmaze) ENV_NAME=pointmaze-medium-navigate-singletask-task1-v0; FLOW=pointmaze-medium-navigate ;;
  *) echo "unknown ENVKEY: $ENVKEY" >&2; exit 1 ;;
esac

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
echo "    steps=$STEPS extra='$EXTRA'"

srun "$PY" main.py \
    agent="$AGENT" \
    agent.flow_ckpt_path="$FLOW_DIR" agent.flow_ckpt_epoch="$EPOCH" \
    agent.preimage_path="$PREIMAGES" $EXTRA \
    env_name="$ENV_NAME" \
    offline_steps="$STEPS" online_steps=0 \
    log_interval="$LOG_INT" eval_interval="$EVAL_INT" eval_episodes="$EVAL_EPS" \
    save_interval="$SAVE_INT" save_dir="$PSM_DATA/exp" \
    run_group="$GROUP" seed="$SEED"

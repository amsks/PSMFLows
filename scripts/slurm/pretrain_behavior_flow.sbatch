#!/usr/bin/env bash
# Stage A on a SLURM cluster: fit the frozen behaviour flow G(s, u) that every later stage
# decodes through. This is `scripts/pretrain_behavior_flow.sh` for a scheduler --
# that script hardcodes midi-01's REPO/`/var/local` paths and backgrounds seeds with nohup,
# neither of which survives sbatch.
#
# It also does the step the shell launcher leaves to the operator: after training it
# INSTALLS the checkpoint into $PSM_DATA/flow/<FLOW>/ as `params_<EPOCH>.pkl` + `flags.json`.
# That flat layout is not cosmetic -- scripts/hf_preimages.py, fetch_preimages_hf.sh,
# run_recovery_tests.sh, eval500.sbatch and precompute_preimages.sbatch all address the
# frozen flow as `$PSM_DATA/flow/<name>/params_<epoch>.pkl` and nothing else.
#
# Submit:
#   sbatch --partition=kisski-inference --account=general \
#     --export=ALL,ENVKEY=scene,SEED=0 scripts/slurm/pretrain_behavior_flow.sbatch
#
# Required in the environment:
#   PSM_REPO    path to this checkout on the cluster
#   PSM_DATA    scratch dir; the flow lands in $PSM_DATA/flow/<FLOW>/
#
#SBATCH --job-name=psm-stagea
# SLURM does NOT expand environment variables in #SBATCH directives, so this is the
# literal $PSM_DATA/logs/slurm on KISSKI rather than the variable. A relative path here
# lands the log in the submit CWD -- i.e. the repo root. Launchers override both flags.
#SBATCH --output=/mnt/home/amohan/psm-data/logs/slurm/%x-%j.out
#SBATCH --error=/mnt/home/amohan/psm-data/logs/slurm/%x-%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
set -euo pipefail

: "${PSM_REPO:?PSM_REPO=<path to the PSMFLows checkout> required}"
: "${PSM_DATA:?PSM_DATA=<scratch dir for the flow> required}"
: "${ENVKEY:?ENVKEY=cube|antmaze|pointmaze|scene required}"
SEED="${SEED:-0}"
STEPS="${STEPS:-500000}"
EPOCH="${EPOCH:-$STEPS}"
LOG_INT="${LOG_INT:-5000}"
# The preimage inversion needs the forward and inverse maps to share a discretization at
# >= 100 steps (tools/precompute_preimages.py asserts it), so Stage A trains at 100. The
# antmaze and pointmaze flows in $PSM_DATA/flow are both at 100; the older cube one is at
# the FQL default of 10 and is inverted by overriding agent.flow_steps=100.
FLOW_STEPS="${FLOW_STEPS:-100}"
SAVE_INT="${SAVE_INT:-$STEPS}"
EXTRA="${EXTRA:-}"
# Skip the install step (e.g. a smoke run that must not clobber a published flow).
INSTALL="${INSTALL:-1}"

case "$ENVKEY" in
  cube)      ENV_NAME=cube-single-play-singletask-v0;               FLOW=cube-single-play ;;
  antmaze)   ENV_NAME=antmaze-medium-navigate-singletask-v0;        FLOW=antmaze-medium-navigate ;;
  pointmaze) ENV_NAME=pointmaze-medium-navigate-singletask-task1-v0; FLOW=pointmaze-medium-navigate ;;
  scene)     ENV_NAME=scene-play-singletask-v0;                     FLOW=scene-play ;;
  *) echo "unknown ENVKEY: $ENVKEY" >&2; exit 1 ;;
esac

GROUP="${GROUP:-bcflow_${FLOW}_$(date +%Y%m%d_%H%M%S)}"

cd "$PSM_REPO"
PY="${PY:-$PSM_REPO/.venv/bin/python}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${MEM_FRAC:-0.90}"
export MUJOCO_GL=egl                    # no DISPLAY on a compute node
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="$PSM_DATA/wandb"
export OGBENCH_DATASET_DIR="${OGBENCH_DATASET_DIR:-$HOME/.ogbench/data}"
mkdir -p "$PSM_DATA/exp" "$PSM_DATA/wandb"

echo "=== stage A: $ENV_NAME seed=$SEED group=$GROUP ==="
echo "    steps=$STEPS flow_steps=$FLOW_STEPS save_interval=$SAVE_INT extra='$EXTRA'"
echo "    install -> $PSM_DATA/flow/$FLOW (INSTALL=$INSTALL)"

# eval_interval=0: task success is meaningless for a reward-free BC flow, and the 50-ep
# rollouts would cost more than the training step they interrupt.
srun "$PY" main.py \
    agent=fql agent.bc_only=true agent.flow_steps="$FLOW_STEPS" $EXTRA \
    env_name="$ENV_NAME" \
    offline_steps="$STEPS" online_steps=0 \
    log_interval="$LOG_INT" eval_interval=0 \
    save_interval="$SAVE_INT" save_dir="$PSM_DATA/exp" \
    run_group="$GROUP" seed="$SEED"

RUN_DIR="$(ls -d "$PSM_DATA/exp/PSMFLows/$GROUP"/sd*/ | head -1)"
echo "=== run dir: $RUN_DIR ==="
[ -f "$RUN_DIR/params_$EPOCH.pkl" ] || { echo "no params_$EPOCH.pkl in $RUN_DIR" >&2; exit 1; }

if [ "$INSTALL" = "1" ]; then
  DEST="$PSM_DATA/flow/$FLOW"
  mkdir -p "$DEST"
  cp "$RUN_DIR/params_$EPOCH.pkl" "$DEST/params_$EPOCH.pkl"
  cp "$RUN_DIR/flags.json" "$DEST/flags.json"
  echo "installed:"
  ls -l "$DEST"
fi

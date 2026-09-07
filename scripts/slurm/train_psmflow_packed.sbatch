#!/usr/bin/env bash
# K seeds of ONE Stage-C config in ONE allocation, sharing ONE GPU.
#
# MEASURED AND REJECTED -- do not use this for real runs. It is checked in so the next
# person does not re-run the experiment. (docs/design/2026-09-07-gpu-packing.md)
#
# The hypothesis was that a psmflow training job leaves the device idle: it holds 5.2 GB
# of an 80 GB H100 and 0.53 of one core, so three seeds should share a GPU cheaply. They
# do not. `utilization.gpu` for ONE seed is already 66-68% -- time with a kernel
# resident -- and without MPS the driver TIME-SLICES between CUDA contexts rather than
# overlapping their kernels, so a second process can only use the ~33% the first leaves.
#
# Smoke on cube, 2000 steps, steady-state `time/epoch_time` from train.csv:
#
#     K=1 (job 2491928)              0.01699 s/step   1.00x   aggregate 1.00x
#     K=3 (job 2491927)              0.0580  s/step   3.41x   aggregate 0.88x
#     K=3 + USE_MPS=1 (job 2491950)  0.0517  s/step   3.04x   aggregate 0.99x
#
# i.e. three packed seeds take 3.4x the wall clock of one and deliver 12% LESS total
# throughput than three separate GPUs; MPS recovers a third of that and reaches
# break-even. Training stays one seed per GPU -- scripts/slurm/train_psmflow.sbatch.
#
# This is the PACKED counterpart of scripts/slurm/train_psmflow.sbatch. Every knob is the
# same and the hydra line is copied verbatim, so a packed sweep and an unpacked one differ
# only in placement, not in configuration -- which is what makes the comparison above fair.
#
# Submit (ONE sbatch for the whole seed sweep):
#   sbatch --partition=kisski-inference --account=general \
#     --job-name=psm_packed_<name> --output=$PSM_DATA/logs/%x-%j.out \
#     --export=ALL,ENVKEY=antmaze,PREIMAGES=$PSM_DATA/preimages/<name>.npz,\
# SEEDS="0 1 2",GROUP=<name> \
#     scripts/slurm/train_psmflow_packed.sbatch
#
# Required in the environment: PSM_REPO, PSM_DATA, ENVKEY, PREIMAGES  (as unpacked).
# SEEDS is a space-separated list; K is its length.
#
# Each seed runs as a background process with its own log under $PSM_DATA/logs/, its own
# share of the device memory pool, and `wait` at the end so the allocation lives until the
# last one finishes. A seed that dies takes the job's exit status with it (see the exit
# roll-up below) rather than silently leaving a hole in the sweep.
#
#SBATCH --job-name=psm-stagec-packed
# SLURM does NOT expand environment variables in #SBATCH directives, so this is the
# literal $PSM_DATA/logs/slurm on KISSKI rather than the variable. A relative path here
# lands the log in the submit CWD -- i.e. the repo root. Launchers override both flags.
#SBATCH --output=/mnt/home/amohan/psm-data/logs/slurm/%x-%j.out
#SBATCH --error=/mnt/home/amohan/psm-data/logs/slurm/%x-%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
# 6 cores and ~32 GB of host RAM per packed seed (measured: 5.6 GB max RSS, ~2.3 cores in
# steady state, more during the dataset load). Sized for K=3; gpu00[1-9] have 208 cores
# and ~500 GB per node across 4 GPUs, so this stays inside one GPU's fair share.
#SBATCH --cpus-per-task=18
#SBATCH --mem=96G
set -euo pipefail

: "${PSM_REPO:?PSM_REPO=<path to the PSMFLows checkout> required}"
: "${PSM_DATA:?PSM_DATA=<scratch dir with flow/ and the npz> required}"
: "${ENVKEY:?ENVKEY=cube|antmaze|pointmaze|scene required}"
: "${PREIMAGES:?PREIMAGES=<preimage npz> required}"
SEEDS="${SEEDS:-0 1 2}"
STEPS="${STEPS:-500000}"
EPOCH="${EPOCH:-500000}"
EVAL_INT="${EVAL_INT:-50000}"
EVAL_EPS="${EVAL_EPS:-50}"
LOG_INT="${LOG_INT:-5000}"
SAVE_INT="${SAVE_INT:-250000}"
EXTRA="${EXTRA:-}"
AGENT="${AGENT:-psmflow}"

case "$ENVKEY" in
  cube)      ENV_NAME=cube-single-play-singletask-v0;                FLOW=cube-single-play ;;
  antmaze)   ENV_NAME=antmaze-medium-navigate-singletask-v0;         FLOW=antmaze-medium-navigate ;;
  pointmaze) ENV_NAME=pointmaze-medium-navigate-singletask-task1-v0; FLOW=pointmaze-medium-navigate ;;
  scene)     ENV_NAME=scene-play-singletask-v0;                      FLOW=scene-play ;;
  *) echo "unknown ENVKEY: $ENVKEY" >&2; exit 1 ;;
esac

FLOW_DIR="$PSM_DATA/flow/$FLOW"
GROUP="${GROUP:-psmflow_${ENVKEY}_$(date +%Y%m%d_%H%M%S)}"

# Fail here rather than 40 s into JIT, and before K processes have been started.
[ -f "$PREIMAGES" ] || { echo "no such preimage npz: $PREIMAGES" >&2; exit 1; }
[ -d "$FLOW_DIR" ]  || { echo "no such flow dir: $FLOW_DIR" >&2; exit 1; }

read -r -a SEED_LIST <<< "$SEEDS"
K="${#SEED_LIST[@]}"
[ "$K" -ge 1 ] || { echo "SEEDS is empty" >&2; exit 1; }

cd "$PSM_REPO"
PY="${PY:-$PSM_REPO/.venv/bin/python}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# Split the pool K ways. With preallocation off this is a CAP, not a reservation, so the
# only thing it has to clear is the ~5.2 GB one seed actually uses (6.5% of an 80 GB
# H100); 0.90/K leaves an order of magnitude of headroom even at K=6.
MEM_FRAC_TOTAL="${MEM_FRAC_TOTAL:-0.90}"
export XLA_PYTHON_CLIENT_MEM_FRACTION
XLA_PYTHON_CLIENT_MEM_FRACTION="$(awk -v t="$MEM_FRAC_TOTAL" -v k="$K" 'BEGIN{printf "%.4f", t/k}')"
# One BLAS thread per core per seed, so K numpy runtimes do not each spawn
# $(nproc)=208 threads and thrash the cgroup's cpuset.
CPUS="${SLURM_CPUS_PER_TASK:-18}"
THREADS="$(( CPUS / K ))"
[ "$THREADS" -ge 1 ] || THREADS=1
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS"
# --- optional: NVIDIA MPS ------------------------------------------------------------
# Without MPS, kernels from different CUDA CONTEXTS cannot run concurrently: the driver
# time-slices the device between processes. That is the whole cost of packing here --
# psmflow's step is hundreds of tiny kernels (batch 256 MLPs, a 10-step flow unroll), so
# three contexts spend their time context-switching rather than filling the 114 SMs.
# MPS funnels them through ONE context and lets the kernels genuinely overlap.
#
# USE_MPS=1 starts a daemon private to THIS job: the pipe directory is per-job, so no
# other job of ours -- on this node or elsewhere -- can connect to it or be affected by
# it, and it is torn down in the EXIT trap. Needs the GPU in Default compute mode
# (kisski-inference H100s are).
if [ "${USE_MPS:-0}" = "1" ]; then
  export CUDA_MPS_PIPE_DIRECTORY="${TMPDIR:-/tmp}/mps-pipe-${SLURM_JOB_ID:-$$}"
  export CUDA_MPS_LOG_DIRECTORY="${TMPDIR:-/tmp}/mps-log-${SLURM_JOB_ID:-$$}"
  mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
  nvidia-cuda-mps-control -d
  # Leaving a daemon behind would keep serving a GPU this job no longer holds.
  trap 'echo quit | nvidia-cuda-mps-control >/dev/null 2>&1 || true' EXIT
  echo "    MPS daemon up, pipe=$CUDA_MPS_PIPE_DIRECTORY"
fi
export MUJOCO_GL=egl                    # no DISPLAY on a compute node
export OGBENCH_DATASET_DIR="${OGBENCH_DATASET_DIR:-$PSM_DATA/ogbench}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="$PSM_DATA/wandb"
LOGDIR="${PACKED_LOGDIR:-$PSM_DATA/logs}"
mkdir -p "$PSM_DATA/exp" "$PSM_DATA/wandb" "$LOGDIR"

echo "=== stage C PACKED: agent=$AGENT $ENV_NAME group=$GROUP ==="
echo "    seeds=[${SEED_LIST[*]}]  K=$K on one GPU"
echo "    flow=$FLOW_DIR@$EPOCH"
echo "    preimages=$PREIMAGES"
echo "    steps=$STEPS extra='$EXTRA'"
echo "    XLA_PYTHON_CLIENT_MEM_FRACTION=$XLA_PYTHON_CLIENT_MEM_FRACTION each, OMP_NUM_THREADS=$THREADS, cpus-per-task=$CPUS"
echo "    per-seed logs: $LOGDIR/${SLURM_JOB_NAME:-packed}-${SLURM_JOB_ID:-local}-sd<seed>.log"

PIDS=()
for s in "${SEED_LIST[@]}"; do
  LOG="$LOGDIR/${SLURM_JOB_NAME:-packed}-${SLURM_JOB_ID:-local}-sd${s}.log"
  # No `srun` per seed: one step with K tasks would give each seed 1/K of the cores as a
  # hard cpuset and serialise the dataset load. These are plain background processes in
  # the batch step, which is also what makes `wait` below the right join.
  # shellcheck disable=SC2086  -- EXTRA is a deliberate word-split list of hydra overrides
  "$PY" main.py \
      agent="$AGENT" \
      agent.flow_ckpt_path="$FLOW_DIR" agent.flow_ckpt_epoch="$EPOCH" \
      agent.preimage_path="$PREIMAGES" $EXTRA \
      env_name="$ENV_NAME" \
      offline_steps="$STEPS" online_steps=0 \
      log_interval="$LOG_INT" eval_interval="$EVAL_INT" eval_episodes="$EVAL_EPS" \
      save_interval="$SAVE_INT" save_dir="$PSM_DATA/exp" \
      run_group="$GROUP" seed="$s" > "$LOG" 2>&1 &
  PIDS+=("$!")
  echo "    seed $s -> pid $! -> $LOG"
done

# Roll the seeds' exit codes up into the job's. `wait -n` would return on the first one
# to finish; we need all of them, and we need a failure to be visible in sacct rather
# than only in one of K logs.
RC=0
for i in "${!PIDS[@]}"; do
  if ! wait "${PIDS[$i]}"; then
    echo "seed ${SEED_LIST[$i]} (pid ${PIDS[$i]}) FAILED" >&2
    RC=1
  fi
done
echo "=== packed job done, rc=$RC ==="
exit "$RC"

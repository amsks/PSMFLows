#!/usr/bin/env bash
# One 500-episode eval per SLURM job -- the scheduler-side wrapper around
# scripts/eval500.sh, the same way train_psmflow.sbatch wraps main.py.
#
# eval500.sh's paths default to midi-01; here they are pointed at PSM_DATA, which on this
# cluster holds flow/<name>/ and preimages/<name>.npz (the layout hf_preimages.py pulls
# into). Report JSONs land in $PSM_DATA/logs, which is the --logs dir to hand
# tools/make_tables.py on this machine.
#
# Submit:
#   sbatch --partition=kisski-inference --account=general --time=02:00:00 \
#     --job-name=e500_<name> --output=$PSM_DATA/logs/%x-%j.out \
#     --export=ALL,MODE=psmflow,ENVKEY=cube,RUN_DIR=<run dir>,OUT=<basename>,SEED=0 \
#     scripts/slurm/eval500.sbatch
#
# EXTRA is forwarded verbatim as trailing hydra overrides and is normally EMPTY: the agent
# config comes from the run's own flags.json (tools/eval_checkpoint.py). Pass something
# only to evaluate a checkpoint deliberately off its own config.
#SBATCH --job-name=e500
# SLURM does NOT expand environment variables in #SBATCH directives, so this is the
# literal $PSM_DATA/logs/slurm on KISSKI rather than the variable. A relative path here
# lands the log in the submit CWD -- i.e. the repo root. Launchers override both flags.
#SBATCH --output=/mnt/home/amohan/psm-data/logs/slurm/%x-%j.out
#SBATCH --error=/mnt/home/amohan/psm-data/logs/slurm/%x-%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
# Unchanged from the single-process era, and that is the point: EVAL_WORKERS=4 (below)
# needs 4 cores plus the parent, and even 8 workers on antmaze peaked at 16.8 GB total
# RSS (~2.1 GB each). Raise these only if you raise EVAL_WORKERS -- but see
# docs/design/2026-09-07-gpu-packing.md: rollout throughput on one H100 saturates at
# ~1.45x by N=3, so more workers buy cores, not speed.
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
set -euo pipefail

: "${PSM_REPO:?PSM_REPO=<path to the PSMFLows checkout> required}"
: "${PSM_DATA:?PSM_DATA=<scratch dir with flow/ and preimages/> required}"
: "${MODE:?MODE=psmflow|bc required}"
: "${ENVKEY:?ENVKEY=cube|antmaze|pointmaze|scene required}"
OUT="${OUT:?OUT=<report json basename> required}"
RUN_DIR="${RUN_DIR:--}"
# Eval seed. Unset by default, which is the protocol every recorded eval500 number used:
# no eval line has ever passed one, so cfg.seed stays at the config default (0) for every
# training seed and the 500 episode inits are paired across seeds.
SEED="${SEED:-}"
EXTRA="${EXTRA:-}"
# Full OGBench env id, overriding ENVKEY's default. Set it to
# <env>-singletask-task{1..5}-v0 to evaluate a checkpoint on another task (see
# scripts/eval500.sh); unset means the ENVKEY default, i.e. cube task 2 / maze task 1.
export ENV_NAME="${ENV_NAME:-}"

case "$ENVKEY" in
  cube)      FLOW=cube-single-play ;;
  antmaze)   FLOW=antmaze-medium-navigate ;;
  pointmaze) FLOW=pointmaze-medium-navigate ;;
  scene)     FLOW=scene-play ;;
  *) echo "unknown ENVKEY: $ENVKEY" >&2; exit 1 ;;
esac

export PSM_REPO
export FLOW_DIR="$PSM_DATA/flow/$FLOW"
export PRE_NPZ="$PSM_DATA/preimages/$FLOW.npz"
# EVAL_LOGS defaults to $PSM_DATA/logs (make_tables' --logs dir); the BC controls are
# written to $PSM_DATA/evals instead, so let a preset value through.
export EVAL_LOGS="${EVAL_LOGS:-$PSM_DATA/logs}"
export MEM_FRAC="${MEM_FRAC:-0.90}"
# Worker processes per eval (see scripts/eval500.sh). 4 is the measured knee: 8 workers
# on antmaze gave 1.43x (92.3 -> 64.5 min) and three concurrent cube rollouts gave 1.50x,
# i.e. the aggregate saturates by N=3 however many contexts you add. EVAL_WORKERS=1
# reproduces a pre-2026-09-07 number bit for bit; anything else is a fresh 500-episode
# sample of the same distribution.
export EVAL_WORKERS="${EVAL_WORKERS:-4}"
export GPU=0
# ogbench's own default; $PSM_DATA/ogbench does not exist on this cluster.
export OGBENCH_DATASET_DIR="${OGBENCH_DATASET_DIR:-$HOME/.ogbench/data}"

echo "=== eval500: mode=$MODE env=$ENVKEY seed=${SEED:-<config default>} out=$OUT workers=$EVAL_WORKERS"
echo "    run_dir=$RUN_DIR"
echo "    flow=$FLOW_DIR  preimages=$PRE_NPZ"
echo "    extra='$EXTRA'"

cd "$PSM_REPO"
# shellcheck disable=SC2086  -- EXTRA is a deliberate word-split list of hydra overrides
SEED_ARG=()
if [ -n "$SEED" ]; then SEED_ARG=(seed="$SEED"); fi
srun bash scripts/eval500.sh "$MODE" "$ENVKEY" "$RUN_DIR" "$OUT" ${SEED_ARG[@]+"${SEED_ARG[@]}"} $EXTRA

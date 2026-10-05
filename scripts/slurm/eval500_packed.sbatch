#!/usr/bin/env bash
# Five-task eval500 of ONE (run, checkpoint, readout) in ONE SLURM job (2026-10-05).
#
# eval500.sbatch runs one 500-episode eval per job, so a five-task readout is five jobs that
# each wait in the queue and each load the dataset and JIT the agent. A cube readout other
# than lp runs for 1-4 minutes once started (sacct, 143 jobs, 2026-09-28 to 10-04), so the
# queue wait dominates. This file runs the five tasks sequentially in one allocation and
# writes the SAME report names, <OUT>_task<t>.json, so tools/make_tables.py and the
# aggregation scripts are unchanged. Tasks whose JSON already exists are skipped; a task
# that fails does not stop the others and the job exits non-zero at the end.
#
# Submit:
#   sbatch --partition=kisski-inference --account=general --nice=600 \
#     --job-name=e500p_<out> --output=$PSM_DATA/logs/slurm/%x-%j.out \
#     --export=ALL,PSM_REPO=...,PSM_DATA=...,MODE=psmgoal,ENVKEY=cube,RUN_DIR=<run dir>,\
#   OUT=<group>_<sd>_<step>_<tag>,RESTORE_EPOCH=<step>,EXTRA="agent.coef_source=hgoal_each agent.acting=gpi" \
#     scripts/slurm/eval500_packed.sbatch
#
# Knobs are eval500.sbatch's (MODE, ENVKEY, RUN_DIR, OUT, RESTORE_EPOCH, SEED, EXTRA,
# EVAL_WORKERS, EVAL_LOGS, MEM_FRAC) plus TASKS (default "1 2 3 4 5"). OUT is the report
# basename WITHOUT the _task<t> suffix. ENV_NAME is set per task here and must not be passed.
#
#SBATCH --job-name=e500p
#SBATCH --output=/mnt/home/amohan/psm-data/logs/slurm/%x-%j.out
#SBATCH --error=/mnt/home/amohan/psm-data/logs/slurm/%x-%j.err
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
# 4 eval workers plus the parent; peak RSS of a 4-worker cube eval was 9.8 GB (sacct).
#SBATCH --cpus-per-task=5
#SBATCH --mem=16G
set -uo pipefail

: "${PSM_REPO:?PSM_REPO=<path to the PSMFLows checkout> required}"
: "${PSM_DATA:?PSM_DATA=<scratch dir with flow/ and preimages/> required}"
: "${MODE:?MODE=psmflow|psmgoal|bc required}"
: "${ENVKEY:?ENVKEY=cube|antmaze|pointmaze|scene required}"
OUT="${OUT:?OUT=<report json basename without _task<t>> required}"
RUN_DIR="${RUN_DIR:--}"
SEED="${SEED:-}"
EXTRA="${EXTRA:-}"
TASKS="${TASKS:-1 2 3 4 5}"
[ -z "${ENV_NAME:-}" ] || { echo "ENV_NAME is set per task by this file; do not pass it" >&2; exit 1; }

case "$ENVKEY" in
  cube)      FLOW=cube-single-play;          ENV_BASE=cube-single-play ;;
  antmaze)   FLOW=antmaze-medium-navigate;   ENV_BASE=antmaze-medium-navigate ;;
  pointmaze) FLOW=pointmaze-medium-navigate; ENV_BASE=pointmaze-medium-navigate ;;
  scene)     FLOW=scene-play;                ENV_BASE=scene-play ;;
  *) echo "unknown ENVKEY: $ENVKEY" >&2; exit 1 ;;
esac

export PSM_REPO
export FLOW_DIR="$PSM_DATA/flow/$FLOW"
export PRE_NPZ="$PSM_DATA/preimages/$FLOW.npz"
export EVAL_LOGS="${EVAL_LOGS:-$PSM_DATA/logs}"
export MEM_FRAC="${MEM_FRAC:-0.90}"
export EVAL_WORKERS="${EVAL_WORKERS:-4}"
export GPU=0
export OGBENCH_DATASET_DIR="${OGBENCH_DATASET_DIR:-$HOME/.ogbench/data}"
export RESTORE_EPOCH="${RESTORE_EPOCH:-500000}"

echo "=== eval500 packed: mode=$MODE env=$ENVKEY tasks='$TASKS' epoch=$RESTORE_EPOCH out=$OUT workers=$EVAL_WORKERS job=$SLURM_JOB_ID node=$(hostname)"
echo "    run_dir=$RUN_DIR  extra='$EXTRA'"

cd "$PSM_REPO"
SEED_ARG=()
if [ -n "$SEED" ]; then SEED_ARG=(seed="$SEED"); fi
status=0
for t in $TASKS; do
  out="${OUT}_task${t}"
  if [ -f "$EVAL_LOGS/$out.json" ]; then echo "--- task $t: $out.json exists, skip"; continue; fi
  export ENV_NAME="${ENV_BASE}-singletask-task${t}-v0"
  echo "--- task $t: $ENV_NAME -> $out  ($(date +%F_%T))"
  # shellcheck disable=SC2086  -- EXTRA is a deliberate word-split list of hydra overrides
  if ! srun --ntasks=1 bash scripts/eval500.sh "$MODE" "$ENVKEY" "$RUN_DIR" "$out" ${SEED_ARG[@]+"${SEED_ARG[@]}"} $EXTRA; then
    echo "--- task $t FAILED" >&2
    status=1
  fi
done
echo "=== done ($(date +%F_%T)) status=$status"
exit $status

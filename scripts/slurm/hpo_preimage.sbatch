#!/usr/bin/env bash
# The inversion HPO for a scheduler: 50 SMAC trials of tools/hpo_preimage_inversion.py over
# (alpha, prior_scale, n_steps) against ONE fixed neighbourhood batch, so a difference
# between trials is the setting and nothing else.
#
# This is the step that picks Stage B's ALPHA / PRIOR_SCALE / N_STEPS. Do not take them from
# the frontier table in configs/inversion/default.yaml -- that is a coarser grid, and the
# 2026-08-29 cube sweep beat its hand grid (coverage 0.990 at decode 0.196, against
# 0.96 at 0.227 which is over budget).
#
# The trials are STRICTLY SEQUENTIAL by design (configs/hpo_preimage.yaml:
# max_parallelization 0.0): they share the in-process dataset/batch/agent cache, so one
# GPU and ~65 s/trial at d_a=5, ~127 s at d_a=8.
#
# Submit:
#   sbatch --partition=kisski-inference --account=general \
#     --export=ALL,ENVKEY=scene,BUDGET=0.22 scripts/slurm/hpo_preimage.sbatch
#
# Required in the environment:
#   PSM_REPO    path to this checkout on the cluster
#   PSM_DATA    scratch dir holding flow/<name>/
#
#SBATCH --job-name=psm-hpo
# SLURM does NOT expand environment variables in #SBATCH directives, so this is the
# literal $PSM_DATA/logs/slurm on KISSKI rather than the variable. A relative path here
# lands the log in the submit CWD -- i.e. the repo root. Launchers override both flags.
#SBATCH --output=/mnt/home/amohan/psm-data/logs/slurm/%x-%j.out
#SBATCH --error=/mnt/home/amohan/psm-data/logs/slurm/%x-%j.err
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
set -euo pipefail

: "${PSM_REPO:?PSM_REPO=<path to the PSMFLows checkout> required}"
: "${PSM_DATA:?PSM_DATA=<scratch dir with flow/> required}"
: "${ENVKEY:?ENVKEY=cube|antmaze|pointmaze|scene required}"
EPOCH="${EPOCH:-500000}"
FLOW_STEPS="${FLOW_STEPS:-100}"        # forward AND inverse discretization; must be >= 100
N_ROWS="${N_ROWS:-1280}"
SAMPLE_K="${SAMPLE_K:-128}"
COV_K="${COV_K:-64}"
N_TRIALS="${N_TRIALS:-50}"
# Decode-error budget in action units, 0.1*sqrt(d_a): 0.22 at d_a=5 (every manipspace env,
# cube and scene alike), 0.28 at d_a=8 (antmaze). configs/hpo_preimage.yaml defaults to 0.22.
BUDGET="${BUDGET:-0.22}"
EXTRA="${EXTRA:-}"
# smac  -- configs/hpo_preimage.yaml, Bayesian optimisation. Needs hypersweeper + smac +
#          ConfigSpace, which are NOT installed in this cluster's .venv and cannot safely be
#          (the resolve downgrades wandb and scipy under every other queued job).
# grid  -- configs/hpo_preimage_grid.yaml, hydra's stock sweeper over GRID_* below. Same
#          target, same cost, same fixed batch; only the point selection differs.
SWEEP_MODE="${SWEEP_MODE:-grid}"
GRID_ALPHA="${GRID_ALPHA:-5,10,20,40,80}"
GRID_PRIOR_SCALE="${GRID_PRIOR_SCALE:-0.5,0.7,1.0}"
GRID_N_STEPS="${GRID_N_STEPS:-8,12,16}"

case "$ENVKEY" in
  cube)      ENV_NAME=cube-single-play-singletask-v0;               FLOW=cube-single-play ;;
  antmaze)   ENV_NAME=antmaze-medium-navigate-singletask-v0;        FLOW=antmaze-medium-navigate ;;
  pointmaze) ENV_NAME=pointmaze-medium-navigate-singletask-task1-v0; FLOW=pointmaze-medium-navigate ;;
  scene)     ENV_NAME=scene-play-singletask-v0;                     FLOW=scene-play ;;
  *) echo "unknown ENVKEY: $ENVKEY" >&2; exit 1 ;;
esac

FLOW_DIR="$PSM_DATA/flow/$FLOW"
[ -f "$FLOW_DIR/params_$EPOCH.pkl" ] || { echo "no Stage-A ckpt: $FLOW_DIR/params_$EPOCH.pkl" >&2; exit 1; }

SWEEP_DIR="${SWEEP_DIR:-$PSM_DATA/hpo/${ENVKEY}_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$SWEEP_DIR"

cd "$PSM_REPO"
PY="${PY:-$PSM_REPO/.venv/bin/python}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${MEM_FRAC:-0.90}"
export MUJOCO_GL=egl
export OGBENCH_DATASET_DIR="${OGBENCH_DATASET_DIR:-$HOME/.ogbench/data}"

echo "=== inversion HPO: $ENV_NAME  flow=$FLOW_DIR@$EPOCH ==="
echo "    mode=$SWEEP_MODE budget=$BUDGET n_rows=$N_ROWS sample_k=$SAMPLE_K cov_k=$COV_K"
if [ "$SWEEP_MODE" = "smac" ]; then echo "    n_trials=$N_TRIALS"; else
  echo "    grid alpha=$GRID_ALPHA prior_scale=$GRID_PRIOR_SCALE n_steps=$GRID_N_STEPS"; fi
echo "    sweep dir: $SWEEP_DIR"

COMMON=(agent=fql env_name="$ENV_NAME"
        restore_path="$FLOW_DIR" restore_epoch="$EPOCH"
        agent.flow_steps="$FLOW_STEPS" inversion.n_initial_steps="$FLOW_STEPS"
        hpo_decode_budget="$BUDGET"
        +n_rows="$N_ROWS" +sample_k="$SAMPLE_K" +cov_k="$COV_K"
        hydra.sweep.dir="$SWEEP_DIR")

if [ "$SWEEP_MODE" = "smac" ]; then
  srun "$PY" tools/hpo_preimage_inversion.py -m --config-name=hpo_preimage \
      "${COMMON[@]}" \
      hydra.sweeper.n_trials="$N_TRIALS" \
      hydra.sweeper.sweeper_kwargs.optimizer_kwargs.scenario.n_trials="$N_TRIALS" $EXTRA
else
  # Grid over the SAME target and the SAME cost; see configs/hpo_preimage_grid.yaml for why
  # SMAC is not always available. GRID_* bracket the SMAC search space (alpha log [2,100],
  # prior_scale [0.3,1.0], n_steps [5,20]) and cube's winner (a 14.1 / ps 0.69 / ns 14).
  srun "$PY" tools/hpo_preimage_inversion.py -m --config-name=hpo_preimage_grid \
      "${COMMON[@]}" \
      inversion.alpha="$GRID_ALPHA" \
      inversion.prior_scale="$GRID_PRIOR_SCALE" \
      inversion.n_steps="$GRID_N_STEPS" $EXTRA
fi

echo "=== trials -> $SWEEP_DIR ==="
ls -1 "$SWEEP_DIR"/*/hpo_trial.json 2>/dev/null | wc -l

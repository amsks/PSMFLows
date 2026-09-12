#!/usr/bin/env bash
# 2026-09-08 Item 1: the corrected GPI-selection probe on the affine_strict_cube ladder.
# docs/design/2026-09-08-critic-signal-and-dsrl-na.md 2.
#
# The probe's Monte-Carlo ground truth now uses success ONSET states (the old pool put ten
# of sixteen states inside a success that had already happened), 64 of them, discounted at
# the training gamma, with three rankers -- see the tool's docstring.
#
# Three continuations per candidate (held / onestep / onestep_bc); the pre-registration is
# read on onestep_bc, the in-support one. The returns depend only on the FROZEN flow, the
# dataset and the simulator, so they are identical in every job: the first job writes
# MC_CACHE and the rest wait on it via --dependency=afterok, which is also why this is a
# script and not a bare for-loop -- two concurrent jobs would race on the cache file. The
# priming job pays ~800k simulator steps, hence the 5 h wall clock.
#
#   PSM_REPO=... PSM_DATA=... bash scripts/slurm/launch_gpi_selection_onsets.sh [smoke]
set -euo pipefail
: "${PSM_REPO:?}"; : "${PSM_DATA:?}"
EXP="$PSM_DATA/exp/PSMFLows"
LOGDIR="$PSM_DATA/logs/slurm/gpisel"; mkdir -p "$LOGDIR"
# N_MC states, and a cache + report tag keyed on it: the 64-state pass left only 15 states
# where the in-support continuation separated the candidates at all (at the other 49 the BC
# tail fails from every one of them), which is too few to rest a verdict on.
N_MC="${N_MC:-64}"
SUFFIX=""
if [ "$N_MC" != "64" ]; then SUFFIX="_n$N_MC"; fi
CACHE="$PSM_DATA/logs/diag_gpi_selection_mc_cache_cube${SUFFIX}.npz"
ORACLE=$(ls -d "$EXP"/fqlexpert_cube_a300/sd000* 2>/dev/null | head -1)
[ -z "$ORACLE" ] && { echo "no fqlexpert_cube_a300 run -- the oracle ranker needs one" >&2; exit 1; }
SB=(--partition=kisski-inference --account=general --gres=gpu:1 --cpus-per-task=8
    --mem=64G --time=05:00:00)

# seed  epoch   500-episode success on record (blank = never scored at 500 ep)
ROWS=(
  "sd000  250000  -"
  "sd000  350000  0.704_best_of_all_seeds"
  "sd000  500000  0.086_worst_of_all_seeds"
  "sd001  250000  -"
  "sd001  500000  0.548"
  "sd002  250000  0.246"
  "sd002  450000  0.564"
  "sd002  500000  0.292"
)
[ "${1:-}" = "smoke" ] && ROWS=("${ROWS[0]}")

dep=""
for row in "${ROWS[@]}"; do
  read -r sd epoch why <<<"$row"
  rd=$(ls -d "$EXP/affine_strict_cube/${sd}"* 2>/dev/null | head -1)
  [ -z "$rd" ] && { echo "MISSING affine_strict_cube/$sd" >&2; exit 1; }
  [ -f "$rd/params_${epoch}.pkl" ] || { echo "MISSING $rd/params_${epoch}.pkl" >&2; exit 1; }
  tag="cube_${sd}${SUFFIX}"
  args=(--job-name="gpisel_${sd}_${epoch}" --output="$LOGDIR/%x-%j.out"
        --error="$LOGDIR/%x-%j.err"
        --export="ALL,PSM_REPO=$PSM_REPO,PSM_DATA=$PSM_DATA,ENVKEY=cube,RUN_DIR=$rd,EPOCH=$epoch,TAG=$tag,N_MC=$N_MC,MC_CACHE=$CACHE,INBOX_CLIP=${INBOX_CLIP:-1.5},ORACLE_DIR=$ORACLE")
  # The first job primes the cache; everything else waits for it to land.
  if [ -n "$dep" ]; then args+=(--dependency="afterok:$dep"); fi
  jid=$(sbatch "${SB[@]}" "${args[@]}" scripts/slurm/diag_gpi_selection.sbatch | awk '{print $NF}')
  echo "submitted $jid  $sd @ $epoch  (${why})"
  [ -z "$dep" ] && dep="$jid"
done
echo
echo "reports  -> $PSM_DATA/logs/diag_gpi_selection_cube_sd00*_<epoch>.json"
echo "mc cache -> $CACHE"

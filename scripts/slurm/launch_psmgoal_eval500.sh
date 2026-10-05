#!/usr/bin/env bash
# Five-task 500-episode evals of every seed of one psmgoal run group, one SLURM job per
# (seed, checkpoint, task, arm). Reports land in $PSM_DATA/logs as
#   <GROUP>_sd00<s>_<epoch>_<tag>_task<t>.json
# Existing reports are skipped, so re-running after a partial failure only fills the gaps.
#
#   GROUP=psmgoal_f1_ortho10_cube EPOCHS="250000 500000" COEFS="lp regression" \
#     bash scripts/slurm/launch_psmgoal_eval500.sh
#
# An arm is a (tag, override string) pair: the tag goes into the report name, the overrides
# sit on top of the run's own flags.json (eval_checkpoint.py reads the rest from it). COEFS
# is the short form: each coefficient source c is the arm (c, "agent.coef_source=c"). ARMS
# names arms in full, ';'-separated "tag|overrides", and takes precedence over COEFS. The
# overrides may hold spaces but no commas (they travel through sbatch --export). lp runs the
# 20000-step Lagrangian inference, so give it the time.
#
# The four readouts of a goal-conditioned run (policy_index=goal actor_input=goal,
# docs/design/2026-10-01-psmgoal-goal-conditioned.md):
#   GROUP=psmgoal_gc_sq_cube EPOCHS="250000 500000" ARMS="\
#   hgoal|agent.coef_source=amortized agent.acting=gpi;\
#   lp|agent.coef_source=lp agent.acting=gpi;\
#   actor_rel|agent.acting=distill agent.eval_redistill=false agent.eval_goal_source=relabel;\
#   actor_env|agent.acting=distill agent.eval_redistill=false agent.eval_goal_source=env" \
#     bash scripts/slurm/launch_psmgoal_eval500.sh
set -euo pipefail

main() {
  : "${GROUP:?GROUP=<run group under \$PSM_DATA/exp/PSMFLows>}"
  PSM_REPO="${PSM_REPO:-$(cd "$(dirname "$0")/../.." && pwd)}"
  PSM_DATA="${PSM_DATA:-/mnt/home/amohan/psm-data}"
  EPOCHS="${EPOCHS:-500000}"
  COEFS="${COEFS:-lp regression}"
  ARMS="${ARMS:-}"
  TASKS="${TASKS:-1 2 3 4 5}"
  TIME="${TIME:-03:00:00}"
  LOGS="$PSM_DATA/logs"
  mkdir -p "$LOGS/slurm"

  # The arm list: tags[i] names the report, overrides[i] is its hydra override string.
  tags=()
  overrides=()
  if [ -n "$ARMS" ]; then
    IFS=';' read -r -a arm_specs <<< "$ARMS"
    for spec in "${arm_specs[@]}"; do
      [ -n "${spec// /}" ] || continue
      tag="${spec%%|*}"
      ovr="${spec#*|}"
      [ "$tag" != "$spec" ] || { echo "arm '$spec' is not 'tag|overrides'" >&2; exit 1; }
      case "$ovr" in *,*) echo "arm '$tag': overrides must hold no commas" >&2; exit 1 ;; esac
      tags+=("$(echo "$tag" | tr -d '[:space:]')")
      overrides+=("$ovr")
    done
  else
    for coef in $COEFS; do
      tags+=("$coef")
      overrides+=("agent.coef_source=${coef}")
    done
  fi
  [ "${#tags[@]}" -gt 0 ] || { echo "no arms: set COEFS or ARMS" >&2; exit 1; }

  shopt -s nullglob
  runs=("$PSM_DATA"/exp/PSMFLows/"$GROUP"/sd*/)
  [ "${#runs[@]}" -gt 0 ] || { echo "no runs under $GROUP" >&2; exit 1; }
  n=0
  for run in "${runs[@]}"; do
    run="${run%/}"
    sd="$(basename "$run" | cut -c1-5)"
    for ep in $EPOCHS; do
      [ -f "$run/params_${ep}.pkl" ] || { echo "missing $run/params_${ep}.pkl, skip" >&2; continue; }
      for i in "${!tags[@]}"; do
        for t in $TASKS; do
          out="${GROUP}_${sd}_${ep}_${tags[$i]}_task${t}"
          [ -f "$LOGS/$out.json" ] && continue
          sbatch --parsable --partition=kisski-inference --account=general --time="$TIME" \
            --job-name="e500_${out}" --output="$LOGS/slurm/%x-%j.out" --error="$LOGS/slurm/%x-%j.err" \
            --export=ALL,PSM_REPO="$PSM_REPO",PSM_DATA="$PSM_DATA",MODE=psmgoal,ENVKEY=cube,RUN_DIR="$run",OUT="$out",RESTORE_EPOCH="$ep",EVAL_WORKERS=4,ENV_NAME="cube-single-play-singletask-task${t}-v0",EXTRA="${overrides[$i]}" \
            "$PSM_REPO/scripts/slurm/eval500.sbatch" >/dev/null
          n=$((n + 1))
        done
      done
    done
  done
  echo "submitted $n jobs for $GROUP"
}

main "$@"

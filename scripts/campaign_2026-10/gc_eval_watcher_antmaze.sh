#!/usr/bin/env bash
# Five-task eval500 of psmgoal_gc_{sq,sm}_antmaze at 250k and 500k, four readouts per checkpoint
# (docs/design/2026-10-01-psmgoal-goal-conditioned.md). MODE=psmgoal, ENVKEY=antmaze.
# Copy of gc_eval_watcher.sh with the antmaze groups, env ids, marker dir and time limit.
# JSON: {arm}_{sd}_{step}_{tag}_task{t}.json in $PSM_DATA/logs. Exits when all 240 are submitted.
PSM_DATA=/mnt/home/amohan/psm-data; REPO=/mnt/home/amohan/git/Austin/PSMFLows
MARK=$PSM_DATA/gc_scripts/submitted_antmaze; mkdir -p $MARK $PSM_DATA/logs/slurm
TAGS=(hgoal lp actor_rel actor_env)
declare -A OV=(
  [hgoal]="agent.coef_source=amortized agent.acting=gpi"
  [lp]="agent.coef_source=lp agent.acting=gpi"
  [actor_rel]="agent.acting=distill agent.eval_redistill=false agent.eval_goal_source=relabel"
  [actor_env]="agent.acting=distill agent.eval_redistill=false agent.eval_goal_source=env"
)
while true; do
  n=0
  for arm in psmgoal_gc_sq_antmaze psmgoal_gc_sm_antmaze; do
    for run in $PSM_DATA/exp/PSMFLows/${arm}/sd00*; do
      [ -d "$run" ] || continue; sd=$(basename "$run" | cut -c1-5)
      for ep in 250000 500000; do
        ck=$run/params_${ep}.pkl; ready=0
        [ -f "$ck" ] && [ $(( $(date +%s) - $(stat -c %Y "$ck") )) -gt 120 ] && ready=1
        for tag in "${TAGS[@]}"; do for t in 1 2 3 4 5; do
          out=${arm}_${sd}_${ep}_${tag}_task${t}
          if [ -f $MARK/$out ] || [ -f $PSM_DATA/logs/$out.json ]; then n=$((n+1)); continue; fi
          [ $ready = 1 ] || continue
          # 500 antmaze episodes of up to 1000 steps on 4 workers, plus the 20000-step
          # coefficient inference in each worker for the lp readout: estimated up to ~1.7 h.
          id=$(sbatch --parsable --partition=kisski-inference --account=general --time=04:00:00 \
            --job-name=e500_$out --output=$PSM_DATA/logs/slurm/%x-%j.out --error=$PSM_DATA/logs/slurm/%x-%j.err \
            --export=ALL,PSM_REPO=$REPO,PSM_DATA=$PSM_DATA,MODE=psmgoal,ENVKEY=antmaze,RUN_DIR=$run,OUT=$out,RESTORE_EPOCH=$ep,EVAL_WORKERS=4,ENV_NAME=antmaze-medium-navigate-singletask-task${t}-v0,EXTRA="${OV[$tag]}" \
            $REPO/scripts/slurm/eval500.sbatch) && { echo "$(date +%F_%T) submitted $out job $id"; echo $id > $MARK/$out; n=$((n+1)); }
        done; done
      done
    done
  done
  [ $n -ge 240 ] && { echo "$(date +%F_%T) ALLDONE 240 evals submitted"; exit 0; }
  sleep 300
done

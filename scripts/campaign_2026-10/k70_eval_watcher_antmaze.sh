#!/usr/bin/env bash
# Five-task eval500 of psmgoal_ja_fbc3_mix_k70_antmaze (antmaze_k70_arms.sh). Readouts: best-of-64 by the
# critic (hgoal_each) and the actor alone (actor_rel) at 250k and 500k. MODE=psmgoal, ENVKEY=antmaze.
# Copy of db_eval_watcher_antmaze.sh with the group and readouts of the ja cube watcher; --nice=600.
# JSON: {arm}_{sd}_{step}_{tag}_task{t}.json in $PSM_DATA/logs. Exits when all 60 are submitted.
PSM_DATA=/mnt/home/amohan/psm-data; REPO=/mnt/home/amohan/git/Austin/PSMFLows
MARK=$PSM_DATA/gc_scripts/submitted_k70_antmaze; mkdir -p $MARK $PSM_DATA/logs/slurm
declare -A OV=(
  [hgoal_each]="agent.coef_source=hgoal_each agent.acting=gpi"
  [actor_rel]="agent.acting=distill agent.eval_redistill=false agent.eval_goal_source=relabel"
)
TOTAL=60   # 1 arm x 3 seeds x 5 tasks x 2 readouts x 2 checkpoints
while true; do
  n=0
  for arm in psmgoal_ja_fbc3_mix_k70_antmaze; do
    for run in $PSM_DATA/exp/PSMFLows/${arm}/sd00*; do
      [ -d "$run" ] || continue; sd=$(basename "$run" | cut -c1-5)
      for tag in hgoal_each actor_rel; do
        for ep in 250000 500000; do
          ck=$run/params_${ep}.pkl; ready=0
          [ -f "$ck" ] && [ $(( $(date +%s) - $(stat -c %Y "$ck") )) -gt 120 ] && ready=1
          for t in 1 2 3 4 5; do
            out=${arm}_${sd}_${ep}_${tag}_task${t}
            if [ -f $MARK/$out ] || [ -f $PSM_DATA/logs/$out.json ]; then n=$((n+1)); continue; fi
            [ $ready = 1 ] || continue
            id=$(sbatch --parsable --nice=600 --partition=kisski-inference --account=general --time=04:00:00 \
              --job-name=e500_$out --output=$PSM_DATA/logs/slurm/%x-%j.out --error=$PSM_DATA/logs/slurm/%x-%j.err \
              --export=ALL,PSM_REPO=$REPO,PSM_DATA=$PSM_DATA,MODE=psmgoal,ENVKEY=antmaze,RUN_DIR=$run,OUT=$out,RESTORE_EPOCH=$ep,EVAL_WORKERS=4,ENV_NAME=antmaze-medium-navigate-singletask-task${t}-v0,EXTRA="${OV[$tag]}" \
              $REPO/scripts/slurm/eval500.sbatch) && { echo "$(date +%F_%T) submitted $out job $id"; echo $id > $MARK/$out; n=$((n+1)); }
          done
        done
      done
    done
  done
  [ $n -ge $TOTAL ] && { echo "$(date +%F_%T) ALLDONE $TOTAL evals submitted"; exit 0; }
  sleep 300
done

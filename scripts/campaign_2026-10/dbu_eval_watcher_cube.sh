#!/usr/bin/env bash
# Five-task eval500 of psmgoal_dbu_{sq,sm}_cube (uniform hindsight goals; the 2026-10-04
# ablation of docs/design/2026-10-03-psmgoal-data-bootstrap.md, expectations in
# $PSM_DATA/logs/dbu_hold_EXPECTED.txt). Readouts: hgoal and hgoal_each at 250k and 500k, lp at
# 250k only. MODE=psmgoal, ENVKEY=cube. Copy of db_eval_watcher.sh with the dbu groups and its
# own marker dir; eval jobs carry --nice=600 so they queue behind training.
# JSON: {arm}_{sd}_{step}_{tag}_task{t}.json in $PSM_DATA/logs. Exits when all 150 are submitted.
PSM_DATA=/mnt/home/amohan/psm-data; REPO=/mnt/home/amohan/git/Austin/PSMFLows
MARK=$PSM_DATA/gc_scripts/submitted_dbu_cube; mkdir -p $MARK $PSM_DATA/logs/slurm
declare -A OV=(
  [hgoal]="agent.coef_source=amortized agent.acting=gpi"
  [hgoal_each]="agent.coef_source=hgoal_each agent.acting=gpi"
  [lp]="agent.coef_source=lp agent.acting=gpi"
)
declare -A STEPS_OF=( [hgoal]="250000 500000" [hgoal_each]="250000 500000" [lp]="250000" )
TOTAL=150   # 2 arms x 3 seeds x 5 tasks x (2 + 2 + 1) readout-checkpoints
while true; do
  n=0
  for arm in psmgoal_dbu_sq_cube psmgoal_dbu_sm_cube; do
    for run in $PSM_DATA/exp/PSMFLows/${arm}/sd00*; do
      [ -d "$run" ] || continue; sd=$(basename "$run" | cut -c1-5)
      for tag in hgoal hgoal_each lp; do
        for ep in ${STEPS_OF[$tag]}; do
          ck=$run/params_${ep}.pkl; ready=0
          [ -f "$ck" ] && [ $(( $(date +%s) - $(stat -c %Y "$ck") )) -gt 120 ] && ready=1
          for t in 1 2 3 4 5; do
            out=${arm}_${sd}_${ep}_${tag}_task${t}
            if [ -f $MARK/$out ] || [ -f $PSM_DATA/logs/$out.json ]; then n=$((n+1)); continue; fi
            [ $ready = 1 ] || continue
            id=$(sbatch --parsable --nice=600 --partition=kisski-inference --account=general --time=03:00:00 \
              --job-name=e500_$out --output=$PSM_DATA/logs/slurm/%x-%j.out --error=$PSM_DATA/logs/slurm/%x-%j.err \
              --export=ALL,PSM_REPO=$REPO,PSM_DATA=$PSM_DATA,MODE=psmgoal,ENVKEY=cube,RUN_DIR=$run,OUT=$out,RESTORE_EPOCH=$ep,EVAL_WORKERS=4,ENV_NAME=cube-single-play-singletask-task${t}-v0,EXTRA="${OV[$tag]}" \
              $REPO/scripts/slurm/eval500.sbatch) && { echo "$(date +%F_%T) submitted $out job $id"; echo $id > $MARK/$out; n=$((n+1)); }
          done
        done
      done
    done
  done
  [ $n -ge $TOTAL ] && { echo "$(date +%F_%T) ALLDONE $TOTAL evals submitted"; exit 0; }
  sleep 300
done

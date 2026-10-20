#!/usr/bin/env bash
# Five-task eval500 of the advantage-guided arm (ARM, default psmgoal_ag_cube) at STEPS (default 250000):
# hgoal_each (best-of-64 by the measure) and the guided actor alone at ag_w_eval in {0.5, 1, 2, 4}
# (acting=distill from the run's flags; ag_w_eval is a typed override). One packed job per (run, readout).
PSM_DATA=/mnt/home/amohan/psm-data; REPO=/mnt/home/amohan/git/Austin/PSMFLows
arm=${ARM:-psmgoal_ag_cube}
MARK=$PSM_DATA/gc_scripts/submitted_$arm; mkdir -p $MARK $PSM_DATA/logs/slurm
declare -A OV=(
  [hgoal_each]="agent.coef_source=hgoal_each agent.acting=gpi"
  [ag_w05]="agent.ag_w_eval=0.5"
  [ag_w1]="agent.ag_w_eval=1.0"
  [ag_w2]="agent.ag_w_eval=2.0"
  [ag_w4]="agent.ag_w_eval=4.0"
)
STEPS=${STEPS:-"250000"}
TOTAL=${TOTAL:-15}
while true; do
  n=0
  for run in $PSM_DATA/exp/PSMFLows/${arm}/sd00*; do
    [ -d "$run" ] || continue; sd=$(basename "$run" | cut -c1-5)
    for tag in hgoal_each ag_w05 ag_w1 ag_w2 ag_w4; do
      for ep in $STEPS; do
        out=${arm}_${sd}_${ep}_${tag}
        if [ -f $MARK/$out ]; then n=$((n+1)); continue; fi
        done_json=0; for t in 1 2 3 4 5; do [ -f $PSM_DATA/logs/${out}_task${t}.json ] && done_json=$((done_json+1)); done
        [ $done_json = 5 ] && { n=$((n+1)); continue; }
        ck=$run/params_${ep}.pkl
        { [ -f "$ck" ] && [ $(( $(date +%s) - $(stat -c %Y "$ck") )) -gt 120 ]; } || continue
        id=$(sbatch --parsable --partition=kisski-inference --account=general --time=03:00:00 \
          --job-name=e500p_$out --output=$PSM_DATA/logs/slurm/%x-%j.out --error=$PSM_DATA/logs/slurm/%x-%j.err \
          --export="ALL,PSM_REPO=$REPO,PSM_DATA=$PSM_DATA,MODE=psmgoal,ENVKEY=cube,RUN_DIR=$run,OUT=$out,RESTORE_EPOCH=$ep,EVAL_WORKERS=4,EXTRA=${OV[$tag]}" \
          $REPO/scripts/slurm/eval500_packed.sbatch) && { echo "$(date +%F_%T) submitted $out job $id"; echo $id > $MARK/$out; n=$((n+1)); }
      done
    done
  done
  [ $n -ge $TOTAL ] && { echo "$(date +%F_%T) ALLDONE $TOTAL packed evals submitted"; exit 0; }
  sleep 300
done

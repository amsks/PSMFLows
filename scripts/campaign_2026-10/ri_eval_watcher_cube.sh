#!/usr/bin/env bash
# Five-task eval500 of the reward-index arm psmgoal_ri_fbc3_cube (docs/design/2026-10-19-reward-index.md)
# at 250k under the three readouts of the design: ri_actor (the flowbc actor at the ridge w), ri_gpi
# (best-of-64 by the goal mass at w) and ri_gpi_reward (best-of-64 by the expected relabel reward).
# coef_source=ridge and ridge_score=goal come from the run's own flags.json. One packed job per
# (run, readout) (scripts/slurm/eval500_packed.sbatch) writes <group>_<sd>_250000_<tag>_task<t>.json in
# $PSM_DATA/logs. No --nice: these decide the 2026-10-19 six-hour question. Exits when all 9 are submitted.
PSM_DATA=/mnt/home/amohan/psm-data; REPO=/mnt/home/amohan/git/Austin/PSMFLows
MARK=$PSM_DATA/gc_scripts/submitted_ri_cube; mkdir -p $MARK $PSM_DATA/logs/slurm
declare -A OV=(
  [ri_actor]="agent.acting=distill agent.eval_redistill=false"
  [ri_gpi]="agent.acting=gpi"
  [ri_gpi_reward]="agent.acting=gpi agent.ridge_score=reward"
)
STEPS=${STEPS:-"250000"}
TOTAL=${TOTAL:-9}   # 3 seeds x 1 checkpoint x 3 readouts
arm=psmgoal_ri_fbc3_cube
while true; do
  n=0
  for run in $PSM_DATA/exp/PSMFLows/${arm}/sd00*; do
    [ -d "$run" ] || continue; sd=$(basename "$run" | cut -c1-5)
    for tag in ri_actor ri_gpi ri_gpi_reward; do
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

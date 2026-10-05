#!/usr/bin/env bash
# Five-task eval500 of psmgoal_sm_code_cube (softmax loss, policy label = the binary code z, no
# actor) at 250k and 500k, two readouts per checkpoint. MODE=psmgoal.
#   trained: w = projected mean of w(z) over 256 fixed codes (the coefficient M was trained with)
#   lp:      the Lagrangian w
# JSON: {arm}_{sd}_{step}_{tag}_task{t}.json in $PSM_DATA/logs. Exits when all 60 are submitted.
PSM_DATA=/mnt/home/amohan/psm-data; REPO=/mnt/home/amohan/git/Austin/PSMFLows
MARK=$PSM_DATA/gc_scripts/submitted_sm_code; mkdir -p $MARK $PSM_DATA/logs/slurm
TAGS=(trained lp)
declare -A OV=(
  [trained]="agent.coef_source=trained agent.acting=gpi"
  [lp]="agent.coef_source=lp agent.acting=gpi"
)
while true; do
  n=0
  for arm in psmgoal_sm_code_cube; do
    for run in $PSM_DATA/exp/PSMFLows/${arm}/sd00*; do
      [ -d "$run" ] || continue; sd=$(basename "$run" | cut -c1-5)
      for ep in 250000 500000; do
        ck=$run/params_${ep}.pkl; ready=0
        [ -f "$ck" ] && [ $(( $(date +%s) - $(stat -c %Y "$ck") )) -gt 120 ] && ready=1
        for tag in "${TAGS[@]}"; do for t in 1 2 3 4 5; do
          out=${arm}_${sd}_${ep}_${tag}_task${t}
          if [ -f $MARK/$out ] || [ -f $PSM_DATA/logs/$out.json ]; then n=$((n+1)); continue; fi
          [ $ready = 1 ] || continue
          id=$(sbatch --parsable --partition=kisski-inference --account=general --time=03:00:00 \
            --job-name=e500_$out --output=$PSM_DATA/logs/slurm/%x-%j.out --error=$PSM_DATA/logs/slurm/%x-%j.err \
            --export=ALL,PSM_REPO=$REPO,PSM_DATA=$PSM_DATA,MODE=psmgoal,ENVKEY=cube,RUN_DIR=$run,OUT=$out,RESTORE_EPOCH=$ep,EVAL_WORKERS=4,ENV_NAME=cube-single-play-singletask-task${t}-v0,EXTRA="${OV[$tag]}" \
            $REPO/scripts/slurm/eval500.sbatch) && { echo "$(date +%F_%T) submitted $out job $id"; echo $id > $MARK/$out; n=$((n+1)); }
        done; done
      done
    done
  done
  [ $n -ge 60 ] && { echo "$(date +%F_%T) ALLDONE 60 evals submitted"; exit 0; }
  sleep 300
done

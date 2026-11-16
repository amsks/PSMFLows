#!/usr/bin/env bash
# Submits actor-alone 500-episode evals (five tasks) for every psmz_fix_* checkpoint as it lands; marks in submitted_eval.txt.
PSM_DATA=/mnt/home/amohan/psm-data; REPO=/mnt/home/amohan/git/Austin/PSMFLows; F=$REPO/psmgoal-vs-dfb-2026-11-09
while true; do n=0
 for run in $PSM_DATA/exp/PSMFLows/psmz_fix_*/sd00*/; do [ -d "$run" ] || continue; G=$(basename $(dirname $run)); sd=$(basename $run | cut -c1-5)
  case $G in *cd_*) K=cubedouble;; *scene*) K=scene;; *) K=cube;; esac
  for ck in $run/params_*.pkl; do [ -f "$ck" ] || continue; ep=$(basename $ck .pkl | sed 's/params_//'); [ $(( $(date +%s) - $(stat -c %Y "$ck") )) -gt 120 ] || continue
   out=${G}_${sd}_${ep}_actor_e500; grep -q "^$out " $F/submitted_eval.txt && continue
   for t in 1 2 3 4 5; do id=$(sbatch --parsable --partition=kisski-inference --account=general --time=04:00:00 --job-name=e5fix_${G#psmz_fix_}_${sd}_${ep}_t$t --output=$PSM_DATA/psmgoal-vs-dfb-2026-11-09/slurm/%x-%j.out --error=$PSM_DATA/psmgoal-vs-dfb-2026-11-09/slurm/%x-%j.err \
     --export="ALL,PSM_REPO=$REPO,PSM_DATA=$PSM_DATA,MODE=psmz,ENVKEY=$K,RUN_DIR=${run%/},OUT=$out,RESTORE_EPOCH=$ep,TASKS=$t,EVAL_WORKERS=4,EXTRA=agent.acting=distill agent.eval_redistill=false" $REPO/scripts/slurm/eval500_packed.sbatch) && { echo "$out task$t $id" >> $F/submitted_eval.txt; n=$((n+1)); }; done
   echo "$(date -u +%FT%TZ) submitted $out"; done; done
 [ $(ls $PSM_DATA/logs/psmz_fix_*_task*.json 2>/dev/null | wc -l) -ge 300 ] && exit 0; sleep 600; done

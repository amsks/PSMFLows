#!/usr/bin/env bash
# Five-task eval500 of the joint-actor arms psmgoal_ja_{fbc3_sh,fbc0_sh,dsrl_sh,fbc3_pt}_cube
# (2026-10-05; expectations in $PSM_DATA/logs/ja_runs_EXPECTED.txt). Readouts hgoal_each
# (best of 64 prior draws by M, each goal at its own h(g)) and actor_rel (the actor fed a
# rewarding dataset state) at 250k and 500k. One PACKED eval job per (run, step, readout)
# runs the five tasks sequentially (scripts/slurm/eval500_packed.sbatch) and writes the
# usual <group>_<sd>_<step>_<tag>_task<t>.json in $PSM_DATA/logs. Jobs carry --nice=600 so
# they queue behind training. Exits when all 48 packed jobs are submitted.
PSM_DATA=/mnt/home/amohan/psm-data; REPO=/mnt/home/amohan/git/Austin/PSMFLows
MARK=$PSM_DATA/gc_scripts/submitted_ja_cube; mkdir -p $MARK $PSM_DATA/logs/slurm
declare -A OV=(
  [hgoal_each]="agent.coef_source=hgoal_each agent.acting=gpi"
  [actor_rel]="agent.acting=distill agent.eval_redistill=false agent.eval_goal_source=relabel"
)
STEPS="250000 500000"
TOTAL=60   # 5 arms x 3 seeds x 2 checkpoints x 2 readouts
while true; do
  n=0
  for arm in psmgoal_ja_fbc3_sh_cube psmgoal_ja_fbc0_sh_cube psmgoal_ja_dsrl_sh_cube psmgoal_ja_fbc3_pt_cube psmgoal_ja_fbc03_sh_cube; do
    for run in $PSM_DATA/exp/PSMFLows/${arm}/sd00*; do
      [ -d "$run" ] || continue; sd=$(basename "$run" | cut -c1-5)
      for tag in hgoal_each actor_rel; do
        for ep in $STEPS; do
          out=${arm}_${sd}_${ep}_${tag}
          if [ -f $MARK/$out ]; then n=$((n+1)); continue; fi
          done_json=0; for t in 1 2 3 4 5; do [ -f $PSM_DATA/logs/${out}_task${t}.json ] && done_json=$((done_json+1)); done
          [ $done_json = 5 ] && { n=$((n+1)); continue; }
          ck=$run/params_${ep}.pkl
          { [ -f "$ck" ] && [ $(( $(date +%s) - $(stat -c %Y "$ck") )) -gt 120 ]; } || continue
          id=$(sbatch --parsable --nice=600 --partition=kisski-inference --account=general --time=03:00:00 \
            --job-name=e500p_$out --output=$PSM_DATA/logs/slurm/%x-%j.out --error=$PSM_DATA/logs/slurm/%x-%j.err \
            --export="ALL,PSM_REPO=$REPO,PSM_DATA=$PSM_DATA,MODE=psmgoal,ENVKEY=cube,RUN_DIR=$run,OUT=$out,RESTORE_EPOCH=$ep,EVAL_WORKERS=4,EXTRA=${OV[$tag]}" \
            $REPO/scripts/slurm/eval500_packed.sbatch) && { echo "$(date +%F_%T) submitted $out job $id"; echo $id > $MARK/$out; n=$((n+1)); }
        done
      done
    done
  done
  [ $n -ge $TOTAL ] && { echo "$(date +%F_%T) ALLDONE $TOTAL packed evals submitted"; exit 0; }
  sleep 300
done

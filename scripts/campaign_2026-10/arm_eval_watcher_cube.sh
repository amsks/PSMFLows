#!/usr/bin/env bash
# Generic five-task eval500 watcher for psmgoal cube arms (2026-10-17). For every run dir of
# every arm in ARMS, once params_<step>.pkl is complete, submits one PACKED eval job per
# (run, step, readout) (scripts/slurm/eval500_packed.sbatch, five tasks sequentially) and writes
# the usual <group>_<sd>_<step>_<tag>_task<t>.json in $PSM_DATA/logs. Jobs carry --nice=600 so
# they queue behind training. Exits when every (run, step, readout) is submitted or already
# evaluated.
#
#   ARMS="psmgoal_ja_fbc3_pt_bdsneg_cube psmgoal_ja_fbc3_pt_nf2_cube" STEPS="250000 500000" \
#   READOUTS="hgoal_each" nohup bash scripts/campaign_2026-10/arm_eval_watcher_cube.sh \
#     > $PSM_DATA/logs/slurm/watcher_bdsneg_nf2.log 2>&1 &
#
#   ARMS      space-separated run groups under $PSM_DATA/exp/PSMFLows (required)
#   STEPS     checkpoints (default "250000 500000")
#   READOUTS  hgoal_each and/or actor_rel (default "hgoal_each")
#   SEEDS_N   runs expected per arm (default 3); the watcher exits once ARMS x SEEDS_N x STEPS x READOUTS are done
PSM_DATA=${PSM_DATA:-/mnt/home/amohan/psm-data}; REPO=${PSM_REPO:-/mnt/home/amohan/git/Austin/PSMFLows}
: "${ARMS:?ARMS=<space-separated run groups> required}"
STEPS=${STEPS:-"250000 500000"}; READOUTS=${READOUTS:-hgoal_each}; SEEDS_N=${SEEDS_N:-3}
MARK=$PSM_DATA/gc_scripts/submitted_arm_${ENVKEY:-cube}; mkdir -p "$MARK" "$PSM_DATA/logs/slurm"
declare -A OV=(
  [hgoal_each]="agent.coef_source=hgoal_each agent.acting=gpi"
  [actor_rel]="agent.acting=distill agent.eval_redistill=false agent.eval_goal_source=relabel"
)
n_arms=$(wc -w <<<"$ARMS"); n_steps=$(wc -w <<<"$STEPS"); n_tags=$(wc -w <<<"$READOUTS")
TOTAL=$((n_arms * SEEDS_N * n_steps * n_tags))
echo "$(date +%F_%T) watching ARMS='$ARMS' STEPS='$STEPS' READOUTS='$READOUTS' TOTAL=$TOTAL"
while true; do
  n=0
  for arm in $ARMS; do
    for run in "$PSM_DATA"/exp/PSMFLows/"${arm}"/sd00*; do
      [ -d "$run" ] || continue; sd=$(basename "$run" | cut -c1-5)
      for tag in $READOUTS; do
        for ep in $STEPS; do
          out=${arm}_${sd}_${ep}_${tag}
          if [ -f "$MARK/$out" ]; then n=$((n+1)); continue; fi
          done_json=0; for t in 1 2 3 4 5; do [ -f "$PSM_DATA/logs/${out}_task${t}.json" ] && done_json=$((done_json+1)); done
          [ $done_json = 5 ] && { n=$((n+1)); continue; }
          ck=$run/params_${ep}.pkl
          { [ -f "$ck" ] && [ $(( $(date +%s) - $(stat -c %Y "$ck") )) -gt 120 ]; } || continue
          id=$(sbatch --parsable --nice=600 --partition=kisski-inference --account=general --time=${EVAL_TIME:-03:00:00} \
            --job-name=e500p_$out --output="$PSM_DATA"/logs/slurm/%x-%j.out --error="$PSM_DATA"/logs/slurm/%x-%j.err \
            --export="ALL,PSM_REPO=$REPO,PSM_DATA=$PSM_DATA,MODE=psmgoal,ENVKEY=${ENVKEY:-cube},RUN_DIR=$run,OUT=$out,RESTORE_EPOCH=$ep,EVAL_WORKERS=4,EXTRA=${OV[$tag]}" \
            "$REPO"/scripts/slurm/eval500_packed.sbatch) && { echo "$(date +%F_%T) submitted $out job $id"; echo "$id" > "$MARK/$out"; n=$((n+1)); }
        done
      done
    done
  done
  [ "$n" -ge "$TOTAL" ] && { echo "$(date +%F_%T) ALLDONE $TOTAL packed evals submitted"; exit 0; }
  sleep 300
done

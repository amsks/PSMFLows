#!/usr/bin/env bash
# 2026-09-09: finish the oscillation-stability campaign's verdict, for the two arms whose
# verdict is genuinely open. docs/design/2026-09-08-oscillation-stability.md 8.
#
# All 12 runs of that campaign are COMPLETE, but none of the four arms has a single
# 500-episode point, so `stability_ladder.py` scores the control from eval500 and every arm
# from the in-loop 50-episode CSV. Those two bases are not comparable and the tool says so.
# On the SAME (in-loop) basis, over the 300-500k window:
#
#   arm                 mean   swing    vs control 0.432
#   affine_strict_cube  0.432  0.353    (control)
#   tau1e3_cube         0.119  0.193    3.6x below; best seed 0.182 < control's worst 0.356
#   lrsf1e5_cube        0.104  0.173    4.2x below; best seed 0.127 < control's worst 0.356
#   oc1e4_cube          0.296  0.533    inside the band -- OPEN
#   oc1e4_lrsf1e5_cube  0.237  0.320    inside the band -- OPEN
#
# tau1e3 and lrsf1e5 are settled without eval500: every one of their seeds sits below the
# control's WORST seed, which no amount of 50-episode noise explains. They bought a small
# swing by collapsing the mean. The two ortho arms are inside the noise band, so only they
# are evaluated here: 2 arms x 3 seeds x 2 epochs = 12 jobs.
#
# Priority: these must never delay the Item 1 / DSRL-NA / Arm B / Arm C queue. Both belts
# are used -- `--nice` so the scheduler never prefers them, and `--dependency=afterany` on
# the long training jobs in flight, so they cannot occupy a GPU one of those is waiting for.
#
#   PSM_REPO=... PSM_DATA=... bash scripts/slurm/launch_stability_eval500.sh [dry]
set -euo pipefail
: "${PSM_REPO:?}"; : "${PSM_DATA:?}"
EXP="$PSM_DATA/exp/PSMFLows"
LOGDIR="$PSM_DATA/logs/slurm/e500stab"; mkdir -p "$LOGDIR"
STAB_GROUPS="${STAB_GROUPS:-oc1e4_cube oc1e4_lrsf1e5_cube}"  # not GROUPS: bash built-in
EPOCHS="${EPOCHS:-250000 500000}"

# Everything long that is already in flight. Short diagnostic jobs are deliberately not
# listed: they finish in minutes and waiting on them would only delay this pointlessly.
DEPS=$(squeue -u "$USER" -h -o "%i %j" \
       | grep -E "dsrlna_|usignal_arm_b|armc_" | awk '{print $1}' | paste -sd: -)
DEP_ARG=""
if [ -n "$DEPS" ]; then DEP_ARG="--dependency=afterany:$DEPS"; fi
echo "waiting behind: ${DEPS:-<nothing in flight>}"

n=0
for g in $STAB_GROUPS; do
  for sd in "$EXP/$g"/*/; do
    sd="${sd%/}"
    seed=$(basename "$sd" | cut -c1-6)
    for ep in $EPOCHS; do
      [ -f "$sd/params_${ep}.pkl" ] || { echo "SKIP $g/$seed @ $ep (no checkpoint)"; continue; }
      out="${g}_${seed}_${ep}"
      if [ "${1:-}" = "dry" ]; then echo "would submit $out"; n=$((n+1)); continue; fi
      # shellcheck disable=SC2086  -- DEP_ARG is one optional flag or empty
      jid=$(sbatch --partition=kisski-inference --account=general --gres=gpu:1 \
        --cpus-per-task=8 --mem=64G --time=02:00:00 --nice=10000 $DEP_ARG \
        --job-name="e500_$out" --output="$LOGDIR/%x-%j.out" --error="$LOGDIR/%x-%j.err" \
        --export="ALL,PSM_REPO=$PSM_REPO,PSM_DATA=$PSM_DATA,MODE=psmflow,ENVKEY=cube,RUN_DIR=$sd,OUT=$out,RESTORE_EPOCH=$ep" \
        scripts/slurm/eval500.sbatch | awk '{print $NF}')
      echo "submitted $jid  $out"
      n=$((n+1))
    done
  done
done
echo "$n jobs"
echo "then: .venv/bin/python tools/stability_ladder.py --exp \$PSM_DATA/exp/PSMFLows \\"
echo "  --groups affine_strict_cube,oc1e4_cube,oc1e4_lrsf1e5_cube --logs \$PSM_DATA/logs"

#!/usr/bin/env bash
# 2026-09-09 Arm C: shrunken-mixture latent sampling, all three published envs.
# docs/design/2026-09-08-critic-signal-and-dsrl-na.md 4.2.
#
# Same seam as Arm B -- the measure head is fitted at 4 latents per transition instead of 1,
# the extra 3 carrying the same (s, s') target -- but the extra latents come from the stored
# EM posterior with each component's covariance scaled by c^2, instead of from a round ball.
#
# Why the shape should matter: the EM covariance is stretched along the directions in which
# the decode barely moves, which are the directions the preimage set actually extends along.
# A round ball spends its budget equally on directions that change the action. MEASURED, at
# a matched decode error (tools/diag_mixture_decode.py, 4096 rows per env):
#
#   env        c    err    dist     matched jitter          ratio
#   cube      0.5  0.119   1.675    sigma 0.5: 0.124/1.068  1.57x further
#   antmaze   1.0  0.314   1.574    sigma 0.3: 0.326/0.833  1.89x further
#   pointmaze 0.5  0.130   1.236    sigma 0.1: 0.120/0.125  9.9x further
#
# so the hypothesis holds on all three, most strongly on pointmaze (d_a = 2, where a round
# ball is nearly useless). c is the largest whose mean decode error stays inside the point
# inverse's own p90 for that env; on pointmaze c=1.0 sat AT the gate and was called marginal,
# so the dose one step down is used.
#
# Note what c does NOT do: c -> 0 collapses onto the posterior MEAN, not onto u_data, and
# that mean is 1.14 / 1.74 / 2.79 away from the point inverse. Shrinking cannot buy fidelity
# below the mean's own displacement, which is why antmaze's sweep is flat in c.
#
# Every env runs on the SAME npz its affine_strict control trained on, so there is no
# preimage confound anywhere and no extra control seed is needed. That required reading the
# mixture out of a legacy-target npz; see main.py for why the prior_scale gate does not
# apply to this path (it asks a question about the latent ACTOR's sampling distribution, and
# on cube it points the wrong way -- the prior_scale=0.691 file's samples decode WORSE).
#
#   PSM_REPO=... PSM_DATA=... bash scripts/slurm/launch_arm_c.sh [table|smoke] [envkey]
set -euo pipefail
: "${PSM_REPO:?}"; : "${PSM_DATA:?}"
LOGDIR="$PSM_DATA/logs/slurm/armc"; mkdir -p "$LOGDIR"
SEEDS="${SEEDS:-0 1 2}"
STEPS="${STEPS:-500000}"

# envkey | preimage npz basename | shrink c | control group
ROWS=(
  "cube|cube-single-play|0.5|affine_strict_cube"
  "antmaze|antmaze-medium-navigate|1.0|affine_strict_antmaze_g99"
  "pointmaze|pointmaze-medium-navigate|0.5|affine_strict_pointmaze"
)

for row in "${ROWS[@]}"; do
  IFS='|' read -r envkey npz c ctl <<<"$row"
  if [ -n "${2:-}" ] && [ "$2" != "$envkey" ]; then continue; fi
  NPZ="$PSM_DATA/preimages/${npz}.npz"
  OV="agent.measure_u_samples=4 agent.measure_u_source=mixture agent.measure_u_mixture_shrink=$c"
  GROUP="armc_${envkey}"
  if [ "${1:-}" = "table" ]; then
    echo "$GROUP  seeds='$SEEDS' steps=$STEPS  control=$ctl"
    echo "  npz  $NPZ"
    echo "$OV" | tr ' ' '\n' | sed '/^$/d' | sed 's/^/  /'
    continue
  fi
  [ -f "$NPZ" ] || { echo "no such preimage npz: $NPZ" >&2; exit 1; }
  if [ "${1:-}" = "smoke" ]; then
    sbatch --partition=kisski-inference --account=general --gres=gpu:1 \
      --cpus-per-task=8 --mem=64G --time=00:40:00 --job-name="armc_smoke_${envkey}" \
      --output="$LOGDIR/%x-%j.out" --error="$LOGDIR/%x-%j.err" \
      --export=ALL,PSM_REPO="$PSM_REPO",PSM_DATA="$PSM_DATA",ENVKEY="$envkey",\
PREIMAGES="$NPZ",GROUP="armc_smoke_${envkey}",SEED=0,STEPS=200,\
EVAL_INT=200,EVAL_EPS=2,LOG_INT=50,SAVE_INT=1000000,EXTRA="$OV" \
      "$PSM_REPO/scripts/slurm/train_psmflow.sbatch"
    continue
  fi
  for s in $SEEDS; do
    # gamma comes from train_psmflow.sbatch's per-env default: 0.99 on antmaze
    # (docs/design/2026-09-06-antmaze-failure-tests.md), 0.98 elsewhere.
    sbatch --partition=kisski-inference --account=general --gres=gpu:1 \
      --cpus-per-task=8 --mem=64G --time=12:00:00 --job-name="${GROUP}_sd${s}" \
      --output="$LOGDIR/%x-%j.out" --error="$LOGDIR/%x-%j.err" \
      --export=ALL,PSM_REPO="$PSM_REPO",PSM_DATA="$PSM_DATA",ENVKEY="$envkey",\
PREIMAGES="$NPZ",GROUP="$GROUP",SEED="$s",STEPS="$STEPS",SAVE_INT=50000,EXTRA="$OV" \
      "$PSM_REPO/scripts/slurm/train_psmflow.sbatch"
  done
done
[ "${1:-}" = "table" ] && exit 0
echo
echo "score with: .venv/bin/python tools/stability_ladder.py --exp \$PSM_DATA/exp/PSMFLows \\"
echo "  --groups affine_strict_cube,armc_cube,affine_strict_antmaze_g99,armc_antmaze,\\"
echo "affine_strict_pointmaze,armc_pointmaze --logs \$PSM_DATA/logs"

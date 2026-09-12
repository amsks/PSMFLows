#!/usr/bin/env bash
# 2026-09-08 Item 3: raise the signal over u, or lower the noise around it. Zero-shot.
# docs/design/2026-09-08-critic-signal-and-dsrl-na.md 4.
#
# The problem both arms attack: over a fixed candidate roster, the std over u of
# max_{u'} psi(s,u,u')^T w is 57 while the two critics disagree by ~124, so the deployed
# argmax is mostly a max over the ensemble's own noise. Two independent handles:
#
#   arm_a   DROPPED BEFORE LAUNCH, 2026-09-08. It was: num_parallel 2 -> 8 with the
#           ensemble MEAN at acting (gpi_select=mean), on the premise that the argmax is a
#           max over ensemble noise. Item 1 killed the premise -- a frozen FQL expert's
#           action critic, trained by a real max-backup on real rewards and with a direct
#           action pathway, ranks the same roster at rho +0.066, i.e. no better than the
#           measure. Two independently-trained critics missing the same signal is not a
#           noise problem that averaging eight of them fixes. The command it would have
#           run, if the 256-state pass reverses that:
#             agent.num_parallel=8 agent.gpi_select=mean, npz $STD_NPZ, control
#             affine_strict_cube (mean 0.415, swing 0.371, step 0.163).
#
#   arm_b   RAISE THE SIGNAL. measure_u_samples 1 -> 4: the measure head is fitted at four
#           latents per transition instead of one, the extra three carrying the same
#           (s, s') target.
#
#           NOT from the stored EM posterior, which was the plan until it was measured.
#           tools/diag_mixture_decode.py, 4096 cube rows, ||a|| = 0.875, one-step decode:
#
#             latent            ||G(s,u) - a||   fraction of the way from point to prior
#             point inverse           0.0885                 0.00   (p90 0.128)
#             posterior mean          0.1014                 0.07
#             posterior SAMPLE        0.1667                 0.40
#             prior draw              0.2851                 1.00
#
#           A posterior sample is 40% of the way to an uninverted prior draw (60% on the
#           prior_scale=0.691 npz, the only one main.py lets read the mixture). Those are
#           not other preimages of the same action, so three of every four rows would carry
#           the point inverse's target while decoding elsewhere -- the fiction
#           `mask_invalid_preimages` exists to prevent, at 75% of the loss.
#
#           So the extra latents are a JITTER ball around u_data, with the width taken off
#           the same probe's ladder rather than from the inversion temperature:
#
#             sigma      0.1     0.2     0.3     0.5     0.75    1.0
#             decode   0.090   0.096   0.103   0.121   0.154   0.190
#
#           The point inverse's own p90 is 0.128, so at sigma <= 0.5 the extra latents
#           still decode to this transition's action. Two doses, 0.3 and 0.5. This runs on
#           the SAME npz as affine_strict_cube, so that arm is the control directly and no
#           npz-matched control is needed.
#           Honest limit, pre-registered: a ball of radius ~0.6 around u_data in a box of
#           half-width 3 teaches the head the LOCAL shape of Q near the data latents, while
#           the deployed argmax scans the whole prior box. Expect the extrapolation problem
#           to be reduced, not removed.
#
#   PSM_REPO=... PSM_DATA=... bash scripts/slurm/launch_u_signal.sh [smoke|table] [arm]
set -euo pipefail
: "${PSM_REPO:?}"; : "${PSM_DATA:?}"
ENVKEY=cube; FLOW=cube-single-play
LOGDIR="$PSM_DATA/logs/slurm/usignal"; mkdir -p "$LOGDIR"
SEEDS="${SEEDS:-0 1 2}"
STEPS="${STEPS:-500000}"

# The npz every affine_strict_cube run trained on, so that arm is the control for all three.
STD_NPZ="$PSM_DATA/preimages/$FLOW.npz"
JIT="agent.measure_u_samples=4 agent.measure_u_source=jitter"

# name       npz        overrides
ROWS=(
  "arm_b03|$STD_NPZ|$JIT agent.measure_u_jitter_std=0.3"
  "arm_b05|$STD_NPZ|$JIT agent.measure_u_jitter_std=0.5"
)

if [ "${1:-}" = "table" ]; then
  for row in "${ROWS[@]}"; do
    IFS='|' read -r name npz ov <<<"$row"
    echo "usignal_${name}_cube  seeds='$SEEDS' steps=$STEPS"
    echo "  npz  $npz"
    echo "$ov" | tr ' ' '\n' | sed '/^$/d' | sed 's/^/  /'
  done
  exit 0
fi

for row in "${ROWS[@]}"; do
  IFS='|' read -r name npz ov <<<"$row"
  if [ -n "${2:-}" ] && [ "$2" != "$name" ]; then continue; fi
  [ -f "$npz" ] || { echo "no such preimage npz: $npz" >&2; exit 1; }
  GROUP="usignal_${name}_cube"
  if [ "${1:-}" = "smoke" ]; then
    sbatch --partition=kisski-inference --account=general --gres=gpu:1 \
      --cpus-per-task=8 --mem=64G --time=00:40:00 --job-name="usig_smoke_${name}" \
      --output="$LOGDIR/%x-%j.out" --error="$LOGDIR/%x-%j.err" \
      --export=ALL,PSM_REPO="$PSM_REPO",PSM_DATA="$PSM_DATA",ENVKEY="$ENVKEY",\
PREIMAGES="$npz",GROUP="usignal_smoke_${name}",SEED=0,STEPS=200,\
EVAL_INT=200,EVAL_EPS=2,LOG_INT=50,SAVE_INT=1000000,EXTRA="$ov" \
      "$PSM_REPO/scripts/slurm/train_psmflow.sbatch"
    continue
  fi
  for s in $SEEDS; do
    sbatch --partition=kisski-inference --account=general --gres=gpu:1 \
      --cpus-per-task=8 --mem=64G --time=12:00:00 --job-name="${GROUP}_sd${s}" \
      --output="$LOGDIR/%x-%j.out" --error="$LOGDIR/%x-%j.err" \
      --export=ALL,PSM_REPO="$PSM_REPO",PSM_DATA="$PSM_DATA",ENVKEY="$ENVKEY",\
PREIMAGES="$npz",GROUP="$GROUP",SEED="$s",STEPS="$STEPS",SAVE_INT=50000,EXTRA="$ov" \
      "$PSM_REPO/scripts/slurm/train_psmflow.sbatch"
  done
done
echo
echo "score with: .venv/bin/python tools/stability_ladder.py --exp \$PSM_DATA/exp/PSMFLows \\"
echo "  --groups affine_strict_cube,usignal_arm_b03_cube,usignal_arm_b05_cube \\"
echo "  --logs \$PSM_DATA/logs"

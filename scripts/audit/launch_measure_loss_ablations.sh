#!/usr/bin/env bash
# Measure-loss audit ablations (docs/design/2026-09-08-measure-loss-audit.md §7).
#
# antmaze-medium-navigate at discount=0.995 -- the fastest-failing setting, where the
# divergence is fully visible by 150k. One seed per job, one GPU per job, 150k steps,
# save + in-loop eval at 50k/100k/150k, 3 h wall.
#
# EVERY arm is an existing config key. The two levers the brief asked for that do NOT
# exist:
#   * agent.num_parallel=1  -- UNRUNNABLE. utils/psm_common.targets_uncertainty divides by
#     (P^2 - P) = 0, so `unc` is NaN and the TD target is NaN from step 1. Verified
#     directly. `agent.pessimism_penalty=0.0` at P=2 is the same semantics (plain ensemble
#     mean target) without the NaN, and is arm A1.
#   * a diagonal-vs-off-diagonal weighting knob -- DOES NOT EXIST. contrastive_loss
#     hardcodes `* P` on the diagonal and `0.5 / (B^2 - B)` on the off-diagonal. Ablating
#     it would need a copy of the agent under scripts/audit/, deliberately not created.
#
# Usage:
#   SMOKE=1 bash scripts/audit/launch_measure_loss_ablations.sh    # one 400-step wiring job
#   bash scripts/audit/launch_measure_loss_ablations.sh            # the 9 real jobs
#   DRY=1  ...                                                     # print, submit nothing
set -euo pipefail

PSM_REPO="${PSM_REPO:-/mnt/home/amohan/git/Austin/PSMFLows}"
PSM_DATA="${PSM_DATA:-/mnt/home/amohan/psm-data}"
PREIMAGES="${PREIMAGES:-$PSM_DATA/preimages/antmaze-medium-navigate.npz}"
OGBENCH_DATASET_DIR="${OGBENCH_DATASET_DIR:-/mnt/home/amohan/.ogbench/data}"
PART="${PART:-kisski-inference}"
ACCT="${ACCT:-general}"
DRY="${DRY:-0}"
SMOKE="${SMOKE:-0}"

[ -f "$PREIMAGES" ] || { echo "no such preimage npz: $PREIMAGES" >&2; exit 1; }

submit () {  # submit <jobname> <group> <seed> <extra>
  local name="$1" group="$2" seed="$3" extra="$4"
  local exp="ALL,PSM_REPO=$PSM_REPO,PSM_DATA=$PSM_DATA,OGBENCH_DATASET_DIR=$OGBENCH_DATASET_DIR"
  exp="$exp,ENVKEY=antmaze,PREIMAGES=$PREIMAGES,DISCOUNT=$DISCOUNT,SEED=$seed,GROUP=$group"
  exp="$exp,STEPS=$STEPS,EVAL_INT=$EVAL_INT,EVAL_EPS=$EVAL_EPS,SAVE_INT=$SAVE_INT,LOG_INT=$LOG_INT"
  exp="$exp,EXTRA=$extra"
  echo "--- $name  group=$group seed=$seed"
  echo "    EXTRA='$extra'  steps=$STEPS discount=$DISCOUNT"
  if [ "$DRY" = "1" ]; then return 0; fi
  sbatch --partition="$PART" --account="$ACCT" --gres=gpu:1 --time="$WALL" \
         --job-name="$name" --export="$exp" \
         "$PSM_REPO/scripts/slurm/train_psmflow.sbatch"
}

DISCOUNT=0.995

if [ "$SMOKE" = "1" ]; then
  # One wiring job: every novel override value at once, 400 steps, no eval, no save.
  # Confirms hydra accepts the keys and that pessimism_penalty=0.0 does not NaN.
  STEPS=400; EVAL_INT=1000000; EVAL_EPS=1; SAVE_INT=1000000; LOG_INT=100; WALL=00:30:00
  submit mla_smoke mla_smoke 0 \
    "agent.pessimism_penalty=0.0 agent.ortho_coef=100000.0 agent.lr_sf=1.0e-5 agent.tau=0.001"
  exit 0
fi

STEPS=150000; EVAL_INT=50000; EVAL_EPS=10; SAVE_INT=50000; LOG_INT=5000; WALL=03:00:00

# A1  no target pessimism -- THE test of H-PESS.        predicted BOUNDED
submit mla_pess0_sd0   mla_g995_pess0    0 "agent.pessimism_penalty=0.0"
submit mla_pess0_sd1   mla_g995_pess0    1 "agent.pessimism_penalty=0.0"
# A2  half pessimism  -- dose-response below baseline.  predicted 5.1e5 steps/decade
submit mla_pess025_sd0 mla_g995_pess025  0 "agent.pessimism_penalty=0.25"
# A3  double pessimism -- dose-response above baseline. predicted 9.5e3 steps/decade
submit mla_pess10_sd0  mla_g995_pess10   0 "agent.pessimism_penalty=1.0"
# A5/A6  stronger orthonormality.                       predicted RATE UNCHANGED
submit mla_oc1e4_sd0   mla_g995_oc1e4    0 "agent.ortho_coef=10000.0"
submit mla_oc1e5_sd0   mla_g995_oc1e5    0 "agent.ortho_coef=100000.0"
# A7  slower psi.                                       predicted < 10x slower
submit mla_lrsf1e5_sd0 mla_g995_lrsf1e5  0 "agent.lr_sf=1.0e-5"
# A8  slower Polyak.                                    predicted ~10x slower
submit mla_tau1e3_sd0  mla_g995_tau1e3   0 "agent.tau=0.001"
# A9  free psi -- did the affine head move gamma_crit?
submit mla_freepsi_sd0 mla_g995_freepsi  0 "agent.psi_form=free"

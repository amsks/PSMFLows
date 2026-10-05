#!/usr/bin/env bash
#SBATCH --partition=kisski-inference
#SBATCH --account=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --job-name=fourrooms_rhat
# tools/diag_fourrooms_induced_reward.py, learned stage: one encoding (ENC=onehot|xyphase),
# SEEDS in parallel on one GPU. Each run trains the psmgoal measure on four-rooms data, then
# compares M as critic with scalar critics on induced rewards for 8 goals.
# Optional: STEPS (measure steps, default 30000), QSTEPS (xyphase Q-network steps, default 50000),
# LPSTEPS (0 = psmgoal.yaml num_inference_steps), OUT (run directory).
set -uo pipefail
export PSM_DATA=/mnt/home/amohan/psm-data
export PSM_REPO=/mnt/home/amohan/git/Austin/PSMFLows
export XLA_PYTHON_CLIENT_PREALLOCATE=false
cd "$PSM_REPO"
OUT=${OUT:-$PSM_DATA/logs/fourrooms_induced_reward}
mkdir -p "$OUT"
pids=()
for sd in ${SEEDS:-0 1 2}; do
  .venv/bin/python tools/diag_fourrooms_induced_reward.py --stage learned --enc "$ENC" --seed "$sd" \
    --steps "${STEPS:-30000}" --q_steps "${QSTEPS:-50000}" --lp_steps "${LPSTEPS:-0}" --reuse_measure \
    --report_out "$OUT/${ENC}_sd${sd}.json" > "$OUT/${ENC}_sd${sd}.log" 2>&1 &
  pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
for sd in ${SEEDS:-0 1 2}; do echo "=== ${ENC} sd${sd}"; tail -n 12 "$OUT/${ENC}_sd${sd}.log"; done
exit $rc

#!/usr/bin/env bash
# E2: re-evaluate the PSMFlow point-arm checkpoints with the EXACT (100-step ODE)
# decoder instead of the one-step distilled net the agent deploys with.
#
# The point preimage is exact under ODE-100 (decode error 0.00012) but off by 0.0886
# under the one-step net -- ~10% of action scale. This asks how much of the deployed
# number that mismatch costs. Evaluation only: no training, no new checkpoints, and no
# code change (agents/psmflow.py `decode` already branches on gpi_decode).
#
# The paired control is re-measured here rather than quoted. The 0.236 +/- 0.071 baseline
# comes from eval500_latentpsm_cube_sd0..4.json, written 08-05/08-10 -- BEFORE the 08-14
# fix that pinned the action-noise stream to cfg.seed. Those runs drew their action stream
# from OS entropy, so they are not reproducible and cannot be differenced against a
# pinned-stream ODE number: re-running sd0 onestep under today's harness gives 0.240
# [0.205, 0.279] against the recorded 0.318. Every seed is therefore re-evaluated under
# BOTH decoders here, and the ODE-vs-onestep comparison is made only within this script.
#
# Usage: GPU=0 bash scripts/reeval_ode_e2.sh
set -euo pipefail

REPO=/u/amsks/git/PSMFLows
RUNS=/data-local/amsks/PSMFLows/exp/PSMFLows/psmflow_latentpsm_cube_021456
GPU="${GPU:-0}"
cd "$REPO"

declare -A SD=(
  [0]="$RUNS/sd000_20260805_021459"
  [1]="$RUNS/sd001_20260805_100228"
  [2]="$RUNS/sd002_20260805_211248"
  [3]="$RUNS/sd003_20260805_211248"
  [4]="$RUNS/sd004_20260806_000039"
)

for s in 0 1 2 3 4; do
  echo "=== sd$s acting=actor, onestep (paired control) ==="
  GPU="$GPU" bash scripts/eval500.sh psmflow cube "${SD[$s]}" "e2_onestep_actor_sd$s" \
    agent.gpi_decode=onestep
done

for s in 0 1 2 3 4; do
  echo "=== sd$s acting=gpi, onestep (paired control) ==="
  GPU="$GPU" bash scripts/eval500.sh psmflow cube "${SD[$s]}" "e2_onestep_gpi_sd$s" \
    agent.acting=gpi agent.gpi_decode=onestep
done

for s in 0 1 2 3 4; do
  echo "=== sd$s acting=actor, ODE-100 ==="
  GPU="$GPU" bash scripts/eval500.sh psmflow cube "${SD[$s]}" "e2_ode_actor_sd$s" \
    agent.gpi_decode=ode agent.flow_decode_steps=100
done

for s in 0 1 2 3 4; do
  echo "=== sd$s acting=gpi, ODE-100 ==="
  GPU="$GPU" bash scripts/eval500.sh psmflow cube "${SD[$s]}" "e2_ode_gpi_sd$s" \
    agent.acting=gpi agent.gpi_decode=ode agent.flow_decode_steps=100
done

echo "all E2 reports in /data-local/amsks/PSMFLows/logs/e2_*.json"

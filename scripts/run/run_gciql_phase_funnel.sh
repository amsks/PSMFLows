#!/usr/bin/env bash
# Re-run gciql_profile on the 10 state seeds with the extended phase_funnel
# columns (final_cube_lift, final_grip, final_cube_goal_dist) so we can split
# transport-fails into grasp-lost vs misplaced for GCIQL.
# Uses .venv-jax-cpu (jax). Each seed: 5 tasks x 10 episodes.
set -euo pipefail

cd "$(dirname "$0")/.."

RUN_ROOT="results/gciql_20260518_201030/factored-fb/factored-fb-gciql"
OUT_ROOT="analysis/profiles/gciql_profile_v2"
DATA_PATH="datasets/cube-single-play-v0"

mkdir -p "$OUT_ROOT"
for sd in $(ls "$RUN_ROOT" | sort); do
  # sd001 -> s1, sd010 -> s10
  s_num=$(echo "$sd" | sed -E 's/^sd0*([0-9]+).*$/\1/')
  out="$OUT_ROOT/s${s_num}_final"
  if [[ -f "$out/phase_funnel.parquet" ]]; then
    echo "[skip] s${s_num} already done"
    continue
  fi
  echo "[probe] s${s_num} <- $sd -> $out"
  .venv-jax-cpu/bin/python -m scripts.profiles.gciql_profile \
    --run-dir "$RUN_ROOT/$sd" --step 1000000 \
    --out "$out" --tasks 1,2,3,4,5 --n-episodes 10 \
    --dataset-path "$DATA_PATH" \
    > "$out.log" 2>&1
done
echo "[done] -> $OUT_ROOT"

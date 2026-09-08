#!/usr/bin/env bash
# 2026-09-08 gpi_num_u sweep. K is the size of Lambda_K, the prior roster GPI's argmax
# scans. docs/design/2026-09-08-measure-loss-audit.md 8.4 measured Spearman(Q, MC) at
# 0.06-0.16 with a negative 10th percentile, so max over K draws is close to a max over
# noise and should get WORSE with K. K=8 already took the dead cube seed 0.002 -> 0.140.
#
# Eval only: no training, checkpoints are frozen, the sole hydra override is gpi_num_u.
# Every other key comes from each run's own flags.json (tools/eval_checkpoint.py).
#
#   PSM_REPO=... PSM_DATA=... bash launch_gpi_k.sh [smoke]
set -euo pipefail
: "${PSM_REPO:?}"; : "${PSM_DATA:?}"
EXP="$PSM_DATA/exp/PSMFLows"
LOGDIR="$PSM_DATA/logs/slurm/gpik"; mkdir -p "$LOGDIR"
SB=(--partition=kisski-inference --account=general --gres=gpu:1 --cpus-per-task=8 --mem=64G)

# name              envkey   group                     seed_dir_glob     epoch  why
ROWS=(
  "cubeH0   cube     affine_strict_cube        sd000  400000  healthy_kappa05_default"
  "cubeH1   cube     affine_strict_cube        sd001  400000  healthy_kappa05_default"
  "cubeD0   cube     affine_strict_cube_mpess0 sd000  350000  dead_seed_base0.002"
  "amzH2    antmaze  affine_strict_antmaze_g99 sd002  400000  best_antmaze_seed"
)
KS=(4 8 16 32 64)
[ "${1:-}" = "smoke" ] && KS=(8) && ROWS=("${ROWS[2]}")

for row in "${ROWS[@]}"; do
  read -r name envkey group sd epoch why <<<"$row"
  rd=$(ls -d "$EXP/$group/${sd}"* 2>/dev/null | head -1)
  [ -z "$rd" ] && { echo "MISSING $group/$sd" >&2; exit 1; }
  [ -f "$rd/params_${epoch}.pkl" ] || { echo "MISSING $rd/params_${epoch}.pkl" >&2; exit 1; }
  for k in "${KS[@]}"; do
    out="gpik_${name}_${epoch}_K${k}"
    if [ "${1:-}" = "smoke" ]; then
      echo "SMOKE would submit: $out  rd=$rd  why=$why"; continue
    fi
    sbatch "${SB[@]}" --time=02:00:00 --job-name="$out" \
      --output="$LOGDIR/%x-%j.out" --error="$LOGDIR/%x-%j.err" \
      --export=ALL,MODE=psmflow,ENVKEY="$envkey",RUN_DIR="$rd",OUT="$out",RESTORE_EPOCH="$epoch",EXTRA="agent.gpi_num_u=$k" \
      "$PSM_REPO/scripts/slurm/eval500.sbatch"
  done
done

#!/usr/bin/env bash
# 2026-09-08 Item 2's agreed diagnostic: does a REWARD-SPECIFIC latent critic rank the same
# roster the zero-shot measure cannot? docs/design/2026-09-08-critic-signal-and-dsrl-na.md 4.1.
#
# Same 256 onset states, same PRNGKey(12345) roster, same cached returns as the Item 1
# table -- so the two new rows (`na_qw`, `na_qa`) drop straight into it. Every ranker is
# also scored on the roster subset inside the DSRL-NA arm's own box (INBOX_CLIP=1.5),
# because qw was never fitted outside it and a bare low number would be ambiguous.
#
# RUN_DIR is pinned to one affine_strict checkpoint throughout: the na rows do not depend on
# it, and holding it fixed keeps the psi rows identical across these jobs so a difference
# can only come from the na run.
#
# Run only once the DSRL-NA checkpoints exist (dsrlna_cube, 250k and 500k).
#   PSM_REPO=... PSM_DATA=... bash scripts/slurm/launch_gpi_selection_na.sh [dry]
set -euo pipefail
: "${PSM_REPO:?}"; : "${PSM_DATA:?}"
EXP="$PSM_DATA/exp/PSMFLows"
LOGDIR="$PSM_DATA/logs/slurm/gpisel"; mkdir -p "$LOGDIR"
CACHE="$PSM_DATA/logs/diag_gpi_selection_mc_cache_cube_n256.npz"
ORACLE=$(ls -d "$EXP"/fqlexpert_cube_a300/sd000* 2>/dev/null | head -1)
PSI_RUN=$(ls -d "$EXP"/affine_strict_cube/sd000* 2>/dev/null | head -1)
PSI_EPOCH="${PSI_EPOCH:-350000}"
EPOCHS="${EPOCHS:-250000 500000}"
[ -f "$CACHE" ] || { echo "no MC cache at $CACHE -- run launch_gpi_selection_onsets.sh with N_MC=256 first" >&2; exit 1; }
SB=(--partition=kisski-inference --account=general --gres=gpu:1 --cpus-per-task=8
    --mem=64G --time=02:00:00)

for na in "$EXP"/dsrlna_cube/*/; do
  na="${na%/}"
  sd=$(basename "$na" | cut -c1-6)
  for ep in $EPOCHS; do
    [ -f "$na/params_${ep}.pkl" ] || { echo "SKIP $sd @ $ep (no checkpoint yet)"; continue; }
    tag="cube_na_${sd}_${ep}"
    if [ "${1:-}" = "dry" ]; then echo "would submit $tag  na=$na"; continue; fi
    jid=$(sbatch "${SB[@]}" --job-name="gpisel_na_${sd}_${ep}" \
      --output="$LOGDIR/%x-%j.out" --error="$LOGDIR/%x-%j.err" \
      --export="ALL,PSM_REPO=$PSM_REPO,PSM_DATA=$PSM_DATA,ENVKEY=cube,RUN_DIR=$PSI_RUN,EPOCH=$PSI_EPOCH,TAG=$tag,N_MC=256,MC_CACHE=$CACHE,INBOX_CLIP=1.5,ORACLE_DIR=$ORACLE,NA_DIR=$na,NA_EPOCH=$ep" \
      scripts/slurm/diag_gpi_selection.sbatch | awk '{print $NF}')
    echo "submitted $jid  $tag"
  done
done
echo "reports -> $PSM_DATA/logs/diag_gpi_selection_cube_na_*_*.json"

#!/usr/bin/env bash
# scripts/train/fb_after_crl.sh — wait until the GPUs are free (the CRL state campaign has
# finished), then auto-launch the coverage-balanced FB-reweighted sweep. Detached:
#   setsid bash scripts/train/fb_after_crl.sh > /dev/shm/fb_after_crl.log 2>&1 &
set -uo pipefail
trap '' HUP
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"

THRESH_MIB="${THRESH_MIB:-20000}"   # each GPU must have >= this MiB free to count as idle
POLL="${POLL:-120}"                 # seconds between checks
MAX_POLLS="${MAX_POLLS:-720}"       # ~24h safety cap, then give up (don't launch)
export ALPHA="${ALPHA:-0.5}"

echo "[fb-after-crl] $(date) waiting until all GPUs have >= ${THRESH_MIB} MiB free (poll ${POLL}s, cap ${MAX_POLLS})"
for ((i = 0; i < MAX_POLLS; i++)); do
    minfree=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | sort -n | head -1)
    echo "[fb-after-crl] $(date +%T) min_free=${minfree:-?} MiB"
    if [ -n "${minfree:-}" ] && [ "$minfree" -ge "$THRESH_MIB" ]; then
        echo "[fb-after-crl] $(date +%T) GPUs free -> launching FB sweep (alpha=$ALPHA)"
        exec env REWEIGHT_ALPHA="$ALPHA" bash scripts/agents/fb/state.sh
    fi
    sleep "$POLL"
done
echo "[fb-after-crl] gave up after $MAX_POLLS polls; GPUs never freed."
echo "[fb-after-crl] launch manually with: REWEIGHT_ALPHA=$ALPHA bash scripts/agents/fb/state.sh"

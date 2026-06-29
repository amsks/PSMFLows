#!/usr/bin/env bash
# scripts/train/monitor_crl_campaign.sh — periodic health check for the CRL campaign.
#
# Every INTERVAL seconds, for the latest logs/crl_flowbc_campaign_* dir, prints a
# per-run status line: state (TRAIN/EVAL/DONE/CRASHED/STALLED), latest step + %,
# training it/s, and the current eval rate (s/episode) when an eval is running.
# Alerts (!!!) on a Python Traceback or a log that stopped updating. Exits once
# every run is DONE or CRASHED. Output goes to stdout (tee'd to a log by the
# tmux launcher). Liveness = log mtime (tqdm refreshes the bar during both train
# and eval, so a stale log means genuinely stuck).
#
#   bash scripts/train/monitor_crl_campaign.sh            # 15-min ticks, latest campaign
#   INTERVAL=300 CDIR=logs/crl_flowbc_campaign_X bash scripts/train/monitor_crl_campaign.sh
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
INTERVAL="${INTERVAL:-900}"
STALE="${STALE:-1800}"            # log mtime older than this (s) = stuck
TRAIN_STEPS="${TRAIN_STEPS:-1000000}"
CDIR="${CDIR:-$(ls -dt logs/crl_flowbc_campaign_* 2>/dev/null | head -1)}"

[[ -z "$CDIR" || ! -d "$CDIR" ]] && { echo "no campaign dir found"; exit 1; }
echo "[monitor] watching $CDIR  (tick=${INTERVAL}s, stale>${STALE}s, target=${TRAIN_STEPS})"

declare -A PREV_STEP
tick=0
while true; do
    tick=$((tick + 1))
    now=$(date +%s)
    n_done=0 n_crash=0 n_alert=0 n_run=0 n_total=0
    echo ""
    echo "════════ [monitor] $(date '+%F %T')  tick $tick ════════"
    shopt -s nullglob
    for f in "$CDIR"/*.log; do
        name=$(basename "$f" .log)
        n_total=$((n_total + 1))
        # only the tail carries the latest bar state; whole-file tr is too slow
        body=$(tail -c 200000 "$f" 2>/dev/null | tr '\r' '\n')
        step=$(grep -oE "[0-9]+/${TRAIN_STEPS} \[" <<< "$body" | tail -1 | grep -oE "^[0-9]+")
        step=${step:-0}
        itps=$(grep -oE "[0-9]+/${TRAIN_STEPS} \[[0-9:]+<[0-9:]+, +[0-9.]+it/s\]" <<< "$body" | tail -1 | grep -oE "[0-9.]+it/s")
        evrate=$(grep -oE "/51 \[[0-9:]+<[0-9:]+, +[0-9.]+s/it\]" <<< "$body" | tail -1 | grep -oE "[0-9.]+s/it")
        mtime=$(stat -c %Y "$f" 2>/dev/null || echo 0)
        age=$((now - mtime))
        crashed=$(grep -c "Traceback (most recent call last)" "$f" 2>/dev/null)
        done=0
        { [[ "$step" -ge "$TRAIN_STEPS" ]] || grep -q "params_${TRAIN_STEPS}.pkl" "$f" 2>/dev/null; } && done=1

        pct=$(( step * 100 / TRAIN_STEPS ))
        flag=""; st=""
        if [[ "$done" == 1 ]]; then
            st="DONE   "; n_done=$((n_done + 1))
        elif [[ "$crashed" -gt 0 ]]; then
            st="CRASHED"; flag=" !!! Traceback"; n_crash=$((n_crash + 1)); n_alert=$((n_alert + 1))
        elif [[ "$age" -gt "$STALE" ]]; then
            st="STALLED"; flag=" !!! no log update for $((age / 60))m"; n_alert=$((n_alert + 1))
        else
            n_run=$((n_run + 1))
            prev="${PREV_STEP[$name]:-0}"
            if [[ -n "$evrate" && "$step" == "$prev" ]]; then st="EVAL   "; else st="TRAIN  "; fi
        fi
        ev=""; [[ -n "$evrate" ]] && ev="eval=${evrate}"
        printf "  %-7s %-42s %3d%% step=%-8s %-10s %s%s\n" \
            "$st" "$name" "$pct" "$step" "${itps:-}" "$ev" "$flag"
        PREV_STEP[$name]="$step"
    done
    echo "  ---- $n_run running, $n_done done, $n_crash crashed, $n_total total ----"
    [[ "$n_alert" -gt 0 ]] && echo "  !!!!! $n_alert run(s) need attention !!!!!"

    if [[ "$n_total" -gt 0 && $((n_done + n_crash)) -ge "$n_total" ]]; then
        echo "[monitor] all runs finished ($n_done done, $n_crash crashed) — exiting."
        exit 0
    fi
    sleep "$INTERVAL"
done

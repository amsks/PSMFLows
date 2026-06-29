#!/usr/bin/env bash
# scripts/train/resume_crl_flowbc_campaign.sh — resume/complete the CRL+FlowBC campaign
# that was killed ~3h in (SSH disconnect -> SIGHUP; the old launcher trapped only
# INT/TERM). Run this INSIDE tmux so it survives disconnects:
#
#   tmux new -s crl
#   bash scripts/train/resume_crl_flowbc_campaign.sh 2>&1 | tee /dev/shm/crl_resume.log
#   # detach with Ctrl-b d ; reattach with: tmux attach -t crl
#
#   DRY_RUN=1 bash scripts/train/resume_crl_flowbc_campaign.sh   # print the plan only
#
# What it does (matches a single 0..1M run's on-disk layout per seed):
#   STATE 0-5 : RESUME from /dev/shm params_600000 (+400k steps), writing
#               params_700000..1000000 into the SAME sd00X_* dir (exp_name pinned,
#               ckpt epoch + wandb/CSV step offset by 600000 via the shim).
#   STATE 6-9 : FRESH 0..1M (never started in the killed run).
#   PIXEL 0-4 : FRESH 0..1M (killed run reached <=12% with no usable checkpoints).
#
# Storage: EVERYTHING stays in /dev/shm (231G free). The durable disk (/) is 100%
# full (the 465G ~/git/offline-rl-aditya repo), so there is no rsync-to-durable
# here. /dev/shm is RAM-backed => a reboot loses it. TMPDIR/WANDB_DIR/logs are also
# pinned to /dev/shm so the full / is never written to.
#
# wandb_mode MUST be online: main.py builds save_dir from wandb.run.project, which
# the shim sets to 'factored-fb' only when wandb runs online (offline/disabled ->
# project 'dummy' -> wrong dir).
set -euo pipefail
trap '' HUP                     # belt-and-suspenders: ignore SSH-disconnect SIGHUP
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"

PYTHON="${PYTHON:-$REPO/.venv/bin/python}"
WANDB_MODE="${WANDB_MODE:-online}"
OUT="/dev/shm/gciql_outputs"
GROUP_STATE="factored-fb-crl-flowbc"
GROUP_PIXEL="factored-fb-crl-flowbc-pixel-drq"
RESTORE_EPOCH="${RESTORE_EPOCH:-600000}"
STATE_TOTAL="${STATE_TOTAL:-1000000}"
PIXEL_TOTAL="${PIXEL_TOTAL:-1000000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-100000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-100000}"

STATE_SEEDS="${STATE_SEEDS:-0 1 2 3 4 5 6 7 8 9}"   # resume vs fresh auto-detected
PIXEL_SEEDS="${PIXEL_SEEDS:-0 1 2 3 4}"
STATE_GPUS="${STATE_GPUS:-5 6 7}"
STATE_JOBS_PER_GPU="${STATE_JOBS_PER_GPU:-2}"
PIXEL_GPUS="${PIXEL_GPUS:-0 1 2 3 4}"
CLEAN_STALE_PIXEL="${CLEAN_STALE_PIXEL:-1}"
DRY_RUN="${DRY_RUN:-0}"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="/dev/shm/crl_resume_logs_$TS"; mkdir -p "$LOG_DIR"
export TMPDIR="/dev/shm/tmp_crl"; mkdir -p "$TMPDIR"
export WANDB_DIR="/dev/shm/wandb_crl"; mkdir -p "$WANDB_DIR"

echo "[resume] dry_run=$DRY_RUN wandb=$WANDB_MODE python=$PYTHON"
echo "[resume] STATE seeds=[$STATE_SEEDS] gpus=[$STATE_GPUS] (${STATE_JOBS_PER_GPU}/GPU)  total=$STATE_TOTAL restore_epoch=$RESTORE_EPOCH"
echo "[resume] PIXEL seeds=[$PIXEL_SEEDS] gpus=[$PIXEL_GPUS] (1/GPU)  total=$PIXEL_TOTAL"
echo "[resume] logs=$LOG_DIR  outputs=$OUT (VOLATILE /dev/shm)"
echo ""

# Existing resume dir basename for a state seed (only if its restore ckpt exists).
state_dir_for() {  # seed -> dirname | ""
    local d; d="$(ls -d "$OUT/factored-fb/$GROUP_STATE/$(printf 'sd%03d' "$1")_"* 2>/dev/null | head -1 || true)"
    if [[ -n "$d" && -f "$d/params_${RESTORE_EPOCH}.pkl" ]]; then basename "$d"; fi
}

# Per-seed Hydra overrides (space-separated, no spaces inside any value).
state_args() {  # seed -> overrides
    local seed="$1" dir; dir="$(state_dir_for "$seed")"
    if [[ -n "$dir" ]]; then
        echo "train_steps=$((STATE_TOTAL - RESTORE_EPOCH)) restore_path=$OUT/factored-fb/$GROUP_STATE/$dir restore_epoch=$RESTORE_EPOCH exp_name=$dir"
    else
        echo "train_steps=$STATE_TOTAL"
    fi
}
pixel_args() { echo "train_steps=$PIXEL_TOTAL"; }

launch() {  # cfg seed gpu group log  extra...
    local cfg="$1" seed="$2" gpu="$3" group="$4" log="$5"; shift 5
    local -a cmd=("$PYTHON" run_gciql.py --config-name "$cfg"
        seed="$seed" save_interval="$SAVE_INTERVAL" eval_interval="$EVAL_INTERVAL"
        wandb_mode="$WANDB_MODE" run_group="$group"
        "hydra.run.dir=$LOG_DIR/hydra_${group}_s${seed}" "$@")
    CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" > "$log" 2>&1
}

# Generic slot scheduler. $1=cfg $2=group $3=args-builder-fn $4="seeds" $5..=slot->gpu
run_track() {
    local cfg="$1" group="$2" builder="$3"; shift 3
    read -ra seeds <<< "$1"; shift
    local -a slot_gpu=("$@"); local n="${#slot_gpu[@]}"
    local -a pid=(); for ((s=0;s<n;s++)); do pid+=(""); done
    for seed in "${seeds[@]}"; do
        local slot=""
        while [[ -z "$slot" ]]; do
            for ((s=0;s<n;s++)); do
                if [[ -z "${pid[$s]}" ]] || ! kill -0 "${pid[$s]}" 2>/dev/null; then slot="$s"; break; fi
            done
            [[ -z "$slot" ]] && sleep 3
        done
        local gpu="${slot_gpu[$slot]}" log="$LOG_DIR/${group}_s${seed}.log"
        local -a extra; read -ra extra <<< "$("$builder" "$seed")"
        local kind="FRESH"; [[ "${extra[*]}" == *restore_path=* ]] && kind="RESUME"
        echo "[resume] $(date +%T) $group s=$seed -> GPU $gpu  [$kind] ${extra[*]}"
        if [[ "$DRY_RUN" == 1 ]]; then sleep 1; pid[$slot]=""; continue; fi
        launch "$cfg" "$seed" "$gpu" "$group" "$log" "${extra[@]}" &
        pid[$slot]="$!"
    done
    for p in "${pid[@]}"; do [[ -n "$p" ]] && wait "$p" 2>/dev/null || true; done
}

# Drop the killed run's partial pixel dirs so each pixel seed has one clean dir.
if [[ "$CLEAN_STALE_PIXEL" == 1 ]]; then
    for d in "$OUT/factored-fb/$GROUP_PIXEL"/sd*; do
        [[ -d "$d" ]] || continue
        echo "[resume] stale partial pixel dir -> remove: $d"
        [[ "$DRY_RUN" == 1 ]] || rm -rf "$d"
    done
fi

# Slot lists: state packs STATE_JOBS_PER_GPU per GPU; pixel is 1/GPU.
state_slots=(); for g in $STATE_GPUS; do for ((j=0;j<STATE_JOBS_PER_GPU;j++)); do state_slots+=("$g"); done; done
pixel_slots=(); for g in $PIXEL_GPUS; do pixel_slots+=("$g"); done

run_track cube_single_state_crl_flowbc      "$GROUP_STATE" state_args "$STATE_SEEDS" "${state_slots[@]}" &
STATE_TRACK="$!"
run_track cube_single_visual_crl_flowbc_drq "$GROUP_PIXEL" pixel_args "$PIXEL_SEEDS" "${pixel_slots[@]}" &
PIXEL_TRACK="$!"
wait "$STATE_TRACK" "$PIXEL_TRACK"

echo ""
echo "[resume] ALL DONE. logs=$LOG_DIR outputs=$OUT (VOLATILE — copy out before reboot)"

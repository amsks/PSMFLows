#!/usr/bin/env bash
# scripts/train/run_crl_flowbc_campaign.sh — CRL+FlowBC large-run campaign.
#
# Launches both tracks concurrently across 8 GPUs:
#   PIXEL: cube_single_visual_crl_flowbc_drq  seeds 0-4  GPUs 0-4 (1 job/GPU)
#   STATE: cube_single_state_crl_flowbc       seeds 0-9  GPUs 5-7 (2 jobs/GPU, queued)
# Both at TRAIN_STEPS (default 1M), wandb online, checkpoints every SAVE_INTERVAL
# to /dev/shm, rsynced to a durable dir every SYNC_INTERVAL. Pixel needs a
# dedicated GPU (~17 GB); state is light so it packs 2/GPU.
#
#   DRY_RUN=1 bash scripts/train/run_crl_flowbc_campaign.sh   # print plan, launch nothing
#   bash scripts/train/run_crl_flowbc_campaign.sh             # real launch
# Run inside tmux so it survives SSH disconnect:
#   tmux new -s crl; bash scripts/train/run_crl_flowbc_campaign.sh 2>&1 | tee logs/crl_campaign.log
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"

DRY_RUN="${DRY_RUN:-0}"
WANDB_MODE="${WANDB_MODE:-online}"
TRAIN_STEPS="${TRAIN_STEPS:-1000000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-100000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-100000}"
PIXEL_SEEDS="${PIXEL_SEEDS:-0 1 2 3 4}"
STATE_SEEDS="${STATE_SEEDS:-0 1 2 3 4 5 6 7 8 9}"
PIXEL_GPUS="${PIXEL_GPUS:-0 1 2 3 4}"
STATE_GPUS="${STATE_GPUS:-5 6 7}"
STATE_JOBS_PER_GPU="${STATE_JOBS_PER_GPU:-2}"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$REPO/logs/crl_flowbc_campaign_$TS"
SHM_OUT="/dev/shm/gciql_outputs"
DURABLE_OUT="${DURABLE_OUT:-$HOME/gciql_runs/crl_flowbc_$TS}"
SYNC_INTERVAL="${SYNC_INTERVAL:-600}"
mkdir -p "$LOG_DIR"

echo "[crl-campaign] dry_run=$DRY_RUN wandb=$WANDB_MODE train_steps=$TRAIN_STEPS save_intvl=$SAVE_INTERVAL"
echo "[crl-campaign] PIXEL seeds=[$PIXEL_SEEDS] gpus=[$PIXEL_GPUS] (1/GPU)"
echo "[crl-campaign] STATE seeds=[$STATE_SEEDS] gpus=[$STATE_GPUS] (${STATE_JOBS_PER_GPU}/GPU)"
echo "[crl-campaign] logs=$LOG_DIR  durable=$DURABLE_OUT"
echo ""

launch() {  # config seed gpu run_group logfile
    local cfg="$1" seed="$2" gpu="$3" group="$4" log="$5"
    local -a cmd=(python run_gciql.py --config-name "$cfg"
        seed="$seed" train_steps="$TRAIN_STEPS"
        save_interval="$SAVE_INTERVAL" eval_interval="$EVAL_INTERVAL"
        wandb_mode="$WANDB_MODE" run_group="$group"
        "hydra.run.dir=$LOG_DIR/hydra_${group}_s${seed}")
    if [[ "$DRY_RUN" == 1 ]]; then
        echo "[plan] GPU $gpu <- $cfg seed=$seed group=$group  (log: ${log##*/})"
        sleep 2  # hold the slot so the scheduler rotates GPUs realistically
        return 0
    fi
    CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" > "$log" 2>&1
}

# Schedule a track over a fixed slot list (slot -> gpu), queuing extra seeds.
run_track() {  # config run_group "seeds" "slot_gpus..."
    local cfg="$1" group="$2"; shift 2
    read -ra seeds <<< "$1"; shift
    local -a slot_gpu=("$@")
    local n_slots="${#slot_gpu[@]}"
    local -a slot_pid=(); for ((s=0;s<n_slots;s++)); do slot_pid+=(""); done
    for seed in "${seeds[@]}"; do
        local slot=""
        while [[ -z "$slot" ]]; do
            for ((s=0;s<n_slots;s++)); do
                local pid="${slot_pid[$s]}"
                if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then slot="$s"; break; fi
            done
            [[ -z "$slot" ]] && sleep 3
        done
        local gpu="${slot_gpu[$slot]}"
        local log="$LOG_DIR/${group}_s${seed}.log"
        echo "[crl-campaign] $(date +%T) $group seed=$seed -> GPU $gpu"
        launch "$cfg" "$seed" "$gpu" "$group" "$log" &
        slot_pid[$slot]="$!"
    done
    for pid in "${slot_pid[@]}"; do [[ -n "$pid" ]] && wait "$pid" 2>/dev/null || true; done
}

# Build state slot list: each state GPU repeated STATE_JOBS_PER_GPU times.
state_slots=()
for g in $STATE_GPUS; do for ((j=0;j<STATE_JOBS_PER_GPU;j++)); do state_slots+=("$g"); done; done
# Pixel slot list: one slot per GPU.
pixel_slots=(); for g in $PIXEL_GPUS; do pixel_slots+=("$g"); done

if [[ "$DRY_RUN" == 1 ]]; then
    run_track cube_single_visual_crl_flowbc_drq factored-fb-crl-flowbc-pixel-drq "$PIXEL_SEEDS" "${pixel_slots[@]}"
    run_track cube_single_state_crl_flowbc      factored-fb-crl-flowbc           "$STATE_SEEDS" "${state_slots[@]}"
    echo ""; echo "[crl-campaign] DRY RUN complete — nothing launched."
    exit 0
fi

mkdir -p "$DURABLE_OUT"
sync_once() { rsync -a "$SHM_OUT/" "$DURABLE_OUT/" 2>/dev/null || true; }
( while true; do sleep "$SYNC_INTERVAL"; sync_once; echo "[crl-campaign] $(date +%T) synced -> $DURABLE_OUT"; done ) &
SYNC_PID="$!"
cleanup() { echo "[crl-campaign] signal — stopping"; kill "$SYNC_PID" 2>/dev/null||true; pkill -P $$ 2>/dev/null||true; sync_once; exit 1; }
trap cleanup INT TERM

# Run both tracks concurrently (each in its own scheduler subshell).
run_track cube_single_visual_crl_flowbc_drq factored-fb-crl-flowbc-pixel-drq "$PIXEL_SEEDS" "${pixel_slots[@]}" &
PIXEL_TRACK="$!"
run_track cube_single_state_crl_flowbc      factored-fb-crl-flowbc           "$STATE_SEEDS" "${state_slots[@]}" &
STATE_TRACK="$!"
wait "$PIXEL_TRACK" "$STATE_TRACK"

kill "$SYNC_PID" 2>/dev/null || true
sync_once
echo "[crl-campaign] ALL DONE. logs=$LOG_DIR durable=$DURABLE_OUT"

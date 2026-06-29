#!/usr/bin/env bash
# scripts/train/run_sweep.sh — Antmaze-medium + Cube-single sweep across N GPUs.
#
# Mirrors td_jepa's sweep grid exactly:
#   ortho_coef ∈ {100, 1000}  x  lr_b ∈ {1e-4, 1e-5}  =  4 combos per domain
#
# td_jepa ran 10 seeds; we default to 3. Adjust SEEDS to taste.
#
# Usage:
#   bash scripts/train/run_sweep.sh                                      # 8 GPUs, full grid
#   N_GPUS=4 bash scripts/train/run_sweep.sh                             # 4 GPUs
#   JOBS_PER_GPU=2 bash scripts/train/run_sweep.sh                       # 2 jobs per GPU
#   SEEDS="1 2 3" bash scripts/train/run_sweep.sh                        # more seeds
#   DOMAINS="antmaze_medium" bash scripts/train/run_sweep.sh             # single domain
#   DATA_PATH=/dev/shm/datasets bash scripts/train/run_sweep.sh          # custom data path
#
# Run inside tmux so the session survives SSH disconnect:
#   tmux new -s sweep
#   bash scripts/train/run_sweep.sh 2>&1 | tee logs/sweep_master.log
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$REPO/logs/$TIMESTAMP"
N_GPUS="${N_GPUS:-8}"
JOBS_PER_GPU="${JOBS_PER_GPU:-10}"
N_PARALLEL=$(( N_GPUS * JOBS_PER_GPU ))
SEEDS="${SEEDS:-1 2 3 4 5 6 7 8 9 10}"
DOMAINS="${DOMAINS:-antmaze_medium cube_single}"
ORTHOS="${ORTHOS:-100 1000}"
LR_BS="${LR_BS:-1e-4 1e-5}"
DATA_PATH="${DATA_PATH:-/dev/shm/datasets}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "$LOG_DIR"

# ── Build job list ────────────────────────────────────────────────────────
declare -a JOB_DOMAINS JOB_ORTHOS JOB_LR_BS JOB_SEEDS JOB_GROUPS

for domain in $DOMAINS; do
    case "$domain" in
        antmaze_medium) group_base="antmaze-medium" ;;
        cube_single)    group_base="cube-single"    ;;
        *)              group_base="$domain"        ;;
    esac
    for ortho in $ORTHOS; do
        for lr_b in $LR_BS; do
            for seed in $SEEDS; do
                JOB_DOMAINS+=("$domain")
                JOB_ORTHOS+=("$ortho")
                JOB_LR_BS+=("$lr_b")
                JOB_SEEDS+=("$seed")
                JOB_GROUPS+=("${group_base}-ortho${ortho}-lrb${lr_b}")
            done
        done
    done
done

N_JOBS="${#JOB_DOMAINS[@]}"
echo "[sweep] $N_JOBS jobs  |  $N_GPUS GPUs  |  $JOBS_PER_GPU job(s)/GPU ($N_PARALLEL parallel)  |  logs → $LOG_DIR"
echo "[sweep] domains:    $DOMAINS"
echo "[sweep] ortho_coef: $ORTHOS"
echo "[sweep] lr_b:       $LR_BS"
echo "[sweep] seeds:      $SEEDS"
echo "[sweep] data_path:  $DATA_PATH"
echo "[sweep] extra_args: ${EXTRA_ARGS:-(none)}"
echo ""

# ── Graceful shutdown on Ctrl-C ───────────────────────────────────────────
declare -a ACTIVE_PIDS=()
cleanup() {
    echo ""
    echo "[sweep] Caught signal — terminating running jobs..."
    for pid in "${ACTIVE_PIDS[@]+"${ACTIVE_PIDS[@]}"}"; do
        kill "$pid" 2>/dev/null || true
    done
    exit 1
}
trap cleanup INT TERM

# ── GPU slot tracker ──────────────────────────────────────────────────────
# Maps slot index → pid. Slot i lives on GPU (i % N_GPUS).
declare -a SLOT_PIDS=()
for (( s=0; s<N_PARALLEL; s++ )); do SLOT_PIDS+=(""); done

find_free_slot() {
    while true; do
        for (( s=0; s<N_PARALLEL; s++ )); do
            pid="${SLOT_PIDS[$s]}"
            if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
                if [[ -n "$pid" ]]; then
                    wait "$pid" && status=0 || status=$?
                    echo "[sweep] $(date +%T)  pid=$pid slot=$s finished (exit=$status)"
                fi
                echo "$s"
                return
            fi
        done
        sleep 3
    done
}

# ── Launch jobs ───────────────────────────────────────────────────────────
for i in "${!JOB_DOMAINS[@]}"; do
    domain="${JOB_DOMAINS[$i]}"
    ortho="${JOB_ORTHOS[$i]}"
    lr_b="${JOB_LR_BS[$i]}"
    seed="${JOB_SEEDS[$i]}"
    group="${JOB_GROUPS[$i]}"

    lr_tag="${lr_b//-/}"
    log="$LOG_DIR/${domain}_ortho${ortho}_lrb${lr_tag}_s${seed}.log"
    run_name="${domain}__s${seed}__ortho${ortho}__lrb${lr_b}"

    slot=$(find_free_slot)
    gpu=$(( slot % N_GPUS ))

    echo "[sweep] $(date +%T)  launching job $((i+1))/$N_JOBS on GPU $gpu (slot $slot): $run_name"
    (
        cd "$REPO"
        CUDA_VISIBLE_DEVICES="$gpu" python train.py \
            domain="$domain" \
            ortho_coef="$ortho" \
            lr_b="$lr_b" \
            seed="$seed" \
            data_path="$DATA_PATH" \
            use_wandb=true \
            wandb_entity=amsks \
            wandb_group="$group" \
            "wandb_tags=[sweep-${TIMESTAMP}]" \
            wandb_run_name="$run_name" \
            $EXTRA_ARGS \
        > "$log" 2>&1
        echo "[sweep] $(date +%T)  DONE: $run_name"
    ) &
    pid="$!"
    SLOT_PIDS[$slot]="$pid"
    ACTIVE_PIDS+=("$pid")
done

# ── Wait for stragglers ───────────────────────────────────────────────────
echo "[sweep] $(date +%T)  all jobs launched — waiting for stragglers..."
for pid in "${ACTIVE_PIDS[@]+"${ACTIVE_PIDS[@]}"}"; do
    wait "$pid" && status=0 || status=$?
    echo "[sweep] $(date +%T)  pid=$pid finished (exit=$status)"
done

echo ""
echo "[sweep] ══════════════════════════════════════════════════"
echo "[sweep] Sweep complete: $N_JOBS runs in $LOG_DIR"
echo "[sweep] Quick tail of each log (last line):"
for i in "${!JOB_DOMAINS[@]}"; do
    lr_tag="${JOB_LR_BS[$i]//-/}"
    log="$LOG_DIR/${JOB_DOMAINS[$i]}_ortho${JOB_ORTHOS[$i]}_lrb${lr_tag}_s${JOB_SEEDS[$i]}.log"
    last="$(tail -1 "$log" 2>/dev/null || echo '(no output)')"
    tag="${JOB_DOMAINS[$i]}_s${JOB_SEEDS[$i]}_ortho${JOB_ORTHOS[$i]}_lrb${JOB_LR_BS[$i]}"
    printf "  %-52s  %s\n" "$tag" "$last"
done
echo "[sweep] ══════════════════════════════════════════════════"

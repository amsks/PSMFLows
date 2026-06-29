#!/usr/bin/env bash
# scripts/agents/_common.sh — shared helpers + sweep entry points for scripts/agents/*.
# Sourced (never executed) by each leaf. Provides:
#   _gpu_for_seed <seed> <gpus_str>          -> echoes GPUS[seed % nGPU]
#   _log_dir <run_group> <seed>              -> echoes /dev/shm/<run_group>_<TS>/seed_<seed>.log
#   _dry_run_preview <cmd...>                -> if DRY_RUN=1, echoes "[DRY] <cmd...>" and returns 0;
#                                               otherwise returns 1 (caller proceeds)
#   _assert_data <path>                      -> exits 1 with a clear message if path missing
#   _setup_dev_shm                           -> mkdir -p the volatile run dirs under /dev/shm
#   _print_usage                             -> prints $USAGE to stderr
#   _maybe_help <args...>                    -> if any arg is --help/-h, print usage and exit 0
#   _run_fb_sweep <domain> <run_group> [extra hydra args...]
#                                            -> PyTorch FB sweep via train.py
#   _run_jax_sweep <config-name> <run_group> [extra args...]
#                                            -> JAX OGBench sweep via run_gciql.py
# Per-leaf env knobs (read by _run_*_sweep):
#   SEEDS (default "0 1 2 3 4"), GPUS (default "0 1"), TRAIN_STEPS, RUN_GROUP, DATA_PATH,
#   STAGGER (default 3 for fb / 2 for jax), DRY_RUN (default 0),
#   WANDB_MODE (jax, default online), STORAGE (jax, default shm), WANDB_ENTITY (fb, default amsks).
set -uo pipefail
trap '' HUP

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-$REPO/.venv/bin/python}"

_gpu_for_seed() {
    local seed="$1"
    local -a gpus
    read -ra gpus <<< "$2"
    local n=${#gpus[@]}
    echo "${gpus[$(( seed % n ))]}"
}

_TS_SHARED="${_TS_SHARED:-$(date +%Y%m%d_%H%M%S)}"
_log_dir() {
    local group="$1" seed="$2"
    local dir="/dev/shm/${group}_${_TS_SHARED}"
    mkdir -p "$dir"
    echo "${dir}/seed_${seed}.log"
}

_dry_run_preview() {
    if [[ "${DRY_RUN:-0}" == 1 ]]; then
        echo "[DRY] $*"
        return 0
    fi
    return 1
}

_assert_data() {
    [[ -d "$1" ]] || { echo "MISSING dataset $1 — run scripts/data/extract_ogbench.py" >&2; exit 1; }
}

_setup_dev_shm() {
    mkdir -p /dev/shm/tmp_fb /dev/shm/wandb_fb \
             /dev/shm/tmp_crl /dev/shm/wandb_crl \
             /dev/shm/factored-fb/runs /dev/shm/gciql_outputs
}

_print_usage() { printf '%s\n' "${USAGE:-(no usage defined)}" >&2; }

_maybe_help() {
    for a in "$@"; do
        [[ "$a" == "--help" || "$a" == "-h" ]] && { _print_usage; exit 0; }
    done
}

# ── PyTorch FB sweep (train.py) ────────────────────────────────────────────
_run_fb_sweep() {
    local domain="$1" run_group="$2"; shift 2
    local -a extra_args=("$@")

    local seeds="${SEEDS:-0 1 2 3 4}"
    local gpus="${GPUS:-0 1}"
    local steps="${TRAIN_STEPS:-1000000}"
    local data="${DATA_PATH:-/dev/shm/factored-fb/datasets}"
    local stagger="${STAGGER:-3}"
    local wandb_entity="${WANDB_ENTITY:-amsks}"
    local save_eval_videos="${SAVE_EVAL_VIDEOS:-false}"

    _setup_dev_shm
    export TMPDIR="${TMPDIR:-/dev/shm/tmp_fb}"
    export WANDB_DIR="${WANDB_DIR:-/dev/shm/wandb_fb}"
    _assert_data "$data/cube-single-play-v0/buffer"

    echo "[fb-sweep] domain=$domain group=$run_group steps=$steps seeds=[$seeds] gpus=[$gpus]"
    echo "[fb-sweep] data=$data  save_eval_videos=$save_eval_videos  extra_args=(${extra_args[*]:-})"

    local -a seed_arr; read -ra seed_arr <<< "$seeds"
    local s gpu log run_dir
    for s in "${seed_arr[@]}"; do
        gpu="$(_gpu_for_seed "$s" "$gpus")"
        log="$(_log_dir "$run_group" "$s")"
        run_dir="/dev/shm/factored-fb/runs/${_TS_SHARED}__${run_group}__s${s}"
        local cmd=("$PYTHON" train.py
            "domain=$domain"
            "num_train_steps=$steps"
            "data_path=$data"
            "seed=$s" device=cuda
            "save_eval_videos=$save_eval_videos"
            use_wandb=true "wandb_entity=$wandb_entity" "wandb_group=$run_group"
            "wandb_run_name=${run_group}_s${s}"
            hydra.job.chdir=true "hydra.run.dir=$run_dir"
            "${extra_args[@]}")
        echo "[fb-sweep] seed=$s -> GPU $gpu  log=$log"
        if _dry_run_preview "CUDA_VISIBLE_DEVICES=$gpu" "${cmd[@]}"; then
            continue
        fi
        CUDA_VISIBLE_DEVICES="$gpu" nohup "${cmd[@]}" > "$log" 2>&1 &
        sleep "$stagger"
    done
    if [[ "${DRY_RUN:-0}" != 1 ]]; then
        wait
        echo "[fb-sweep] ALL DONE"
    fi
}

# ── JAX OGBench sweep (run_gciql.py) ───────────────────────────────────────
_run_jax_sweep() {
    local config="$1" run_group="$2"; shift 2
    local -a extra_args=("$@")

    local seeds="${SEEDS:-0 1 2 3 4}"
    local gpus="${GPUS:-0 1}"
    local steps="${TRAIN_STEPS:-1000000}"
    local stagger="${STAGGER:-2}"
    local wandb_mode="${WANDB_MODE:-online}"
    local storage="${STORAGE:-shm}"

    _setup_dev_shm
    export TMPDIR="${TMPDIR:-/dev/shm/tmp_crl}"
    export WANDB_DIR="${WANDB_DIR:-/dev/shm/wandb_crl}"

    echo "[jax-sweep] config=$config group=$run_group steps=$steps seeds=[$seeds] gpus=[$gpus]"
    echo "[jax-sweep] wandb_mode=$wandb_mode storage=$storage  extra_args=(${extra_args[*]:-})"

    local -a seed_arr; read -ra seed_arr <<< "$seeds"
    local s gpu log
    for s in "${seed_arr[@]}"; do
        gpu="$(_gpu_for_seed "$s" "$gpus")"
        log="$(_log_dir "$run_group" "$s")"
        local cmd=("$PYTHON" run_gciql.py
            --config-name "$config"
            "seed=$s" "train_steps=$steps"
            "storage=$storage" "wandb_mode=$wandb_mode" "run_group=$run_group"
            "hydra.run.dir=/dev/shm/${run_group}_${_TS_SHARED}/hydra_s${s}"
            "${extra_args[@]}")
        echo "[jax-sweep] seed=$s -> GPU $gpu  log=$log"
        if _dry_run_preview "CUDA_VISIBLE_DEVICES=$gpu" "${cmd[@]}"; then
            continue
        fi
        CUDA_VISIBLE_DEVICES="$gpu" nohup "${cmd[@]}" > "$log" 2>&1 &
        sleep "$stagger"
    done
    if [[ "${DRY_RUN:-0}" != 1 ]]; then
        wait
        echo "[jax-sweep] ALL DONE"
    fi
}

#!/usr/bin/env bash
# scripts/train/run_sweep_slurm.sh — Submit the antmaze + cube HP sweep to SLURM via
# Hydra's submitit launcher.
#
# Each (domain, ortho_coef, lr_b, seed) combination becomes one SLURM job.
# Total: 2 domains × 2 orthos × 2 lr_bs × 10 seeds = 80 jobs.
#
# Prerequisites on the cluster:
#   pip install -e ".[slurm]"       # installs hydra-submitit-launcher
#
# Usage:
#   bash scripts/train/run_sweep_slurm.sh --partition gpu
#   bash scripts/train/run_sweep_slurm.sh --partition gpu --account myproject
#   bash scripts/train/run_sweep_slurm.sh --partition gpu --seeds "42 1 2"  # custom seeds
#
# Outputs land in multirun/YYYY-MM-DD_HH-MM-SS/ (Hydra default sweep dir).
# SLURM logs: multirun/.../.submitit/<job_id>/

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ── Defaults (mirror td_jepa's sweep exactly) ─────────────────────────────
PARTITION=""
ACCOUNT=""
QOS=""
SEEDS="3917,3502,8948,9460,4729,2226,1744,7742,4501,6341"
ORTHOS="100,1000"
LR_BS="1e-4,1e-5"
DOMAINS="antmaze_medium,cube_single"
WANDB_ENTITY="amsks"
WANDB_GROUP="slurm-sweep"
DRY_RUN=0

usage() {
    grep '^#' "$0" | sed 's/^# \?//'
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --partition) PARTITION="$2"; shift 2 ;;
        --account)   ACCOUNT="$2";   shift 2 ;;
        --qos)       QOS="$2";       shift 2 ;;
        --seeds)     SEEDS="$2";     shift 2 ;;
        --orthos)    ORTHOS="$2";    shift 2 ;;
        --lr-bs)     LR_BS="$2";     shift 2 ;;
        --domains)   DOMAINS="$2";   shift 2 ;;
        --group)     WANDB_GROUP="$2"; shift 2 ;;
        --dry-run)   DRY_RUN=1;      shift   ;;
        -h|--help)   usage ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$PARTITION" ]]; then
    echo "[slurm] ERROR: --partition is required (e.g. --partition gpu)"
    exit 1
fi

# ── Build launcher overrides ──────────────────────────────────────────────
LAUNCHER_OVERRIDES=(
    "hydra/launcher=slurm"
    "hydra.launcher.partition=${PARTITION}"
)
[[ -n "$ACCOUNT" ]] && LAUNCHER_OVERRIDES+=("hydra.launcher.account=${ACCOUNT}")
[[ -n "$QOS"     ]] && LAUNCHER_OVERRIDES+=("hydra.launcher.qos=${QOS}")

# ── Sweep overrides (Hydra multirun comma-separated) ─────────────────────
SWEEP_OVERRIDES=(
    "domain=${DOMAINS}"
    "ortho_coef=${ORTHOS}"
    "lr_b=${LR_BS}"
    "seed=${SEEDS}"
    "use_wandb=true"
    "wandb_entity=${WANDB_ENTITY}"
    "wandb_group=${WANDB_GROUP}"
)

CMD=(
    python train.py
    --multirun
    "${LAUNCHER_OVERRIDES[@]}"
    "${SWEEP_OVERRIDES[@]}"
)

echo "[slurm] Submitting sweep: domains=${DOMAINS}  orthos=${ORTHOS}  lr_bs=${LR_BS}  seeds=${SEEDS}"
echo "[slurm] SLURM partition=${PARTITION} account=${ACCOUNT:-none} qos=${QOS:-none}"
echo "[slurm] Command: ${CMD[*]}"

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[slurm] DRY RUN — nothing submitted."
    exit 0
fi

cd "$REPO"
"${CMD[@]}"

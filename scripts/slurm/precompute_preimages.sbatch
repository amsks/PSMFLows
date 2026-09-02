#!/usr/bin/env bash
# Stage B on a SLURM cluster: pull the frozen behaviour flow from Hugging Face, invert the
# whole dataset, push the resulting npz back. Nothing else on the cluster needs to know
# about this repo's directory layout -- the HF dataset repo is the only shared state.
#
# Inverting 1M rows takes 12-25 h depending on `num_samples` and `n_steps`, so the wall
# clock below is deliberately generous; raise it rather than lose a 20 h job.
#
# Submit:
#   sbatch --export=ALL,ENVKEY=cube,NAME=cube-single-play-a20p6-ps0p69-ns12-N200,\
# ALPHA=20.57,PRIOR_SCALE=0.691,N_STEPS=12,NUM_SAMPLES=200 \
#     scripts/slurm/precompute_preimages.sbatch
#
# Required in the environment (or edit below):
#   HF_TOKEN     write token for the dataset repo (it is private)
#   PSM_REPO     path to this checkout on the cluster
#   PSM_DATA     scratch dir with room for ~2 GB per environment
#
#SBATCH --job-name=psm-preimage
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --time=36:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
# Cluster-specific; set on the command line rather than editing, e.g.
#   sbatch --partition=gpu --account=<acct> ...
set -euo pipefail

: "${PSM_REPO:?PSM_REPO=<path to the PSMFLows checkout> required}"
: "${PSM_DATA:?PSM_DATA=<scratch dir for npz + flow> required}"
: "${ENVKEY:?ENVKEY=cube|antmaze|pointmaze required}"
: "${NAME:?NAME=<artifact name in the HF repo> required}"
ALPHA="${ALPHA:?ALPHA required}"
PRIOR_SCALE="${PRIOR_SCALE:?PRIOR_SCALE required}"
N_STEPS="${N_STEPS:?N_STEPS (EM iterations) required}"
NUM_SAMPLES="${NUM_SAMPLES:-200}"
FLOW_STEPS="${FLOW_STEPS:-100}"        # forward AND inverse discretization; must be >= 100
HF_REPO="${HF_REPO:-amsks/psmflows-preimages}"

case "$ENVKEY" in
  cube)      ENV_NAME=cube-single-play-singletask-v0;               FLOW=cube-single-play ;;
  antmaze)   ENV_NAME=antmaze-medium-navigate-singletask-v0;        FLOW=antmaze-medium-navigate ;;
  pointmaze) ENV_NAME=pointmaze-medium-navigate-singletask-task1-v0; FLOW=pointmaze-medium-navigate ;;
  *) echo "unknown ENVKEY: $ENVKEY" >&2; exit 1 ;;
esac

cd "$PSM_REPO"
PY="${PY:-$PSM_REPO/.venv/bin/python}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${MEM_FRAC:-0.90}"
export MUJOCO_GL=egl                    # no DISPLAY on a compute node
export OGBENCH_DATASET_DIR="${OGBENCH_DATASET_DIR:-$PSM_DATA/ogbench}"
mkdir -p "$PSM_DATA" "$OGBENCH_DATASET_DIR"

EPOCH="${EPOCH:-500000}"
echo "=== 1/3 pull the frozen Stage-A flow ($FLOW @ $EPOCH) ==="
# Decoder only. A Stage-B job produces latents; it has no use for anyone else's, and the
# canonical npz is 0.3-0.5 GB of download that would be discarded.
"$PY" scripts/hf_preimages.py --repo "$HF_REPO" pull \
    --name "$FLOW" --flow-name "$FLOW" --dest "$PSM_DATA" --flow-only --flow-epoch "$EPOCH"
FLOW_DIR="$PSM_DATA/flow/$FLOW"

OUT="$PSM_DATA/preimages_${NAME}.npz"
echo "=== 2/3 invert the dataset -> $OUT ==="
echo "    env=$ENV_NAME flow=$FLOW_DIR@$EPOCH flow_steps=$FLOW_STEPS"
echo "    alpha=$ALPHA prior_scale=$PRIOR_SCALE n_steps=$N_STEPS num_samples=$NUM_SAMPLES"
srun "$PY" tools/precompute_preimages.py agent=fql \
    env_name="$ENV_NAME" \
    restore_path="$FLOW_DIR" restore_epoch="$EPOCH" \
    agent.flow_steps="$FLOW_STEPS" \
    inversion.n_initial_steps="$FLOW_STEPS" \
    inversion.alpha="$ALPHA" inversion.prior_scale="$PRIOR_SCALE" \
    inversion.n_steps="$N_STEPS" inversion.num_samples="$NUM_SAMPLES" \
    preimage_out="$OUT"

echo "=== 3/3 publish ==="
"$PY" scripts/hf_preimages.py --repo "$HF_REPO" push \
    --npz "$OUT" --name "$NAME" \
    --flow-dir "$FLOW_DIR" --flow-epoch "$EPOCH" --flow-name "$FLOW"

echo "done. On the training machine:"
echo "  python scripts/hf_preimages.py pull --name $NAME --dest \$PSM_DATA --with-flow"

#!/usr/bin/env bash
# Plain-FB baseline for the cube-single ladder, with the actor's BC term OFF.
#
# WHAT THIS RUNS
#   Repo    https://github.com/LUH-AI/Factored-FB   branch density-fb   commit b62dc9d
#           (checked out at $FB_REPO; `density-fb` is simply where the newest code sits --
#           the AGENT here is the plain Forward-Backward critic `fb`, not density_fb.)
#   Method  critic `fb`  = Touati-style Forward-Backward successor measure M = F(s,a,z) B(g)^T,
#                          impls/critics/fb.py, config impls/configs/critic/fb.yaml
#           actor `ddpgbc` = Gaussian DDPG+BC policy, impls/actors/ddpgbc.py
#   BC OFF  --override actor.alpha=0.0
#           DDPGBCActor.loss is  q_loss + bc_loss  with
#               q_loss  = -q.mean() / sg(|q|.mean() + 1e-6)     (q_normalize, default true)
#               bc_loss = -(alpha * dist.log_prob(dataset_action)).mean()
#           alpha=0 zeroes bc_loss EXACTLY and leaves the |q| normaliser in place, so the
#           actor ascends Q = <F(s,a,z), z> and nothing else. This is the repo's own no-BC
#           idiom -- scripts/launch_exorl.sh pins ACTOR=ddpgbc, ACTOR_OV=actor.alpha=0.0 for
#           exactly this reason ("flowbc carries bc_coeff=3.0 ... i.e. it clones an
#           explorer. Neither reference implementation does that").
#           `ddpgbc` is also what makes this a NO-FLOW arm: the repo's canonical cube-single
#           pairing is `--actor flowbc`, whose bc_coeff=3.0 distils a flow-matching BC policy.
#
# ARMS
#   fb_nobc_cube      fb x ddpgbc, actor.alpha=0.0                      seeds 0,1,2   <- the baseline
#   fb_bc_cube        fb x flowbc (bc_coeff=3.0), the repo's default    seed 0        <- BC-ON reference
#   fb_nobc_ortho1000 fb x ddpgbc alpha=0, ortho_coef=1000              seed 0        <- probe
#           The probe exists because PSMFlows' own archived FB (archive/agents/fb.py, cube
#           0.721) ran ortho_coef=1000 while this repo's fb.yaml ships the reference-native
#           1.0; without it a low baseline number cannot be attributed.
#
# EVAL
#   OGBench task_id=1 only, 500 episodes, which is what PSMFlows means by
#   `cube-single-play-singletask-v0`. The sibling repo's scripts/reeval_checkpoint.py loops
#   all five task_infos, so scripts/baselines/fb_eval500.py calls the same
#   impls.utils.evaluation.evaluate at one task and writes PSMFlows' report schema.
#
# Usage:  bash scripts/baselines/fb_nobc.sh {table|train|eval}
set -euo pipefail

FB_REPO=${FB_REPO:-/mnt/home/amohan/git/Austin/Factored-FB}
PSM_REPO=${PSM_REPO:-/mnt/home/amohan/git/Austin/PSMFLows}
PSM_DATA=${PSM_DATA:-/mnt/home/amohan/psm-data}
EXP=$PSM_DATA/exp/FactoredFB
LOGS=$PSM_DATA/logs
GROUP=${GROUP:-fb_nobc_cube}
SEEDS=${SEEDS:-"0 1 2"}
TRAIN_STEPS=${TRAIN_STEPS:-500000}
SAVE_INTERVAL=${SAVE_INTERVAL:-50000}
EPOCHS=${EPOCHS:-"50000 100000 150000 200000 250000 300000 350000 400000 450000 500000"}
NOBC_OV="actor.alpha=0.0"

case "${1:-table}" in

table)
  # The full merged hyperparameter table, printed BEFORE any launch (standing discipline).
  cd "$FB_REPO"
  JAX_PLATFORMS=cpu PYTHONPATH=$FB_REPO "$FB_REPO/.venv/bin/python" - <<'PY'
import json
from env import make_config
arms = [("fb_nobc_cube   (fb x ddpgbc, BC OFF)", "ddpgbc", ["actor.alpha=0.0"]),
        ("fb_bc_cube     (fb x flowbc, BC ON)",  "flowbc", []),
        ("fb_nobc_ortho1000",                    "ddpgbc", ["actor.alpha=0.0", "ortho_coef=1000"])]
for name, actor, ov in arms:
    print("=" * 78); print(name)
    print(json.dumps(dict(make_config("fb", actor, "cube_single", ov)), indent=1, default=str))
PY
  ;;

train)
  cd "$FB_REPO"
  for S in $SEEDS; do
    CRITIC=fb ACTOR=ddpgbc DOMAIN=cube_single SEED=$S TRAIN_STEPS=$TRAIN_STEPS \
    SAVE_INTERVAL=$SAVE_INTERVAL EVAL_INTERVAL=100000 EVAL_EPISODES=10 LOG_INTERVAL=10000 \
    OVERRIDES="$NOBC_OV" SAVE_DIR=$EXP/$GROUP \
      sbatch --job-name=fbnobc_sd$S --export=ALL scripts/kisski_train_fb.sbatch
  done
  # BC-ON reference: the repo's own canonical cube-single pairing.
  CRITIC=fb ACTOR=flowbc DOMAIN=cube_single SEED=0 TRAIN_STEPS=$TRAIN_STEPS \
  SAVE_INTERVAL=$SAVE_INTERVAL EVAL_INTERVAL=100000 EVAL_EPISODES=10 LOG_INTERVAL=10000 \
  OVERRIDES="" SAVE_DIR=$EXP/fb_bc_cube \
    sbatch --job-name=fbbc_sd0 --export=ALL scripts/kisski_train_fb.sbatch
  # ortho probe.
  CRITIC=fb ACTOR=ddpgbc DOMAIN=cube_single SEED=0 TRAIN_STEPS=$TRAIN_STEPS \
  SAVE_INTERVAL=$SAVE_INTERVAL EVAL_INTERVAL=100000 EVAL_EPISODES=10 LOG_INTERVAL=10000 \
  OVERRIDES="$NOBC_OV ortho_coef=1000" SAVE_DIR=$EXP/fb_nobc_ortho1000 \
    sbatch --job-name=fbnobc_o1k --export=ALL scripts/kisski_train_fb.sbatch
  ;;

eval)
  # 500 episodes of cube-single costs ~45 min on a shared H100 (~5 s/episode, unbatched
  # policy + MuJoCo), so ten checkpoints in one job would be a 7.5 h serial chain. The
  # ladder is CHUNKED into three jobs per seed instead; each writes its own JSONs and they
  # can land in any order.
  CHUNKS=("50000 100000 150000" "200000 250000 300000" "350000 400000 450000 500000")
  for S in $SEEDS; do
    RUN=$EXP/$GROUP/cube-single-play-v0_fb_ddpgbc_seed_$S
    for i in 0 1 2; do
      SEED=$S RUN_DIR=$RUN EPOCHS="${CHUNKS[$i]}" OUT_PREFIX=$LOGS/eval500_${GROUP} \
      CRITIC=fb ACTOR=ddpgbc \
        sbatch --job-name=fbev_sd${S}_c$i --export=ALL "$PSM_REPO/scripts/baselines/fb_eval500.sbatch"
    done
  done
  ;;

eval-ref)
  # The two single-seed reference arms, at the late checkpoints only.
  SEED=0 RUN_DIR=$EXP/fb_bc_cube/cube-single-play-v0_fb_flowbc_seed_0 \
  EPOCHS="300000 400000 500000" OUT_PREFIX=$LOGS/eval500_fb_bc_cube CRITIC=fb ACTOR=flowbc \
    sbatch --job-name=fbev_bc --export=ALL "$PSM_REPO/scripts/baselines/fb_eval500.sbatch"
  SEED=0 RUN_DIR=$EXP/fb_nobc_ortho1000/cube-single-play-v0_fb_ddpgbc_seed_0 \
  EPOCHS="300000 400000 500000" OUT_PREFIX=$LOGS/eval500_fb_nobc_ortho1000 CRITIC=fb ACTOR=ddpgbc \
    sbatch --job-name=fbev_o1k --export=ALL "$PSM_REPO/scripts/baselines/fb_eval500.sbatch"
  ;;

*) echo "usage: $0 {table|train|eval}" >&2; exit 1 ;;
esac

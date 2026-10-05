#!/usr/bin/env bash
# 200-step smoke of the joint-actor launch path (2026-10-05), one GPU, sequential: for each
# of fbc3_sh, dsrl_sh, fbc3_pt train 200 steps through main.py with the exact launch
# overrides (in-loop eval at steps 1 and 200 = the actor_rel readout, 4 episodes), then run
# tools/eval_checkpoint.py on params_200.pkl for the hgoal_each and actor_rel readouts
# (4 episodes, 1 worker). Prints flags.json's relevant keys and the actor stats.
# Submit: sbatch --partition=kisski-inference --account=general --time=01:00:00 --gres=gpu:1 \
#   --cpus-per-task=4 --mem=8G --job-name=ja_smoke --output=$PSM_DATA/logs/slurm/%x-%j.out \
#   --export=ALL $PSM_DATA/gc_scripts/ja_smoke.sh
set -uo pipefail
PSM_DATA=/mnt/home/amohan/psm-data; REPO=/mnt/home/amohan/git/Austin/PSMFLows
source $PSM_DATA/gc_scripts/ja_arms.sh
export PSM_REPO=$REPO PSM_DATA OGBENCH_DATASET_DIR=/mnt/home/amohan/.ogbench/data
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 MUJOCO_GL=egl WANDB_MODE=offline WANDB_DIR=$PSM_DATA/wandb
PY=$REPO/.venv/bin/python
FLOW=$PSM_DATA/flow/cube-single-play; PRE=$PSM_DATA/preimages/cube-single-play.npz
cd $REPO
echo "node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"; nvidia-smi -L 2>/dev/null | head -2
status=0
for arm in psmgoal_ja_fbc3_sh_cube psmgoal_ja_dsrl_sh_cube psmgoal_ja_fbc3_pt_cube; do
  G=smoke_${arm}_$(date +%H%M%S)
  echo "===== $arm -> $G"; echo "overrides: ${JA_ARMS[$arm]}"
  $PY main.py agent=psmgoal agent.flow_ckpt_path=$FLOW agent.flow_ckpt_epoch=500000 agent.preimage_path=$PRE \
     ${JA_ARMS[$arm]} env_name=cube-single-play-singletask-v0 offline_steps=200 online_steps=0 \
     log_interval=100 eval_interval=200 eval_episodes=4 save_interval=200 save_dir=$PSM_DATA/exp run_group=$G seed=0 \
     > $PSM_DATA/logs/slurm/${G}.train.log 2>&1 || { echo "TRAIN FAILED rc=$?"; tail -30 $PSM_DATA/logs/slurm/${G}.train.log; status=1; continue; }
  run=$(ls -d $PSM_DATA/exp/PSMFLows/$G/sd000* | head -1)
  echo "--- flags.json (agent keys that differ from psmgoal_db_sm_cube):"
  $PY - "$run/flags.json" <<'EOF'
import json, sys, glob
a = json.load(open(sys.argv[1]))["agent"]
ref = json.load(open(glob.glob("/mnt/home/amohan/psm-data/exp/PSMFLows/psmgoal_db_sm_cube/sd000*/flags.json")[0]))["agent"]
for k in sorted(set(a) | set(ref)):
    if a.get(k) != ref.get(k): print(f"  {k}: {ref.get(k)!r} -> {a.get(k)!r}")
EOF
  echo "--- train.csv (last row, actor stats):"
  $PY - "$run/train.csv" <<'EOF'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
r = rows[-1]
keys = [k for k in r if any(t in k for t in ("actor_", "psm_loss", "m_softmax_diag", "epoch_time", "alpha"))]
print("  step", r["step"]); [print(f"  {k}: {r[k]}") for k in sorted(keys)]
import math; bad = [k for k in r if r[k] not in ("",) and not math.isfinite(float(r[k]))]
print("  NONFINITE:", bad)
EOF
  echo "--- eval.csv (in-loop actor_rel, 4 episodes):"; tail -2 $run/eval.csv | cut -c1-300
  for tag in hgoal_each actor_rel; do
    case $tag in hgoal_each) OV="agent.coef_source=hgoal_each agent.acting=gpi";; actor_rel) OV="agent.acting=distill agent.eval_redistill=false agent.eval_goal_source=relabel";; esac
    out=$PSM_DATA/logs/slurm/${G}_200_${tag}_smoke.json
    EVAL_WORKERS=1 $PY tools/eval_checkpoint.py agent=psmgoal env_name=cube-single-play-singletask-task2-v0 \
      agent.flow_ckpt_path=$FLOW agent.flow_ckpt_epoch=500000 agent.preimage_path=$PRE agent.use_point_preimage=true \
      restore_path=$run restore_epoch=200 eval_episodes=4 report_out=$out $OV > ${out%.json}.log 2>&1 \
      && echo "--- eval $tag: $($PY -c "import json;d=json.load(open('$out'));print({k:d.get(k) for k in ('success','actor_kind','actor_value_kind','coef_source','acting','actor_restored','eval_goal_source','fb_bc_coeff','bootstrap_source')})")" \
      || { echo "EVAL $tag FAILED rc=$?"; tail -20 ${out%.json}.log; status=1; }
  done
done
echo "===== smoke done status=$status"; exit $status

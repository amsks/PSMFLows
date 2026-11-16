#!/bin/bash
# CPU smoke of diag_critic_compare on one task, small sizes, both kinds.
set -e
cd /mnt/home/amohan/git/Austin/PSMFLows
export PSM_DATA=/mnt/home/amohan/psm-data OGBENCH_DATASET_DIR=/mnt/home/amohan/.ogbench/data MUJOCO_GL=egl JAX_PLATFORMS=cpu
OUT=$PSM_DATA/psmgoal-vs-dfb-2026-11-09/smoke
mkdir -p $OUT
.venv/bin/python psmgoal-vs-dfb-2026-11-09/diag_critic_compare.py --kind dfb \
  --run_dir $PSM_DATA/exp/PSMFLows/dfb_latent_oc_bc0p3_cube/sd000_aws_500k --epoch 500000 \
  --tasks 1 --n_states 16 --n_u 8 --report_out $OUT/smoke_dfb.json
.venv/bin/python psmgoal-vs-dfb-2026-11-09/diag_critic_compare.py --kind psmgoal \
  --run_dir $(ls -d $PSM_DATA/exp/PSMFLows/psmgoal_recipe_bc03_cube/sd000_*) --epoch 500000 \
  --tasks 1 --n_states 16 --n_u 8 --report_out $OUT/smoke_psmgoal.json

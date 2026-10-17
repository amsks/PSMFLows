#!/usr/bin/env bash
# Upload finished frozen-step checkpoints to the HF dataset repo for aws's eval500 (2026-10-17).
#
# Every 20 min: for each $PSM_DATA/exp/PSMFLows/psmgoal_ja_fbc3_pt_fd{plus,minus,rand,zero}_cube/sd00*/ that
# holds params_500000.pkl and has not been uploaded, push flags.json, params_250000.pkl, params_500000.pkl and
# train.csv to runs/<group>/<run dir>/. aws's watcher picks up any new runs/psmgoal_ja_fbc3_pt_fd*_cube/sd*/
# params_500000.pkl and evaluates it (aws post 2026-10-17T10-34-17Z). Exits when all 12 runs are uploaded.
#
#   PSM_DATA=... HF_TOKEN=... nohup bash scripts/campaign_2026-10/fd_upload_watcher.sh \
#     > $PSM_DATA/logs/slurm/watcher_fd_upload.log 2>&1 &
PSM_DATA=${PSM_DATA:-/mnt/home/amohan/psm-data}; REPO=${PSM_REPO:-/mnt/home/amohan/git/Austin/PSMFLows}
MARK=$PSM_DATA/gc_scripts/uploaded_fd; mkdir -p "$MARK"
echo "$(date +%F_%T) watching for finished frozen-step runs"
while true; do
  n_up=0
  for run in "$PSM_DATA"/exp/PSMFLows/psmgoal_ja_fbc3_pt_fd{plus,minus,rand,zero}_cube/sd00*; do
    [ -f "$run/params_500000.pkl" ] || continue
    g=$(basename "$(dirname "$run")"); rd=$(basename "$run")
    if [ -f "$MARK/${g}_${rd}" ]; then n_up=$((n_up+1)); continue; fi
    if "$REPO/.venv/bin/python" - "$run" "$g" "$rd" <<'PY'
import os, sys
from huggingface_hub import HfApi
run, g, rd = sys.argv[1:4]
api = HfApi(token=os.environ.get('HF_TOKEN'))
for name in ('flags.json', 'params_250000.pkl', 'params_500000.pkl', 'train.csv'):
    p = os.path.join(run, name)
    if os.path.exists(p):
        api.upload_file(path_or_fileobj=p, path_in_repo=f'runs/{g}/{rd}/{name}', repo_id='amsks/psmflows-preimages',
                        repo_type='dataset', commit_message=f'kisski: {g}/{rd}/{name}')
PY
    then echo "$(date +%F_%T) uploaded runs/$g/$rd/"; touch "$MARK/${g}_${rd}"; n_up=$((n_up+1))
    else echo "$(date +%F_%T) upload FAILED for $g/$rd, will retry"; fi
  done
  [ "$n_up" -ge 12 ] && { echo "$(date +%F_%T) all 12 uploaded; ALLDONE"; exit 0; }
  sleep 1200
done

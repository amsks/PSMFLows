#!/usr/bin/env bash
# Three-way check of the nf2 500k checkpoints trained on luis (seeds 1 and 2), 2026-10-17.
#
# Polls the HF dataset repo for runs/psmgoal_ja_fbc3_pt_nf2_cube/sd00{1,2}*/params_500000.pkl that
# did not come from kisski (kisski's sd001/sd002 dirs hold 250k only). For each: downloads
# params_500000.pkl + flags.json into $PSM_DATA/exp/PSMFLows/psmgoal_ja_fbc3_pt_nf2l_cube/<run dir>/,
# checks the flags carry norm_future_k=2, and submits one packed five-task eval500 (hgoal_each + gpi,
# 500 episodes per task) named psmgoal_ja_fbc3_pt_nf2l_cube_<sd>_500000_hgoal_each so that
# scripts/campaign_2026-10/agg_results.py aggregates it as its own group. luis evaluates the same
# checkpoints on luis and aws on aws (mailbox post 2026-10-17); together with kisski's numbers that
# is the three-machine parity table and, with sd000 (kisski, 0.912), the three-seed nf2 500k number.
# When kisski's ten JSONs exist the watcher prints the aggregate beside critics A and B and exits.
#
#   PSM_DATA=... PSM_REPO=... HF_TOKEN=... nohup bash scripts/campaign_2026-10/nf2_threeway_watcher.sh \
#     > $PSM_DATA/logs/slurm/watcher_nf2_threeway.log 2>&1 &
PSM_DATA=${PSM_DATA:-/mnt/home/amohan/psm-data}; REPO=${PSM_REPO:-/mnt/home/amohan/git/Austin/PSMFLows}
GROUP_SRC=psmgoal_ja_fbc3_pt_nf2_cube; GROUP_DST=psmgoal_ja_fbc3_pt_nf2l_cube
DST=$PSM_DATA/exp/PSMFLows/$GROUP_DST; MARK=$PSM_DATA/gc_scripts/submitted_nf2_threeway
mkdir -p "$DST" "$MARK" "$PSM_DATA/logs/slurm"
KISSKI_DIRS="sd001_s_2525380.0.20261015_164910 sd002_s_2525381.0.20261015_164911 sd000_s_2525384.0.20261015_165409"
echo "$(date +%F_%T) watching HF runs/$GROUP_SRC for luis sd001/sd002 params_500000"
while true; do
  # 1. find and fetch new luis checkpoints
  "$REPO/.venv/bin/python" - "$GROUP_SRC" "$DST" "$KISSKI_DIRS" <<'EOF'
import os, sys, json, shutil
from huggingface_hub import HfApi, hf_hub_download
src, dst, kisski = sys.argv[1], sys.argv[2], sys.argv[3].split()
api = HfApi(token=os.environ.get('HF_TOKEN'))
files = api.list_repo_files('amsks/psmflows-preimages', repo_type='dataset')
for f in files:
    if not (f.startswith(f'runs/{src}/') and f.endswith('/params_500000.pkl')):
        continue
    run = f.split('/')[2]
    if run in kisski or not (run.startswith('sd001') or run.startswith('sd002')):
        continue
    out = os.path.join(dst, run)
    if os.path.exists(os.path.join(out, 'params_500000.pkl')):
        continue
    os.makedirs(out, exist_ok=True)
    for name in ('flags.json', 'params_500000.pkl'):
        p = hf_hub_download('amsks/psmflows-preimages', f'runs/{src}/{run}/{name}', repo_type='dataset',
                            local_dir='/tmp/nf2_threeway_dl', token=os.environ.get('HF_TOKEN'))
        shutil.move(p, os.path.join(out, name))   # /tmp and $PSM_DATA are different filesystems
    fl = json.load(open(os.path.join(out, 'flags.json'))).get('agent', {})
    ok = int(fl.get('norm_future_k', 0)) == 2 and str(fl.get('actor_kind')) == 'flowbc' and float(fl.get('fb_bc_coeff', 0)) == 3.0
    print(f'fetched {run}: norm_future_k={fl.get("norm_future_k")} actor_kind={fl.get("actor_kind")} fb_bc_coeff={fl.get("fb_bc_coeff")} -> {"OK" if ok else "FLAGS MISMATCH, not evaluated"}')
    if not ok:
        os.rename(out, out + '.flags_mismatch')
EOF
  # 2. submit evals for fetched checkpoints
  n_done=0
  for run in "$DST"/sd00*; do
    [ -f "$run/params_500000.pkl" ] || continue
    sd=$(basename "$run" | cut -c1-5); out=${GROUP_DST}_${sd}_500000_hgoal_each
    done_json=0; for t in 1 2 3 4 5; do [ -f "$PSM_DATA/logs/${out}_task${t}.json" ] && done_json=$((done_json+1)); done
    [ $done_json = 5 ] && { n_done=$((n_done+1)); continue; }
    [ -f "$MARK/$out" ] && continue
    id=$(sbatch --parsable --nice=600 --partition=kisski-inference --account=general --time=03:00:00 \
      --job-name=e500p_$out --output="$PSM_DATA"/logs/slurm/%x-%j.out --error="$PSM_DATA"/logs/slurm/%x-%j.err \
      --export="ALL,PSM_REPO=$REPO,PSM_DATA=$PSM_DATA,MODE=psmgoal,ENVKEY=cube,RUN_DIR=$run,OUT=$out,RESTORE_EPOCH=500000,EVAL_WORKERS=4,EXTRA=agent.coef_source=hgoal_each agent.acting=gpi" \
      "$REPO"/scripts/slurm/eval500_packed.sbatch) && { echo "$(date +%F_%T) submitted $out job $id"; echo "$id" > "$MARK/$out"; }
  done
  # 3. done?
  if [ "$n_done" -ge 2 ]; then
    echo "$(date +%F_%T) both luis seeds evaluated on kisski; aggregate:"
    (cd "$REPO" && .venv/bin/python scripts/campaign_2026-10/agg_results.py 2>/dev/null | grep -E 'nf2l|nf2_cube +500000|fbc3_pt_cube +500000 +hgoal_each|fbcBC_pt_cube +500000')
    echo "$(date +%F_%T) ALLDONE"; exit 0
  fi
  sleep 1800
done

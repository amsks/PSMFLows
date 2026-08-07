# Preimages: how to get them and how to run

The PSMFlow pipeline has three steps. Step 2 is the expensive one (4-19 h on a GPU), and
its output is published so you do not have to run it.

| step | what it does | how long | do you need to run it? |
|---|---|---|---|
| behaviour flow | fits `G(s, u)`, a flow that maps Gaussian noise to dataset actions | ~1 h | no, checkpoint published |
| inversion | finds, per transition, the `u` that decodes to the recorded action | 4-19 h | **no, output published** |
| representation | trains `phi` / `psi` / the latent actor on those latents | ~3 h | yes |

`HF_REPO` below is the Hugging Face dataset repo holding the published artifacts.

## 1. Download

```bash
pip install huggingface_hub
export HF_REPO=<user>/<name>          # private repo: also `hf auth login`
export PSM_DATA=/path/for/big/files   # ~1.1 GB for all three environments

python - <<'EOF'
import os
from huggingface_hub import snapshot_download
snapshot_download(os.environ['HF_REPO'], repo_type='dataset',
                  local_dir=os.environ['PSM_DATA'])
EOF
```

You get:

```
$PSM_DATA/preimages/<env>.npz              the training input
$PSM_DATA/preimages/<env>.npz.meta.json    provenance sidecar
$PSM_DATA/flow/<env>/params_500000.pkl     the frozen behaviour flow
$PSM_DATA/flow/<env>/flags.json
```

`<env>` is one of `pointmaze-medium-navigate`, `cube-single-play`,
`antmaze-medium-navigate`. To fetch one environment instead of all three, pass
`allow_patterns=['*cube-single-play*']`.

## 2. Point the pairing guard at your copy

Latents mean nothing except for the exact flow that produced them, so `main.py` refuses to
start unless the checkpoint you pass matches the one recorded in the sidecar. The
published sidecars record the absolute path on the machine that produced them, which will
not exist on yours. Rewrite it once, after downloading:

```bash
python - <<'EOF'
import glob, json, os
data = os.environ['PSM_DATA']
for meta in glob.glob(f'{data}/preimages/*.meta.json'):
    env = os.path.basename(meta).replace('.npz.meta.json', '')
    with open(meta) as f:
        m = json.load(f)
    m['restore_path'] = f'{data}/flow/{env}'
    with open(meta, 'w') as f:
        json.dump(m, f, indent=2)
    print('repaired', meta)
EOF
```

The guard compares resolved paths, so `flow_ckpt_path` must then be exactly
`$PSM_DATA/flow/<env>` (or a glob resolving to it).

## 3. Set up the repo

```bash
git clone https://github.com/amsks/PSMFLows.git && cd PSMFLows
python -m venv .venv && .venv/bin/pip install -r requirements.txt
```

Three environment variables matter:

```bash
export OGBENCH_DATASET_DIR=/path/for/ogbench   # keeps ~10 GB of datasets off your home dir
export MUJOCO_GL=egl                           # headless machines; pointmaze builds a renderer
export XLA_PYTHON_CLIENT_PREALLOCATE=false     # so several runs can share a GPU
```

The OGBench datasets download themselves on first use. Run one seed per GPU: two seeds
sharing a device run about twice as slow each, for no gain in throughput.

## 4. Train

```bash
ENV=cube-single-play-singletask-v0
NAME=cube-single-play

CUDA_VISIBLE_DEVICES=0 .venv/bin/python main.py \
  agent=psmflow \
  env_name=$ENV \
  agent.flow_ckpt_path=$PSM_DATA/flow/$NAME \
  agent.flow_ckpt_epoch=500000 \
  agent.preimage_path=$PSM_DATA/preimages/$NAME.npz \
  agent.use_point_preimage=true \
  offline_steps=500000 eval_interval=50000 eval_episodes=50 \
  save_dir=$PSM_DATA/exp seed=0
```

Env names: `pointmaze-medium-navigate-singletask-task1-v0`,
`cube-single-play-singletask-v0`, `antmaze-medium-navigate-singletask-v0`.

`scripts/launch_psmflow.sh` wraps this for multiple seeds:

```bash
FLOW_CKPT=$PSM_DATA/flow/$NAME FLOW_EPOCH=500000 \
PREIMAGES=$PSM_DATA/preimages/$NAME.npz \
SEEDS="0" bash scripts/launch_psmflow.sh $ENV 0 500000 offline
```

### Which preimage arm to use

`agent.use_point_preimage=true` uses the single backward-ODE latent `u*` per transition.
This is what every current result uses, and the one to start from.

`false` samples from the stored Gaussian mixture instead. It is healthy on pointmaze and
cube, but **broken on antmaze** — mean ESS 7.6 with only 6% of rows above 20, because
`alpha=20` was tuned at `d_a=5` and does not transfer to `d_a=8`.

### Evaluating a checkpoint

```bash
MUJOCO_GL=egl .venv/bin/python tools/eval_checkpoint.py agent=psmflow env_name=$ENV \
  agent.flow_ckpt_path=$PSM_DATA/flow/$NAME agent.flow_ckpt_epoch=500000 \
  agent.preimage_path=$PSM_DATA/preimages/$NAME.npz \
  restore_path='<run dir>' restore_epoch=500000 eval_episodes=500
```

Use 500 episodes for any number you intend to report; the 50-episode evaluations logged
during training swing by ±0.15 between consecutive points. Report the mean and 95%
confidence interval across seeds, never the best seed or the peak of a curve. The
comparison that matters is against the behaviour flow acting alone, evaluated from the
same checkpoint the agent decodes through — `agent.bc_only=true` makes `sample_actions`
draw a fresh prior latent each step and decode it, which is exactly that control:

```bash
MUJOCO_GL=egl .venv/bin/python tools/eval_checkpoint.py agent=fql agent.bc_only=true \
  env_name=$ENV restore_path=$PSM_DATA/flow/$NAME restore_epoch=500000 eval_episodes=500
```

## 5. Regenerating preimages (only if you change the flow)

```bash
.venv/bin/python tools/precompute_preimages.py agent=fql env_name=$ENV \
  restore_path=$PSM_DATA/flow/$NAME restore_epoch=500000 \
  agent.flow_steps=100 inversion.n_initial_steps=100 \
  inversion.alpha=20 inversion.num_samples=200 \
  preimage_out=$PSM_DATA/preimages/$NAME.npz
```

Constraints the script asserts:

- `agent.flow_steps` must equal `inversion.n_initial_steps`, and both must be >= 100. The
  implicit-Euler inverse diverges at the training default of 10.
- The flow must be trained. Preimages of an untrained flow carry no behaviour information.

Almost all of the runtime is one term: every EM iteration re-decodes all
`inversion.num_samples` proposals through the full `flow_steps` ODE. `tools/bench_forward_steps.py`
measured on cube that this scoring pass holds up at **20** steps while the inverse stays
pinned at 100 — 4.9x faster, ESS 79 vs 84, same decode error. This is not yet a flag:
`precompute_preimages.py` asserts `flow_steps == n_initial_steps`, so taking the speedup
means decoupling the two there, as the benchmark does.

Uploading a regenerated set: `scripts/upload_preimages_hf.py --repo-id <repo> --dry-run`
lists what it would push; drop `--dry-run` to upload.

## Where the files came from

Each `.npz` carries the OGBench transitions plus the preimage arrays, so it is a drop-in
training input. Array shapes and per-environment quality numbers are in the dataset card
on Hugging Face.

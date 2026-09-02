# Preimages: how to get them and how to run

The PSMFlow pipeline has three steps. Step 2 is the expensive one (4-19 h on a GPU), and
its output is published so you do not have to run it.

| step | what it does | how long | do you need to run it? |
|---|---|---|---|
| behaviour flow | fits `G(s, u)`, a flow that maps Gaussian noise to dataset actions | ~1 h | no, checkpoint published |
| inversion | finds, per transition, the `u` that decodes to the recorded action | 4-19 h | **no, output published** |
| representation | trains `phi` / `psi` / the latent actor on those latents | ~3 h | yes |

The artifacts live in the Hugging Face dataset repo `amsks/psmflows-preimages` (private;
ask for access).

## 1. Download

```bash
pip install huggingface_hub
export HF_REPO=amsks/psmflows-preimages   # private repo: also `hf auth login`
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

### Checking that the latents recover the data

Both checks below run against the published artifacts in one command, downloading them
first if they are not already in `$PSM_DATA`:

```bash
PSM_DATA=/path/for/big/files bash scripts/run_recovery_tests.sh all   # SMOKE=1 for a wiring check
```

It calls `scripts/fetch_preimages_hf.sh`, which is also usable on its own: it downloads
one or all environments and rewrites the sidecars' `restore_path` to your copy, which is
step 2 above and what the pairing guard needs. Reports land in `$PSM_DATA/logs`. What the
two tools measure:

`tools/validate_decode_recovery.py` reads an npz plus the flow it was inverted from — no
env, no MuJoCo — and answers two questions:

1. **Decode recovery.** Draw `u` from each row's stored preimage mixture, decode it, and
   compare against the recorded action: MSE, and the min/median/max of the per-row L2.
   `+t1_sources=[point]` reproduces the stored `preimage_roundtrip` (the exact backward-ODE
   latent, i.e. the floor); `[prior]` decodes a fresh prior latent at the same state, the
   uninformed reference.

   `nan_decode_frac` is reported per latent source and is not always zero: the mixture
   draws far outside the N(0, I) support the flow was fitted on (60% of cube draws exceed
   `||u||=3`, 12% exceed 5), and out there the velocity field extrapolates into a runaway
   — `||v||=116` at step 0 for `||u||=9.4`, `||x||` reaching 1e14 by step 30 and NaN by
   step 70. It is 0% below `||u||=5` and 18% above 9. `+u_clip=3.0` clamps the latents the
   way `psmflow` already clamps them before use (`sample_step_inputs`), which removes the
   divergences and is what training actually consumes; leaving it unset measures the
   stored posterior as it is. Clipping the *state* to the action box each Euler step also
   stops the runaway but is not neutral — at `t=0` the state is the latent, which lives in
   a Gaussian of radius ~2.2, so it moves healthy decodes by 0.16 where clamping `u` moves
   them by 0.03.

   The point preimage only matches the stored value on the *same device* that wrote the
   npz.
   The 100-step implicit-Euler inverse is float32-sensitive: re-inverting the published
   pointmaze rows on CPU moves the latent by 0.0065 (against `||u|| ~ 1.25`), and the
   decode turns that into 0.0037 of action error where the npz records 7.6e-5. Both are
   far below the mixture, so it changes no conclusion — but read a point-preimage number
   in the 1e-3 range as the device, not the inversion.
2. **Buffer recovery.** At each of `t2_states` states, decode `t2_latents` prior latents
   and search *all* buffer rows for the nearest states, then report the smallest distance
   to their recorded actions. Read it against the two baselines printed beside it: the
   query row's own action scored the same way, and a uniform random action.

```bash
.venv/bin/python tools/validate_decode_recovery.py agent=fql env_name=$ENV \
  restore_path=$PSM_DATA/flow/$NAME restore_epoch=500000 agent.flow_steps=100 \
  +preimage_npz=$PSM_DATA/preimages/$NAME.npz
```

`agent.flow_steps` must equal the `flow_steps` in the sidecar (asserted): a decode under a
different discretization measures that mismatch instead. Sizes are `preimage_limit`
(test 1 rows, default 50000), `+t2_states` / `+t2_latents` / `+t2_k`; `+tests=[1]` runs one
of the two. Keys absent from `configs/config.yaml` need the leading `+`.

### Checking that the decoded actions move the simulator where the data did

`tools/validate_dynamics_recovery.py` scores the same latents against the environment
instead of against the recorded actions. It puts MuJoCo back at a recorded transition with
`set_state(qpos, qvel)` — OGBench keeps those arrays row-aligned with `observations`, and
`envs/env_utils.make_env_and_datasets(..., add_info=True)` is what stops them being
dropped — then steps the decoded action from that state.

```bash
MUJOCO_GL=egl .venv/bin/python tools/validate_dynamics_recovery.py agent=fql env_name=$ENV \
  restore_path=$PSM_DATA/flow/$NAME restore_epoch=500000 agent.flow_steps=100 \
  +preimage_npz=$PSM_DATA/preimages/$NAME.npz +n_states=512
```

Read `next_obs_vs_replay`: the decoded action's next state against the *recorded* action's
next state, both stepped from the same restored state in the same process. Its scales are
`data_step` (how far one recorded step moves the state at all) and `random`, both
printed beside it. `+latent_sources=[mixture,point,prior]` selects where the latents come from.

`next_obs_vs_dataset` compares against the recorded next observation instead, and is only
as tight as the `replay_vs_dataset` floor above it — replay the *recorded* action and you
land 1.4e-6 from the recorded next state on pointmaze but **0.022** on cube (mean step
0.47), because the local MuJoCo is not the one that generated the datasets. The same
mismatch shows in `restore_obs_err`: exactly 0 on pointmaze, while on cube qpos/qvel are
set exactly but the gripper-contact bit and ~0.5 mm at the effector site do not reproduce
(`restore_err_per_dim_mean` says which coordinates). Restore-and-step is deterministic to
the bit, so nothing here is noise you can average away — use `next_obs_vs_replay`, where
it cancels.

### Tuning the inversion on a sampled batch

`docs/TUNING_INVERSION.md` is the short version: which knob moves what, what to read off the
report, and the commands. The rest of this section is the measurements behind it.

Inverting 1M rows costs 4-19 h, which is the wrong loop to tune in. Two pieces make the
short loop:

```bash
# score several settings against ONE sampled batch, in one process
MUJOCO_GL=egl .venv/bin/python tools/tune_preimage_inversion.py agent=fql env_name=$ENV \
  restore_path=$PSM_DATA/flow/$NAME restore_epoch=500000 agent.flow_steps=100 \
  +n_rows=2000 '+sweep={prior_scale: [0.0, 1.0], alpha: [5.0, 20.0, 50.0]}'

# then persist the batch under a chosen setting and run the full recovery tests on it
.venv/bin/python tools/precompute_preimages.py agent=fql env_name=$ENV \
  restore_path=$PSM_DATA/flow/$NAME restore_epoch=500000 \
  agent.flow_steps=100 inversion.n_initial_steps=100 \
  +preimage_sample=2000 preimage_out=$PSM_DATA/preimages/${NAME}_batch.npz
```

Measured on 256 cube rows at the production inversion settings (`alpha=20`,
`num_samples=200`, `n_steps=10`), against the published flow:

| `prior_scale` | E‖u‖² (want 5) | KS | beyond χ²₉₉ | NaN | decode | ESS/200 |
|---|---|---|---|---|---|---|
| 0.0 (published npz) | 15.97 | 0.528 | 0.309 | 0.0039 | 0.221 | 81.0 |
| 1.0 (posterior) | 3.84 | 0.172 | 0.000 | 0.0000 | 0.164 | 117.5 |

**What the mixture is for, and how the sweep scores it.** The mixture densifies the data
for the successor measure: instead of one latent per transition it attaches a whole
ε-region of latents to the recorded outcome, so the measure is fitted over latent space
rather than at isolated points. Every latent admitted asserts "policy `u` would have taken
`a` here", which is exactly true only at `u*`, so ε trades **latent-space coverage against
label noise** — `alpha` is `1/ε` and is the primary knob.

The sweep reports both sides. `covered_at_k` asks the coverage question directly: draw
`u ~ p0` at a state, and is it captured under any mixture component of that state (k=0) or
of its k nearest states? `covered_by_prior_quantile` says where the misses fall — a miss
beyond `u_clip` costs nothing, since psmflow clamps every draw to that box, whereas a miss
in the bulk is the failure the metric exists to catch. `decode_mix` is the label noise the
coverage was bought with. Coverage alone is maximised by inflating the covariances, so
read the two together and pick the widest setting inside a decode-error budget.

Coverage is strongly k-dependent, so read the curve, not a number. Pointmaze at
`prior_scale=1`, 1280 rows sampled as neighbourhoods:

| | k=0 | k=1 | k=4 | k=16 | k=64 | decode |
|---|---|---|---|---|---|---|
| `alpha=20` | 0.03 | 0.06 | 0.14 | 0.38 | 0.74 | 0.077 |
| `alpha=50` | 0.01 | 0.02 | 0.05 | 0.15 | 0.40 | 0.035 |

It roughly doubles per doubling of k and has not saturated by 64, so "how well is the
prior covered" is mostly a question about how far the measure generalises across states.
The misses concentrate in the tail — at `alpha=20`, k=64 the prior's inner quartile is 88%
covered and its outer 5% is 17% — which is the benign case, since `u_clip` means the actor
never emits the tail anyway.

Cube (`d_a=5`, `prior_scale=1`, `n_steps=4`, `num_samples=32`, k=64) is the case that
matters, and it reads very differently from pointmaze:

| `alpha` | coverage | inner quartile | decode | E‖u‖² (want 5) |
|---|---|---|---|---|
| 2 | 0.60 | 0.98 | 0.232 | 4.81 |
| 5 | 0.53 | 0.95 | 0.193 | 4.66 |
| 10 | 0.41 | 0.85 | 0.153 | 4.56 |
| 20 (current default) | 0.19 | 0.48 | 0.109 | 4.65 |
| 50 | 0.02 | 0.04 | 0.052 | 5.07 |

Four things to know before picking a setting:

- **Coverage falls off geometrically with `d_a`.** The same `alpha` covers 0.74 of the
  prior on pointmaze (`d_a=2`) and 0.19 on cube (`d_a=5`), because an ellipsoid of fixed
  per-dimension width covers a volume fraction that shrinks with dimension. Antmaze
  (`d_a=8`) will be worse again.
- **With the prior in the target, MORE EM iterations shrink the mixture.** On cube,
  `n_steps` 4 -> 10 drops coverage 0.19 -> 0.01. Without the prior the target is improper
  along the decode level set and EM drifted outward (the ~4x-per-iteration covariance
  growth noted in `_condition`); with it, EM converges onto the point preimage as it
  should. So `n_steps` is now a width knob pointing the opposite way, and the
  `num_samples=200, n_steps=10` production setting is a *concentrating* one.
- **`num_samples` dominates, and the table above is at a starved budget.** It was
  measured at `num_samples=32`, an eighth of the production 200. Re-run at 128 with
  `alpha=5`, coverage goes 0.53 -> 0.96 at k=64 and 0.02 -> 0.29 at k=0, for a decode
  error of 0.227 against 0.193. The EM proposal is fitted from those draws, so too few of
  them yield a covariance shrunk toward the point preimage. Read the `alpha` frontier as
  the *shape* of the trade-off, not as absolute coverage.
- **More mixture components do not buy coverage.** Sweeping `num_clusters` 1/2/3/5 at
  fixed `num_samples=32` (cube, k=64) moves coverage 0.53/0.39/0.22/0.11 at `alpha=5` and
  0.19/0.13/0.05/0.01 at `alpha=20`, monotonically down, with the prior's inner quartile
  falling the same way (0.95 -> 0.28) and the decode error improving only ~6%. The union
  of K components fitted from N/K draws each is smaller than one component fitted from N.
  At `num_samples=128` the K dependence mostly disappears (k=64: 0.962 at K=1, 0.978 at
  K=3), so `num_clusters=1` is a defensible default and K is at best a refinement once the
  sample budget is adequate.

Typicality stays put across the whole `alpha` range (E‖u‖² 4.5-5.1 against 5) — with
`prior_scale=1` the prior pins where the mass sits while `alpha` sets how wide it spreads,
which is the separation of concerns the likelihood-only target did not have.

The sweep runs every combination against the same rows and the same draw seed, so a
difference between rows is the setting. Note that ESS is bounded by `num_samples` and is
measured against whatever target the setting defines, so compare it as ESS/N and only
within a column — it is not a yardstick across `prior_scale` values, and it says nothing
about typicality (the published cube npz scored ESS 81/200 while its draws sat 3x outside
the prior). It reports typicality (mean `||u||^2` against
`E[chi^2_{d_a}]`, KS, fraction past the 99th percentile), the mixture and point decode
errors, `nan_decode_frac`, and ESS, plus seconds per setting.

`+preimage_sample=N` inverts a random N-row batch rather than the whole buffer and records
`source_index`, the buffer row each sample came from — `validate_dynamics_recovery.py`
needs it to look up the right `qpos`/`qvel`. The sidecar marks these files
`sampled_batch: true`: they are tuning artifacts, and `main.py` will reject them as
training input, which is intended.

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
on Hugging Face (`amsks/psmflows-preimages`).

---

## Generating preimages on another cluster (SLURM)

Stage B is 12-25 h of single-GPU work per environment and needs nothing from this machine
except the frozen Stage-A flow. The Hugging Face dataset repo is the only shared state
between clusters: pull the flow, invert, push the npz back.

`scripts/hf_preimages.py` is the transport (`scripts/upload_preimages_hf.py` still
publishes the three canonical environments from hardcoded paths; this one moves arbitrary
artifacts under arbitrary names, which is what variants need).

### One-time setup on the cluster

```bash
git clone https://github.com/amsks/PSMFLows.git && cd PSMFLows
# create .venv per the repo README, then:
export HF_TOKEN=hf_...            # the repo is private; a WRITE token to push back
export PSM_REPO=$PWD
export PSM_DATA=/scratch/$USER/psmflows     # ~2 GB per environment
python scripts/hf_preimages.py list         # confirms auth and shows what exists
```

OGBench datasets download themselves on first use into `$OGBENCH_DATASET_DIR`; they are
deterministic, so the rows a cluster inverts are the same rows this machine trains on.
`main.py` verifies that anyway (row count, plus the first and last 1000 rows compared
against the env dataset).

### Submit

```bash
sbatch --partition=<gpu-partition> --account=<acct> \
  --export=ALL,ENVKEY=antmaze,NAME=antmaze-medium-navigate-a26p5-ps0p60-ns5-N200,\
ALPHA=26.51,PRIOR_SCALE=0.597,N_STEPS=5,NUM_SAMPLES=200 \
  scripts/slurm/precompute_preimages.sbatch
```

`ENVKEY` is `cube|antmaze|pointmaze`. `NAME` is free-form; put the inversion settings in it,
because two npz files that differ only in `alpha` are otherwise indistinguishable after
download. The job pulls the flow, runs `tools/precompute_preimages.py`, and pushes the npz
plus its sidecar back. Wall clock defaults to 36 h -- raise it rather than lose a 20 h job.

Pick `ALPHA` / `PRIOR_SCALE` / `N_STEPS` from an HPO sweep on the same corrected target
(`tools/hpo_preimage_inversion.py`, ~30 min for 50 trials), not from the frontier table in
`configs/inversion/default.yaml`, which is a coarser grid.

### Retrieve on the training machine

```bash
python scripts/hf_preimages.py pull --name <NAME> --dest $PSM_DATA --with-flow
```

`--with-flow` is not optional if you intend to train. Every npz records the ABSOLUTE path of
the checkpoint it was inverted from, on the producing machine, and `main.py` compares
realpaths and refuses a mismatch; `--with-flow` downloads the flow and rewrites the sidecar
to point at the local copy. The command prints the exact
`agent.preimage_path=... agent.flow_ckpt_path=... agent.flow_ckpt_epoch=...` to paste into a
launch line.

Verified end to end on 2026-09-02: `pull --with-flow` of the published pointmaze artifact
into a clean directory, then 20 training steps through `main.py`, pairing guards passing.

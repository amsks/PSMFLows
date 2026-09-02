# Handoff: antmaze preimage generation on a SLURM cluster

**Date:** 2026-09-02 · **Repo:** https://github.com/amsks/PSMFLows ·
**Branch:** `feat/inversion-integration` at `89c03a6` or later — **not `main`**

## What you are doing

Stage B of the PSMFlows pipeline for antmaze. A conditional behaviour-cloning flow
`G(s, u)` maps Gaussian noise to dataset actions; this job inverts it over a 1M-transition
OGBench dataset to recover, for every transition, the latent `u` that decodes to the
recorded action, plus a fitted Gaussian mixture over the neighbourhood of that latent.

Output is one `.npz` (~530 MB) plus a provenance sidecar, published to the Hugging Face
dataset repo `amsks/psmflows-preimages`. **That repo is the only shared state between your
cluster and the training machine** — you never need filesystem access to ours, and we never
need any to yours.

All the machinery exists and has been tested end to end. Your job is to wire it to your
scheduler and submit one job. **Do not write a new precompute script.**

## Setup

```bash
git clone https://github.com/amsks/PSMFLows.git && cd PSMFLows
git checkout feat/inversion-integration
# build .venv per the repo README (JAX with CUDA, flax, optax, ogbench, huggingface_hub)

export HF_TOKEN=hf_...                        # WRITE token; the dataset repo is private
export PSM_REPO=$PWD
export PSM_DATA=/scratch/$USER/psmflows       # needs ~2 GB free

.venv/bin/python scripts/hf_preimages.py list
```

`list` should print three preimages and three flows. **If it fails, stop and fix auth** —
everything downstream depends on that token, and the job's final step is a push.

## Smoke test first — five minutes, do not skip

```bash
.venv/bin/python scripts/hf_preimages.py pull \
    --name antmaze-medium-navigate --dest $PSM_DATA --flow-only

MUJOCO_GL=egl OGBENCH_DATASET_DIR=$PSM_DATA/ogbench \
.venv/bin/python tools/precompute_preimages.py agent=fql \
    env_name=antmaze-medium-navigate-singletask-v0 \
    restore_path=$PSM_DATA/flow/antmaze-medium-navigate restore_epoch=500000 \
    agent.flow_steps=100 inversion.n_initial_steps=100 \
    inversion.alpha=26.51 inversion.prior_scale=0.597 inversion.n_steps=5 \
    inversion.num_samples=200 preimage_limit=512 \
    preimage_out=/tmp/smoke.npz
```

This is the full code path on 512 rows instead of 1,000,000. If it completes and writes
`/tmp/smoke.npz`, submit the real job. A 20-hour job that dies in minute three on a shape
error is the failure mode this prevents.

## Submit

```bash
sbatch --partition=<your-gpu-partition> --account=<your-account> \
  --export=ALL,ENVKEY=antmaze,NAME=antmaze-medium-navigate-a26p5-ps0p60-ns5-N200,\
ALPHA=26.51,PRIOR_SCALE=0.597,N_STEPS=5,NUM_SAMPLES=200 \
  scripts/slurm/precompute_preimages.sbatch
```

The script has no `--partition` or `--account` baked in, deliberately: those are
cluster-specific and it is better for a submit to fail loudly than to inherit a wrong
default.

What the job does, in three steps:

1. pulls the frozen Stage-A flow — decoder only, 56 MB, not the 530 MB canonical npz;
2. runs `tools/precompute_preimages.py` over the full dataset;
3. pushes `preimages/<NAME>.npz` and its sidecar back to the HF repo.

One GPU, ~20 h expected, 36 h wall clock in the script. Raise the wall clock rather than
lose a 20-hour job.

### Where the hyperparameters come from

`ALPHA=26.51`, `PRIOR_SCALE=0.597`, `N_STEPS=5` are the incumbent of a 50-trial SMAC sweep
run 2026-09-02 (`tools/hpo_preimage_inversion.py`) against the **corrected** inversion
target. Two earlier sweeps exist and are invalid — they optimized an un-squared likelihood
with a mismatched Laplace covariance, both fixed in commit `313948e`.

**Do not substitute values from `configs/inversion/default.yaml`.** That file's table is a
coarse grid answering a different question, and its `alpha: 50.0` default is tuned for cube.

`NUM_SAMPLES=200` matches the cube file being regenerated in parallel, so the two stay
comparable. If your wall clock is tight, 128 is the documented fallback at some ESS cost.

## Things that will bite you

- **`agent.flow_steps` must equal `inversion.n_initial_steps`, and both must be ≥ 100.**
  Asserted in the tool. The implicit-Euler inversion *diverges* at the training default of
  10 — measured round-trip error is NaN at 10 steps, KS 0.217 at 30, and 1.2e-4 at 100. The
  sbatch sets both from `FLOW_STEPS` (default 100). Do not lower it to save time.
- **`MUJOCO_GL=egl`** — compute nodes have no DISPLAY and the environment builds a renderer.
- **`XLA_PYTHON_CLIENT_PREALLOCATE=false`** — set in the script; leave it alone.
- Expect a warning that a few hundred of 1M preimages diverged to non-finite and were reset
  to the prior. **This is normal and known** (antmaze 881/1M ≈ 0.09%). It is only a problem
  above 1%, which the tool asserts on.
- OGBench downloads its dataset on first use into `$OGBENCH_DATASET_DIR`. Deterministic, so
  your rows match ours; the training side re-verifies row count plus the first and last
  1000 rows before it will pair the file.
- The sidecar records the **absolute path** of the checkpoint on *your* machine. That is
  expected — the consumer's `pull --with-flow` rewrites it. Do not hand-edit it.

## Report back

- the HF path (`preimages/<NAME>.npz`),
- job wall time,
- the reported invalid-preimage count,
- the tail of the job log.

We retrieve it with:

```bash
python scripts/hf_preimages.py pull --name <NAME> --dest $PSM_DATA --with-flow
```

## Context, so you do not over-interpret the result

**This artifact is expected to be of limited use, and that is not a reason to tune it.**

On its own HPO sweep, antmaze reaches **3.6% coverage** at its optimum, against **98.2%**
for the identical procedure on cube. Several explanations have already been eliminated:

- *local flow geometry* — a Jacobian probe over 2048 transitions found antmaze's smallest
  singular value is **smaller** than cube's (0.050 vs 0.068) with a **larger** free radius,
  so antmaze has more latent slack per unit of action change, not less;
- *the inversion proposal itself* — the settings above are the corrected target, and they
  still give 3.6%.

We are generating this to close the question under a correct inversion, not because we
expect it to work. **Do not sweep further hyperparameters to raise coverage.** The
width/fidelity frontier is understood and monotone: coverage is bought one-for-one with
decode error, because raising `alpha` shrinks the fitted mixture onto the unique point
inverse. The exact inverse of a diffeomorphism is a point, not a set — there is no hidden
multiplicity to find by tuning.

If the job reveals something that contradicts any of the above, that is a genuine finding
and worth reporting. Improving the coverage number by re-tuning is not.

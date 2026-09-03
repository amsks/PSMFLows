# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Branch

Work happens on `feat/inversion-integration`. Do not commit to `main` unless asked.

## What this is

Zero-shot offline RL. A behaviour-cloned conditional flow `G(s,u)` is fit on the dataset and
**frozen**; all RL then happens in that flow's latent space, so every action the model
evaluates, bootstraps or executes is a flow decode and the bootstrap distribution equals the
data distribution. `PAPER/main.tex` (section *LatentFlowPSM*) is the write-up;
`docs/COMPENDIUM.md` is the cold-start summary of theory, code seams, every verified number
and the live hypotheses — **read it before proposing experiments**, it records which
hypotheses are already settled negative.

## Three-stage pipeline

| stage | what | how | cost |
|---|---|---|---|
| A behaviour flow | fits `G(s,u)` by flow matching | `main.py agent=fql agent.bc_only=true` | ~1 h |
| B inversion | per transition, the `u` that decodes to the recorded action | `tools/precompute_preimages.py` | 4–19 h |
| C representation | phi / psi / latent actor | `main.py agent=psmflow` | ~3 h |

A and B are **already published** for `cube-single-play`, `antmaze-medium-navigate`,
`pointmaze-medium-navigate` (HF dataset `amsks/psmflows-preimages`, private). Normal work is
stage C. `docs/PREIMAGES.md` is the operational manual: download, sidecar repair, training,
eval, regeneration.

## Commands

```bash
# tests (CPU is fine; no CI, no pytest config — plain pytest from the repo root)
.venv/bin/python -m pytest tests/ -x -q
.venv/bin/python -m pytest tests/test_psmflow_agent.py::test_name -x
# tests gated on a Stage-A checkpoint skip unless PSMFLOWS_STAGE_A_CKPT points at one

.venv/bin/ruff check .        # line-length 120, single quotes, py310

# stage C training (Hydra; configs/config.yaml + configs/agent/<agent>.yaml)
CUDA_VISIBLE_DEVICES=0 .venv/bin/python main.py agent=psmflow env_name=$ENV \
  agent.flow_ckpt_path=$PSM_DATA/flow/$NAME agent.flow_ckpt_epoch=500000 \
  agent.preimage_path=$PSM_DATA/preimages/$NAME.npz agent.use_point_preimage=true \
  offline_steps=500000 eval_interval=50000 eval_episodes=50 save_dir=$PSM_DATA/exp seed=0

# multi-seed launcher (one seed per GPU; two seeds sharing a device is ~2x slower each)
FLOW_CKPT=... FLOW_EPOCH=500000 PREIMAGES=... SEEDS="0 1 2" GROUP=<name> \
  bash scripts/launch_psmflow.sh $ENV <GPU> 500000 offline

# the only evals that count
GPU=0 bash scripts/eval500.sh psmflow cube <run_dir> <out_name>
GPU=0 bash scripts/eval500.sh bc      cube -        bc_control
.venv/bin/python tools/make_tables.py     # regenerate docs/tables/results.md

# recovery diagnostics against the published artifacts (SMOKE=1 for a wiring check)
PSM_DATA=... bash scripts/run_recovery_tests.sh all

# move preimage artifacts between machines (HF dataset repo is the only shared state)
python scripts/hf_preimages.py list
python scripts/hf_preimages.py pull --name <NAME> --dest $PSM_DATA --with-flow
sbatch --export=ALL,ENVKEY=antmaze,NAME=...,ALPHA=...,PRIOR_SCALE=...,N_STEPS=...,NUM_SAMPLES=200 \
  scripts/slurm/precompute_preimages.sbatch          # stage B elsewhere, ~36 h wall clock
```

Environment: `OGBENCH_DATASET_DIR`, `MUJOCO_GL=egl` (headless), `XLA_PYTHON_CLIENT_PREALLOCATE=false`,
`PSM_DATA` (artifacts), `HF_TOKEN` (the dataset repo is private; a write token to push).
`/var/local` is small and has filled mid-run before — point `save_dir`/`STORE` at the large disk.

## Architecture

`main.py` is the single Hydra entry point for every agent in `agents/__init__.py`'s registry
(`psmflow`, `fql`, `psm`, `fb`, `affine_psm`, `latentrl`, plus baselines). It converts the
Hydra `agent` group into the `ml_collections.ConfigDict` the agents expect, builds the env +
dataset, and for latent-space agents loads the preimage-augmented dataset.

- `agents/fql.py` — the behaviour flow *and its inverse*. `compute_full_proposal_distribution_em`
  is stage B's core: implicit-Euler backward ODE for the point preimage, then EM for a
  Gaussian(-mixture) posterior over `u`.
- `agents/psmflow.py` — the algorithm. Its module docstring carries the code↔write-up symbol
  map (`phi`→basis, `psi`→measure head, `actor`→latent actor, `infer_z`→closed-form reward
  inference) and the TD-target semantics; read it before touching the losses. Switchable
  seams that correspond to write-up variants: `policy_index` (`task_vector`|`latent`),
  `acting` (`actor`|`gpi`), `train_actor`, `backup_explore_frac`, `action_critic.*`.
- `utils/flow_inversion.py` — preimage validity, repair, augmented-dataset IO, mixture sampling.
- `utils/psm_networks.py`, `utils/networks.py` — `PhiMap`, `PsiMap`, `FlowVectorField`, actors.
- `tools/` — `precompute_preimages.py` (stage B), `eval_checkpoint.py`, `validate_*.py`
  (recovery gates), `diag_*.py` (diagnostics, all write JSON via `report_out`), `fig_*.py`,
  `hpo_preimage_inversion.py` (pick `alpha`/`prior_scale`/`n_steps` from a sweep on the
  corrected target, not from the coarse frontier table in `configs/inversion/default.yaml`).
- `scripts/hf_preimages.py` — push/pull arbitrary named artifacts through the HF dataset
  repo, the transport for stage B run on another cluster
  (`scripts/slurm/precompute_preimages.sbatch`). `upload_preimages_hf.py` still publishes the
  three canonical envs from hardcoded paths; the two coexist in the same repo layout.
  Name variants after their inversion settings — two npz files differing only in `alpha` are
  otherwise indistinguishable after download.

### Invariants that bite

- **`utils.xla_guard` must be imported before jax** in every GPU entry point. XLA:GPU's
  autotuner miscompiles the unrolled flow ODE past ~30 steps and silently returns actions
  pinned at the clip. `tests/conftest.py` exists solely to enforce this ordering.
- **Latents are only valid for the exact flow that produced them.** `main.py` refuses to
  start unless the `.npz.meta.json` sidecar's env, `restore_path`, `restore_epoch` and
  dataset subset match the checkpoint being passed — and every npz records the *absolute*
  checkpoint path on the machine that produced it, so a downloaded one never matches until
  its sidecar is repaired. Don't weaken the guard; repair instead —
  `hf_preimages.py pull --with-flow` does it automatically, `docs/PREIMAGES.md` §2 by hand.
- Files marked `sampled_batch: true` are inversion-tuning artifacts, never training input.
- Stage B asserts `agent.flow_steps == inversion.n_initial_steps >= 100`; the implicit-Euler
  inverse diverges at the training default of 10. Changing `alpha`, `prior_scale`,
  `num_samples` or `n_steps` invalidates existing preimage npz files.
- Arm B (`agent.policy_index=latent train_actor=false acting=gpi`) needs those flags at
  **eval as well as training** — psi's index slot changes width, so restoring without them fails.

## Reporting results

500 episodes (`tools/eval_checkpoint.py` / `scripts/eval500.sh`) for anything reported; the
in-loop 50-episode evals swing ±0.15 between consecutive points. Report mean and 95% CI
across seeds, never a peak or a best seed. Always quote the behaviour-cloning control beside
it — the frozen flow acting alone (`agent=fql agent.bc_only=true`) from the same checkpoint
the agent decodes through — since that is what the method has to beat.

## Working discipline (these rules exist because they were violated)

- Before any launch: print the full hyperparameter table, smoke-test the exact code path for
  ~200 steps, and after launch re-read the run's own `flags.json` to confirm the values
  landed. Long jobs go in a named tmux session.
- State expected outcomes — including expected failures — before results arrive.
- `docs/HANDOFF.md` is the dated session record (newest entry first, Marp slides); append an
  entry for work that produced numbers. Design docs and plans live in `docs/design/` and
  `docs/plans/` as `YYYY-MM-DD-slug.md`. `docs/tables/results.md` is generated, not hand-edited.
- Diagnostics must persist their JSON (`report_out`); a 2026-08-03 result was lost to
  stdout-only output.

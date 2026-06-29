# Resuming training runs (crash-recovery)

Crash-recovery resume restores **weights + optimizer + step** and continues the
same wandb run. It is *not* bit-exact (RNG is re-seeded; CUDA+TF32 make mid-stream
bit-exactness impossible anyway). Design:
`PAPER/specs/2026-06-09-training-resume-design.md`.

## PyTorch (FB / PSM / RLDP — state & pixel, `train.py`)

Per-run output dirs are deterministic (`outputs/${wandb_run_name}`, falling back to
`${domain}__${agent}__s${seed}`), so a relaunch lands in the same `save_dir`.

- Auto-resume from the latest `step_*.pt` in the run's dir:
  `python train.py ... wandb_run_name=<same-name> resume=true`
- Resume from an explicit checkpoint:
  `python train.py ... resume_from=outputs/<run>/checkpoints/step_400000.pt`
- A fresh run refuses to overwrite a non-empty `save_dir` (pass `force=true` to override).

Each checkpoint writes a `train_state.json` sidecar (`step`, `wandb_run_id`,
`latest`) next to `step_<N>.pt`; resume reads it to continue the loop from `N` and
re-attach the wandb run. `step_<N>.pt` itself is unchanged (raw `agent.state_dict()`,
which already includes optimizer state), so all eval/analysis `agent.load()` paths
keep working.

### Grid launcher

`scripts/run_grid_sweep.sh` with `RESUME=1` appends `resume=true` to every job, so
re-running the same launcher invocation continues jobs that have a checkpoint and
starts the rest fresh.

## JAX / GCIQL (`run_gciql.py`)

Already supported via the vendored OGBench restore path:
`++restore_path=<run_dir> ++restore_epoch=<N>` restores params+opt state from
`params_<N>.pkl`; `GCIQL_STEP_OFFSET`/`GCIQL_CSV_APPEND` (set automatically when
`restore_epoch` is given) shift saved epochs and continue the CSV/wandb axes. Pin
`++exp_name=<name>` to keep the same output dir. (Use `++` because the base
`cube_single_*` configs do not predefine these keys.)

## Smoke tests

- PyTorch: `scripts/smoke_resume.sh` (state); `DOMAIN=visual_cube_single
  AGENT=fb_flowbc DATA_PATH=/dev/shm/factored-fb/datasets_visual
  bash scripts/smoke_resume.sh` (pixel).
- GCIQL: `scripts/smoke_resume_gciql.sh`.

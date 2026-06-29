# wandb_mode_shim

This directory is **not a package** — it is a `sitecustomize.py` deliberately
placed on `PYTHONPATH` by `run_gciql.py` for the OGBench child process only.
It is inert for any unrelated Python process and never edits `third_party/`
(which must stay byte-for-byte upstream per its `PROVENANCE.md`).

## Two independent, opt-in behaviors

### 1. Force wandb mode — `GCIQL_WANDB_FORCE_MODE=offline|disabled`

OGBench's vendored `setup_wandb` hardcodes `wandb.init(mode='online')`, and
wandb (>= 0.27) lets that explicit kwarg override `WANDB_MODE` /
`WANDB_DISABLED`. Since `third_party/ogbench/` must not be edited,
`sitecustomize.py` monkeypatches `wandb.init` at interpreter startup to force
the requested mode regardless of the caller's kwarg.

Set by `run_gciql.py` only when `wandb_mode=offline` or `wandb_mode=disabled`
is in the Hydra config. No-op for all other modes.

### 2. Mirror eval metrics to FB wandb keys — `GCIQL_FB_ENV=<env_name>`

GCIQL (OGBench) logs evaluation results under keys like:

```
evaluation/overall_success
evaluation/task1_horizontal_success
evaluation/task5_diagonal2_success
```

FB logs the same concept under:

```
eval/reward/eval/success                          (aggregate)
eval/reward/<env_base>-singletask-task<N>-v0/success  (per task)
```

When `GCIQL_FB_ENV` is set (e.g. `cube-single-play-v0`), the shim wraps
`wandb.log` to **add** FB-style keys alongside OGBench's originals so FB and
GCIQL runs overlay on the same wandb panels. The original keys are kept
unchanged. `env_base` is the env name with any trailing `-v0` stripped.

Set unconditionally by `run_gciql.py` (the shim is always on the child
PYTHONPATH; this flag enables the mirroring behavior for real comparison runs).

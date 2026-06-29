.PHONY: install install-pip smoke train-ant-medium train-ant-large train-cube sweep-ant-medium verify-ortho-fix verify-ortho-broken verify-lr-b-fix verify-lr-b-broken verify-both data clean help install-jax-shm install-jax test-gciql test-wandb-shim run-gciql run-gciql-state run-gciql-state-sweep smoke-gciql

# ── Installation ──────────────────────────────────────────────────────────── #

## Install dependencies with uv (recommended; uses .venv/)
install:
	uv pip install -r requirements.txt

## Install dependencies with plain pip
install-pip:
	python -m pip install -r requirements.txt

# ── OGBench dataset ──────────────────────────────────────────────────────── #

OUTPUT ?= datasets

## Download + extract all state-based OGBench domains into ./datasets/ (or OUTPUT=<path>)
data:
	python scripts/data/extract_ogbench.py --output_folder $(OUTPUT)

# ── Pipeline check ────────────────────────────────────────────────────────── #

## Synthetic-data pipeline smoke (~20s, NO OGBench dataset needed)
smoke:
	python scripts/dev/smoke.py

# ── Training (requires extracted OGBench data under ./datasets/) ─────────── #

## Antmaze medium (standard FB)
train-ant-medium:
	python train.py domain=antmaze_medium

## Antmaze large (standard FB)
train-ant-large:
	python train.py domain=antmaze_large

## Cube single (FlowBC)
train-cube:
	python train.py domain=cube_single

## Antmaze medium with seeds 1-3
sweep-ant-medium:
	python train.py --multirun seed=1,2,3 domain=antmaze_medium

# ── FB collapse diagnostic (50k steps, asserts metrics stay bounded) ─────── #

## Verify ortho_coef=100 prevents FB collapse on antmaze (~10 min on GPU)
verify-ortho-fix:
	python scripts/dev/verify_ortho_fix.py

## Reproduce the collapse with the broken config (ortho_coef=1.0, no clip)
verify-ortho-broken:
	python scripts/dev/verify_ortho_fix.py --ortho-coef 1.0 --clip-grad-norm 0

## Verify lr_b=1e-5 prevents FB collapse on antmaze (~10 min on GPU)
verify-lr-b-fix:
	python scripts/dev/verify_lr_b_fix.py

## Reproduce the collapse with the broken lr_b=1e-4 (other knobs identical)
verify-lr-b-broken:
	python scripts/dev/verify_lr_b_fix.py --lr-b 1.0e-4

## Run both verifications back-to-back; ortho first, then lr_b
verify-both: verify-ortho-fix verify-lr-b-fix

# ── Housekeeping ──────────────────────────────────────────────────────────── #

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	rm -rf outputs multirun

help:
	@echo ""
	@echo "Factored-FB — available targets"
	@echo "────────────────────────────────────────────────────────"
	@echo "  install          Install deps with uv into .venv/"
	@echo "  install-pip      Install deps with plain pip"
	@echo "  data             Download + extract OGBench to ./datasets/"
	@echo ""
	@echo "  smoke            Synthetic pipeline check (no data needed)"
	@echo "  train-ant-medium Antmaze medium run"
	@echo "  train-ant-large  Antmaze large run"
	@echo "  train-cube       Cube single run (FlowBC)"
	@echo "  sweep-ant-medium Antmaze medium, seeds 1-3"
	@echo ""
	@echo "  verify-ortho-fix    50k-step antmaze run with ortho_coef=100; asserts no collapse"
	@echo "  verify-ortho-broken Same but with ortho_coef=1.0 to reproduce collapse"
	@echo "  verify-lr-b-fix     50k-step antmaze run with lr_b=1e-5;     asserts no collapse"
	@echo "  verify-lr-b-broken  Same but with lr_b=1e-4 to reproduce collapse"
	@echo "  verify-both         Run ortho-fix then lr-b-fix (~20 min on GPU)"
	@echo ""
	@echo "  clean            Remove pycache, outputs"
	@echo ""
	@echo "  install-jax-shm  install isolated JAX env in tmpfs /dev/shm (no sudo, default)"
	@echo "  install-jax      Install isolated JAX env on the NVMe (GCIQL track)"
	@echo "  test-gciql       Run the run_gciql argv/isolation test"
	@echo "  run-gciql        Launch GCIQL (vendored OGBench, JAX) state config"
	@echo "  run-gciql-state  Multi-seed GCIQL state run (1M steps, online wandb)"
	@echo "  smoke-gciql      End-to-end GCIQL smoke (state + visual)"
	@echo "────────────────────────────────────────────────────────"

# ── GCIQL / JAX track ─────────────────────────────────────────────────────── #

## Install isolated JAX venv in tmpfs at /dev/shm (no sudo, default)
install-jax-shm:
	uv venv --python 3.11 /dev/shm/.venv-jax
	UV_CACHE_DIR=/dev/shm/uv-cache uv pip install --python /dev/shm/.venv-jax/bin/python -r requirements-jax.txt
	@mkdir -p /dev/shm/ogbench_data /dev/shm/gciql_outputs
	@test -e ~/.ogbench || ln -s /dev/shm/ogbench_data ~/.ogbench
	@echo "[install-jax-shm] done (tmpfs; ephemeral on reboot). Verify: /dev/shm/.venv-jax/bin/python -c 'import jax; print(jax.devices())'"

## Install isolated JAX venv on NVMe at /mnt/scratch1 (run once per instance)
install-jax:
	@test -d /mnt/scratch1 || { echo "Mount the NVMe at /mnt/scratch1 first (see spec Prerequisites)"; exit 1; }
	uv venv --python 3.11 /mnt/scratch1/.venv-jax
	uv pip install --python /mnt/scratch1/.venv-jax/bin/python -r requirements-jax.txt
	@mkdir -p /mnt/scratch1/ogbench_data /mnt/scratch1/gciql_outputs
	@test -e ~/.ogbench || ln -s /mnt/scratch1/ogbench_data ~/.ogbench
	@echo "[install-jax] done. Verify: /mnt/scratch1/.venv-jax/bin/python -c 'import jax; print(jax.devices())'"

## Argv / isolation unit test for the GCIQL launcher
test-gciql:
	python scripts/dev/test_run_gciql.py

## Unit test for the wandb shim video GIF + re-key behavior
test-wandb-shim:
	/dev/shm/.venv-jax/bin/python scripts/dev/test_wandb_shim_video.py

## Launch GCIQL with cube_single_state config
run-gciql:
	python run_gciql.py --config-name cube_single_state

## Full multi-seed GCIQL state run (cube-single-play-v0, 1M steps, online wandb)
run-gciql-state:
	bash scripts/run_gciql_state.sh

## Parallel 10-seed GCIQL state sweep across GPUs (mirrors run_sweep.sh)
run-gciql-state-sweep:
	bash scripts/run_gciql_state_sweep.sh

## End-to-end smoke test for GCIQL (state + visual)
smoke-gciql:
	bash scripts/dev/smoke_test_gciql.sh

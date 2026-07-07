---
marp: true
title: PSMFlows — Session Handoff
theme: default
paginate: true
---

<!-- _class: lead -->

# PSMFlows — Session Handoff

Integrating the Factored-FB actors into JAX PSM,
jitting the update, and benchmarking vs the FB+Flow reference.

Branch: `feat/psm-integration` · Machine: `midi-01` (UT CS)
Date: 2026-07-06

---

## What this project is

- **Goal:** scale Proto Successor Measures (PSMs) to constrained offline datasets by
  indexing policies with behavior-flow noise `u₀` (see `PAPER/main.tex`).
- Built on the **Flow Q-Learning (FQL)** JAX/Hydra codebase.
- The JAX **PSM agent** lives in `agents/psm.py` (port of a PyTorch reference,
  verified byte-equivalent via fixtures in `tests/`).

---

## Done this session (1/2) — actor integration

Ported both actors from **`/u/amsks/git/Factored-FB`** (PyTorch FB / `td_jepa`) into JAX PSM.
Selectable via `agent.actor.type`:

- **`ddpgbc`** — TD3 mean + truncated exploration + optional BC term (`bc_coeff`).
  `bc_coeff=0` ⇒ **byte-identical** to the old PSM actor (equiv tests still pass).
- **`flow`** — FQL-style: BC flow-matching velocity field `v(s,xₜ,t)` + one-step,
  z-conditioned distilled actor. Q from PSM's `sf_psi` (`Q=(sf_psi·z)`).

Cube defaults match the reference: `bc_coeff=3`, `flow_steps=10`, `lr_actor_vf=3e-4`.

---

## Done this session (2/2) — jit + perf

- PSM `update()` was **un-jitted** → **1.15 s/step** (GPU 98% idle, dispatch-bound).
- **Jitted** `update` + `sample_actions` → **~9 ms/step (≈126× faster)**.
  - `proto` table → traced pytree leaf; `nets`/`txs` → hashable `_HashableDict` static aux.
  - Equiv tests untouched: they call the **un-jitted** `apply_update`/`compute_static`.
- **13/13 tests pass** (`tests/test_psm_*`).
- wandb wired to **`amsks/PSMFLows`** (config `wandb_project`/`wandb_entity`);
  entity forced to `amsks` to avoid the `al-laq-fb` team default.

500k run now ≈ **76 min solo** (≈ hours under GPU sharing).

---

## Environment gotchas (READ before running)

- **`midi-01` = UT CS HTCondor box, NO Slurm/`sbatch`.** Condor is broken on this host
  (`Can't read config .../etc/local/midi-01`). Run directly (nohup/tmux + `CUDA_VISIBLE_DEVICES`).
- **Home NFS quota (`/u/amsks`) is tiny (~6 GB).** A `jax[cuda12]` install blows it.
  Install with `PIP_NO_CACHE_DIR=1 TMPDIR=/var/local pip install --no-cache-dir ...`.
- **All bulk data → node-local `/var/local/amsks/`** (off quota, ~21 GB free):
  `~/.ogbench` symlinked there; `WANDB_DIR`, `save_dir`, logs all under `/var/local/amsks`.
- GPUs shared (hss963, idutta) — check `nvidia-smi` first. We use **GPU 1**,
  `XLA_PYTHON_CLIENT_PREALLOCATE=false`, `MEM_FRACTION=0.30`.

---

## Currently running — the comparison experiment

**wandb group `psm_cube_match_ortho1000_20260706_205705`** on GPU 1:

- Our **PSM**, matched config **`ortho_coef=1000, z_dim=50`**, seeds **5, 10, 7**,
  actors **flow + ddpgbc**, 500k steps, eval+ckpt @ 100k.
- Purpose: apples-to-apples vs the reference FB+Flow at the **same seeds / config / task**.
  Only intended difference = **representation (PSM vs FB)**.

Flow run URLs: `zjglyxq7` (s5), `ikf5dwjq` (s10), `jxxuqorl` (s7) — under `amsks/PSMFLows`.

A background monitor auto-emits the 100k table once all 3 flow seeds eval.

---

## Reference targets (FB+Flow, `amsks/factored-fb`)

`cube_single__sN__ortho1000__lrb1e-4` · **task2** (= our `singletask-v0`) success:

| seed | ref @100k | ref peak |
|------|----------:|---------:|
| 5    | 0.10 | 1.00 |
| 10   | 0.10 | 1.00 |
| 7    | 0.20 | 0.90 |

Full numbers (5-task mean + task2, all seeds) in **`docs/reference_benchmarks.md`**.
Reference seeds are **late bloomers** — low @100k, peak 0.9–1.0 later.

---

## How to run

```bash
# From repo root. GPU-1, online wandb (needs `wandb login` first — entity amsks).
GROUP=my_group SEEDS="5 10 7" ACTORS="flow ddpgbc" \
EXTRA="agent.ortho_coef=1000 agent.z_dim=50" EVAL_INT=100000 SAVE_INT=100000 \
bash scripts/launch_psm_cube.sh 1 500000 online

# single run
CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
python main.py agent=psm agent.actor.type=flow \
  agent.ortho_coef=1000 agent.z_dim=50 \
  env_name=cube-single-play-singletask-v0 offline_steps=500000 \
  save_dir=/var/local/amsks/exp run_group=... seed=5
```

Do **not** `pkill -f "main.py agent=psm"` inline — it matches the launching shell and
self-kills. Kill by PID or a narrower pattern.

---

## Next steps / open items

1. **Collect the 100k comparison** (monitor pending) — our PSM+Flow vs ref (0.10/0.10/0.20).
2. **Peak / 500k comparison** — the fairer picture (ref seeds peak at 0.9–1.0 later).
3. Consider a **5-task eval** in `main.py` to match the reference's 5-task-mean metric
   (currently we eval only task2 via `singletask-v0`).
4. Original seeds-0,1,2 sweep (ortho=1.0, z_dim=128) was **killed/superseded** — relaunch
   if the un-matched PSM-default numbers are still wanted.
5. Nothing committed yet — changes are in the working tree on `feat/psm-integration`.

---

<!-- _class: lead -->

## Pointers

- Agent: `agents/psm.py` · Networks: `utils/psm_networks.py`, `utils/networks.py`
- Config: `configs/agent/psm.yaml`, `configs/config.yaml`
- Launcher: `scripts/launch_psm_cube.sh` · Tests: `tests/test_psm_*.py`
- Reference numbers: `docs/reference_benchmarks.md`
- Reference PyTorch impl: `/u/amsks/git/Factored-FB` (`agents/fb`, `agents/fb/flow_bc`)
- Memory: `cluster-and-quota`, `psm-run-workflow`, `reference-fb-flow-benchmarks`

# 2026-09-07 — more work per H100: what packing buys (1.43x on eval, nothing on training)

**Problem.** 36 H100s, all ours, all busy, ~20 jobs queued. The queue is two job shapes and
neither looks like it uses the device it holds:

* **500-episode evals** (`tools/eval_checkpoint.py`) — one MuJoCo rollout at a time, one
  batch-1 flow decode per env step. antmaze/pointmaze ~95 min, cube ~20 min, each on a
  whole H100. 3.2 GB of 80. 0.42 of one core out of 8 held.
* **Stage-C training** (`main.py agent=psmflow`) — small MLPs at batch 256 plus a 10-step
  flow unroll. 17 ms/step on cube, 5.2 GB of 80, 0.53 of a core.

Both look starved. Neither is. **`utilization.gpu` was already 66-76% for a single job of
either kind**, and that — not memory, not cores — is what caps everything below.

**Result.**

| lever | measured | verdict |
|---|---|---|
| N=8 eval workers, antmaze 500 ep | 92.3 min → **64.5 min**, 1.43x | shipped, default N=**4** |
| K=3 packed training seeds, cube | 17.0 → **58.0 ms/step**, 3.41x per seed | rejected (0.88x aggregate) |
| K=3 packed + MPS | 17.0 → **51.7 ms/step**, 3.04x per seed | rejected (0.99x aggregate) |

The eval change is a real 1.43x on the slowest job class in the queue and it is in. Packed
training is a loss and stays unused; the launcher is checked in as a measured negative.

---

## 1. Measurements

### 1.1 The cluster

`scontrol show partition kisski-inference`: 9 nodes, `cpu=1872, mem=4500000M,
gres/gpu:h100=36` → **208 cores, ~500 GB and 4 H100 PCIe (80 GB) per node**; one GPU's
fair share is 52 cores and 125 GB. `MaxCPUsPerNode=UNLIMITED`,
`DefMemPerNode=UNLIMITED`, `JobDefaults=(null)` (no `DefCpuPerGPU`). Both sbatch headers
asked for `--cpus-per-task=8 --mem=64G`.

Compute mode is `Default` (not `EXCLUSIVE_PROCESS`), so processes may share a device — but
without MPS the driver **time-slices between CUDA contexts** instead of running their
kernels concurrently.

### 1.2 A running training job — 2491814 `psm-pm-sd0` (pointmaze Stage C, gpu002)

```
nvidia-smi (srun --overlap)   utilization.gpu 66-68 %   memory.used 5225 / 81559 MiB (6.4%)
sstat -j 2491814              AveCPU 01:05:04 over RunTime 02:02:12  → 0.53 of one core
                              MaxRSS 5.6 GB   NTasks 1
scontrol show job             NumCPUs=8  CPUs/Task=8  mem=64G
```

### 1.3 A running eval job — 2491902 `e500_pm50k_sd0` (pointmaze eval500, gpu005)

```
nvidia-smi        utilization.gpu 73-76 %   memory.used 3173 / 81559 MiB (3.9%)
nvidia-smi pmon   sm 73-75 %, mem 15-16 %, fb 3153 MB   (one python process)
ps                41.7 % of ONE core, RSS 2.1 GB        (of 8 cores / 64 GB allocated)
```

**`utilization.gpu` is time-with-a-kernel-resident, not occupancy** — `mem` (memory
controller busy) is only 15%, and these are microsecond batch-1 kernels touching a handful
of the H100's 114 SMs. But for *packing* it is exactly the right number: a second process
can only use the 25% of wall time the first leaves idle, because their kernels cannot
overlap. Two such jobs already oversubscribe the device. That is the whole story of §2.5
and §3.

### 1.4 Single-process eval wall times (`sacct`, 8 CPUs each)

| job | env | 500 episodes | per episode |
|---|---|---|---|
| 2491641 `e250k_strict_ant_sd1` | antmaze | **1:32:18** | 11.0 s |
| 2491630 `e500k_strict_antmaze_sd1` | antmaze | 1:32:54 | 11.0 s |
| 2491629 `e500k_strict_antmaze_sd0` | antmaze | 1:36:44 | — |
| 2491625 `e500k_strict_cube_sd0` | cube | 0:21:39 | 2.6 s |

2491641 is the exact run this work is validated against.

---

## 2. Parallel eval episodes (`eval_workers` / `EVAL_WORKERS`)

### 2.1 What changed

* `tools/eval_checkpoint.py` — the part of `main` that builds an env, restores the agent
  and calls `evaluate` moved verbatim into `_evaluate_shard(payload)`. `main` splits
  `eval_episodes` into contiguous shards, runs them (in-process at N=1, in a **spawn**
  `Pool` at N>1) and folds the results with `aggregate_worker_results`. Four pure helpers
  (`resolve_num_workers`, `split_episodes`, `worker_seed_plan`, `aggregate_worker_results`)
  hold all the arithmetic and are unit-tested without jax.
* `configs/config.yaml` — `eval_workers: 1`.
* `scripts/eval500.sh` — `EVAL_WORKERS`, **defaulting to 1** (see §2.6).
* `scripts/slurm/eval500.sbatch` — exports `EVAL_WORKERS=4`; header **unchanged** at
  `--cpus-per-task=8 --mem=64G` (§2.5 says more workers are not worth more cores).

### 2.2 Invariants kept

* **`utils.xla_guard` before jax in every worker.** Workers are `spawn`ed, so the child
  re-imports `tools/eval_checkpoint.py` from the top, and `import utils.xla_guard` is that
  module's first import, ahead of `agents`/jax. Fork was never an option: a forked child
  inherits a half-initialised CUDA context and MuJoCo's EGL display.
* **Device memory.** The parent never touches a jax device in multi-worker mode; it sets
  `XLA_PYTHON_CLIENT_PREALLOCATE=false` and divides `XLA_PYTHON_CLIENT_MEM_FRACTION` by N
  *before* `Pool`, and children inherit it through `os.environ`. Recorded in the report as
  `xla_mem_fraction_per_worker` (0.1125 in the N=8 run). `MUJOCO_GL=egl` and per-worker
  `OMP/MKL/OPENBLAS_NUM_THREADS` are set the same way.
* **Report schema is a superset.** `success`, `num_success`, `num_episodes`, `wilson95`,
  `per_episode_success` are unchanged, plus `num_workers`, `worker_episodes`,
  `worker_seeds`, `worker_seconds`, `eval_seed_scheme`, `xla_mem_fraction_per_worker`,
  `wall_seconds`. `tools/make_tables.py` reads only the old keys and is untouched.

### 2.3 Seeding — N>1 is a *different* 500 episodes, not a reordering

`utils.evaluation.evaluate` seeds the env's episode-init RNG **once** per call and lets
per-episode resets advance it, and draws action keys from **one sequential** PRNG stream.
Episode *i* is a function of episodes 0..*i*-1, so no worker that did not run them can
reproduce it. A split can preserve the distribution, never the individual episodes.

The scheme, recorded in the report's `eval_seed_scheme`:

```
worker w of N runs shard w with   evaluate(seed = cfg.seed * N + w)
every worker infers z from        np.random.seed(cfg.seed)      # the SAME task vector
```

* **N=1 → `seed`**, so the single-process path is byte-identical to the pre-2026-09-07
  tool and every recorded eval500 JSON stays reproducible.
* seeds are disjoint across `cfg.seed` at fixed N, so an eval at seed 0 and one at seed 1
  never share an episode-init stream.

**For reporting:** an N>1 number agrees with an old N=1 number *within the Wilson
interval*, not exactly. Use `EVAL_WORKERS=1` when a number must be byte-reproduced.

### 2.4 Validation

**N=1 parity, CPU** — `agent=psmflow`, cube sd0 @500k, 4 episodes, `seed=0`, jax on CPU,
this tool against `git show HEAD:tools/eval_checkpoint.py`:

```
orig: success 0.25  num_success 1/4  wilson95 [0.0456, 0.6994]  per_episode [0,0,0,1]
w1  : success 0.25  num_success 1/4  wilson95 [0.0456, 0.6994]  per_episode [0,0,0,1]
IDENTICAL: True
```

N=2 and N=4 were run on the same checkpoint to exercise the spawn path end to end
(`worker_episodes` [2,2] and [1,1,1,1], `worker_seeds` [0,1] and [0,1,2,3]).

**GPU, the real thing** — job 2491929, affine-strict antmaze sd1 @250k, 500 episodes,
`EVAL_WORKERS=8`, against 2491641 which produced
`eval500_affine250k_strict_antmaze_sd1.json` at N=1:

| | N=1 (2491641) | N=8 (2491929) |
|---|---|---|
| wall | 92.3 min | **64.5 min** (3868 s) |
| success | 89/500 = 0.178 | 93/500 = **0.186** |
| Wilson 95% | [0.147, 0.214] | [0.154, 0.223] |
| device memory | 3.2 GB | 25.9 GB (8 × 3.15 GB) |
| host MaxRSS | ~2.6 GB | 16.8 GB |
| load balance | — | worker_seconds 3682-3821 s (±1.9%) |

**1.43x**, and the two intervals overlap over almost their whole width — exactly the
agreement §2.3 predicts for a re-sample of the same distribution.

**Unit tests** — `tests/test_eval_workers.py`, 22 cases over the pure helpers: the shard
list is a partition of `eval_episodes` balanced to within one episode; the remainder goes
to the low indices deterministically; `worker_seed_plan(s, 1) == [s]`; seed plans do not
collide across `cfg.seed`; aggregation is order-independent, thresholds at 0.5 rather than
summing, rejects a missing worker or a short shard, and falls back to the
**episode-weighted** mean when the env exposes no per-step `success`; and worker-count
resolution (CLI > env > 1, clamped to the episode count, garbage falls back to 1).

### 2.5 Why 4 and not 8

1.43x from **eight** processes is a 5.6x-per-worker slowdown, and it is not CPU (each
worker sat at ~11-21% of a core, and 16 cores were allocated) or memory (25.9 GB of 80).
It is §1.3: `nvidia-smi` read **100% utilisation** for the whole 8-worker run, against
73-76% for one rollout alone. The contexts time-slice.

The saturation point is visible from two independent measurements:

| concurrent rollouts on one H100 | per-process slowdown | aggregate |
|---|---|---|
| 3 (the packed job's `i == 1` eval, cube) | 5.13 / 2.57 = 2.0x | 1.50x |
| 8 (this eval, antmaze) | 5.6x | 1.43x |

Three processes already reach the ceiling; five more add nothing. Hence the default of
**4** — all of the win, half the cores, and the sbatch header stays where it was.

### 2.6 Existing queued jobs are unaffected — deliberately

SLURM snapshots the batch script at **submit** time, but `srun bash scripts/eval500.sh`
reads that file off disk at **launch** time. A job queued before this change therefore
runs its old sbatch body against the new `eval500.sh`. A default of 8 in `eval500.sh`
would silently re-shape other people's queued jobs onto 8 workers with the 8 cores their
old header asked for — which is what happened for a few minutes today (jobs 2491907-2491916
printed `parallel eval: 8 workers`) before the default was moved back to 1.

| where | value | effect |
|---|---|---|
| `scripts/eval500.sh` | `EVAL_WORKERS=1` | anything already queued behaves exactly as before |
| `scripts/slurm/eval500.sbatch` | exports `EVAL_WORKERS=4` | every NEW submission gets the parallel path |
| `configs/config.yaml` | `eval_workers: 1` | `main.py`'s in-loop eval ignores it entirely |

(Those 2491907-2491916 jobs died on `KeyError: 'entropy'`, which is the in-flight
`agents/psmflow.py` / `configs/agent/psmflow.yaml` drift over `actor.entropy`, not this
change — the same failure reproduces on `git show HEAD:tools/eval_checkpoint.py`.)

---

## 3. Packed training seeds — measured, rejected, kept as a record

`scripts/slurm/train_psmflow_packed.sbatch` runs K seeds of one config in one allocation as
background processes with `wait`, each with `XLA_PYTHON_CLIENT_MEM_FRACTION=0.90/K`, its
own log under `$PSM_DATA/logs/`, and a roll-up of the seeds' exit codes into the job's.
`USE_MPS=1` additionally starts a **job-private** MPS daemon (its own
`CUDA_MPS_PIPE_DIRECTORY`, torn down in an EXIT trap) so kernels from the K processes go
through one context and can actually overlap.

Smoke: cube, 2000 steps, `time/epoch_time` from `train.csv`, first 200-step bucket dropped
(it carries JIT and the `i == 1` eval).

| job | placement | s/step per seed | vs. K=1 | aggregate |
|---|---|---|---|---|
| 2491928 `gpupack_k1` | 1 seed, 1 H100 | **0.01699** | 1.00x | 1.00x |
| 2491927 `gpupack_k3` | 3 seeds, 1 H100 | **0.0580** (0.0584/0.0582/0.0575) | **3.41x** | **0.88x** |
| 2491950 `gpupack_k3mps` | 3 seeds + MPS | **0.0517** (0.0516/0.0518/0.0516) | **3.04x** | **0.99x** |

The bar was "a win if the per-seed slowdown is < 1.5x". It is 3.41x, or 3.04x with MPS —
three packed seeds finish in 3.4x the wall clock of one and deliver 12% *less* aggregate
throughput than three separate GPUs (0.99x, i.e. break-even, with MPS). `nvidia-smi`
during the packed run: **100% utilisation, 8829 MiB**, `pmon` sm% summing to ~95 across the
three processes, against 66-68% for one seed alone. The device was already the bottleneck
at K=1; the 5.2 GB and 0.53 cores were never the constraint.

MPS *does* work — `pmon` showed the three pythons as `M+C` clients behind one
`nvidia-cuda-mps-server` at 88-90% sm — it just cannot manufacture headroom that is not
there. It recovers about a third of the packing loss (3.41x → 3.04x) and no more.

No real runs were launched with it. It is checked in so the next person does not re-run
this experiment.

---

## 4. Recommended defaults

1. **Evals: `EVAL_WORKERS=4`** — already `scripts/slurm/eval500.sbatch`'s export, with the
   header left at `--cpus-per-task=8 --mem=64G`. Worth ~1.4x on antmaze/pointmaze evals
   (95 → ~68 min), which is the slowest thing in the queue.
2. **Anything that must byte-reproduce an existing number: `EVAL_WORKERS=1`.** The
   per-episode sequence is not preserved across a split (§2.3).
3. **Training stays one seed per GPU** (`scripts/slurm/train_psmflow.sbatch`). Do not pack.
4. **Do not raise `--cpus-per-task` on either job type** to chase this. Neither is
   CPU-bound (0.42 and 0.53 of a core), and the GPU ceiling is reached at 3 contexts.
5. **The next real lever is a batched evaluator, not more processes.** Stepping M MuJoCo
   envs in one process and decoding one batch-M action per step replaces M batch-1 kernels
   with one batch-M kernel — it needs no extra CUDA contexts, so it is not subject to the
   §2.5 ceiling, and the H100 is at 15% memory-controller utilisation on a batch of 1. That
   is a change to `utils/evaluation.py` (vectorised env, batched `sample_actions`,
   per-env done bookkeeping) and was out of scope here. Expected payoff is much larger than
   1.43x; this is the recommended follow-up.

## 5. Files touched

```
tools/eval_checkpoint.py                    parallel-episode split, 4 pure helpers, report fields
configs/config.yaml                         eval_workers: 1
scripts/eval500.sh                          EVAL_WORKERS (default 1)
scripts/slurm/eval500.sbatch                exports EVAL_WORKERS=4; header unchanged
scripts/slurm/train_psmflow_packed.sbatch   NEW: K seeds in one allocation, USE_MPS knob
tests/test_eval_workers.py                  NEW: 22 cases over the pure helpers
docs/design/2026-09-07-gpu-packing.md       this file
```

Artifacts: `$PSM_DATA/logs/gpupack/` (job logs and
`eval500_affine250k_strict_antmaze_sd1_w8.json`),
`$PSM_DATA/exp/PSMFLows/gpupack_smoke_k1|k3|k3mps/` (the packing `train.csv`s).

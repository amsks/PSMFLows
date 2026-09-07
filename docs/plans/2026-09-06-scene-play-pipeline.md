# scene-play end to end: Stage A -> HPO -> Stage B -> Stage C (affine paper-strict)

Date opened: 2026-09-06. Branch `feat/inversion-integration`. Machine: KISSKI (SLURM, H100,
partition `kisski-inference`, account `general`, max wall **3-00:00:00**).

Executes recommendation 1 of `docs/plans/2026-09-06-env-roadmap.md`: add `scene-play` as the
second working manipulation environment beside cube, so "OGBench" in the paper is represented
by more than one environment where the method works.

`PSM_DATA=/mnt/home/amohan/psm-data`, `OGBENCH_DATASET_DIR=/mnt/home/amohan/.ogbench/data`.

## 0. The environment, verified on disk

`scene-play-singletask-v0` resolves (ogbench drops the dataset-type token) to gym id
`scene-singletask-v0` + dataset file `scene-play-v0.npz`. Both npz are present and load:

| | value |
|---|---|
| rows (train / val) | 1,000,000 / 100,000 |
| `d_obs` | 40 (cube 28, antmaze 29) |
| `d_a` | **5** — identical to cube, the only regime where the method works |
| action box | `[-1, 1]`, `max abs = 0.99999` |
| mean action L2 | **0.978** (cube 0.87, antmaze 1.99) |
| default task | task 2 (`ogbench/manipspace/envs/scene_env.py`) |

`d_a=5` and an action norm near cube's are exactly the two surviving candidate explanations
of the cube/antmaze split (`COMPENDIUM:473-483`), so scene-play sits on the cube side of both.
That is the reason to expect it to work, and the reason a failure would be informative.

## Pre-registered expectations

Stated before any number arrives.

### Stage A — behaviour flow

- Trains in **1.5–2 h** at `flow_steps=100`, 500k steps (pointmaze measured 1h27m). The
  4 h wall on `pretrain_behavior_flow.sbatch` is ~2x headroom.
- **BC control**, 500 episodes of the frozen flow acting alone: scene-play is a
  longer-horizon multi-object task than cube, so the BC floor is expected **at or below
  cube's 0.072** — point estimate **0.03, 90% interval [0.00, 0.15]**. A BC control above
  ~0.3 would mean the task is nearly solved by the data alone and the arm has little room;
  a BC control of exactly 0.0 is fine and is what pointmaze reads.

### Inversion HPO

- `hpo_decode_budget=0.22` reused verbatim from cube (the budget is `0.1*sqrt(d_a)` and
  `d_a=5` is identical). 50 SMAC trials, ~65 s/trial at `d_a=5` => **~1 h**.
- Expect the optimum to land **near cube's** `alpha≈20.6, prior_scale≈0.69, n_steps≈12`,
  since the objective, the budget, and `d_a` all match: **alpha in [10, 40], prior_scale in
  [0.5, 1.0], n_steps in [8, 16]**. Coverage@k=64 at the optimum: **0.85–0.98** (cube reads
  0.945 at alpha=20 under the corrected target). If coverage lands below ~0.3 — antmaze
  territory — that is the pre-registered *failure* signal and Stage B should not be spent.

### Stage B — inversion of 1M rows

- `tools/precompute_preimages.py` has **no sharding and no resume**: `preimage_limit` is a
  prefix slice and `preimage_sample` a random batch, neither an index range, and nothing is
  written until the whole buffer is inverted. Per the brief, sharding is NOT added. One
  single-GPU job it is.
- Cost is ~linear in `num_samples * n_steps * rows`. Cube at `n_steps=12, N=200` took ~23 h.
  scene-play has the same row count and `d_a` but `d_obs` 40 vs 28, so the state MLP is
  marginally wider: expect **20–30 h**. Wall booked at **36 h** (under the partition's 72 h
  cap, so no chaining is needed).
- Expect `num_invalid_preimages` **< 2000 of 1M** (cube's arm is clean; antmaze's 881/1M is
  the pathological one and it still passed).

### Recovery gates (`scripts/run_recovery_tests.sh scene`)

- Decode recovery, point arm: **< 1e-3** action L2 (the floor; device-limited).
- Decode recovery, mixture arm: **at or under the 0.22 decode budget** the HPO enforced.
- Both well below the prior arm, which should read ~0.8–1.0 (the mean action norm).
- Dynamics recovery `next_obs_vs_replay` for the point arm: **small against `data_step`**;
  scene is a manipulation env like cube, so expect a `replay_vs_dataset` floor of the same
  order as cube's 0.022 rather than pointmaze's 1e-6 — read `next_obs_vs_replay`, not
  `next_obs_vs_dataset`.

### Stage C — affine paper-strict, 3 seeds, 500k

Repo-default `agent=psmflow` (affine psi, `policy_index=latent`, `train_actor=false`,
`acting=gpi`), `agent.use_point_preimage=true`.

- Runtime: cube-class, **~2.7 h/seed**; 12 h wall is ample.
- **The number**: cube's affine-strict late mean is 0.415 against BC 0.072. scene-play is the
  same action regime but a harder, longer task with more objects, so the pre-registered effect
  is *smaller*: **late-checkpoint mean success 0.20, 90% interval [0.05, 0.45]**, and the
  falsifiable claim is **scene-play beats its own BC control by at least 3x** (the "working
  regime" prediction). Anything inside [BC, 1.5x BC] is a null and should be reported as one.
- Expect the same **non-convergence** cube shows: ±0.4 swings between 50k checkpoints on some
  seeds (HANDOFF 09-06). This is why the eval is a full ladder at every 50k for all seeds and
  why the headline is a late-checkpoint *mean*, never a peak.
- Expected failure mode worth naming in advance: if scene-play reads null while its coverage
  and ESS match cube's, then `d_a` is exonerated as the explanation of the cube/antmaze split
  and the cube result is about cube. That is a publishable negative, not a wasted pipeline.

## Code touched (shell only; no Python)

Nine `case "$ENVKEY"` blocks got a `scene)` arm, per the roadmap's inventory:
`scripts/slurm/{precompute_preimages,train_psmflow,eval500,eval_gpi_ablation,diag_flow_fit,diag_gpi_selection}.sbatch`,
`scripts/{eval500.sh,run_recovery_tests.sh,fetch_preimages_hf.sh}`.

Two further shell changes, both additive:

- `scripts/slurm/precompute_preimages.sbatch` now uses a **local** `flow/<name>/params_<epoch>.pkl`
  when one exists instead of pulling from HF. A flow trained on this cluster is not in the HF
  repo until step 3/3 pushes it, and the unconditional pull would have failed before a 20 h
  inversion started.
- `scripts/slurm/pretrain_behavior_flow.sbatch` is new: Stage A for a scheduler.
  `scripts/pretrain_behavior_flow.sh` hardcodes midi-01 paths and `nohup`. It also installs
  the checkpoint into `$PSM_DATA/flow/<name>/`, the flat layout every downstream script
  addresses.

## Log

- 2026-09-06 — env verified, nine case blocks patched, Stage-A sbatch written, plan opened.
- 2026-09-07 — **Stage A done** (job 2491918, smoke 2491917). **35m11s**, not the 1.5-2 h
  predicted: the pointmaze 1h27m figure is from another machine, and an H100 at
  `flow_steps=100` is roughly 2.5x faster. Flow installed at
  `$PSM_DATA/flow/scene-play/params_500000.pkl` (56 MB), `flags.json` byte-identical to the
  antmaze/pointmaze Stage-A configs except `env_name`. Fit is healthy and NOT overfitting:
  `bc_flow_loss` train 0.232 -> 0.179, validation 0.248 -> 0.174 over 500k, i.e. the
  validation curve tracks training the whole way and ends fractionally BELOW it.
- 2026-09-07 — **BC control done** (job 2491925, 500 ep, 8 workers, 229 s):
  **5/500 = 0.010, Wilson 95% [0.004, 0.023]** -> `$PSM_DATA/evals/bc_scene.json`.
  Inside the pre-registered [0.00, 0.15] but at the LOW end of it (I said 0.03), and well
  under cube's 0.072. Consequence worth stating before Stage C rather than after: the
  "beats its own BC by 3x" claim is now a bar at 0.03, which is nearly free and therefore a
  weak test. **The primary falsifiable claim for scene-play is the ABSOLUTE
  pre-registration, 0.20 with 90% interval [0.05, 0.45]** -- the ratio test is demoted to a
  sanity floor. A result of, say, 0.04 would pass "3x BC" and still be a null; it must be
  reported as one.
  The 10-episode probe (2491924) exited 2 AFTER writing a valid report; the 500-ep job over
  the same script exited clean, and `scripts/eval500.sh` / `eval500.sbatch` were being
  edited by another agent in that window (the probe ran 8 workers where the file on disk now
  defaults to 4). Treated as a launch-time race against a concurrently edited script, not a
  defect. Eval cost for sizing later walls: 500 BC episodes = 229 s at 8 workers.
- 2026-09-07 — **HPO failed and was rerouted.** `sceneHPO` (2491926) died in 23 s at config
  composition: `Could not find 'hydra/sweeper/HyperSMAC'`. `hypersweeper`, `smac` and
  `ConfigSpace` are absent from this cluster's `.venv` and from `requirements.txt` -- the
  harness was built on midi-01 and has never run here. **Not installed, on purpose**:
  `uv pip install --dry-run hypersweeper` resolves to a DOWNGRADE of wandb 0.29.0 -> 0.16.6
  and scipy 1.17.1 -> 1.15.3, and ~97 queued jobs read this `.venv` at launch (wandb 0.29 is
  the version `utils/log_utils.py` was fixed for, HANDOFF 2026-08-29). Rerouted to
  `configs/hpo_preimage_grid.yaml`: same target, same cost, same fixed batch and draw seed,
  hydra's stock sweeper over alpha {5,10,20,40,80} x prior_scale {0.5,0.7,1.0} x n_steps
  {8,12,16} = 45 trials. Smoke 2492039 passed; full sweep is job **2492088** (6 h wall,
  sized from the smoke at ~80 s/trial at n_steps=5 rising to ~210 s at 16).
- 2026-09-07 — **First real inversion signal, and it is the good one.** The smoke's single
  trial (alpha 20, prior_scale 1.0, n_steps 5, N=128, 128 rows, cov_k=8) reads
  **coverage@k8 0.828, decode_mix 0.190 (inside the 0.22 budget), decode_pt 1.7e-4,
  E||u||^2 4.61 vs 5.0 (typicality ratio 0.92), ESS 31.4/128, 0 invalid, 0 NaN.** antmaze
  could not clear 0.115 coverage inside budget at ANY of 50 BO trials. So scene-play is in
  cube's inversion regime, clearing the pre-registered abort threshold (coverage < ~0.3) by
  a wide margin. This is the first evidence bearing on the d_a hypothesis and it points the
  predicted way; it is one trial on 128 rows, so it is a signal, not the answer.


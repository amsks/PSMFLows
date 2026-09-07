# Does the discount horizon matter beyond antmaze? — cube and pointmaze at `gamma=0.99`

Date: 2026-09-07 · Branch `feat/inversion-integration` · Design + **pre-registration**.
Machine: KISSKI (SLURM, H100, partition `kisski-inference`, account `general`).
Subject: the repo-default paper-strict affine agent (`agent=psmflow`, i.e. `psi_form=affine
policy_index=latent index_agg=max train_actor=false acting=gpi`) on
`cube-single-play-singletask-v0` and `pointmaze-medium-navigate-singletask-task1-v0`.
Code touched: **none**. This is a retrain with a single config-only change, `agent.discount`.
Written and committed to **before any job was submitted**; the results section is appended
below and nothing above it is edited afterwards.

## 0. Why this exists

`docs/design/2026-09-06-antmaze-failure-tests.md` §3 (H2) retrained the same agent on
antmaze at `gamma = 0.99` and `0.995` against the repo default `0.98` (effective horizons
100 / 200 / 50 steps). The antmaze runs are still in flight at 400k, but two things are
already visible and neither was predicted at the level of detail they arrived in:

1. **The horizon moved antmaze off the floor, hard.** 500-episode evals of seed 2 @100k:
   `gamma=0.99` → **0.534**, `gamma=0.995` → **0.678**, against the `gamma=0.98` arm's
   late-checkpoint mean **0.081 ± 0.050** and the BC control **0.072**. That is a 6–8x
   move from a one-number config change, on the env this project had written off.
   (`$PSM_DATA/logs/eval500_affine100k_strict_antmaze_g{99,995}_sd2.json`.)
2. **`gamma=0.995` then dies and `gamma=0.99` does not.** In-loop 50-episode success,
   read from each run's `eval.csv`:

   | step | g995 sd0 | g995 sd1 | g995 sd2 | | g99 sd0 | g99 sd1 | g99 sd2 |
   |---|---|---|---|---|---|---|---|
   | 50k | 0.50 | 0.32 | 0.58 | | 0.34 | 0.08 | 0.46 |
   | 100k | 0.32 | 0.56 | 0.78 | | 0.16 | 0.02 | 0.66 |
   | 200k | 0.22 | 0.08 | 0.12 | | 0.38 | 0.00 | 0.48 |
   | 250k | 0.14 | 0.12 | 0.10 | | 0.38 | 0.16 | 0.58 |
   | **300k** | **0.00** | **0.00** | **0.00** | | 0.28 | 0.24 | 0.64 |
   | **350k** | **0.00** | **0.00** | **0.00** | | 0.28 | 0.12 | 0.46 |
   | 400k | 0.06 | 0.00 | 0.02 | | 0.14 | 0.16 | 0.26 |

   All three `gamma=0.995` seeds go to exactly 0.00 at 300k and 350k; all three
   `gamma=0.99` seeds hold. This is the failure mode §3 of the 09-06 note listed **in
   advance** ("a longer discount can destabilise the TD backup … `gamma=0.995` diverging or
   collapsing earlier than `gamma=0.98` is a plausible outcome"), and it fired.

The hypothesis under test here is the generalisation of that: **the discount horizon is a
first-order knob for this agent on every env, not an antmaze-specific fix.** `gamma=0.99`
is the value to carry, because it is the one that both moved and survived.

The two envs are chosen because they are the two remaining published-artifact envs and
because they make **opposite** predictions, which is the only reason the pointmaze arm is
worth a GPU: cube is the arm that works and could be improved or destabilised; pointmaze is
a settled negative with a known mechanism that the discount should be unable to touch. A
sweep that predicts the same thing everywhere tests nothing.

## 1. Reference points (all 500 episodes; BC = the frozen flow acting alone)

| reference | success | source |
|---|---|---|
| **cube** BC control | **0.072** | `$PSM_DATA/evals/bc_cube.json` |
| cube, affine strict `g=0.98`, late mean (300k–500k, 15 cells) | **0.415 ± 0.083** (95% CI) | `docs/tables/affine_cube_ladder.md` |
| cube, affine strict `g=0.98`, min / max cell | 0.078 / 0.704 | same |
| **pointmaze** BC control | **0.002** | `$PSM_DATA/logs/bc_control_pointmaze.json` |
| pointmaze, affine strict `g=0.98` @50k (3 seeds) / @100k (2 seeds) | **0.000** on all 5 cells | `eval500_affine{50,100}k_strict_pointmaze_sd?.json` |
| antmaze, affine strict `g=0.98`, late mean | 0.081 ± 0.050 vs BC 0.072 | 09-06 note §1 |
| antmaze, affine strict `g=0.99` @100k sd2 | 0.534 | this note §0 |

Wilson 95% half-width at n=500 is ±0.026 at p=0.07, ±0.042 at p=0.30 and ±0.044 at p=0.45,
so a shift of the cube late mean by more than ~0.09 is resolvable and a single pointmaze
cell at 0.05 (25/500) is separable from 0/500 at p < 1e-7.

## 2. The full cube `gamma=0.98` ladder this is measured against

500 episodes per cell, three seeds, every 50k. Regenerated from the JSONs, not retyped.

| epoch | sd0 | sd1 | sd2 | mean ± sd |
|---|---|---|---|---|
| 50k | 0.192 | 0.412 | 0.224 | 0.276 ± 0.097 |
| 100k | 0.078 | 0.330 | 0.178 | 0.195 ± 0.104 |
| 150k | 0.088 | 0.268 | 0.596 | 0.317 ± 0.210 |
| 200k | 0.264 | 0.298 | 0.466 | 0.343 ± 0.088 |
| 250k | 0.532 | 0.620 | 0.246 | 0.466 ± 0.160 |
| 300k | 0.286 | 0.532 | 0.360 | 0.393 ± 0.103 |
| 350k | **0.704** | 0.494 | 0.350 | 0.516 ± 0.145 |
| 400k | 0.282 | 0.346 | 0.546 | 0.391 ± 0.112 |
| 450k | 0.272 | 0.568 | 0.564 | 0.468 ± 0.139 |
| 500k | 0.086 | 0.548 | 0.292 | 0.309 ± 0.189 |

Largest jump between *adjacent* 50k checkpoints, per seed: **0.422** (sd0, 350k→400k),
0.322 (sd1, 200k→250k), **0.418** (sd2, 100k→150k). That oscillation is the second thing
this experiment is about.

---

## 3. H3a — cube: the horizon is **not** the binding constraint, so `gamma=0.99` moves the
## mean by less than the noise, and does **not** fix the oscillation

**Claim.** Cube episodes are 200 steps and the task reward is reachable well inside the
50-step effective horizon of `gamma=0.98`. Unlike antmaze — where the goal is 200–400 steps
away and `0.98^300 ≈ 2e-3` discounts it into the numerical floor — nothing about cube is
horizon-starved. Doubling the horizon to 100 steps therefore buys no reachability and costs
bootstrap variance: the same TD target is now accumulated over twice as many steps of a
measure fitted on a latent index that the 09-05/09-06 work showed is an optimism device
rather than a policy.

**Pre-registered point prediction.** Cube late mean (300k–500k, 15 cells) at `gamma=0.99`:
**0.42**, with an interval of **0.33–0.50**. Formally: I predict the `g=0.99` late mean lies
within ±0.09 of the `g=0.98` late mean 0.415, i.e. **no significant change**.

**Pre-registered prediction on the oscillation, stated separately because it is the more
falsifiable half.** The ladder will oscillate **just as much**: at least **two of three
seeds** will show an adjacent-checkpoint jump of **>= 0.30**, and the per-seed range
(max − min over the ten checkpoints) will be **>= 0.35** on at least two seeds. I am
explicitly predicting that the discount does *not* stabilise this arm, because nothing in
the diagnosed mechanism is a horizon effect: the 09-06 three-seed ladder tied the swing to
training-time non-stationarity with `w_enc_spread` (the policy-encoder collapse) decaying
smoothly to the same place on all seeds while success swung ±0.4 around it, and the 09-06
H1 result tied the arm's whole score to a per-step lottery over 64 fresh policy indices.
Neither is a function of `gamma`.

**Outcomes and what each means.**

- **Confirmed (expected):** late mean in 0.33–0.50 and the oscillation criteria above both
  met. Reading: the horizon is antmaze-specific; `gamma=0.99` is a safe default change that
  costs cube nothing, and the cube instability has a different cause that this sweep does
  not touch.
- **Cube improves, late mean >= 0.55:** falsifies the "not horizon-starved" argument for
  cube and makes the discount a general win. It would need explaining — the obvious
  candidate is that a longer horizon smooths the index lottery rather than lengthening
  reach — and would justify a `gamma=0.995` cube arm, which is *not* being run now.
- **Cube degrades, late mean <= 0.30:** the bootstrap-variance cost is real and the antmaze
  gain is being bought with stability elsewhere. That makes `gamma` an env-level
  hyperparameter, not a fix, and is the outcome that would keep the repo default at 0.98.
- **Oscillation actually shrinks** (no seed with a >= 0.30 adjacent jump): would be the most
  interesting single result in this note and directly contradicts the 09-06 diagnosis.
  It is the one I consider least likely and I am recording that here so it cannot be
  claimed as expected afterwards.

## 4. H3b — pointmaze: the discount **cannot** fix a policy family with no goal-reaching
## member, so the answer is 0.000 everywhere

**The settled negative and its mechanism.** `docs/COMPENDIUM.md` §4.11 ("the Rung-1 root
cause, 2026-08-05 — closed, do not re-audit") records that every Stage-C variant reads
**0.0** on pointmaze task 1 and that the cause is **not a bug**:

1. **Exhaustive reachability.** `d_a = 2`, so a 13x13 grid tiles the entire `[-3,3]^2`
   latent box; plus 64 dataset/goal preimages, each rolled for two full 1000-step episodes.
   **0 of 233 latents ever reach the goal**, and 75% never get closer than ~23.5 in a maze
   ~30 across.
2. **The expert's route is latent white noise.** Within-episode preimage variance /
   marginal variance = **0.99**; lag-1 autocorrelation 0.27, ~0 by lag 50. The BC flow
   factorises behaviour as (state → conditional, `u` → quantile), and the expert's direction
   choice is driven by a goal that is **not in the observation**, so that variance is forced
   into `u` independently at every step. **Routes exist only as latent *sequences*.**
3. **What the family contains.** Typical `u` gives orbiters (path length 164, net
   displacement 1.4); saturated `u` gives constant headings that cannot turn.

**Can the discount plausibly affect that? Almost entirely no, and I am pre-registering the
"no".** `agent.discount` is a parameter of the TD backup that fits `psi`. It changes *which
member of the policy family `psi` prefers* and over what horizon that preference is
accumulated. It does not change the family. Every member of that family has already been
enumerated and rolled out, and none of them reaches the goal — so there is no ranking over
that family, at any horizon, whose argmax reaches the goal. The failure is in the frozen
Stage-A/B artifacts, upstream of everything `gamma` touches. Nothing in this experiment
re-opens §4.11's audit and it should not be read as doing so.

**The one channel through which it is not a flat no, stated honestly because it is the only
reason to spend the GPU.** The deployed acting rule does not execute a fixed `u`. It redraws
`K = 64` policy indices every step and takes the max, so the executed trajectory *is* a
latent sequence — precisely the object §4.11 says routes live in. §4.11's reachability test
used **fixed** `u` and therefore does not by itself rule out that a per-step ranking could
*stitch* a goal-reaching sequence out of members none of which reaches the goal alone. If
such stitching is possible at all, a 50-step horizon in a maze that needs ~200 steps is
exactly the regime where it would be invisible, and lengthening the horizon is the one
intervention that addresses it. That is a narrow channel and I do not expect it to carry:
the same per-step-max rule already reads 0.000 on all five `gamma=0.98` cells evaluated so
far, and the 09-06 H1 result showed the per-step max behaves as an optimism device rather
than as a route planner on a 1000-step maze.

**Pre-registered prediction.** **0.000 on every one of the 30 cells** (10 checkpoints x 3
seeds, 500 episodes each = 15000 episodes without a success). Late mean (300k–500k)
**0.000 ± 0.000**. Point prediction for the single best cell: **0.000**; I will accept
anything <= 0.004 (2/500) as consistent, since the BC control itself is 0.002.

**Outcomes and what each means.**

- **Confirmed (expected):** all cells 0.000–0.004. Reading: §4.11 stands, and it now stands
  against the *acting rule* as well as the fixed-`u` family — 15000 episodes of the
  per-step-max rule at a 100-step horizon produce nothing. The stitching channel above is
  closed and pointmaze stays a settled negative. This is the outcome that makes the run a
  cheap confirmation rather than a discovery, and it is worth the 3 GPU-hours only because
  the antmaze result made the "discount is env-specific" claim testable at all.
- **Any cell >= 0.05** (25/500, separable from 0/500 at p < 1e-7 and from BC 0.002 at
  p < 1e-6): the stitching channel is real and §4.11 is **not** settled for
  per-step-reselecting acting rules. That would be a genuine reopening and would justify
  re-running the §4.11 reachability probe with a per-step-argmax rollout instead of a
  fixed-`u` one. I consider this unlikely and am recording the threshold now so that a
  0.01–0.03 cell cannot be narrated into a result later.
- **Cells in 0.004 < x < 0.05:** noise at the BC floor, reported as such, **not** as a
  partial rescue.

## 5. Failure modes to watch, named in advance

1. **The `gamma=0.995`-style late collapse (the one that already fired once).** All three
   antmaze `g995` seeds read exactly 0.00 in-loop at 300k **and** 350k while `g99` held.
   The monitor is each run's `eval.csv` at every 50k. Signature: in-loop success pinned at
   **exactly 0.00** on **all three seeds simultaneously** from some epoch onward, not one
   seed dipping. If it appears on cube `g=0.99` it means the collapse is not specific to
   the 200-step horizon and the whole discount direction is a stability trade, not a win.
   **It does not abort the ladder:** every checkpoint still gets its 500-episode eval,
   because the in-loop number is the thing this project has repeatedly caught over- and
   under-reading (antmaze sd2 @300k: 0.40 in-loop, 0.088 over 500).
2. **Encoder collapse arriving earlier.** `training/w_enc_spread` in `train.csv`. On cube at
   `g=0.98` it peaks ~1.31 and decays to 0.36 / 0.25 / 0.31 by 500k, first crossing 0.6 at
   190k / 215k / 305k. If `g=0.99` crosses 0.6 markedly earlier on all three seeds that is
   the mechanism by which a longer horizon would hurt, and it is checkable without any eval.
3. **TD divergence.** `training/psm_loss`, `training/psi_q_spread` and
   `training/psi_q_range_rel` in `train.csv`; NaN or monotone blow-up is the loud version of
   failure mode 1. `training/orth_loss` / `orth_diag` / `orth_offdiag` for the
   orthonormaliser.
4. **Reading the answer off the in-loop eval.** Named as a failure mode because it is a
   process failure this project has committed before. Every reported number in this note is
   500 episodes at every 50k for every seed; the in-loop 50-episode eval appears only as
   collapse *monitoring*, never as a reported success rate and never as a checkpoint picker.
5. **Cross-arm contamination.** Six jobs land in a queue that already holds ~60 belonging to
   other work. Groups are namespaced `affine_strict_{cube,pointmaze}_g99` and the eval
   outputs `eval500_affine{N}k_strict_{env}_g99_sd{S}.json`, disjoint from every existing
   basename, so nothing overwrites and nothing is double-counted by `make_tables.py`'s globs.

## 6. Hyperparameters (identical to `affine_strict_{cube,pointmaze}` except `discount`)

Read back from `affine_strict_cube/sd000_*/flags.json` and
`affine_strict_pointmaze/sd000_*/flags.json`; the only key that differs between this sweep
and those runs is `agent.discount`.

| key | value | key | value |
|---|---|---|---|
| `agent_name` | psmflow | `z_dim` | 128 |
| `psi_form` | affine | `affine.w_dim` | 128 (`norm_w=true`) |
| `policy_index` | latent | `affine.encoder` | 256 x 2 |
| `index_agg` | max | `num_parallel` | 2 |
| `acting` | gpi | `pessimism_penalty` | 0.5 |
| `train_actor` | false | `actor_pessimism_penalty` | 0.5 |
| `gpi_num_u` | 64 | **`discount`** | **0.99** (was 0.98) |
| `gpi_select` | argmax | `tau` | 0.01 |
| `gpi_decode` | onestep | `ortho_coef` | 1000.0 |
| `u_clip` | 3.0 | `backup_explore_frac` | 0.0 |
| `use_point_preimage` | true | `mix_ratio` | 0.5 |
| `action_critic.enabled` | false | `batch_size` | 1024 |
| `lr_phi` | 1e-05 | `lr_sf` | 1e-04 |
| `expectile_mu` | 0.9 | `index_panel` | 16 |
| `offline_steps` | 500000 | `save_interval` | 50000 |
| `eval_interval` | 50000 | `eval_episodes` | 50 |
| `eval_relabel_size` | 10000 | `eval_reward_shift` | 1.0 |
| cube env | `cube-single-play-singletask-v0` (OGBench task 2) | pointmaze env | `pointmaze-medium-navigate-singletask-task1-v0` |
| cube flow | `$PSM_DATA/flow/cube-single-play` @500000 | cube preimages | `$PSM_DATA/preimages/cube-single-play.npz` |
| pointmaze flow | `$PSM_DATA/flow/pointmaze-medium-navigate` @500000 | pointmaze preimages | `$PSM_DATA/preimages/pointmaze-medium-navigate.npz` |

Seeds 0, 1, 2 per env. Groups `affine_strict_cube_g99`, `affine_strict_pointmaze_g99`.

## 7. Protocol

**Training.** `scripts/slurm/train_psmflow.sbatch`, one job per seed per env (6 jobs),
`--partition=kisski-inference --account=general --gres=gpu:1`, `--time=12:00:00` (the
sbatch header default; measured wall clock for this agent is 2 h 42 m on cube and ~4 h on
the mazes), `SAVE_INT=50000 EVAL_INT=50000 EVAL_EPS=50 STEPS=500000`,
`EXTRA="agent.discount=0.99"` — the same override, passed the same way, as the antmaze
`g99` runs (SLURM 2491831-2491833).

**Evaluation.** 500 episodes at **every** 50k checkpoint for **all** seeds: 30 cells per env,
60 jobs, `scripts/slurm/eval500.sbatch` with `EVAL_WORKERS=1`. `EVAL_WORKERS=1` is the
serial path that reproduces every pre-2026-09-07 number bit for bit (`scripts/eval500.sh`:
N>1 seeds worker `w` as `seed*N + w` and therefore draws a *different* sample of 500 episode
inits, agreeing only within the Wilson interval). The `gamma=0.98` ladders this sweep is
compared against were all produced serially, so the comparison is uniform only at N=1.
Cost: cube ~20 min/eval, pointmaze ~95 min/eval, all concurrent, `--time=03:00:00`.
Outputs `$PSM_DATA/logs/eval500_affine{N}k_strict_{cube|pointmaze}_g99_sd{S}.json`.

**Deliverables.** `docs/tables/affine_discount_ladder.md` (both envs, 0.98 beside 0.99),
`docs/figures/2026-09-07-affine-discount-ladder.{png,json}`, append-only rows in
`tools/make_tables.py`, and a `docs/HANDOFF.md` section under the 2026-09-06 entry.

**Discipline.** Hyperparameter table printed above before submission; a 200-step smoke of the
exact training path per env before launch; every run's `flags.json` re-read after launch and
confirmed to carry `discount: 0.99` and nothing else changed; expected outcomes — including
the expected *failures* — stated above before any number arrives; every reported number from
a persisted JSON.

---

## 8. Results

*(appended after the runs return; nothing above this line is edited)*

### 8.0 Provenance — config drift since the antmaze `g99` runs, checked before any number

Recorded here rather than in §6 so the pre-registration above stays exactly as submitted.

The two 200-step smokes (SLURM **2491981** cube, **2491982** pointmaze) both returned
COMPLETED 0:0 in ~2 min, and both `flags.json` carry `agent.discount: 0.99`. Diffed against
`affine_strict_cube/sd000` and `affine_strict_pointmaze/sd000`, they differ in more keys
than the antmaze `g99` runs did 14 hours earlier — `agents/psmflow.py` and
`configs/config.yaml` gained keys in between (the 09-06 DSRL-actor work and the 09-07
GPU-packing work). The new names, and why each is inert **for this arm specifically**:

| key | value | why it changes nothing here |
|---|---|---|
| `agent.actor_mode` | `ddpg` | the default, and `psmflow.py:223` **asserts** it is the only legal value when `train_actor=false`. |
| `agent.actor.index_panel` | 0 | read only in `_actor_q` (`:517`) and the DSRL-NA path (`:548`), both actor-only. 0 = "read psi at the ONE prior index this batch element drew" = prior behaviour; its assertions (`:229-234`) bite only when > 0. Distinct from the top-level `agent.index_panel=16` that the expectile head uses (`:719`), which is unchanged. |
| `agent.actor.{entropy,target_entropy,init_alpha,lr_alpha,log_std_min,log_std_max,q_coeff,na_*}` | defaults | tanh-Gaussian head parameters; the head is never built or trained under `train_actor=false` (the actor loss is gated at `:946`). |
| `eval_workers` | 1 | top-level key defaulting to 1 in `configs/config.yaml`, read only by `tools/eval_checkpoint.py`. Inert during training, and 1 is the serial eval protocol §7 already pins. |
| `agent.gpi_{select,topm,index_seed}` | `argmax`, 8, 0 | the 09-05 eval-time selection-rule knobs; the antmaze `g99` runs carry them too. `argmax` is the shipped rule. |
| `agent.gpi_prior_shrink` | 0.8 | read at exactly one site (`:1122`), inside the `mode == "prior_shrunk"` branch of the `gpi_select` dispatch, so unreachable under `gpi_select=argmax`. `:1374` records that `argmax` "is the shipped rule and the only one training ever uses; the rest are eval-time ablations". |

**A hazard this exposed, worth stating.** `gpi_prior_shrink` was **not** in the smoke's
`flags.json` (01:5x) but **was** in cube sd0's (07:49): another agent added the
`prior_shrunk` selection mode to `configs/agent/psmflow.yaml` while these six jobs sat
queued, and a SLURM job reads the config fresh at launch. Six jobs that start hours apart
can therefore each pick up a different config, and a smoke's `flags.json` is only a
guarantee about the smoke. The mitigation used here is to diff all six runs' own
`flags.json` **against each other** as well as against the `gamma=0.98` references, so a
mid-queue edit that is *not* inert would be caught in the runs rather than assumed away by
the smoke. That check is reported in §8.1.

### 8.1 Post-launch `flags.json` verification — all six runs

All six trainings started 07:49-08:10 on 2026-09-07 and every run's own `flags.json` was
re-read (not the smoke's, not the submit line).

| env | seed | SLURM | run dir |
|---|---|---|---|
| cube | 0 | 2492017 | `affine_strict_cube_g99/sd000_s_2492017.0.20260907_074919` |
| cube | 1 | 2492018 | `affine_strict_cube_g99/sd001_s_2492018.0.20260907_075715` |
| cube | 2 | 2492019 | `affine_strict_cube_g99/sd002_s_2492019.0.20260907_075755` |
| pointmaze | 0 | 2492020 | `affine_strict_pointmaze_g99/sd000_s_2492020.0.20260907_075756` |
| pointmaze | 1 | 2492021 | `affine_strict_pointmaze_g99/sd001_s_2492021.0.20260907_080034` |
| pointmaze | 2 | 2492022 | `affine_strict_pointmaze_g99/sd002_s_2492022.0.20260907_081037` |

Two checks, both passed:

1. **Cross-seed consistency** (the mitigation §8.0 calls for). Every key of all three
   `flags.json` within each env, `seed` and `run_group` excluded, is **identical across all
   seeds** — for cube and for pointmaze. The mid-queue config edit landed before the first
   job started, so all six picked up the same `configs/agent/psmflow.yaml`; the seeds are
   not differentiated by it.
2. **Against the `gamma=0.98` references**, ignoring the keys §8.0 established are inert,
   the only substantive difference on **all six** runs is
   **`agent.discount: 0.99` (was 0.98)** — nothing else.

Confirmed on every run: `offline_steps=500000`, `save_interval=50000`,
`eval_interval=50000`, `eval_episodes=50`, and the arm
`psi_form=affine policy_index=latent index_agg=max train_actor=false acting=gpi`, against
the same frozen flow and preimage npz as the `gamma=0.98` runs.

`ACTOR_DEFAULTS` (`psmflow.py:96`) exists precisely so "a flags.json written before a key
existed still builds", so the `gamma=0.98` checkpoints restore under these same defaults:
the comparison is sound in both directions. **The reasoning is arm-specific** — it holds
because this arm is `train_actor=false`, and would not transfer to the latent-actor arm.

Net: across all six runs the only substantive difference from their `gamma=0.98` references
is `agent.discount`, which is what the experiment is.

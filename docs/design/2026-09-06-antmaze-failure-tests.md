# Why is affine paper-strict null on antmaze? — two pre-registered tests

Date: 2026-09-06 · Branch `feat/inversion-integration` · Design + pre-registration.
Machine: KISSKI (SLURM, H100, partition `kisski-inference`, account `general`).
Subject: `$PSM_DATA/exp/PSMFLows/affine_strict_antmaze/sd00{0,1,2}_*`, the repo-default
paper-strict agent (`psi_form=affine policy_index=latent index_agg=max train_actor=false
acting=gpi`), env `antmaze-medium-navigate-singletask-v0`.
Code touched: none. H1 is an **eval-time** switch that already exists
(`agent.gpi_select=fixed_index`, `docs/design/2026-09-05-gpi-ablations.md`); H2 is a
retrain with a **config-only** change (`agent.discount`).

## 0. The state of play

Affine strict on antmaze is null. 500-episode evals of the three seeds:

| epoch | sd0 | sd1 | sd2 |
|---|---|---|---|
| 50k | 0.112 | 0.000 | — |
| 250k | — | **0.178** | — |
| 300k | — | — | 0.088 |
| 350k | 0.082 | — | — |
| 400k | 0.056 | 0.134 | — |
| 500k | 0.002 | 0.096 | 0.008 |

Late-checkpoint mean (>=250k, 8 cells) **0.081 ± 0.050** against the BC control **0.072**:
the interval contains BC, so this is not a result. Episodes hit the 1000-step timeout.

Two things about antmaze make it different from cube, and each is one of the hypotheses
below.

1. **Antmaze is a 1000-step navigation task; cube is a short manipulation task.** The
   deployed acting rule redraws `K=64` policy indices `u'` *every step* and takes the max.
   The 09-05 cube ablations showed that this per-step index max **is the mechanism**: pin
   `u'` for the whole eval and cube 0.704 collapses to 0.000 (seed 0) / 0.180 (seed 1).
   On cube, per-step reselection is what makes the number. On a maze it may be what breaks
   it — a policy that re-picks its index every step has no committed direction, and the ant
   dithers until the horizon. The arms with a **persistent** policy (the affine *actor*
   arm, which decodes one amortized actor latent per step from a single learned policy)
   score ~0.21 on the same env, checkpoints and flow — 3x this arm and 3x BC.
2. **`discount=0.98`** is an effective horizon of ~50 steps. Antmaze-medium episodes are
   1000 steps and the goal is typically several hundred steps away, so the value of the
   goal is discounted to ~`0.98^300 ≈ 2e-3` at the start state: the TD fixed point may
   simply carry no gradient toward the goal from most of the maze. Cube's episodes are
   200 steps and its reward is reachable inside the 50-step horizon.

They are not exclusive and neither is guaranteed; both are cheap.

## 1. Reference points (all 500 episodes, Wilson 95% at n=500)

| reference | success | note |
|---|---|---|
| BC control (frozen flow, per-step prior) | **0.072** | `bc_control_antmaze.json` |
| affine strict, late-ckpt mean | 0.081 ± 0.050 | the arm under test |
| affine strict, best single cell (sd1 @250k) | 0.178 | best of 8 |
| **affine latent-actor arm (persistent policy)** | **~0.21** | 0.224/0.206 @500k, 0.218/0.224 @50k, 0.176/0.216 @100k |

At n=500 the Wilson half-width is ~±0.026 at p=0.10 and ~±0.036 at p=0.21, so the
BC-vs-persistent gap (0.072 vs 0.21) is resolvable many times over; a fixed_index cell at
0.15 is separable from BC (p<0.001) and a cell at 0.10 is not (p≈0.08).

---

## 2. H1 — per-step GPI re-selection is temporally incoherent over a 1000-step episode

**Claim.** The affine-strict arm fails on antmaze not because `psi` is uninformative but
because the acting rule has no temporal commitment: every step it draws a fresh panel of
64 policy indices and maximises over them, so the executed behaviour is a different policy
at every step of a 1000-step episode. Any consistent policy — even a mediocre one — beats
an incoherent max on a navigation task.

**Test (eval only, no retraining).** `agent.gpi_select=fixed_index`: `u'` is pinned to one
draw `PRNGKey(agent.gpi_index_seed)` and held for the whole evaluation (all 500 episodes,
all 1000 steps), while the inner `argmax_u` over the 64 action latents runs per step as
usual. Everything else is inherited from the run's own `flags.json`. Four index seeds
`{0,1,2,3}` per checkpoint, because the 09-05 cube result showed the pinned index is a
lottery (0.000 vs 0.180 for two draws), so a single draw is uninterpretable.

| cell | run dir | epoch | argmax reference |
|---|---|---|---|
| sd1 @250k | `affine_strict_antmaze/sd001_s_2491603.0.20260904_181115` | 250000 | 0.178 |
| sd0 @350k | `affine_strict_antmaze/sd000_s_2491599.0.20260904_181113` | 350000 | 0.082 |
| sd2 @300k | `affine_strict_antmaze/sd002_s_2491698.0.20260906_033853` | 300000 | 0.088 |

12 jobs (3 checkpoints x 4 index seeds), 500 episodes each, `scripts/eval500.sh` via
`scripts/slurm/eval500.sbatch` with
`EXTRA="agent.gpi_select=fixed_index agent.gpi_index_seed=<I>"` and `RESTORE_EPOCH=<epoch>`;
~1 h 40 m each (measured, job 2491799), all concurrent, `--time=03:00:00`.
Reports: `$PSM_DATA/logs/eval500_antmaze_fixedidx_sd{S}_{epoch}k_is{I}.json`.

The "redraw the index every T=50 steps" variant is **not** run: `gpi_select` has no such
mode and adding one is code, which this note deliberately avoids.

**Pre-registered expectations.**

- **Under H1:** several of the 12 cells reach **>= 0.15**, i.e. the persistent-policy band
  (~0.21) rather than the BC band (~0.07), and the spread *across index seeds within a
  checkpoint* is large (some indices are good directions, some are not) while the spread
  across checkpoints is comparatively small. The specific signature is: pinning helps on a
  1000-step maze exactly where it destroyed cube, because the per-step index max is an
  optimism device that a short manipulation task can absorb and a long navigation task
  cannot.
- **Under not-H1:** every cell stays **<= 0.10**, indistinguishable from BC 0.072 and from
  the argmax references (0.082-0.178). The failure is then in `psi` itself, not in the
  temporal structure of the acting rule, and the ~0.21 of the actor arm comes from the
  amortized actor being trained (a different learning signal) rather than from persistence.
- **Third outcome, explicitly allowed:** cells land *below* BC, as cube's `fixed_index`
  seed 0 did (0.000, significantly below BC). That says the pinned index selects actively
  harmful actions and reproduces the cube finding on a second env — informative, and still
  not-H1.
- **Note on the best cell:** sd1 @250k already scores 0.178 with the shipped rule, within
  noise of the persistent band. If *only* that checkpoint lifts, the result is about that
  checkpoint, not about the rule; H1 is confirmed only if the lift appears on checkpoints
  whose argmax score is at the floor (sd0 @350k = 0.082, sd2 @300k = 0.088).

## 3. H2 — `discount=0.98` is too short a horizon for antmaze-medium

**Claim.** With `gamma=0.98` the value function's effective horizon is `1/(1-gamma) = 50`
steps. Antmaze-medium requires hundreds. The measure/critic therefore has no signal that
distinguishes states by their distance to a goal that is 200-400 steps away, and every
acting rule built on it is choosing between indistinguishable options — which is also
consistent with the observed 1000-step timeouts and with the arm sitting exactly at BC.

**Test (retrain).** Two groups, 3 seeds each, everything else at the repo default (the
exact config that produced `affine_strict_antmaze`):

| group | `agent.discount` | horizon `1/(1-g)` | seeds |
|---|---|---|---|
| `affine_strict_antmaze_g99` | 0.99 | 100 | 0,1,2 |
| `affine_strict_antmaze_g995` | 0.995 | 200 | 0,1,2 |

500k offline steps, `save_interval=50000`, `eval_interval=50000`, `eval_episodes=50`,
`scripts/slurm/train_psmflow.sbatch`, `--time=08:00:00` (measured 4 h 06 m for the
`gamma=0.98` antmaze seed).

**Pre-registered expectations.**

- **Under H2:** the in-loop 50-episode eval leaves the floor by ~250k on `gamma=0.995`,
  and the 500-episode success at 500k is **>= 0.15** for `gamma=0.995`, with `gamma=0.99`
  intermediate (0.10-0.20). Monotone in the horizon is the signature; a lift on 0.99 with
  nothing on 0.995, or vice versa, is noise, not H2.
- **Under not-H2:** all six runs stay in the 0.00-0.10 band at 500k, like the `gamma=0.98`
  runs. Longer horizons then do not rescue the arm and the problem is elsewhere (`psi`
  capacity, the encoder collapse, or the acting rule of H1).
- **Anticipated failure mode, stated in advance:** a longer discount can *destabilise* the
  TD backup — the affine critic already shows the ±0.4 within-run swings documented on cube
  and the `w_enc_spread` collapse. `gamma=0.995` diverging or collapsing to 0.000 earlier
  than `gamma=0.98` is a plausible outcome and would be reported as such, not as H2 being
  untestable.
- **The in-loop eval over-reads on antmaze and must not be used to pick the answer.** sd2's
  300k checkpoint read **0.40 over 50 episodes** and **0.088 over 500**. In-loop numbers are
  used only to choose *which* checkpoint to spend a 500-episode eval on; every reported
  number is 500 episodes.

**Follow-up evals (after training).** For each of the 6 runs: 500 episodes at 500k and at
the best in-loop checkpoint = 12 more jobs, `--time=03:00:00`, concurrent. Reports:
`$PSM_DATA/logs/eval500_affine{N}k_strict_antmaze_g{99|995}_sd{S}.json`.

## 4. Hyperparameters (H2 training; identical to `affine_strict_antmaze` except `discount`)

Read back from `affine_strict_antmaze/sd001_*/flags.json`.

| key | value | key | value |
|---|---|---|---|
| `agent_name` | psmflow | `z_dim` | 128 |
| `psi_form` | affine | `affine.w_dim` | 128 (`norm_w=true`) |
| `policy_index` | latent | `affine.encoder` | 256 x 2 |
| `index_agg` | max | `num_parallel` | 2 |
| `acting` | gpi | `pessimism_penalty` | 0.5 |
| `train_actor` | false | `actor_pessimism_penalty` | 0.5 |
| `gpi_num_u` | 64 | **`discount`** | **0.99 / 0.995** (was 0.98) |
| `gpi_decode` | onestep | `tau` | 0.01 |
| `u_clip` | 3.0 | `ortho_coef` | 1000.0 |
| `use_point_preimage` | true | `backup_explore_frac` | 0.0 |
| `action_critic.enabled` | false | `mix_ratio` | 0.5 |
| `offline_steps` | 500000 | `save_interval` | 50000 |
| `eval_interval` | 50000 | `eval_episodes` | 50 |
| flow | `$PSM_DATA/flow/antmaze-medium-navigate` @500000 | preimages | `$PSM_DATA/preimages/antmaze-medium-navigate.npz` |
| `eval_relabel_size` | 10000 | `eval_reward_shift` | 1.0 |

## 5. Results — H1

Twelve concurrent jobs (SLURM 2491818-2491829), all COMPLETED, **11 min each** — not the
budgeted 1 h 40 m, because `fixed_index` sets `n_idx=1` and the `K x K = 4096` pair scan
collapses to `K = 64` psi evaluations per step. Reports:
`$PSM_DATA/logs/eval500_antmaze_fixedidx_sd{S}_{epoch}k_is{I}.json`. Section 2 was not
edited afterwards.

| checkpoint | argmax ref | is0 | is1 | is2 | is3 | mean |
|---|---|---|---|---|---|---|
| sd1 @250k | 0.178 | 0.002 | **0.142** | 0.000 | 0.000 | 0.036 |
| sd2 @300k | 0.088 | 0.020 | 0.026 | 0.004 | 0.000 | 0.013 |
| sd0 @350k | 0.082 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

Full cells with Wilson 95% intervals and two-proportion z-tests at n=500 (BC = 36/500):

| cell | k/500 | success | Wilson 95% | vs BC 0.072 | vs its own argmax |
|---|---|---|---|---|---|
| sd1 @250k is0 | 1 | 0.002 | [0.000, 0.011] | z=-5.86, p=4.5e-09 | z=-9.72, p=2.4e-22 |
| sd1 @250k is1 | 71 | **0.142** | [0.114, 0.175] | z=+3.58, p=3.4e-04 | z=-1.55, p=0.12 |
| sd1 @250k is2 | 0 | 0.000 | [0.000, 0.008] | z=-6.11, p=9.9e-10 | z=-9.88, p=4.9e-23 |
| sd1 @250k is3 | 0 | 0.000 | [0.000, 0.008] | z=-6.11, p=9.9e-10 | z=-9.88, p=4.9e-23 |
| sd2 @300k is0 | 10 | 0.020 | [0.011, 0.036] | z=-3.92, p=8.7e-05 | z=-4.76, p=2.0e-06 |
| sd2 @300k is1 | 13 | 0.026 | [0.015, 0.044] | z=-3.37, p=7.5e-04 | z=-4.23, p=2.4e-05 |
| sd2 @300k is2 | 2 | 0.004 | [0.001, 0.015] | z=-5.62, p=1.9e-08 | z=-6.34, p=2.3e-10 |
| sd2 @300k is3 | 0 | 0.000 | [0.000, 0.008] | z=-6.11, p=9.9e-10 | z=-6.78, p=1.2e-11 |
| sd0 @350k is0-3 | 0 | 0.000 | [0.000, 0.008] | z=-6.11, p=9.9e-10 | z=-6.54, p=6.2e-11 |

Pooled over all 12 cells: **97 / 6000 = 0.0162**, against BC 36/500 = 0.072,
z = -8.47, p = 2.4e-17.

### 5.1 Verdict: **H1 is refuted, and refuted in the opposite direction**

Pinning the policy index does not rescue antmaze — it **destroys** it. Eleven of twelve
cells are at or below 0.026 and **every one of those eleven is significantly BELOW the BC
control** (p <= 8.7e-05 in each case, p = 2.4e-17 pooled). Six of the twelve are exactly
0/500. The pre-registered not-H1 threshold was "every cell <= 0.10"; the observed cells
are an order of magnitude below that, and the single exception is discussed next.

The one cell above the threshold, **sd1 @250k with index seed 1 at 0.142**, is *not* a
lift. It sits on the one checkpoint that already scores **0.178** with the shipped per-step
argmax (a difference of -0.036, p = 0.12, i.e. no significant change), and the same
checkpoint's other three index seeds return 0.002 / 0.000 / 0.000. Section 2 pre-registered
this exact caveat: a move confined to the checkpoint that was already the best of the eight
is about that checkpoint, not about the rule. The three checkpoints whose argmax score is at
the floor — the ones the hypothesis needed — return **0.000 on all eight of their cells**
except sd2's 0.020 / 0.026.

So the third pre-registered outcome fired, not the first: the pinned index selects actively
harmful actions, exactly as `fixed_index` seed 0 did on cube (0.000 there too). **The
result reproduces the 09-05 cube finding on a second environment and a second task family:
the per-step max over 64 fresh policy indices is load-bearing, and pinning `u'` costs
everything the rule had.** It is an optimism device averaged over the index lottery every
step, not a policy choice, and it behaves identically on a 200-step manipulation task and a
1000-step navigation task.

### 5.2 What that leaves

- **Temporal incoherence of the acting rule is not why antmaze fails.** The rule's per-step
  reselection is, if anything, the only thing keeping this arm at BC rather than at zero.
- **The ~0.21 of the affine latent-actor arm is therefore not "persistence".** A pinned
  index is maximally persistent and scores 0.016 pooled. Whatever the actor arm has, it
  comes from the amortized actor being *trained* — a distillation over the index panel with
  its own learning signal — not from committing to one index. That reframes the actor arm
  as the interesting object: it is the only antmaze arm above BC, and the reason is now
  narrowed to its training, not its temporal structure.
- **The index lottery is enormous on antmaze.** Within one checkpoint, four draws of `u'`
  give 0.142 / 0.002 / 0.000 / 0.000. `psi`'s preference over action latents, conditioned
  on a *fixed* index, is worth between -0.072 and +0.070 against BC depending on which
  index you drew. The learned measure has no index-independent notion of a good action.
- **Cheap and unregistered:** `fixed_index` is 64x cheaper per step (64 psi evaluations
  instead of 4096). That is worth remembering only as an ablation cost, since the mode is
  useless as a policy.

## 6. Results — H2 (COMPLETE: 51 cells, 500 episodes each)

Six trainings, SLURM 2491831-2491836, all COMPLETED in 3 h 53 m - 4 h 10 m. Every
`flags.json` was re-read after launch and differs from `affine_strict_antmaze/sd001` in
`discount` alone (plus the three eval-only `gpi_select` keys added 09-05, which training
never reads). A 200-step smoke of the exact path (2491830) was checked the same way before
submission. The full ladder is 51 500-episode cells — `gamma=0.99` at every 50k checkpoint
50k-500k (30) and `gamma=0.995` at 50k-300k plus a 500k endpoint (21) — all on the serial
eval path (`EVAL_WORKERS=1`), the one the 12 H1 cells and every historical antmaze number
used. Per-cell table with Wilson intervals: `docs/tables/affine_antmaze_discount_ladder.md`.
Figure: `docs/figures/2026-09-07-affine-antmaze-discount-ladder.png`.
Sections 2 and 3 were not edited after any result returned.

### 6.1 Verdict: **H2 is confirmed. The discount was the blocker — and 0.99, not 0.995.**

Pooled over (checkpoint x seed) cells, each cell one 500-episode measurement:

| arm | horizon | window | n | mean | 95% CI | min | max |
|---|---|---|---|---|---|---|---|
| gamma=0.98 (default) | 50 | all measured | 10 | **0.076** | ± 0.043 | 0.000 | 0.178 |
| **gamma=0.99** | 100 | **all measured** | 30 | **0.294** | **± 0.070** | 0.004 | 0.624 |
| gamma=0.99 | 100 | 50k-250k | 15 | 0.336 | ± 0.097 | 0.050 | 0.624 |
| gamma=0.99 | 100 | 300k-500k | 15 | 0.252 | ± 0.100 | 0.004 | 0.590 |
| gamma=0.995 | 200 | all measured | 21 | 0.214 | ± 0.092 | 0.002 | 0.678 |
| gamma=0.995 | 200 | 50k-250k | 15 | 0.296 | ± 0.102 | 0.074 | 0.678 |
| gamma=0.995 | 200 | **300k-500k** | 6 | **0.010** | ± 0.008 | 0.002 | 0.020 |

Reference: BC control **0.072**; affine latent-actor arm **~0.21**.

`gamma=0.99` pooled over its whole 30-cell ladder is **0.294 [0.224, 0.364]** against
`gamma=0.98`'s **0.076 [0.033, 0.119]** — **disjoint intervals, ~3.9x BC**, and above the
latent-actor arm (~0.21) that was previously the only antmaze arm off the floor. It is also
the first antmaze arm whose *late* window stands on its own: 300k-500k pooled 0.252 ± 0.100,
n=15. **Antmaze-medium is not structurally closed to this agent**; the 50-step effective
horizon of `gamma=0.98` was the obstruction, and the failure it produced — episodes running
to the 1000-step timeout with success pinned at the BC control — is exactly what a value
function blind past 50 steps does on a maze whose goal is hundreds of steps away.

### 6.2 What did NOT happen: the pre-registered signature, on both halves

§3 pre-registered "in-loop leaves the floor by ~250k **and** 500-episode success at 500k
>= 0.15 for gamma=0.995, **monotone in the horizon**". Both halves are wrong, and the way
they are wrong is the more useful result:

1. **Not monotone.** `gamma=0.995` beats `gamma=0.99` only at 50k-100k (0.492/0.519 vs
   0.377/0.305). Pooled over any window it does not: 50k-250k 0.296 vs 0.336, all-measured
   0.214 vs 0.294. Horizon 100 is enough; horizon 200 is past the useful point.
2. **`gamma=0.995` collapses, and the collapse is now dated at 500 episodes.** Its
   300k-500k window is **0.010 ± 0.008 over 6 cells** — every cell between 0.002 and 0.020,
   i.e. *below the BC control*, on all three seeds at both checkpoints (300k: 0.008 / 0.020
   / 0.018; 500k: 0.006 / 0.004 / 0.002). This was the destabilisation failure mode §3
   named in advance ("a longer discount can destabilise the TD backup ... reported as such,
   not as H2 being untestable"), and it is the single most reproducible behaviour this arm
   has ever shown: three of three seeds, two checkpoints apart, same near-zero value.
3. **A 500k-only evaluation would have got this wrong in both directions**: it would have
   read `gamma=0.995` as null (0.004) and `gamma=0.99` as marginal (0.170 ± 0.227, its
   *worst* checkpoint). The originally specified "500k plus best in-loop" protocol was
   inadequate; the 50k ladder is what makes the result readable.

**So nothing here may be quoted at a single checkpoint.** The honest forms are the
per-checkpoint table with seeds pooled and the pooled-window rows above.

### 6.3 The other findings

**Seed spread remains the dominant term**, exactly as on cube. Within `gamma=0.99`, seed 2
runs 0.43-0.62 across the whole ladder while seed 1 wanders 0.004-0.414; the per-checkpoint
across-seed std is 0.14-0.28 at every epoch. The pooled row exists for this reason.

**The in-loop 50-episode eval is well calibrated on these runs**, contrary to the caution in
§3. `gamma=0.995` in-loop at 100k read .32 / .56 / .78 against 500-episode .332 / .546 /
.678, and its in-loop 0.000 from 300k is confirmed at 0.002-0.020 over 500 episodes. The
0.40-in-loop -> 0.088-at-500-episodes over-read recorded for `gamma=0.98` is therefore a
property of an agent **sitting at the floor** — a 50-episode sample of a near-zero rate is
almost all noise — not a property of the environment. In-loop numbers are still not
reportable, but they are usable for choosing which checkpoint to spend 500 episodes on once
an arm is off the floor.

### 6.4 Provenance

* Training: SLURM 2491831-2491836. Eval: 51 COMPLETED jobs, 2491922-2491923, 2491961-2491964,
  2491984-2492016, 2492023-2492025, 2492030-2492038.
* 2491912/2491913 died at startup on another agent's in-flight `agents/psmflow.py` change
  (`KeyError: 'entropy'`), wrote no JSON, and were resubmitted as 2491922/2491923. No other
  eval failed.
* All 51 JSONs verified programmatically: 500 episodes, `restore_epoch` matching the
  filename's epoch token, `num_workers=1`.
* The generator is idempotent and partial-safe and prints a coverage line; re-running it is
  how this section was produced and how it would be extended:

```bash
.venv/bin/python tools/fig_affine_antmaze_discount_ladder.py \
    --logs $PSM_DATA/logs --exp $PSM_DATA/exp/PSMFLows
.venv/bin/python tools/make_tables.py --logs $PSM_DATA/logs
```

  It prints `gamma=0.99: 30, gamma=0.995: 21` when the ladder is complete, which it is.

### 6.5 What remains

1. **Make `discount=0.99` the antmaze setting** and decide whether it becomes the repo
   default. It is a one-line config change with a 30-cell, three-seed ladder behind it.
2. **Cube and pointmaze at `gamma=0.99`** — whether this is an antmaze-horizon fix or a
   general one. Another agent has that in flight (`tools/fig_affine_discount_ladder.py`,
   `docs/design/2026-09-07-discount-sweep.md`); cube should be unaffected (its reward is
   inside a 50-step horizon) and pointmaze should stay at zero for the reason COMPENDIUM
   4.11 records, so those are the two pre-registered negatives that would confirm the
   mechanism is horizon and not "a bigger gamma helps everything".
3. **The `gamma=0.995` collapse is a new and unusually clean object.** Three of three seeds
   go to 0.002-0.020 and stay there, which is more reproducible than anything else this
   agent does. `w_enc_spread` and the psi-spread diagnostics are logged straight through it
   (third panel of the figure) and are where to look for the mechanism — and whatever
   explains it may also explain the cube arm's ±0.4 within-run swings.
4. **The gamma=0.98 antmaze ladder has gaps** (10 of 30 cells) because that batch was never
   evaluated at every checkpoint. Filling it would make the three-arm comparison exact
   rather than pooled-over-what-exists; it is 20 evals of already-trained checkpoints.
5. **H1 closed the acting-side question** (§5). Combined with §6.1, the antmaze story is
   **"the critic was horizon-starved"**, not "the acting rule was temporally incoherent".

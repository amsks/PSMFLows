# What breaks at `discount=0.995` on antmaze between 250k and 300k

Date: 2026-09-07 · Branch `feat/inversion-integration` · Diagnosis + pre-registration.
Machine: KISSKI (SLURM, H100, `kisski-inference`, `general`).
Subject: `$PSM_DATA/exp/PSMFLows/affine_strict_antmaze_g995/sd00{0,1,2}_*` against
`affine_strict_antmaze_g99/sd00{0,1,2}_*` — the repo-default paper-strict agent
(`psi_form=affine policy_index=latent index_agg=max train_actor=false acting=gpi`),
env `antmaze-medium-navigate-singletask-v0`, identical in every key but `agent.discount`.
Code touched: none in `agents/`, `utils/`, `tools/` except the new figure generator
`tools/fig_antmaze_g995_collapse.py`.

Follows `docs/design/2026-09-06-antmaze-failure-tests.md` §6.5 item 3, which named this
object and pointed at `w_enc_spread` / the psi-spread diagnostics as where to look.

## 0. The object

500-episode ladder (from `$PSM_DATA/logs/eval500_affine*_strict_antmaze_g99*_sd*.json`;
BC control 0.072):

| epoch | g995 sd0 | sd1 | sd2 | mean | | g99 sd0 | sd1 | sd2 | mean |
|---|---|---|---|---|---|---|---|---|---|
| 50k  | 0.568 | 0.348 | 0.560 | **0.492** | | 0.442 | 0.218 | 0.470 | 0.377 |
| 100k | 0.332 | 0.546 | 0.678 | **0.519** | | 0.330 | 0.050 | 0.534 | 0.305 |
| 150k | 0.208 | 0.294 | 0.102 | 0.201 | | 0.372 | 0.162 | 0.568 | 0.367 |
| 200k | 0.192 | 0.076 | 0.158 | 0.142 | | 0.322 | 0.064 | 0.544 | 0.310 |
| 250k | 0.178 | 0.074 | 0.124 | 0.125 | | 0.230 | 0.112 | 0.624 | 0.322 |
| 300k | 0.008 | 0.020 | 0.018 | **0.015** | | 0.198 | 0.230 | 0.568 | 0.332 |
| 500k | 0.006 | 0.004 | 0.002 | **0.004** | | 0.078 | 0.004 | 0.428 | 0.170 |

The first thing to say about the framing is that the pooled windows quoted in the H2 note
(0.296 over 50k–250k, 0.010 over 300k–500k) hide the shape: `gamma=0.995` is **monotone
decreasing from 100k onward on the seed mean** (0.519 → 0.201 → 0.142 → 0.125 → 0.015).
The 250k→300k step is the last drop of a decay that started at 150k, not a cliff out of a
flat trace. `gamma=0.99` has no trend over the same window (0.305 → 0.367 → 0.310 → 0.322 →
0.332).

## 1. Step 1 — the training trace (done before any eval job was submitted)

Every scalar `train.csv` logs, all six runs, at 5k resolution. Figure:
`docs/figures/2026-09-07-antmaze-g995-collapse.png`, data
`docs/figures/2026-09-07-antmaze-g995-collapse.json`, generator
`tools/fig_antmaze_g995_collapse.py`.

Definitions, from `agents/psmflow.py::index_spread` and `utils/psm_common.py`:

* `psm_loss` = the contrastive PSM objective `0.5*mean_offdiag((M - gamma*targetM)^2) -
  P*mean(diag(M - gamma*targetM))`, `M = psi(s,u',u) phi(s')^T`. It is the TD term.
* `orth_diag` = `-mean_i ||phi_i||^2`, pinned at `-128` in every run (phi is projected to
  the sphere of radius `sqrt(z_dim)`), so `orth_offdiag` carries all the information:
  `0.5*mean_{i!=j}(phi_i . phi_j)^2`. Its **maximum is `0.5*128^2 = 8192`, reached only when
  every state's phi is the same direction** (rank-1 collapse); a random-ish basis reads ~64.
  Define the residual off-rank-1 energy `1 - orth_offdiag/8192 = 1 - E[cos^2]`.
* `psi_q_spread_rel` = std of `Q` over 64 prior ACTION latents `u` at the batch index,
  divided by `|Q|` — what the inner argmax consumes. `psi_q_index_spread_rel` is the same
  over 64 POLICY indices `u'` — what the outer GPI max consumes. `|Q|` level is recoverable
  as `psi_q_spread / psi_q_spread_rel`.

### 1.1 What moves first, and when

| event | g995 sd0 | sd1 | sd2 | g99 sd0 | sd1 | sd2 |
|---|---|---|---|---|---|---|
| `psm_loss` > 1e4 | **85k** | **90k** | **85k** | 425k | 275k | 275k |
| `psm_loss` > `ortho_coef * \|orth_loss\|` | **125k** | **125k** | **115k** | never | 400k | never |
| `orth_offdiag` > 2x baseline (phi starts collapsing) | **145k** | 140k | 140k | never | 460k | never |
| `orth_offdiag` > 0.99 x 8192 | 220k | 210k | 235k | never | never | never |
| `orth_offdiag` > 0.999 x 8192 | **250k** | **240k** | **275k** | never | never | never |

`|Q|` level (`psi_q_spread / psi_q_spread_rel`), the same six runs:

| epoch | g995 sd0 | sd1 | sd2 | g99 sd0 | sd1 | sd2 |
|---|---|---|---|---|---|---|
| 200k | 1.7e5 | 5.5e4 | 1.5e5 | 1.2e3 | 7.3e2 | 1.0e3 |
| 250k | 9.7e5 | 2.1e6 | 2.1e6 | 7.7e2 | 7.0e2 | 8.5e2 |
| 300k | 1.7e7 | 4.7e7 | 1.9e7 | 9.3e2 | 1.0e3 | 6.6e2 |
| 500k | ~1e9 | ~1e9 | ~1e9 | ~1e3 | ~2e3 | ~1e3 |

**The answer to "which scalar moves first" is `psm_loss`, and it is divergence, not drift:**
monotone multiplicative growth from ~2.5e3 at 50k to 8e17 at 500k, roughly one decade per
25k steps, on all three `gamma=0.995` seeds and on none of the `gamma=0.99` seeds except
sd1 (see §1.3). `w_enc_spread` moves *second* and only mildly (1.12 → 0.75 → back to ~0.96);
`psi_q_index_spread_rel` moves third (0.35 → 0.5 → 0.8); `psi_q_spread_rel`, the action
signal the inner argmax reads, **does not move at all** (0.004–0.01 throughout) — so the
`u`-side signal does not go flat, it is simply outgrown by the index side and by `|Q|`.

### 1.2 The causal chain the numbers support

1. `gamma=0.995` makes the contrastive backup non-contractive on this data: `psm_loss`
   leaves the `gamma=0.99` band by 85k and grows exponentially.
2. At **115–125k** `psm_loss` exceeds `ortho_coef * |orth_loss| = 1000 * 64 = 6.4e4`. The
   loss is `psm + 1000 * ortho`, so from that step on the orthonormality regulariser is a
   rounding error against the TD term, and phi is free to buy TD loss with basis collapse.
3. At **140–145k** `orth_offdiag` leaves its baseline; `phi` starts collapsing to one
   direction. **This is where the eval trace takes its largest single step down: seed-mean
   500-ep success 0.519 @100k → 0.201 @150k.**
4. Between 250k and 300k the residual off-rank-1 energy `1 - orth_offdiag/8192` falls by
   ~an order of magnitude on every seed: sd0 8.55e-4 → 5.86e-5, sd1 4.61e-4 → 3.25e-5,
   sd2 2.86e-3 → 3.73e-4. Equivalently the RMS angle between two states' phi vectors falls
   from 1.2–3.1 degrees to 0.3–1.1 degrees (§4.2). **That is the 250k→300k event: the last
   off-rank-1 energy phi has, goes.** What that costs is worked out in §5.1 — the
   pre-registered guess below, that it makes `Q` state-independent, is refuted there.

### 1.3 The within-arm replication

`gamma=0.99` **seed 1** does exactly the same thing, three times later and only just
started: `psm_loss` > 1e4 at 275k, crosses the ortho term at 400k, `orth_offdiag` leaves
baseline at 460k, and reaches 1.5e6 / `orth_loss=+223` at 500k. Its 500-episode ladder is
the one the H2 note flagged as "wanders 0.004–0.414", and the 460k onset falls exactly
between its best cell and its worst: **450k = 0.414, 500k = 0.004**, the only checkpoint
saved after `orth_offdiag` left baseline. Seeds 0 and 2 of `gamma=0.99` never cross and
never collapse (0.078 / 0.428 at 500k). So divergence-vs-not tracks collapse-vs-not **within** the
`gamma=0.99` arm as well as between arms — 4 diverging runs collapse, 2 non-diverging runs
do not.

## 2. Step 3 — what the collapsed policy does (in-loop `eval.csv`, 50 episodes)

`evaluation/xy` is `info['xy']` (the ant's maze coordinates) averaged over every step of
every eval episode, i.e. a coarse "how far into the maze did it get".

| g995 sd0 | 50k | 100k | 150k | 200k | 250k | **300k** | 350k | 500k |
|---|---|---|---|---|---|---|---|---|
| success | 0.50 | 0.32 | 0.26 | 0.22 | 0.14 | **0.00** | 0.00 | 0.00 |
| episode length | 757 | 821 | 936 | 948 | 971 | **1000** | 1000 | 1000 |
| mean visited xy | 18.5 | 15.7 | 13.5 | 13.5 | 11.1 | **3.9** | 4.5 | 7.0 |

Every `gamma=0.995` seed reads episode length exactly 1000 (the timeout) at 300k, 350k and
500k, and mean visited xy drops by ~3x at 300k. `gamma=0.99` sd2 at 300k reads 0.64 /
667 steps / xy 17.4.

## 3. Step 2 — PRE-REGISTRATION (written before the three diag jobs were submitted)

Probe: `tools/diag_gpi_selection.py` (existing, unmodified) via
`scripts/slurm/diag_gpi_selection.sbatch`, three jobs:

| cell | run | epoch | 500-ep reference |
|---|---|---|---|
| A | `affine_strict_antmaze_g995/sd000_s_2491834.0.20260906_221817` | 250000 | 0.178 (good) |
| B | same run | 300000 | 0.008 (collapsed) |
| C | `affine_strict_antmaze_g99/sd000_s_2491831.0.20260906_221812` | 300000 | 0.198 (control) |

`N_EP=20`, `N_FIXED=256` (the same 256 dataset rows and the same single `PRNGKey(12345)`
64x64 candidate roster in all three jobs, as on cube), `--time=02:00:00`.
**`N_MC=0`**: the MC section rolls a fixed `u` for `H=50` steps and scores return. On
antmaze-medium every state is unwinnable in 50 steps, so every MC state would be degenerate
by the probe's own criterion (this is the failure the cube note recorded for uniformly
drawn states) and the section would return nothing but ties. Dropping it also removes the
dominant env-stepping cost, which matters because antmaze episodes are 1000 steps against
cube's ~180 and the rollout section is therefore ~5x the cube job at the same `N_EP`.

**Two named hypotheses.**

* **H-TD (expected).** The mechanism is TD divergence plus basis collapse, and it is
  *not* the cube lottery. Predictions, in order of how strongly they discriminate:
  1. **`fixed`-state argmax-`u` concentration rises sharply at B.** With `phi` rank-one,
     `w = E_D[r phi]` is parallel to the one surviving direction and `Q(s,u,u') =
     psi(s,u,u')^T w` is the same function of `(u,u')` at every state up to a positive
     scale, so one candidate should win at most of the 256 probe states. Cube read
     0.098–0.203 at every checkpoint, good and bad. **Predict B >= 0.5, A and C <= 0.25.**
     This is the prediction that separates H-TD from the cube mechanism, and the one I
     would most expect to be wrong if I am wrong.
  2. **`|Q|` level at B is >= 10x A and >= 1e4 x C** (train.csv says 1.7e7 / 9.7e5 / 9.3e2).
     Near-certain; it is a wiring check that the probe is reading the same weights.
  3. **Rank correlation of the fixed-roster per-`u` ranking between A and B is <= 0.2**,
     i.e. at or below cube's 0.21–0.36 band — but with prediction 1 attached, so the
     verdict is "the ranking is destroyed by collapse", not "the ranking is a lottery".
     A **high** A↔B correlation (>= 0.6) with prediction 1 also firing would still be
     H-TD: a state-independent ranking is a *stable* ranking.
  4. **Not flatness (b).** Top-1 minus top-2 over the per-`u` reduction, in units of the
     roster std, stays in cube's permanent 0.30–0.45 band at all three cells. If B's gap
     collapses toward 0 the failure is flatness after all and H-TD's account is wrong.
  5. **Not edge-seeking (a).** Selected `|u|` ~ 2.8 against the roster's ~2.13 and selected
     clip fraction ~4x the roster's at **all three** cells, unchanged by the collapse —
     as on cube, a permanent property of the rule rather than a checkpoint property.
  6. **No numerical degeneracy.** Zero NaN, zero all-equal candidate sets. fp32 overflows
     near 3.4e38 and B sits at ~1e7, so "the critic overflowed" is predicted **false**;
     the collapse is representational, not arithmetic. Finding NaNs at B would refute the
     mechanism above and replace it with a plain numerical one.
  7. **Rollout success reproduces the eval500 ordering**: A ~0.1–0.3 over 20 episodes,
     B 0.00, C ~0.1–0.3.
* **H-LOTTERY (the alternative).** The 250k→300k step is the same re-randomisation cube
  shows between neighbouring checkpoints (per-`u` Spearman 0.21–0.36 for *any* pair
  including two that both work), and `gamma=0.995` merely lowered the mean of the lottery
  so every ticket now loses. Its signature: A↔B rank correlation **inside** cube's
  0.21–0.36 band, argmax-`u` concentration **unchanged** between A and B (both ~0.1–0.2),
  no state-independence, and the A↔C and B↔C correlations in the same band — i.e. nothing
  about B is special except its score.

Both can be true in part; the discriminator is prediction 1 (state-independence) and the
`|Q|` level, not the drift number alone.

---

## 4. Results — step 1 in numbers

SLURM: nothing; `train.csv` only. Generator
`tools/fig_antmaze_g995_collapse.py --logs $PSM_DATA/logs --exp $PSM_DATA/exp/PSMFLows`
(idempotent, reads only, writes only `docs/figures/`).

### 4.1 `psm_loss` diverges, and at a seed-independent rate

`log10(psm_loss)` growth from 50k to 500k:

| run | 50k | 500k | decades | steps per decade |
|---|---|---|---|---|
| g995 sd0 | 2.34e3 | 7.77e17 | 14.52 | **31.0k** |
| g995 sd1 | 2.46e3 | 1.02e18 | 14.62 | **30.8k** |
| g995 sd2 | 3.63e3 | 1.05e18 | 14.46 | **31.1k** |
| g99 sd0 | 6.4e2 | 1.19e4 | 1.27 | 355k |
| g99 sd1 | 6.9e2 | 1.48e6 | 3.33 | 135k |
| g99 sd2 | 9.1e2 | 5.25e4 | 1.76 | 255k |

Three decimal places of agreement across seeds at `gamma=0.995` — the divergence rate is a
property of the objective at that discount, not of the initialisation. Note the second
reading: **`gamma=0.99` is also divergent, ~4x more slowly.** The contrastive objective has
no visible fixed point at either discount; 0.99 simply does not get far enough in 500k steps
for the basis to notice, except on seed 1 (§1.3).

### 4.2 phi's collapse, in degrees

`orth_offdiag` as the RMS angle between two states' `phi` (`cos = sqrt(2*offdiag)/z_dim`);
`resid = 1 - orth_offdiag/8192` is the off-rank-1 energy:

| run | 150k | 200k | 250k | **300k** | 350k | 500k |
|---|---|---|---|---|---|---|
| g995 sd0 | 81.3d | 17.5d | **1.7d** | **0.4d** | 0.2d | 0.3d |
| g995 sd1 | 76.6d | 8.3d | **1.2d** | **0.3d** | 0.2d | 0.2d |
| g995 sd2 | 76.9d | 27.3d | **3.1d** | **1.1d** | 0.6d | 0.0d |
| g99 sd0-2 | 84.9d | 84.9d | 84.9d | 84.9d | 84.9d | 84.9 / 78.1 / 84.8d |

`gamma=0.99` seeds 0 and 2 hold a near-orthogonal basis (84.9 degrees, `resid` 0.992) for
the whole run. Every `gamma=0.995` seed is inside 3 degrees by 250k and inside ~1 degree by
300k. **That is the 250k->300k event: a 3-8x drop in the last off-rank-1 energy phi has.**

### 4.3 What did NOT move

* **`psi_q_spread_rel`** — the action-latent signal the inner argmax consumes — is
  0.004-0.010 at `gamma=0.995` and 0.008-0.019 at `gamma=0.99`, flat in both, with no step
  at 250k->300k. The selection signal is not flattened; it is outgrown.
* **`w_enc_spread`**, which §6.5 of the H2 note nominated, dips to 0.75 around 200k and then
  **recovers to ~0.95 while the eval score is collapsing**. On cube it falls monotonically
  1.30 -> 0.25 across a run whose scores swing up and down. It tracks neither. It is a red
  herring for both failures.
* **`orth_diag`** is pinned at -128 by construction (phi is projected to the sphere of
  radius `sqrt(z_dim)`), so it carries no information in any run.
* **`epoch_time`** is identical between arms; nothing pathological in the compute path.

### 4.4 Cube at `gamma=0.98`, for contrast (`affine_strict_cube/sd00{0,1,2}`)

| | psm_loss | orth_offdiag | \|Q\| | resid |
|---|---|---|---|---|
| cube, 50k -> 500k, all 3 seeds | 1.5e3 - 3.6e4, **no trend** | **64 throughout** | 400-1600 | **0.992 throughout** |

**Cube never diverges and its phi never collapses.** Whatever produces cube's +-0.4 swings
does so on a healthy basis at a stable `|Q|`. That already separates the two failures before
any eval-side probe.

## 5. Results — step 2, `tools/diag_gpi_selection.py`

Three jobs, SLURM **2492096 / 2492097 / 2492098**, all COMPLETED in ~10 min of the 2 h
budget (`N_EP=20`, `N_FIXED=256`, `N_MC=0`, `CAND_KEY=12345`, `PROBE_ROW_SEED=7`).
Reports `$PSM_DATA/logs/diag_gpi_selection_antmaze_{g995_sd0_250000,g995_sd0_300000,
g99_sd0_300000}.json` + npz; cross-checkpoint table
`..._antmaze_g995_sd0_compare.json`; the derived readout used below (degeneracy checks,
rollout geometry, dataset reference) is persisted as
`$PSM_DATA/logs/diag_antmaze_g995_collapse_readout.json`.

| quantity | A: g995 @250k | B: g995 @300k | C: g99 @300k | cube band (09-05) |
|---|---|---|---|---|
| 500-ep reference | 0.178 | 0.008 | 0.198 | 0.086-0.704 |
| probe rollout success (20 ep) | 0.30 | **0.00** | 0.10 | — |
| probe rollout episode length | 946 | **1000** | 964 | — |
| **`Q` mean level** | **-1.97e6** | **-3.42e7** | **-3.1e3** | -3.8e3 .. -6.4e3 |
| `Q` spread / \|mean\| (rollout) | 0.533 | 0.508 | 0.310 | 0.56-1.05 |
| top1-top2 / std (pairs) | 0.0023 | 0.0012 | 0.0038 | 0.011-0.017 |
| top1-top2 / std (per-`u`) | 0.414 | 0.393 | 0.343 | 0.31-0.43 |
| **top1-top2 / \|Q\| (per-`u`)** | **7.3e-3** | **6.3e-4** | **1.3e-3** | — |
| ensemble unc at pick / \|Q\| | 0.018 | 0.005 | 0.007 | 0.016-0.019 |
| selected \|u\| | 3.53 | 3.51 | 3.36 | 2.77-2.85 |
| candidate \|u\| (prior) | 2.74 | 2.74 | 2.74 | 2.13 |
| selected-`u` clip fraction | 0.0110 | 0.0111 | 0.0084 | 0.0108-0.0128 |
| candidate-`u` clip fraction | 0.0027 | 0.0027 | 0.0027 | 0.0028 |
| action \|a\| / clip fraction | 1.93 / 0.087 | 1.85 / 0.075 | 1.95 / 0.090 | 0.75-0.81 / 0.011-0.018 |
| persistence cos(u_t, u_t-1) | 0.255 | 0.280 | 0.119 | 0.094-0.139 |
| fixed: `Q` spread / \|mean\| | 0.409 | 0.368 | 0.348 | 0.339-0.586 |
| fixed: gap/std (per-`u`) | 0.540 | 0.515 | 0.368 | 0.312-0.431 |
| **fixed: argmax-`u` concentration** | **0.199** | **0.231** | **0.137** | **0.098-0.203** |
| fixed: index_matters (max-min over `u'`) | 3.8e6 | 7.0e7 | 5.5e3 | 5.5e3-1.3e4 |
| fixed: rank(pessimistic) vs rank(mean-Q) | 0.839 | 0.890 | 0.843 | 0.926-0.945 |
| fixed: rank(online) vs rank(target) | 0.938 | 0.892 | 0.895 | 0.915-0.960 |
| **NaN / Inf / all-tied candidate sets** | **0 / 0 / 0** | **0 / 0 / 0** | **0 / 0 / 0** | — |
| median top1-top2 in fp32 ulps of \|Q\| | 2.2e4 | 4.9e4 | 2.4e4 | — |

Cross-checkpoint drift on the shared roster, A vs B (256 states, `compare`):

| | A vs B (g995, 250k vs 300k) | cube, ANY pair of 250k/350k/400k/500k |
|---|---|---|
| Spearman, per-`u` GPI score | **0.213** | 0.21-0.36 |
| Spearman, `K x K` pair matrix | 0.780 | 0.64-0.83 |
| Spearman, target critic per-`u` | 0.255 | 0.23-0.40 |
| top-5 overlap, per-`u` | 0.219 | 0.20-0.28 |
| argmax-`u` agreement | 0.160 | 0.11-0.19 |

### 5.1 Scoring the pre-registration

| # | prediction | outcome |
|---|---|---|
| 1 | argmax-`u` concentration >= 0.5 at B (state-independence) | **REFUTED.** 0.199 -> 0.231, inside cube's 0.098-0.203 band. `Q` is still a function of the state. |
| 2 | \|Q\| at B >= 10x A and >= 1e4 x C | **CONFIRMED.** 17x A, 1.1e4 x C. Matches `train.csv`'s `psi_q_spread/_rel` to within a factor of 2, so the probe reads the same weights. |
| 3 | A<->B per-`u` Spearman <= 0.2 | **NEAR MISS, and the wrong frame.** 0.213 — the bottom of cube's band, not below it. |
| 4 | not flatness in units of the roster std | **CONFIRMED.** 0.414 -> 0.393, cube band 0.31-0.43. |
| 5 | edge-seeking unchanged and permanent | **CONFIRMED.** selected \|u\| 3.5 against a roster 2.74 at all three cells, clip fraction 4x the roster's at all three. |
| 6 | no NaN, no all-equal candidate sets | **CONFIRMED.** Zero NaN/Inf/ties; margins are 2e4-5e4 fp32 ulps. The arithmetic is fine; the collapse is representational, not numerical. |
| 7 | rollout reproduces the eval500 ordering | **CONFIRMED.** 0.30 / 0.00 / 0.10 against 0.178 / 0.008 / 0.198. |

**The prediction I flagged as most likely to be wrong is the one that was wrong.** phi being
rank-one to 0.4 degrees does *not* make `Q` state-independent, because `Q(s,u,u') =
psi(s,u,u')^T w` keeps its state dependence through `psi`'s own towers `A(s,u), beta(s,u)`,
which nothing constrains. What the collapse removes is different and worse: `phi` is the
basis in which the *future state* is represented, so a rank-one `phi` makes the measure
`m(s,u,u',x) = psi^T phi(x)` blind to **where you end up**, and makes the task vector
`w = E_D[r phi]` proportional to the single surviving direction. `Q` then still varies over
states and candidates — it just no longer varies with anything about reaching the goal.
That account fits every row of the table: normal spread, normal gap-over-std, normal
concentration, normal drift, and a policy below the BC control.

### 5.2 The one row that is not normal

`top1-top2 over |Q|` for the per-`u` ranking falls **7.3e-3 -> 6.3e-4** between A and B, an
11.6x drop, and B sits 2x below the healthy control C (1.3e-3) despite `|Q|` being 1.1e4
times larger. The margin the argmax acts on, measured against the scale of the thing being
compared, is what the divergence destroys — while the margin measured against the roster's
own std (the statistic the cube probe reported) is unchanged. **The two normalisations
disagree, and only the `|Q|`-relative one sees the failure.** That is the single new
diagnostic this probe produced.

## 6. Results — step 3, what the collapsed policy does

From the probe's own 20-episode rollouts (`step_obs[:, :2]` is the ant's maze xy), against
the dataset's own visited-state statistic (`observations[:, :2]`, 1.001e6 transitions):

| | A: g995 @250k | B: g995 @300k | C: g99 @300k | dataset |
|---|---|---|---|---|
| start (x+y)/2 | -0.05 | -0.05 | -0.05 | 9.96 (episode starts) |
| **mean visited (x+y)/2** | 8.22 | **2.98** | 8.06 | **9.80** (p5 2.12, p95 18.08) |
| net displacement \|end - start\| | 20.4 | **8.2** | 18.9 | — |
| max radius reached from start | 21.0 | **10.9** | 21.9 | — |
| **path length (per 1000 steps)** | 94.0 | **107.3** | 122.8 | — |
| mean per-step \|dxy\| | 0.099 | 0.107 | 0.127 | — |
| straightness (net / path) | 0.217 | **0.076** | 0.154 | — |
| episodes never leaving r < 10 | 0.20 | **0.55** | 0.00 | — |

**It does not freeze and it does not leave the maze — it mills.** Per-step speed at B is
within 8% of A's and 84% of C's, and its total path length is *longer* than A's: the ant
walks a normal amount. What changes is where the walking goes — net displacement falls 2.5x,
straightness falls 2.9x, and 11 of 20 episodes never get more than 10 units from the start,
against 4 of 20 at A and 0 of 20 at C. The in-loop `evaluation/xy` trace says the same for
all three seeds (3.9 / 9.1 / 8.7 at 300k against 11.1 / 10.3 / 10.8 at 250k and 18.5-18.8 at
50k), and every `gamma=0.995` seed reads episode length exactly 1000 from 300k onward.
Against the dataset's mean visited xy of 9.80, cells A and C sit at 8.1-8.2 (the behaviour
distribution) and B at 2.98 (a third of it): the collapsed policy's state distribution has
left the data.

## 7. Verdict

### 7.1 What breaks at 0.995, and when

**The contrastive TD term diverges from the first 50k steps, and the collapse of the eval
score is the delayed, monotone consequence.** In order, with the dates the six runs agree on:

1. **~50-85k** `psm_loss` leaves the `gamma=0.99` band and starts growing at one decade per
   31k steps — the first scalar to move, on all three seeds, at a rate that agrees to three
   significant figures across them.
2. **115-125k** it exceeds `ortho_coef * |orth_loss|`. The orthonormality regulariser stops
   being a constraint.
3. **140-145k** `orth_offdiag` leaves its baseline: phi starts collapsing. The seed-mean
   500-episode score takes its largest step here — 0.519 @100k -> 0.201 @150k.
4. **250k -> 300k** the last off-rank-1 energy goes: RMS angle between two states' phi
   falls from 1.2-3.1 degrees to 0.3-1.1 degrees, `|Q|` goes 1e6 -> 3e7, the per-`u` argmax
   margin relative to `|Q|` falls 12x, and success goes 0.125 -> 0.015.

The 250k->300k step is therefore the **end** of a process, not an event. The question as
posed ("what breaks between 250k and 300k") presupposes a cliff; the ladder is monotone from
100k and the pooled-window framing is what makes it look like a cliff.

### 7.2 Is the mechanism shared with cube's swings? **No — but the acting rule is.**

Two layers, and only the lower one is shared.

* **Shared (the acting rule).** Every selection statistic at the collapsed checkpoint is
  inside cube's band: argmax-`u` concentration 0.231 (cube 0.098-0.203), gap/std 0.393
  (0.31-0.43), selected-`u` clip fraction 4x the roster's, and — the number the task asks
  for — **per-`u` rank correlation between neighbouring checkpoints 0.213, against cube's
  0.21-0.36**, with top-5 overlap 0.219 (cube 0.20-0.28) and argmax agreement 0.160 (cube
  0.11-0.19). GPI over 64 prior draws is the same weakly-informed lottery on both
  environments, at every checkpoint, good and bad. It is not what changed at 300k.
* **Not shared (the cause of this collapse).** Cube's three seeds run 500k steps with
  `psm_loss` in 1.5e3-3.6e4 and `orth_offdiag` pinned at 64 — **no divergence, no basis
  collapse, `|Q|` flat at 400-1600.** Cube's swings happen on a healthy basis; they are the
  lottery resolving differently at each checkpoint, which is exactly what the 09-05 note
  concluded. antmaze at `gamma=0.995` adds a second, distinct failure underneath the
  lottery: the TD term diverges, phi goes rank-one, and the lottery is then drawn over a
  measure that cannot see where a trajectory ends up. That is why its floor is 0.004-0.020
  (*below* BC) rather than cube's 0.086 (*at* BC): a lottery over a reward-blind ranking
  with a large-norm-latent bias is worse than a uniform prior draw, the same way H1's
  `fixed_index` cells came in below BC.

So the shared quantity (rank drift ~0.21) is a property of the **rule**, present everywhere;
the discriminating quantities are `psm_loss`, `orth_offdiag`, `|Q|` and the `|Q|`-relative
margin, and on those the two failures have nothing in common. A stabiliser for this collapse
will do nothing for cube's swings, and vice versa.

### 7.3 Stabilisation candidates, ranked by the evidence for each

1. **Make the geometry regulariser scale-relative — `ortho_coef` proportional to
   `stop_grad(psm_loss)`, or normalise the TD term.** *Evidence: direct and dated.* The
   crossing of `psm_loss` past `ortho_coef*|orth_loss|` precedes phi's departure from
   baseline by 20-25k steps on all three `gamma=0.995` seeds (125k->145k, 125k->140k,
   115k->140k), happens on the one `gamma=0.99` seed that later collapses (400k->460k), and
   never happens on the two `gamma=0.99` seeds that do not. Four for four, two for two. This
   is the only candidate with a measured, dated, per-seed antecedent.
2. **Bound `psi` — the divergence lives entirely in the psi tower.** *Evidence: strong,
   structural.* `phi` is projected to the sphere (`orth_diag` pinned at -128) and `w(u')` is
   unit-norm (`norm_w`), so `A(s,u)` and `beta(s,u)` are the only unconstrained magnitudes,
   and `psi_q_spread` is what runs 7 -> 2e8. A norm penalty or projection on `psi`, or
   clipping the bootstrap `target_M`, attacks the growth at its only free direction.
   `psm_loss ~= psm_offdiag` in every diverged row (7.77e17 vs 7.77e17), i.e. the blow-up is
   in the quadratic off-diagonal residual, so a clipped target directly bounds the loss.
3. **Use `gamma=0.99` on antmaze and treat 0.995 as outside the stable region.** *Evidence:
   the strongest of any row, and the least interesting.* 30 cells, 3 seeds, 0.294 +- 0.070
   (H2 §6.1). It is a workaround, not a fix: §4.1 shows `gamma=0.99` is divergent too, ~4x
   more slowly, and its seed 1 does collapse by 500k. **If the run were 1.5M steps instead
   of 500k, `gamma=0.99` would land where `gamma=0.995` is now.** Any claim that 0.99 is
   "stable" is a claim about the step budget.
4. **Lower `lr` for phi/psi at long horizons.** *Evidence: indirect.* The instability is a
   feedback loop whose gain scales with the step size, and the seed-independence of the
   31k-steps-per-decade rate says the loop, not the noise, sets the rate — so halving the lr
   should roughly halve the rate in *wall* terms while leaving it unchanged per *unit of
   learning*. Cheap to test, but nothing measured here predicts it removes the fixed point
   problem rather than deferring it, which is the same objection as (3).
5. **Slower Polyak (`tau` < 0.01) at long horizons.** *Evidence: measured, and weak.*
   `rank(online) vs rank(target)` is 0.89-0.94 at all three probe cells and 0.92-0.96 on
   cube: the target head is already tracking the online head closely enough that it is not
   acting as an independent anchor. Slowing it changes the loop gain, but the 09-05 note
   already recorded target-critic selection as a no-op, and this probe reproduces that.
   Try it only after (1) and (2).
6. **Reward / return scaling — predicted NO-OP, do not spend a run on it.** *Evidence:
   the code.* Under the paper-strict arm (`policy_index=latent`, `train_actor=false`) the
   measure loss never sees a reward: `M = psi(s,u',u) phi(s')^T`, the bootstrap index is a
   prior draw, and `task_w` is a mixture of a Gaussian and `phi(goal)` — reward-free
   throughout. Rewards enter only at eval, through `w = E_D[r phi]`. Scaling them cannot
   change `psm_loss`'s trajectory by construction. This was on the candidate list in the
   brief and it should come off it.
7. **Anything on the acting side** (`gpi_select`, candidate-ball shrinkage, checkpoint
   averaging). *Evidence: measured no-op for this failure.* §5 shows the selection
   statistics at the collapsed checkpoint are cube's; H1 (09-06 §5) already showed pinning
   the index costs everything. These remain live for cube's swings and are simply the wrong
   layer here.

### 7.4 What would settle it, cheaply

* **Measure `w`'s degeneracy directly.** §5.1's account — reward-blindness rather than
  state-blindness — is inference from `orth_offdiag` plus the refutation of prediction 1.
  The direct test is the spectrum of `phi` over a batch and the angle between
  `w = E_D[r phi]` and the top phi singular vector, at 250k vs 300k. No new training; it is
  a few lines against the existing `infer_z` path (which this note deliberately did not
  write, having been asked not to add tools beyond the figure script).
* **One `gamma=0.995` rerun with candidate (1)** — `ortho_coef` tracking `psm_loss` — three
  seeds, 500k steps, `--time=08:00:00`. The pre-registered signature is `orth_offdiag`
  staying at ~64 while `psm_loss` still grows; if the score then holds near its 100k value
  (~0.5) the mechanism is confirmed end to end, and if it collapses anyway with a healthy
  basis then the basis was not the load-bearing part.
* **A `gamma=0.99` within-run 250k-vs-300k `diag_gpi_selection` pair**, to give §5's drift
  row a same-arm control instead of cube's. Not run here: the brief capped this at three
  jobs and the C cell was spent on the collapsed-vs-healthy contrast instead.

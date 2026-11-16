# psmgoal (joint critic + Density FB recipe) vs latent Density FB — 2026-11-09

User question: is the port of the Density FB recipe onto psmgoal correct, and what can the psmgoal
checkpoints not do that the latent Density FB checkpoints can?

Files here: `diag_critic_compare.py` (same measurements on both checkpoint kinds), `diag.sbatch`, `smoke.sh`,
`port_audit.md` (code audit of the port, written by a separate read-only pass). Outputs:
`$PSM_DATA/psmgoal-vs-dfb-2026-11-09/*.json`.

Jobs 2530235-42 (2026-11-09): psmgoal_recipe_bc03_cube seeds 0-2 at 250k and 500k; dfb_latent_oc_bc0p3_cube
seed 0 at 500k; dfb_latent_oc_bc0p1_cube seed 0 at 500k (only seed 0 of the DFB runs is on local disk).

Expected before results (from the smoke, 1 task, 16 states):
- index fit: psmgoal held-out Pearson 0.2-0.4, DFB 0.6-0.8 (the index feature is deviation 2 of the port).
- measure vs true futures: both rank the true near future above random states (percentile > 0.8 at h <= 10).
- action sensitivity of the measure (u_data minus random u): small for both (< 0.05 percentile).
- slope: psmgoal's actor gains about zero critic score over the BC latent (in units of the within-state
  spread); DFB's actor gains a clearly positive amount.

## Results (2026-11-09, jobs 2530235-42, five-task means; table in $PSM_DATA/psmgoal-vs-dfb-2026-11-09/table.md)

| measurement | DFB bc0.1 sd0 500k | DFB bc0.3 sd0 500k | psmgoal recipe bc0.3, 3 seeds 500k |
|---|---|---|---|
| index reward view vs task reward, held-out Pearson | 0.77 | 0.73 | 0.28-0.29 |
| true future at h=1, percentile among 256 random states | 0.99 | 0.99 | 1.00 |
| same at h=10 | 0.98 | 0.98 | 0.77-0.78 |
| same at h=50 | 0.71 | 0.72 | 0.59-0.60 |
| logit std over the random pool | 6.0 | 3.8 | 0.42-0.52 |
| measure's reaction to u (h=1 percentile, data u minus random u) | 0.0004 | 0.0006 | 0.018-0.029 |
| within-state / across-state score std | 0.09 | 0.11 | 0.08 |
| pooled Spearman, score vs one-step cube progress | 0.02 | 0.03 | 0.10-0.11 |
| gradient norm / within-state std | 1.05 | 1.08 | 1.13-1.19 |
| actor's score gain over its BC latent, in within-state stds | 0.18 | 0.10 | 0.02-0.03 |
| actor distance from BC latent | 1.00 | 0.78 | 0.58-0.62 |

Against expectations: index fit as expected; measure action sensitivity was NOT small for psmgoal (larger than DFB);
the actor gain is as expected. Port audit (port_audit.md): no undocumented code difference; index feature has
1-3 effective directions vs 21-30 in DFB; phi effective rank 1.00.

## Factorised vs joint, base recipe (reward index, flowbc delta actor, fb_bc_coeff 3), 3 seeds each at 500k
Jobs 2530271-6; table in $PSM_DATA/psmgoal-vs-dfb-2026-11-09/table_base.md. Score = the actor's own value
(softmax over 256 fixed relabel next states, weighted by the reward view at the task w).

| measurement | factorised (sd0-2) | joint (sd3-5) |
|---|---|---|
| index reward view vs task reward, held-out Pearson | 0.56-0.57 | 0.31-0.33 |
| true future at h=50, percentile among random states | 0.94 | 0.72-0.74 |
| logit std over the random pool | 2.7-2.8 | 0.31-0.50 |
| pooled Spearman, score vs one-step cube progress | 0.09-0.13 | -0.01-0.06 |
| actor distance from BC latent | 0.017-0.020 | 0.004-0.005 |
| actor's score gain over BC, in within-state stds | 0.011-0.012 | ~0 |

## Factorised critic with a weak anchor (launched 2026-11-09)
Arms (arms_weak.tsv), each = the ladder's factorized arm (reward index, softmax loss, one head) plus:
- fwk03: delta actor, fb_bc_coeff 0.3 (ladder had 3)
- fwk01: delta actor, fb_bc_coeff 0.1
- fvf01: Density FB's own actor (flowbc_vf, own vector field, bc 0.1, q-centred, pool-softmax readout)
3 seeds x 500k, cube-single, checkpoints 250k/500k.
Expected: if the critic form was what held the actor at BC, the actor alone rises well above the ladder's
factorised 0.157 (0.4 or more). If it stays at 0.10-0.20, the factorised psmgoal critic still lacks what Density
FB's has (z as an input to F, two heads, LayerNorm, B without the sphere projection).

## Result (2026-11-09 17:00Z): five-task, 500 episodes per task, 3 seeds, 95% CI = 4.303 SE, BC 0.111
```
fwk03 250000 actor 0.609 ± 0.262 [0.526 0.575 0.728] tasks [0.74 0.67 0.85 0.47 0.32]
fwk03 250000 bon 0.311 ± 0.244 [0.202 0.338 0.392] tasks [0.45 0.32 0.41 0.13 0.25]
fwk03 500000 actor 0.530 ± 0.173 [0.482 0.498 0.61 ] tasks [0.5  0.55 0.79 0.49 0.32]
fwk03 500000 bon 0.261 ± 0.068 [0.273 0.23  0.28 ] tasks [0.16 0.33 0.42 0.24 0.14]
fwk01 250000 actor 0.636 ± 0.051 [0.62  0.63  0.659] tasks [0.84 0.69 0.86 0.56 0.23]
fwk01 250000 bon 0.255 ± 0.098 [0.268 0.285 0.21 ] tasks [0.29 0.32 0.34 0.16 0.16]
fwk01 500000 actor 0.678 ± 0.128 [0.62  0.696 0.718] tasks [0.63 0.79 0.86 0.62 0.5 ]
fwk01 500000 bon 0.234 ± 0.098 [0.228 0.198 0.276] tasks [0.22 0.28 0.26 0.2  0.22]
fvf01 250000 actor 0.674 ± 0.164 [0.611 0.67  0.742] tasks [0.78 0.79 0.8  0.57 0.43]
fvf01 250000 bon 0.184 ± 0.201 [0.231 0.091 0.231] tasks [0.16 0.22 0.19 0.2  0.14]
fvf01 500000 actor 0.712 ± 0.131 [0.763 0.658 0.716] tasks [0.79 0.78 0.86 0.66 0.47]
fvf01 500000 bon 0.271 ± 0.290 [0.354 0.137 0.32 ] tasks [0.34 0.27 0.33 0.19 0.23]
```
Reading: the factorised critic at a weak anchor lifts the actor from 0.157 (anchor 3) to 0.68-0.71 at 500k; the joint critic at the same anchors stays at 0.11-0.13; latent Density FB 0.95. Best-of-64 falls to 0.23-0.27.

## psmz on cube-double and scene (launched 2026-11-10)
`agent=psmz` (the 0.712 config) + the env's training-reward block spec (cube-double both cubes
`19:22:0.4,28:31:0.4`; scene five blocks from Factored-FB scene_play.yaml); discount kept at 0.98; 1M steps,
checkpoints 250k/500k/750k/1M; jobs 2530604-9.
Expected (psmz feeds the full state to f, so the matching Density FB rows are its full-state ones): cube-double
actor 0.03-0.15 (latent DFB full-state 0.05, BC 0.019); scene graded index 0.3-0.45 (latent DFB full-state 0.45).

## psmz cube-single arms (launched 2026-11-10), 3 seeds x 500k, each = psmz + one change
- vgoals: Density FB future-state mix (0.2 current / 0.5 geometric trajectory at 0.99 / 0.3 random)
- b1024: batch 1024 (psmz 256)
Expected: each within the seed CI of psmz's 0.712 unless it is one of the things separating psmz from latent
Density FB's 0.84 (same full-state setting); a real effect would put the actor near 0.8.

### Result 2026-11-11 01:20Z (cube-single, five tasks x 500 episodes, 3 seeds, 95% CI = 4.303 SE; psmz 0.712 at 500k)
```
psmz_vgoals_cube 250000 actor: 0.591 ± 0.041 seeds [0.575 0.608 0.591] tasks [0.89 0.74 0.46 0.46 0.41]
psmz_vgoals_cube 250000 bon: 0.152 ± 0.072 seeds [0.119 0.172 0.165] tasks [0.3  0.08 0.07 0.05 0.26]
psmz_vgoals_cube 500000 actor: 0.774 ± 0.132 seeds [0.835 0.736 0.752] tasks [0.88 0.89 0.89 0.71 0.51]
psmz_vgoals_cube 500000 bon: 0.220 ± 0.111 seeds [0.242 0.25  0.169] tasks [0.33 0.38 0.1  0.09 0.2 ]
psmz_b1024_cube 250000 actor: 0.883 ± 0.056 seeds [0.87  0.87  0.909] tasks [0.92 0.97 0.96 0.8  0.76]
psmz_b1024_cube 250000 bon: 0.429 ± 0.183 seeds [0.493 0.348 0.444] tasks [0.43 0.44 0.37 0.31 0.58]
psmz_b1024_cube 500000 actor: 0.884 ± 0.025 seeds [0.893 0.884 0.873] tasks [0.9  0.96 0.96 0.86 0.74]
psmz_b1024_cube 500000 bon: 0.396 ± 0.066 seeds [0.391 0.372 0.424] tasks [0.36 0.46 0.4  0.31 0.46]
```
Reading: batch 1024 lifts the psmz actor to 0.884 ± 0.025 at 500k (latent Density FB full state 0.84, coordinates 0.95); the goal mix gives 0.774 ± 0.132, inside the psmz CI.

### psmz cube-double / scene result (2026-11-12; batch 256, discount 0.98; five tasks x 500 episodes, 3 seeds)
| env | 500k actor | 1M actor | 1M best-of-64 | latent DFB full-state B (actor) |
|---|---|---|---|---|
| cube-double (BC 0.019) | 0.013 ± 0.010 | 0.011 ± 0.009 (tasks 0.05/0/0/0/0) | 0.001 ± 0.003 | 0.05 |
| scene, graded index | 0.280 ± 0.117 | 0.248 ± 0.050 (tasks 0.68/0.46/0.07/0.03/0) | 0.007 ± 0.027 | 0.45 |
Against expectation (cube-double 0.03-0.15, scene 0.3-0.45): both below. These ran before the batch-1024 result.

## Why cube-double fails (2026-11-12). Arms launched (cube-double, 3 seeds x 1M, psmz + ...):
- hitblocks: goal_input=hit_blocks (both cubes' xyz into f, as latent DFB's coordinate B)
- b1024g99: batch 1024, discount 0.99 (DFB's)
- hb_b1024g99: both
Expected: hitblocks is the decisive one (latent DFB: coords 0.52, full state 0.05); if hitblocks alone stays at BC
while hb_b1024g99 rises, the batch/discount is needed as well; if all stay at BC, the cause is elsewhere (index
fit on a 10k relabel batch with very few rewarding rows; see diag).

### Cube-double diagnostics (2026-11-12, jobs 2531230-7; full table $PSM_DATA/psmgoal-vs-dfb-2026-11-09/table_cubedouble.md)
Five-task means, 256 states x 64 draws. Success in brackets.
| measurement | DFB coords 1M [0.52] | DFB full state 500k, 3 sd [0.05] | psmz 1M, 3 sd [0.011] | cube-single psmz b1024 500k [0.88] |
|---|---|---|---|---|
| rewarding states per 10k relabel | 7.6 | 7.6 | 7.6 | 227 |
| training positives per 256-batch (goal 5 / 25 steps ahead) | 0.76 / 0.51 | same | same | 4.0 / 3.8 |
| index reward view vs task reward, held-out Pearson | 0.64 | 0.40 | 0.45-0.46 | 0.65 |
| true future at h=50, percentile among random states | 0.98 | 0.99 | 1.00 | 0.95 |
| actor's score gain over BC latent, in within-state stds | 0.18 | 0.06-0.09 | 0.01-0.04 | 0.25 |
| actor distance from BC latent | 1.09 | 0.91 | 0.69-0.73 | 1.09 |
| actor one-step cube progress (BC about -0.0005) | +0.0012 | +0.0006 | -0.0003 to -0.0007 | +0.0014 |
Reading: the measure is not the problem on cube-double (true future ranked at 0.98-1.00 even 50 steps ahead). The
training reward is 30x sparser than on cube-single (0.5-0.8 positives per batch, i.e. the goal itself). The index
fit and the actor's gain track success: coords 0.64 / 0.18, full state 0.40 / 0.07, psmz 0.45 / 0.03. psmz's actor
moves the cubes away from the goal at one step; DFB's moves them toward it.

## Toward >0.3 on cube-double with the FULL STATE into the future-state network (user target, 2026-11-13)
Best known full-state numbers: raw DFB 0.16, latent DFB 0.05, psmz 0.011. Diagnostics: the task-vector fit is the
lever and it is starved (8 rewarding states per 10k relabel; <1 other positive per training batch).
Step 1 (eval-only, existing checkpoints): fit the task vector on 1,000,000 dataset rewards (~760 positives) instead
of 10,000. psmz_cubedouble 1M sd0-2 and dfb_latent_stateB_cubek_bc0p1_cubedouble 500k sd0-2, actor alone, five
tasks x 500 episodes. Expected: if the fit was the limit, success rises well above 0.011 / 0.05; if the features
cannot carry the two cubes, it stays flat.
Step 2 (training side, after step 1 and the b1024g99 arm): more positives per batch (batch 1024 running; a graded
cube-distance kernel instead of the 4 cm hit), the task vector as input to the (s,u) network, two heads.

### hitblocks 250k (2026-11-13): actor alone 0.150 ± 0.038 (0.167 / 0.138 / 0.144); tasks 0.54 / 0.02 / 0.05 / 0.02 / 0.13
psmz full state at 250k-1M: 0.011-0.013; latent DFB coords at 250k: 0.46. 500k / 1M evals queued.
hitblocks 500k: 0.085 ± 0.035 (0.101 / 0.082 / 0.074), tasks 0.28 / 0.03 / 0.03 / 0.01 / 0.07; 1M: 0.101 ± 0.075 (0.104 / 0.069 / 0.129),
tasks 0.29 / 0.04 / 0.05 / 0.01 / 0.12; 1M best-of-64 0.007. Peaks at 250k (0.150) and falls, as the DFB coordinate critic
does at anchor 0.3 on cube-double. b1024g99 250k (seeds 1-2): 0.063 ± 0.021, tasks 0.25 / 0.03 / 0.01 / 0 / 0.02.

### 2026-11-13 18:40Z, cube-double, actor alone (seeds 1-2 for the batch arms; seed 0 died, seed 3 still training)
| arm | 250k | 500k | 1M | 1M best-of-64 |
|---|---|---|---|---|
| hitblocks (coords into f) | 0.150 ± 0.038 | 0.085 ± 0.035 | 0.101 ± 0.075 | 0.007 |
| b1024g99 (full state) | 0.063 ± 0.021 | 0.057 ± 0.009 | 0.015 ± 0.008 | 0.000 |
| hb_b1024g99 (both) | queued | 0.168 ± 0.212 (0.218 / 0.119; tasks 0.45 / 0.11 / 0.14 / 0.02 / 0.11) | 0.081 ± 0.050 | 0.005 |
| latent DFB full state, 1M-row task-vector fit (eval-only, 500k, 3 seeds) | — | 0.062 ± 0.026 (10k-row fit: 0.05) | — | — |
Every arm peaks early and falls by 1M; task 1 carries nearly all of it. The 1M-row fit does not move the DFB
full-state critic (0.05 → 0.06). psmz 1M-row-fit evals: see slurm logs (no JSON written).

### Cube-double, final (2026-11-15)
| cube-double, actor alone, 3 seeds (1-3 for the batch arms), 95% CI = 4.303·SE | 250k | 500k | 1M | 1M best-of-64 |
|---|---|---|---|---|
| psmz (full state, batch 256, discount 0.98) | — | 0.013 ± 0.010 | 0.011 ± 0.009 | 0.001 |
| + both cubes' xyz into f (hitblocks) | 0.150 ± 0.038 | 0.085 ± 0.035 | 0.101 ± 0.075 | 0.007 |
| + batch 1024, discount 0.99 (b1024g99) | 0.055 ± 0.035 | 0.051 ± 0.025 | 0.016 ± 0.007 | 0.000 |
| + both (hb_b1024g99) | 0.180 ± 0.074 (0.214 / 0.158 / 0.168) | 0.154 ± 0.137 | 0.081 ± 0.029 | 0.004 |
| psmz, task vector fitted on 100k rows instead of 10k (eval-only, 1M) | — | — | 0.013 ± 0.023 | — |
| latent DFB full state, fitted on 1M rows (eval-only, 500k) | — | 0.062 ± 0.026 (10k: 0.05) | — | — |
| latent DFB, coords into B (reference) | 0.46 | 0.50 | 0.52 | ~0.02 |
Per task, hb_b1024g99 250k: 0.53 / 0.10 / 0.12 / 0.02 / 0.13. BC 0.019.
Reading: every psmz arm peaks at 250k and falls by 1M; task 1 carries nearly all of it; best-of-64 at 0 everywhere. The two changes that closed cube-single (coords, batch/discount) give 0.18 at best on cube-double against DFB's 0.46-0.52. More reward samples in the test-time fit move nothing (psmz 0.011 -> 0.013, DFB 0.05 -> 0.06). Left: the task vector as an input to the (s,u) network, two heads, layer norm; a tighter anchor against the post-250k decay.

### Decay diagnostics on cube-double (2026-11-16, jobs on hb_b1024g99 sd1-3 at 250k / 1M, hitblocks sd0-2 at 250k, DFB coords sd0 at 250k / 1M; table_cubedouble_decay.md)
| metric | hb 250k | hb 1M | hitblocks 250k | dfb coords 250k | dfb coords 1M |
|---|---|---|---|---|---|
| index_heldout.pearson | 0.668 | 0.729 | 0.641 | 0.618 | 0.64 |
| measure.h50.pct_true_future_u_data | 0.963 | 0.968 | 0.975 | 0.961 | 0.976 |
| measure.logit_std_over_pool | 2.91 | 2.76 | 2.49 | 4.37 | 4.7 |
| ranking.within_over_across | 0.0542 | 0.0455 | 0.0441 | 0.0472 | 0.0439 |
| ranking.pooled_spearman_q_vs_progress_cube | 0.0267 | 0.00856 | 0.0798 | 0.0628 | 0.0476 |
| slope.actor_minus_bc_q_over_spread | 0.12 | 0.0766 | 0.159 | 0.252 | 0.177 |
| slope.actor_shift | 0.8 | 0.97 | 0.494 | 0.928 | 1.09 |
| slope.actor_percentile | 0.527 | 0.519 | 0.537 | 0.558 | 0.525 |
| slope.actor_progress_cube | 0.000208 | 0.000182 | 0.000404 | 0.00146 | 0.00124 |
| slope.bc_progress_cube | -0.000582 | -0.000336 | -0.000598 | -0.000529 | -0.000613 |
cd_hb_sd1_250000.json {'1': (0.67, 0.82, 2.6), '2': (0.69, 0.79, -0.11), '3': (0.64, 0.81, 1.33), '4': (0.67, 0.76, -2.33), '5': (0.66, 0.76, -1.0)}
cd_hb_sd1_1000000.json {'1': (0.75, 0.99, 2.25), '2': (0.75, 0.95, 0.59), '3': (0.71, 0.98, 0.25), '4': (0.72, 0.93, -2.03), '5': (0.73, 0.97, -0.88)}
Reading: with coordinates in f the task-vector fit is as good as DFB's (0.67-0.73 vs 0.62-0.64) and rises from 250k to 1M while success falls; the measure is fine; the actor moves as far as DFB's (0.8-0.97 vs 0.93-1.09) but gains less critic score (0.12 -> 0.08 vs 0.25 -> 0.18) and its one-step cube movement is ~0 on average (DFB +0.0013): toward the goal on tasks 1 and 3, away on tasks 4 and 5 at both checkpoints. The ridge-whitening (cube 2 ignored) hypothesis is not supported by the fit numbers; the gap sits in the critic's ranking of actions at a state.

### Cube sensitivity (2026-11-16, diag_cube_sensitivity.py; 256 states x 64 draws, five-task means; $PSM_DATA/psmgoal-vs-dfb-2026-11-09/table_cube_sensitivity.md)
| metric | hb_sd1_250k | hb_sd2_250k | hb_sd1_1M | hitblocks_sd0_250k | psmz_fullstate_sd0_1M | dfb_coords_sd0_250k | dfb_coords_sd0_1M | dfb_fullstate_sd0_500k |
|---|---|---|---|---|---|---|---|---|
| reward_view.dkg_cube1_frac1.0 | 3.46 | 3.43 | 3.95 | 3.28 | 1.96 | 3.22 | 3.25 | 1.76 |
| reward_view.dkg_cube2_frac1.0 | 2.49 | 2.51 | 2.83 | 2.47 | 1.57 | 2.22 | 2.38 | 1.34 |
| ranking.spearman_q_vs_progress_cube1 | 0.034 | 0.0301 | 0.00287 | 0.0164 | 0.0445 | 0.00629 | 0.0396 | 0.047 |
| ranking.spearman_q_vs_progress_cube2 | 0.0303 | 0.0288 | 0.0187 | 0.0606 | 0.0401 | 0.0483 | 0.0209 | 0.0599 |
| ranking.q_moved_any_minus_still | 0.536 | 1.02 | 0.235 | 0.168 | 0.145 | 0.544 | 0.073 | -0.00924 |
| ranking.spearman_q_vs_total_displacement | 0.0138 | 0.0125 | 0.00411 | -0.0142 | 0.00449 | -0.0384 | -0.014 | -0.0145 |
| actor.actor_progress_cube1 | 0.000702 | 0.000608 | 0.00044 | 0.000797 | 0.000265 | 0.00125 | 0.00133 | 0.000782 |
| actor.actor_progress_cube2 | 0.00132 | 0.00133 | 0.001 | 0.00148 | 0.00093 | 0.00209 | 0.00211 | 0.00165 |
| actor.rand_progress_cube1 | 0.000108 | 0.000108 | 0.000108 | 0.000108 | 0.000108 | 0.000108 | 0.000108 | 0.000108 |
| actor.rand_progress_cube2 | 0.000826 | 0.000826 | 0.000826 | 0.000826 | 0.000826 | 0.000826 | 0.000826 | 0.000826 |
Reading: (1) the reward view responds to both cubes on every coordinate critic, psmz and DFB alike (cube 1 3.2-4.0 stds, cube 2 2.2-2.8); the full-state critics respond about half as much; no cube-2 blindness, no psmz/DFB difference. (2) the critic's one-step action ranking vs per-cube progress is 0.00-0.06 for every critic. (3) no critic prefers stillness: draws that move a cube score 0.1-1.0 within-state stds ABOVE draws that keep both still (DFB 1M and DFB full state ~0). The actor's one-step progress is the only separating row: DFB coords +0.0013 / +0.0021 per cube, psmz coords +0.0004-0.0008 / +0.0010-0.0015, with the same fraction of steps moving a cube (0.25 / 0.30, set by the flow). The three one-step suspects are cleared; the difference must be in multi-step credit, which a one-step probe cannot see.

## HQL (ryanxhr.github.io/hql) read 2026-11-16 (subagent summary)
Critic = scalar Q, 2-net ensemble; policy = BC flow sampler, acting and bootstrap = best of N=4 flow samples under Q.
Target: windows of 16 logged transitions walked backward; the logged action's downstream multi-step correction is
carried only when Q_target(s,a_logged) >= max over N policy samples (hard gate), else one-step. Cube-double offline:
HQL 67, chunked-HQL 73, LQL 45, QC 28 (online 87-100). Reason given for cube-double hardness: 500-step horizon x
suboptimal logged actions (one-step TD error grows ~sqrt(H); multi-step adds the negative advantage of spanned
actions). No hierarchy / goal representation / dwell content. Portable, ranked: (1) gated multi-step measure target;
(2) same small-N best-of-N policy for gate, bootstrap and acting; (3) 8-action chunks (new flow); (4) dense window
targets (small effect).

## Fix arms launched 2026-11-16 00:1xZ (psmz_fix_*, seeds 1-3; seed 0's relabel draw is empty on cube-double)
Code: `critic_heads>1` now wired for measure_form=factorized (independent heads, min at every scoring);
`factorized_index_input=true` feeds w into the A(s,u)/beta(s,u) trunk (Density FB's F(s,u,z)). Golden tests
bit-identical (13 pass), new tests 5 pass, 200-step smokes OK for all six arms.
| arm | base | change |
|---|---|---|
| cd_heads2 / cd_windex / cd_both | hb_b1024g99 (coords into f, batch 1024, discount 0.99), 1M | two heads / w input / both |
| cs_heads2 / cs_windex / cs_both | psmz b1024 (0.884), 500k | two heads / w input / both |
Expected: if the actor exploiting a single critic is the fault, cd_heads2 removes the post-250k decay and ends above
0.18 (DFB coords 0.46-0.52 if that was the whole gap); cd_windex tests the task-vector-input hypothesis; on
cube-single each should stay >= 0.88 (a drop means the change costs something). HQL idea 2 (bootstrap and act with
best-of-N=4 prior draws under the critic) is the next arm if these do not move.
- cd_bon4h2 (00:4xZ): + critic_heads=2 + bootstrap_bon_n=4 (HQL idea 2: the measure bootstraps through the best of 4 prior draws under the critic's reward readout at the target basis, instead of the actor). Code: bon_bootstrap / _select_u_next; tests/test_psmgoal_bootstrap_bon.py 6 pass; goldens intact.

### Fix arms, first results (2026-11-16 08:00Z; BC cube-double 0.019, latent DFB coords 0.46-0.52)
| actor alone, 3 seeds, 95% CI = 4.303·SE | 250k | 500k | 750k | 1M |
|---|---|---|---|---|
| cube-double base (coords, batch 1024, disc 0.99) | 0.180 ± 0.074 | 0.154 ± 0.137 | — | 0.081 ± 0.029 |
| + two heads (cd_heads2) | 0.222 ± 0.116 | 0.202 ± 0.140 | | |
| + w into the (s,u) net (cd_windex) | 0.294 ± 0.097 | 0.241 ± 0.028 | 0.140 ± 0.028 | 0.108 ± 0.081 |
| + both (cd_both) | **0.319 ± 0.027** (0.318 / 0.331 / 0.310; tasks 0.71 / 0.30 / 0.35 / 0.07 / 0.17) | 0.303 ± 0.143 (0.346 / 0.324 / 0.237) | | |
| + two heads + best-of-4 bootstrap (cd_bon4h2) | 0.035 ± 0.017 | | | |
| cube-single psmz b1024 (reference 0.884) + two heads | 0.897 ± 0.047 | 0.897 ± 0.050 | | |
| cube-single + w input | 0.877 ± 0.014 | 0.864 ± 0.055 | | |
| cube-single + both | 0.888 ± 0.060 | 0.880 ± 0.020 | | |
Reading: the task-vector input is the largest single change (0.18 -> 0.29 at 250k); two heads add ~0.03-0.04 and slow the decay (both: 0.30 at 500k vs windex 0.24); windex alone still decays to 0.11 by 1M. Best-of-4 bootstrap (HQL idea 2) hurts (0.035). Cube-single unchanged at 0.86-0.90 by any of the three.

## Normalized (Density) FB, full state, reward-sample index — what we hold (2026-11-16, subagent collation; scratch $CLAUDE_JOB/tmp/fbfs)
Raw density_fb_z_state (anchor 3, cube kernel, 1M): cube-single 0.909 (paper protocol, 10 eps/task), kisski rerun 0.863;
cube-double 0.158 ± 0.038 (permuting env). Raw at anchor 0.1 collapses (0.003-0.008). Latent full-state B + cube kernel,
anchor 0.1: cube-single 0.844 (500k), cube-double 0.052 (500k). Config = density_fb_z.yaml + backward_input=state (lr 3e-4,
batch 1024, 3x512 LayerNorm, disc 0.99, tau 0.005, z_dim 50, z_rand_frac 0.5, ridge 0.01 / floor 1e-6, enc tau 5e-4,
sigma tau 5e-3, pool 256, goal mix 0.2/0.5/0.3, actor bc 3.0). Diagnosis on record: held-out R^2 of the reward on B(s+)
0.29 (full state) vs 0.47-0.60 (coords), identical raw vs latent; 85 % of the full-state distance is the arm.
Missing: 500-episode Route B of raw full state at anchor 3; raw full state on the non-permuting cube-double env.

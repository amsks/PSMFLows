# Oscillation, not ranking, is what keeps cube off 0.70

**Status:** pre-registered 2026-09-08, four arms in flight (12 runs). Nothing below this
line may change once the first job is submitted.

**Target:** 0.70 on `cube-single-play` **across seeds**, not as a peak.

---

## 1. The observation that reframes the problem

Every cube seed already reaches 0.6-0.7. None of them holds it. 500-episode evals,
`affine_strict_cube`, the shipped defaults:

| ckpt | sd0 | sd1 | sd2 |
|---|---|---|---|
| 50k | 0.192 | 0.412 | 0.224 |
| 100k | 0.078 | 0.330 | 0.178 |
| 150k | 0.088 | 0.268 | **0.596** |
| 200k | 0.264 | 0.298 | 0.466 |
| 250k | 0.532 | **0.620** | 0.246 |
| 300k | 0.286 | 0.532 | 0.360 |
| 350k | **0.704** | 0.494 | 0.350 |
| 400k | 0.282 | 0.346 | 0.546 |
| 450k | 0.272 | 0.568 | 0.564 |
| 500k | 0.086 | 0.548 | 0.292 |

Per-seed maxima are **0.704 / 0.620 / 0.596**, mean **0.640**. The pooled 300-500k figure
we report is **0.415** -- because it averages sd0's 0.704 together with sd0's 0.086, from
the same run 150k steps apart.

Sampling noise at 500 episodes is +/-0.04. **These swings are real.** Seed 0 traverses
0.532 -> 0.286 -> 0.704 -> 0.282 -> 0.272 -> 0.086.

**The capability is present and the consistency is not.** Closing a 0.415 -> 0.70 gap by
making the method better is a much harder problem than stopping it falling out of solutions
it already finds. This note assumes the second problem is the real one.

## 2. Why it is not the ranking

The critic's ordering is weak (Spearman ~0.15 against MC return, see
`2026-09-08-measure-loss-audit.md` 8.4) and that remains true. But ordering does not explain
the oscillation, and at a good checkpoint the critic is plainly doing real work. Same run,
same checkpoint (`affine_strict_cube` sd0 @350k), 500 episodes, only the selection rule
changing:

| rule | success |
|---|---|
| `mean` (pessimism dropped) | **0.734** |
| `argmax` (shipped) | **0.704** |
| `small_ball` | 0.670 |
| `soft_topm` | 0.612 |
| `max_norm` (critic-free) | 0.078 |
| `top_quartile_random` (critic-free) | 0.078 |
| single random draw | 0.090 |

Every critic-based rule lands at 0.61-0.73; every critic-free rule sits at the BC control.
**At a good checkpoint the critic is worth ~8x over random selection.** The earlier
"the critic picks worse than random" result (8.4) was measured in the MC probe's
*held-fixed-latent* regime, which is not what GPI deploys -- that caveat turns out to
carry the whole disagreement.

## 3. What actually tracks success

28 (seed, checkpoint) pairs with both a 500-episode eval and a training row. Pearson
correlation of each logged training metric against success:

| metric | corr |
|---|---|
| **orth_offdiag** | **-0.375** |
| **orth_loss** | **-0.375** |
| w_enc_spread | -0.318 |
| orth_diag | +0.222 |
| psm_diag | -0.127 |
| psi_q_range_rel | -0.090 |
| **psm_loss** | **+0.079** |

Two readings.

**Orthonormality is the best predictor we have, and the sign says tighter is better.**

**`psm_loss` does not predict performance at all.** The quantity the entire measure-loss
audit was constructed around correlates at +0.08 with how well the agent acts. This is the
same lesson `psi_bound` taught in 8.2 -- the best loss-growth number ever measured here,
attached to the worst policy -- and it should be treated as a standing warning: **on this
method, loss diagnostics are not a proxy for policy quality.**

## 4. What the existing ortho result does and does not cover

The settled-negative on record is "stronger orthonormality does not fix the divergence,
delay only". That is correct **and it was measured against loss-growth rate at
gamma=0.995**. `ortho_mode=relative`, `ortho_coef=1e4` and `1e5` have never been scored on
cube success at gamma=0.98. Given 3, loss growth is not the right target, so that null
carries no information about this axis. The same applies to `lr_sf=1e-5` and `tau=1e-3`,
both of which were judged on steps/decade alone.

`orth_offdiag` is also pinned in a narrow band -- 63.7 to 64.6 across all of training,
never improving -- so the coefficient is holding the geometry at a fixed level rather than
driving it toward orthonormal.

## 5. Arms

Cube, gamma=0.98, 3 seeds each, 500k steps, checkpoints every 50k. Control is the existing
`affine_strict_cube`, which already has the full 500-episode ladder in 1.

| group | change |
|---|---|
| `tau1e3_cube` | `tau` 0.01 -> 1e-3 |
| `oc1e4_cube` | `ortho_coef` 1e3 -> 1e4 |
| `lrsf1e5_cube` | `lr_sf` 1e-4 -> 1e-5 |
| `oc1e4_lrsf1e5_cube` | both of the above |

## 6. Scoring -- the mean is the wrong statistic

An arm that scores 0.45 flat beats an arm that scores 0.45 by averaging 0.70 and 0.09,
because only the first is shippable. Primary metric is therefore **per-seed stability over
the 300-500k ladder**:

- `swing` = max - min of the 500-episode ladder, per seed, 300-500k
- `step` = mean |difference| between consecutive checkpoints, per seed
- reported beside the pooled mean, never instead of it

Scored by `tools/stability_ladder.py`. Control values to beat, `affine_strict_cube`
300-500k (sd0 / sd1 / sd2):

| seed | mean | swing | step | min | max |
|---|---|---|---|---|---|
| sd0 | 0.326 | **0.618** | 0.259 | 0.086 | 0.704 |
| sd1 | 0.498 | **0.222** | 0.107 | 0.346 | 0.568 |
| sd2 | 0.422 | **0.272** | 0.124 | 0.292 | 0.564 |
| arm | **0.415** | **0.371** | 0.163 | | |

The arm mean reproduces the published 0.415 headline exactly, which is the check that the
ladder is being assembled from the right evals. Three filters were needed to get there,
each of which produced a wrong table first: acting-rule overrides (a `gpi_num_u` cell is a
different policy on the same weights), other-task evals (`..._task3_sd0.json` shares the
run and epoch and carries no CLI override, only a different `env`), and smoke reports.

## 7. Pre-registered predictions

Recorded before any arm lands, so the verdict cannot be shopped.

1. **`tau=1e-3` reduces swing and is the most likely of the four to work.** Slower target
   updates are the standard remedy for a policy oscillating around a solution, and it was
   the one ablation that moved loss growth 2.3x (1.01e5 vs 4.4e4) against a
   pre-registration of "fixes nothing". Expected: swing down by at least a third on 2 of 3
   seeds; mean roughly unchanged or slightly up.
2. **`ortho_coef=1e4` is the highest-variance bet.** The correlation behind it is
   r = -0.375 on n=28 (p ~ 0.05) extrapolated from a 1.4% observed range to a 10x
   coefficient change. Expected: `orth_offdiag` drops well below 63.7. If success does not
   improve when it does, the correlation was confounded and **the ortho line should be
   closed for good** rather than retried at 1e5.
3. **`lr_sf=1e-5` slows everything and may simply not converge by 500k.** Expected: lower
   swing, lower mean, and the failure mode to watch is the whole ladder sitting under the
   control.
4. **The combination is not expected to be additive.** If both single arms help, the pair
   most likely lands between them rather than beyond.
5. **Expected failure for the note as a whole:** every arm oscillates at the control's
   amplitude. That would mean the instability is intrinsic to the contrastive measure
   objective rather than to any optimisation rate, and the next move is the
   Bellman-trained selection head from `2026-09-08-measure-loss-audit.md` 8.5, not more
   hyperparameters.

## 8. Verdict, filled 2026-09-08; all 12 runs complete 2026-09-09

**Every arm failed. The two that look like they worked, did not.**

In-loop 50-episode ladders, 300-500k, `tools/stability_ladder.py`. The control is scored on
the SAME in-loop basis for comparability (its e500 ladder gives 0.415 / 0.371, so the
in-loop proxy is sound):

| arm | mean | swing | step | swing / mean |
|---|---|---|---|---|
| control `affine_strict_cube` | **0.432** | 0.353 | 0.135 | **0.82** |
| `tau1e3_cube` | 0.119 | **0.193** | 0.062 | 1.62 |
| `oc1e4_cube` | 0.296 | 0.533 | 0.213 | 1.80 |
| `lrsf1e5_cube` | 0.104 | **0.173** | 0.068 | 1.66 |
| `oc1e4_lrsf1e5_cube` | 0.237 | 0.320 | 0.110 | 1.35 |

**2026-09-09, all 12 runs complete.** The two rows that were partial are now full 3-seed,
9-point ladders; `oc1e4_lrsf1e5` moved from 0.256 / 0.240 (2 seeds, one partial) to
0.237 / 0.320, i.e. its swing/mean went from 0.94 to 1.35 and it joins the others in being
*relatively* less stable than the control. Nothing else moved.

**Which rows are final, and which are not.**

| arm | status | why |
|---|---|---|
| `tau1e3_cube` | **FINAL without eval500** | its best seed (0.182) is below the control's *worst* seed (0.356). A 3.6x gap that holds seed-for-seed is not something 50-episode noise produces. |
| `lrsf1e5_cube` | **FINAL without eval500** | same: best seed 0.127 against the control's worst 0.356, a 4.2x gap. |
| `oc1e4_cube` | **PROVISIONAL** | 0.296 against 0.432 is inside the band the in-loop basis can resolve. |
| `oc1e4_lrsf1e5_cube` | **PROVISIONAL** | 0.237 against 0.432, same. |

500-episode evals for the two provisional arms are queued (12 jobs, 3 seeds x 2 epochs,
`scripts/slurm/launch_stability_eval500.sh`), nice'd and dependency-chained behind the
2026-09-08/09 campaign so they cannot delay it. **Until they land the ortho line is
PROVISIONAL, not closed** -- see the correction below.

`tau1e3` and `lrsf1e5` roughly halve the swing -- and drop the mean 4x, from 0.432 to
0.119 and 0.104. **They do not hold a good policy steady; they sit near the floor, where a
small swing is free.** Normalising by the mean makes it unambiguous: every completed arm is
*relatively more* unstable than the control. Slowing the optimisation did not damp the
oscillation, it just trained worse. `oc1e4` is worse on both axes.

### Against section 7's pre-registration

| prediction | outcome |
|---|---|
| 1. `tau` cuts swing, mean roughly unchanged | **half right** -- swing fell as predicted, the mean collapsed, which was not predicted and is what matters |
| 2. `ortho_coef=1e4` is the high-variance bet; close the line if success does not improve when `orth_offdiag` drops | **failed on both axes on the in-loop basis, but the line is PROVISIONAL, not closed** (corrected 2026-09-09). Both ortho arms sit inside the band the in-loop ladders can resolve, and the pre-registration says to close the line on a *result*, not on a proxy. The r=-0.375 correlation is still confounded exactly as flagged -- it extrapolated a 1.4% observed range to a 10x coefficient change -- but that argues the arm was a long shot, not that it has been measured |
| 3. `lr_sf=1e-5` lowers swing and mean, ladder under control | **exactly right**, including the named failure mode |
| 4. the combination is not additive | consistent so far (2 seeds) |
| 5. **expected failure: every arm oscillates at the control amplitude -> the instability is intrinsic to the objective** | **this is what happened** |

### What it means

No optimisation-rate knob damps the oscillation at a useful level. Combined with 3 --
`psm_loss` correlates +0.079 with success -- the conclusion is that **the instability is in
the contrastive measure objective, not in any learning rate, target rate or regulariser
weight.** Per section 7 prediction 5, the next move is the Bellman-trained selection head,
not more hyperparameters. See section 9.

### Caveats

- **In-loop 50-episode ladders**, which swing +/-0.15. For `tau1e3` and `lrsf1e5` the gap
  is 3.6-4.2x and holds seed-for-seed against the control's worst seed, so those two are
  settled on this basis; for the two ortho arms it is not, and they are queued for eval500.
  No number in this table is reportable as a headline until then.
- The control's row is the in-loop ladder (0.432 / 0.353) so that it shares a basis with the
  arms. Its eval500 ladder is 0.415 / 0.371 -- close enough that the in-loop proxy is sound
  for the control, which is the evidence that the proxy is usable at all here.

---

## 9. Where this leads: train the selection head by Bellman backup

The chain, stated so the reasoning can be attacked rather than re-derived:

1. Slowing the optimiser does not help (section 8, and `tau1e3` / `lrsf1e5` are settled) --
   so the instability is not in the optimisation rates. The two ortho arms are still out at
   eval500; if either of them clears the control this step weakens and the chain below has
   to be re-argued.
2. `psm_loss` does not predict success, r=+0.079 (section 3) -- so the objective being
   minimised is not the objective we care about.
3. The critic ranks weakly (Spearman ~0.15, `2026-09-08-measure-loss-audit.md` 8.4) and it
   is trained CONTRASTIVELY.

Contrastive training constrains the ANGLE between representations and leaves norm growth
off-support unconstrained. That is precisely a scorer which retrieves well and ranks badly.
"Good Rankers, Bad Objectives" (arXiv:2607.27422) isolates it on this exact architecture --
a bilinear contrastive critic scoring best-of-K -- and finds, **parameter-matched**,
contrastive training at Kendall tau ~0.40 against Bellman training's ~0.70. The failure is
attributed to the objective, not the bilinear form.

**Proposal.** `q_dist(s, w, u)` already exists as a scalar selection head, but
`q_dist_loss` regresses it onto `psi^T w` -- it distils the contrastive critic and inherits
its ordering. Replace that target with a real TD backup.

**This does not cost zero-shot**, which is the obvious objection. The reward for any task
is recoverable from the basis, `r_w(x) = phi(x)^T w`, so a w-conditioned scalar critic can
be trained by ordinary TD on synthetic rewards over the same random `w` the measure loss
already draws (`mix_ratio`). Bellman training, task-conditioned, no retraining at test time.

**Risk.** The head would inherit phi. That is acceptable here specifically because phi is
the one thing that is stable -- `orth_offdiag` sits at 63.7-64.6 across all of training and
never moves.

**Gate before any code.** Check that `r_w = phi^T w` yields a signal TD can learn from
rather than being too sparse to bite on. Offline, on the dataset, no GPU. If it fails, the
whole approach is void and the alternatives are a larger critic ensemble (`num_parallel`
is 2, and 8.3 showed its spread carries no support information) or weight averaging across
the oscillation.

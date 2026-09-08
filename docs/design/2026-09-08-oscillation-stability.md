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

Control values to beat (sd0 / sd1 / sd2, 300-500k): swing **0.618 / 0.222 / 0.254**,
pooled mean 0.415.

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

## 8. Verdict

To be filled in when the arms land.

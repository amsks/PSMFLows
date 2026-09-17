# 2026-09-08 — the critic's signal over `u`, a real DSRL-NA upper bound, and two noise fixes

Status: **pre-registered, running.** Written before any of the three items produced a number.
Supersedes nothing; it sits beside `docs/design/2026-09-08-oscillation-stability.md`, whose
verdict (no optimisation rate damps the swing) is what sent us back to the critic itself.

---

## 0. The one-paragraph statement

The affine agent's acting rule is a max over `u` of `psi(s,u,u')^T w`. We now think that max
is a max over noise: the part of `Q` that actually varies with `u` is smaller than the
disagreement between the two critics in the ensemble. If that is right, the ladder does not
swing because training is unstable — it swings because the argmax is re-rolling a die every
checkpoint. Three items follow: **measure the signal honestly** (Item 1), **measure whether
the frozen flow can be steered at all when a reward-specific critic is allowed** (Item 2),
and **raise signal / lower noise in the zero-shot critic** (Item 3).

---

## 1. What three audits found (verified against the code, 2026-09-08)

**No wiring bug in the default arm.** The TD target reads the target `psi` and target `phi`
under `stop_gradient` (`agents/psmflow.py:531-536`), the affine head is the proposition
literally, and under `policy_index=latent` the bootstrap is the same prior index `u'` the
online side carries, so the backup is fixed-index and contains no max.

**The critic barely moves with `u`.** From the saved `256 x 64 x 64` `Q` tensor of the
`kappa05_sd0` dead-seed probe (`$PSM_DATA/logs/diag_deadseeds/`):

| quantity | value |
|---|---|
| mean \|Q\| | 3604 |
| std of Q over the policy index `u'` | 1713 |
| std over `u` of `max_{u'} Q` | 57 (1.6% of \|Q\|) |
| ensemble disagreement | ≈124 |

The quantity the argmax consumes — how `max_{u'} Q` varies across `u` — is **half the size of
the disagreement between the two critics**. Top-5 overlap between neighbouring checkpoints is
~0.25, which is roughly what independent re-draws would give.

**Why the head has nothing to say about other `u`.** `measure_loss` fits `psi` at exactly one
action latent per row, `u_data` (`agents/psmflow.py:427`, read at `:527-531`). Every other `u`
in the box is only ever seen through the bootstrap. So the head is trained as a function of
`u` at a single point per state and extrapolates everywhere else.

**The Spearman number from this morning is not usable.** `tools/diag_gpi_selection.py` built
its Monte-Carlo states from *every* dataset row with reward above the minimum, stepped back 25
(old `:381`). On cube a success **run** is ~44 rows long: there are 477 success onsets against
20 810 rewarding rows, so stepping back 25 from a randomly chosen rewarding row usually lands
*inside* a success that has already happened. Ten of sixteen probe states were degenerate, and
the reported 0.06–0.16 was a mean over five live states. Three further defects: the return was
undiscounted while training uses `gamma = 0.98`; `u` was held fixed for 50 steps while `Q`
depends on `u` for one step; and the returns tie heavily at `-50`. **The 0.06–0.16 is retired.**

**What we called the faithful DSRL arm copied DSRL's actor, not DSRL's critic.** DSRL-NA is a
*dual* critic. Ours has one. Specifically, DSRL-NA trains an action-space critic
`Q_A(s, a)` by TD on real `(s, a, r, s')` with bootstrap action `a' = G(s', pi_W(s'))`, then
fits a latent critic `Q_W(s, w)` by regression onto `Q_A(s, G(s, w))` at prior draws `w`, and
the actor climbs `Q_W` only, with no gradient through the frozen flow. Ours has the actor
climbing `psi^T w` at a random task vector each batch (`:438-443`, `:727-734`); the
`action_critic` branch `psi_a` (`:833-861`) is off by default, is never distilled into
anything, and never reaches the actor.

---

## 2. Item 1 — fix the critic diagnostic (CPU/1 GPU, done first)

### What changed in `tools/diag_gpi_selection.py`

| before | after |
|---|---|
| states = any rewarding row, minus 25 | states = **success onsets** only (`r[i] > min`, `r[i-1] == min`), 448 available on cube |
| `n_mc_states = 12` | `n_mc_states = 64` |
| undiscounted sum of reward | discounted at the **training** `gamma` read from the run's own config |
| one ground truth (`held`: `u` fixed for 50 steps) | **three**: `held`; `onestep` (decode `u` once, then replay the dataset's recorded actions); `onestep_bc` (decode `u` once, then the frozen flow **closed loop** on a per-state fixed prior latent stream) |
| ranked by `psi^T w` only (plus `-\|\|u\|\|` as an aside) | ranked **four ways**: `psi^T w`; a frozen FQL expert's action critic read at `G(s,u)` (`+oracle_path`); `-\|\|u\|\|`; and `-\|\|G(s,u) - a_expert(s)\|\|`, a check on the ground truth rather than on a critic. Plus two more (`na_qw`, `na_qa`) when `+na_path` points at a trained DSRL-NA run |
| one Spearman per ranker | each also computed on the roster subset inside a smaller box (`+inbox_clip`, default the DSRL-NA arm's `u_clip = 1.5`), so a low `qw` number is not ambiguous between "does not rank" and "never saw half the roster" |
| ground truth recomputed per checkpoint | `+mc_cache=<npz>` — the returns use only the frozen flow, the dataset and the simulator, so one job pays for them |

The candidate roster is unchanged: the same 64 clipped prior draws from `PRNGKey(12345)`, so
the new numbers are comparable across checkpoints exactly as before.

Why three ground truths. `held` is what the old probe measured and it is **not what GPI
deploys** — GPI redraws `u` every step — so a flat `held` ranking does not by itself convict
the critic.

`onestep` was the first attempt at the deployed question and it is **confounded**, so it is
reported and not tested on: once `u` knocks the state off the recorded trajectory, the
replayed actions no longer fit the state, and the return mostly measures how far `u` pushed
the state off the recorded path — which is exactly what `-||u||` measures. (Caught in review
by the oversight session before any job produced a number.)

`onestep_bc` is the test: decode `u` once, then run the frozen behaviour flow **closed
loop** for the remaining 49 steps. The continuation is in-support, which is what a one-step
`Q` is about, and the latent stream is **fixed per state** — every candidate at a state gets
the same continuation noise, so the comparison between candidates is paired and the BC
policy's own variance does not enter the spread.

### Pre-registration (written before the runs)

Read on the mean per-state Spearman over the 64 onset states, **`onestep_bc`** ground truth.

| outcome | reading |
|---|---|
| the FQL expert critic ranks (ρ ≥ 0.3) and `psi^T w` does not (ρ ≤ 0.15) | **the measure form is at fault.** A critic with a direct action pathway can rank these candidates; ours cannot. Item 3 Arm B (fit the head at more than one `u` per row) is the indicated fix. |
| neither ranks, and the `onestep_bc` return spread over `u` is under ~1 reward unit per state | **no critic can rank `u` here.** The one-step choice does not change the return, so GPI-over-`u` is picking between equivalent options and should become sampling, not argmax. |
| neither ranks, but the spread is large (≥ 5 reward units) | the signal exists and both critics miss it — a representation problem, not a deployment one. Item 3 Arm A will not help. |
| `psi^T w` ranks (ρ ≥ 0.3) | the 09-08 reading was an artifact of the broken state pool and the ranking story needs rewriting. Least expected. |

Expected, honestly: the second row. The dead-seed probe already found 79% of decoded prior
candidates succeed under MC, which is the shape of a decision that does not matter much.

Secondary, non-blocking: `-||u||` is expected to rank *better* than `psi^T w` on `held`
(smaller latents behave better over 50 open-loop steps) and no better than chance on
`onestep_bc`. If `-||u||` beats the critic on `onestep_bc` too, the deployed argmax is a
norm filter. The same comparison on `onestep` proves nothing, for the reason above.

A three-state CPU smoke at `H=25` (not a result — it is three states) put the `onestep_bc`
spread at 4.5 reward units, so the "flat, nothing to rank" branch is not the foregone
conclusion the `held` numbers suggested.

**The ground-truth check.** E1 measured that executing the roster candidate closest to a
frozen FQL expert's action scores 0.934. So `-||G(s,u) - a_expert(s)||` ought to rank the
`onestep_bc` returns. If it does not, the per-candidate return under a fixed continuation is
not a stable target and **nothing in this table may be read**, including the negative
results. Added after the 64-state pass, at the oversight session's request; the expert's
action is its one-step head at zero noise (its `sample_actions` ignores `temperature`), so
it differs from E1's sampled action by the head's own noise.

### Result — the 64-state pass (8 checkpoints, landed 2026-09-08)

Mean per-state Spearman over the live states, averaged over the 8 checkpoints:

| ranker | `onestep_bc` | `held` | `onestep` |
|---|---|---|---|
| `psi^T w` (deployed rule) | +0.041 | +0.007 | +0.026 |
| FQL expert action critic at `G(s,u)` | +0.066 | −0.019 | +0.007 |
| `-||u||` | +0.059 | +0.041 | +0.070 |

**Nothing ranks.** Not the measure, and not a critic whose own policy scores 0.949 on this
task. Nothing is near either pre-registered line. `rho_q` does not track success either:
sd0@350k (0.704) reads +0.100, sd0@500k (0.086) reads +0.020, sd2@450k (0.564) reads −0.017.

**But the candidates are not equivalent.** Spread over `u` on the live states: **12.6**
reward units under `onestep_bc`, 15.1 under `held` — an order of magnitude above the
1-unit threshold. Regret against the best candidate: `psi^T w` 9.04, the FQL critic 7.18,
`-||u||` 9.03, a random pick 9.53.

**Pre-registered verdict: the third row.** The signal exists and both critics miss it — a
representation problem, not a deployment one. On its own terms this says Item 3 Arm A
(averaging away ensemble noise) will not fix it.

**The weakness, stated plainly.** Only **15 of 64** states are live under `onestep_bc` (21
under `held`): at the other 49, the BC continuation fails from every candidate, so the first
action cannot matter there. That is the same small-n objection this section raises against
the retired number. The pass was **relaunched at `n_mc_states=256`**, which fixes the n —
and then failed a different check.

### The verdict above is WITHDRAWN: the ground truth does not measure what it must

The 256-state pass added the ranker that checks the *ground truth* rather than a critic:
`-||G(s,u) - a_expert(s)||`, the distance from each candidate's decoded action to a frozen
FQL expert's own action. E1 established that executing the roster candidate closest to that
action **at every step** scores 0.934. So it must rank the one-step return, or the return is
not measuring first-action quality.

**It does not rank it.** Final table, all 8 checkpoints, 256 states (59 live under
`onestep_bc`, 110 under `held`). The `inbox` columns repeat each Spearman on the 36 of 64
candidates inside `|u| <= 1.5`, so a low number cannot be blamed on the box:

| ranker | ρ `onestep_bc` | inbox | ρ `held` | inbox | ρ `onestep` | inbox |
|---|---|---|---|---|---|---|
| `psi^T w` | +0.037 | +0.019 | +0.002 | +0.004 | −0.004 | −0.013 |
| FQL expert critic | +0.099 | +0.111 | +0.005 | +0.044 | −0.046 | −0.022 |
| `-||u||` | +0.020 | −0.005 | +0.046 | +0.062 | +0.053 | +0.043 |
| **`-||G(s,u) - a_expert||`** | **+0.022** | **+0.049** | **+0.039** | **+0.118** | +0.050 | +0.067 |

Regret against the best candidate, `onestep_bc`: `psi^T w` 6.48, the expert critic 6.32,
`-||u||` 8.37, the **expert-distance ranker 8.74**, a random pick **8.58**. Picking the
near-expert candidate is **worse than random**, and restricting to the in-box candidates does
not rescue any row.

**And it is not for want of a good candidate in the roster.** Measured over the same 256
states, K=64: the closest candidate sits **0.049** from the expert's mode action while the
roster averages 0.174 (mean `||a||` = 1.231), and 0.086 from a sampled expert action against
E1's 0.062 at K=512. There is a near-expert action available at essentially every state, and
a 0.124 range to rank over. The null is real.

**What that costs and what it buys.** It costs the reading above: on `onestep_bc` the
Spearman column cannot convict `psi^T w`, because a ranker known to be right does no better.
The pre-registered rule (agreed with the oversight session before the run) is to say so
rather than draw the representation conclusion, so **"the signal exists and both critics miss
it" is withdrawn.**

What it buys is a sharper statement about the task: **one near-expert action followed by 49
steps of behaviour cloning is indistinguishable from one random in-support action followed by
the same 49 steps.** The value of acting well on cube is not located in any single step. That
is consistent with the two anchors — 0.934 when you aim at every step (E1), 0.072 when you
never do (the BC control) — and it says per-step ranking quality may be the wrong quantity to
measure in the first place; sustained selection is the thing.

One column survives and is worth keeping: on **regret**, both learned critics beat a random
pick (6.48 and 6.32 against 8.58, on a 12.29-unit spread) while their rank correlations are
~0.04–0.10. Their *argmax* carries something their *ordering* does not. That is a different
claim from the one this section set out to test, and it is not yet backed by all 8
checkpoints — the `psi^T w` rows above are one checkpoint; the other 7 are queued.

Ground truth level, checkpoint-independent: BC-continuation-only return −30.72, dataset
replay −19.48, all-fail −31.79; per-candidate success 0.072 under `onestep_bc` (i.e. exactly
the BC control, which is what the tail is).

---

### 2.1 STEP A — score the ordering against a critic, not against a rollout (2026-09-09)

If the rollout return cannot score a ranker, score the ranker against another ranker. Success
on cube comes from small per-step edges compounding over 50+ steps, so the quantity of
interest is whether a checkpoint **orders** the roster the way a known-good critic does, and
whether that agreement tracks the checkpoint's own 500-episode success. No simulator is
involved, so this can be read on any checkpoint mid-training.

`tools/diag_ranker_agreement.py`, 256 onset states, the same `PRNGKey(12345)` roster, per
state, against a frozen FQL expert's action critic at `G(s,u)` and against
`-||G(s,u) - a_expert(s)||`:

| checkpoint | 500-ep | ρ(q, expert critic) | top-8 overlap |
|---|---|---|---|
| sd0@250k | — | +0.254 | 0.283 |
| sd0@350k | **0.704** | +0.241 | 0.250 |
| sd0@450k | — | +0.162 | 0.207 |
| sd0@500k | **0.086** | +0.020 | 0.157 |
| sd1@250k | — | +0.028 | 0.194 |
| sd1@350k | — | +0.198 | 0.250 |
| sd1@450k | — | +0.267 | 0.271 |
| sd1@500k | **0.548** | +0.157 | 0.233 |
| sd2@250k | **0.246** | +0.150 | 0.229 |
| sd2@350k | — | +0.363 | 0.361 |
| sd2@450k | **0.564** | +0.226 | 0.267 |
| sd2@500k | **0.292** | +0.245 | 0.234 |

Two seeds move in **opposite directions** and agreement follows each: seed 0 decays into its
dead checkpoint (0.704 → 0.086) and agreement decays with it (+0.254 → +0.020); seed 1
improves toward its best checkpoint (450k, 0.568 on the ladder) and agreement rises to its
maximum there (+0.267). That is a better sign than one monotone trend would be.

**Across checkpoints, against 500-episode success.** Successes are read off the eval500
JSONs through `stability_ladder.ladder_from_eval500` rather than a hand-kept table — the
first version of this analysis used a hardcoded dict built from the cells the Item 1 jobs
happened to target, and **missed four cells that had already been evaluated** (sd0@450k
0.272, sd1@350k 0.494, sd1@450k 0.568, sd2@350k 0.350). So n = 10, not 6. Permutation p,
20000 samples at n = 10 and exhaustive at n ≤ 8:

| statistic | ρ vs success (n=10) | p | ρ (Item-1 6 cells) | p |
|---|---|---|---|---|
| **top-8 overlap with the expert CRITIC** | **+0.721** | **0.024** | +0.886 | 0.033 |
| top-8 overlap with expert DISTANCE | +0.382 | 0.279 | +0.886 | 0.033 |
| ρ(q, expert critic), full ordering | +0.564 | 0.096 | +0.657 | 0.175 |
| ρ(q, expert distance), full ordering | +0.576 | 0.088 | +0.771 | 0.103 |

**One statistic survives at n = 10: top-8 overlap with the expert's action CRITIC.** The two
top-8 statistics were indistinguishable on the 6 Item-1 cells and separate cleanly on all 10
— agreeing with a **value** ordering predicts success, agreeing with an **imitation**
ordering does not (+0.721 vs +0.382). That is the same split §2 found from the other side,
where expert *distance* was useless as a first-action criterion while the expert *critic* was
mildly useful. The full-ordering Spearmans sit at p ≈ 0.09 and stay suggestive.

The distinction that matters mechanically: GPI acts on the **argmax**, so the top of the
ordering is the part that reaches the policy, and it is the part that tracks success.

Note the scale change against §2: the same critic scores ρ ≈ 0.16–0.27 against the expert's
*ordering* where it scored ≈ 0.04 against rollout *returns*. That is the withdrawal in §2
seen from the other side — the return was the insensitive instrument.

**Adopted as the Arm B / Arm C readout** alongside ladder swing, per the pre-registration.
The readout is **top-8 overlap with the expert's action critic** specifically — not the
expert-distance variant, which does not survive n = 10. The full-ordering Spearman is
recorded as suggestive only and is not used to decide anything.

> ### RETIRED 2026-09-09. The statistic does not survive its own ladder being finished.
>
> `tools/diag_ranker_agreement.py` held the successes in a hardcoded six-cell dict keyed on
> `(seed, epoch)` alone. It now reads them through `stability_ladder.ladder_from_eval500`,
> keyed by run directory, which both fixes the n and makes it impossible to score one
> group's orderings against another group's successes. Rerun on the control, using every
> `eval500` cell on disk:
>
> | n | ρ(top-8 overlap, success) | p | source |
> |---|---|---|---|
> | 6 | +0.886 | 0.033 | this section, first pass |
> | 10 | +0.721 | 0.024 | this section, the adopted readout |
> | **12** | **+0.371** | **0.238** | 2026-09-09, all cells from disk; permutation, 200k samples |
>
> The two cells the n = 10 analysis lacked are the two whose evals landed on 09-09:
> `sd000@250k` overlap 0.283 against success 0.532, and `sd001@250k` overlap 0.194 against
> 0.652. Both are low agreement with high success, and adding them removes the effect. This
> statistic was chosen as the one of four that passed at n = 10, which is exactly the
> condition under which a statistic is most likely to be selected by noise.
>
> **Arm B is the evidence, not a puzzle.** Jitter 0.5 raised this statistic against the
> control by +0.058 paired, 9 of 12 cells, while its 500-episode success FELL, 0.393 →
> 0.334. When the diagnostic and the outcome disagree, the diagnostic is what is on trial.
>
> **Arm B and Arm C are judged on the eval500 ladder — mean and swing on matched cells,
> through `stability_ladder.py` — and on nothing else.** The pre-registration in point 2
> below ("top-8 overlap above the control's per-seed values") is withdrawn. Agreement
> numbers are still produced and are a description of a checkpoint, not a predictor of its
> success.

Two readings recorded with the oversight session, neither of which prompts an action:

1. **There is headroom.** Absolute agreement is low — 0.16 to 0.36 — even at the best
   checkpoints. A checkpoint that scores 0.704 still overlaps the expert's top 8 only a
   quarter of the time. Whatever the measure is ordering by, it is mostly not what the
   expert orders by, and the ceiling on this diagnostic is nowhere near reached.
2. **Arm B and Arm C now have a concrete success criterion**, which they did not before:
   top-8 overlap at 250k and 500k **above the control's per-seed values**, and **less spread
   between adjacent checkpoints**. That is a per-checkpoint readout available without any
   500-episode eval, so it can be read the moment those runs checkpoint.

Two caveats carried:
- n = 6, and the two missing evals (`sd0@250k`, `sd1@250k`) are queued at low priority.
- This measures agreement with **one** expert's ordering. It inherits whatever that expert
  gets wrong, and it is not a measurement against success directly.

**One bug found and fixed here.** The `live`-state mask was computed as
`onestep_bc_returns.std(1) > 0` on the stored **float32** array, where the standard deviation
of a constant row is ~1e-7 rather than exactly 0. That counted 249 of 256 states as live
where the true figure is **59**. The all-states columns above are unaffected (they use every
state); the mask now casts to float64 first, as the probe that wrote the cache does.

---

## 3. Item 2 — a real DSRL-NA arm on cube (the upper bound)

**This arm is NOT zero-shot.** `Q_A` is trained on the task's real reward. It answers one
question and no other: *can the frozen flow be steered on cube at all, when the critic is
allowed to be reward-specific and to see actions directly?* Its number may not be quoted
beside any zero-shot row.

Recipe, from `ajwagen/dsrl` (offline OGBench configs), ported to latents:

- `Q_A(s, a)`: TD on real `(s, a, r, s')`, bootstrap `a' = G(s', pi_W(s'))`, 2 critics,
  min over the pair, target nets at `tau = 0.005`, `gamma = 0.99`.
- `Q_W(s, w)`: regression onto `Q_A(s, G(s, w))` at prior draws `w`, several inner steps per
  outer update. No gradient through `G`.
- Actor: the existing tanh-Gaussian latent actor climbs `Q_W` only. Auto-tuned `alpha`,
  target entropy 0, box 1.5, **no BC term**.
- Nets: 3 x 2048 hidden, LayerNorm on every hidden layer, `lr = 3e-4`, batch 256.
- Eval at the mode action.

### Pre-registration

| outcome | reading |
|---|---|
| ≥ 0.5 by 250k | the flow can be steered; the zero-shot representation is the whole gap, and Item 3 is the right place to spend. |
| between BC 0.072 and GPI 0.415 | steering works but is worse than the zero-shot argmax, which would be a strange and interesting result — it would say the argmax is doing something an amortised actor cannot. |
| near BC 0.072 | **the flow/inversion is the binding constraint** and Item 3 is moot. Nothing downstream of a frozen `G` will work on cube, and the project's next move is Stage A/B, not Stage C. |

Expected: above 0.415. The per-task FQL reference on this env is 0.949 and the per-task
latent-RL residual arm reached 0.905, so a reward-specific critic over this flow has a lot of
room. If it does not clear BC, that is the single most informative negative available.

### Result, 2026-09-09 — the flow can be steered, and it is not close

Three seeds, 500k steps. **In-loop 50-episode ladders, which are NOT reportable** (they swing
±0.15); 500-episode evals are queued (2492559–64) and the numbers below must not be quoted
until those land.

| seed | 50k | 150k | 250k | 350k | 450k | 500k |
|---|---|---|---|---|---|---|
| sd0 | 0.92 | 1.00 | 0.90 | 0.94 | 0.90 | 0.88 |
| sd1 | 0.92 | 0.92 | 0.94 | 0.96 | 0.94 | 0.90 |
| sd2 | 0.80 | 0.82 | 0.96 | 0.94 | 0.84 | 0.90 |

Against: BC control **0.072**, actor-free GPI **0.415**, the 09-08 "faithful DSRL" arm
**0.136 / 0.141**, and the per-task FQL reference **0.949**.

**The pre-registered top branch, by a distance.** The bar was ≥ 0.5 by 250k; the arm is at
~0.9 by **50k**. So:

- **The frozen flow is not the binding constraint on cube.** A critic that is allowed to know
  the reward and to see raw actions steers this exact decoder to near the per-task reference.
  Item 3 is therefore the right place to spend, and the "the flow/inversion is the problem"
  branch is closed for this environment.
- **The gap is the zero-shot representation.** 0.9 versus 0.415 is the price of not knowing
  the reward, on one substrate, with everything else held fixed.

**A second observation, not pre-registered.** The arm is *flat*: 0.80–1.00 across eleven
checkpoints and three seeds, against the zero-shot arm's 0.086 ↔ 0.704 traverse on a single
seed. **The oscillation is a property of the zero-shot measure, not of the substrate.** The
whole 09-08 stability campaign searched for an optimisation-rate knob and found none; this
says it was looking in the right place but at the wrong object.

**And the internal readout agrees.** `na_signal_over_disagreement` — the std of `Q_W` over
`u` divided by `Q_A`'s own ensemble disagreement — sits at **1.4–2.0** through training,
where the zero-shot measure reads 57/124 = **0.46** (§1). The reward-specific latent critic
has three to four times the signal-to-noise over `u`, measured the same way.

**Not zero-shot. Not comparable to any zero-shot row.** `Q_A` is trained on cube's real
reward. This number's only job is to say whether the substrate can be steered at all, and it
says yes.

#### 500-episode confirmation (2026-09-09)

| seed | 250k | 500k |
|---|---|---|
| sd0 | 0.902 | 0.842 |
| sd1 | 0.946 | 0.868 |
| sd2 | 0.966 | 0.936 |

**Pooled 0.910** (min 0.842, max 0.966, n = 6), against BC **0.072**, actor-free GPI
**0.415**, the 09-08 "faithful DSRL" arm **0.136 / 0.141**, and the per-task FQL reference
**0.949**. The arm reaches **96% of the per-task reference** while decoding through the same
frozen flow, and the in-loop reading held exactly. The flatness holds too: the whole range is
0.842–0.966 where the zero-shot arm traverses 0.086–0.704 inside one seed.

#### The agreed diagnostic: a critic at 0.910 ranks the roster no better than one at 0.415

Both DSRL-NA critics scored the same 256-onset roster as §2's table:

| ranker | ρ `onestep_bc` | inbox | regret vs best | its own deployed success |
|---|---|---|---|---|
| `psi^T w` (zero-shot) | +0.065 | +0.052 | 7.88 | 0.415 |
| **`Q_A(s, G(s,u))`** | **+0.062** | +0.055 | 6.49 | **0.910** |
| **`Q_W(s, u)`** | **+0.062** | +0.056 | 6.67 | **0.910** |
| FQL expert critic | +0.099 | +0.111 | 6.32 | 0.949 (its own policy) |
| expert distance | +0.022 | +0.049 | 8.74 | — |
| a random pick | — | — | 8.58 | 0.072 |

**The oversight session's pre-registration fires exactly.** It predicted `Q_A` would rank
"no better than the FQL expert did, about +0.07", with success decided by the actor's local
climb rather than by roster ranking. `Q_A` reads **+0.062** against the predicted +0.07, is
indistinguishable from the zero-shot measure's **+0.065**, and the arm scores **0.910**
against 0.415.

So roster-ranking ability does not determine success: the arm that more than doubles the
zero-shot number orders the same candidates no better.

**The verdict is CONDITIONAL, and the condition matters.** The pre-registration's wording
was that this "settles that GPI-over-a-roster is the wrong deployment and distillation to an
actor is the right one". The first half stands. The second half does **not** stand as an
unconditional claim about deployment, and the counter-example is already on the record: the
2026-09-08 `dsrlfaithful` arm deployed exactly that way — a latent actor, no roster argmax —
climbing the MEASURE readout at a random `w`, and scored **0.136 / 0.141**. Same deployment,
a sixth of the number.

What separates the two is not the deployment but the critic: 0.910 has a **Bellman critic on
the real reward**, 0.136 has the measure readout. So the defensible statement is:

> **Actor deployment wins WHEN the critic behind it is a Bellman one.** Roster argmax is the
> wrong deployment either way, but replacing it buys nothing on its own.

Whether a *zero-shot* Bellman critic keeps that property is precisely what **D2** decides,
and **D1** prices what the linear readout costs on the way there. Neither is answered by this
table.

**This does not contradict §2.1.** Step A measured agreement *within the roster-argmax
family*, across checkpoints that all deploy by per-step argmax — and there ordering quality
does track success (ρ = +0.72). DSRL-NA does not deploy that way at all. Both hold: **if you
deploy by roster argmax, your ordering had better be right; the way to win is not to deploy
by roster argmax.**

---

## 4. Item 3 — raise the signal, lower the noise (zero-shot, launched after Item 2)

Both arms are default-off. 3 seeds each on cube, 500k steps, in-loop ladders every 50k, then
500-episode evals at 250k and 500k. Scored with `tools/stability_ladder.py` on **swing and
step**, not on the pooled mean. Control: `affine_strict_cube` (mean 0.415, swing 0.371,
step 0.163).

**Arm A — DROPPED before launch, on Item 1's evidence.** It was to be `num_parallel = 8`
with the ensemble **mean** at acting, on the premise that the argmax is a max over ensemble
noise. Item 1 killed the premise: a frozen FQL expert's action critic — trained by a real
max-backup on real rewards, with a direct action pathway and no ensemble of ours — ranks the
same roster at ρ +0.066, no better than the measure. Two independently trained critics
missing the same signal is not a noise problem that averaging eight of them fixes. The
command it would have run is kept in `scripts/slurm/launch_u_signal.sh` in case the
256-state pass reverses that.

**Arm B — fit the head at more than one `u` per row.** The measure loss sees one action
latent per state. `measure_u_samples = 4` adds 3 more per row, each carrying the **same**
`(s, s')` target, the loss expression otherwise identical.

**Not from the stored posterior**, which was the plan until it was measured.
`tools/diag_mixture_decode.py`, 4096 cube rows, `||a|| = 0.875`, one-step decode:

| latent | `\|\|G(s,u) - a\|\|` | fraction of the way from point to prior |
|---|---|---|
| point inverse | 0.0885 (p90 0.128) | 0.00 |
| posterior mean | 0.1014 | 0.07 |
| **posterior sample** | **0.1667** | **0.40** |
| prior draw | 0.2851 | 1.00 |

A posterior sample is 40% of the way to an uninverted prior draw — 60% on the
`prior_scale = 0.691` npz, the only one `main.py` permits for the mixture. Those are not
other preimages of the same action, so three of every four rows would carry the point
inverse's `(s, s')` target while decoding somewhere else: the fiction
`mask_invalid_preimages` exists to prevent, at 75% of the loss. This also sharpens
COMPENDIUM §4.9 — the mixture is not merely blurred, at these α it is most of the way to
carrying no inversion information at all.

So the extra latents are a **jitter ball** around `u_data`, with the width taken off the same
probe's ladder rather than from the inversion's temperature:

| σ | 0.1 | 0.2 | 0.3 | 0.5 | 0.75 | 1.0 |
|---|---|---|---|---|---|---|
| decode error | 0.090 | 0.096 | 0.103 | 0.121 | 0.154 | 0.190 |

The point inverse's own p90 is 0.128, so at σ ≤ 0.5 the extra latents still decode to *this
transition's* action. **Two doses, σ = 0.3 and 0.5, 3 seeds each**, on the **same** npz as
`affine_strict_cube` — so that arm is the control directly and no npz-matched control is
needed. The `mixture` source stays in the config, unused. Measured on the smoke:
`u_extra_dist` = 0.633 at σ = 0.3, as expected for a 5-dim ball.

> Prediction: the std of `Q` over `u` rises relative to the ensemble disagreement — i.e. the
> ratio 57 : 124 moves toward and past 1. Success is secondary; this arm is scored first on
> whether it creates signal at all.
>
> Honest limit, pre-registered: a ball of radius 0.63 around `u_data` in a box of half-width
> 3 teaches the head the *local* shape of `Q` near the data latents, while the deployed
> argmax scans the whole prior box. Expect the extrapolation problem reduced, not removed.

### Expected failures, stated now

- Arm B raises the `u`-spread of `Q` and *lowers* success: a head fitted on a small ball
  around `u_data` gains local smoothness and no new (state, action) information, and the
  extra rows dilute the gradient the single exact preimage used to carry.
- Arm B moves nothing at either dose, in which case Item 2's number decides whether this
  line continues at all.
- σ = 0.5 helps where σ = 0.3 does not, or the reverse, with no monotone trend across the
  two — at n = 3 seeds and a ladder that swings 0.37, a two-point dose is not enough to call
  a trend, and the arm would need a third dose before anything is claimed.

---

### 4.1 The Item 2 diagnostic (agreed with the oversight session)

At 250k and 500k, the trained DSRL-NA arm's own two critics score **the same** 256-onset
roster, appended to the Item 1 table as two more rows: `qw(s, u)` and `qa(s, G(s,u))`. This
asks whether a critic that is *allowed to know the reward* ranks where the zero-shot measure
does not. `+na_path` / `+na_epoch` on `tools/diag_gpi_selection.py`.

Read `qw` beside its `_inbox` column: the arm trains at `u_clip = 1.5` and the shared roster
is drawn at 3.0, so about half of it is outside the box `qw` ever saw. `qa` has no such
problem — it reads actions, and `G` maps the whole box into `[-1, 1]`.

Two pre-registrations, recorded separately:

- **This session:** if `qa` ranks (ρ ≥ 0.3) where `psi^T w` and the FQL critic do not, the
  difference is reward-specificity, and the zero-shot inference `w = E[r φ]` is implicated
  rather than the head.
- **Oversight session:** `qa` ranks the roster no better than the FQL expert did (≈ +0.07),
  and the arm's success is decided by the actor's local climb, not by roster ranking. If the
  arm scores well above 0.415 with `qa` still not ranking, that settles that GPI-over-a-roster
  is the wrong deployment scheme and distillation into an actor is the right one.

---

### 4.2 Arm C — the same seam, but the posterior's SHAPE (2026-09-09)

Arm B's jitter ball spends its sampling budget equally in every direction. The EM posterior
does not: its covariance is stretched along the directions in which the decode barely moves,
which are the directions the preimage set actually extends along. Arm C draws the extra
latents from that posterior with every component's covariance scaled by `c²`, so the shape
is kept and only the width is chosen — `sample_preimage_noise(..., scale=c)`, the existing
sampler, given a scale rather than a second sampler beside it.

**Probe, all three envs** (`tools/diag_mixture_decode.py --shrink`, 4096 rows, one-step
decode, on the npz each env's `affine_strict` control actually trained on):

| env | `\|\|a\|\|` | point (gate = its p90) | chosen `c` | sample err | dist from `u_data` | matched jitter | ratio |
|---|---|---|---|---|---|---|---|
| cube | 0.875 | 0.0885 (0.128) | **0.5** | 0.119 | 1.675 | σ=0.5: 0.124 / 1.068 | **1.57×** |
| antmaze | 1.984 | 0.2661 (0.450) | **1.0** | 0.314 | 1.574 | σ=0.3: 0.326 / 0.833 | **1.89×** |
| pointmaze | 0.948 | 0.0852 (0.180) | **0.5** | 0.130 | 1.236 | σ=0.1: 0.120 / 0.125 | **9.9×** |

**The hypothesis holds on all three.** At a matched decode error the shrunk anisotropic draw
covers 1.6–9.9× more latent distance than the round ball. It is strongest on pointmaze,
where `d_a = 2` and a round ball is nearly useless.

Three things the sweep also settled, none of them expected:

1. **Shrinking has a floor, and it is the posterior's MEAN.** `c → 0` collapses onto the
   component mean, not onto `u_data`, and that mean sits **1.14 / 1.74 / 2.79** away from the
   point inverse (cube / antmaze / pointmaze). So on antmaze the sweep is nearly flat in `c`
   (dist 1.411 at c=0.1, 1.574 at c=1.0) — the covariance was never the binding constraint
   there; the mean's displacement is. `c` is chosen against the decode gate, which is why
   antmaze's answer is `c = 1.0`, i.e. **no shrink at all**.
2. **`num_clusters = 1` in every published npz**, so the per-component breakdown is a single
   row and the drop rule is vacuous. It is computed anyway (a future multi-component
   inversion is covered) with one guard added after it fired: dropping *every* component
   would empty the mixture, so that case is reported as a MARGINAL verdict on `c` instead.
   It fired exactly once — pointmaze at `c = 1.0`, which sat at the gate (0.1784 vs 0.1799);
   hence the dose one step down.
3. **The box, not the posterior, decides where the extra latents land on pointmaze.** Its
   posterior mean is 2.787 from the point inverse and a real fraction of rows have it outside
   `[-u_clip, u_clip]`, so the draws clip. That is also why pointmaze's `dist` barely moves
   with `c`.

**The `prior_scale` gate does not apply to this path, and on cube it points the wrong way.**
`main.py` asserted `prior_scale > 0` before reading the mixture. That gate asks whether the
fit sits inside the prior the latent **actor** samples from — its original target,
`use_point_preimage=false`, which is untouched. This path never samples the actor from the
mixture: it fits the measure head at extra latents, where the only question is whether they
decode to the transition's recorded action. Measured, that proxy is **inverted on cube** —
the `prior_scale = 0.691` npz's samples decode at **0.207** against the legacy npz's
**0.167** (point inverse 0.089), and no `c` in the swept range brings the former inside the
gate — and on pointmaze **no `prior_scale > 0` npz exists at all**. The clause covering
`measure_u_samples` (added earlier the same day, in this campaign, not a standing invariant)
is replaced by a gate on deliberateness: `measure_u_mixture_shrink` has **no default** and
`create` refuses the mixture source without it, while `main.py` prints the sidecar's
`prior_scale` so a shrink chosen on a different npz is visible in the log.

The payoff is that every env runs on the **same npz its `affine_strict` control trained on**:
no preimage confound anywhere, and no extra control seeds.

> Prediction: the same as Arm B — the std of `Q` over `u` rises relative to the ensemble
> disagreement, and swing/step falls. Plus: **Arm C beats Arm B on cube at matched decode
> error** if the shape is what matters. On antmaze the bar is clearing the BC control 0.072
> with a pooled late mean above the current 0.081.
>
> Expected failure, stated now: `c = 1.0` on antmaze means Arm C there *is* plain mixture
> sampling, so if antmaze moves and cube does not, the result is about the extra latents
> existing at all, not about their shape.

---

## 4.3 The GATE — can phi express the reward at all? (2026-09-09)

Item 2 put the whole gap inside the critic: 0.9 flat with a real reward against 0.415
oscillating with `psi^T w`. Four differences separate those arms, and the agreed plan is to
walk them one at a time. The first is the **linear readout**: every zero-shot critic here
reads its reward as `phi(x)^T w` with `w = E_D[r phi]`. If the best linear readout of `phi`
cannot reconstruct the reward, no `psi` on that `phi` can represent the task.

`tools/diag_reward_readout.py`, 200k dataset rows, 12 checkpoints of `affine_strict_cube`.
Pre-registered: **R² > 0.5 viable, R² < 0.2 capped**.

| statistic | value |
|---|---|
| topline R² (least squares, shifted reward) | **0.116** (range 0.107–0.120) |
| closed-form `w = E[r phi]`, after optimal rescale | 0.116 |
| cosine between the closed form and the least-squares optimum | **0.990–0.998** |
| R² gain from adding an intercept | +0.0003 |
| topline R² on the **raw** (unshifted) reward | **−0.20** |

**Verdict: CAPPED.** 0.116 is far below the 0.2 line. This reproduces COMPENDIUM §4.7's D2
(R² 0.129) independently, on today's checkpoints rather than the July ones, so it is a
property of the method and not of one stale run.

> **AMENDED 2026-09-09, after the baseline this gate was missing.** The 0.2 line was set
> with nothing to compare 0.116 against. Measured on the same 200k cube rows, same shifted
> reward, least squares with intercept: raw observations (28-d) R² 0.049; random tanh
> features at phi's own width (128-d, untrained) 0.065; random Fourier features 512-d
> 0.096; random Fourier features 2048-d 0.169 in-sample and 0.107 held out. Trained phi
> (128-d) reads 0.116 — about twice a matched random basis, and level with a 2048-d
> nonlinear one. `psm-data/logs/diag_reward_baseline_cube.json`.
>
> So the sentence "phi cannot express the reward" is **withdrawn**. What holds: no linear
> basis of this size expresses a reward that 2.1% of rows pay, and phi is 2x a matched
> random one. What the agent optimises is the reward PROJECTED onto phi's span — a blurred
> version of the goal. Whether a blurred goal is enough for a value is not settled by any
> R², and it is what Arm D1b measures. The conclusion below that the work belongs on phi
> rather than psi does not follow from 0.116 alone; §3's D1/D2 numbers are the evidence
> that carries it, and D1 has since been found to be an unfair test of the channel.

Four things the fit says that R² alone does not:

1. **The estimator is not the problem — the basis is.** `w = E[r phi]` sits at cosine
   0.99+ to the least-squares optimum and lands within 0.003 R² of the topline. There is
   nothing to gain by estimating `w` better, which closes that direction for good.
2. **The cap is FLAT across the ladder.** R² runs 0.107 to 0.120 while 500-episode success
   swings eight-fold. sd0@350k (**0.704**) reads 0.1089; sd0@500k (**0.086**) reads 0.1103 —
   the *dead* checkpoint has the marginally better readout. So reward expressiveness is a
   constant ceiling, **not** the variable that separates a good checkpoint from a dead one.
   Whatever the oscillation is, it is not this.
3. **Direction right, magnitude wrong, and the magnitude is what matters here.** The readout
   separates rewarding rows from the rest by **2.3–2.4 sd** and predicts 0.128 on them
   against 0.019 elsewhere — a 6.7× ratio, clearly not noise. But the target is 1.0 against
   0.0. Because rewarding rows are only 2.1% of the data, **86.5% of the total predicted
   reward mass sits on rows that pay no reward** (range 86–87% over the 12 checkpoints),
   where the true reward puts none. A successor measure integrating `phi^T w` along a
   trajectory is therefore accumulating mostly the wrong thing, and that is the cap
   expressed mechanically rather than as a correlation.
4. **`eval_reward_shift=1.0` is load-bearing, not cosmetic.** On the raw −1/0 reward the
   topline R² is **−0.20**: a no-intercept linear readout of `phi` does worse than predicting
   the mean, because it cannot represent the constant offset. The shift is what makes the
   readout usable at all.

**What this does to the plan.** The pre-registration says the work redirects to `phi` rather
than `psi`, and COMPENDIUM's live hypothesis 3 (a `phi`-grounding auxiliary) is the standing
proposal for that. Two qualifications carried forward to D1/D2:

- **D1 is now a quantification, not a test.** Training `Q_A` on `r_hat = phi^T w` feeds it a
  reward whose mass is 86% misplaced, so a large drop from 0.9 is expected. The number it
  produces is the one nobody has: what the readout costs in success.
- **The gate does NOT cap D2's training, only its eval.** D2 trains on synthetic rewards
  `r_w = phi(s')^T w`, which are exactly expressible by construction — the readout loss
  cannot appear during training. It appears only at deployment, when the real task's `w` is
  inferred and the induced reward is the 86%-misplaced one. So a D2 that trains beautifully
  and evaluates poorly is the predicted outcome, not a surprise, and the diagnostic that
  separates the two is whether the inferred eval `w` lies inside the training `w` mixture.

### 4.4 The gate extension — the cap is not one thing (2026-09-09)

Two questions: does the orthonormality weight move the cap, and is the cap the method's or
cube's? Neither pre-registered branch is what happened.

**The ortho coefficient moves it the WRONG WAY.** 12 checkpoints per arm, cube:

| arm | `ortho_coef` | topline R² | deployed R² | cos |
|---|---|---|---|---|
| `affine_strict_cube` | 1e3 | **0.115** | 0.114 | 0.997 |
| `oc1e4_cube` | 1e4 | **0.091** | 0.090 | 0.999 |
| `oc1e4_lrsf1e5_cube` | 1e4 | 0.093 | 0.092 | 0.999 |

R² *falls* as the weight rises. The direction fits the proposed mechanism — a `phi` forced
to spread its mass evenly is worse at isolating a 2% success set — but the effect is far too
small to be the wall. **The regulariser is not the lever.**

**And the cap is not one thing across environments.** Same probe, with `phi`'s Gram measured:

| env | topline R² | deployed R² | cos(deployed, optimal) | Gram dev from I | cond | separation |
|---|---|---|---|---|---|---|
| cube | 0.115 | 0.114 | **0.997** | 0.035 | 1.3 | 2.4 sd |
| antmaze | 0.104 | 0.083 | 0.893 | 0.423 | 15.4 | 3.4 sd |
| **pointmaze** | **0.514** | 0.149 | **0.075** | 2.39 | **2e9** | 9.5 sd |

Pointmaze's basis is the **most** expressive of the three — above the pre-registered
viability line — and its deployed estimator points **almost orthogonally** to the best
readout. `affine_strict_pointmaze` scores exactly **0.000** on every seed and checkpoint.

**The mechanism was in the write-up all along.** Cor. `reward-inference` states that
`w = E_D[r phi]` is the least-squares projection *exactly when* `E_rho[phi phi^T] = I` —
which is, in the write-up's own words, "why the orthonormality loss is part of the
specification and not a stability regulariser". The condition was stated; it was never
checked. On cube it holds (Gram deviates 3.5%) and the estimator is optimal. On pointmaze it
has collapsed and the estimator is worthless.

#### The Gram ladder: the regulariser is not blind, it is losing

30 pointmaze checkpoints (3 seeds × 10 epochs), the measured Gram beside the `orth_loss` the
run logged at that same step:

| relationship | Spearman |
|---|---|
| `orth_loss` vs Gram deviation from I | **+1.000** |
| `orth_loss` vs Gram condition number | +0.954 |
| Gram condition vs **deployed** R² | −0.829 |
| Gram condition vs cos to the optimal readout | −0.848 |
| Gram condition vs **topline** R² | +0.400 |

| quantity | range over the ladder |
|---|---|
| topline R² (what the basis can do) | 0.25 – 0.60 |
| deployed R² (what the estimator gets) | **0.017 – 0.305**, a 17× swing |
| Gram condition number | **4.8e3 – 7.8e7**, four orders of magnitude |
| logged `orth_loss` | 18 – 1942 |

**Pre-registered branch (a) — "the regulariser is blind to the collapse" — is refuted, at
rank correlation 1.000.** The ortho loss tracks the Gram deviation *perfectly*; the number
was on the dashboard the whole time and nobody was reading it as what it is. Branch (b) —
"large from the first checkpoint, hence a capacity question" — is also wrong: the condition
number **oscillates by four orders of magnitude across checkpoints**, non-monotonically,
which is the same signature as cube's success ladder under the same loss.

So the ortho term is not blind and not absent. It is simply **losing**, intermittently, and
when it loses the reward inference stops working while the basis underneath stays fine.

This also gives the 09-08 stability campaign's strongest correlate a mechanism: `orth_offdiag`
was the best single predictor of cube success (r = −0.375) and no one could say why. It is
measuring whether `w = E[r phi]` is still the right estimator.

**Fix, and it needs no retraining.** `reward_inference='whitened'` (default off) solves the
normal equations instead of taking the mean — the same object wherever the Gram is the
identity, so a **no-op on cube by construction**. Eval-only tests are queued on pointmaze
(2492571–73) and antmaze (2492574–76). Pre-registered with the oversight session: on
pointmaze *any* non-zero 500-episode result is decisive, and **above 0.2 means the estimator
was the entire pointmaze failure**. Added to `stability_ladder.ACTING_OVERRIDES` first, so a
whitened eval cannot overwrite a run's own ladder entry.

#### REFUTED by its own pre-registered test (2026-09-09, later the same day)

The whitened estimator was the eval-only test of "the collapsed Gram breaks reward inference,
and that is why pointmaze fails". It was pre-registered as: **any non-zero 500-episode result
on pointmaze is decisive; above 0.2 means the estimator was the entire pointmaze failure.**

| env | checkpoint | closed form | whitened | delta |
|---|---|---|---|---|
| **pointmaze** | sd0 @500k | 0.000 | **0.000** | +0.000 |
| **pointmaze** | sd1 @500k | 0.000 | **0.000** | +0.000 |
| **pointmaze** | sd2 @500k | 0.000 | **0.000** | +0.000 |
| antmaze | sd0 @500k | 0.078 | 0.010 | **−0.068** |
| antmaze | sd1 @500k | 0.004 | 0.042 | +0.038 |
| antmaze | sd2 @500k | **0.428** | **0.062** | **−0.366** |
| cube | sd0 @350k (healthy) | 0.704 | 0.698 | −0.006 |
| cube | sd0 @500k (dead) | 0.086 | 0.160 | +0.074 |

**Pointmaze does not move at all — exactly 0.000 on every seed.** By the pre-registered rule
the hypothesis is dead: the estimator was *not* the pointmaze failure. Something else is
broken there, and the reward channel is the best of the three environments while the agent
scores zero.

**Worse, whitening HURTS antmaze**, and most on the seed that was working: 0.428 → 0.062.

**And the topline that motivated the hypothesis was partly overfitting.** Refitting on half
the rows and scoring on the other half:

| env | in-sample | held-out | gap | Gram cond | `\|\|w\|\|` |
|---|---|---|---|---|---|
| cube | 0.121 | 0.104 | 0.017 | 1.4 | 0.05 |
| antmaze | 0.102 | 0.102 | 0.000 | 29.5 | 0.03 |
| pointmaze | 0.584 | **0.395** | **0.189** | 3.5e6 | **2.00** |

Unregularised least squares on a Gram with a 2e-4 eigenvalue puts unbounded weight on
directions where `phi` barely varies — `||w||` is 40-70x the other environments. The claim
"pointmaze's basis is the most expressive" **survives weakened**: 0.395 held-out still beats
cube's 0.104 and still clears the 0.2 line, but the 0.514 headline was inflated by ~0.12.

**Why whitening HURTS, mechanically.** The whitened `w` loads on the low-variance
directions of `phi` — that is what inverting an ill-conditioned Gram does, and it is why
`||w||` is 40-70x larger. But `psi` was never *trained* where `phi` barely varies: those
directions carry almost no data. So `psi^T w` reads `psi`'s **noise** there. Reward
reconstruction and `psi`'s reliability pull in opposite directions, and no choice of
estimator can satisfy both — which is why **no estimator arm follows from this**, including
the training-side whitening that was parked pending this result. It is cancelled.

### FINDING: neither natural proxy for critic quality predicts policy quality

Two independent measurements, the same day, the same shape:

| proxy | the comparison | verdict |
|---|---|---|
| **roster-ranking ability** | a critic scoring **0.910** ranks the roster at ρ +0.062; one scoring **0.415** ranks it at +0.065 | does not predict |
| **reward-reconstruction** | on antmaze the whitened `w` reconstructs reward *equally well* out of sample (0.102 either way) and drops the policy 0.428 → 0.062 | does not predict |

Both are the obvious things to measure about a critic, both are what this project has been
measuring, and neither tracks the outcome. **The one internal quantity that has lined up with
outcome so far is the signal-over-disagreement ratio over `u`** — the spread of `Q` across
latents divided by the critic ensemble's own disagreement — reading **1.4-2.0** on the arm
that scores 0.910 against **0.46** on the arm that scores 0.415. It is logged on D1 and D2
(`na_signal_over_disagreement`, emitted by `na_spread` whenever `dsrl_na.enabled`), so
whether it survives as the proxy for the actor family is decided by those two arms rather
than asserted here.

What survives from §4.4 unchanged: the Gram *does* collapse on pointmaze; `orth_loss` *does*
track it at rank correlation 1.000; the write-up *did* state the precondition and it *was*
never checked. What does not survive is the causal claim that this is why pointmaze fails.

**Scope discipline.** None of this changes the cube conclusion. On cube the Gram is the
identity to 3.5%, the estimator is optimal, and the basis is the wall — so D1 and D2 remain
the right cube experiments and are launched as planned. What changes is that "the work is on
`phi`" is a **cube-only** statement. COMPENDIUM §4.7's D2 has been marked cube-only with
these numbers beside it.

---

## 5. Bookkeeping

- The 0.06–0.16 Spearman in the 2026-09-08 HANDOFF entry (§2, "Dead seeds are a SELECTION
  failure") is **retired**: it was a mean over 5 live states. The conclusion it supported may
  survive the corrected probe, but it is not currently supported.
- Every behaviour added here is a default-off seam. `agent=psmflow` with no flags is
  unchanged, and no published number moves.
- **One guard clause changed, and it is not the one people will assume.** `main.py`'s
  `prior_scale > 0` assertion on `use_point_preimage=false` — the standing gate, protecting
  the latent **actor**'s sampling distribution — is **untouched, byte for byte**. What was
  replaced is the clause extending that assertion to `measure_u_samples > 1`, which was
  added earlier the same day in this campaign and was never a standing invariant. It was a
  proxy, and the thing it proxies for is now measured directly and points the other way: on
  cube the `prior_scale = 0.691` npz's samples decode **0.207** from the recorded action
  against the legacy npz's **0.167** (point inverse 0.089), and pointmaze has no
  `prior_scale > 0` npz at all. The replacement gate is that `measure_u_mixture_shrink` has
  no default and `create` refuses the mixture source without it; `main.py` prints the
  sidecar's `prior_scale` so a shrink picked on a different npz shows up in the log.
- `u_extra_clipfrac` is logged per step beside `u_extra_dist`: the fraction of extra-latent
  components sitting on the box wall. On pointmaze the box, not the posterior, decides where
  the draws land, so that number is part of the result rather than telemetry.

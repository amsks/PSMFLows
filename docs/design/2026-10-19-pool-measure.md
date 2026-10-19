# 2026-10-19 — the pool measure: a successor distribution over a fixed reference set

## Why

The goal-indexed measure of arm A and nf2 is a detector of a trajectory's own next and later
states. Measured on 2026-10-15 to 10-18 (`docs/PSMGOAL_COMPENDIUM.md` sections 5 and 6.2):
total variation between its distribution over 1024 dataset states and the rollout occupancy
is 0.87 to 0.93 against a split-half noise floor of 0.35, for critic A and nf2, whether the
rollouts follow best-of-64, the run's own actor, or BC; the mass on a trajectory's own later
states does not change with the action latent (ratio 1.00 to 1.05); mass does not transfer to
states on other trajectories at the same offset. The brainstorm of 10-17 (transcript line 1052,
summarised in the compendium) proposed an explicit successor distribution relative to a stated
reference measure as the first build. This document fixes that design.

The cause, as read from the loss (compendium 2.4): the candidates are the batch's other next
states, each present once, each another row's positive; the one-hot term anchors the row's own
next state, which is never a candidate for any other row; the continuation term at (s', u')
sees no state that is downstream of s'. The fixed point "spike on the exact own next state,
flat elsewhere" satisfies the loss, and the triple network b(s, u, y) can represent it by
memorising distinct triples. nf2 added own future states as candidates, which gave the
continuation term something on-trajectory to hit and raised success, and left the rest.

## What changes

One thing: the candidate set. It becomes a fixed reference set R of `pool_size` dataset states,
drawn once, shared by every transition and every training step, and the same set the readout
integrates over. Everything on R is then a proper distribution over a fixed finite set of
states, and the Bellman recursion closes on R:

```
R = { y_1 .. y_P }                       P = pool_size, fixed for the run
p_i(j)      = softmax_j( M(s_i, u_i, y_j; w_i) / t )                     online, over R
a_i         = argmin_j || (s'_i - y_j) / sigma ||_2                       the anchor: R's state nearest the next state
target_i(j) = (1 - gamma) 1[j = a_i] + gamma softmax_j( M_bar(s'_i, u'_i, y_j; w_bar_i) / t )
L_pool      = -(1/N) sum_i sum_j target_i(j) log p_i(j)
```

M, M_bar, w, w_bar, u'_i, t, gamma are exactly those of compendium 2.4 and 2.5. sigma is the
per-dimension standard deviation of the observations in R. Nothing else in the agent changes:
the same phi, b, h, the same bootstrap sources (actor, frozen_delta, data, argmax, bc_mean),
the same actor loss, the same Polyak targets.

Why the anchor is R's nearest state and not the next state itself. If s'_i were appended as
its own candidate, the loss would again be satisfiable by a spike on a triple that occurs once.
With the anchor on R, about N_data / P transitions share each anchor state (1,000 at P = 1024
on cube), so b(s, u, y_j) has to represent a region of next states, and the target
distribution at (s', u') over R is a distribution over states that recur every step. Transfer
across trajectories is forced by construction instead of hoped for from generalisation. The
price is quantisation at the pool's resolution; the fullness diagnostic already measures
occupancy at that resolution (1024 nearest-neighbour cells), so the two are on the same footing.

What is deliberately not changed, against the brainstorm's wider list: no factorised
phi(s, u)^T f(y) (the user's notation and modelling rule; the triple network stays), no reward-
sampled policy indices (first the goal index and a one-member bank, below), no affine
construction (research direction after a calibrated measure), no importance-weighted future-
biased proposals for the training pool (the pool is uniform over the data).

## Policy index

Two settings, both through the existing `policy_index` seam.

`policy_index=goal` (as arm A): w_i = h(g_i) with g_i the geometric hindsight goal, u'_i from
the configured bootstrap source. This is arm A's recipe with the pool loss. The first arm uses
`bootstrap_source=frozen_delta frozen_delta_mode=plus` with the field from A seed 0 at 500k,
because the frozen-step arms showed a frozen field reproduces A's gain (0.741 against 0.805,
fdplus) and a frozen policy makes the target a plain policy evaluation. A second arm trains the
actor live as A does.

`policy_index=bank`: n_bank = 1, the member is one clipped prior draw (BC), w is a single
learned vector (the existing `w` net evaluated at a constant code, or a fresh (z,) parameter;
the implementer picks whichever reuses `_coef` with least new code), u'_i = clip(eps_i). This is
a clean policy evaluation of the behaviour policy with a full measure; its readout is a one-step
improvement over BC. It is the arm whose fullness number says whether the construction works
at all, because its policy is known and its occupancy can be rolled out exactly.

## Readout

Two readouts on every pool checkpoint.

`coef_source=hgoal_each` (unchanged, goal index only): the share of the goal among
[goal + 224 refs], so pool critics are comparable with A and nf2 on the number that decides.

`coef_source=pool_reward` (new): value by integrating the reward over a bank with importance
weights. Per task, from the relabel batch of 10,000 rows (compendium 2.8 step 1), draw a bank
B of `pool_eval_size` next states, n_r of them from rewarding rows (reward above 0.5 after the
+1 shift) and the rest from non-rewarding rows, with weights omega_j = (count of that class in
the relabel batch) / (count of that class in B), so B is an importance-weighted uniform sample
of the data. Then

```
Q(s, u; w) = sum_j omega_j exp(M(s, u, y_j; w) / t) r(y_j)  /  sum_j omega_j exp(M(s, u, y_j; w) / t)
u*         = argmax_k  max_z  Q(s, u_k; w_z),   u_k ~ N(0, I) clipped, k = 1..64
```

with w_z = h(g_c) over the 32 rewarding goals for the goal index (GPI over goals, as now) and
the single w for the bank index. n_r = min(pool_eval_size / 4, number of rewarding rows).
Defaults: pool_eval_size 512, so one environment step costs 64 x 32 x 512 triple evaluations
for the goal index (2.3 times the current readout) and 64 x 512 for the bank.

## Diagnostics the training log must carry

| key | meaning | arm A / nf2 for comparison |
|---|---|---|
| `measure_p_anchor` | mean mass on the anchor state | `measure_p_own_next` 0.45 to 0.49 / 0.34 |
| `pool_entropy` | mean entropy of p_i over R, nats; max ln P = 6.93 at 1024 | fullness H(p) 6.79 / 5.5 |
| `pool_u_tv` | mean total variation between p_i under u_i and under a fresh clipped prior draw, first 32 rows | trajectory-mass ratio 1.00 to 1.05 (no u dependence) |
| `phi_eff_rank` | as now | 1.0 to 1.1 |

`pool_u_tv` is the number that says whether the measure depends on the action. It is computed
every log interval, on 32 rows, with the online network, no gradient.

## Expected outcome, stated before results

| arm | fullness tv(p, rho) against the arm's own policy (noise floor 0.35) | pool_u_tv | best-of-64 five-task cube |
|---|---|---|---|
| pool, bank (BC) | 0.5 to 0.7 at 250k (from 0.9) | above 0.05 by 100k | 0.6 to 0.72, the data-bootstrap and one-draw level |
| pool, goal, frozen field | 0.5 to 0.7 | above 0.05 | 0.70 to 0.80 under hgoal_each; pool_reward within 0.05 of it |
| pool, goal, live actor | as above | as above | as above |

The construction fails if tv stays above 0.85 and pool_u_tv stays below 0.02 at 250k on the
bank arm: then the triple network does not transfer even when forced, and the next lever is the
network (a y-embedding shared across (s, u)), which this document does not authorise. The
construction succeeds on fullness and fails on success if tv falls below 0.6 and five-task
success drops below 0.6: then the detector's success came from something the full measure
does not keep, and the readout, not the measure, is the next object. Antmaze: the same arms
after cube, expected to move the own-anchor share off 0.08 for the first time; success
unknown.

## Cost

Per update, N x P = 256 x 1024 triple evaluations online and the same for the target, against
256 x 256 for A and 256 x 768 for nf2: about 3 times A's step, 0.06 s on an H100, 8.5 h per
500k. `pool_size` 512 halves it and is the fallback if the queue requires a 10 h limit.

## Configuration keys (all new keys default to the old behaviour)

| key | default | meaning |
|---|---|---|
| `agent.measure_loss` | squared | new value `pool` |
| `agent.pool_size` | 1024 | P, states in R |
| `agent.pool_seed` | 0 | seed of the draw of R from the dataset's valid-preimage rows |
| `agent.policy_index` | goal | new value `bank` |
| `agent.n_bank` | 1 | members of the bank; only 1 is implemented |
| `agent.coef_source` (eval) | lp | new value `pool_reward` |
| `agent.pool_eval_size` (eval) | 512 | bank size of the pool_reward readout |

R must be reproducible from (dataset, pool_seed) alone, so that `tools/eval_checkpoint.py`
rebuilds the same R without reading it from the checkpoint; the training log records
`pool_hash` (sha1 of R's row indices) and eval asserts it matches when the checkpoint carries it.
`LEGACY_AGENT_DEFAULTS` in `tools/eval_checkpoint.py` gets `measure_loss: squared`,
`pool_size: 0`, `policy_index: ...` as already there, so old checkpoints restore unchanged.

## Record

Plan: `docs/plans/2026-10-19-pool-measure.md`. Arms: `psmgoal_pool_bank_cube`,
`psmgoal_pool_fdplus_cube`, `psmgoal_pool_ja_cube`, 3 seeds, 500k, eval500 at 250k and 500k with
both readouts; fullness (`tools/diag_measure_fullness.py`, `+rollout_policy=bc` for the bank arm,
`actor` for the live-actor arm, `bon64` for all) on the first 250k checkpoint before any success
number is quoted. Companion work the same day: the actor recipe and nf2 on antmaze (kisski jobs
2525719 to 2525724), the distillation follow-ups on luis.

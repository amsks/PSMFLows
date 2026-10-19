# 2026-10-19 — the reward index: Density FB's z arm in psmgoal

## Why

Density FB (Factored-FB, branch `density-fb`, `impls/critics/density_fb.py`) has a second arm,
`measure_density_z`, in which the critic's task index is not a goal but a vector inferred from
reward samples by a centred ridge solve on the state tower, the same solve at training time (on
hindsight kernel rewards) and at test time (on the relabelled rewards). On cube-single it scores
82.9 ± 4.0 (5 seeds, state tower on cube position) and 91.7 ± 2.2 (state tower on the full state)
against 93.1 ± 1.8 for the goal arm. The loss is the goal arm's: softmax over the batch's next
states, one-hot at the own next state (`measure_density_hard: true` in every shipped cube run).

psmgoal's goal arm (critic A, 0.805; nf2 0.866) is that recipe. Its test-time coefficients from
rewards (`coef_source=regression`, 0.30; `amortized`) were never trained on: the critic saw
w = h(g) in training and a regressed w at test time. The 2026-09-30 "Factored-FB trick" arm
trained the ridge index but on a BC measure (policy_index=code, squared loss, no bootstrap from
the actor at the index) and scored 0.07. This arm puts the index inside the goal arm's recipe.

## What changes

One new policy index, `policy_index=reward`. Everything else is critic A's.

- Index feature: f(y) = `fb_feature` (the existing projected phi: psm_norm of the mean of
  phi(y, u, a) over fb_k_u fixed prior latents and fb_k_anchor fixed anchor states, read at the
  slow basis copy `fb_zenc`). No new network. mu, C are the existing EMA moments.
- Training rows: w_i = ridge(f(batch next states), r_i) with r_i the kernel reward
  (`fb_hit_spec`, cube xyz, width 0.4) of the row's measure goal g_i (the geometric hindsight goal
  as in A) on the batch's next states; with probability fb_rand_frac = 0.5 replaced by a random
  sqrt(D)-sphere draw. w_i enters M = phi^T w_i + b and the target, stop-gradded (the index is a
  solve, not a net; no Polyak copy).
- Loss: the softmax loss of A over the batch's next states, one-hot at the own next state.
- Bootstrap: the flowbc actor at (s'_i, w_i): u' = clip(eps + delta(s', w_i, eps)). delta takes w
  (z_dim) in its goal slot.
- Actor: flowbc, rows indexed by w_a from the same solve on the actor goal (`fb_goals`, a uniform
  future state) with its own random-draw mix; value Q = sum_j softmax_j(M(s,u,y_j; w_a)/t) r(y_j)
  with r(y) = (f(y) - mu)^T w_a / sqrt(D), the index's own reward view (Density FB's shipped
  `latent_centered`); loss -q_coeff Q/sg|Q| + fb_bc_coeff mean delta^2 as in A.
- Index state: after each update, fb_zenc EMA toward the pre-step basis (rate fb_enc_tau), mu and
  C EMA over the batch's f (rate fb_sigma_tau), anchors fixed from the first batch (as the
  2026-09-30 arm did; `fb_actor_step` order).
- Test time: `coef_source=ridge`, w = the same solve on the relabel batch's rewards (10,000 rows,
  shifted). Readouts: `acting=distill` with the flowbc actor at w (Density FB's own readout);
  `acting=gpi` best-of-64 by the softmax mass on the 32 rewarding goals at w (A's readout at the
  ridge w, `_goal_score`); `ridge_score=reward` best-of-64 by the expected relabel reward over
  eval_ref at w (the reward integral).

## Diagnostics the training log must carry

| key | meaning | gate |
|---|---|---|
| `index_proxy_corr` | correlation on the hindsight rows between (f(y_j) - mu)^T w_i and the kernel reward k(g_i, y_j), over the batch's next states | above 0.3 by 100k, else the index space does not carry the reward |
| `index_w_norm`, `index_hindsight_frac`, `index_cov_trace` | as the 09-30 arm | |
| `measure_p_own_next`, `phi_eff_rank`, `actor_delta_norm` | as A | p_own 0.3 to 0.5 (A's regime) |

## Expected outcome, stated before results

Five-task cube, 500 episodes, 3 seeds, 250k: best-of-64 at the ridge w 0.70 to 0.80; the actor at
w within 0.10 of it; `index_proxy_corr` above 0.5 by 250k. Failure line: at or below 0.30 (the
regression coefficient of 10-01), which reads as "the projected-phi index space does not carry
the reward the way Density FB's state tower does", and the next lever is the index feature. A
result between 0.30 and 0.70 with proxy_corr above 0.5 reads as an actor or bootstrap gap, and
the direction diagnostic of 10-17 applies.

## Keys

`policy_index=reward` (new value); `ridge_score: goal | reward` (eval, new, default goal). All
existing: `coef_source=ridge`, `fb_*`, `actor_kind=flowbc`, `fb_bc_coeff`, `q_coeff`,
`measure_loss=softmax`, `measure_temp`, `goal_sampling=geometric`, `goal_random_frac=0`,
`goal_cur_frac=0`. Asserts: reward index needs measure_loss=softmax, train_actor=true,
actor_kind=flowbc, actor_input=w, bootstrap_source=actor, actor_value=none, norm_future_k=0 (first
arm), coef_source in (ridge, lp) at eval (lp only as a diagnostic).

## Record

Arm `psmgoal_ri_fbc3_cube` (reward index, flowbc actor, fb_bc_coeff 3), seeds 0-2, 250k first
(the user asked for results within six hours of 18:15 CEST), eval500 at 250k under the three
readouts; luis runs the training on the H200s, kisski is full with the pool runs until about
01:00. Companion: `docs/HANDOFF.md` 2026-10-19 entry.

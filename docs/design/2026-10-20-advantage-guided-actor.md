# The advantage-guided actor (2026-10-20)

User decision 03:30 CEST, results wanted by 09:30 CEST. Reference: Zhou & Levine, "Reinforcement Learning via
Advantage Guidance" (arXiv 2610.19783); their policy is a residual flow on a frozen BC flow, conditioned on the
per-state standardised advantage, trained by supervised flow matching on the dataset action plus K imagined BC
draws, and sampled with the contrastive guidance v_BC + w (v(y+) - v(y-)), y+ = 1, y- = -1.

## Why

The bootstrap ladder (HANDOFF 10-20 03:00) says what critic A's actor gives the critic: a consistent small shift
of the bootstrap latent (82% a fixed vector per state and goal) that is in-support, barely better by the critic's
own score (+0.003, a 52nd-percentile BC draw) and nearly invisible in the TD target per step; the critic trained
with it acts 0.05 better than one without, and the critic trained with the better of two draws (a larger but
draw-dependent improvement) collapsed. The deployed policy (student of best-of-64, 0.91) is a second network
trained after the critic. One network that (a) gives the bootstrap a stable, tilted-toward-value action with
a knob for how far, and (b) is the deployment policy at a larger knob, removes the second network and the
argmax from training.

## What changes (`agents/psmgoal.py`, `ag_actor=true`; default off, every existing run unchanged)

Critic A's recipe (policy_index goal, softmax loss, flowbc actor, point value, bootstrap from the actor) with
the actor's training and read-out replaced:

- The delta net's goal slot takes [g, c], c a scalar; mu(s, g, c) := delta(s, [g, c], eps = 0).
- Actor step: K = ag_k (64) prior draws u_k = clip(eps_k) per row at the actor goal; q_k = the point value of
  (s, u_k) at w = h(g) under the stop-gradded online basis (the value A's actor climbed); per row
  c_k = (q_k - mean_k q) / sqrt(var_k q + eps). Loss = mean_k mean_i ||mu(s_i, g_i, c_ik) - u_ik||^2. No
  critic gradient, no BC term (every target is a BC draw).
- Slow copy: delta_target <- Polyak(delta, ag_delta_tau = 0.005), a new agent field (None when off).
- TD bootstrap: u' = clip(eps + ag_w_boot (mu(s', g, +1) - mu(s', g, -1))) read from delta_target, ag_w_boot 0.5.
- Acting (acting=distill): u = clip(eps + ag_w_eval (mu(+1) - mu(-1))) from delta, ag_w_eval a typed eval
  override; 0 is BC, best-of-64 (hgoal_each) unchanged.

Deviation from the paper, stated: their guidance acts on the velocity along the denoising path and is the exact
KL tilt; ours acts on the start noise of the frozen flow once, so the +1 minus -1 shift is a heuristic in the
same direction (the latent direction from low- to high-scoring draws at the state). No action-chunking, no
dataset action in the K set (the recorded latent is not needed for the bootstrap).

## Diagnostics (train.csv)

actor_loss (the regression); actor_delta_norm (the bootstrap shift's norm at ag_w_boot); actor_q (point value
of the guided first draw); ag_q_gain (guided minus the plain first draw); ag_q_gain_bok (best of K minus the
mean draw, the ceiling); ag_q_draw_std; ag_c_absmean (about 0.8 for a standardised score); ag_shift_cos_best
(cosine between the shift and the direction from the first draw to the best draw).

## Expected outcome (stated before results), five-task cube, 500 episodes, 250k, 3 seeds

- best-of-64 (hgoal_each): at or above A's 0.79 (A at 250k) if the guided bootstrap is at least as good a
  policy for the critic as the nudge; 0.66 to 0.72 (B's level) if the regression actor is too slow to help;
  below 0.5 reads collapse (the tilt at w_boot 0.5 too large for the target).
- the guided actor alone: 0.5 to 0.8 at the best ag_w_eval in {0.5, 1, 2, 4}, rising with w until it leaves
  the data; within 0.05 of BC (0.11) at every w reads "the actor ignores the score" (their naive-conditioning
  failure).
- ag_q_gain at w_boot 0.5 between A's nudge (+0.003) and best-of-2 (+0.004) by 100k; ag_shift_cos_best above
  0.3 by 100k.

## Record

Arm `psmgoal_ag_cube` seeds 0-2, 250k, kisski (launcher `scripts/campaign_2026-10/ag_arms.sh`), one extra seed
at ag_w_boot 0.25 (`psmgoal_ag_wb025_cube` sd0) as the tilt-size control; watcher
`scripts/campaign_2026-10/ag_eval_watcher_cube.sh` (hgoal_each + ag_w{0.5,1,2,4}). Tests
`tests/test_psmgoal_ag_actor.py`.

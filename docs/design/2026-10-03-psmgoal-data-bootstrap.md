# psmgoal, goal-conditioned measure with the data's next action as bootstrap

Date: 2026-10-03. Status: audited (correct with conditions), approved in chat; implementation behind
default-off keys. Extends `docs/design/2026-10-01-psmgoal-goal-conditioned.md`.

## Why

In every psmgoal version so far the policy inside M was defined by us: a fresh random latent per
row (= BC on non-repeating states), one latent held for the episode, or a goal-fed actor that sits
near u = 0. None reaches goals. The OGBench data itself is a mixture of goal-reaching policies
(navigate: a noisy expert heading to random goals; play: a scripted controller doing random
pick-and-place; arXiv 2410.20092). So the family we want is already in the data: for goal g, the
policy pi_g = "the move the data makes at s on trajectories that reach g at offset >= 0".

## Notation

u action latent; u' next action; s, s', s+ states; g goal; phi(s,u,s+), b(s,u,s+); w = h(g);
M(s,u,s+) = phi^T w + b; gamma discount. Row i of the dataset is (s_i, u_i, s'_i); u_{i+1} is the
latent of the move the data made at s'_i.

## What changes

1. **Bootstrap.** Under `policy_index=goal` with `bootstrap_source=data`, the bootstrap latent is
   u'_i = u_{i+1} (`batch['next_noise_preimage']`) instead of the actor's output. The target is
   `(1-gamma)[s+ = s'_i] + gamma M(s'_i, u_{i+1}, s+)` at w = h(g_i). No actor is trained.
2. **Goals.** Only hindsight goals from the row's own future, geometric offset at `goal_discount`
   (`goal_random_frac=0`, `goal_cur_frac=0`). The audit shows: for g_i != s'_i, (s'_i, u_{i+1}, g_i)
   has exactly the offset distribution a fresh draw at row i+1 would have, so u_{i+1} is a sample of
   pi_g at s'_i. For g_i = s'_i (2% of rows) u_{i+1} is "what the data does after arriving".
3. **Sampling restrictions.** Row i is not sampled if `terminals[i] > 0` (time-limit end, no next
   row; 1000 rows) or if `preimage_valid[i+1] == 0` (13 cube, 881 antmaze; their stored latent is 0).
   Rows already excluded as row i (invalid preimage) stay excluded.
4. **Losses.** `measure_loss=squared` and `measure_loss=softmax` unchanged; they take `u_next` as an
   argument.
5. **Readouts.** `hgoal` as before (w = normalised mean of h over the 32 rewarding states, best of
   64 prior draws by M); new `hgoal_each`: each goal scored with its own coefficient,
   `score(u) = mean_j M(s,u,g_j; h(g_j))` (softmax: for each j the share of g_j among
   [g_j + reference states] at w = h(g_j), averaged over j). The audit flags the averaged h as an
   approximation; `hgoal_each` is the exact per-goal reading. `lp` kept for the ablation.

## Keys

| key | default | meaning |
|---|---|---|
| `bootstrap_source` | `actor` | `data`: item 1; requires `policy_index=goal`; `train_actor` may be false |
| `coef_source=hgoal_each` | (value) | item 5, eval only |
| `Dataset.return_next_preimage` | off | emits `next_noise_preimage`; row sampling per item 3 |

Defaults leave the agent and the dataset sampler bit-identical (digest test must pass).

## Runs

Cube (discount 0.98) and antmaze (discount 0.99, goal_discount 0.99, as the gc antmaze runs), 500k
steps, seeds 0 1 2, checkpoints 250k and 500k.

| group | overrides |
|---|---|
| `psmgoal_db_sq_<env>` | `policy_index=goal bootstrap_source=data train_actor=false goal_random_frac=0 goal_cur_frac=0 measure_loss=squared` |
| `psmgoal_db_sm_<env>` | same with `measure_loss=softmax` |

Eval: `hgoal` and `hgoal_each` at 250k and 500k; `lp` at 250k. 500 episodes per task, five tasks.
Quote BC beside every number (cube 0.111, antmaze 0.07-0.09).

## Expected outcomes, stated before launch

- Cube softmax, hgoal: above 0.665 (the goal-label run with the stuck actor). Squared: above 0.405.
- `hgoal_each` at or above `hgoal`.
- Antmaze: unknown. The data policy reaches goals, but one ant move changes little; I expect a gain
  over BC (0.07-0.09) but not a cube-like number.
- Failure that refutes the idea: cube softmax hgoal at or below 0.665 with three seeds agreeing.

## Risks (from the audit)

- 43.5% of cube rows have the goal in a later pick-place segment; u_{i+1} then serves the current
  sub-target. The measure stays that of pi_g as defined, but M will be low and flat over u for far g.
- With random goals off, h(g) is never trained on (s, g) from different trajectories; at test time
  s is a rollout state and g a dataset state. phi and b still see all s+ as columns.
- pi_g at s' exists only in hindsight; states with few trajectories reaching g give noisy targets.
- Candidate u at test time are prior draws, not pi_g's moves; their ranking rests on generalisation,
  as before.

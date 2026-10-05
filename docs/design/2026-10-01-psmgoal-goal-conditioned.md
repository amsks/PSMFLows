# psmgoal, goal-conditioned training

Date: 2026-10-01. Status: approved in chat; implementation behind default-off switches.

## Why

Measured on the current psmgoal (cube, `docs/design/2026-09-30-psmgoal-tests.md`, memory
`psmgoal-failing-point-audit`):

- Every policy index denotes the behaviour-cloning policy. M has no improved policy inside it.
- M changes with u by under 1% of its variance within a state.
- M is at its head bound at the row's own next state and about 0.005 at other states.

Goal-conditioned Density FB (Factored-FB, branch `density-fb`, `impls/critics/density_fb.py`)
scores 90.8 on cube-single with the full state as s+ where plain FB scores 17.7 (raw actions,
goal-conditioned eval, 3 seeds). It has three things psmgoal never ran together: an actor that
takes the goal, a bootstrap from that actor at the same goal, and a softmax loss over the batch's
next states. This arm brings those into psmgoal. Two training runs differ only in the loss.

## Notation

u action (flow latent); u' next action; s, s', s+ states; g goal; phi(s,u,s+), b(s,u,s+);
w = h(g), normalised as the code already normalises w; M(s,u,s+) = phi(s,u,s+)^T w + b(s,u,s+);
l multiplier. No other symbols.

## What changes (all behind keys; defaults leave the agent bit-identical)

1. **Goal per row.** Measure goal g_i: 20% the row's own s', 50% a future state on the same
   trajectory (geometric offset, existing `goal_discount`), 30% a random row. Actor goal: a
   uniformly drawn future state on the same trajectory (`batch['fb_goals']`).
2. **Policy index = goal.** The measure loss uses w_i = h(g_i) (online h for M, a new Polyak
   target copy of h for the target M). h is trained by the measure loss gradient, as w(code) is
   today. `goal_head_loss` is not used in this arm.
3. **Actor fed the goal.** actor(s, g) -> u, raw goal observation as input. Loss:
   `-q_coeff * Q / sg|Q| + actor_temp * logp + fb_bc_coeff * mean (u - u_data)^2`, `fb_bc_coeff` 3.0,
   measure frozen for this loss. Q is the reward-weighted measure over the batch's next states,
   with the reward a normalised Gaussian kernel around the actor goal on the success coordinates
   (`fb_hit_spec`, cube xyz, width 0.4):
   - squared run: `Q = sum_j M(s,u,s+_j) k(g, s+_j)`
   - softmax run: `Q = sum_j softmax_j(M(s,u,s+_j) / measure_temp) k(g, s+_j)`
4. **Bootstrap.** u'_i = actor mode at (s'_i, g_i), the same measure goal as the row.
5. **Loss, two runs.**
   - `measure_loss=squared`: the existing loss, unchanged. `0.5 mean_{i!=j} (M_ij - discount * Mbar_ij)^2 - (1-discount) mean_i M_ii`.
   - `measure_loss=softmax`: cross-entropy between `softmax_j(M_ij / measure_temp)` and the target
     `(1-discount) * onehot(j=i) + discount * softmax_j(Mbar_ij / measure_temp)`, target stop-gradded.
     Columns are the batch's next states in both.
6. **Test time, same checkpoint, four readouts.**
   | tag | coefficient | acting |
   |---|---|---|
   | hgoal | w = normalised mean of h over the 32 rewarding states | best of 64 prior draws by M |
   | lp | Lagrangian w (existing loop) | best of 64 prior draws by M |
   | actor_rel | none | actor fed one rewarding dataset state |
   | actor_env | none | actor fed the env's task goal (`info['goal']`) |
   Under `measure_loss=softmax` the best-of-64 score is the softmax mass on the goal columns over
   [goal columns + reference next states], because the logits are only defined up to a per-(s,u)
   shift. Reference columns: `batch_size - k_goals` non-rewarding next states of the relabel batch.
   The Lagrangian under softmax maximises that same mass with the penalty term off (softmax mass
   is non-negative for every w); it is a diagnostic there.

## Keys (in both `get_config()` and `configs/agent/psmgoal.yaml`)

| key | default | meaning |
|---|---|---|
| `policy_index` | `code` | `goal`: items 1-4. Requires `train_actor=true`. |
| `measure_loss` | `squared` | `softmax`: item 5. |
| `measure_temp` | `1.0` | softmax temperature, loss and readouts |
| `actor_input` | `w` | `goal`: actor takes the raw goal |
| `goal_cur_frac` | `0.0` | share of rows whose measure goal is their own s' |
| `eval_goal_source` | `relabel` | `env`: actor fed the env task goal |

Reused unchanged: `train_actor`, `goal_discount`, `goal_random_frac`, `fb_bc_coeff`, `q_coeff`,
`actor_temp`, `fb_hit_spec`, `acting`, `eval_redistill`, `coef_source`, `tau`, `discount`,
`batch_size`. No existing default value changes.

Asserts in `create`: `policy_index=goal` with `train_goal_head=true`; `policy_index=goal` with
`actor_value != none`; `measure_loss=softmax` with `measure_form=factorized`; `measure_loss=softmax`
with `ortho_coef != 0`; `actor_input=goal` with `eval_redistill=true` or `acting=sfbc`;
`policy_index=goal` without `train_actor=true`.

## Code mapping

- `agents/psmgoal.py`: field `target_w_star`; `_coef(z, goals, params, target)` used by
  `measure_loss`, `_select_u_next`, `total_loss`; pure `softmax_td_loss`; `actor_loss_goal`;
  `apply_update` routes the measure-loss gradient to h and Polyak-updates its target;
  `_goal_score` used by `select_latent` and `_obj_and_constraint`; `sample_actions` and
  `infer_eval_goals` for the actor goal and the reference columns. No extra `split(rng)` on the
  default path. A checkpoint without `target_w_star` must restore.
- `utils/datasets.py`: `cur_frac` argument to `hindsight_goal_idxs`, drawing no extra random
  numbers at 0.
- `main.py`: emit `goals` and `fb_goals` when `policy_index=goal`.
- `tools/eval_checkpoint.py`: env goal hand-off; record `coef_source`, `measure_loss`,
  `actor_input`, `acting`, `eval_goal_source` in the report.
- `scripts/slurm/launch_psmgoal_eval500.sh`: an arm list of (tag, override string), tag in the
  output name.
- Tests: `tests/test_psmgoal_goalcond.py` (defaults unchanged; goal shares; goal-mode loss equals
  a hand computation and does not depend on the code; Polyak of the h target; bootstrap uses the
  same goal; `softmax_td_loss` against numpy and shift invariance; actor loss gradients; actor
  input width; four readouts give finite actions; each assert fires; old checkpoint restores).
  `tests/test_psmgoal_fbtrick_defaults_off.py` must pass untouched.

## Runs

Cube-single-play, 500k steps, seeds 0 1 2, checkpoints at 250k and 500k.

| group | overrides |
|---|---|
| `psmgoal_gc_sq_cube` | `policy_index=goal train_actor=true actor_input=goal goal_cur_frac=0.2 goal_random_frac=0.3 measure_loss=squared` |
| `psmgoal_gc_sm_cube` | same with `measure_loss=softmax` |

Eval: 500 episodes per task, five tasks, the four readouts, both checkpoints. Quote BC (0.111)
beside every number. Five-task mean and 95% CI across seeds only.

## Expected outcomes, stated before any run

- Squared run: M still at the head bound at the row's own next state (the loss and the
  never-repeating states are unchanged). Acting above the current default (lp 0.001) because the
  policy inside M now improves towards the goal; well below the Density FB numbers.
- Softmax run: the larger gain. M's softmax mass spread over more than one column.
- `lp` against `hgoal`: no prediction for the squared run. Under softmax the Lagrangian has no
  active constraint, so I expect `lp` at or below `hgoal`.
- Failure that would refute the plan: both runs at or below BC on all four readouts.

## Risks

- The actor sits inside the bootstrap. The Density FB config records this as the cause of a
  collapse on another task. The brakes are the BC pull (3.0) and `u_clip`.
- 30% of measure goals are random rows; the actor is queried at goals it was not trained on, as
  in Density FB.
- The head is RMSNorm-bounded (|M| <= 129, spread about 11 at init). At temperature 1 the softmax
  may start near one-hot. Check the logit spread in the smoke run.
- The kernel in the actor's Q uses the success coordinates (cube xyz), as Density FB does.
- Density FB's numbers are with raw actions and its own eval protocol; these runs go through
  the frozen flow and the 500-episode protocol.
- `restore_agent` keeps fresh weights on a shape mismatch with only a warning. An eval with the
  wrong `actor_input` would run an untrained actor; the report must record `actor_input`.
- The softmax actor loss adds a second batch x batch mesh with gradients. Expect about double
  the 3 h training time.

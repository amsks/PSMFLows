# psmgoal with the softmax measure loss: results 2026-10-01..07

All numbers: cube-single-play, mean success over the five tasks, 500 episodes per task, seeds
0 1 2 unless a row says otherwise. Behaviour cloning (the frozen flow acting alone) scores
0.111 on cube and 0.07-0.09 on antmaze. Means come from `scripts/campaign_2026-10/agg_results.py`
(run 2026-10-07 over `$PSM_DATA/logs/*.json`); the 95% interval is the t-interval over the
three per-seed means. The code-label groups (`psmgoal_lift_cube_gc`, `psmgoal_sm_code_cube`)
are not matched by that script's file pattern and were aggregated with the same code and a
wider pattern.

## Current best agent

Group `psmgoal_ja_fbc3_pt_cube`. Five-task mean, best of 64 by M with per-goal h:

| step | mean | 95% interval | seeds |
|---|---|---|---|
| 250k | 0.794 | ± 0.134 | 0.818 / 0.831 / 0.732 |
| 500k | 0.804 | ± 0.104 | 0.756 / 0.825 / 0.832 |

The same checkpoints with the actor acting alone score 0.108 (250k and 500k).

What the agent is:

- **Measure.** M(s, u, s+) = phi(s, u, s+)^T w + b(s, u, s+) on the frozen flow's latents.
- **Policy label.** The goal. The row's coefficient is w = h(g), h a learned network of the goal
  state (`policy_index=goal`).
- **Measure goals.** Hindsight goals from the row's own trajectory at a geometric offset
  (`goal_sampling=geometric`, `goal_discount=0.98`). No random goals and no current-state goals
  (`goal_random_frac=0`, `goal_cur_frac=0`).
- **Measure loss.** Softmax over the batch's next states (`measure_loss=softmax`, `measure_temp=1`).
- **Bootstrap action.** u' at (s', g) comes from a flowbc actor, u = clip(eps + Delta(s, g, eps)),
  eps a fresh prior draw (`bootstrap_source=actor actor_kind=flowbc`).
- **Actor training.** Loss = -Q / mean|Q| + 3 * mean(Delta^2) (`fb_bc_coeff=3.0`). Q is the
  softmax share M puts on the actor goal g among {g} plus the batch's next states
  (`actor_value_kind=point`). The actor goal is a uniform draw from the row's own future
  (`batch['fb_goals']`), not the geometric measure goal. Only Delta moves; phi, b and h are
  held fixed in the actor step.
- **Acting.** Draw 64 clipped prior latents at the current state. Score each by the softmax
  share M puts on each of 32 rewarding goals among that goal plus 224 non-rewarding reference
  states, each goal read with its own h(g); average over goals; decode the best
  (`coef_source=hgoal_each acting=gpi`, `gpi_num_u=64`).

Training overrides on top of `configs/agent/psmgoal.yaml` (from `scripts/campaign_2026-10/ja_arms.sh`,
confirmed in the run's `flags.json`):

```
agent.policy_index=goal agent.bootstrap_source=actor agent.train_actor=true agent.actor_input=goal
agent.goal_random_frac=0.0 agent.goal_cur_frac=0.0 agent.goal_sampling=geometric
agent.measure_loss=softmax agent.acting=distill agent.eval_redistill=false agent.eval_goal_source=relabel
agent.actor_kind=flowbc agent.fb_bc_coeff=3.0 agent.actor_value_kind=point
```

Values left at the yaml default in that run: `discount=0.98`, `goal_discount=0.98`,
`measure_temp=1.0`, `z_dim=128`, `batch_size=256`, `k_goals=32`, `gpi_num_u=64`, `q_coeff=1.0`.
`acting=distill` in training only sets the in-loop 50-episode eval. The readout above is the
eval override `agent.coef_source=hgoal_each agent.acting=gpi`.

## The three places the softmax is used

1. **M's loss.** Row i is (s_i, u_i) with w = h(g_i); the columns are the batch's next states.
   The loss is the cross-entropy between softmax_j(M_ij) and the target
   (1 - discount) * [j = i] + discount * softmax_j(M_bar(s'_i, u'_i, s'_j)), M_bar the target
   network (`softmax_td_loss`).
2. **The actor's value** (`actor_value_kind=point`). Q_i = exp(M(s_i, u_i, g_i)) / (exp(M(s_i, u_i, g_i))
   + sum_j exp(M(s_i, u_i, s'_j))), the share of the actor goal among the goal plus the batch's
   next states.
3. **Test-time scoring** (`_hgoal_each_score`). For each goal g_j the share of g_j among g_j plus
   224 reference states, read at w = h(g_j), averaged over the 32 goals.

Scoring must use shares. The softmax loss only sees M through softmax_j(M_ij). Adding any number
c(s, u) to every column of row i leaves the loss unchanged. So the raw value M(s, u, g) carries an
arbitrary per-(s, u) offset, and comparing raw M across candidate latents u compares those
offsets. A share divides the offset out, so it is comparable across u.

## Every cube arm, 2026-10-01..07

Readouts: `trained` = mean of the training coefficient w(z) (code label); `hgoal` = mean of h over
the goal set, best of 64; `hgoal_each` = per-goal h, best of 64; `lp` = the Lagrangian coefficient,
best of 64; `actor_rel` = the actor fed one rewarding dataset state. A dash means not evaluated.

| arm | group | readout | 250k | 500k | lp 250k | lp 500k |
|---|---|---|---|---|---|---|
| code label, squared | `psmgoal_lift_cube_gc` | trained | 0.327 | 0.240 | - | - |
| code label, softmax | `psmgoal_sm_code_cube` | trained | 0.524 | 0.383 | 0.119 | 0.265 |
| goal label + tanh actor, squared | `psmgoal_gc_sq_cube` | hgoal | 0.405 | 0.277 | 0.000 | 0.000 |
| goal label + tanh actor, softmax | `psmgoal_gc_sm_cube` | hgoal | 0.665 | 0.594 | 0.367 | 0.308 |
| | | actor_rel | 0.137 | 0.144 | | |
| data bootstrap, squared | `psmgoal_db_sq_cube` | hgoal | 0.597 | 0.516 | 0.000 | - |
| | | hgoal_each | 0.602 | 0.520 | | |
| data bootstrap, softmax | `psmgoal_db_sm_cube` | hgoal | 0.412 | 0.446 | 0.26 (1 seed) | - |
| | | hgoal_each | 0.619 | 0.646 | | |
| uniform goals, squared | `psmgoal_dbu_sq_cube` | hgoal | 0.482 | 0.436 | 0.000 | - |
| | | hgoal_each | 0.479 | 0.448 | | |
| uniform goals, softmax | `psmgoal_dbu_sm_cube` | hgoal | 0.534 | 0.477 | 0.285 | - |
| | | hgoal_each | 0.506 | 0.501 | | |
| orthonormal phi, squared, data bootstrap | `psmgoal_dbo_sq_cube` | hgoal | 0.062 | 0.036 | 0.031 | 0.061 |
| | | hgoal_each | 0.068 | 0.039 | | |
| joint flowbc actor, bc 3, shaped value | `psmgoal_ja_fbc3_sh_cube` | hgoal_each | 0.669 | 0.655 | - | - |
| | | actor_rel | 0.106 | 0.103 | | |
| joint flowbc actor, bc 0.3, shaped value | `psmgoal_ja_fbc03_sh_cube` | hgoal_each | 0.758 | 0.706 | - | - |
| | | actor_rel | 0.132 | 0.117 | | |
| **joint flowbc actor, bc 3, point value** | `psmgoal_ja_fbc3_pt_cube` | hgoal_each | **0.794** | **0.804** | - | - |
| | | actor_rel | 0.108 | 0.108 | | |
| joint flowbc actor, bc 0, shaped value (2 seeds) | `psmgoal_ja_fbc0_sh_cube` | hgoal_each | 0.021 | 0.033 | - | - |
| | | actor_rel | 0.001 | 0.003 | | |
| joint DSRL actor (no BC), shaped value | `psmgoal_ja_dsrl_sh_cube` | hgoal_each | 0.014 | 0.062 | - | - |
| | | actor_rel | 0.041 | 0.088 | | |
| behaviour cloning | | | 0.111 | 0.111 | | |

Other readouts on these groups:

| test | result |
|---|---|
| `psmgoal_gc_sq_cube` actor_env (actor fed the env goal) 250k / 500k | 0.110 / 0.098 |
| `psmgoal_gc_sm_cube` actor_env 250k / 500k | 0.131 / 0.116 |
| `psmgoal_db_sq_cube` 250k, hold the chosen latent 4 env steps (2 seeds) | 0.440 |
| `psmgoal_db_sq_cube` 250k, hold 8 env steps (1 seed) | 0.135 |
| `psmgoal_lift_cube_gc` regression w 250k / 500k | 0.079 / 0.176 |
| `psmgoal_lift_cube_gc` 750k, 10,000 inference goals: lp / regression | 0.000 / 0.288 |

95% intervals of the main rows (hgoal_each, or hgoal where that is the only readout):

| group | 250k | 500k |
|---|---|---|
| `psmgoal_ja_fbc3_pt_cube` | 0.794 ± 0.134 | 0.804 ± 0.104 |
| `psmgoal_ja_fbc03_sh_cube` | 0.758 ± 0.062 | 0.706 ± 0.127 |
| `psmgoal_ja_fbc3_sh_cube` | 0.669 ± 0.314 | 0.655 ± 0.267 |
| `psmgoal_gc_sm_cube` (hgoal) | 0.665 ± 0.055 | 0.594 ± 0.181 |
| `psmgoal_db_sm_cube` | 0.619 ± 0.146 | 0.646 ± 0.286 |
| `psmgoal_db_sq_cube` | 0.602 ± 0.100 | 0.520 ± 0.420 |
| `psmgoal_sm_code_cube` (trained) | 0.524 ± 0.140 | 0.383 ± 0.048 |

The best three rows overlap within their intervals at three seeds.

## Actor and measure statistics, last logged training step (500k)

From the last row of each seed's `train.csv`; seeds listed in order 0 / 1 / 2.

| group | mean abs u per dim | mean norm of Delta | actor Q | M's softmax share on own next state | phi effective rank |
|---|---|---|---|---|---|
| `psmgoal_ja_fbc3_pt_cube` | 0.80 / 0.80 / 0.79 | 0.026 / 0.040 / 0.048 | 0.037 / 0.047 / 0.048 | 0.285 / 0.277 / 0.274 | 1.01 / 1.03 / 1.03 |
| `psmgoal_ja_fbc3_sh_cube` | 0.80 / 0.80 / 0.79 | 0.010 / 0.008 / 0.008 | 0.0077 / 0.0080 / 0.0093 | 0.234 / 0.258 / 0.266 | 1.01 / 1.03 / 1.03 |
| `psmgoal_ja_fbc03_sh_cube` | 0.80 / 0.80 / 0.79 | 0.052 / 0.043 / 0.052 | 0.011 / 0.013 / 0.013 | 0.292 / 0.321 / 0.315 | 1.02 / 1.05 / 1.04 |
| `psmgoal_ja_fbc0_sh_cube` (2 seeds) | 2.85 / 2.90 | 1730 / 3520 | 0.060 / 0.057 | 0.352 / 0.328 | 1.13 / 1.14 |
| `psmgoal_ja_dsrl_sh_cube` | 1.92 / 1.92 / 1.97 | - | 0.068 / 0.071 / 0.067 | 0.369 / 0.331 / 0.364 | 1.11 / 1.12 / 1.12 |
| `psmgoal_db_sm_cube` (no actor) | - | - | - | 0.318 / 0.334 / 0.331 | 1.06 / 1.08 / 1.12 |
| `psmgoal_dbo_sq_cube` (no actor) | - | - | - | - | 128 / 128 / 128 |

A clipped prior draw has mean abs u about 0.8 per dim, so the bc 3 and bc 0.3 actors stay at
prior scale and move eps by a small Delta. Q under the point value (a share on one goal) and Q
under the shaped value (a kernel-weighted sum) are different quantities and are not comparable
across those two columns. Without the BC term (bc 0, DSRL) the actor's latent leaves the prior
scale (mean abs u 1.9-2.9) and the measure collapses (0.01-0.06).

## Antmaze (antmaze-medium-navigate, discount 0.99)

| group | readout | 250k | 500k |
|---|---|---|---|
| `psmgoal_gc_sq_antmaze` | hgoal | 0.032 | 0.061 |
| | lp | 0.058 | 0.040 |
| | actor_rel / actor_env | 0.090 / 0.088 | 0.096 / 0.098 |
| `psmgoal_gc_sm_antmaze` | hgoal | 0.044 | 0.121 (0.008 / 0.185 / 0.170) |
| | lp | 0.018 | 0.002 |
| | actor_rel / actor_env | 0.089 / 0.095 | 0.090 / 0.091 |
| `psmgoal_db_sq_antmaze` | hgoal | 0.118 | 0.096 |
| | hgoal_each | 0.115 | 0.101 |
| `psmgoal_db_sm_antmaze` | hgoal | 0.095 | 0.051 |
| | hgoal_each | 0.086 | 0.088 |
| behaviour cloning | | 0.07-0.09 | |

Every antmaze arm and readout is at the BC level. M's softmax share on the own next state is
0.08-0.09 on antmaze (0.45-0.49 on cube in the gc runs).

## What is not working

- The actor acting alone scores 0.10-0.14 on cube in every arm and readout (gc, joint actor).
  Its gain shows only through the measure it bootstraps.
- Actors without a BC term collapse: bc 0 flowbc 0.021 / 0.033, DSRL 0.014 / 0.062.
- The Lagrangian coefficient (`lp`) scores below the h(goal) or trained readout of the same
  checkpoint on every cube run where both exist. The largest lp is 0.367 (`psmgoal_gc_sm_cube`
  250k, h(goal) 0.665). The one exception is the orthonormal arm at 500k (lp 0.061, h(goal)
  0.036), where both are below BC.
- Orthonormality on phi (`ortho_coef=1.0`) raises phi's effective rank from about 1 to 128. Acting
  falls to 0.068 / 0.039 and lp to 0.031 / 0.061. M stays at the head bound at the own next state
  (128.4-129.0) under the squared loss in that arm.
- Antmaze is unsolved (table above).
- Uniform goal sampling scores below geometric goals (sq 0.479 / 0.448 vs 0.602 / 0.520;
  sm 0.506 / 0.501 vs 0.619 / 0.646, hgoal_each).
- Holding the chosen latent for several env steps lowers success (1 / 4 / 8 steps: 0.602 / 0.440 / 0.135).
- phi's effective rank is 1.0-1.14 in every run without the orthonormality term.

## Caveats

- One dataset (cube-single-play) carries every positive result.
- Three seeds per arm (two for bc 0). The top three cube arms overlap within their 95% intervals.
- 500 episodes per task, five tasks. Two checkpoints (250k, 500k) per run; no later checkpoint.
- The pre-launch expectation (`docs/campaign_2026-10/ja_runs_EXPECTED.txt`) said the point value
  would score below the shaped value. It scored above (0.794 / 0.804 vs 0.669 / 0.655).

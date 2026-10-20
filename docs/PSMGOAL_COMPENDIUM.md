# psmgoal compendium: from the 2026-09-17 spec to nf2 (written 2026-10-17)

This document records every design decision, every arm, every diagnostic and every number on the
path from the goal-indexed successor measure spec (2026-09-17) to the winning configuration
`psmgoal_ja_fbc3_pt_cube` (2026-10-07), through the dissection of what its actor does
(2026-10-07 to 2026-10-17), to the own-future normalisation arm `norm_future_k=2` ("nf2",
2026-10-15/16). It is written so that a reader with no other context can reconstruct the
reasoning, re-run every number, and continue the work. `docs/COMPENDIUM.md` is the older
cold-start summary of the project up to 2026-09-01 and the comparator agent; this document
starts where that one's live-hypothesis list stops.

Conventions. Every success number is the cube-single-play five-task mean over 500 episodes per
task unless stated otherwise. The behaviour-cloning control, the frozen flow acting alone, is
0.111 on cube and 0.07 to 0.09 on antmaze. "250k / 500k" are training steps. "A" is
`psmgoal_ja_fbc3_pt_cube`, "B" is `psmgoal_ja_fbcBC_pt_cube`. Sources are given as `file:line`
in this checkout at commit e61ff1f, as HANDOFF entry dates, as mailbox message ids
(`<machine>/<utc>` in the HF dataset `amsks/psmflows-preimages`, folder `mailbox/`), or as JSON
paths under `$PSM_DATA/logs/` with `$PSM_DATA=/mnt/home/amohan/psm-data` on kisski.

Notation. State s, next state s', a later state of the same trajectory g or s+ (the goal when it
indexes the policy), action a, action latent u, bootstrap latent u', frozen behaviour flow
G(s,u) -> a, measure M(s,u,g) = phi(s,u,g)^T w + b(s,u,g), coefficient w, goal head h(g) so that
w = h(g), temperature t, discount gamma, Lagrange multiplier l, parameters theta. The flowbc
actor's prior draw is eps and its learned correction is delta, as in the code. N is the batch
size (256), D the width of phi and of h (128), d_a the action dimension (5 on cube).

Table of contents

1. Prehistory and the question
2. Objects and math, matched to the code
3. The path to the winning configuration, 2026-09-16 to 2026-10-07
4. Taking the actor apart, 2026-10-07 to 2026-10-17
5. The route to a full successor measure
6. nf2
7. Open questions and the next arms, with predictions
8. Proposed write-up
9. Appendix: readouts, configuration, locations, regeneration, record discrepancies

---

## 1. Prehistory and the question

### 1.1 What the project is

A behaviour-cloned conditional flow G(s,u) is fit on an offline dataset and frozen. Every action
the method evaluates, bootstraps or executes is a decode of some latent u through that flow, so
every action is in the support of the behaviour policy by construction. All reinforcement
learning happens in the latent space. The object learned is a successor measure over flow
latents: how much discounted future time the policy spends at states like g, starting from
state s with latent u. Zero-shot transfer means a task is specified at test time by a reward
over states, the measure is read against that reward, and the policy acts by picking, at each
step, the latent the measure ranks best among a set of prior draws.

The agent before psmgoal was the affine LatentFlowPSM (`agents/f_psmflow.py`), which indexes
the policy by a latent code and scored 0.284 on five-task cube on 2026-09-04. The comparator
stays in the repo. `docs/COMPENDIUM.md` sections 1 to 8 cover it.

### 1.2 Why a goal index

On 2026-09-15 and 09-16 the record established three facts about the code-indexed measure
(`docs/COMPENDIUM.md` section 6; memory of the 09-15 diagnostics):

- The index slot carried 92 to 99 percent of the within-state variance of the measure, while
  the latent u carried under 1 percent. The measure could not rank actions at a state.
- Every policy in the indexed family was the behaviour policy with a different random mode.
  Fixed-noise policies scored 0.006.
- The Lagrangian inference of the coefficient (PSM Eq. 10) had a linear objective and an inert
  constraint on a frozen head. Its coefficient pointed along the dominant direction of a
  rank-one feature matrix. Success 0.000 to 0.012.

The 2026-09-16 collation `docs/design/2026-09-16-goal-indexed-measure.md` proposed indexing the
policy by a goal state g through a head h(g), with the reward indicator r_g(s+) = 1[s+ = g] so
that the measure read at the goal is the goal-reaching value Q_g(s,u) = phi(s,u,g)^T h(g) +
b(s,u,g). The stated reason: a goal-indexed policy has a target to reach, so the measure of
that policy must depend on u.

### 1.3 The question this document answers

How did a goal-indexed measure trained with a FlowBC-style actor in its TD bootstrap reach
0.805 on five-task cube, what does the actor do, why is the trained measure not a successor
measure, and what did adding the transition's own later states to the training target do.

---

## 2. Objects and math, matched to the code

Everything in this section is for the configuration of arm A. File abbreviations: `P` =
`agents/psmgoal.py`, `Y` = `configs/agent/psmgoal.yaml`, `DS` = `utils/datasets.py`, `NW` =
`utils/psm_networks.py`, `MN` = `main.py`.

### 2.1 The overrides that define arm A

```
agent.policy_index=goal agent.bootstrap_source=actor agent.train_actor=true agent.actor_input=goal
agent.goal_random_frac=0.0 agent.goal_cur_frac=0.0 agent.goal_sampling=geometric
agent.measure_loss=softmax agent.acting=distill agent.eval_redistill=false agent.eval_goal_source=relabel
agent.actor_kind=flowbc agent.fb_bc_coeff=3.0 agent.actor_value_kind=point
```

`acting=distill` in training only sets the in-loop 50-episode evaluation. The scoring readout is
the evaluation override `agent.coef_source=hgoal_each agent.acting=gpi`. Arm nf2 adds
`agent.norm_future_k=2`. The sign-flip arm adds `agent.bootstrap_delta_scale=-1.0`. Defaults left
in place that matter: discount 0.98, goal_discount 0.98, measure_temp 1.0, z_dim 128,
batch_size 256, k_goals 32, gpi_num_u 64, q_coeff 1.0, measure hidden 1024 x 2, tau 0.01,
lr_measure 1e-4, lr_goal 1e-4, lr_actor 3e-4, u_clip 3.0. Section 9.2 lists all of them.

### 2.2 Networks

Measure network, phi and b (`RLUMeasure`, `NW:547-569`, built `P:385-389`). One MLP on the
concatenation [s, u, g]:

```
x    = tanh(LayerNorm(Dense(1024)([s, u, g])))
x    = relu(Dense(1024)(x))
head = RMSNorm_noscale(Dense(129)(x))        # joint normalisation of [phi, b]
phi  = head[:128],  b = head[128]
```

The joint RMSNorm bounds sum_k phi_k^2 + b^2 near 129. There is no sphere projection of phi
alone and no separate bound on b. M(s,u,g; w) = phi^T w + b (`P:474-478`). The mesh form
evaluates M for N transitions against S candidate states by running the network on N x S
triples (`P:480-504`). Every latent the network sees is clipped to [-3, 3]. Adam, lr 1e-4.
Single critic, no ensemble.

Goal head h(g) (`GoalCoefficient`, `NW:514-526`, built `P:408-413`):

```
x    = tanh(LayerNorm(Dense(256)(g))); x = relu(Dense(256)(x)); x = Dense(128)(x)
h(g) = sqrt(128) * x / max(||x||, 1e-12)
```

So ||h(g)|| = sqrt(128) = 11.31 always. Adam, lr 1e-4. A Polyak copy `target_w_star` (h_bar)
is kept at tau 0.01.

The flowbc actor delta(s, g, eps) (`LatentResidualActor`, `NW:678-702`, built `P:417-421`):

```
g_emb = relu(Dense(256)(tanh(LayerNorm(Dense(512)([s, g, eps])))))
s_emb = relu(Dense(256)(tanh(LayerNorm(Dense(512)([s, eps])))))
hid   = relu(Dense(512)(relu(Dense(512)([s_emb, g_emb]))))
delta = Dense(d_a, kernel_init=zeros)(hid)      # zero at init: the actor starts as BC
u     = clip(eps + delta, -3, 3)                 # flowbc_latent, P:1123-1127
```

eps is an unclipped N(0, I) draw in the actor loss, in the bootstrap and in acting. Only the sum
is clipped. Adam, lr 3e-4.

Frozen flow decode G(s,u) (`decode`, `P:1481-1491`). Default `gpi_decode=onestep`: a four-layer
512-wide MLP on [s, u] from the Stage-A checkpoint, output clipped to [-1, 1]. The ODE path
(`flow_decode_steps=10`, Euler from a_0 = u) exists and is not used. The flow is never updated.

A tanh-Gaussian actor is built so that checkpoints restore, and is neither trained nor used
under `actor_kind=flowbc`.

### 2.3 One sampled batch

`main.py:331` samples N = 256 transitions among dataset rows whose point preimage is valid
(`DS:244-301`). The preimage of row i is the latent u_i with G(s_i, u_i) = a_i, computed once
by Stage B and stored. Fields per transition i with dataset index idx_i (`DS:329-380`):

| field | value |
|---|---|
| `observations` | s_i |
| `next_observations` | s'_i |
| `noise_preimage` | u_i, the point preimage, clipped to [-3, 3] when used |
| `goals` | g_i, the measure goal, drawn as below |
| `fb_goals` | the actor goal, drawn as below |
| `future_states` | (N, K, ob) own later states, only when `norm_future_k` = K > 0 |
| `actions`, `rewards`, `terminals`, `masks`, `index` | carried, not read by this arm's training |

Trajectory end: end(i) = the first row with terminals > 0 at or after idx_i, else the last row.

Measure goal g_i, geometric (`DS:120-152`):

```
d_i ~ Geometric(1 - goal_discount),  P(d = k) = goal_discount^(k-1) (1 - goal_discount),  k >= 1
g_i = next_observations[min(idx_i + d_i - 1, end(i))]
```

With goal_discount 0.98 the mean offset is 50 steps, clipped at the trajectory end. d = 1 gives
the transition's own next state. `goal_random_frac` would replace a fraction of goals by uniform
random dataset states and `goal_cur_frac` by the own next state. Both are 0 in arm A, so every
measure goal is a later state of the transition's own trajectory.

Actor goal fb_goal_i (`DS:155-168`): a uniform draw among the next states from idx_i to end(i).

Future states for K > 0 (`DS:106-117`): K independent draws of the measure-goal rule per
transition. They are drawn last so that K = 0 leaves the random stream unchanged.

### 2.4 The measure loss, `measure_loss=softmax`

`measure_loss` `P:767-801`, `_softmax_measure_loss` `P:836-872`, `softmax_td_loss` `P:201-210`.

Candidate states S (`P:784-789`): the N next states of the batch, followed by the N x K future
states when K > 0. |S| = 256 for arm A and 768 for nf2. The same S serves every transition in
the batch. Candidate i is transition i's own next state.

Coefficients: online w_i = h(g_i), target w_bar_i = h_bar(g_i).

Online logits: M_ij = phi(s_i, u_i, S_j)^T w_i + b(s_i, u_i, S_j), with the online measure
network and u_i the clipped point preimage.

Target logits: M_bar_ij = phi_bar(s'_i, u'_i, S_j)^T w_bar_i + b_bar(s'_i, u'_i, S_j), with the
target measure network, stop-gradient, and u'_i the bootstrap latent of section 2.5.

Loss, with t = 1.0 and gamma = 0.98:

```
p_ij      = softmax_j( M_ij / t )
target_ij = (1 - gamma) * 1[j = i]  +  gamma * softmax_j( M_bar_ij / t )
L_measure = -(1/N) sum_i sum_j target_ij * log p_ij
```

In words: the measure at (s_i, u_i) is read as a distribution over the candidate states. The
target puts (1 - gamma) on the transition's own next state and gamma on wherever the target
measure at the next state, under the bootstrap latent, puts its mass. The loss is the
cross-entropy. The gradient reaches phi, b and h. Both targets are Polyak-updated at tau 0.01.
No other term is active: `ortho_coef` must be 0 and `train_goal_head` must be false for this
arm (`P:324-336`).

Why shares and not raw values are read at test time: adding any constant c(s,u) to every
candidate of transition i leaves this loss unchanged, so the raw M(s,u,g) carries an arbitrary
per-(s,u) offset. A share divides the offset out (`docs/results/2026-10-07-psmgoal-softmax.md:70-73`).

Logged every update (`P:848-871`): `measure_p_own_next` (equal to `m_softmax_diag`), the mean
over transitions of p_ii; `measure_p_future`, the mean total mass on the transition's own K
future candidates, 0 when K = 0; `phi_eff_rank` = (tr G)^2 / ||G||_F^2 with G the mean of
phi phi^T over the whole mesh, 128 if isotropic and 1 if rank one; `m_softmax_max`,
`m_logit_spread`, `td_target_absmean`, `w_norm` (constant 11.31).

### 2.5 The bootstrap latent u' (`_select_u_next`, `P:672-715`)

Computed before any parameter step, from the pre-step parameters. For arm A:

```
eps_i  ~ N(0, I_{d_a})                       key fold_in(rng, 103), fresh every update
delta_i = delta(s'_i, g_i, eps_i)              the actor at the NEXT state and the measure goal
u'_i   = clip(eps_i + m * delta_i, -3, 3)      m = bootstrap_delta_scale, 1.0 in A, -1.0 in the sign-flip arm
```

`bootstrap_switch` can replace u'_i by clip(eps_i) before or after `bootstrap_switch_step`
training steps while the actor keeps training (the p2a and a2p arms). Other sources, used by
other arms: `data` takes the next transition's stored preimage u_{i+1}; `argmax` takes the best
of `bootstrap_n` clipped prior draws scored by the target measure's share of g_i among
[g_i + the N next states]; `bc_mean` makes the target softmax the mean of the softmax rows over
`bootstrap_n` prior draws.

### 2.6 The actor loss (`goal_actor_step` `P:1177-1193`, `actor_loss_flowbc` `P:1129-1146`, `actor_value` `P:1090-1121`)

Inputs: s_i, the pool of N next states (never the future states, even in nf2), the actor goal
g_hat_i = fb_goals_i, eps_i ~ N(0, I) from key fold_in(rng, 102). The measure parameters and
h are stop-gradded. Only delta's parameters move.

```
u_i  = clip(eps_i + delta(s_i, g_hat_i, eps_i))
M_ij = phi(s_i, u_i, s'_j)^T h(g_hat_i) + b(s_i, u_i, s'_j),   j over the N next states
m_i  = phi(s_i, u_i, g_hat_i)^T h(g_hat_i) + b(s_i, u_i, g_hat_i)
q_i  = exp(m_i/t) / ( exp(m_i/t) + sum_j exp(M_ij/t) )          the point value: the share of the goal
L_actor = - q_coeff * mean_i q_i / sg(max(mean_i |q_i|, 1e-6))  +  fb_bc_coeff * mean_{i,k} delta_ik^2
```

The anchor is on delta, that is toward u = eps, which decodes to a behaviour-cloning sample. It
is not an anchor toward the data latent u_i. There is no entropy term. `actor_value_kind=shaped`
would replace q_i by sum_j softmax_j(M_ij/t) k(g_hat_i, s'_j) with a kernel k on cube position;
arm A uses point.

Order within one update (`apply_update` `P:1206-1265`): u' from pre-step parameters; measure
gradient and step; target update; actor step. The actor step reads the pre-step online measure
and h, so measure and actor are updated from the same parameters, and the measure sees the
actor's new step only at the next update.

### 2.7 The goal head

h(g) is trained only through the measure loss, as the second gradient argument (`P:1215-1221`).
The separate amortised objective `goal_head_loss` (`P:875-896`) runs only under
`train_goal_head=true`, which arm A forbids. h is stop-gradded in every actor loss.

### 2.8 Test-time inference and acting, `coef_source=hgoal_each`, `acting=gpi`

Once per evaluation (`infer_eval_goals` `P:1637-1779`, called from `tools/eval_checkpoint.py`):

1. A relabel batch of 10,000 dataset rows is drawn. Rewards are shifted by 1.0 so cube's {-1, 0}
   becomes {0, 1}. Rows with reward above 0.5 are rewarding.
2. The goal set: 32 rewarding rows' next states, `eval_goals` (32, ob), full observation
   vectors.
3. The reference set: 224 non-rewarding rows' next states, `eval_ref` (224, ob). 224 = batch
   size minus k_goals.
4. For `hgoal_each`: `eval_goal_w = h(eval_goals)`, one coefficient per goal.

Every environment step (`select_latent` `P:1493-1512`, `_hgoal_each_score` `P:578-609`):

```
u_k ~ N(0, I) clipped to [-3, 3],  k = 1..64
for each goal c with w_c = h(g_c):
    m_g[k,c]   = phi(s, u_k, g_c)^T w_c + b(s, u_k, g_c)
    m_r[k,c,r] = phi(s, u_k, ref_r)^T w_c + b(s, u_k, ref_r),  r = 1..224
    share[k,c] = exp(m_g[k,c]/t) / ( exp(m_g[k,c]/t) + sum_r exp(m_r[k,c,r]/t) )
q_k = mean_c share[k,c]
u*  = u_{argmax_k q_k};   a = G(s, u*)
```

Goals are combined by the mean of per-goal shares. A new argmax is taken every step
(`gpi_hold=1`). The other readouts are defined in section 9.1.

### 2.9 One training step, pseudocode

```
update(batch):
  rng, new_rng = split(rng)
  # bootstrap from pre-step parameters
  eps_b  = N(0, I)                                     key fold_in(rng, 103)
  u'_i   = clip(eps_b_i + m * delta(s'_i, g_i, eps_b_i))
  # measure loss
  u_i    = clip(noise_preimage_i)
  S      = [s'_1 .. s'_N]  (+ future_states flattened if K > 0)
  M_ij   = phi(s_i,u_i,S_j)^T h(g_i) + b(s_i,u_i,S_j)              online, with gradient
  Mb_ij  = sg( phi_bar(s'_i,u'_i,S_j)^T h_bar(g_i) + b_bar(s'_i,u'_i,S_j) )
  tgt_ij = (1-gamma) 1[j=i] + gamma softmax_j(Mb_ij/t)
  L      = -(1/N) sum_ij tgt_ij log softmax_j(M_ij/t)
  (measure, h) <- Adam steps on grad L
  target_measure <- 0.01 measure + 0.99 target_measure;  h_bar likewise
  # actor step, reading the PRE-step measure and h with stop-gradient
  eps_a  = N(0, I)                                     key fold_in(rng, 102)
  u_i    = clip(eps_a_i + delta(s_i, fb_goal_i, eps_a_i))
  q_i    = share of fb_goal_i among [fb_goal_i + the N next states] at w = h(fb_goal_i)
  L_a    = -mean q / sg(mean |q|) + 3.0 * mean delta^2
  delta  <- Adam step on grad L_a
  log measure_p_own_next, measure_p_future, phi_eff_rank, actor_delta_norm, actor_q, ...
```

Every 50k steps an in-loop 50-episode evaluation runs with the actor acting alone
(`acting=distill`, `eval_redistill=false`). That number swings by 0.15 between consecutive
points and is not used for reporting. Checkpoints at 250k and 500k. The 500-episode five-task
evaluation of a checkpoint is `scripts/eval500.sh psmgoal cube <run_dir> <out>` with the
readout override.

---

## 3. The path to the winning configuration, 2026-09-16 to 2026-10-07

Each step gives the decision, the reason written at the time, the arms, their overrides, the
numbers, the expectation stated before the result, and whether it held.

### 3.1 2026-09-16: the collation (`docs/design/2026-09-16-goal-indexed-measure.md`)

Decisions written:
- Index the policy by a goal state g through h(g), with r_g(s+) = 1[s+ = g].
- Replace the per-transition multiplier of Eq. 10 by a per-triple multiplier l(s,u,s+), because
  the per-transition form "cannot weight the particular s+ that violates" and "the penalty never
  bound" in the previous run.
- Keep from the previous agent: the split phi = A(s,u)^T f(s+) as a first version, sphere
  normalisation, ensemble-min pessimism, the EMaQ argmax over 8 draws as the bootstrap,
  500-episode five-task evaluation.
- Noted as optional: a general network phi(s,u,s+) in place of A^T f.

Reason recorded for the whole move: on a frozen head the coefficient ascent is closed-form and
measured near zero; the proposal is new only because the head is fit on the coefficients it is
queried at, and feasibility becomes a training-time property.

### 3.2 2026-09-17: the spec (`docs/design/2026-09-17-psmgoal.md`)

Objects: general networks phi(s,u,s+) and b(s,u,s+) on the triple, no A/beta split, no state
basis; h(g) with w*(g) = h(g)/||h(g)||; an optional per-triple multiplier; pessimism as a min
over 2 heads; bootstrap by the EMaQ argmax over 8 prior draws under the target networks;
hindsight goals at a geometric horizon plus 30 percent random-state goals; a contrastive TD
loss on an N x (C+1) matrix with C = 32 negative next states; test time w = the normalised sum
of w*(g) over 32 rewarding goals and an argmax over 64 prior draws.

Pre-registered readings: five-task above 0.328 at any step means goal indexing was the missing
piece; nocon equal to con means the constraint is unnecessary; con above nocon means negative
mass was hurting acting; both at or below 0.284 with stable TD means the affine measure over
flow latents does not carry a goal-reaching value on this substrate.

Arms: `psmgoal_cube_nocon` (constraint_coef 0) and `psmgoal_cube_con` (constraint_coef 1),
3 seeds, 1M steps, batch 1024, trunk 512 x 2, lr 1e-4 / 1e-4 / 3e-5.

First launch diverged at about 15k steps: the measure at the positive state reached 2.4e5 to
4.4e5 and the violation fraction 0.998 to 1.000. Cause written: the spec removed the state
basis and its orthonormality term and put no scale control in their place, and the contrastive
term is unbounded below. Fix (commit cfbb285): phi on the sphere of radius sqrt(D), w*(g) on
the same sphere, b bounded by D tanh(b/D), so |M| <= 256.

Result, 500 episodes, 3 seeds, mean and 95 percent interval:

| arm | 250k | 500k | 750k | 1M |
|---|---|---|---|---|
| nocon | 0.075 ± 0.078 | 0.057 ± 0.105 | 0.002 ± 0.003 | 0.002 ± 0.003 |
| con | 0.118 ± 0.131 | 0.007 ± 0.010 | 0.004 ± 0.004 | 0.029 ± 0.031 |

Per seed, con at 250k: 0.000 / 0.231 / 0.123. The "above 0.328" reading was not met. The
"both at or below 0.284 with stable TD" reading was met. More TD made the capped measure worse
for ranking latents. The constraint did not change the verdict. The eval watcher had submitted
nothing because `MODE=psmgoal` and `PSM_REPO` were missing; both are now required.

The "lp best 0.118" quoted in `docs/COMPENDIUM.md` and CLAUDE.md is this arm's number. The
promoted agent of 3.3 reads lp 0.001.

### 3.3 2026-09-17/18: the rewrite to a faithful RLU port (the code every later arm runs on)

The spec agent was replaced the same day by a one-to-one port of the Proto Successor Measure
of RLU (arXiv 2411.19418), commit b1108be: general phi(s,u,g) + b with no scale anchor, a
single critic, w(z) = sqrt(D) enc(z)/||enc(z)|| on the sphere indexed by a binary code z, the
basis fit by off-diagonal squared TD on a state x next-state mesh plus a -(1 - gamma) diagonal
occupancy pull, the bootstrap from a fixed z-indexed proto policy, Stage-2 coefficient inference
by the Eq. 10 Lagrangian with a per-triple multiplier, gpi acting or a minimal DSRL distill.

Fixes in the following two days, each with its reason:
- 0828e4b (09-18): the diagonal pull ran the measure to about 1e4 by 15k steps on continuous
  cube and antmaze, because every (s,u,g) triple is distinct and the pull is uncoupled from the
  off-diagonal regressions that bound it in RLU's gridworld. A no-affine LayerNorm on the joint
  [phi, b] head hard-bounds |M| near z_dim. Verified: the diagonal flattens at about 128 by 12k
  steps and the target converges. This refuted the audit's claim that the no-anchor loss was
  safe.
- 64f707a (09-18): `train_goal_head` adds an amortised h(g) trained by J(theta | g) with a
  non-negativity hinge, kept "so it does not collapse to FB"; `coef_source=lp|amortized`
  selects the coefficient at eval; goal mixture 70/30 hindsight/random.
- 7703c51 (09-18): inference uses the dataset preimages, not prior draws, so the expectation is
  over (s,u) from the data; RMSNorm replaces the centering LayerNorm on the head, because
  centering made E[phi] degenerate and the linear LP objective pointed w at random; capacity
  raised, max_log_seed 8 to 16 and measure hidden 512 to 1024.
- b302f92 (09-18): rename to the paper notation, coef to w, goal_coef to w_star, mult to l,
  z_bin to z.
- aca38cf (09-19): eval_checkpoint splices the preimages into the eval dataset.
- bca4b8e (09-22): psmgoal promoted to the main agent as-is; the comparator stays.

The three documented deviations of this code from the written spec: phi and b are one MLP on
[s, u, s+] and not A(s,u)^T f(s+) and beta(s,u)^T f(s+); the policy index is a binary code z
with its own w(z) network and not u' through f; inference uses 32 sampled rewarding goals and
not all s+.

Resulting squared loss in code (`P:767-834`), for the record of what the softmax loss later
replaced: candidates = the batch's next states; M = phi^T w + b on the N x N mesh;
off_diag = 0.5 mean over j != i of (M_ij - gamma M_bar_ij)^2; diag = -(1 - gamma) mean_i M_ii;
loss = off_diag + diag + ortho_coef * ortho.

Default configuration after promotion: batch 256 (the loss is a batch x batch mesh), z_dim 128,
discount 0.98, tau 0.01, ortho_coef 0, measure_form joint, lr_measure 1e-4, lr_w 1e-4,
lr_l 3e-4, max_log_seed 16, lr_infer 3e-4, num_inference_steps 20000, use_dgd true,
inf_coeff 5.0, infer_batch 1024, k_goals 32, acting gpi, gpi_num_u 64, train_actor false,
fb_bc_coeff 3.0, train_goal_head false, coef_source lp, goal_discount 0.98,
goal_random_frac 0.3, policy_index code, measure_loss squared, measure_temp 1.0,
actor_input w, goal_cur_frac 0.0, eval_goal_source relabel, bootstrap_source actor,
bootstrap_n 8, goal_sampling geometric, gpi_hold 1, actor_kind tanh, actor_value_kind shaped,
measure hidden 1024 x 2, u_clip 3.0.

### 3.4 2026-09-18: `psmgoal_lift_cube_gc`, the RLU port with the amortised head

Overrides: `train_goal_head=true coef_source=amortized`, 1M steps, 3 seeds. No launch entry
or expectation was written; the run directory is dated 2026-09-18 16:15. Readouts measured
later:

| readout | 250k | 500k | 750k |
|---|---|---|---|
| lp (Lagrangian) | 0.001 | | 0.000 with 10,000 goals |
| regression (least-squares w on goal-averaged phi) | 0.079 | 0.176 | 0.301 ± 0.210 |
| trained (mean of w(z) over 256 codes) | 0.327 | 0.240 | |

Per seed, regression: 0.095/0.066/0.076, 0.103/0.160/0.264, 0.231/0.395/0.276. The `trained`
readout was expected at BC level, 0.05 to 0.2, because the measure was at its bound at the
transition's own next state and near zero elsewhere. It scored 0.327. Conclusion written on
10-05: the readout was the fault of the old psmgoal, not the measure.

### 3.5 2026-09-21 to 09-23: diagnostics on the lift run and the first fixes

Rank-one collapse (`tools/diag_psmgoal_gram.py`; `docs/design/2026-09-22-psmgoal-fixes.md`
section 1): phi's effective rank 1.01 to 1.02 of 128 at every checkpoint and seed, top
eigenvalue share 0.990 to 0.993; cosine between the Lagrangian direction and the least-squares
coefficient 0.0000 to 0.0001; whitening the LP gradient by the feature Gram recovers least
squares, cosine 0.997 to 0.999. Reading: the Lagrangian objective is linear in w and returns
the direction of its gradient, which is the collapsed direction; regression works because it
reads the small residual directions.

Reward relabel (`docs/design/2026-09-21-psmgoal-reward-relabel.md`): the regression readout
correlates with reward at 0.374; a scalar critic trained on the induced reward phi^T w reaches
0.876 ± 0.005 against 0.877 ± 0.047 with the real reward. Random weights give correlation
0.357, so training adds little. Conclusion at the time: the features carry reward, the fault is
the coefficient. Correction on 10-05: the reward override replaced rewards only and the critic
kept the task's true episode-end mask; in four rooms a critic with reward -1 everywhere plus
the true episode end scores 1.000; cube without the mask is untested.

F1 orthogonality arms, 2026-09-23, `psmgoal_f1_ortho{1,10,100}_cube`, 2 seeds, 500k, every
other key as the lift run. Expected: effective rank well above 1, cosine between lp and least
squares above 0.9, lp close to regression. Expected failure: the TD fit and the ortho term
fight.

| arm | 250k lp / regression | 500k lp / regression |
|---|---|---|
| ortho 1 | 0.011 / 0.033 | 0.036 / 0.047 |
| ortho 10 | 0.071 / 0.058 | 0.061 / 0.058 |
| ortho 100 | 0.055 / 0.062 | 0.061 / 0.070 |
| control, no ortho, 3 seeds | — / 0.079 | — / 0.176 |

Effective rank 127.7 to 127.9 (control 1.01 to 1.02); cosine between E[phi r] and the
least-squares w 0.75 to 0.98 (control 0.0003); cosine between the lp direction and least
squares 0.29 to 0.50 (control 0.0000). The mechanism targets were met and acting fell below BC
and below the control. Default stays ortho_coef 0. F2 (two-head pessimism), F3 (u_ball 4.1)
and F4 (greedy bootstrap) were specified in the same document and never run.

### 3.6 2026-09-23 to 09-30: policy count, factorisation, the Factored-FB trick (`docs/design/2026-09-30-psmgoal-tests.md`)

| arm | change | hypothesis | result |
|---|---|---|---|
| `psmgoal_fewpol` | max_log_seed 2 (4 policies) | fewer policies lets w(z) separate | partial: regression 0.194 at 250k on 2 of 3 seeds; at 500k lp 0.001, regression 0.212 |
| `psmgoal_factor` | measure_form factorized, phi = A(s,u)^T f(g), b = beta(s,u)^T f(g), f on the sphere, f_ortho 1.0 | factorising s+ out removes the spike | no: lp 0.012, regression 0.026 at 500k; M at own next state about 38, elsewhere 0.15 |
| `psmgoal_fewpol_factor` | both | | no: lp 0.012, regression 0.032 |
| Factored-FB trick, raw and softmax | actor(s, w) trained on Q = sum_j M(s,u,s+_j) r_w(s+_j), tasks 50 percent hindsight / 50 percent random w, w by ridge on reward labels, act with the actor | | no: 0.073 ± 0.036 and 0.066 ± 0.049 at 500k, below BC |
| actor following M with no pull to the data | | | 0.001 |
| BC actor proposes 64 latents, pick the highest M (`acting=sfbc`) | | | 0.206 at 250k |

The failing-point audit in the same document, 24 rows, is the prior to everything after:
M at the transition's own next state = 129, the head bound; elsewhere about 0.005; all w(z)
the same vector, cosine 0.99999999; z explains 1e-13 to 3e-8 of M's variance; u explains
0.0001 to 0.009; the diagonal gradient is aligned with the mean phi at the own next state
(cosine 0.9999) and the TD gradient has cosine -0.005 with it; a 7 x 7 gridworld with 4
learnable policies recovers the true measure (correlation 0.98), with 2^16 codes w(z) cannot
learn the hash; the rank correlation of M's ordering of 64 latents with a scalar critic is
about 0 and the top pick agrees 4 to 7 percent of the time (chance 1.6).

First-latent simulator test (`tools/diag_first_u_effect_bc.py`): the share of success
variance explained by the first latent, bias-corrected, is 0.0009 on cube and 0.0002 on
antmaze. Written conclusion: the measure of BC has nothing to rank u by; index the policy by
the goal and bootstrap from a goal policy.

Squared-loss saturation (HANDOFF 10-05; `docs/figures/gridworld/gridworld_7x7/EXPECTED.txt`):
under the squared loss the fixed point for the own next state is (1 - gamma) x N divided by
the number of indistinguishable transitions, 20,020 for the 28-dim cube state, so M sits at
its bound 128.9 to 129.0 at the own next state and 0.005 to 0.03 elsewhere. The 7 x 7 grid
reproduces this on unique states and recovers M on one-hot states. The softmax loss removes
the saturation: own-state weight 0.45 to 0.49 on cube, 0.08 to 0.09 on antmaze. Decision:
`measure_loss=softmax`.

Action-from-future (`tools/diag_action_from_future.py`): top-1 of 64 identification of the
first action from the future at k = 1 / 5 / 50 / 100 steps: 0.277 / 0.123 / 0.102 / 0.084 on
cube, no-future 0.083, chance 0.016; antmaze 0.11 at every k. The first action is identifiable
from the future only within a few steps on cube and not at all on antmaze.

Four-rooms gridworld (`tools/diag_fourrooms_induced_reward.py`): exact M greedy 1.000; learned
M with its training w 0.731 (one-hot) / 0.565 (xy-phase); regression w 0.084 / 0.239;
Lagrangian w 0.044 / 0.134; random 0.251. Decision: the `trained` readout, and the mask
correction above.

### 3.7 2026-10-01: the goal-conditioned design (`docs/design/2026-10-01-psmgoal-goal-conditioned.md`)

Why, as written: every policy index denotes the behaviour-cloning policy; M changes with u by
under 1 percent of its variance within a state; M is at its head bound at the own next state
and about 0.005 elsewhere. Density FB (Factored-FB) scores 90.8 on cube-single where plain FB
scores 17.7, with an actor that takes the goal, a bootstrap from that actor at the same goal,
and a softmax loss over the batch's next states.

What changes: a goal per transition, 20 percent own next state / 50 percent geometric future /
30 percent random; w_i = h(g_i) with a Polyak copy of h; a tanh-Gaussian actor(s, g) trained on
-q_coeff Q / sg|Q| + actor_temp log p + fb_bc_coeff mean (u - u_data)^2 with fb_bc_coeff 3.0
and Q the kernel-weighted measure on cube position (width 0.4); the bootstrap is the actor's
mode at (s'_i, g_i); the softmax loss of section 2.4.

Arms `psmgoal_gc_sq_cube` and `psmgoal_gc_sm_cube` with
`policy_index=goal train_actor=true actor_input=goal goal_cur_frac=0.2 goal_random_frac=0.3
measure_loss={squared|softmax}`, plus the antmaze pair with discount 0.99, goal_discount 0.99,
fb_hit_spec 0:2:0.5. Expectations: squared above the default lp 0.001 and well below Density
FB; softmax the larger gain; lp at or below hgoal under softmax; refutation = both at or below
BC on all four readouts.

| group | hgoal 250k / 500k | lp | actor_rel | actor_env |
|---|---|---|---|---|
| gc_sq_cube | 0.405 / 0.277 | 0.000 / 0.000 | — | 0.110 / 0.098 |
| gc_sm_cube | 0.665 ± 0.055 / 0.594 ± 0.181 | 0.367 / 0.308 | 0.137 / 0.144 | 0.131 / 0.116 |
| gc_sq_antmaze | 0.032 / 0.061 | 0.058 / 0.040 | 0.090 / 0.088 | 0.096 / 0.098 |
| gc_sm_antmaze | 0.044 / 0.121 | 0.018 / 0.002 | 0.089 / 0.095 | 0.090 / 0.091 |

All three cube expectations held. Antmaze stayed at BC. The actor sat near u = 0 (|u| 0.15 to
0.31 against 0.8 for a prior draw) because the BC term pulled toward the dataset latent. The
new readout `hgoal` (section 9.1) replaced lp as the scoring readout.

### 3.8 2026-10-03: the data bootstrap (`docs/design/2026-10-03-psmgoal-data-bootstrap.md`)

Why, as written: in every psmgoal version so far the policy inside M was defined by us and none
reaches goals; the dataset itself is a mixture of goal-reaching policies; for goal g define the
policy as "the move the data makes at s on trajectories that reach g".

Changes: bootstrap u'_i = u_{i+1}, the next transition's stored preimage; hindsight goals only,
geometric at goal_discount, goal_random_frac 0 and goal_cur_frac 0; transitions at a trajectory
end or with an invalid next preimage not sampled; new readout `hgoal_each`, each goal scored at
its own h(g) (the exact per-goal reading, where `hgoal` averages h over goals).

Overrides: `policy_index=goal bootstrap_source=data train_actor=false goal_random_frac=0.0
goal_cur_frac=0.0 measure_loss={squared|softmax}`. Expectations: cube softmax hgoal above
0.665, squared above 0.405, hgoal_each at or above hgoal; refutation = cube softmax hgoal at or
below 0.665 with three seeds agreeing.

| group | hgoal 250k / 500k | hgoal_each 250k / 500k | lp 250k |
|---|---|---|---|
| db_sq_cube | 0.597 / 0.516 | 0.602 ± 0.100 / 0.520 ± 0.420 | 0.000 |
| db_sm_cube | 0.412 / 0.446 | 0.619 ± 0.146 / 0.646 ± 0.286 | 0.26 (1 seed) |
| db_sq_antmaze | 0.118 / 0.096 | 0.115 / 0.101 | |
| db_sm_antmaze | 0.095 / 0.051 | 0.086 / 0.088 | |

Squared above 0.405 held. Softmax hgoal above 0.665 did not hold (0.412; hgoal_each 0.619).
hgoal_each at or above hgoal held on cube. The softmax-versus-squared ordering reversed under
hgoal and held under hgoal_each, so hgoal_each became the scoring readout.

Hold ablation on db_sq 250k: holding the chosen latent for 1 / 4 / 8 steps gives 0.602 / 0.440
/ 0.135. Expected equal or better at 4; it was worse.

### 3.9 2026-10-04/05: uniform goals and orthonormality under the data bootstrap

`psmgoal_dbu_{sq,sm}_cube`, `goal_sampling=uniform`. Expected above the geometric arms because
test-time goals are far from most rollout states and uniform draws train h and the measure on
far goals about 10 times more often. Result: dbu_sq hgoal_each 0.479 / 0.448, dbu_sm 0.506 /
0.501, both below geometric. Did not hold.

`psmgoal_dbo_sq_cube`, `ortho_coef=1.0` on the data bootstrap, to re-test the 09-23 penalty
under the trained-coefficient readout. Result: hgoal_each 0.068 / 0.039, phi rank 128, M at the
head bound at the own next state. Orthonormality lowers acting under every readout.

### 3.10 2026-10-05: the joint-actor arms (`scripts/campaign_2026-10/ja_arms.sh`)

Base: the data-bootstrap softmax arm with the actor back in the bootstrap,

```
JA_BASE = policy_index=goal bootstrap_source=actor train_actor=true actor_input=goal
          goal_random_frac=0.0 goal_cur_frac=0.0 goal_sampling=geometric measure_loss=softmax
          acting=distill eval_redistill=false eval_goal_source=relabel
```

Two new actor kinds (commit of 10-05): `flowbc`, u = clip(eps + delta(s,g,eps)) with the BC
anchor on delta and a zero-initialised output so the actor starts as BC; `dsrl`, the
tanh-Gaussian actor with no BC term and a learned entropy weight. Two actor values: `shaped`,
the kernel-weighted measure over the batch's next states; `point`, the share of the actor goal
among [goal + the batch's next states]. The actor goal is a uniform later state (`fb_goals`).

| group | extra overrides |
|---|---|
| `psmgoal_ja_fbc3_sh_cube` | actor_kind=flowbc fb_bc_coeff=3.0 actor_value_kind=shaped |
| `psmgoal_ja_fbc0_sh_cube` | actor_kind=flowbc fb_bc_coeff=0.0 actor_value_kind=shaped |
| `psmgoal_ja_dsrl_sh_cube` | actor_kind=dsrl fb_bc_coeff=0.0 actor_value_kind=shaped |
| `psmgoal_ja_fbc3_pt_cube` | actor_kind=flowbc fb_bc_coeff=3.0 actor_value_kind=point |
| `psmgoal_ja_fbc03_sh_cube` | actor_kind=flowbc fb_bc_coeff=0.3 actor_value_kind=shaped |

Launch: 500k steps, eval every 50k with 50 episodes, save at 250k and 500k, 3 seeds packed on
one GPU, 200-step smokes with a flags.json diff against the data-bootstrap run.

Pre-launch expectations (`docs/campaign_2026-10/ja_runs_EXPECTED.txt`): fbc3_sh hgoal_each at
or above 0.68 with actor_rel within 0.05 of it and |u| about 0.8 per dimension; fbc0_sh and
dsrl_sh below fbc3_sh with a risk of collapse; fbc3_pt below fbc3_sh because the point value at
a far goal is flat in u; if every arm sits at or below 0.68 the actor in the bootstrap does not
improve on the data's own next move.

| group | hgoal_each 250k / 500k | 95 percent interval | actor alone 250k / 500k |
|---|---|---|---|
| `psmgoal_ja_fbc3_pt_cube` | 0.794 / 0.804 | ± 0.134 / ± 0.104 | 0.108 / 0.108 |
| `psmgoal_ja_fbc03_sh_cube` | 0.758 / 0.706 | ± 0.062 / ± 0.127 | 0.132 / 0.117 |
| `psmgoal_ja_fbc3_sh_cube` | 0.669 / 0.655 | ± 0.314 / ± 0.267 | 0.106 / 0.103 |
| `psmgoal_ja_fbc0_sh_cube` (2 seeds) | 0.021 / 0.033 | | 0.001 / 0.003 |
| `psmgoal_ja_dsrl_sh_cube` | 0.014 / 0.062 | | 0.041 / 0.088 |

Per seed for the winner: 250k 0.818 / 0.831 / 0.732; 500k 0.756 / 0.825 / 0.832. With seeds 3
and 4 added on 10-08: 250k 0.789, 500k 0.805 ± 0.014 (0.756 / 0.825 / 0.832 / 0.815 / 0.798).

Training statistics at 500k, seeds 0/1/2: mean |u| 0.80 / 0.80 / 0.79; mean step norm 0.026 /
0.040 / 0.048; actor value 0.037 / 0.047 / 0.048; measure share on the own next state 0.285 /
0.277 / 0.274; phi effective rank 1.01 / 1.03 / 1.03. Without the BC term the actor's latent
leaves the prior scale (mean |u| 1.9 to 2.9, step norm 1730 to 3520) and the measure collapses.

Expectations resolved: fbc3_sh at the 0.68 reference, actor_rel not within 0.05; collapse of
the no-BC arms, yes; fbc3_pt below fbc3_sh, no, it scored above, and this reversal is recorded
as unexplained. The winning configuration is JA_BASE plus `actor_kind=flowbc fb_bc_coeff=3.0
actor_value_kind=point`, scored with `coef_source=hgoal_each acting=gpi`.

### 3.11 The state of knowledge on 2026-10-07

The summary table of `docs/results/2026-10-07-psmgoal-softmax.md`:

| arm | group | readout | 250k | 500k |
|---|---|---|---|---|
| code label, squared | lift_cube_gc | trained | 0.327 | 0.240 |
| code label, softmax | sm_code_cube | trained | 0.524 | 0.383 |
| goal label + tanh actor, squared | gc_sq_cube | hgoal | 0.405 | 0.277 |
| goal label + tanh actor, softmax | gc_sm_cube | hgoal | 0.665 | 0.594 |
| data bootstrap, squared | db_sq_cube | hgoal_each | 0.602 | 0.520 |
| data bootstrap, softmax | db_sm_cube | hgoal_each | 0.619 | 0.646 |
| uniform goals, squared / softmax | dbu_sq / dbu_sm | hgoal_each | 0.479 / 0.506 | 0.448 / 0.501 |
| orthonormal phi, data bootstrap | dbo_sq_cube | hgoal_each | 0.068 | 0.039 |
| flowbc actor, bc 3, shaped | ja_fbc3_sh | hgoal_each | 0.669 | 0.655 |
| flowbc actor, bc 0.3, shaped | ja_fbc03_sh | hgoal_each | 0.758 | 0.706 |
| flowbc actor, bc 3, point | ja_fbc3_pt | hgoal_each | 0.794 | 0.804 |
| flowbc actor, bc 0, shaped | ja_fbc0_sh | hgoal_each | 0.021 | 0.033 |
| DSRL actor, shaped | ja_dsrl_sh | hgoal_each | 0.014 | 0.062 |
| behaviour cloning | | | 0.111 | 0.111 |

What was not working, as recorded: the actor acting alone at BC level in every arm; actors
without a BC term collapse; lp below the h(goal) readouts everywhere, best 0.367; orthonormality
raises rank to 128 and lowers acting; antmaze at BC on every arm and readout, with the measure's
own-next-state share 0.08 to 0.09 there against 0.45 to 0.49 on cube; uniform goals below
geometric; holding the latent lowers success; phi rank 1.0 to 1.14 everywhere without ortho;
one dataset carries every positive result; the point-beats-shaped reversal unexplained.

The decisions carried forward: the softmax loss, the goal index with w = h(g), geometric
hindsight goals only, the flowbc actor with BC weight 3 in the bootstrap, the point value, the
hgoal_each readout, and the 500-episode five-task protocol with the BC control beside every
number.

---

## 4. Taking the actor apart, 2026-10-07 to 2026-10-17

The question from 10-07: arm A scores 0.805 and its actor acting alone scores 0.108. The
actor's whole effect is through the latent it supplies to the TD target. What does that latent
do? Three machines worked on it: kisski (this checkout, H100 under SLURM), luis (H200 for
training, 16-core CPU nodes for simulator diagnostics) and, from 10-15, aws (8 L4 GPUs, no
scheduler). Coordination went through the mailbox folders in the HF dataset; the task split of
10-09 is in `mailbox/tasks/2026-10-09-design-decisions.md`. Parity between machines was
checked before numbers were pooled: training within 1 percent on psm_loss at 5k steps and
1.8e-5 on the diagonal mean at 10k; the same 250k checkpoint of A evaluated at 0.818 (kisski),
0.826 (luis), 0.820 (aws); mujoco 3.15 against 3.12 within 0.012 on 15 groups.

### 4.1 The run list of 10-07 and what each run was for

1. Best-of-N bootstrap without an actor: the target uses the best of N prior draws scored by
   the target measure. If within A's seed interval, the actor is not needed; if lower, the plain
   argmax picks latents the target overestimates.
2. BC weight 0.3 with the point value: combine the two changes that each raised the score;
   expected at or above 0.80.
3. The actor as the proposal at test time: draw the 64 candidates from the actor instead of
   the prior; expected within 0.05 of 0.80.
4. BC weights 1 and 10: if the gain comes from the step size, bc 10 falls toward 0.62 to 0.65
   and bc 1 sits at or above 0.80; if all alike, step size does not explain it.
5. Antmaze with the best configuration: expected at BC level.

### 4.2 Chronological log

2026-10-05 to 10-09, luis, best-of-N (`bootstrap_source=argmax`, `bootstrap_n=N`, actor
removed, everything else A's):

| group | N | cube 250k / 500k | antmaze 250k / 500k |
|---|---|---|---|
| bon1 | 1, one BC draw | 0.713 / 0.708 (seeds 0.748/0.740, 0.749/0.755, 0.643/0.630) | 0.020 / 0.046 |
| bon8 | 8 | 0.020 / 0.096 | 0.092 / 0.046 |
| bon64 | 64 | 0.006 / cancelled at 420k | 0.061 / cancelled |

Reading: N at or above 8 collapses on cube in every seed; the max over noisy target estimates
feeds back through TD. Also from luis, squared-loss arms: dbo_sq 0.055 / 0.049; ja_dsrl_sq
0.004 / 0.000; jao_dsrl_sq (plus ortho) 0.142 / 0.080; jd_dsrl_sq (DSRL actor trained, data
bootstrap) 0.419 / 0.399; jdo_dsrl_sq 0.061 / 0.059. On antmaze every arm's in-loop task-1
success rose to 0.26 to 0.60 at 50k to 150k and fell to about 0 by 200k to 250k.

2026-10-08, kisski:

| group | change from A | seeds | 250k / 500k |
|---|---|---|---|
| ja_fbc3_pt | seeds 3, 4 added | 5 | 0.789 / 0.805 |
| ja_fbc3_sh | seeds 3, 4 added | 5 | 0.700 / 0.710 |
| ja_fbcBC_pt (critic B) | fb_bc_coeff 1000 | 3 | 0.626 / 0.657 (0.716 / 0.569 / 0.687) |
| ja_nostitch_pt | every direct start-to-goal stretch dropped from the data (278k rows) | 3 | 0.730 / 0.786 |
| ja_randdrop_pt | a random 258k rows dropped | 3 | 0.786 / 0.787 |

Conclusion: cube success does not need direct start-to-goal trajectories in the data. Critic B
and bon1 both sit below A.

Actor-ranking diagnostic (`tools/diag_actor_ranking.py`, 2000 dataset transitions, 63 fixed
prior draws plus the data latent, hindsight goal k steps ahead, score = per-goal share among
255 reference states): under A at 500k with k = 5 the data latent ranks 27.1 of 64 (chance
32.5), the actor's latent 30.3, its own base draw 32.2; the actor beats its draw 84 percent of
the time; the 64 scores differ by 4 to 5 percent of the across-state spread. The shaped-value
actor ranks 32.5, at chance.

2026-10-09, the task split. New keys on kisski (commit 9eacf71 on 10-13): `actor_goal_source`
(future | measure), `bootstrap_switch` (none | prior_then_actor | actor_then_prior) with
`bootstrap_switch_step`, `acting=flowbc_gpi`.

2026-10-10, kisski, pair-ranking diagnostic (`tools/diag_measure_pair_ranking.py`, A against B,
seeds 0 to 2, 2000 states, the same 63 draws): the two critics order the same draws with rank
correlation 0.38 to 0.68 depending on step and horizon, agree on the top pick 20 to 32 percent
of the time (chance 1.6), and rank the data latent equally (26.9 to 32.0). No written
conclusion exists; the numbers say the two critics are correlated but far from identical in
their rankings.

2026-10-12, kisski, the switch and goal arms, and readout 3:

| run | 250k | 500k mean ± SE | per seed 500k |
|---|---|---|---|
| ja_p2a_pt, BC draw for 100k then the actor | 0.763 | 0.806 ± 0.031 | 0.862 / 0.755 / 0.800 |
| ja_a2p_pt, the actor for 100k then BC draw | 0.690 | 0.727 ± 0.018 | 0.756 / 0.695 / 0.730 |
| ja_actmg_pt, actor goals = the measure's goals | 0.727 | 0.751 ± 0.038 | 0.825 / 0.699 / 0.730 |
| ja_ogmix_pt, goal mix 0.2 own / 0.5 trajectory / 0.3 random | 0.784 | 0.755 ± 0.021 | 0.764 / 0.714 / 0.786 |
| ja_rand30_pt, 30 percent random goals | 0.753 | 0.704 ± 0.047 | 0.699 / 0.625 / 0.788 |
| A, 64 candidates from the actor (`flowbc_gpi`), 5 seeds | 0.785 | 0.801 | 0.755 / 0.814 / 0.825 / 0.808 / 0.803 |

Readout 3's expectation held: 0.801 against 0.805. The actor's candidates are prior draws moved
by 0.03 to 0.09, so this was expected. None of the goal variants scored above A.

2026-10-12/13, luis, five seeds and curves. bon1 at 5 seeds: 250k 0.712 ± 0.025, 500k 0.696 ±
0.030 against A's 0.805 ± 0.014, a gap of 0.11 and about 3.4 standard errors. Best-of-2 0.541 /
0.399, best-of-4 0.218 / 0.213, so the ordering is 1 > 2 > 4 > 8 > 64. The BC-weight sweep
with the point value, 3 seeds:

| fb_bc_coeff | 250k | 500k | actor alone 500k |
|---|---|---|---|
| 0.3 | 0.757 ± 0.019 | 0.777 ± 0.050 | 0.187 |
| 1 | 0.782 ± 0.038 | 0.771 ± 0.014 | 0.114 |
| 3 (A, 5 seeds) | 0.789 | 0.805 ± 0.014 | 0.10 |
| 10 | 0.743 ± 0.031 | 0.719 ± 0.029 | 0.104 |
| 1000 (B) | 0.626 | 0.657 ± 0.045 | |

Flat from 0.3 to 3 within standard error; lower at 10; the actor alone rises as the BC weight
weakens, to 0.19 at 0.3. Against the 10-07 expectations: bc 0.3 at or above 0.80 did not hold;
bc 10 toward 0.62 to 0.65 held in direction only. Curves every 25k, bon1 (5 seeds) against A
(3 seeds): tied from 75k to 150k (0.39 to 0.60 both), the actor ahead from 175k (0.694 against
0.641), bon1 flat at 0.70 to 0.74 from 225k on. Antmaze bon1 at goal_discount 0.99: task 1
reaches 0.48 to 0.51 at 175k in all three seeds and collapses by 225k; five-task peak 0.165.

2026-10-13, both machines, three simulator checks of the critics. One-action rollout check:
at six probe steps of A's episodes, 64 prior latents scored by both critics, A's top, B's top
and 6 random candidates each executed once and then A continues; rank correlation of each
critic's score with the real outcome 0.036 (A) and 0.026 (B) against a split-half noise ceiling
of 0.033; the top pick's success 0.617 / 0.615 against a random candidate's 0.617. No power:
one action barely changes the outcome. Pick-modes on A's own rollouts: the 64 decoded
candidates form one cluster at 96.5 percent of steps; the pick's gripper command is opposite to
the candidate majority 24 percent of the time overall (random candidate 12), 35 percent in
failed episodes (16) and 17 percent in successes (10).

2026-10-13, luis, failure stages (T1; 3 seed pairs x 5 tasks x 20 episodes per side, same
reset seeds):

| share of episodes | A | B |
|---|---|---|
| success | 0.817 ± 0.026 | 0.663 ± 0.003 |
| never grasped | 0.047 | 0.087 |
| grasped then dropped | 0.117 | 0.167 |
| held, not placed | 0.020 | 0.083 |

First grasp at step 32 (A) against 44 (B), first lift 40 against 52, length of successes 74
against 90. No single stage carries the gap; B is slower at every stage.

2026-10-13 to 10-15, luis, segment swap (T2) in both directions. At undecided states of an
episode owned by one critic (success probability between 0.2 and 0.8 over 8 branches), hand
20 or 50 steps to the other critic and then return. Episodes owned by A: handing to B raises
success by 0.07 (k = 20) and 0.09 (k = 50). Episodes owned by B: handing to A raises success
by 0.06 and 0.13. Whichever critic owns the episode, switching policy at an undecided state
helps. The test does not rank the critics; the ranking stays with full-episode numbers.

2026-10-13 to 10-15, luis, mean-of-8 bootstrap (T3, `bootstrap_source=bc_mean`,
`bootstrap_n=8`): the target softmax is the mean over 8 prior draws' softmax rows, so target
noise is reduced without any max. Pre-stated: if the actor's gain is lower target noise, this
reaches about 0.80; if it stays near bon1, the gain needs the actor's push. Result 0.735 ±
0.031 at 250k (0.674 / 0.755 / 0.775) and 0.683 ± 0.036 at 500k over 3 seeds (0.708 / 0.728 /
0.612; posted by luis 2026-10-17T01-11-35Z), against bon1 0.712 / 0.701 in the same pipeline.
Within noise of one draw at both checkpoints.

2026-10-15, kisski, the fullness and actor-mechanism diagnostics, then the trajectory-mass gate
and the nf2 decision. These are section 5.

2026-10-17, kisski, the sign-flip arm (`psmgoal_ja_fbc3_pt_bdsneg_cube`, A with
`bootstrap_delta_scale=-1.0`: target latent clip(eps - delta) instead of clip(eps + delta), the
actor trained unchanged; plan `docs/plans/2026-10-17-sign-flip-bootstrap.md`). Pre-stated:
direction carries the gain -> at or below B 0.63 / 0.66; only the step's presence matters -> at
A 0.79 / 0.80. Result: 0.623 (0.706 / 0.633 / 0.529; seed 0 per task 0.782 0.614 0.892 0.702 0.540) at 250k
and 0.640 (0.642 / 0.527 / 0.752) at 500k, B's level at both. Caveat known before the result: the arm's actor shrank its step to
0.022 to 0.031 by 500k against A's 0.088, so the mirrored target is also a shorter one. The
frozen-step arms (section 7, item 1) fix the length.

2026-10-15, aws, the state-reward comparison on A seed 0 at 500k (`tools/eval_state_reward.py`;
`coef_source=reward_gpi`, GPI over goal coefficients against a continuous state reward;
`hgoal_topk`, the usual readout with the 32 highest-reward states as goals; `bc`):

| reward | readout | mean reward per step | final | env success |
|---|---|---|---|---|
| cube height | reward_gpi | 0.450 | 0.366 | 0.031 |
| cube height | hgoal_topk | 0.674 | 0.702 | 0.030 |
| cube height | bc | 0.915 | 0.548 | 0.120 |
| effector height | reward_gpi | 1.850 | 1.898 | 0.036 |
| effector height | hgoal_topk | 1.990 | 2.034 | 0.014 |
| effector height | bc | 1.354 | 0.683 | 0.120 |

The reward-weighted readout beats the goal readout in no row and no task. On cube height BC
wins because BC rollouts sometimes lift the cube high (max 3.05 against 1.75) and the
measure-guided ones almost never do. This is consistent with a measure that recognises goals
and carries nothing about the states leading to them.

2026-10-15 to 10-16, aws, the step-scale sweep (`bootstrap_delta_scale` = m, the actor's step
in the target only, m = 1 is A exactly; 100k steps, seed 0, checkpoints every 25k; pre-stated:
success follows m if the step-compounding reading is right, m = 4 may collapse like best-of-N):

| m | 25k | 50k | 75k | 100k |
|---|---|---|---|---|
| 0.5 | 0.338 | 0.339 | 0.404 | 0.532 |
| 2 | 0.413 | 0.480 | 0.557 | 0.637 |
| 4 | 0.420 | 0.502 | 0.575 | 0.628 |

m = 2 and m = 4 level within 0.025 at every checkpoint; m = 0.5 lower by 0.08 to 0.16; m = 4
did not collapse. There is no m = 1 checkpoint evaluated at 100k, and A and B were tied until
150k, so this sweep may be too early to see the A-versus-B effect. Clean L4 speed for A's
configuration is 0.28 s per step, 14 times slower than an H100, which is why aws runs 100k
arms and evaluations and kisski trains full-length seeds.

2026-10-15, kisski, time-to-success (`tools/diag_time_to_success.py`, A and B seed 0 at 500k,
100 episodes per task, horizon 200): best-of-64 with A 0.696 ± 0.021, best-of-8 with A 0.894 ±
0.014, BC 0.128, the actor alone 0.108, best-of-64 with B 0.714 ± 0.020; median steps to
success 59 (A) against 68 (B). The best-of-8 above best-of-64 and the 0.696 below the
500-episode evaluation of the same checkpoint (0.756) are unexplained in the record.

2026-10-17, kisski, the shift-direction diagnostic and the sign-flip arm. These are 4.5 and 7.

### 4.3 Every bootstrap variant in one table

| bootstrap latent in the TD target | group | seeds | 250k | 500k |
|---|---|---|---|---|
| the dataset's own next latent, softmax, geometric goals | db_sm_cube | 3 | 0.619 | 0.646 |
| same, squared | db_sq_cube | 3 | 0.602 | 0.520 |
| same, uniform goals, softmax / squared | dbu_sm / dbu_sq | 3 | 0.506 / 0.479 | 0.501 / 0.448 |
| same, squared, ortho 1.0 | dbo_sq | 3 | 0.068 | 0.039 |
| data bootstrap plus a trained DSRL actor, squared | jd_dsrl_sq | 3 | 0.419 | 0.399 |
| one BC prior draw | bon1 (3) / bon1_sm25 (5) | 3 / 5 | 0.713 / 0.712 | 0.708 / 0.696 |
| best of 2 BC draws | bon2 | 3 | 0.541 | 0.399 |
| best of 4 | bon4 | 3 | 0.218 | 0.213 |
| best of 8 | bon8 | 3 | 0.020 | 0.096 |
| best of 64 | bon64 | 3 | 0.006 | cancelled |
| mean of 8 BC draws' softmax rows | bcmean8 | 3 | 0.735 | 0.683 |
| flowbc actor step mirrored in the target (scale -1), actor trained as A | ja_fbc3_pt_bdsneg | 3 | 0.623 | 0.640 |
| frozen flowbc field from A seed 0 at 500k, +step | ja_fbc3_pt_fdplus | 3 | 0.770 | 0.741 |
| same field, -step | ja_fbc3_pt_fdminus | 3 | 0.596 | 0.577 |
| same field, random direction of the same length | ja_fbc3_pt_fdrand | 3 | 0.692 | 0.683 |
| same field, zero step (one BC draw) | ja_fbc3_pt_fdzero | 3 | 0.727 | 0.687 |
| flowbc actor, bc 0.3, shaped | ja_fbc03_sh | 3 | 0.758 | 0.706 |
| flowbc actor, bc 3, shaped | ja_fbc3_sh | 5 | 0.700 | 0.710 |
| flowbc actor, bc 0.3, point | ja_fbc03_pt | 3 | 0.757 | 0.777 |
| flowbc actor, bc 1, point | ja_fbc1_pt | 3 | 0.782 | 0.771 |
| flowbc actor, bc 3, point (A) | ja_fbc3_pt | 5 | 0.789 | 0.805 |
| flowbc actor, bc 10, point | ja_fbc10_pt | 3 | 0.743 | 0.719 |
| flowbc actor, bc 1000, point (B) | ja_fbcBC_pt | 3 | 0.626 | 0.657 |
| flowbc actor, bc 0, shaped | ja_fbc0_sh | 2 | 0.021 | 0.033 |
| DSRL actor, no BC, shaped / squared / squared plus ortho | ja_dsrl_sh / ja_dsrl_sq / jao_dsrl_sq | 3 | 0.014 / 0.004 / 0.142 | 0.062 / 0.000 / 0.080 |
| BC draw for 100k, then the actor | ja_p2a_pt | 3 | 0.763 | 0.806 |
| the actor for 100k, then BC draw | ja_a2p_pt | 3 | 0.690 | 0.727 |
| actor goals = the measure's goals | ja_actmg_pt | 3 | 0.727 | 0.751 |
| OGBench goal mix | ja_ogmix_pt | 3 | 0.784 | 0.755 |
| 30 percent random measure goals | ja_rand30_pt | 3 | 0.753 | 0.704 |
| no-stitch data mask | ja_nostitch_pt | 3 | 0.730 | 0.786 |
| random-drop data mask | ja_randdrop_pt | 3 | 0.786 | 0.787 |
| actor step x 0.5 / x 2 / x 4 in the target, 100k only | bds05 / bds2 / bds4 | 1 | 0.532 / 0.637 / 0.628 at 100k | |
| actor step x -1 in the target | ja_fbc3_pt_bdsneg | 3 | running | |
| A plus own-future normalisation | ja_fbc3_pt_nf2 | 3 | 0.865 | 0.866 |
| A, candidates from the actor at test time only | flowbc_gpi readout | 5 | 0.785 | 0.801 |

The actor acting alone is 0.098 to 0.132 in every actor arm except bc 0.3 with the point value
(0.176 / 0.187) and the collapsed no-BC arms.

### 4.4 Hypotheses about the actor: the deciding measurement for each

| hypothesis | deciding measurement | number | status |
|---|---|---|---|
| The actor is not needed; one BC draw does the same | bon1 at 5 seeds against A at 5 seeds | 0.696 ± 0.029 against 0.805 ± 0.014 | refuted |
| A greedy max over BC draws replaces the actor | bon2 / 4 / 8 / 64 | 0.399 / 0.213 / 0.096 / 0.006 | refuted; the max over noisy target estimates overestimates |
| The gain is lower target noise | mean of 8 draws | 0.735 / 0.683 against bon1 0.712 / 0.701 | refuted by the pre-stated criterion of about 0.80 |
| The actor regularises early training | curves; p2a against a2p | tied to 150k, gap opens after; 0.806 against 0.727 | refuted; the effect lives in the fixed point |
| The gain scales with the step size | BC weight 0.3 / 1 / 3 / 10 / 1000 | 0.777 / 0.771 / 0.805 / 0.719 / 0.657 | flat from 0.3 to 3; a threshold, not a slope |
| The gain scales with the step in the target alone | m = 0.5 / 2 / 4 at 100k, 1 seed | 0.532 / 0.637 / 0.628 | m at or above 2 level, 0.5 lower; open at 100k |
| Actor candidates at test time beat prior candidates | flowbc_gpi | 0.801 against 0.805 | refuted |
| Shape: the critic's preferred latents form a set a Gaussian actor cannot output | top-5 percent set shape | fraction in the fitted Gaussian ball 0.600 (exact Gaussian 0.587, control 0.579); 2-means separation 1.13 (control 0.95); kurtosis -0.12 | refuted: one Gaussian-shaped cluster 1.96 from the prior centre, narrower than the prior |
| Optimism: A's target rates the actor's latent higher than B does | q(actor latent) - q(base draw) | +1.4 percent under A online and target, +1.5 / +1.4 under B | refuted |
| Sharpening: A separates latents at a state more than B | within-state over across-state spread; gradient norm | 0.0335 against 0.0317; 0.0076 against 0.0054 | refuted |
| A ranks single actions more like real outcomes | one-action rollout check | 0.036 against 0.026, noise ceiling 0.033 | no power |
| A acts better than B over 20 to 50 step segments | segment swap both ways | switching to the other critic helps by 0.06 to 0.13 either way | does not rank the critics |
| B loses at one stage | failure stages | never grasped +0.04, dropped +0.05, held-not-placed +0.06 | refuted; B slower everywhere |
| Direct start-to-goal trajectories carry the result | nostitch / randdrop | 0.786 / 0.787 | refuted |
| The goal mix or measure-matched actor goals help | ogmix / rand30 / actmg | 0.755 / 0.704 / 0.751 | refuted |
| Actors fail at inference by not reaching the preferred cluster | membership; step length | 5.4 percent against 4.2 for raw draws; step 0.088 on a draw of 2.13 | supported |
| Exploration in noise space: only the perturbation off the exact draw matters, direction irrelevant | frozen-step arms (same field, same length): random direction against zero step | random 0.692 / 0.683 against zero 0.727 / 0.687 at 250k / 500k; prediction was at A 0.79 / 0.80 | refuted |
| Gradient step: the direction of the step carries the gain | frozen-step arms: plus against random and minus; sign-flip; shift diagnostic | plus 0.770 / 0.741, random 0.692 / 0.683, minus 0.596 / 0.577; sign-flip 0.623 / 0.640; step along the gradient at cosine 0.35 | supported; the gain over no step is 0.04 to 0.05, the loss from reversing it 0.11 to 0.13 |
| Why A beats B at all | the only difference found | A's share level 0.151 against 0.107 | open |

### 4.5 Where the step points (2026-10-17, `tools/diag_actor_shift_direction.py`)

A seed 0 at 500k, 512 dataset states with their hindsight goal, 2000 prior draws per state,
critic A's online network. The step is delta(s, g, eps); its constant part is the mean over
eps. References at each state: the centroid of the top 5 percent latents under the critic's
share; the data latent; the critic's mean gradient in u at the draws; the draw itself.

| cosine of the constant part of the step with | value ± SE |
|---|---|
| the top-set centroid | 0.340 ± 0.021 |
| the critic's mean gradient | 0.348 ± 0.022 |
| the data latent | 0.078 ± 0.021 |

The gradient and the top-set centroid point the same way, cosine 0.947, so the critic's local
slope and its global preference agree. The centroid and the data latent do not, cosine 0.098.
Per draw: the step points from the draw toward the centroid with cosine 0.245 (70 percent
positive), along the gradient 0.297 (70 percent), toward the data latent 0.087 (58 percent),
outward from the prior centre -0.059 (45 percent). Two random directions in 5 dimensions have
mean cosine 0 and mean absolute cosine 0.375. The step is 0.088 long, its constant part 0.080,
82 percent of its second moment is the per-(s, g) constant. 88 percent of the step's variance is
across goals at a fixed state; the hindsight-goal step and a random-goal step agree with cosine
0.51. In action space the decoded action moves 0.010 while the recorded action is 0.298 away;
the move points toward the decoded top-set centroid with cosine 0.24 and toward the recorded
action with 0.06; the gripper sign flips in 0.9 percent of draws.

Reading: the step is a goal-conditioned push toward what the critic ranks highest, 4 percent
of the way, following the critic's local gradient, not toward the recorded action. This is the
form of a deterministic-actor-critic improvement step. It does not show that the direction is
what produces the gain; the sign-flip arm tests that.

### 4.6 What the actor is, in plain terms, as of 10-17

The actor takes a prior draw eps, which decodes to a behaviour-cloning action, and adds a step
delta(s, g, eps) that is 4 percent of the draw's length and mostly a fixed vector per state and
goal. It is trained to raise the critic's share on the goal with a penalty of 3 on the squared
step, and the penalty wins. The critic bootstraps from the stepped latent at the next state.
At test time the actor is not used by the scoring readout, which takes the best of 64 prior
draws under the critic. Acting with the actor alone gives BC-level success. The gain from
having the step in the target is 0.15 of five-task success. It switches on at a small step and
saturates: zero step 0.66, half step at 100k below the others, step 0.80, 2x and 4x the step no
better, a larger greedy step by argmax over draws collapses. It does not come from averaging
noise, from the critic favouring its own actor, from a sharper critic, from early training, from
the data's direct trajectories, from the goal mix, or from the shape of the critic's preferred
set. Whether it comes from the direction of the step or from the step's mere presence is the
open question, and the sign-flip arm decides it.

Update 2026-10-18, after the frozen-step arms (section 7, item 1). The direction is what matters.
With the step field frozen from critic A and the step length held fixed, a step in the actor's
direction scores 0.741 at 500k, no step 0.687, the same length in a random direction 0.683, and
the reversed direction 0.577. The gain over no step is 0.04 to 0.05 of five-task success, not
the 0.15 read off critics A and B, because B's three seeds sit at the low end of the no-step
level (B 0.657, bon1 0.701, fdzero 0.687). The reversed step costs 0.11 to 0.13. A co-trained
actor is not needed to get the gain; whether it adds the remaining 0.06 between fdplus and A
is not separable with three seeds.

---

## 5. The route to a full successor measure

### 5.1 The question posed on 2026-10-15

A successor measure says how much discounted future time the policy spends near each state,
starting from a state and action. The 10-01 failing-point audit had shown, for the squared-loss
agent, that the measure sat at its bound at the transition's own next state and at 0.005
elsewhere, with no dependence on u. The softmax agents of 10-05 onward reached 0.80, so the
question was put directly: is the trained measure of arm A a successor measure, or a detector of
the exact next state and goal? Design decisions fixed by the user the same morning, before any
measurement: the form stays M = phi(s,u,s+)^T h(g) + b(s,u,s+); no factorised A(s,u)^T f(s+)
path; no goal fed into phi; no cube-position-only s+ for the measure.

Pre-stated expectations for the two diagnostics of the day: the measure's distribution over
dataset states concentrated on the next state for both critics and as far from the rollout
occupancy as uniform; for the actor-mechanism tool, A's spread larger than B's, a small
A-target-only optimism, and a non-Gaussian top set. All three actor expectations failed
(section 4.4); the fullness expectation held.

### 5.2 Fullness (`tools/diag_measure_fullness.py`)

Method. Take probe states from deployed episodes (5 tasks x 2 episodes, steps 0 to 80). For
each probe, take a goal g, either the task goal with the highest share or a random dataset
state. Read the measure at (s, u; h(g)) for the actor's latent u as a distribution p over 1024
fixed random dataset next states, by softmax of M over those states at temperature 1. Roll the
g-policy (best of 64 under h(g), re-drawn each step) out 8 times for 100 steps, map each visited
state to its nearest of the 1024 states, and form the discounted visitation rho. Compare p and
rho by total variation tv = 0.5 sum |p - rho| (1 = disjoint), by rank correlation, and by
entropy H(p) (maximum ln 1024 = 6.93). Controls: tv between the uniform distribution and rho,
and the split-half tv between two halves of the rollouts (the noise floor).

| critic, step | goal | tv(p, rho) | tv(uniform, rho) | noise floor | rank corr | p on the state nearest the actual next state | H(p) | H(rho) |
|---|---|---|---|---|---|---|---|---|
| A 500k | task | 0.965 ± 0.004 | 0.972 | 0.160 | 0.081 | 0.0012 | 6.79 | 2.12 |
| A 500k | random | 0.952 ± 0.004 | 0.956 | 0.232 | 0.049 | 0.0013 | 6.79 | 2.58 |
| B 500k | task / random | 0.967 / 0.954 | same | same | | 0.0012 / 0.0010 | 6.79 | |

tv between A's and B's distributions 0.09 to 0.10. Matching rollouts on cube position only
gives the same picture (tv 0.968 / 0.956 against uniform 0.974 / 0.966).

Conclusion written in the 10-15 entry: the measure's distribution over dataset states is flat
(entropy 6.79 of 6.93), as far from the rollout occupancy as the uniform distribution, with
1/1024 on the dataset state nearest the actual next state. Critic B is identical. The measure
carries no occupancy information about states other than the exact goal it is indexed by; it is
a goal detector. Two consequences were written at once: a reward-weighted readout has nothing
to read (confirmed by the aws state-reward run, section 4.2), and adding own-future states to
the normalisation set "would supervise exact trained states only; the measure does not transfer
to the nearest dataset neighbour, so a full measure over 28-dim states is not expected from it
either." That counter-expectation is on the record and the nf2 result is read against it.

### 5.3 Why the measure is a goal detector: the reasoning written for the fix

The softmax loss of section 2.4 reads the measure at (s_i, u_i) as a distribution over the
candidate states. In arm A the candidates are the batch's N next states. The target puts
(1 - gamma) on the transition's own next state and gamma on wherever the target measure at the
next state puts its mass over the same candidates. The only candidate that transition i's
trajectory ever visits is its own next state. Every other candidate is the next state of a
different, randomly drawn transition, which the trajectory of transition i does not pass
through. So the target has no candidate where the trajectory actually goes later, and the only
mass it can learn to place is on the own next state, with the rest as flat as the loss allows.
Measured: the own-next-state share is 0.27 to 0.29 at 500k in A; the remainder is spread over
255 states that carry no information.

A full successor measure needs candidates the policy reaches later. The proposed change: add,
for each transition, K of its own later states to the candidate set, drawn at offsets with the
same discount as the goal sampling. The target can then put mass where the trajectory goes.

### 5.4 The gate: trajectory mass (`tools/diag_measure_trajectory_mass.py`)

Before launching, one question was asked of the existing critic: does it put any mass at all
on its own trajectory's later states when those states are offered as candidates? If not, the
own-future candidates would be as uninformative as random ones and the plan should stop in
favour of a different estimator. Method: 512 dataset transitions; for each, its own state at
offsets k = 1, 2, 5, 10, 20, 50 and its hindsight goal are added to a set of 1024 random
reference states; the ratio reported is the measure's mass on the own state divided by the
mean mass of a random state. The control is another trajectory's state at the same offset.
Both the data latent and a fresh prior draw are used as u.

| k | A, data u / prior u | B, data u / prior u | A top-5 percent rate | other trajectory's state |
|---|---|---|---|---|
| 1 | 112.6 / 108.4 | 95.1 / 91.8 | 1.00 | 0.97 / 0.98 |
| 5 | 98.3 / 95.8 | 81.9 / 79.5 | 1.00 | 0.98 |
| 10 | 85.2 / 82.8 | 63.4 / 61.8 | 0.97 | 0.98 / 0.99 |
| 20 | 55.5 / 53.6 | 39.4 / 38.4 | 0.89 | 0.98 |
| 50 | 13.8 / 13.4 | 9.2 / 9.5 | 0.49 | 1.05 / 1.04 |
| hindsight goal | 42.1 / 40.3 | 28.4 / 27.2 | 0.69 | |

Conclusion written: the measure recognises states of its own trajectory out to 50 steps and
gives other trajectories' states nothing. The mass does not depend on u (data latent and prior
draw agree within 4 percent), so what is recognised is trajectory membership, not where the
action takes the policy. This is consistent with the flat rollout result, because live-episode
states belong to no dataset trajectory. The gate passed as posed: the measure does transfer to
its own later states, so the target has something to move mass onto.

### 5.5 The design, as decided (`docs/plans/2026-10-15-norm-future.md`)

`norm_future_k = K`. The dataset sampler draws K later states of the transition's own
trajectory at offsets d ~ Geometric(1 - goal_discount), d >= 1, clipped at the trajectory end;
this is the hindsight-goal draw repeated K times. The candidate set becomes
S = [the N next states, then the N x K future states], shared by every transition; candidate i
is still transition i's own next state. The loss and target are those of section 2.4 over the
larger S. K = 0 is bit-identical to the previous code, verified on a fixed batch over three
updates for both the A arm and the squared default.

Left on the N next states, deliberately: the actor's point-value pool, the kernel pool for the
shaped value, the argmax-bootstrap scoring pool, and every evaluation readout. The squared loss
refuses K > 0. New training-log keys: `measure_p_own_next` (the own-next-state share),
`measure_p_future` (the total share on the transition's own K future candidates),
`phi_eff_rank`. Memory: a 256 x 768 mesh through the 1024-wide network, about 0.8 GB of
activations per layer, held on an H100 without chunking.

Pre-stated expectations: own-next-state share below 0.27; own-future share far above the
2/768 = 0.0026 that K uninformative candidates would get; phi's effective rank above 1 without
the ortho term; five-task success may fall from 0.805; mass on other trajectories' states may
stay flat (the open risk).

The arm: A's configuration plus `agent.norm_future_k=2`, group `psmgoal_ja_fbc3_pt_nf2_cube`,
3 seeds, 500k steps, one seed per GPU. aws measured 0.62 s per step on a shared L4, so training
moved to kisski; the packed three-seed job failed after 20 s and the per-seed jobs were
submitted with a 12-hour limit on 10-15 at 16:48.

---

## 6. nf2

### 6.1 Results

Five-task success, best of 64 prior draws under the per-goal share:

| step | seeds | mean | per seed | per task (seed 0) |
|---|---|---|---|---|
| 50k | 0 | 0.836 | | 0.850 0.814 0.950 0.822 0.746 |
| 100k | 0 | 0.876 | | 0.920 0.922 0.954 0.700 0.882 |
| 150k | 0 | 0.841 | | 0.888 0.812 0.978 0.842 0.686 |
| 200k | 0 | 0.888 | | 0.886 0.940 0.978 0.866 0.768 |
| 250k | 0, 1, 2 | 0.865 | 0.914 / 0.829 / 0.852 | 0.904 0.916 0.970 0.884 0.894 |
| 500k | 0, 1, 2 | **0.866** | 0.912 / 0.825 / 0.861 | 0.914 0.938 0.950 0.850 0.910 |

Seeds 1 and 2 at 500k were trained on luis (jobs 8674932 / 8674933, 11.9 h each on an H200) and
evaluated on kisski (0.825 / 0.861; per task sd1 0.938 0.774 0.704 0.838 0.870, sd2 0.868 0.908
0.918 0.666 0.946) and on aws (0.815 / 0.870): machines within 0.01. The three-seed 500k mean
equals the 250k mean (0.866 against 0.865); the pre-stated 0.88 to 0.92 was not met, and seed
0's 0.912 is the high seed, not a 500k gain.

Against A: 0.789 at 250k over 5 seeds and 0.805 at 500k. The same checkpoint at 50k evaluated
on aws gave 0.834 (parity). Seeds 1 and 2 hit the 12-hour limit at step 375,000 and have only
their 250k checkpoint; seed 0 was relaunched under a 16-hour limit and finished in 15 h 56 min.
Each step costs 2.3 times A's because the candidate set is three times larger.

Training signals, seed 0 (5k-step means):

| step | own-next-state share | own-future share | phi effective rank | psm_loss |
|---|---|---|---|---|
| 50k | 0.382 | 0.359 | 1.050 | 1.69 |
| 100k | 0.388 | 0.321 | 1.083 | 1.87 |
| 250k | 0.358 | 0.292 | 1.064 | 2.44 |
| 500k | 0.344 | 0.250 | 1.049 | 2.90 |

Seeds 1 and 2 at 375k: own-next 0.352 / 0.356, own-future 0.260 / 0.292, rank 1.084 / 1.103.
In A the own-next share at 500k was 0.27 to 0.29 over 256 candidates.

Against the pre-stated expectations: own-next share below 0.27 did not hold (0.33 to 0.36 over
768 candidates). Own-future share far above 0.0026 held (0.25 to 0.36). Rank above 1 held
marginally (1.05 to 1.10 against A's 1.01 to 1.03). Success did not fall; it rose by 0.08 at
250k over 3 seeds and by 0.11 at 500k on one seed.

### 6.2 Diagnostics on the nf2 critic (seed 0)

Fullness, the same method as 5.2:

| step | goal | tv(p, rho) | tv(uniform, rho) | noise floor | rank corr | p on the state nearest the next state | mass beyond the next state | H(p) |
|---|---|---|---|---|---|---|---|---|
| A 500k | task / random | 0.965 / 0.952 | 0.972 / 0.956 | 0.160 / 0.232 | 0.081 / 0.049 | 0.0012 / 0.0013 | 0.034 / 0.047 | 6.79 |
| nf2 50k | task / random | 0.928 / 0.878 | 0.966 / 0.951 | 0.207 / 0.263 | 0.157 / 0.157 | 0.004 / 0.004 | 0.068 / 0.118 | 3.81 / 4.78 |
| nf2 250k | task / random | 0.920 / 0.894 | 0.968 / 0.952 | 0.195 / 0.248 | 0.146 / 0.163 | 0.0030 / 0.0039 | 0.077 / 0.102 | 4.99 / 5.62 |
| nf2 500k | task / random | 0.942 / 0.914 | 0.971 / 0.949 | 0.175 / 0.284 | 0.127 / 0.126 | 0.0037 / 0.0031 | 0.054 / 0.083 | 5.55 / 5.98 |

The distribution is no longer flat: entropy fell from 6.79 to 3.8 at 50k and climbed back to
5.6 by 500k. It moved toward the rollout occupancy by 0.03 to 0.07 in total variation against a
noise floor of 0.18 to 0.28, so it is still far from it. On cube position alone, nf2's tv is no
better than uniform at any checkpoint. tv between the nf2 and B distributions is 0.55 to 0.60,
where A against B was 0.09, so the training change moved the measure substantially, mostly not
toward the rollouts.

Fullness against the actor policy and BC (aws, 2026-10-17, `+rollout_policy=actor|bc|bon64`,
code 52892fd; JSONs on HF `runs/measure_fullness/fullness_{A_actor,A_bc,nf2_actor,nf2_bc,nf2_bon64}_sd000_500k_aws.json`).
The 10-15 test and the table above compare the measure with rollouts of the best-of-64 policy.
The measure is trained to evaluate the actor policy u = clip(eps + delta(s', g, eps)), so the
comparison was rerun with the rollouts (and the probe episode) driven by the run's own actor,
by one clipped prior draw (BC), and by best-of-64 as the control. Same settings as 5.2
(2 episodes per task, 8 rollouts of 100 steps per probe, 1024 reference states, seed 0, 500k).

| critic | rollout policy | goal | tv(p, rho) | tv(uniform, rho) | split-half noise | rank corr | H(p) | H(rho) |
|---|---|---|---|---|---|---|---|---|
| A | actor | task / random | 0.917 / 0.921 | 0.931 / 0.930 | 0.352 / 0.354 | 0.131 / 0.106 | 6.79 | 3.22 / 3.21 |
| A | bc | task / random | 0.911 / 0.916 | 0.925 / 0.926 | 0.385 / 0.391 | 0.126 / 0.101 | 6.79 | 3.43 / 3.43 |
| nf2 | actor | task / random | 0.881 / 0.874 | 0.930 / 0.927 | 0.354 / 0.376 | 0.213 / 0.176 | 5.39 / 5.92 | 3.27 / 3.37 |
| nf2 | bc | task / random | 0.877 / 0.891 | 0.929 / 0.927 | 0.355 / 0.353 | 0.206 / 0.163 | 5.35 / 5.99 | 3.30 / 3.31 |
| nf2 | bon64 | task / random | 0.935 / 0.911 | 0.966 / 0.950 | 0.201 / 0.261 | 0.138 / 0.134 | 5.59 / 5.95 | 2.49 / 2.87 |

Critic B against A's actor rollouts: tv 0.920 / 0.921. On cube position alone, nf2 under the
actor and BC is worse than uniform (0.93 to 0.95 against 0.88); under best-of-64 it is not.
Pre-stated: the rollout occupancy under the actor and BC is broader (met: H(rho) 2.5 to 2.9
under best-of-64, 3.2 to 3.4 under the actor and BC); tv falls below the best-of-64 value but
stays above the noise floor (met: 0.87 to 0.92 against a floor of 0.35); if nf2's tv under its
own actor had dropped near the floor the 10-15 conclusion would have been withdrawn (not met).
The uniform baseline falls as much as tv does, so measured against uniform A is 0.010 to 0.014
better and nf2 0.036 to 0.053 better, against 0.031 to 0.039 under best-of-64. nf2 fits the
actor's occupancy no better than BC's. The measure is not a successor measure of the policy it
bootstraps from either. Caveat from aws: the probe episode is walked by the same policy, so the
probe states differ across the three policies (probe episodes succeeded 0/10 under A's actor,
2/10 under nf2's, 8/10 under best-of-64). Rerun with the probe episode walked by best-of-64
(`+probe_policy=bon64`, aws 2026-10-17T05-23-01Z, JSONs `*_bonprobe_aws.json`): nf2 under its
actor tv 0.895 / 0.900 against uniform 0.939 / 0.938 and noise 0.32; nf2 under BC 0.889 / 0.895;
A under its actor 0.925 / 0.929 against uniform 0.934 / 0.935; H(rho) 2.8 to 3.2. tv moves by at
most 0.02 between the two probe sets. The caveat is closed; the conclusion stands.

Trajectory mass, the same method as 5.4, ratio of the own later state's mass to a random
state's:

| k | A 500k | nf2 50k | nf2 250k | nf2 500k |
|---|---|---|---|---|
| 1 | 112.6 | 819.7 | 536.1 | 415.5 |
| 10 | 85.2 | 662.8 | 394.5 | 304.5 |
| 50 | 13.8 | 293.2 | 175.4 | 136.6 |
| hindsight goal | 42.1 | 448.1 | 269.7 | 221.5 |
| other trajectory's state at the same offset | 0.97 to 1.05 | 0.45 to 0.64 | 0.86 to 0.99 | 0.89 to 1.00 |

Dependence on u: the ratio of the own-future mass under the data latent to that under a prior
draw is 1.00 to 1.05 at every offset and checkpoint (A: 1.03 to 1.04). The mass still does not
depend on the action latent. Mass on other trajectories' states at the same offset stays at the
random level.

### 6.3 What nf2 did and did not do

Did: gave the target candidates the trajectory actually reaches; the measure put a quarter to a
third of its mass on them; the own-next-state share fell from about 0.5 of 256 candidates to
0.34 of 768; the distribution over dataset states concentrated; five-task success rose above A.

Did not: make the mass depend on u, so the measure still does not say where the action takes
the policy; transfer to states on other trajectories, even at the same offset; move the
distribution to the rollout occupancy beyond a small step; raise phi's effective rank beyond
1.1. The measure after nf2 is a stronger own-trajectory detector: "this state lies some steps
ahead on my trajectory", with the same answer for every action latent.

Why success rose while the action dependence did not: unmeasured. Two candidate accounts are
on record only as conjecture. One is that the hindsight-goal column now shares the target's
mass with its neighbours along the trajectory, so the goal share at test time is smoother along
the path to the goal and the ranking of candidates improves through the goal column alone. The
other is that the larger candidate set changes the temperature-like scale of the shares and
makes the ranking at test time less noisy. Neither has a measurement.

---

### 6.4 The pool measure (2026-10-19, `docs/design/2026-10-19-pool-measure.md`)

The brainstorm of 10-17 asked for the full successor measure: a distribution over a fixed set of states
instead of a detector of the trajectory's own next state. Built as `measure_loss=pool`: the softmax runs
over a fixed reference pool R of 1024 dataset next states, the one-hot sits on the pool state nearest the
row's next state (standardised full-state distance), the target is (1 - gamma) one-hot + gamma softmax of the
target measure over R, everything else as arm A. Three arms, 3 seeds, 250k: `bank` (policy index = one BC
draw, no actor, a policy evaluation of BC; readout `pool_reward`, the importance-weighted reward integral
over 512 relabel states), `fdplus` (A's goal index with the frozen +step field), `ja` (A's goal index with the
live flowbc actor).

| five-task cube, 250k, 3 seeds | hgoal_each | pool_reward |
|---|---|---|
| pool bank | | 0.252 ± 0.063 |
| pool fdplus | 0.326 ± 0.041 | 0.292 ± 0.033 |
| pool ja | 0.328 ± 0.092 | 0.260 ± 0.085 |
| arm A / frozen +step arm (batch softmax), same readout | 0.789 / 0.74 | |
| BC | 0.111 | |

Expected before the runs: 0.70 to 0.80 for the goal arms, 0.6 to 0.72 for the bank. Fullness
(`tools/diag_measure_fullness.py`, bank sd000 and fdplus sd000 at 250k, kisski and aws in parity): in
full-state cells the pool measure's distribution over the reference states is 0.07 to 0.19 above the split-half
floor under its own policy (bank under BC tv 0.45 / floor 0.38; fdplus under its actor 0.60 to 0.69 / floor 0.33
to 0.40), where critic A and nf2 were 0.65 to 0.80 above it; under best-of-64 rollouts 0.65 to 0.84 against
floors of 0.17 to 0.26. In cube-position cells it is at the uniform level in every line (tv 0.91 to 0.97,
Spearman 0.00 to 0.07): the occupancy puts 0.3 to 0.5 of its mass on the next state's cube cell and the measure
puts 0.01 there. Training signals: anchor mass 0.08 to 0.10 (A's own-next mass 0.45 to 0.49), pool entropy 3.7
to 4.2 nats of 6.93, pool_u_tv 0.05 to 0.08 flat from 5k, phi effective rank 1.15 to 1.25.

What this settles. The acting of A and nf2 comes from the one-hot on the trajectory's own next state over the
batch's next states: with that replaced by a real distribution over a fixed set, success falls from 0.79 to 0.33
with the same actor, index, bootstrap and readout. The full-state agreement the pool loss buys is in the arm
and gripper coordinates that dominate the standardised distance; the cube position, the only thing the reward
reads, is not resolved. The pool measure is a diagnostic result in this form. The next lever written down and
not built: weight the cube position in the distance that defines the anchor and the cells. The 500k pool
checkpoints are evaluated for the record only.

### 6.5 The reward index (2026-10-19, `docs/design/2026-10-19-reward-index.md`; 250k three seeds: best-of-64 at w 0.389 ± 0.052, actor at w 0.107 ± 0.009, against A 0.789)

Density FB's second arm, `measure_density_z`: the critic's task index is a vector solved from reward samples
(centred ridge fit of the reward on the state tower, at training on hindsight kernel rewards, at test on the
relabelled rewards), with the goal arm's one-hot own-next-state loss. Cube-single in their stack: 82.9 ± 4.0
(state tower on cube position), 91.7 ± 2.2 (full state), goal arm 93.1 ± 1.8. The point against plain FB and
against psmgoal's earlier reward coefficients (regression 0.30): the index map used at test time is the one the
critic trains on. Ported as `policy_index=reward` (2d48a11): index feature = the projected phi `fb_feature` at the
slow basis copy (no new net), training rows w_i = ridge of the kernel reward of the geometric hindsight goal over
the batch's next states (half random), A's softmax loss, the flowbc actor at w with Q = sum_j softmax_j(M/t)
(f(y_j) - mu)^T w / sqrt(D), bootstrap from that actor at (s', w); `coef_source=ridge` at test with three
readouts (actor at w; best-of-64 by the goal mass at w; best-of-64 by the expected relabel reward). Arm
`psmgoal_ri_fbc3_cube`, 3 seeds, 250k, luis (H200) and kisski. Stated before results: best-of-64 at w 0.70 to
0.80, actor within 0.10, failure line 0.30; gate `index_proxy_corr` above 0.3 by 100k.
Result 2026-10-20 00:50, sd000 250k, five tasks, 500 episodes: best-of-64 by the goal mass at the ridge w
0.424 (0.682 / 0.360 / 0.638 / 0.348 / 0.092), best-of-64 by the expected relabel reward 0.272, the actor at w
0.115 (BC 0.111); critic A sd000 250k 0.818. The index gate passed (proxy correlation 0.63 at every 25k point)
while the spread of the index feature over the anchors fell 50x (index_cov_trace 0.08 to 0.0015). The
best-of-64 readout missed the 0.70 to 0.80 band by 0.28 and sits 0.12 above the failure line; tasks 1 and 3
keep A's shape, tasks 2, 4, 5 lose most of it; the actor at the index acts at BC, as every actor-at-w arm
since 10-05. aws reproduces the kisski cells within 0.05. Pending: seeds 1-2 at 250k (about 03:00 CEST), the
sd000 ladder 50k to 200k under the best-of-64 readout (does the readout fall as the feature spread shrinks),
luis's H200 seeds as replication once the luis login node returns.
Ladder (01:20): sd000 best-of-64 at w 0.260 / 0.283 / 0.392 / 0.328 / 0.424 at 50k / 100k / 150k / 200k / 250k,
rising 0.16 over the run while the feature spread shrinks 50x, so the shrink does not cost the readout and 250k
is not the end of the curve. 500k line launched (seeds 3-5, `psmgoal_ri_fbc3_500k_cube`); expected 0.50 to 0.65
at 500k if the slope continues at half rate.
Three seeds at 250k (10-20 04:10): best-of-64 at the ridge w 0.389 ± 0.052 (0.424 / 0.336 / 0.405), by expected
reward 0.214 ± 0.073, the actor at w 0.107 ± 0.009 (BC 0.111); tasks 1 and 3 held, tasks 4 and 5 lost. Half of
the goal index on the same critic; the actor at BC as in every actor-at-w arm.
Kernel actor reward (01:50): `reward_actor_reward=kernel` (8817b60), the actor climbs the exact kernel reward of
its own hindsight goal on the rows that keep one (Density FB's kernel mode; the paper's Eq. pretraining-reward
is the centered view the 250k arm ran). Arm `psmgoal_ri_fbc3_kern_cube` seeds 0-2, 250k. Expected: actor at w
0.30 or above if the reward rows were the limit; within 0.05 of BC if the BC term holds it.
Result (10-20 11:30, three seeds, 250k): the actor at w 0.104 ± 0.005 (BC 0.111), best-of-64 at w 0.408 ±
0.211, by expected reward 0.265 ± 0.142; the actor's latent step 0.007 (A's nudge 0.09) with a value term six
times the centered arm's. The kernel actor reward changes nothing; the reward index is closed at 250k, the 500k
line pending.

### 6.6 The advantage-guided actor (2026-10-20, `docs/design/2026-10-20-advantage-guided-actor.md`; 50k seed 0: the actor alone 0.480, best-of-64 0.507, BC 0.111)

The user's ask of 10-20: one network that is trained with the critic, supplies the TD bootstrap and acts at
test time, in place of A's nudge actor for the bootstrap plus an offline distilled student for acting. Port of
Zhou and Levine's advantage guidance (arXiv 2610.19783) onto A's recipe (`ag_actor`, 9bfebd3, 846808d): the
flowbc delta's goal slot takes [g, c]; per update ag_k = 16 prior draws are scored by the point value at the
actor goal, the scores standardised per state (the normaliser reads ag_pool = 32 of the batch's next states),
and delta regresses each draw conditioned on its standardised score (mean squared error, plain supervised
learning, no critic gradient through the actor). The bootstrap is clip(eps + ag_w_boot (mu(+1) - mu(-1))) read
from a Polyak copy of delta (tau 0.005), acting is the same at ag_w_eval. Cost at 16 / 32 equals A's step (0.26
s per update on an L4). Arms: `psmgoal_ag_cube` (w_boot 0.5, 3 seeds), `psmgoal_ag_wb025_cube` (0.25),
`psmgoal_ag_wb01_cube` (0.1); aws runs sd0 of each plus sd1-2 at 0.5, kisski sd1-2 of the two controls, luis
sd0-2 at 0.5 as the second copy. Stated before results (design doc): best-of-64 at or above A's 0.79 at 250k,
0.66 to 0.72 if the regression actor is too slow, below 0.5 if the tilt collapses the target; the actor alone
0.5 to 0.8 at the best ag_w_eval, within 0.05 of BC at every w reads "the actor ignores the score".
Signals at 25k and 50k (aws): the shift's size is 1.2 x w_boot in every run and its direction does not depend on
its size (cos to the best draw 0.30 to 0.37); own-next mass 0.39 to 0.45 at 50k (A 0.42). Ladder at 50k
(`tools/diag_bootstrap_ladder.py`, b5ce219 reads the guided bootstrap from the Polyak copy): the bootstrap is a
64th / 58th / 54th percentile BC draw at w_boot 0.5 / 0.25 / 0.1 (A's nudge 52nd), nearest-data distance 0.2146
/ 0.2133 / 0.2129 against the BC draw's 0.2127 and the best-of-64 pick's 0.224, cos to the critic's gradient
0.70 to 0.79 (A 0.32 to 0.35), TD loss with vs without the shift equal to the 4th decimal.
50k, seed 0, five tasks, 500 episodes (aws post 06-47-45Z):

| sd0 at 50k | t1 | t2 | t3 | t4 | t5 | five-task |
|---|---|---|---|---|---|---|
| w_boot 0.5, best-of-64 by the critic | 0.452 | 0.740 | 0.462 | 0.404 | 0.478 | 0.507 |
| w_boot 0.5, the actor alone at w 1 | 0.498 | 0.324 | 0.606 | 0.502 | 0.472 | 0.480 |
| w_boot 0.5, the actor alone at w 2 | 0.266 | 0.386 | 0.540 | 0.518 | 0.374 | 0.417 |
| w_boot 0.25, best-of-64 | 0.328 | 0.696 | 0.644 | 0.402 | 0.336 | 0.481 |
| w_boot 0.25, the actor alone at w 1 | 0.306 | 0.460 | 0.560 | 0.808 | 0.042 | 0.435 |
| w_boot 0.1, best-of-64 | 0.414 | 0.538 | 0.636 | 0.246 | 0.256 | 0.418 |
| w_boot 0.1, the actor alone at w 1 | 0.302 | 0.178 | 0.370 | 0.622 | 0.076 | 0.310 |
| BC | | | | | | 0.111 |

The actor that supplies the bootstrap acts at 0.31 to 0.48 at 50k; every actor-at-w arm since 10-05 and A's
nudge actor act at BC. Both readouts order with w_boot (0.5 above 0.25 above 0.1) at 50k. The reward-index
best-of-64 at 50k was 0.26; no 500-episode A number exists at 50k (the 25k-save rerun is queued).
Three seeds at 50k, w_boot 0.5 (09:35): best-of-64 by the critic 0.523 ± 0.103 (0.507 / 0.493 / 0.570), the
actor alone at w 1 0.431 ± 0.131 (0.480 / 0.438 / 0.376), BC 0.111.
100k (13:10 to 13:50): best-of-64 0.694 ± 0.182 (0.696 / 0.620 / 0.766), the actor alone 0.434 ± 0.240 (0.537 /
0.346 / 0.418); w_boot 0.25 sd0 0.692 / 0.522, w_boot 0.1 sd0 0.581 / 0.618. The critic's pick rises 0.17 from
50k and the actor stays flat, a 0.26 gap. The actor's regression loss is 0.92 on unit-variance draws from 5k
on (the score explains 8% of a draw), its gain relative to the critic's spread rises slightly, and the critic's
spread over draws shrinks 25% while its scores get more peaked. A larger tilt at test time does not recover
the gap: w 2 is 0.13 / 0.30 below w 1 on the 0.5 / 0.1 runs, w 4 is below BC. The ladder at 100k equals 50k
(bootstrap at the 64th percentile, 0.002 above a BC draw in distance to the data, TD loss equal). Reading:
the actor learns a smooth shift along the average score; the critic's gain is in which few draws it ranks on
top, and a smooth shift does not reproduce that choice. Pending: 250k three seeds (aws about 22:00 CEST), the
controls' seeds 1-2 on kisski, luis's second copy of the 0.5 line.

## 7. Open questions and the next arms, with predictions

Each item gives the question, the measurement that decides it, and the predicted outcome under
each live reading. Numbers are what to compare against.

1. Does the direction of the actor's step carry the gain, or only its presence?
   Arm `psmgoal_ja_fbc3_pt_bdsneg_cube` = A with `bootstrap_delta_scale=-1.0`, 3 seeds, 500k,
   running on kisski since 2026-10-17 00:57 (seed 0) and 01:35 (seeds 1 and 2), evaluations
   automatic at 250k and 500k. Gradient-step reading: at or below critic B, 0.63 / 0.66,
   possibly lower since the target then steps down the critic's value at every backup.
   Perturbation reading: at A's level, 0.79 / 0.80. In between, 0.70 to 0.75: both contribute,
   and the random-direction control (same length, random direction per draw, a few lines in
   the bootstrap selection) is the next arm. The actor acting alone stays at BC level in every
   case. The actor's own step in this arm was 0.019 at 75k against A's 0.088, so the actor
   learns a smaller step when its target is mirrored; that is a result in itself.
   Result 2026-10-17, eval500 hgoal_each: 0.623 (0.706 / 0.633 / 0.529) at 250k and 0.640
   (0.642 / 0.527 / 0.752) at 500k against B 0.626 / 0.657 and A 0.789 / 0.805. The
   gradient-step outcome (at or below B) is met at both checkpoints; the perturbation outcome
   (at A) is not. The arm's own step is 0.022 / 0.023 / 0.031 at 500k against A's
   0.088, so direction and length change together and the arm alone does not exclude a smaller
   perturbation as the cause. The clean test is the frozen-step arms
   `psmgoal_ja_fbc3_pt_fd{plus,minus,rand,zero}_cube` (`bootstrap_source=frozen_delta`, the
   step field loaded from A seed 0 at 500k and never updated; plan
   `docs/plans/2026-10-17-frozen-delta-bootstrap.md`, kisski jobs 2525526 to 2525537): plus
   expected at A 0.80 (below it if the co-trained actor matters), zero at B 0.66, minus and
   random at or below B under the direction reading and at A under the presence reading. The
   informative contrasts are plus against random and random against zero.
   First frozen-step numbers (250k, 09:55): plus 0.770 (0.726 / 0.792 / 0.791), at A's 0.789,
   so the frozen field reproduces A without co-training; minus seed 0 0.726 (per task 0.790
   0.542 0.956 0.548 0.794), above B and below A, one seed. Random and zero pending.
   At 500k (aws evals, 10-34-17Z): plus 0.742 (0.762 / 0.677 / 0.787; seed 1 fell to 0.430 on
   task 1), below the 0.78 to 0.82 stated beforehand and not separable from A 0.805 with three
   seeds; minus seed 0 0.620 (per task 0.706 0.666 0.602 0.676 0.450), B's level, down 0.106
   from 250k. Plus against minus on seed 0 at 500k: +0.142, plus ahead on four of five tasks.
   All four arms at 250k, 3 seeds each (10-18): plus 0.770, zero 0.727 (0.761 / 0.733 / 0.686),
   random 0.692 (0.809 / 0.718 / 0.550), minus 0.596 (0.726 / 0.521 / 0.542). Presence outcome
   (random at A) not met. Direction outcome met for minus and random; zero sits above B's 0.626,
   and the no-step level over B, bon1 and fdzero (nine seeds) is about 0.69, so the step's gain
   over no step on matched frozen-field seeds is +0.043 at 250k, not the 0.16 of A against B.
   Order: plus > zero > random > minus. 500k for zero, random and the minus seeds 1-2 pending.
   500k, 3 seeds each (10-18 05:27): plus 0.741 (0.756 / 0.671 / 0.795), zero 0.687 (0.728 /
   0.655 / 0.677), random 0.683 (0.709 / 0.654 / 0.687), minus 0.577 (0.614 / 0.475 / 0.642).
   Random equals zero at both checkpoints; minus is 0.11 below zero; plus is 0.054 above zero.
   The presence reading is refuted; the direction reading holds with a gain of 0.04 to 0.05 over
   no step. Open: the 0.06 between fdplus and A (co-trained actor or seed noise), and more seeds
   on plus and zero.

2. Does a step larger than the trained one help at full length?
   The aws sweep is one seed at 100k, and A and B were tied until 150k. Needed: m = 1 at 100k
   on the same evaluation (A seed 0 has no 100k checkpoint; a re-run with save_interval 25k
   gives it), and m = 2 at 500k over 3 seeds. Threshold reading: m = 2 at 500k within A's
   interval. Slope reading: above it.

3. nf2 at 500k over 3 seeds.
   Seeds 1 and 2 need a relaunch under a 24-hour limit, about 16 hours each on an H100. The
   250k three-seed number is 0.865; the 500k one-seed number is 0.912. Prediction from the seed
   0 curve: 0.88 to 0.92 at 500k over 3 seeds. A result at or below 0.865 would say the 500k
   gain on seed 0 is seed noise. luis launched fresh seeds 1 and 2 on 2026-10-17 (H200, 24 h,
   jobs 8674932 / 8674933, smoke flags norm_future_k 2). The same two checkpoints are evaluated
   on all three machines (`scripts/campaign_2026-10/nf2_threeway_watcher.sh` on kisski; luis and
   aws upload to HF `runs/nf2_threeway/<machine>/`); expected agreement within 0.01 on the
   five-task mean. Result 2026-10-19: 500k over 3 seeds **0.866** (0.912 / 0.825 / 0.861), equal
   to the 250k mean; kisski, aws and luis within 0.012 per seed (luis 0.813 / 0.866). The 0.88
   to 0.92 prediction was not met; seed 0 is the high seed. Settled: nf2 is 0.865 at both
   checkpoints, 0.06 to 0.08 above A.

3b. The deployment actor (luis, `tools/distill_bon_student.py`, 2026-10-17 to 10-19). A
   full-covariance Gaussian over the 5-dim latent, conditioned on the state and a fixed per-task
   descriptor (mean of h(g) and of g over the task's 32 rewarding goals), fit by maximum
   likelihood to the raw winning latent of best-of-64 under the nf2 seed 0 500k critic at 20,000
   dataset states plus 50 teacher rollouts per task. With early stopping on validation NLL (best
   at step 3,000) and weight decay 1e-4: eval500 five-task **0.918** with one sample per step and
   0.926 with the mode, against the teacher's 0.912 on the same checkpoint and BC 0.111; per task
   0.936 0.978 0.970 0.882 0.822 (sample). Without early stopping the student overfit (validation
   NLL above the prior on 4 of 5 tasks) and still acted at 0.902 / 0.898 on 500 episodes, so
   calibration changes acting by 0.02 to 0.03. Held-out task (trained on four tasks): mean
   descriptor 0.002; set-encoder descriptor, leave-one-out over the five tasks, 0.072 / 0.230 /
   0.002 / 0.000 / 0.000 (mean 0.061), validation NLL 12 to 21 times the prior's on the held-out
   task's states; with early stopping 0.034 / 0.446 / 0.000 / 0.000 / 0.000 (mean 0.096) and the
   NLL at the prior. With 20 random goal regions as the training tasks (anchor = a random
   row's cube position, the rows within the 4 cm success tolerance as the 32 goals, the set
   encoder, early stopping) and NO real task in training: 0.904 sample / 0.920 mode on the five
   real tasks (0.924 / 0.830 / 0.984 / 0.880 / 0.902), calibrated (validation NLL at the prior),
   teacher 0.912, in-task student 0.918. Real centroids lie 1.6 to 8 cm from the nearest random
   anchor (tasks 1 and 3 within tolerance of one, 2, 4, 5 not). Student-seed spread is large: the
   same labels at student seeds 0 / 1 / 2 give 0.904 / 0.957 / 0.860 (SD 0.049; task 4 moves 0.52 to
   0.93 with the student seed alone, at identical validation NLL), so from 10-19 every cell is a
   three-seed mean ± 95% CI. Three-seed random-region students (20 regions, 500 episodes) against
   their teachers: sd0 0.907 ± 0.121 (teacher 0.912), sd1 0.877 ± 0.033 (teacher 0.813), sd2
   0.867 ± 0.055 (teacher 0.866); mean over teachers 0.884 against 0.864; the student means span
   0.040 where the teachers span 0.099, and the in-task students (0.918 / 0.757, 0.787 / 0.820)
   carry the teacher's spread and more. Coverage control, sd0 teacher, three student seeds:
   keeping only the 12 random regions more than 8 cm from every real goal gives 0.455 ± 0.384,
   keeping the 8 of those that are not sparse gives 0.293 ± 0.209 (BC 0.111); nearest kept anchor
   8.0 to 13.2 cm; both students reach the same held-out validation NLL on their own regions as
   the 20-region student (6.92 to 6.99) and sit above the prior's NLL on the real tasks. So the
   transfer rests on random regions within about 5 cm of each real goal, not on label quality (the
   12-region and 8-region cells differ by 0.16, inside both CIs); task 4 transfers at 8.5 cm in five
   of six of these students and task 3 fails at 10.2 cm in all six, task 2 at 8.0 cm sits at 0.10 to
   0.48, so anchor distance alone does not order the tasks. A 50-region student scores 0.928 (one
   seed; seeds 1 and 2 in flight); adding the four other real tasks to the random regions and
   testing on the fifth lowers tasks 4 and 5 (50 regions: 0.924 to 0.782, 0.850 to 0.666), cause
   open. This closes the "actor fails at inference" item for in-task deployment: the failure was
   the actor objective (critic climb with a BC penalty), not the representability of the target.
   The zero-shot deployment actor exists at the resolution of the training regions' coverage.

4. Why nf2 raised success without action dependence.
   Measurement: the per-goal share at test time along a trajectory toward the goal, for A and
   nf2, at the same states, against the step distance to the goal; and the within-state spread
   of candidate scores for A and nf2. Smoother-share reading: nf2's share rises monotonically
   along the path where A's is flat until the last steps. Scale reading: nf2's within-state
   spread of candidate scores is larger relative to its noise.

5. Does the measure ever depend on the action latent?
   The trajectory-mass ratio under the data latent against a prior draw is 1.00 to 1.05 in A and
   nf2. A successor measure of a goal-reaching policy must give the data latent, which was the
   action the trajectory actually took, more own-future mass than a random draw. The 10-17
   fullness rerun against the actor policy (section 6.2) gives tv 0.87 to 0.92 against a noise
   floor of 0.35, so the measure is not a successor measure of its own bootstrap policy either. Arms that could
   produce this: more future states per transition (K = 8 or 16, at 3 to 5 times A's step cost);
   candidates drawn from other trajectories near the own future states, so the measure has to
   generalise in state space; a target that includes the own-future candidates in the actor's
   value pool as well, so the actor's step is trained on the same object the measure is. None is
   launched. Prediction under the "softmax over batch candidates never asks for action
   dependence" reading: K alone does not create it; the near-neighbour candidates do.

6. Antmaze.
   Every arm and readout sits at BC (0.03 to 0.12). The measure's own-next-state share there is
   0.08 to 0.09 against 0.45 to 0.49 on cube. bon1 on antmaze rose to 0.5 on task 1 at 175k in
   all seeds and collapsed by 225k. nf2 on antmaze is untested. Prediction: nf2 moves the
   own-next share up as on cube; success unknown.

7. Smaller items still open on the record: the time-to-success file's best-of-8 above best-of-64 on A seed 0 is
   unexplained; the actor-goal question (actmg 0.751) is answered on success only; the Factored
   FB z-arm reading of 10-15 (ridge task vector, 50 percent random indices, nonlinear F in z,
   B on cube position) is noted and not acted on.

---

## 8. Proposed write-up

A section-by-section outline for the paper's method and results, with which numbers and
figures go where and which claims the evidence supports. The notation follows section 2.

### 8.1 Title and claim

Zero-shot offline RL in the latent space of a frozen behaviour flow with a goal-indexed
successor measure. Claim supported by the evidence: on cube-single-play, a goal-indexed measure
over flow latents, trained with a softmax TD loss and a BC-anchored latent actor in its
bootstrap, reaches 0.805 five-task success against a 0.111 behaviour-cloning control, with
every executed action a decode of the frozen flow. Claim not supported: that the trained object
is a successor measure in the occupancy sense; the paper should say what it measures.

### 8.2 Method section

1. The frozen flow and the latent space (one paragraph, section 2.2 decode).
2. The goal-indexed measure M(s,u,g) = phi(s,u,g)^T h(g) + b(s,u,g); the reward indicator at the
   goal; the point value as the softmax share (section 2.4 and 2.6 formulas).
3. The softmax TD loss over candidate states, with the (1 - gamma) one-hot on the next state
   (section 2.4). State in words why shares and not raw values are read.
4. The bootstrap from a BC-anchored latent actor: u' = clip(eps + delta(s', g, eps)), the actor
   loss, the zero-initialised step (sections 2.5, 2.6). Cite TD3+BC and FQL for the loss form,
   EMaQ and IQL for the greediness knob.
5. Own-future candidates in the normalisation set (section 5.5), presented as the fix for a
   target that otherwise never contains a reachable state.
6. Test-time inference: 32 rewarding states as goals, 224 reference states, per-goal shares,
   best of 64 prior draws, decode (section 2.8).
7. Pseudocode: section 2.9 verbatim.

### 8.3 Results section

Table 1, the main result: five-task cube, 500 episodes per task, mean and 95 percent interval
over seeds, 250k and 500k: BC 0.111; A 0.789 / 0.805 (5 seeds); nf2 0.865 / 0.866 (3 seeds);
critic B 0.626 / 0.657; one BC draw 0.712 / 0.696 (5 seeds).

Table 2, what the bootstrap does (section 4.3 condensed): the data's next latent 0.646; one BC
draw 0.696; mean of 8 draws 0.735 / 0.683; best of 2 / 4 / 8 draws 0.399 / 0.213 / 0.096; the
actor at BC weight 0.3 / 1 / 3 / 10 / 1000: 0.777 / 0.771 / 0.805 / 0.719 / 0.657; no BC term
0.033; switch arms 0.806 (late) and 0.727 (early). The claim: a small in-support step in the
target is necessary and sufficient for the gain, a greedy step destroys it, averaging does not
produce it.

Table 3, the actor is not the policy: actor alone 0.108 in every arm; actor candidates at test
time 0.801 against prior candidates 0.805.

Figure 1, the critic's preferred set at a state is one Gaussian-shaped cluster 1.96 from the
prior centre (section 4.4 shape row), with the actor's step of 0.088 drawn to scale.

Figure 2, the measure is a trajectory detector: trajectory-mass ratios by offset for A and nf2,
with the other-trajectory control at 1, and the data-latent against prior-draw ratio at 1
(sections 5.4, 6.2).

Figure 3, fullness: entropy and total variation to the rollout occupancy for A and nf2 across
checkpoints, with the uniform and noise-floor references (section 6.2).

Figure 4, the learning curves of one BC draw against the actor bootstrap every 25k (section
4.2, luis 10-13), showing the tie to 150k and the separation after.

### 8.4 Analysis section

What the actor does (section 4.6), stated with the five refuted readings and the sign-flip
result once available. Where the step points (section 4.5). The failure stages and segment-swap
results as the reason the critics cannot be ranked on local decisions.

### 8.5 Limitations to state

One dataset carries every positive result; antmaze is at BC; the measure does not depend on
the action latent in the trajectory-mass sense; the lp inference of the paper's Lagrangian is
inert under the softmax loss and scores below the h(g) readouts; the point-beats-shaped
reversal is unexplained; three seeds for most arms.

### 8.6 Appendix material

The full arm table (section 4.3), the configuration tables (section 9.2), the readout
definitions (section 9.1), the parity checks across machines (section 4 preamble), and the
regeneration commands (section 9.4).

---

## 9. Appendix

### 9.1 Readouts, the evaluation-time overrides on one checkpoint

`tools/eval_checkpoint.py` reads the run's own `flags.json` and lets only typed `agent.*`
overrides sit on top. Each readout below is such an override.

| tag | override | what it does at each step |
|---|---|---|
| `hgoal_each` | `coef_source=hgoal_each acting=gpi` | the scoring readout: 64 clipped prior draws, each scored by the mean over 32 rewarding goals of the softmax share of that goal among [goal + 224 reference states] at w = h(goal); argmax; decode |
| `hgoal` | `coef_source=amortized acting=gpi` | one w = the projected mean of h over the 32 goals; score = total softmax mass on the goal set over [goals + references]; argmax over 64 |
| `lp` | `coef_source=lp acting=gpi` | 20,000 Adam steps on a free w on the sqrt(D) sphere maximising the mean goal-set share over 1024 (s,u) data rows; under the softmax loss the multiplier and constraint are inert; 25 to 60 min per evaluation; below the h(g) readouts on every run |
| `trained` | `coef_source=trained acting=gpi` | code-indexed agents only: w = the projected mean of w(z) over 256 codes |
| `regression` | `coef_source=regression` | least squares of reward on goal-averaged phi over the relabel rows, projected to the sphere |
| `actor_rel` | `acting=distill eval_redistill=false eval_goal_source=relabel` | the actor acting alone, fed the first of the 32 rewarding states as its goal for the whole evaluation; flowbc: one prior draw per step plus delta |
| `actor_env` | same with `eval_goal_source=env` | the actor fed the environment's task goal |
| `flowbc_gpi` | `acting=flowbc_gpi coef_source=hgoal_each` | the 64 candidates are actor outputs, candidate k fed goal k mod 32; scored as hgoal_each |
| `reward_gpi` | `coef_source=reward_gpi` (`tools/eval_state_reward.py`) | 32 candidate coefficients h(c) from the 16 highest-reward and 16 random relabel states; score = max over c of the reward-weighted softmax over 224 references; argmax over 64 |
| `hgoal_topk` | `tools/eval_state_reward.py` | hgoal_each with the 32 highest-reward states as the goal set |
| `hgoal_each_hold{4,8}` | hgoal_each with `gpi_hold` | the chosen latent reused for 4 or 8 steps, decoded at the current state each step |
| `bc` | gpi with `gpi_num_u=1` | the frozen flow acting alone |

The evaluation protocol: 500 episodes per task, five tasks, the mean across tasks, mean and
95 percent interval across seeds, the BC control beside every number
(`scripts/eval500.sh psmgoal cube <run_dir> <out>`, `scripts/slurm/eval500_packed.sbatch` for
one job per (run, step, readout), `scripts/campaign_2026-10/agg_results.py` to aggregate).
Evaluation needs `MODE=psmgoal` and `PSM_REPO` exported. `EVAL_WORKERS=4` on every machine.

### 9.2 Configuration

Keys arm A sets:

| key | value | meaning |
|---|---|---|
| policy_index | goal | w_i = h(g_i) on the transition's measure goal |
| bootstrap_source | actor | u' from the actor at (s'_i, g_i) |
| train_actor | true | the actor is stepped every update |
| actor_input | goal | the actor takes the raw goal state |
| goal_random_frac | 0.0 | no random-state goals (yaml default 0.3) |
| goal_cur_frac | 0.0 | no own-next-state goals |
| goal_sampling | geometric | offset d ~ Geometric(1 - goal_discount) |
| measure_loss | softmax | the cross-entropy loss of section 2.4 (yaml default squared) |
| acting | distill | in-loop evaluation only; scoring uses gpi |
| eval_redistill | false | no evaluation-time distillation |
| eval_goal_source | relabel | the actor readout's goal is a rewarding dataset state |
| actor_kind | flowbc | u = clip(eps + delta) (yaml default tanh) |
| fb_bc_coeff | 3.0 | weight on mean delta^2 |
| actor_value_kind | point | the share of the actor goal (yaml default shaped) |
| norm_future_k | 0; nf2: 2 | own later states added to the candidate set |
| bootstrap_delta_scale | 1.0; sign flip: -1.0 | factor on the actor's step in the target only |

Defaults left untouched that matter:

| key | default | meaning |
|---|---|---|
| batch_size | 256 | N; the mesh is N x (N + N K) |
| z_dim | 128 | D, the width of phi and h |
| discount | 0.98 | gamma in the target |
| goal_discount | 0.98 | the geometric horizon of goals and future states |
| measure_temp | 1.0 | t in the loss, the actor value and every readout |
| tau | 0.01 | Polyak rate of both targets |
| lr_measure / lr_goal / lr_actor | 1e-4 / 1e-4 / 3e-4 | |
| measure | hidden 1024, 2 layers | |
| w_star | hidden 256, 2 layers | h(g) |
| actor | hidden 512, 2 layers | delta |
| u_clip | 3.0 | clip on every latent |
| q_coeff | 1.0 | weight on the normalised value term |
| actor_goal_source | future | the actor trains at a uniform later state |
| bootstrap_switch | none | |
| k_goals | 32 | goal set size at evaluation |
| reference set size | 224 | batch_size minus k_goals |
| gpi_num_u | 64 | prior draws per step |
| gpi_hold | 1 | a new argmax every step |
| eval_relabel_size / eval_reward_shift | 10000 / 1.0 | |
| infer_batch, num_inference_steps, lr_infer, lr_l, use_dgd, inf_coeff | 1024, 20000, 3e-4, 3e-4, true, 5.0 | lp readout only |
| gpi_decode / flow_decode_steps | onestep / 10 | |
| use_point_preimage | true | u_i is the stored point preimage |
| ortho_coef, train_goal_head, actor_value, train_basis | 0.0, false, none, true | all required for this arm |

Training launch, one seed per GPU (`scripts/slurm/train_psmflow.sbatch`): STEPS 500000,
EVAL_INT 50000, EVAL_EPS 50, LOG_INT 5000, SAVE_INT 250000, ENVKEY cube, the flow at
`$PSM_DATA/flow/cube-single-play` epoch 500000, preimages
`$PSM_DATA/preimages/cube-single-play.npz`, 24-hour limit for A (about 3 hours actual) and
nf2 (16 hours actual).

### 9.3 Where things are

Run groups under `$PSM_DATA/exp/PSMFLows/` on kisski: `psmgoal_ja_fbc3_pt_cube` (A, seeds
0 to 4; seed 0 is `sd000_s_2522773.0.20261006_111542`), `psmgoal_ja_fbcBC_pt_cube` (B, seed 0
`sd000_s_2523422.0.20261008_111851`), `psmgoal_ja_fbc3_pt_nf2_cube` (seed 0
`sd000_s_2525384.0.20261015_165409` with checkpoints every 50k; seeds 1 and 2 with 250k only),
`psmgoal_ja_fbc3_pt_bdsneg_cube` (running), and `psmgoal_ja_{fbc3_sh,fbc03_sh,fbc0_sh,dsrl_sh,
p2a_pt,a2p_pt,actmg_pt,ogmix_pt,rand30_pt,nostitch_pt,randdrop_pt}_cube`,
`psmgoal_{db,dbu,dbo}_{sq,sm}_{cube,antmaze}`, `psmgoal_gc_*`, `psmgoal_sm_code_cube`,
`psmgoal_lift_cube_gc`. The luis groups (bon*, bcmean8, fbc03_pt, fbc1_pt, fbc10_pt, jd_*,
jao_*) and the aws groups (bds05/2/4) live on those machines; their numbers are in the mailbox
posts and the 10-13 and 10-16 messages. A's 10-07 checkpoints are also on HF as
`runs/psmgoal_ckpts_2026-10-07.tar.gz`.

Evaluation JSONs: `$PSM_DATA/logs/<group>_<sd>_<step>_<readout>_task<t>.json`.

Diagnostic JSONs under `$PSM_DATA/logs/`: `actor_mechanism/actor_mech_sd000_500k.json`;
`measure_fullness/fullness_fbc3pt_vs_fbcBC_sd000_500k.json`,
`fullness_nf2_sd000_{250000,500000}.json` (HF `runs/measure_fullness/fullness_{A_actor,A_bc,nf2_actor,nf2_bc,nf2_bon64}_sd000_500k_aws.json`
for the 10-17 rollout-policy rerun), `trajectory_mass_sd000_500k.json`,
`trajectory_mass_nf2_sd000_{250000,500000}.json`, `trajectory_mass_nf2_sd000_50k_cpu_Aonly.json`;
`actor_shift/shift_fbc3pt_sd000_500k.json`; `time_to_success/tts_fbc3pt_sd000_500k.json`,
`luis_t1_survival.json`; `diag_actor_ranking.json`; `diag_measure_pair_ranking.json`;
`diag_pick_modes_fbc3pt_sd000_500k.json`. luis's T1, T2 and one-action files are on HF under
`runs/rollout_value_check/`. aws's state-reward and bds files are on HF under
`runs/state_reward/` and `runs/psmgoal_ja_fbc3_pt_bds*_cube/eval_aws/`.

Documents: `docs/design/2026-09-16-goal-indexed-measure.md`, `2026-09-17-psmgoal.md`,
`docs/audits/2026-09-17-psmgoal-rlu-faithfulness.md`, `docs/design/2026-09-21-psmgoal-2p-dsrl.md`,
`2026-09-21-psmgoal-reward-relabel.md`, `2026-09-22-psmgoal-fixes.md`, `2026-09-30-psmgoal-tests.md`,
`2026-10-01-psmgoal-goal-conditioned.md`, `2026-10-03-psmgoal-data-bootstrap.md`,
`docs/results/2026-10-07-psmgoal-softmax.md`, `docs/plans/2026-10-15-norm-future.md`,
`2026-10-15-aws-setup.md`, `2026-10-17-sign-flip-bootstrap.md`, `docs/HANDOFF.md` entries
2026-09-17 through 2026-10-15, `docs/campaign_2026-10/*_EXPECTED.txt`,
`scripts/campaign_2026-10/`, and the mailbox folders `mailbox/{kisski,luis,aws,tasks}` on HF.

Tools: `tools/diag_actor_mechanism.py`, `diag_measure_fullness.py`,
`diag_measure_trajectory_mass.py`, `diag_actor_shift_direction.py`, `diag_time_to_success.py`,
`diag_failure_stages.py`, `diag_segment_swap.py`, `diag_rollout_value_check.py`,
`diag_actor_ranking.py`, `diag_measure_pair_ranking.py`, `diag_pick_modes.py`,
`diag_psmgoal_rollout_ranking.py`, `eval_state_reward.py`, `diag_psmgoal_gram.py`,
`diag_first_u_effect_bc.py`, `diag_action_from_future.py`, `diag_fourrooms_induced_reward.py`,
with sbatch launchers under `scripts/slurm/`.

### 9.4 Regenerating every number

```bash
# train arm A (one seed per GPU; nf2 adds agent.norm_future_k=2; sign flip adds agent.bootstrap_delta_scale=-1.0)
REPO=/mnt/home/amohan/git/Austin/PSMFLows; PSM_DATA=/mnt/home/amohan/psm-data
OVERRIDES="agent.policy_index=goal agent.bootstrap_source=actor agent.train_actor=true agent.actor_input=goal agent.goal_random_frac=0.0 agent.goal_cur_frac=0.0 agent.goal_sampling=geometric agent.measure_loss=softmax agent.acting=distill agent.eval_redistill=false agent.eval_goal_source=relabel agent.actor_kind=flowbc agent.fb_bc_coeff=3.0 agent.actor_value_kind=point"
COMMON="PSM_REPO=$REPO,PSM_DATA=$PSM_DATA,OGBENCH_DATASET_DIR=/mnt/home/amohan/.ogbench/data,ENVKEY=cube,PREIMAGES=$PSM_DATA/preimages/cube-single-play.npz,STEPS=500000,EVAL_INT=50000,EVAL_EPS=50,LOG_INT=5000,SAVE_INT=250000"
for s in 0 1 2; do sbatch --parsable --partition=kisski-inference --account=general --time=24:00:00 --cpus-per-task=4 \
  --job-name=<GROUP>_sd$s --output=$PSM_DATA/logs/slurm/%x-%j.out --error=$PSM_DATA/logs/slurm/%x-%j.err \
  --export="ALL,$COMMON,SEED=$s,GROUP=<GROUP>,EXTRA=$OVERRIDES" $REPO/scripts/slurm/train_psmflow.sbatch; done

# five-task 500-episode evaluation of every checkpoint of an arm, automatically
ARMS="<GROUP>" STEPS="250000 500000" READOUTS="hgoal_each" nohup bash scripts/campaign_2026-10/arm_eval_watcher_cube.sh > $PSM_DATA/logs/slurm/watcher_<GROUP>.log 2>&1 &
.venv/bin/python scripts/campaign_2026-10/agg_results.py          # every group, step, readout, per seed

# the diagnostics (one GPU each; RUN_DIR = a run directory with flags.json and params_<EPOCH>.pkl)
sbatch --partition=kisski-inference --account=general --export=ALL,RUN_DIR=<A>,RUN_DIR_B=<B>,EPOCH=500000,TAG=<tag> scripts/slurm/diag_actor_mechanism.sbatch
sbatch ... scripts/slurm/diag_measure_fullness.sbatch              # fullness; set +episodes for more probes
sbatch ... scripts/slurm/diag_measure_trajectory_mass.sbatch        # trajectory mass
sbatch ... --export=ALL,RUN_DIR=<A>,EPOCH=500000,TAG=<tag> scripts/slurm/diag_actor_shift_direction.sbatch
sbatch ... --export=ALL,RUN_DIR=<A>,RUN_DIR_B=<B>,EPOCH=500000,READOUTS="bon64 bon8 bc actor",EPISODES=100,TAG=<tag> scripts/slurm/diag_time_to_success.sbatch
# state-reward comparison: tools/eval_state_reward.py (see scripts/slurm/eval_state_reward.sbatch)

# unit tests (per file; the whole suite in one process aborts in XLA)
.venv/bin/python -m pytest tests/test_psmgoal_norm_future.py tests/test_psmgoal_bootstrap_delta_scale.py tests/test_diag_actor_shift_direction.py -x -q
```

### 9.5 Discrepancies in the record, so nobody trips on them

- "lp best 0.118" in `docs/COMPENDIUM.md` and CLAUDE.md is the 2026-09-17 spec agent's
  `psmgoal_cube_con` at 250k. The promoted RLU-port agent's lp readout is 0.001.
- Arm A is 0.804 at 500k over the first 3 seeds (`docs/results/2026-10-07-psmgoal-softmax.md`)
  and 0.805 over 5 seeds (aggregator, after seeds 3 and 4 were added on 10-08). Both appear in
  the record; 0.805 with 5 seeds is the number to quote.
- `psmgoal_db_sm_antmaze` at 500k is 0.088 in the results document and 0.118 (2 seeds) in the
  10-07 HANDOFF entry; the results document is the later source.
- The 2026-09-21 induced-reward critic result (0.876) kept the true episode-end mask; cube
  without the mask is untested.
- The 10-15 HANDOFF entry says nf2 was "launched on aws"; training moved to kisski the same
  afternoon when aws measured 0.62 s per step.
- The configuration yaml and the module docstring describe the squared RLU loss and the
  Lagrangian; arm A runs the softmax branch, under which the constraint and multiplier are
  inert. The dgd penalty in code is cons x sg(l), not l x min(M, 0) as the yaml comment writes.
- The time-to-success file's best-of-64 pooled 0.696 on A seed 0 (100 episodes per task,
  horizon 200) is below the 500-episode evaluation of the same checkpoint (0.756), and its
  best-of-8 is 0.894; no explanation is on record.
- There is no HANDOFF entry between 2026-09-23 and 2026-10-05, and none after 2026-10-15 as of
  this writing; the nf2 result and the sign-flip launch are recorded in the plan files, the
  aggregator output, the JSONs and this document.

# Port audit: psmgoal + Density FB recipe (P) against density_fb_latent (L) and Factored-FB (R)

Date 2026-11-09. Read-only. Working tree of `feat/inversion-integration` (uncommitted code is what was read).
P = `agents/psmgoal.py` with the `DFB_RECIPE` keys, L = `agents/density_fb_latent.py`,
R = Factored-FB `impls/critics/density_fb.py` @ 571be6d, FA = `impls/actors/flowbc.py`.
Scripts and raw output in this folder: `actor_grad_check.py` / `.json` / `.log`,
`task_dependence_check.py` / `.json` / `.log`.

## Conclusion

I read every row of the "Ported decisions" table in P's executed path (`policy_index=reward`,
`actor_kind=flowbc_vf`) and compared it line by line with L and R. Each row matches L, or differs only
through a documented deviation. I found no stop_gradient in the wrong place. The actor's Q is evaluated at
the actor's own u. It uses the critic after this update's step, with the critic params stop-gradded. The
sign, the q-centring and the bc/q coefficients are the same as in L.

The actor path does carry a Q gradient. On a trained checkpoint, P's Q-term gradient with respect to u is
larger relative to the BC term than L's is (table below). The actor moves off its BC rollout about as much
as L's does. So the actor is not held at BC by a missing or zero gradient. The measured differences are in
what that gradient points at:

- The index feature f in P has about 2 to 3 effective directions of variation. L's slow B tower has
  about 21 to 30.
- P's index reward view matches the hindsight kernel reward with correlation 0.48. L's matches it with
  0.80.
- In P's logits over next states, the index-free b term varies about 3 times as much as the phi^T w term.

All three follow from documented deviations 1 and 2, not from a coding error. The doc does not state
their size anywhere.

## Per-decision verdicts

Line numbers are for the working tree. "MATCH" means the same computation as L, up to the documented
deviations named in the row.

| # | decision | verdict | P evidence | L evidence | R evidence |
|---|---|---|---|---|---|
| 1 | training index: centred ridge of the hindsight goal's kernel reward over the batch's next states | MATCH (procedure); feature = DOCUMENTED dev 2 | `reward_index_rows` 1811-1839; `fb_sample_tasks` 1731-1745; `kernel_reward` 87-93 (same function L imports) | `_z_at` 233-244; `_draw_injection` 259-260 | 777-801, 1244-1268 |
| 2 | random sphere rows (0.5, sqrt(D) with +1e-6) | MATCH | 1739-1744 | 241-243, `sphere` 87-89 | 201-207, 791-801 |
| 3 | ridge solve (lambda 0.01 tr/D, floor 1e-6 tr/D, const fallback 1e-6, sqrt(D) projection) | MATCH (same function `ridge_task`) | 104-121, `fb_task_fit` 1723-1729 | `_ridge` 221-224 | 727-777 |
| 4 | slow copy toward the PRE-step basis at 0.0005 | MATCH | `index_state_update` 1919-1934, reads `self.basis.params` (pre-step, the agent before `apply_update`); called from `fbc_actor_step` 2024 | 440-445 | 884-888 |
| 5 | moments at 0.005 from the OLD slow copy, symmetrised | MATCH | `index_f` built at `self.fb_zenc` in 1823; EMA 1926-1933 | 446-452 | 890-900 |
| 6 | test-time index: ridge on the 10,000 relabel rewards (+1.0) | MATCH (procedure); feature = dev 2 | `infer_eval_goals` 2797-2799, `fb_eval_task` 2628-2637 | `infer_z` 549-558 | 1310-1340 |
| 7 | frozen readout pool, first update's first 256 next states | MATCH | 2021-2024; checkpoint sd000 500k: `index_pool_filled` 1.0, pool rows non-zero | 453-454 | 902-915 |
| 8 | readout Q = min_h sum_j softmax_j(M_h/1) (f(y_j)-mu)^T w / sqrt(D) | MATCH in form; M = dev 1, f = dev 2 | `reward_softmax_q` 1936-1951; `pool_softmax_score` 2510-2519 | `_softmax_value` 329-339, `q_fn` 349-362 | 1143-1196 |
| 9 | q_center: divide by sg(max(mean abs(Q - mean Q), 1e-6)); Q itself not centred | MATCH | 1982, 1993 | 360-361, 390-391 | 1194-1199; FA 89-91 |
| 10 | best-of-64 clipped prior draws, argmax of the readout at the ridge w | MATCH | `select_latent` 2486-2505 | `bon_candidates` 497-507 | (L's own seam) |
| 11 | FlowBC actor: NoiseConditionedActor(s, w, noise), u = 3 tanh, own flow fit to the clipped preimage, 10-step rollout target, lr 1e-4 / 3e-4 | MATCH | create 762-775; `fbc_actor_u` 1953-1956; `fbc_rollout` 1958-1967; `actor_loss_fbc_vf` 1969-1999 | 189-192, 211-218, 364-396, 648-669 | FA 65-96 |
| 12 | actor index: own solve at the uniform-future goal, own sphere draw | MATCH | 1836-1837 (`fb_goals`, key 108) | 258-260 | 1244-1268 |
| 13 | bootstrap u+ = FlowBC one-step actor at (s', value index), fresh N(0, I), stop-grad | MATCH | `_select_u_next` 1258-1263 (online actor params, `batch["index_w"]`) | 262-264 | 1012-1020; FA 98-121 |
| 14 | update order: critic step, actor on the stepped critic, index stats toward the pre-step basis | MATCH; Polyak timing = DOCUMENTED dev 8 | `apply_update` 2184-2219 (`fbc_actor_step(batch, basis.params)` 2217) | 398-459 | AG `apply_update` |
| 15 | loss: CE, target (1-gamma) one-hot + gamma softmax(target M), temp 1, batch next states | MATCH; head sum = DOCUMENTED dev 3 | `softmax_td_loss` 312-321; 1472-1484 (w = w_t = `index_w`, target params `self.target_basis` per head); 1540 | 287-305 | 1022-1057 |
| 16 | two heads, min at every action score | MATCH for the min; independent heads = DOCUMENTED dev 3 | 690-693; 1950-1951; 2519 | 338 | 1184 |
| 17 | goal slot reads only cube xyz of s+ | DOCUMENTED dev 4 (unscaled, L divides by 0.4) | `goal_input_blocks` 81-84, `_goal_in` 848-854 | `b_input` 92-97 | 320-325 |
| 18 | value goals 0.2 own obs / 0.5 geometric 0.99 / 0.3 random | MATCH | main.py 277-285; 1826-1834 | main.py 304-314; 109-120, 256-257 | DS 484-501 |
| 19 | widths and rates (3x512 LN, lr 3e-4, gamma 0.99, tau 0.005, z_dim 50, norm_future_k 0) | MATCH; net shape = DOCUMENTED dev 4; batch 512 vs 1024 documented | `DFB_RECIPE` 270-290; `_triple_trunk` (psm_networks.py 462-476); RLUMeasure RMSNorm head 552-575 | 644-669 | Y |

### Run flags (item 6 of the brief)

- All three `psmgoal_recipe_bc03_cube` seeds hold every `DFB_RECIPE` value except one:
  `fbc_actor.bc_coeff` is 0.3 instead of 3.0, which is what the run name asks for.
- `dfb_recipe` is false in these flags. The keys were passed one by one, and they landed.
- The batch size is 512.
- Against L's `dfb_latent_oc_bc0p3_cube` flags, the following match: lrs 3e-4 / 1e-4 / 3e-4, discount
  0.99, tau 0.005, z_dim 50, widths 3x512, layer norm, temp 1, hard one-hot, the index knobs (0.5 / 0.01 /
  1e-6 / 1e-6 / 5e-4 / 5e-3 / 256), the hit spec, the goal mix, u_clip 3, bc 0.3, q 1, 64 one-step draws,
  and two heads with min.
- The documented differences are batch 512 vs 1024, and L's run length of 1M steps (its 500k checkpoint
  was compared).

## Numeric checks (CPU, one batch, seed 0 checkpoints at 500k)

The actor objective was evaluated at the actor's own u on one dataset batch. "Q/BC gradient ratio" is the
norm of d/du of (-q_coeff mean Q / q_scale) divided by the norm of d/du of (bc_coeff mean ||u - rollout||^2).
Both runs use bc 0.3.

| quantity | L (batch 1024) | L (batch 512) | P (batch 512) |
|---|---|---|---|
| index reward view vs hindsight kernel reward, correlation on actor rows | 0.800 | 0.793 | 0.476 |
| Q/BC gradient ratio | 0.52 | 0.64 | 1.44 |
| std of Q over 32 prior u draws at fixed (s, index), divided by q_scale | 0.040 | 0.047 | 0.182 |
| actor-to-rollout squared gap (bc_err) | 0.150 | 0.147 | 0.106 |
| cosine of per-row dQ/du: own index vs a permuted index | 0.245 | | 0.078 |
| cosine of per-row dQ/du: two random sphere indices | 0.149 | | 0.167 |

Stored index-feature moments, from the checkpoints. PR is the participation ratio of the centred
covariance, (sum eig)^2 / sum eig^2.

| run | trace(C) / squared norm of mu | PR |
|---|---|---|
| L bc0.3 500k | 0.00071 | 21.5 |
| L bc3 500k | 0.00021 | 29.5 |
| P bc0.3 (3 seeds, 250k and 500k) | 0.00042 to 0.00053 | 2.2 to 3.3 |
| P bc3 recipe (3 seeds, 500k) | 0.00026 to 0.00105 | 1.04 to 1.36 |

In P's logits over the batch's next states (head 0, 64 rows), the phi^T w part has std 0.18. The b part
has std 0.51. P's training log shows `phi_eff_rank` 1.00 for the whole run and `index_proxy_corr`
0.47 to 0.54.

What this shows:
- P's actor receives a Q gradient at least as large, relative to BC, as L's.
- Its Q depends on u more strongly.
- The two cosine rows do not separate P from L, so they do not show that P's push ignores the task.
- P's index is built on a feature with 2 to 3 usable directions. Its reward view is a weaker copy of the
  task reward.

What it does not show: one batch and one seed per agent were measured, and I did not tie these numbers to
success directly.

## Undocumented differences, ranked by possible effect on the actor

1. **Size of the index feature's variation.** Read in the checkpoints for all seeds. The doc states that f is
   the projected phi on the sphere (dev 2). It does not state that f's centred covariance has 2 to 3
   effective directions where L's B has 21 to 30. The ridge index and the actor's per-state reward live in
   that space, and the reward view correlates 0.48 with the kernel reward vs 0.80 in L. This is a
   consequence of dev 2 together with phi's rank-one collapse (`phi_eff_rank` 1.00). It is not a code
   bug. It is the most likely reason the actor climbs a Q that is not the task. Confidence: high for the
   numbers (checked numerically). Medium for the causal link to success.
2. **The index-free b term dominates the measure's variation over next states.** Checked numerically, one
   batch: std 0.51 for b vs 0.18 for phi^T w. A consequence of dev 1. The softmax weights the actor shifts
   are set mostly by b, which does not see the task index. Confidence: medium.
3. **The factorised form cannot run the recipe, and differs from L beyond dev 1.** Read in code.
   - `create` asserts `measure.layer_norm` only for the joint form (588-590). It asserts `critic_heads == 1`
     for the factorised form (678). So `dfb_recipe` with `measure_form=factorized` fails at construction.
   - The factorised arm has one head and no min.
   - Its LayerNorm is on the first trunk layer only.
   - f is on the sqrt(f_dim) sphere (PhiMap with norm). L's B has no output norm.
   - A(s,u) and beta(s,u) see neither w nor the goal. The forward side is A(s,u) w + beta(s,u), affine in
     w, where L's F(s,u,z) takes z as a network input.
   - The index reads the measure's own f (`fb_feature` 1712-1713), which is closer to L's B_zenc than the
     joint arm's projected phi.
   - The only factorised runs on disk (`psmgoal_ladder_factorized_cube`) use `actor_kind=flowbc`, z_dim 128
     and batch 256, not the recipe.
   - The recipe doc (written 11-03) does not mention the factorised form.
4. **Eval diagnostic `actor_restored` checks the wrong net for flowbc_vf.** Read in code. In
   `tools/eval_checkpoint.py` 452-466, it compares the tanh `agent.actor`, not `agent.fbc_actor`. It printed
   True on the recipe actor eval, which says nothing about the FlowBC actor. The restore itself is
   whole-agent (`utils/flax_utils.py restore_agent`), and the checkpoint holds `fbc_actor` at step 500001,
   so the deployed actor is the trained one. Effect on results: none. Effect on reporting: the flag is
   meaningless for this arm.
5. **Bootstrap and actor noise streams.** Read in code. R reuses one noise draw for the actor-goal and
   value-goal bootstrap actions. L and P draw the bootstrap noise independently of the actor-loss noise.
   Only the value-goal draw reaches the target in all three, so the distributions match. No effect.

Nothing else differed. The actor index, the bootstrap index, the stepped-critic order, the sign of the Q
term, the min over heads, `q_coeff` 1 and bc 0.3 are all as in L.

## What would separate the remaining explanations

These are possible next measurements, not run here. Both reuse `actor_grad_check.py`:
- Train P with the index feature replaced by a slow copy of an L-style backward tower. That isolates dev 2.
- Measure L's proxy correlation and gradient ratio at matched steps.

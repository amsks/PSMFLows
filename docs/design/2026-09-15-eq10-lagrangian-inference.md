# 2026-09-15 — PSM Eq. 10 test-time policy inference over the affine coefficient (cube)

Source: `tools/infer_policy_lagrangian.py` (commit 431180c) plus the acting seam
`agent.acting=fixed_coeff agent.fixed_index_coeff_path=<npz>` in `agents/psmflow.py`.
Checkpoints: `$PSM_DATA/exp/PSMFLows/affine_strict_cube/sd00{0,1,2}` @500k, the paper-strict
affine agent (`psi_form=affine policy_index=latent train_actor=false acting=gpi`). Eval-only:
no training, no new checkpoint. Scalar critic for the task-2 block:
`cube_dsrlna_rhat_scaled/sd001` @500k (the one `tools/diag_measure_vs_scalar_q.py` used).

JSONs: `$PSM_DATA/logs/eq10/infer_policy_lagrangian_cube_sd00{k}.json` (+ `.npz` with the
coefficients, multipliers and the held-out Q panels), the deployable coefficients
`$PSM_DATA/logs/eq10/eq10_<variant>_sd00{k}_task{t}.npz`, and the 500-episode evals
`$PSM_DATA/logs/cube_eq10_<variant>_sd00{k}_500000_task{t}.json`. SLURM: inference
2518387-2518389 (one per seed, ~1 h each); evals listed in section 5.

## 1. Paper to ours

PSM (arXiv 2411.19418, Sec. 5.3) writes the successor measure of ANY policy as
`M = Phi w + b`, affine in an unconstrained coefficient `w`, and infers the test-time policy
by solving

    max_w  E[(Phi w + b) r]    s.t.   Phi w + b >= 0  for all (s, a)

as the Lagrangian `max_{lambda >= 0} min_w  -(Phi w) r - sum lambda(s,a) min(Phi w + b, 0)` by
gradient descent-ascent (lr 1e-4, batches of 10^4 dataset transitions), then acts on
`Q* = M* r`.

| paper | ours | code |
|---|---|---|
| `Phi(s,a)`, `b(s,a)` | `A(s,u)` in R^{z x w_dim}, `beta(s,u)` in R^z; `u` = point preimage of the recorded action | `AffinePsiMap.sa_terms` |
| policy coefficient `w` | `c` in R^{w_dim}; the trained family sits at `c = w(u')`, unit norm | `AffinePsiMap.encode_index` |
| `M_w(s,a,x)` | `m_c(s,u,x) = (A^T c + beta)^T phi(x)`, `phi` on the sphere of radius sqrt(z) | `PhiMap` |
| `M_w r` | `Q_c(s,u) = (A^T c + beta)^T w_task`, `w_task = infer_z` (10k rows, shift 1.0, sphere) | `infer_z` |
| objective `E[(Phi w + b) r]` | `J(c) = mean_i Q_c(s_i, u_i)` over N = 10,000 dataset transitions, ensemble mean over P = 2 | `objective_linear` |
| constraint on `(s,a)` | `m_c(i, j) >= 0` on pairs (row i, next state `s'_j` of another row), the measure loss's negatives | `constraint_rows` |
| `lambda(s,a)` | `row`: one multiplier per dataset row, `v_i = mean_j relu(-m_c(i,j))`; `scalar`: one multiplier on the pair mean | `lagrangian_fit` |
| DDPG argmax on `Q*` | argmax over K = 64 clipped prior draws `u` of `[mean_P - 0.5 * unc] Q_c(s,u)`, decoded through the frozen flow | `fixed_coeff_select` |

Differences from the paper that are decisions, not translations:

- `c` is initialised at the best family member: among 4,096 clipped prior draws `u'`, the
  `w(u')` with the largest batch-mean `Q` (GPI's inner max averaged over states). The paper
  starts from scratch; starting inside the family is what makes "did `c*` leave the family"
  a measurement rather than a foregone conclusion.
- The objective is linear in `c`, so its gradient is exact from the batch means `a_bar =
  mean_{P,i} A_i w`, `b_bar = mean_{P,i} beta_i^T w`; only the constraint is stochastic
  (1,024 rows x 256 columns re-sampled per step).
- Both terms are divided by `S` = the batch std of the objective term at `c0`. Adam makes the
  step on `c` scale-free anyway (about `lr` per coordinate per step); `S` matters for how
  fast `lambda` grows, `lambda_i += lam_lr * v_i / S`.
- The ensemble enters the objective and the constraint as the mean; the pessimistic readout
  (mean - 0.5 x disagreement) is what acting uses and is reported beside the mean.
- `sphere` renormalises `c` to the encoder's unit sphere after each step: the paper has no
  such constraint; the variant asks whether a better member of the trained family is enough.

## 2. Procedure

Per checkpoint (3 seeds), one fixed draw of rows: N = 10,000 train rows and H = 2,000
held-out rows from the preimage-augmented dataset (`ROW_SEED`), the same rows for every
task and variant. `A_i`, `beta_i`, `phi(s'_i)` precomputed once. Per task t in 1..5,
`w_task` is inferred exactly as `tools/eval_checkpoint.py` does (global seed 0, one `ex`
draw, then 10k relabel rows). Six variants, 5,000 steps each, Adam on `c`, projected
gradient ascent on `lambda`:

| variant | c | lambda | lr (c) | lr (lambda) |
|---|---|---|---|---|
| `free` (deployed) | free | per row | 1e-3 | 1e-3 |
| `free_lr1e-4` | free | per row | 1e-4 | 1e-3 |
| `free_scalar` | free | scalar | 1e-3 | 1e-3 |
| `free_fastdual` | free | per row | 1e-3 | 1e-1 |
| `sphere` | unit sphere | per row | 1e-3 | 1e-3 |
| `sphere_lr1e-4` | unit sphere | per row | 1e-4 | 1e-3 |

Every reported number is on the held-out rows: objective at `c0` and `c*`; constraint
violation fraction and mean magnitude over the H x H held-out pairs; `||c*||`, cos to `c0`,
to the nearest of the 4,096 family coefficients and to their mean; `Q_{c*}` against
`Q_GPI = max over 64 u' of [mean - 0.5 unc] psi^T w` on the 2,000 held-out states x 64
latents `u` (pooled and per-state Spearman, argmax agreement, `Q_GPI`'s regret at the
`Q_{c*}` argmax normalised by the per-state range; a random pick scores (max - mean)/range).
Task 2 adds the scalar critic `Q_s(s,u) = min_e qa(s, G(s,u))` on the same panel.

Pre-registered reading (written before the numbers):

- `c*` stays inside the family patch (cos to nearest member > 0.95, `||c*||` ~ 1): the
  program has nothing to add over GPI; expect five-task ~0.25 (the same-checkpoint GPI
  mean, see section 5) and ~0.28 for the 300k-500k late mean.
- `c*` leaves the family and five-task > 0.35: the basis spans a good policy the family
  does not contain; GPI's ceiling is the family, not the basis.
- `c*` leaves the family and success collapses toward or below the BC control (0.111):
  extrapolation; the basis is only trustworthy on the family it was fitted over and has to
  be retrained over a diverse family.

## 3. Per-task diagnostics

TABLES_PLACEHOLDER

## 4. Task-2 scalar-critic comparison

SCALAR_PLACEHOLDER

## 5. Five-task 500-episode success

FIVETASK_PLACEHOLDER

## 6. Reading

READING_PLACEHOLDER

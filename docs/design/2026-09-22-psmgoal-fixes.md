# psmgoal fixes: orthogonality, two-head pessimism, latent ball, greedy backup

Date: 2026-09-22. Status: pre-check done -> spec -> implement (flags off by default) -> run.

psmgoal is the main PSMFlows agent since 2026-09-22 (commit bca4b8e). This doc specifies four
flags, each off by default, so the default agent is unchanged until a run says otherwise.
Notation: `u` action latent, `u'` policy index, `s+` future state, `g` goal,
`M(s,u,s+) = phi(s,u,s+)^T w + b(s,u,s+)`, `w*(g) = h(g)/||h(g)||`, `l` multiplier.

## 1. Pre-check: psmgoal's phi has collapsed to rank one

Tool `tools/diag_psmgoal_gram.py`, run `psmgoal_lift_cube_gc` (the 0.301 run), reports
`$PSM_DATA/logs/diag_psmgoal_gram_lift_cube_gc_sd00{0,1,2}_750k.json` (+ sd000 at 250k, 1M).
Mesh Gram = mean over a 256 x 256 training-style mesh of `phi phi^T` (D = 128; an
orthonormal basis has effective rank 128 and every eigenvalue 1).

| seed @ step | effective rank | top eigenvalue share | eigenvalues 2-3 | cos(lp direction, least-squares w) |
|---|---|---|---|---|
| sd0 @750k | 1.02 | 0.990 | 0.55, 0.50 | 0.0001 |
| sd1 @750k | 1.01 | 0.993 | 0.50, 0.31 | 0.0000 |
| sd2 @750k | 1.01 | 0.993 | 0.50, 0.23 | 0.0000 |
| sd0 @250k | 1.02 | 0.990 | — | 0.0001 |
| sd0 @1M | 1.02 | 0.991 | — | 0.0000 |

What this shows:
- phi points in one direction for every (s, u, s+). The RMSNorm head fixes the trace at D, and
  99% of it sits in one eigenvalue. The collapse is present by 250k and does not recover.
- With a rank-one phi, `phi^T w` depends on the policy index only through one scalar. The
  measure cannot tell policies apart beyond that scalar; `b` carries the rest.
- The Lagrangian (`coef_source=lp`) objective is linear in `w`, so it returns the direction of
  its gradient, which is the collapsed direction: cosine 0.0000-0.0001 with the least-squares
  `w`. Whitening that gradient by the feature Gram recovers the least-squares `w`
  (cosine 0.997-0.999). `coef_source=regression` works (0.301 vs lp 0.118) because least squares
  reads the small residual eigen-directions (0.2-0.5) that still carry reward.

What it does not show: why phi collapses. Hypothesis, unverified: the `-(1-gamma)` diagonal pull
raises `M` on the diagonal, and under the bounded RMSNorm head the cheapest way is to align phi
with `w` everywhere.

## 2. The four flags

### F1 `ortho_coef` (float, default 0.0) — fixes the collapse

    L_ortho = || (1/N^2) sum_ij phi_ij phi_ij^T - I ||_F^2,   phi_ij = phi(s_i, u_i, s+_j)

on the mesh `measure_loss` already computes (no extra forward pass), added as
`ortho_coef * L_ortho`. Compatible with the RMSNorm head: it already fixes the trace at about D,
so the loss only spreads the eigenvalues. Logged: `ortho_loss`, and every 50k the mesh effective
rank.

Expected: effective rank well above 1 (target >= 32); cos(lp, least squares) > 0.9;
`coef_source=lp` five-task close to `regression`. Expected failure: the TD fit and the ortho
term fight (psmflow needed `ortho_coef` 1000 with `lr_phi` 1e-5); sweep 1, 10, 100.

F1 comes first. A greedy backup (F4) on a rank-one basis would propagate one scalar.

### F2 `num_heads` (int, default 1) — pessimism against argmax overestimation

Two `RLUMeasure` heads with separate parameters and targets. Both heads' TD targets use the
minimum of the two target heads' `M`. Every argmax (GPI acting, F4's backup) scores
`min over heads of M`. Reason: best-of-64 selection picks overestimated latents; even the FQL
expert's own critic scored 0.032 under best-of-512 (COMPENDIUM §4.6).

### F3 `u_ball` (float or null, default null) — keep scored latents where the measure trained

Project every latent the measure scores (GPI candidates, F4 backup candidates, any actor output)
onto `||u|| <= u_ball`. Measured on cube (1M point preimages): the 99th percentile of
`||u||` is 4.07 (prior N(0,I): 3.89), while the `u_clip=3` box allows `||u||` up to 6.7 and
the collapsed actor arms sat at 5.5-5.9. Setting: `u_ball=4.1`. GPI candidates are already
mostly inside it, so F3 matters mainly for actors and for F4.

### F4 `bootstrap` (proto | greedy, default proto) — iterate improvement

`greedy`: the TD target at `(s'_i, s+_j)` uses the in-support greedy latent

    u+_ij = argmax over k=1..K of  min-heads M_target(s'_i, u_k, s+_j),   u_k = K prior draws
                                                                          (clipped, in u_ball)

so column j's target is the value of the greedy policy for reaching `s+_j`, improved at every
step. Cost: K times the target mesh (K = 8, N = 256: 8 x 256 x 256 measure calls per step).

Modelling choice, for the user: under `greedy` the target no longer depends on the proto
policy of `w(u')`, so the policy index stops indexing distinct policies and `M` becomes the
in-support optimal goal-reaching measure. The alternative that keeps the index: greedy only
for rows indexed by `w*(g)` and proto for the rest. This doc implements the first.

Prior evidence: psmflow's in-support max backup (EMaQ, task-vector index) reached 0.282 at 1M,
still rising; psmgoal's 09-17 EMaQ version reached 0.118 but used the amortized `w*(g)`, since
shown to be near-constant.

## 3. Run plan (after implementation and tests)

cube-single-play, 500k steps, 5 seeds (the 3-seed intervals were +-0.2 to +-0.38), five-task
500-episode evals at 250k and 500k, each checkpoint scored with `coef_source=lp` and
`regression`. Arms: baseline, F1 (coef sweep 1/10/100 on 2 seeds first, then 5 seeds at the
chosen value), F1+F2+F3, F1+F2+F3+F4. Controls: BC 0.111; psmgoal regression 0.301.

## 4. F1 result (2026-09-23): the collapse is removed, acting gets worse

Runs `psmgoal_f1_ortho{1,10,100}_cube` sd0/sd1 (jobs 2519668-73), 500k steps, every other key as
`psmgoal_lift_cube_gc`. Five-task, 500 episodes per task; per-seed five-task means.
Matched control = `psmgoal_lift_cube_gc` (ortho_coef 0) at the same steps, `coef_source=regression`.

| arm | step | lp | regression |
|---|---|---|---|
| ortho 1 | 250k / 500k | 0.011 / 0.036 | 0.033 / 0.047 |
| ortho 10 | 250k / 500k | 0.071 / 0.061 | 0.058 / 0.058 |
| ortho 100 | 250k / 500k | 0.055 / 0.061 | 0.062 / 0.070 |
| control (no ortho), 3 seeds | 250k / 500k | — | 0.079 (0.095/0.066/0.076) / 0.176 (0.103/0.160/0.264) |
| control @750k (the 0.301) | 750k | — | 0.301 (0.231/0.395/0.276) |
| BC | — | 0.111 | |

Gram (`$PSM_DATA/logs/diag_psmgoal_gram_f1_*.json`): mesh effective rank 127.7-127.9 at every
coefficient and step (control 1.01-1.02); cos(E[phi r], least-squares w) 0.75-0.98 (control
0.0003); cos(lp direction, least-squares w) 0.29-0.50 (control 0.0000).

Reading: F1 met its mechanism targets (rank, lp ~ regression) and missed the outcome target.
Every F1 cell is below BC and below the matched control at 500k (0.036-0.070 vs 0.176). The
rank collapse was a real defect in the inference geometry; removing it does not improve acting
and costs about 0.1 at 500k. What it does not show: whether a different placement of the
penalty (on the goal-averaged feature only, or a smaller coefficient than 1) would keep the
control's acting. Default stays `ortho_coef: 0`.

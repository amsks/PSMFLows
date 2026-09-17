# 2026-09-14 — Paper-vs-code diagnosis of PSMFlow (affine, GPI)

Branch `fix/psmflow-paper-strict`, worktree `.claude/worktrees/psmflow-fix`. Nothing on
`feat/inversion-integration` was modified or deleted.

Sources: `git show 5249267:PAPER/main.tex` (PSMFlows sections 5-9, Alg. `pretrain` l.934,
Alg. `gpi` l.967, Prop. `bilinear` l.588), `agents/psmflow.py`, `utils/psm_common.py`,
`utils/psm_networks.py`, `configs/agent/psmflow.yaml`, `docs/COMPENDIUM.md`,
`docs/HANDOFF.md` (09-09 to 09-14), `docs/design/2026-09-13-psm-interface-audit.md`.

## 1. What the paper specifies

Frozen flow `G(s,u)`, `u in R^{d_a}`. Policy `pi_{u'}(s) = G(s,u')`, one fixed noise
vector repeated at every step. Successor measure of that policy from `(s, G(s,u))`:
`m(s,u,u',x) = psi(s,u,u')^T varphi(x)`, `psi = A(s,u)^T w(u') + beta(s,u)`.

Loss (main.tex l.903), batch of B, negatives are other rows' `s'_j`:

```
L = (1/B^2) sum_ij ( psi_i^T varphi(s'_j) - gamma psi_bar(s'_i,u'_i,u'_i)^T varphi_bar(s'_j) )^2
    - (2/B) sum_i psi_i^T varphi(s'_i)
    + lambda ||(1/B) sum_i varphi(s_i) varphi(s_i)^T - I||_F^2
psi_i = psi(s_i, u_i, u'_i),  u'_i ~ N(0,I) fresh per row,  u_i ~ q_alpha(. | s_i, a_i) per batch
```

Task vector `w = mean_D[r(x) varphi(x)]`. Acting: draw K latents `u_i` and K indices `u'_j`
from `N(0,I)`, argmax over all pairs of `psi(s,u_i,u'_j)^T w`, execute `G(s,u_i)`.

## 2. What the code computes (defaults)

| item | paper | code | file |
|---|---|---|---|
| squared term | all `(i,j)` | `i != j` only, `/(B(B-1))` | `utils/psm_common.py:24` |
| positive term | `-psi_i^T varphi(s'_i)` | `-(M_ii - gamma target_ii)`; target is stop-grad, same gradient | `psm_common.py:25` |
| index `u'` | `N(0,I)` per row, same `u'` in all three slots | same, clipped to `[-3,3]` | `agents/psmflow.py:631-652` |
| dataset latent `u_i` | sampled from `q_alpha` each batch | fixed point inverse, clipped to `[-3,3]` | `configs/agent/psmflow.yaml:299`, `utils/datasets.py:176` |
| target | single network | min over 2 critics (`pessimism_penalty 0.5`) | `psm_common.py:46-52` |
| ortho | weight unspecified | 1000, `varphi` on the sphere of radius `sqrt(128)` | `yaml:7`, `psm_networks.py:49-53` |
| `w` | `mean[r varphi]` | `mean[(r+1) varphi]`, projected to the sphere | `main.py:359`, `psmflow.py:1875` |
| GPI | K x K argmax, execute `G(s,u)` by ODE | K=64, K x K argmax, execute one-step distilled decode | `psmflow.py:1643-1670`, `yaml:397` |
| latent support | chi-square ball | box `[-3,3]` | `yaml:368` |

Conclusion of the diff: the loss in the code is the paper's loss. The 2026-09-13 audit
checked it against an independent expansion to 2.5e-7. The deviations are (a) point
latent instead of posterior sample, (b) one-step decode at acting, (c) ensemble-min target,
(d) reward shift and sphere projection of `w`, (e) box instead of ball. Of these, (c) is
measured to help (cube 0.42 with it, 0.14 without), (b) is measured at +0.04 for the
paper's ODE decode, (d) and (e) are rank-preserving or affect under 3% of rows. (a) has
never been measured on cube with the affine head.

## 3. Numbers on record (500 episodes)

| cube task 2 | value |
|---|---|
| BC, frozen flow alone | 0.072 |
| psmflow affine, checkpoints 250k-500k, 3 seeds | 0.424 +- 0.076 |
| FB, raw actions, in-repo | 0.721 |
| latent actor on the real reward (task-specific bound) | 0.910 |

| antmaze-medium-navigate task 1, gamma 0.99 | value |
|---|---|
| BC | 0.072 |
| psmflow affine, checkpoints 50k-250k | 0.336 +- 0.097 |
| FB (TD-JEPA Table 1) | 0.730 |

## 4. What the gap is

The paper's policy set is "repeat one noise vector forever". Pinning any one index and
running it scores below BC (11/12 cells, `HANDOFF` 09-06). So every policy in the set is
worth about BC. GPI takes one improvement step over that set at each state and reaches
0.42. FB iterates improvement through an actor and reaches 0.72. The task-specific latent
actor on the real reward reaches 0.91, so the flow and the latent space are not the
ceiling. Every arm that iterated improvement on the inferred reward `varphi^T w` scored
below GPI (DSRL-NA 0.307, gradient actor 0.11-0.14, latent DSRL-SAC 0.18).

The Walker identity-decoder run (`HANDOFF` 09-14) is not evidence about the objective: with
an identity decoder the bootstrap action is a clipped `N(0,I)` vector outside the
`[-1,1]` action box, so that arm is off-support by construction.

## 5. The run launched here

Single change from the 0.424 baseline: `agent.use_point_preimage=false`, i.e. the paper's
Alg. `pretrain` latent `u_i ~ q_alpha(.|s_i,a_i)` (stored Gaussian posterior, one sample per
visit). Cube, 3 seeds, 500k steps, all other keys identical to the baseline `flags.json`.

Preimage file: `main.py:142-149` refuses `use_point_preimage=false` on the baseline
`cube-single-play.npz`, because that file was inverted with `prior_scale=None` (likelihood-only
target, no `p0(u)` factor), so its stored Gaussian is not the paper's `q_alpha`. The run uses
`cube-single-play-a20p6-ps0p69-ns12-N200.npz` (alpha 20.57, prior weight 0.69, 12 ODE steps),
the only cube file whose posterior includes the prior. Its point preimages differ from the
baseline file's by at most 0.035 (max abs over 1M rows), so the point arm is unchanged in
effect; its posterior means differ by up to 15.7, which is the quantity under test. The
paper's prior weight is 1.0; no cube file at 1.0 exists.

Why this one: GPI queries the critic at `(s, u)` for 64 random `u` per state. Training on
the point inverse fits the critic at exactly one `u` per state. The posterior sample fits
it on a neighbourhood. The stitch runs 2513746-49 / 2513895-98 test the same pair on
antmaze-stitch and finish during this window.

Expected outcome, stated before results: late-checkpoint mean on cube task 2 moves by
less than 0.1 either way; probability of exceeding 0.55 judged below 20%. The free-psi era
comparison was 0.239 (point) vs 0.194 (posterior); the affine head may change the sign
because the bottleneck forces the critic to use `u'`, but there is no measurement that
says it will. If the arm lands at or below 0.42, item (a) joins the settled list and the
remaining gap is the single-improvement-step structure of Section 4, not an
implementation defect.

**Result (2026-09-14, jobs 2516368/69/71, group `cube_affine_posterior_u`, 500 episodes
per cell, checkpoints 250k-500k x 3 seeds, n = 18).** Protocol note added the same day:
these cells were evaluated on the single default cube task, as was the baseline it is paired
with. The reporting rule from 2026-09-14 onward is the mean across all five tasks; the
paired direction below is retained, the absolute numbers are not to be quoted. Five-task
references: BC 0.111, affine GPI 0.284 (300k-500k), FB 0.496, HILP 0.742 (TD-JEPA Table 1).

| arm | pooled mean | 95% CI (1.96 sd/sqrt(18)) | sd | min | max |
|---|---|---|---|---|---|
| posterior latent, `use_point_preimage=false` | 0.252 | ± 0.057 | 0.123 | 0.076 | 0.428 |
| point latent, `affine_strict_cube` baseline | 0.424 | ± 0.076 | 0.164 | 0.086 | 0.704 |

Per-seed means: posterior 0.219 / 0.395 / 0.141; point 0.360 / 0.518 / 0.393. Paired
difference over the 18 (seed, checkpoint) cells: −0.172, 95% CI ± 0.079 (z) or ± 0.085
(t, df 17); 16 of 18 cells below the point arm. The move is outside the pre-registered
±0.1 band and no cell exceeded 0.55. Item (a) is settled: the posterior-sampled latent
scores below the point latent. Per-cell table and the stitch state are in
`docs/HANDOFF.md`, entry 2026-09-14.

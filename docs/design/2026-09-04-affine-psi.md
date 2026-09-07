# Affine measure head — `psi(s,u,u') = A(s,u)^T w(u') + beta(s,u)`

Date: 2026-09-04 · Branch `feat/inversion-integration` · Design + pre-registration.
Code: `utils/psm_networks.py:AffinePsiMap`, `agents/psmflow.py` (`create`, `index_spread`),
`configs/agent/psmflow.yaml:psi_form`. Tests: `tests/test_psmflow_affine.py`.

## 1. Motivation

The write-up derives the measure in three nested forms:

- **Assumption `affine`** — `m(s,u,u',x) = b(s,u,x) + Phi(s,u,x)^T w^{u'}`, with `b` and
  `Phi` *independent of the policy index* `u'`. The policy enters only through a
  finite-dimensional coordinate `w^{u'}`.
- **Assumption `factorised`** + **Prop. `bilinear`** — `Phi(s,u,x) = A(s,u) phi(x)`,
  `b(s,u,x) = beta(s,u)^T phi(x)`, hence
  `m = psi(s,u,u')^T phi(x)` with `psi(s,u,u') = A(s,u)^T w^{u'} + beta(s,u)`.
  psi is **affine in the policy coordinate**.
- **Rem. `tradeoff`** — the shipped agent "parameterises `psi(s,u,u')` as a free network,
  i.e. adopts the bilinear form but *not* the affineness of psi in `w^{u'}`… the policy
  index is absorbed into psi". The stated cost is the constrained (LP) inference over `w`;
  the stated benefit is capacity.

`psi_form=affine` restores the middle row literally. This is the last structural gap
between the write-up's §1–9 object and the code, after the 2026-09-03 audits
(`docs/design/2026-09-03-paper-code-audit.md`, `-latent-actor-audit.md`) closed the index
and actor gaps.

## 2. Equations as implemented

For a state `s`, an action latent `u`, and a policy index latent `u' ~ p0` (clipped to the
`u_clip` box, exactly as `gpi_select` draws it):

```
w(u')      = enc(u') / ||enc(u')||                 in R^{d_w}      (unit sphere)
A(s,u)     in R^{z_dim x d_w},  beta(s,u) in R^{z_dim}             (one trunk, two heads)
psi(s,u,u') = A(s,u)^T w(u') + beta(s,u)           in R^{z_dim}
```

ensembled `num_parallel` times over `(A, beta)`. `m = psi^T phi(x)` and the readout
`Q_w(s,u,u') = psi(s,u,u')^T w_task` are unchanged, so every downstream consumer —
`measure_loss`, the prior-draw bootstrap `psibar(s',u',u')`, GPI over `(u_i, u'_j)` pairs,
the latent-actor DPG term — sees the same signature and needed no call-site change.

## 3. The encoder is our design choice

Assumption `affine` **asserts that `w^{u'}` exists** and says nothing about how it depends
on `u'`; PSM proves the corresponding statement for finite MDPs over all policies, and the
version restricted to the flow family `{pi_{u'}}` is left as Conj. `span`. So the map
`u' -> w^{u'}` is not given by the paper — we choose it:

| choice | what | why |
|---|---|---|
| form | MLP with `PhiMap`'s exact shape: `Dense -> LayerNorm -> tanh -> [Dense -> relu] x (L-1) -> Dense(d_w)` | the basis map in this codebase already has this shape; reusing it means no new activation/init conventions |
| `d_w` | configurable, default 128 = `z_dim` | no reason to make the policy coordinate narrower than the feature space a priori; it is the knob to turn if the bottleneck is too tight or too loose |
| sharing | ONE encoder, shared across the `num_parallel` ensemble | `w^{u'}` is a property of the **policy**, not of a critic member. The ensemble then disagrees only through `(A, beta)`, which is what `pessimism_penalty` is meant to measure |
| normalization | **unit sphere**, `w <- w/||w||` (`norm_w=true`) | `(A, w)` is identified only up to `(cA, w/c)`; fixing `||w||` pins it. Unit (rather than `psm_norm`'s `sqrt(d_w)`) keeps psi's scale independent of `d_w`, so the head is directly comparable to the free `PsiMap` it replaces at the same `hidden_dim`. It also makes collapse legible: on a fixed-radius sphere a dead encoder shows as small *pairwise distance*, not as a shrinking norm |

`A` and `beta` come off **one trunk with two heads**, with the trunk copied from
`_PsiTower`'s `(s,a)` branch (`Dense(h) -> LayerNorm -> tanh -> Dense(h/2) -> relu ->
Dense(h) -> relu`) and no `z`-branch at all: under Assumption `affine` nothing about `u'`
may reach `A` or `beta`, and the cleanest way to guarantee that is to not give them the
input. `A`'s head is the one wide layer, `h -> z_dim * d_w` (1024 -> 16384 at the shipped
widths, 16.8M params per ensemble member).

## 4. Where it plugs in

Nothing else changes; verified by reading each site:

- `measure_loss` (`agents/psmflow.py:88,94`) calls `self.psi(obs, idx, u)` with
  `idx = _index(sampled) = sampled.u_index` under `policy_index=latent`. Both the online
  and the target read go through the affine head; the contrastive/ortho losses are untouched.
- The **backup** is `psibar(s', u', u')` — `sample_step_inputs` sets `u_next = u_index`
  under `policy_index=latent` (`psmflow.py:155-160`), so the bootstrap action is a `p0`
  decode and `backup_explore_frac` stays inert. Unchanged by the head.
- **GPI** (`gpi_select`, `psmflow.py:559-573`) scores every `(u_i, u'_j)` pair. Under the
  affine head this is exact but redundant work (`A` depends only on `u_i`, `w` only on
  `u'_j`); left as-is so the acting path is bit-for-bit the same algorithm.
- **Latent actor** (`flow_actor_loss`, `psmflow.py:226`) reads `psi(obs, _index(sampled), u_a)`
  — since the 2026-09-03 fix it uses the `u'` index rather than a hardcoded `w`, so
  `train_actor=true` under `policy_index=latent` now runs, and runs against the affine head
  without further change. The readout stays `* w`.
- **Reward inference** `infer_z` is a property of `phi`, not of psi, so closed-form `w` is
  unaffected. (The LP inference Rem. `tradeoff` mentions is *newly available in principle*
  under this head — `m` is again linear in `w^{u'}` — but is **not implemented here**.)

**Guard.** `psi_form=affine` asserts `policy_index=latent` in `create()`. With a `z_dim`
task vector in the index slot, `A(s,u)` and `beta(s,u)` would become functions of the task,
which is exactly what Assumption `affine` forbids.

## 5. Diagnostics (in-loop, `index_spread`)

Emitted every update whenever `policy_index=latent` (so the shipped default path's info
dict and cost are untouched), on 16 states x 16 prior draws:

- `psi_q_spread_rel` — relative std of `Q` over prior **action** latents `u`: what
  `gpi_select`'s inner argmax consumes.
- `psi_q_range_rel` — the same as a max-min range.
- `psi_q_index_spread_rel` — relative std of `Q` over prior **policy indices** `u'`: what
  the outer (GPI) argmax consumes, and the quantity the affine head is meant to relieve.
- `w_enc_spread` — mean pairwise `||w(u'_i) - w(u'_j)||` over 64 prior draws. With
  `norm_w`, random directions in `R^{128}` sit near `sqrt(2) ~ 1.414`; a collapsed encoder
  (psi degenerate in its policy slot) reads ~0.

The `_rel` numbers are built like `ac_q_spread_rel` and `q_dist_spread_rel` so they are
directly comparable to the **~1% band** those two have always reported.

## 6. Pre-registration (written before launch)

Arms (seeds 0,1; `cube-single-play` and `antmaze-medium-navigate`; 500k steps;
`use_point_preimage=true`, `u_clip=3.0`):

- `affine_strict` — `psi_form=affine policy_index=latent train_actor=false acting=gpi`
  (the paper's §1–9 agent with the affine head; Arm B's settings otherwise).
- `affine_actor` — `psi_form=affine policy_index=latent train_actor=true acting=actor`
  (DSRL-style latent actor on the affine substrate).

Comparators (500-episode, mean ± 95% CI across seeds):

| comparator | cube | antmaze |
|---|---|---|
| Arm B (free psi, strict) | 0.083 ± 0.191 | — |
| PSMFlow actor (free psi) | 0.230 ± 0.051 | 0.213 ± 0.140 |
| BC control | 0.072 | 0.072 |

**Hypothesis.** The affine bottleneck on the policy slot is a *regularizer*: forcing the
`u'`-dependence through a `d_w`-dimensional coordinate may give psi relief across `u'`
where the free head has consistently sat at ~1% relative spread.

- **Success** = `psi_q_index_spread_rel` >> 1% **and** `affine_strict` > Arm B (0.083).
- **Failure** = the same ~1% band and BC-level (~0.07) strict performance — in which case
  the affine form is *not* the missing piece either, and the flatness is a property of the
  measure objective rather than of psi's parameterisation.
- Expected either way: `affine_actor` >= `affine_strict`, because the actor arm is the one
  with a working improvement loop; and a **finite, non-collapsed** `w_enc_spread` (near
  `sqrt(2)` at init, dropping if the head decides the policy slot is not worth using — a
  drop toward 0 *is itself* the negative result, stated in the units of this design).

Cost note: `A`'s head makes psi ~7x larger in parameters and the GPI pair scan
correspondingly heavier; the smoke measures step time before launch, and if the projected
500k wall clock exceeds the compute budget the seeds, not the steps, get cut.

## 7. Outcome

Recorded in `docs/HANDOFF.md`, entry **2026-09-04 (evening)**. Both halves of the
pre-registration fired, on cube strict:

- `psi_q_index_spread_rel` **0.29-0.88** against the free head's ~1%, rising with training;
  `w_enc_spread` 1.15-1.31 at 100k (uncollapsed), falling to 0.41-0.45 by 250k on cube while
  the index spread keeps rising — concentration, not collapse, but the number to watch.
- 500-episode success, `affine_strict` cube @**250k**: **0.532** and **0.620** (two seeds),
  against Arm B's 0.083 ± 0.191 at 500k and a gpi-matched control of 0.054.

Caveats that belong with the number: two seeds, 250k not 500k, one environment, and only the
`acting=gpi` arm — the latent-actor arms are level with their comparators (cube 0.108,
antmaze 0.196 at 100k). antmaze strict had no eval at the time of writing.

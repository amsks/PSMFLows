# PSMFlows — Compendium

**Purpose.** Everything currently known about this project, assembled so an agent with no
prior context can pick the work up. Theory, algorithm, code seams, every number that
survived verification, and the live hypotheses. Discarded avenues appear only where knowing
they are closed prevents re-running them.

**Status as of 2026-09-01.** The method's core empirical claim is **not** supported on cube.
Three experiments (E1–E4) localized the failure precisely: the action interface is fine, the
decoder is fine, and neither of the two places where the code deviated from the writeup was
load-bearing. What does not work is **ranking latents by a learned critic**, and one further
result says the failure is partly the *deployment scheme* (per-step best-of-K argmax), not
only the critic.

Sources: `PAPER/main.tex` at commit `5249267` (the formal writeup; untracked from the
working tree, read with `git show 5249267:PAPER/main.tex`), `PAPER/ICLR/` (current draft
skeleton), `docs/HANDOFF.md` (dated session record, 1700+ lines), `docs/tables/results.md`
(generated), `docs/design/2026-07-23-psmflows-formal-writeup-design.md`, and the code.

---

## 1. The idea, in one page

Zero-shot RL from offline data wants a representation that answers *any* reward at test
time without retraining. The two incumbent families both break on offline data for the same
reason:

- **FB** (forward-backward) bootstraps with `argmax_a F(s',a,z)ᵀz` — an unconstrained
  maximiser over the whole action space. Its concentrability coefficient is `∞`.
- **PSM** (proto successor measures) bootstraps with `Unif(A)`. Its coefficient is
  `ess sup_a 1/(|A| μ(a|s'))` — finite, but blows up wherever the behaviour policy is
  narrow.

Both evaluate the learned object at state-action pairs the dataset never contained, and TD
backups amplify that error multiplicatively.

**PSMFlows' proposal.** Train a conditional behaviour flow `G_θ(s, u)` on the dataset by
flow matching, then **freeze** it. Do all RL in the flow's *latent* space `u` instead of
action space. Because every action anyone can name is `G_θ(s, u)` for some `u`, and because
`G_θ(s, ·)_# p₀ = μ̂(·|s)` (the cloned behaviour), drawing `u ~ p₀` produces exactly a
behaviour sample. So the bootstrap distribution equals the data distribution and the
concentrability coefficient is **1**.

| method | bootstrap action distribution | C(π_tr) |
|---|---|---|
| FB | `δ_{argmax_a F(s',a,z)ᵀz}` | ∞ |
| PSM | `Unif(A)` | `ess sup_a 1/(\|A\| μ(a\|s'))` |
| PSMFlows | `G_θ(s',·)_# p₀ = μ̂(·\|s')` | **1** |

That table is the paper's formal separation and its reason to exist.

---

## 2. Formal content

All references are to `git show 5249267:PAPER/main.tex`.

### 2.1 Objects

**Successor measure** (Def. `sm`, §Setting): for policy `π`,
`M^π(s, dx) = E_π[Σ_t γ^t 1{s_t ∈ dx} | s_0 = s]`, so `V_r^π(s) = ∫ r(x) M^π(s, dx)`.

**Behaviour flow** (§`sec:flow`). Conditional flow matching (Lipman et al.) on `D`:
with `u_0 ~ p₀ = N(0,I)`, `u_1 = a`, `u_t = (1-t)u_0 + t u_1`,

```
L_flow(θ) = E_{(s,a)~D, u_0~p₀, t~U[0,1]} ‖ v_θ(s, t, u_t) − (a − u_0) ‖²
```

**Decoder / encoder** (Def. `decoder`): `G_θ(s, u)` integrates `du_t/dt = v_θ(s,t,u_t)` from
`u_0 = u` to `t=1`; `E_θ(s,a)` is the backward ODE, its inverse.

**Assumption `lipschitz`** (regularity of `v_θ`) gives, via **Lemma `diffeo`**, that
`G_θ(s,·)` is a diffeomorphism with the Liouville formula for `|det J|`.

**Assumption `exact`**: `μ̂ = μ`, i.e. the flow clones the behaviour exactly. Every
guarantee below is stated under it. See §2.5 — this assumption is conjectured *unattainable*
in the regime the paper targets.

### 2.2 The policy family

**Def. `classes`** gives three nested classes. The one the writeup builds on is the
**fixed-index** family `Π_fix = {π_u : a = G_θ(s,u), u held fixed}` and the **latent**
class `Π_lat` of policies whose latent distribution `ν ≪ p₀`.

**Prop. `isometry`** (the divergence isometry, the writeup calls it the headline result):
behaviour-relative divergences equal Gaussian-relative divergences,
`D_f(π_ν ‖ μ) = D_f(ν ‖ p₀)`. Corollaries: **`mass`** (mass conversion, latents in the
typical set `U_δ` decode to behaviour-typical actions w.p. `1−δ`), **`conc`** (bounded
concentrability for stochastic latent policies), **`resolution`** (density-adapted
resolution).

**Def. `coherence`** + **Assumption `coherence`**: the indexing is useful only if a fixed
`u` means something consistent across states — `ε`-transport coherence. This is the
assumption the 08-05 measurement destroyed (§4.1).

### 2.3 Successor measures over the family

**Def. `Muu`**: the two-argument object `M^{u→u'}(s, dx)` — take `G_θ(s,u)` now, follow
`π_{u'}` after. **Lemma `bellman`** gives its Bellman equation and diagonal closure.

**Assumption `affine`** (affine successor-measure form) + **Assumption `factorised`** yield
**Prop. `bilinear`**: the bilinear form `ψ(s,u,u')ᵀφ(x)` is a special case.

**Prop. `sf-identity`** (successor-feature Bellman identity), **Cor. `reward-identity`**,
**Cor. `reward-inference`**: `w = E_D[r φ]` is the least-squares projection *exactly when*
`E_ρ[φφᵀ] = I`, which is why the orthonormality loss `L_ortho` is part of the
specification and not a stability regulariser. **Cor. `latent-q`**: latent Q-iteration.

### 2.4 The two results the project rests on

**Prop. `insample` (lines 807–835) — the C=1 result.**
> Under Assumption `exact`, if `s' ~ ρ` and `u' ~ p₀` **independently**, the bootstrap
> action `G_θ(s',u') ~ μ(·|s')`. The joint law of `(s',a')` at which `T` evaluates `m` is
> exactly `ρ ⊗ μ`, and the concentrability coefficient equals 1.

Two hypotheses, both load-bearing: `u' ~ p₀`, and `u'` **independent of** `s'`.

**Def. `operator`** — the TD operator:
```
(T m)(s,u,u',x) = E_{s'~P(·|s,G_θ(s,u))} [ δ_{s'}(dx)/ρ(dx) + γ m(s',u',u',x) ]
```
Note both slots carry `u'` at `s'`. **Prop. `contraction`**: `γ`-contraction in `‖·‖_∞`,
fixed point `M^{u→u'}/ρ`, and `‖m̂ − m‖_∞ ≤ ε_TD/(1−γ)`.

**Prop. `gpi` (lines 981–1001) — the GPI bound.** With `Λ_K = {u'_1..u'_K} ~ p₀` and
`û(s) ∈ argmax_{u ∈ U_δ} max_{u' ∈ Λ_K} ψ(s,u,u')ᵀw`, if
`|ψ(s,u,u')ᵀw − Q^{π_{u'}}_r(s, G_θ(s,u))| ≤ ε` then
```
Q^{π_GPI}_r(s,a) ≥ max_{u' ∈ Λ_K} Q^{π_{u'}}_r(s,a) − 2ε/(1−γ)
```
with `ε ≤ ε_flow + ε_TD/(1−γ) + ε_reg/(1−γ)`.

### 2.5 Empirical objective and algorithms, as written (§8, lines 903–916)

`u_i ~ q_α(·|s_i,a_i)` are the inverted latents; **`u' ~ p₀` is a fresh index per batch
element**; `ψ̄, φ̄` are targets.

```
L_SM = (1/B²) Σ_{i,j} ( ψ(s_i, u_i, u')ᵀφ(s'_j) − γ ψ̄(s'_i, u', u')ᵀφ̄(s'_j) )²
       − (2/B) Σ_i ψ(s_i, u_i, u')ᵀφ(s'_i)
       + λ_⊥ L_ortho(φ),      L_ortho(φ) = ‖ (1/B) Σ_i φ(s_i)φ(s_i)ᵀ − I ‖_F²
```

**Remark `load-bearing`:** the bootstrap term evaluates `ψ̄` at `s'_i` with continuation
index `u'`, whose decoded action `G_θ(s'_i, u')` is a `μ̂` sample by Prop. `insample`.
*Every action the backup ever evaluates is a flow decode.*

**Algorithm — pretraining (reward-free):**
```
Stage A.  Fit v_θ by L_flow; FREEZE θ.
Stage B.  For each (s_i,a_i) ∈ D precompute u*_i = E_θ(s_i,a_i) and J(s_i,u*_i) by
          backward ODE; fit q_α(·|s_i,a_i) via Prop. laplace (EM refinement optional). Cache.
Stage C.  repeat
            sample batch {(s_i,a_i,s'_i)}, draw u_i ~ q_α(·|s_i,a_i), draw u' ~ p₀
            update (ψ,φ) by gradient descent on L_SM; update targets ψ̄, φ̄
          until converged
Output:   basis φ, successor features ψ, frozen decoder G_θ.
```

**Algorithm — Rung 1, flow-GPI action selection (no test-time training):**
```
Require: state s, task code w, budget K, level δ
  draw u'_1..u'_K ~ p₀;  draw u_1..u_K ~ p₀
  (î, ĵ) ← argmax_{i,j}  ψ(s, u_i, u'_j)ᵀ w
  optionally refine u_î by a few gradient steps on u ↦ ψ(s,u,u'_ĵ)ᵀw, projected onto U_δ
  return G_θ(s, u_î)
```

Freezing `θ` after Stage A means the representation trains against a **stationary** policy
family — the writeup's stated advantage over FB, which chases a moving actor.

### 2.6 Dataset latents (§`sec:preimage`)

**Lemma `typicality`**: inverted latents are prior-distributed. **Prop. `laplace`**: Laplace
approximation of the preimage posterior. The target is
```
π(u) ∝ N(u; 0, I)^{prior_scale} · exp(−α ‖G_θ(s,u) − a‖²)
```
and the Laplace covariance is `(2α JᵀJ + prior_scale·I)^{-1}`. Both details were wrong in
code until 2026-08-31 (§5.3).

### 2.7 Open conjectures, from §`sec:open` (all `NOTPROVED`)

1. **`conj:perturbation`** — flow-error perturbation. `f`-divergences are not `W₂`-continuous,
   so Prop. `isometry` cannot be perturbed in that metric at all. Two honest routes:
   smoothed divergences, or a density-ratio bound `‖μ̂/μ‖_∞ ≤ 1+η`. **This blocks the honest
   version of the headline result** and the writeup names it the first theoretical priority.
2. **`conj:span`** — affine span over the flow family; what would justify Assumption `affine`
   rather than positing it.
3. **`conj:insample-bound`** — continuous-space in-sample bound depending on *state* coverage
   `C_ρ` alone with no action-space concentrability. **This is the theorem that would formally
   separate PSMFlows from FB and PSM.**
4. **`conj:compact`** — if `μ(·|s)` is supported on a Lebesgue-null set, no Lipschitz `v_θ`
   attains Assumption `exact`. Matters because narrow near-expert data is exactly that regime,
   so every guarantee is approximate, worst where data is thinnest.
5. **`conj:coherence`** — whether CFM produces `ε`-coherent decoders. "Plausibly not provable";
   a property of the trained net, not a theorem.

**Open problem (stated, unaddressed): state-distribution shift.** Cor. `mass` constrains
actions state-by-state and says nothing about where trajectories go. What Prop. `insample`
removes is the *action* half of distribution shift — the half TD amplifies multiplicatively.

**Open problem: is the successor measure doing work?** Rung 1 with `K` samples resembles
critic-weighted resampling of behaviour samples (IDQL). The representation earns its cost
only if it beats a per-task critic ranking the same `K` decoded actions. **E4a (§4.6) now
answers this negatively** — a per-task critic ranking the same K candidates scores 0.032.

---

## 3. What the code actually is

### 3.1 Correspondence

| writeup | code |
|---|---|
| `φ(x)` shared basis | `PhiMap`, `agents/psmflow.py` `self.phi` |
| `ψ(s,u,u')` measure head | `PsiMap`, `self.psi` — see the index-slot caveat below |
| `G_θ` frozen decoder | `flow_vf` (ODE) / `flow_onestep` (distilled), loaded by `_load_flow_params` (`psmflow.py:677`) |
| `u = E_θ(s,a)` | `batch['noise_preimage']`, from the Stage-B npz |
| `w = E[r φ]` | `infer_z` (`psmflow.py:454`) |
| Rung-1 GPI | `gpi_select` (`psmflow.py:489`) |

### 3.2 The two deviations the 2026-08-31 audit found

**Deviation 1 — bootstrap latent.** The writeup requires `u' ~ p₀` independent of `s'`.
The shipped code bootstraps the **actor's** latent at `s'`:
`u_next = u_clip · actor(next_obs, task_w, noise)` (`sample_step_inputs`, `psmflow.py:145`).
That is a function *of* `s'`, so the independence hypothesis fails and **Prop. `insample`'s
C=1 never applied to anything trained**.

**Deviation 2 — the ψ index slot.** The writeup's `ψ(s,u,u')` carries a *policy latent* in
the index slot. The shipped code carries the **task vector**: `ψ(s, w, u)` (`measure_loss`,
`psmflow.py:108`).

Together these mean `agents/psmflow.py` is **FB with a latent action space**, not PSM: one
measure branch, basis trained by the same contrastive term, `w` Gaussian mixed with
`φ(next_obs[perm])`, `w = E[rφ]`. It matches `agents/fb.py` `_fb_loss_fn` line for line with
`F→ψ`, `B→φ`, `a→u`. There is no codebook basis, no affine `b`, no constrained-LP inference.

This was a deliberate 2026-07-20 design call ("the flow family IS the codebook") that held
under Rung 1, where the fixed-`u` family *was* the codebook. The 08-05 redesign moved policy
identity into `w` and made the bootstrap the actor's latent, removing the codebook; nothing
re-derived the argument afterwards.

### 3.3 Both deviations are now switchable

Added 2026-08-31 (`b68a9f4`), defaults preserve shipped behaviour:

- `agent.backup_explore_frac` ∈ [0,1] (`psmflow.py:145`) — fraction of bootstrap latents
  replaced by clipped prior draws. At `1.0` the bootstrap is exactly `u' ~ p₀`.
- `agent.policy_index` ∈ {`task_vector`, `latent`} (`_index`, `psmflow.py:135`) — under
  `latent`, ψ's index slot carries `u' ~ p₀` drawn per batch element, the backup reads
  `ψ̄(s',u',u')` (both slots the same index, per Def. `operator`), and `w` enters only
  through the readout `Q = ψᵀw`. Makes `backup_explore_frac` inert by construction.
- `agent.train_actor` ∈ {true,false} — `false` drops the actor/CFM branch entirely (§8 has
  no actor); `acting=actor` without one is refused.
- `gpi_select` under `policy_index=latent` implements Alg. Rung 1 verbatim: K action latents
  × K policy indices, argmax over pairs, return `G_θ(s, u_î)`.

Tests: `tests/test_psmflow_policy_index.py` (8 tests) pin that the default path's random
draws are unchanged bit-for-bit, the slot width changes `z_dim → d_a`, the backup index
equals the online index, and a gradient step is finite with no actor.

### 3.4 Decoder seam

`decode` (`psmflow.py:476`) branches on `gpi_decode` ∈ {`onestep`, `ode`} with
`flow_decode_steps`. **Every shipped run used `onestep`.** The Stage-B inversion solves
`G_100(s,u) = a`, so the "exact" latent is exact for a decoder the agent never used. See
§4.4 — this turns out not to matter, in the opposite direction from what was expected.

### 3.5 Stage-B inversion

`agents/fql.py`: `_get_preimage_and_jacobian` (`:23`), `_get_predistribution_proposal`
(`:66`, the Laplace proposal), `compute_full_proposal_distribution` (`:99`),
`compute_full_proposal_distribution_em` (`:145`). `utils/flow_inversion.py` holds
`repair_invalid_preimages` (`:60`) and `sample_preimage_noise` (`:121`).

---

## 4. Everything measured

Environment is `cube-single-play-singletask-v0` (task 2, the headline task) unless stated.
Frozen Stage-A flow: `bcflow_cube_single_20260726_135032/sd000_20260726_135037` @ 500000.
All success numbers are 500-episode evals. Multi-seed entries are **mean ± 95% CI across
seeds (t95, n−1 dof)** — the convention `tools/make_tables.py` uses; single seeds show a
Wilson interval.

### 4.1 Headline table

| method | setting | success | seeds |
|---|---|---|---|
| FQL (per-task reference, raw actions) | per-task | 0.949 ± 0.063 | 3 |
| Latent RL, per-task, ε=0.05 @peak | per-task | 0.905 ± 0.020 | 3 |
| Latent RL, per-task, ε=0 (pure decode) | per-task | 0.142 ± 0.025 | 2 |
| **FB (zero-shot, raw actions)** | zero-shot | **0.721 ± 0.020** | 3 |
| **PSMFlow (zero-shot, latent → frozen decode)** | zero-shot | **0.236 ± 0.071** | 5 |
| PSMFlow, HP-matched to FB | zero-shot | 0.234 ± 0.534 | 2 |
| Hybrid (action critic + residual), deployed | zero-shot | 0.162 ± 0.168 | 4 |
| Hybrid, decode-only control | zero-shot | 0.226 ± 0.068 | 4 |
| Hybrid, λ-rank (K=32, no residual) | zero-shot | 0.083 ± 1.004 | 2 |
| Hybrid + FB graft, deployed | zero-shot | 0.095 ± 0.394 | 2 |
| **BC control (per-step prior)** | control | **0.068 [0.049, 0.093]** | 1 |
| PSMFlow re-eval, actor, one-step decode | zero-shot | 0.220 ± 0.037 | 5 |
| PSMFlow re-eval, actor, exact ODE-100 | zero-shot | 0.182 ± 0.019 | 5 |
| PSMFlow re-eval, gpi, one-step decode | zero-shot | 0.054 ± 0.032 | 5 |
| PSMFlow re-eval, gpi, exact ODE-100 | zero-shot | 0.044 ± 0.043 | 5 |
| **Paper-faithful Arm A (u'~p₀ bootstrap)** | zero-shot | **0.171 ± 0.113** | 3 |
| **Paper-faithful Arm B (ψ(s,u,u'), no actor, gpi)** | zero-shot | **0.083 ± 0.191** | 3 |

**The BC control is non-negotiable.** Never quote a Stage-C number without it. Cube = 0.068.

**Provenance warning.** The `0.236 ± 0.071` headline was recorded before the 2026-08-14
eval-seeding fix; those runs drew actions from OS entropy and are **not reproducible**.
Re-running sd0 under today's pinned harness gives 0.240 [0.205, 0.279] against the recorded
0.318. The re-eval rows (0.220 ± 0.037) are the post-fix measurement of the same checkpoints
and **supersede the headline row for any comparison**.

### 4.2 Data-fraction table

| method | 10% | 50% | 100% |
|---|---|---|---|
| Behaviour flow (BC control) | 0.070 [0.051, 0.096] | 0.100 [0.077, 0.129] | 0.068 [0.049, 0.093] |
| FQL (per-task) | 0.408 [0.366, 0.452] | 0.976 [0.959, 0.986] | 0.949 ± 0.063 |
| FB (zero-shot) | 0.030 [0.018, 0.049] | 0.376 [0.335, 0.419] | 0.721 ± 0.020 |
| PSMFlow (zero-shot) | 0.067 ± 0.013 | 0.052 [0.036, 0.075] | 0.236 ± 0.071 |
| Hybrid, deployed | **0.238** [0.203, 0.277] | **0.228** [0.193, 0.267] | 0.162 ± 0.168 |
| Hybrid, decode-only control | 0.072 [0.052, 0.098] | 0.070 [0.051, 0.096] | 0.226 ± 0.068 |

**The one place this project beats everything measured:** at 10% data the hybrid scores
0.238 while **FB collapses to 0.030**, below its own BC control. The residual's sign flips
with dataset size (+0.166 at 10%, +0.158 at 50%, −0.065 at 100%). One seed per fraction
cell — replicate before claiming.

### 4.3 E1 — oracle-aim (`tools/diag_oracle_aim.py`, 500 ep, K=512, ODE-100)

At each step draw K=512 clipped prior latents (identical to the deployed GPI draw), decode
all of them exactly, and execute the one closest to a frozen FQL expert's action.

| arm | success | Wilson 95% |
|---|---|---|
| **oracle_aim** | **0.934** | [0.909, 0.953] |
| oracle (ceiling control) | 0.960 | [0.939, 0.974] |
| random_latent_onestep (floor control) | 0.086 | [0.065, 0.114] |
| random_latent_ode | 0.014 | [0.007, 0.029] |

Mean min-distance to the oracle action over K=512: **0.062** (p90 0.096), against mean
`‖a‖ = 0.875`. Both controls land where the record says they should (0.949 and 0.068 are
inside their intervals), which is what makes the aim number readable.

**Verdict, pre-registered at ≥0.7:** the flow's reachable action set contains near-expert
behaviour at nearly every step. **The entire Stage-C loss is latent *selection*.** No flow
retraining, no interface widening, no ε-residual relaxation is indicated by this number.

### 4.4 E2 — ODE re-evaluation of the 5 shipped checkpoints

500 ep × 5 seeds, post-fix pinned harness, paired within seed.

| acting | one-step | ODE-100 | paired ODE − onestep |
|---|---|---|---|
| actor | 0.220 ± 0.037 | 0.182 ± 0.019 | **−0.038 ± 0.027** (4/5 seeds negative) |
| gpi | 0.054 ± 0.032 | 0.044 ± 0.043 | −0.010 ± 0.016 |

Pre-registered reading was "≥ +0.05 means the decoder mismatch is real deployed loss."
It moved **down**. The exact decoder is a small consistent *loss*, not a lever; keep
`gpi_decode=onestep` at deployment.

The E1 floor pair says the same thing far more loudly for *untrained* latents (0.086 onestep
vs 0.014 ODE, a 6× gap). Reading: the distilled one-step net does not faithfully reproduce
the behaviour distribution — it smooths it toward something like a conditional mean, and
that smoothing rescues the wild tails. A trained actor has already found the well-behaved
part of latent space, so little is left to rescue; hence 6× on random latents and 0.038 on
trained ones.

### 4.5 E3 — the paper-faithful arms (3 seeds × 500k steps each)

Quoted against the control that **acts the same way**. Using the actor-arm 0.220 for Arm B
would price the removal of the actor as if it were the index change.

| arm | success | matched control |
|---|---|---|
| **Arm A** — `backup_explore_frac=1.0`, `acting=actor` | 0.171 ± 0.113 (0.220/0.164/0.130) | point arm actor 0.220 ± 0.037 |
| **Arm B** — `ψ(s,u,u')`, no actor, `acting=gpi` | 0.083 ± 0.191 (0.006/0.160/0.084) | point arm gpi 0.054 ± 0.032 |

**Arm A is within noise of its control.** Satisfying Prop. `insample`'s hypothesis — making
every bootstrap action a genuine `p₀` decode, so C=1 actually applies for the first time —
changes nothing. A correct backup distribution does not by itself create ranking signal.

**Arm B lands at BC level.** It does not separate from 0.068, its seed spread dwarfs any
effect, and it is nowhere near the 0.45 bar or FB's 0.721. **The writeup's construction is
refuted on its own terms on cube.**

Caveat to carry: Arm B varies the ψ index *and* the actor together, as §8 does, so it does
not isolate the policy-index idea alone. What it establishes is that the algorithm as
written does not work here.

### 4.6 E4 — the ranking failure is the deployment scheme

**E4a, the known-good-ranker control.** The E1 harness with candidates scored by the frozen
FQL expert's **own critic** instead of oracle distance: **0.032** [0.020, 0.051] — *below*
the one-step random floor. Per-step Spearman vs the oracle ranking is bimodal (mean 0.285,
median 0.346, 54% of steps above 0.3, p10 −0.33): the expert's critic ranks moderately well
on most steps, but argmax over K=512 reliably lands on its most **overestimated** candidate
— picked action 0.426 from the expert when 0.105 was available. **Winner's curse at K=512.**

Pre-registered verdict: **decode-then-score is dead as a deployment scheme; even a proven
ranker fails under per-step best-of-K GPI.** This exonerates `ψᵀw` as the specific culprit —
λ-rank, FB-graft, Arm B and the FQL critic all fail identically — and kills the "critic with
a direct action pathway" fix on its own pre-registration.

**E4b, mixture checkpoints.** Ranking Spearman 0.054, Q spread 0.86% of |Q| — the same band
as the point arm (0.10 / 1.1%) and Arm B (0.079 / 0.9%). Mixture training does not create
ranking signal.

### 4.7 The critic diagnostics (D1–D3, 2026-08-10)

- **D1 — the critic cannot rank policies.** Ten cube policies spanning 0.068–0.320 measured
  success, all scored by one frozen representation: **Spearman 0.10** (permutation p=0.78).
  Predicted values span −1816 to −1832, a **0.9% spread**, while success varies 4.7×.
- **D2 — inference is fine, the basis is weak.** Closed-form `w = E[rφ]` gets R² **0.129**
  against a ridge topline of **0.127** on the same features — the estimator extracts
  everything `φ` contains. But the topline is low: the best linear read-out of `φ` explains
  ~13% of reward variance. Antmaze identical (0.135/0.134), so no cube/antmaze asymmetry.
- **D3 — Q is flat.** Over 512 prior draws at 64 states, relative Q spread is **0.011 of
  |Q|**; around the actor's own latent, 0.0038. **The actor sits at the 44th percentile of
  the prior-Q distribution** (median 38th) — below the middle. Actor gradient is
  BC-dominated **5:1** (‖∇q‖ 0.168 vs ‖∇distill‖ 0.839). Dispersion does not collapse over
  training (‖u‖² 4.66 → 4.54), so this is not a shrinking-support artifact.

**Honest reading on record since 08-10:** PSMFlow's 3.5× over BC is explained by **in-support
latent behaviour cloning plus ensemble pessimism, not by value-driven improvement.**

### 4.8 The interface, measured three ways

- **C1 (as corrected 08-14).** Against the data-matches-itself baseline: `a_FQL` sits 0.537
  from its k-NN dataset actions while real data sits 0.582 from its own (p95 1.165), and
  only 6.3% of FQL's actions exceed that p95. **FQL is as data-like as the data.** What is
  anomalous is our decode at 0.187, ~3× *tighter*. The original "FQL is off-support" framing
  was uncalibrated and is retracted.
- **W1/W3.** Coverage 0.415–0.418 across all six decode × radius settings. Retraining Stage A
  at `flow_steps=100` lifted coverage 0.42 → 0.63 but left the FQL-action distance at 0.187
  unmoved to three decimals. The flow was not the bottleneck.
- **Jacobian probe (08-30, `tools/diag_flow_jacobian.py`).** Singular spectrum of `dG/du` at
  each transition's own preimage, 2048 rows, `flow_steps=100`:

| | cube (d_a 5, mean‖a‖ 0.87) | antmaze (d_a 8, mean‖a‖ 1.99) |
|---|---|---|
| σ pooled, median | 0.083 | 0.201 |
| σ_min, median | 0.068 | 0.050 |
| condition number, median | **1.79** | 6.67 |
| free radius in u for a 0.05 action move | 0.73 | 1.00 |

  Cube is a near-**isotropic** contraction by ~0.1 with no null direction. Antmaze's smallest
  singular value is *smaller* than cube's with a *larger* free radius. **"Cube has a preimage
  set, antmaze has a point" is wrong and should not be written down.** The cube/antmaze split
  is not local injectivity; whatever explains antmaze's 0.115 coverage and ESS 7/128 is open.

### 4.9 The preimage mixture

Decode distance to `a`, 4096 rows of the cube HPO npz, same Stage-A checkpoint:

| latent | ODE-100 | one-step |
|---|---|---|
| stored point preimage | **0.00012** | 0.0886 |
| stored mixture mean | 0.1153 | 0.1495 |
| prior draw `N(0,I)` | — | 0.2855 |

**The mixture is a blurred point.** Sweeping α shows width, distance to the point inverse
and decode error all shrink together (α=20/100/500 → per-dim var 0.49/0.093/0.020, decode
error 0.165/0.045/0.009): the posterior's width **is the temperature**, not decoder
degeneracy. Samples become faithful only once the fit has collapsed onto the point inverse.
Measured on all three environments. The mixture arm carries no result and is closed.

### 4.10 Inversion ESS, before and after the 08-31 fix

At matched decode fidelity (256 rows, seed 0, `flow_steps=100`, N=200, `prior_scale=1.0`):

| env | ESS before | ESS after | decode | coverage |
|---|---|---|---|---|
| cube | 112.9 | 132.6 | 0.168 → 0.175 | 0.890 → 0.897 |
| antmaze | **8.8** | **18.8** | 0.176 → 0.202 | 0.016 → 0.064 |

Cube was never collapsed (112.9/200). The 7/200 figure that circulated is **antmaze**.
Raising α improves ESS and decode while **coverage collapses** (cube cov_k16 0.945 at α=20
down to 0.003 at α=3200) — α is a position on a frontier, not an optimum. Default retuned
20 → 50 with the full frontier table in `configs/inversion/default.yaml`.

### 4.11 The Rung-1 root cause (2026-08-05) — closed, do not re-audit

All Stage-C variants read **0.0** on pointmaze task1. Root cause is **not a bug**: the
fixed-`u` policy family has no goal-reaching member.

1. **Exhaustive reachability**: `d_a=2` so a 13×13 grid covers the entire `[-3,3]²` box, plus
   64 dataset/goal preimages, each rolled 2 full 1000-step episodes. **0 of 233 latents ever
   reach the goal.** 75% never get closer than ~23.5 in a maze ~30 across.
2. **The expert's route is latent white noise**: within-episode preimage variance / marginal
   variance = **0.99**; lag-1 autocorr 0.27, ~0 by lag 50. The BC flow factorizes behaviour
   as (state → conditional, u → quantile), and the expert's direction choice is driven by a
   goal **not in the observation**, so that variance is forced into `u` independently each
   step. **Routes exist only as latent *sequences*.** Rung 1 assumed `u` is a persistent
   policy index; Stage A/B construct it as per-step noise. This is Assumption `coherence`
   failing empirically.
3. **What the family contains**: typical `u` → orbiters (path length 164, net displacement
   1.4); saturated `u` → constant headings that cannot turn. Per-cell coverage is fine
   (angle circ_var 0.45–0.78) — the deficiency is purely **temporal**.

---

## 5. Bugs found and fixed (with what they invalidate)

### 5.1 Eval seeding (2026-08-14, P0.2)
Action noise was drawn from OS entropy regardless of `seed`, so re-evaluating the same
weights drew a different action stream every time. **Every headline number recorded before
this fix is a single unreproducible draw.** Today's harness is exactly reproducible
(two invocations match episode-for-episode).

### 5.2 The inversion target had no prior factor (2026-08-14)
Target was `exp(−α‖G−a‖)` with **no** `N(0,I)` factor, so it is flat wherever the decoder is
insensitive and the EM fit runs away (covariance eigenvalue 1→6→34→305→2281→6095 over 8
steps; 82–84% of cube rows fitted wider than the prior, worst case 3.8e4, means to |μ|=206).
Fixed via `inversion.prior_scale` (default 1.0; 0.0 reproduces legacy files).

### 5.3 Squared-norm target and Laplace covariance (2026-08-31, `313948e`)
`agents/fql.py:116,201` used an **un-squared** L2 in the target energy; the spec is
`exp(−α‖G−a‖²)`. The un-squared form has a cusp at the mode, so the Laplace approximation the
EM initializes from does not exist there, and its α-to-width scaling is `α^{-1}` not
`α^{-1/2}`. Paired with it, `fql.py:88` used `(α²JᵀJ + prior_scale·I)^{-1}` where the squared
target's curvature is `2αJᵀJ` — at α=20 the proposal was **10× too narrow**. Both fixed.
**Every α tuned against the old target is meaningless**, and every existing mixture npz and
both 08-28 HPO sweeps measure the old target.

### 5.4 `preimage_valid` never reached the loss (2026-08-31, `3ef0afe`)
`repair_invalid_preimages` neutralizes diverged rows (point → `u=0`, mixture → prior) and
records a mask that nothing downstream read, so repaired rows trained as `(s, u=0, a)` where
`G(s,0) ≠ a`. Now dropped from sampling in `Dataset.get_random_idxs`. Counts: cube 13/1M
(negligible), **antmaze 881/1M**.

### 5.5 The D3 ESS statistic was wrong twice
First it returned `num_samples` — its best value — for rows where *nothing* was usable.
Then it averaged ESS over the whole EM trace while Stage B stores only the **final** iterate.
Gate now reads `ess[:, -1]`. Two separate reversals of the same conclusion resulted.

### 5.6 XLA autotuner miscompiles the flow integration
`utils/xla_guard.py` **must** be imported before jax in every tool. Not optional.

### 5.7 Dead code worth knowing about
`utils/datasets.py` computed `next_noise_preimage` (`u_0'`) on every batch that **no agent
read** — a Rung-1 leftover that doubled per-step mixture sampling on the training path. The
`idx+1` pairing rule and its guard in `main.py` existed for that dead field. Removed
2026-08-31 (`c84f719`).

---

## 6. Hypotheses: settled and live

### Settled negative (do not re-run)
| hypothesis | verdict | evidence |
|---|---|---|
| Fixed-`u` is a policy index | **refuted** | 0/233 latents reach goal; within-episode variance ratio 0.99 (§4.11) |
| The decoder/interface is the ceiling | **refuted** | oracle-aim 0.934 (§4.3) |
| The one-step decoder mismatch costs deployed performance | **refuted, sign reversed** | paired −0.038 ± 0.027 (§4.4) |
| Task inference `w = E[rφ]` is lossy | **refuted** | R² 0.129 vs ridge topline 0.127 (§4.7) |
| The mixture preimage is a set, not a point | **refuted, all 3 envs** | width = temperature (§4.9) |
| Antmaze/cube differ by local injectivity | **refuted** | Jacobian condition number 1.79 vs 6.67, σ_min smaller on antmaze (§4.8) |
| Prop. `insample`'s bootstrap is what's missing | **refuted** | Arm A within noise (§4.5) |
| The paper's ψ(s,u,u') construction works | **refuted on cube** | Arm B at BC level (§4.5) |
| `ψᵀw` specifically is the bad ranker | **refuted** | FQL's own critic also fails at K=512 (§4.6) |
| Q is flat because support collapses (H3) | **not supported** | ‖u‖² 4.66 → 4.54 (§4.7) |
| bc_coeff sweep will help | **not indicated** | gated on "Q has relief"; it has 1.1% (§4.7) |
| Pessimism is a lever | **no** | hpmatch measured pessimism=0.0 as a non-lever |
| More training (1M steps) helps | **no** | 0.134/0.162; extending sd1 500k→1M moved 0.302→0.284 |
| HP-matching to FB helps | **no** | 0.276/0.192 vs 0.236 ± 0.071 |
| Offline DSRL-SAC per task (scalar latent critic Q(s,u), latent actor, no decode in training) | **no** | 0.183 ± 0.038 (u_clip 3) / 0.127 (u_clip 1) vs pure-decode 0.142; Q spread 1.1–1.5% (HANDOFF 2026-09-03) |

### Live
1. **Actor-based improvement in latent space.** No argmax over a large candidate set; the
   actor moves smoothly against the (weak but locally usable) critic gradient. This is where
   the ε=0.05 residual's 0.905 ± 0.020 per-task result already lives. E4a's winner's-curse
   mechanism does not apply to a smooth actor.
2. **Small-K or regularized selection.** E4a's Spearman distribution (median 0.346, 54% of
   steps above 0.3) says ranking signal exists; the winner's curse at K=512 destroys it. An
   optimal K may exist and be small. Untested.
3. **A φ-grounding auxiliary.** D2 localized the weakness to the basis: the best linear
   read-out of φ explains ~13% of reward variance. A better estimator will not help; a
   better φ might.
4. **The limited-data regime is where this method wins.** At 10% data the hybrid is 0.238
   while FB collapses to 0.030. One seed per cell — replicate first. This is the only
   direction where the method currently beats a strong baseline.
5. **Antmaze coverage/ESS.** Open after the Jacobian probe excluded the geometric
   explanation. Two axes the probe cannot reach: 8 action dims vs 5, and actions of twice the
   norm, so a fixed decode budget spreads thinner.

### Theory priorities (from §2.7)
`conj:perturbation` first (it blocks the honest headline), then `conj:insample-bound` (the
theorem that would formally separate PSMFlows from FB and PSM).

---

## 7. Running things

Three stages. Stage A and B are expensive and already done for cube/antmaze/pointmaze.

```bash
# Stage C training. STORE is not optional: /var/local is 24G and filled on 2026-08-31,
# killing six runs mid-training with OSError errno 28.
STORE=/data-local/amsks/PSMFLows \
FLOW_CKPT=/var/local/amsks/exp/PSMFLows/bcflow_cube_single_20260726_135032/sd000_20260726_135037 \
FLOW_EPOCH=500000 \
PREIMAGES=/data-local/amsks/PSMFLows/preimages_cube_single_a20_n200.npz \
SEEDS="0 1 2" GROUP=<name> EXTRA='agent.use_point_preimage=true' \
  bash scripts/launch_psmflow.sh cube-single-play-singletask-v0 <GPU> 500000 online

# 500-episode eval (the only numbers that count; in-loop 50-ep evals carry ±0.115)
GPU=0 bash scripts/eval500.sh psmflow cube <run_dir> <out_name> [extra hydra args]
GPU=0 bash scripts/eval500.sh bc     cube -          bc_control

# Regenerate the tables from the eval JSONs
.venv/bin/python tools/make_tables.py
```

Arm B needs `agent.policy_index=latent agent.train_actor=false agent.acting=gpi` at **eval
as well as training** — ψ's index slot is `d_a` wide instead of `z_dim`, so restoring without
the flag fails.

**Process rules that exist because they were violated.** Before any launch: print the full
hyperparameter table, smoke-test the exact code path ~200 steps, and after launch re-read the
run's own `flags.json` to confirm the values landed. Every long job in a named tmux session.
State expected outcomes, including expected failures, before results arrive.

Key artifacts: `docs/tables/results.md` (generated), `docs/HANDOFF.md` (dated record),
`docs/reference/archived-checkpoints.md` (11.8 GiB of July runs on HF),
`/data-local/amsks/PSMFLows/logs/*.json` (every eval and diagnostic).

---

## 8. The state of the argument, in five sentences

The interface is not the problem: an oracle picking among the flow's own decoded candidates
scores 0.934 where the trained agent scores 0.220. The decoder is not the problem: exact
decode costs 0.038, not gains. Neither of the code's two deviations from the writeup was
load-bearing: fixing the bootstrap changes nothing, and running the paper's actual
construction lands at the behaviour-cloning floor. The critic cannot rank — but E4a shows
even a *proven* critic cannot rank under per-step best-of-K argmax, so the deployment scheme
is implicated alongside the representation. What remains defensible today is the
limited-data result (0.238 vs FB's 0.030 at 10% data) and the per-task residual policy
(0.905 ± 0.020), neither of which is the zero-shot claim the writeup makes.

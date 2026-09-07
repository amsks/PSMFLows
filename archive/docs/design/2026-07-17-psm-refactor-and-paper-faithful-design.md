# PSM: JAX-idiomatic refactor + paper-faithful features — design

- **Date:** 2026-07-17
- **Branch:** `feat/psm-integration`
- **Status:** design (awaiting review)
- **Scope:** `agents/psm.py`, `utils/psm_networks.py`, `tests/test_psm_*`, `configs/agent/psm.yaml`. FB (`agents/fb.py`, `utils/fb_networks.py`) is out of scope here — it gets its own follow-up spec reusing this pattern.

---

## 1. Context & motivation

An audit compared our PSM against four references: the paper (arXiv 2411.19418), the canonical
release `agarwalsiddhant10/PSM`, the RLU codebase (`/u/amsks/git/RLU`), and Meta Motivo's FB.
Findings that drive this work:

- Our JAX PSM is a **bit-exact port of Factored-FB**, which is a verbatim port of the **canonical
  released `agarwalsiddhant10/PSM`**. That canonical code implements the paper's *successor-feature*
  instantiation (§6): `M = ψ·φᵀ`, **no bias term**, and **closed-form** reward inference.
- The paper's *general* method — affine measure `M = Φw + b` with a **constrained-LP / dual**
  reward inference — is implemented faithfully only in **RLU `discrete_psm.py`** (discrete/gridworld).
  RLU's continuous `psm.py` has the same skeleton but is buggy/incomplete.
- Our closed-form reward inference (`z = E[r·φ]`, √d-normalized) is **algorithmically identical to
  Meta Motivo's** FB inference. The only zero-shot inference that differs from FB is the paper's
  constrained LP.

Two problems to solve:
1. **Clarity.** The current agent uses bespoke jit-plumbing (`_HashableDict`, `_stages`, `_draw_injection`
   returning an opaque `inj` dict, `compute_static`) that is hard to read. We want code that a human who
   has read the paper can follow, in the repo's idiomatic style (`ModuleDict`/`select()`/named `*_loss`
   methods, as in `agents/fql.py`).
2. **Fidelity.** We want the option to run the paper's *actual* method (bias term + constrained-LP
   inference), not only the FB-equivalent instantiation we currently have.

## 2. Goals / non-goals

**Goals**
- Phase 1: refactor PSM to idiomatic JAX with **identical numerics** (regenerate fixtures; equiv tests
  stay green at atol 1e-10). Separate network definitions from loss/update logic; give plumbing sensible
  names.
- Phase 2: add, **behind flags defaulting to reference-parity**, the paper-faithful features: bias term
  `b`, constrained-LP reward inference, and two small correctness fixes.

**Non-goals**
- No FB refactor here (separate spec).
- No discrete / per-action `φ` (gridworld) variant — out of scope for continuous OGBench.
- No change to default training behavior in Phase 1, nor to the default eval/inference path unless a flag
  is set.

## 3. Reference: code ↔ paper variable map

Paper notation only (no FB terms). This table goes in the module docstring of the refactored agent.

| Code variable | Paper symbol | What it does |
|---|---|---|
| `phi` / `PhiMap` | $\phi_s(s^{+})$ | Maps future state $s^{+}$ (`goal = next_obs`) to the $d$-dim basis vector — the learned basis spanning successor-measure space. |
| `sf_psi` / `PsiMap` | $\psi^{\pi}(s,a)$ | Coefficients for the continuous-$z$ (task) policy; with $\phi$ reconstructs its measure. Carries $w$ inside. |
| `psm_psi` / `PsiMap` | $\psi^{\pi_z}(s,a)$ | Coefficients for the codebook policies $\pi_z$; forces $\phi$ to represent the whole family (objective $\mathbb{E}_z[\mathcal{L}^{\pi_z}]$). |
| `z_cont` (`task_z`) | $w$ | The task coordinates that combine the basis; inferred from reward at test time. |
| `z_psm` (`proto_seed`) | $z$ | The policy seed identifying which codebook policy is represented. |
| `M` / `Ms` | $M^{\pi}(s,a,s^{+}) = \psi^{\pi}(s,a)^{\top}\phi_s(s^{+})$ | The successor measure over the batch$\times$batch grid. |
| `target_M` | $\gamma\,\bar{M}^{\pi}(s',\pi(s'),s^{+})$ | Bellman target (target networks). |
| `proto_sample`/`seed_to_action`/`powers`/`obs_hash` | $\pi(a\mid s,z)=\mathrm{UniformSample}(z+\mathrm{hash}(s))$ | Codebook behavior policy supplying the next action for the proto measure. |
| `contrastive_loss`→`offdiag` | $\tfrac{1}{2}\mathbb{E}[(m(s,a,s^{+})-\gamma\bar{m}(s',\pi(s'),s^{+}))^{2}]$ | Bellman TD residual over $s^{+}\neq s$. |
| `contrastive_loss`→`diag` | $-(1-\gamma)\mathbb{E}[m(s,a,s)]$ | Source term anchoring mass at the transition's own next state. |
| `ortho_loss` | $\phi\phi^{\top}\to I$ | Orthonormality regularizer (weight 1). |
| `Q` / `Qs` | $Q^{\pi}(s,a)=M^{\pi}r=\psi^{\pi}(s,a)^{\top}w$ | Value = coordinates · reward-weighted basis. |
| `infer_z` | $w=\mathbb{E}_{\rho}[r\,\phi]$ | Reward → task vector (closed form; LP form is Phase 2). |
| `off_diag`/`off_diag_sum` | mask $s^{+}\neq s$ | Splits source (diagonal) from TD (off-diagonal). |
| `discount` | $\gamma$ | Successor-measure discount. |
| `norm_z`/`project_z` | $\sqrt{d}$-normalization | Keeps $w$/$\phi$ on the $\sqrt{d}$ sphere. |
| *(absent in code)* | $b$ | Paper's bias term in $M^{\pi}=\Phi w+b$ — added in Phase 2. |
| *(absent in code)* | $w(\cdot)$ net | Paper's explicit coordinate map — folded into $\psi$. |

Not in the paper (implementation/stability only): `mix_ratio` (relabel a fraction of $w$ with $\phi(\text{goal})$),
`num_parallel` (ψ ensemble size), `pessimism_penalty`/`actor_pessimism_penalty` (ensemble-disagreement
penalty), `tau` (Polyak rate).

## 4. Deviations from the paper (present in our current code)

Ranked; "fix" column says how/if this spec addresses it.

| # | Deviation | Detail | Fix |
|---|---|---|---|
| D1 | **No bias term $b$** | Code computes $M=\psi\phi^{\top}$; paper is $M=\Phi w+b$. | Phase 2 (flag `use_bias`) |
| D2 | **Closed-form reward inference** | Code: $w=\mathbb{E}[r\phi]$ (= Meta Motivo FB). Paper: constrained LP $\max_w \mathbb{E}[(\Phi w+b)r]\ \text{s.t.}\ \Phi w+b\ge 0$. | Phase 2 (flag `inference=lp`) |
| D3 | **Diagonal source term form** | Code: $-\mathrm{mean}(\mathrm{diag}(M-\gamma\bar M))\cdot P$ (on the residual, scaled by ensemble count $P$). Paper: $-(1-\gamma)\mathbb{E}[m(s,a,s)]$ (the $(1-\gamma)$ factor is dropped). | Phase 2 (flag `diag_term=paper`) |
| D4 | **Proto action range** | `(rand-1)*2` → actions in $[-2,0]$ (operator-precedence bug from upstream), not $[-1,1]$. | Phase 2 (flag `proto_action_range`) |
| D5 | No explicit $w(z)$ net | Coordinates folded into $\psi$. Structural difference, not a numeric bug; the SF instantiation is valid. | Not changed (documented) |

Notes: `ortho_coef` is dead in the *canonical* code (hardcoded 1 / ×0); **our** port already exposes it as a
live knob (used for the `ortho_coef=1000` runs) — not a bug for us. The canonical "trains on 10 % of buffer"
and row-index hashing issues are **not** in our port.

## 5. Performance / stability tricks inventory

Tricks observed across implementations that matter for performance. "Have" = already in our code.

| Trick | Where | Status |
|---|---|---|
| Orthonormality reg on $\phi$ (diverse basis; makes closed-form inference valid) | paper Table 2; all | Have |
| Two-timescale LRs: slow basis $\mathrm{lr}_\phi=10^{-5}$ vs $\mathrm{lr}_\psi=\mathrm{lr}_{\text{actor}}=10^{-4}$ | our sweep winner | Have |
| Target networks + Polyak $\tau=0.01$ on $\phi,\psi$ | all | Have |
| Ensemble of $\psi$ heads ($P=2$) + disagreement pessimism ($0.5$ on $Q$) | canonical, ours | Have |
| HER-style $w$/goal mixing (`mix_ratio=0.5`) | canonical, ours | Have |
| $\sqrt{d}$ L2 normalization of $w$ and $\phi$ | all | Have |
| Detach $\phi$ in SF branch (basis trained only by proto branch) | canonical, ours | Have |
| LayerNorm+Tanh first layer of $\phi$ (`ntanh`) + orthogonal init | canonical, ours | Have |
| Actor BC term (flow-BC distillation / DDPG+BC `bc_coeff`) for offline stability | Factored-FB flow_bc, ours | Have |
| Terminals/masks forced to always-$\gamma$ (bootstrap every step) | our fix | Have |
| Seeded eval env (reproducible eval) | our fix | Have |
| Codebook of seeded deterministic policies → single-player optimization | paper Eq 8; all | Have (core) |
| **Bias term $b$** (affine, not linear, decomposition) | paper; RLU | **Phase 2** |
| **Constrained-LP inference** (non-negativity via dual gradient descent + learned multiplier) | paper Eq 10; RLU `discrete_psm.py` | **Phase 2** |

## 6. Plan overview

Two phases; ship and validate Phase 1 before starting Phase 2.

- **Phase 1 — idiomatic refactor, behavior-preserving.** Regenerate fixtures from the *same* PyTorch
  reference; `test_psm_networks_equiv` + `test_psm_agent_equiv` green at atol 1e-10.
- **Phase 2 — paper-faithful features, flag-gated.** Each flag defaults to reference-parity so Phase-1
  fixtures still pass with features off.

## 7. Phase 1 — idiomatic refactor (behavior-preserving)

### 7.1 File layout

- `utils/psm_networks.py` — **network modules only**, cleaned and documented. Unchanged math; clearer
  docstrings tying each module to §3.
- `agents/psm.py` — the agent: the §3 variable table in the docstring, named `*_loss` methods, an explicit
  `update`, reward inference. No network layer math inline.

### 7.2 Network layer

**Revised during implementation (2026-07-17):** an empirical probe showed the house `ModuleDict` prefixes its
param keys (`modules_phi`, …) and requires passing the *full* param tree to `select()`, which forces full-tree
gradients per stage and awkward subtree bookkeeping for PSM's per-network sequential update. We therefore use
**one `TrainState` per network** instead — `phi`, `proto_psi`, `sf_psi`, `actor` (+ `actor_vf` for the flow
variant), each carrying its own module + optimizer. Each stage is `grad → state.apply_gradients(grads=g)` with
the per-network learning rate built in (the classic jaxrl multi-optimizer pattern). Targets (`target_phi`,
`target_proto_psi`, `target_sf_psi`) are held as plain param pytrees and soft-updated in `apply_update`. This
meets every goal of this section (idiomatic `TrainState`, separated networks/losses, clear names, bit-exact)
while reading more cleanly for a multi-optimizer agent than the single-`ModuleDict` approach originally sketched.

### 7.3 Update structure (the one real constraint)

The reference update is **3-stage sequential with per-network learning rates and interleaved target
soft-updates**, and the SF stage reads the `φ` the proto stage just stepped. This cannot collapse to the
single-loss / single-optimizer `fql` shape without changing numerics, so we keep the structure but make it
readable:

```
update(batch):
    sampled = sample_step_inputs(batch, rng)      # named struct (replaces `inj`)
    # proto stage
    (loss, aux), grads = value_and_grad(proto_loss, wrt=(phi, proto_psi))(...)
    step phi optimizer;  step proto_psi optimizer
    soft-update target_phi, target_proto_psi
    # sf stage  (reads the just-updated phi)
    (loss, aux), grads = value_and_grad(sf_loss, wrt=sf_psi)(...)
    step sf_psi optimizer;  soft-update target_sf_psi
    # actor stage (reads the just-updated sf_psi)
    (loss, aux), grads = value_and_grad(actor_loss / flow_actor_loss, wrt=actor[, actor_vf])(...)
    step actor optimizer[; step actor_vf optimizer]
```

Optimizers are kept **per network** (each its own `optax.adam(lr)` + state) so each is stepped exactly once
per update — this is what preserves bit-exactness (a single shared optimizer stepped 3× would tick Adam state
on already-stepped subtrees). Params live in the `ModuleDict` tree (single checkpoint); optimizer states are a
sibling dict keyed by network name. This is the jaxrl multi-optimizer pattern; the only house-style deviation
is "not a single shared `tx`", which is required by the algorithm.

Each stage's math is a named method with a house-style signature, e.g.
`proto_loss(self, batch, sampled, grad_params)`, `sf_loss(...)`, `actor_loss(...)`, `flow_actor_loss(...)`.

### 7.4 Naming changes

Keep the paper's domain symbols (`phi`, `psi`); rename the plumbing:

| Now | New | Reason |
|---|---|---|
| `inj` | `sampled` (a small `flax.struct`/`NamedTuple`: `task_z`, `proto_seed`, `proto_next_action`, `sf_next_action`, `flow_noise`, `flow_x0`, `flow_t`, `actor_sample`) | "the randoms sampled for this step", each field named |
| `_draw_injection` | `sample_step_inputs` | says what it does |
| `z_cont` / `z_psm` | `task_z` / `proto_seed` | continuous task vector $w$ vs binary codebook seed $z$ |
| `psm_psi` (net + params) | `proto_psi` | "proto" (codebook) head, clearer than "psm" |
| `_stages` | the three named `*_loss` methods | one responsibility each |
| `compute_static` | `losses_and_grads` (test/debug helper) | plain name |
| `_HashableDict`, `_off`, `_step`, `_soft` | fold into `ModuleDict`/`TrainState` + small named helpers (`off_diagonal_mask`, `adam_step`, `polyak_update`) | drop bespoke jit-plumbing |

The bit-exact test path (injecting fixed sampled values) is preserved via `sample_step_inputs` accepting an
optional override, so fixtures still pin exact numerics.

### 7.5 Parity & tests

- Regenerate `tests/fixtures/psm_reference.npz` from the Factored-FB PyTorch reference (rename map:
  `psm_psi → proto_psi`; module submodule names unchanged where possible to minimize fixture churn).
- `test_psm_networks_equiv.py`, `test_psm_agent_equiv.py`: green at atol 1e-10 (static) / 1e-8 (K-step).
- `test_psm_smoke.py`: updated for renamed structure; existing regressions (flow actor kernel count, proto
  consumes batch index) preserved.

## 8. Phase 2 — paper-faithful features (flag-gated, default = parity)

Config gains a small `paper` block; every flag defaults to the current behavior so Phase-1 fixtures pass with
all features off.

### 8.1 Bias term `b`  (`use_bias`, default `false`)

- Add a bias head so `M = ψ·φᵀ + b`. Options: a scalar bias from a small `b(s,a,s⁺)`-style head, or fold a
  bias column into the measure — final form to be pinned in the implementation plan from RLU
  `discrete_psm.py` (`b_fc`). The continuous adaptation attaches `b` to the measure net used by both
  `proto_loss` and `sf_loss`, and to `infer_z`/`Q`.
- Off ⇒ code path is bit-identical to Phase 1.
- Tests: a numeric test that with `use_bias=false` the measure equals the Phase-1 measure; a shape/finite
  test with `use_bias=true`.

### 8.2 Constrained-LP reward inference  (`inference`: `correlation` | `lp`, default `correlation`)

- New eval-time solver ported from RLU `discrete_psm.py` (`infer_w` / `_infer_step_gc`), adapted to
  continuous: iterate `w` to maximize `E[(Φw+b)·r]` under the non-negativity penalty `min(Φw+b, 0)`, either
  with a fixed penalty coefficient or a learned Lagrange-multiplier net (dual gradient descent). Reuses the
  existing relabel/inference batch plumbing in `main.py`.
- **Training is untouched.** This only adds an alternative `infer_z_lp` selected at eval.
- Tests: a toy problem with a known optimum where LP beats correlation; a guard that `inference=correlation`
  reproduces the Phase-1 `infer_z`.

### 8.3 Small correctness fixes (each its own flag, default = parity)

- `diag_term`: `residual_scaled` (current) | `paper` (the `-(1-γ)·mean(diag(M))` form). Addresses D3.
- `proto_action_range`: `neg2_0` (current, reference parity) | `pm1` (the intended $[-1,1]$). Addresses D4.

## 9. Testing strategy (summary)

- Phase 1 gate: equiv tests at 1e-10/1e-8 after fixture regeneration; smoke green.
- Phase 2 gate: per-feature unit tests; **flag-off equivalence tests** (every new flag off ⇒ Phase-1 numerics);
  LP solver correctness on a toy.
- No PyTorch in `.venv` here, so live torch comparison uses the exported fixture (as today).

## 10. Risks & open questions

- **R1 (parity vs idiom):** keeping per-network optimizers is a deliberate deviation from the single-`network`
  house pattern; accepted because collapsing to one optimizer changes numerics. Documented in-code.
- **R2 (bias form):** the exact continuous form of `b` (scalar head vs folded column) is pinned during the
  implementation plan by reading RLU `discrete_psm.py` end-to-end; the spec fixes the interface, not the head
  internals.
- **R3 (LP at scale):** the constrained-LP inference is iterative and unproven in continuous control; it is
  strictly opt-in and does not affect the default (Meta-Motivo-equivalent) path.
- **R4 (fixture churn):** renaming `psm_psi → proto_psi` changes fixture keys; the regen script carries a
  rename map so the numeric ground truth is unchanged.

## 11. Out of scope / future

- FB (`agents/fb.py`) idiomatic refactor — separate spec reusing this pattern and the shared
  `NoiseConditionedActor` / `FlowVectorField` modules.
- Discrete / per-action `φ` (gridworld) PSM variant.
- Empirical study of whether bias + LP inference beat the closed-form path on OGBench (a runs/eval effort,
  not code).

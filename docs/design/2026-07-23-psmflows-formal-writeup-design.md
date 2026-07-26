# PSMFlows formal write-up — design spec for `PAPER/main.tex`

**Date:** 2026-07-23 · **Branch:** `feat/psm-integration` · **Source:**
`PAPER/RESEARCH_NOTE.md` v1 (2026-07-20), which this supersedes on every point where the
two disagree. **Deliverable:** a single self-contained math-first technical report at
`PAPER/main.tex`, replacing the existing 136-line sketch.

**Not in scope:** intro / related-work / experiments prose (the note keeps those),
any code change, any run.

## 0. What changed relative to the research note

Working the proofs surfaced three corrections. All were user-approved on 2026-07-23.

### F1 — "support conversion" cannot be a support statement

The ODE construction makes $G_\theta(s,\cdot)$ a diffeomorphism of $\mathbb R^{d_a}$
**onto** $\mathbb R^{d_a}$. Its image is therefore everything, and
$\operatorname{supp}\big(G_\theta(s,\cdot)_\#p_0\big) = \mathbb R^{d_a}$. The claim
"every decoded action is in-support" (note §2.3, §3.3) is vacuously false as a
topological statement.

The salvageable — and already half-written in the note — statement is about **mass**:
$\hat\mu_\theta\big(G_\theta(s,U)\mid s\big) = p_0(U)$ exactly, at every state. So the
result is a typical-set / high-probability one. Everywhere the note says "in-support",
the report says "$\hat\mu_\theta$-typical" or gives the mass bound.

Related honest point, promoted to a conjecture in §10: a smooth ODE flow driven by a
Lipschitz field cannot *exactly* represent a compactly-supported $\mu$. It approximates
one with $|\det J| \to 0$ off the data manifold. The idealization A2 (exact flow) is
therefore known to be unattainable in the narrow-data regime the paper targets; the
report says so rather than hiding it.

### F2 — the losslessness proposition is vacuous; the true theorem is stronger

Note Prop. 2 claims $\Pi_{\text{lat}}$ equals the well-covered policy class. Since
$G_\theta(s,\cdot)$ is a bijection, *every* policy on $\mathbb R^{d_a}$ is the pushforward
of some $\nu$, so $\Pi_{\text{lat}}$ is all policies and the claim carries no information.

The content lives one level up. Change of variables applies the same $1/|\det J|$ factor
to $\pi_\nu$ and to $\hat\mu_\theta$, so it cancels:

$$\frac{\pi_\nu(a\mid s)}{\hat\mu_\theta(a\mid s)} \;=\; \frac{\nu(u\mid s)}{p_0(u)},
\qquad u = E_\theta(s,a),$$

exactly and pointwise, at every state. Hence for every convex $f$ with $f(1)=0$,

$$D_f\big(\pi_\nu(\cdot\mid s)\,\|\,\hat\mu_\theta(\cdot\mid s)\big)
 \;=\; D_f\big(\nu(\cdot\mid s)\,\|\,p_0\big).$$

**This becomes the headline result (Prop. 3.1), and old Prop. 1 becomes its corollary**
($f$ = indicator). It is strictly stronger and it sharpens the VC-FB contrast: the
unknown, state-dependent behavior density ratio that conservatism penalties spend their
loss budget *approximating* becomes a known, state-independent, closed-form ratio against
a standard Gaussian. Action-space concentrability becomes a Gaussian-relative quantity.

**Necessary caveat the report must state.** The isometry governs *stochastic* latent
policies. A deterministic $\pi_u$ has $\nu = \delta_u \not\ll p_0$ and infinite
$f$-divergence from $\mu$. So the report splits the two regimes explicitly:

- **training time** — the family is used *in aggregate* with $u' \sim p_0$, so the
  marginal law of bootstrap actions is exactly $\hat\mu_\theta(\cdot\mid s')$ and the
  operator's concentrability coefficient is $1$ (Prop. 7.1);
- **deployment time** — the extracted policy is a state-wise $\arg\max$, deterministic;
  the governing guarantee is the mass bound (Cor. 3.2) with $u$ constrained to
  $U_\delta = \{\|u\|^2 \le \chi^2_{d_a}(1-\delta)\}$.

Conflating these is the single easiest way to over-claim, so they get separate
propositions.

### F3 — two smaller fixes

- **Laplace covariance.** Note §3.2 gives $\alpha^{-2}(J^\top J)^{-1}$. Correct value is
  $(2\alpha)^{-1}(J^\top J)^{-1}$ (from $\exp(-\alpha\|J\delta\|^2)
  = \exp(-\tfrac12\delta^\top(2\alpha J^\top J)\delta)$).
- **Successor-measure convention.** The note is internally inconsistent: §3.1's Bellman
  equation puts the point mass at $s$ ($\mathbf 1_{\{s\in X\}}$, i.e. $\sum_{t\ge 0}$),
  but §3.3's loss uses $\varphi(s_i')$, which is the $\sum_{t\ge1}$ convention. The report
  adopts the **successor convention** $M^{u\to u'}(s,X) = \mathbb E[\sum_{t\ge1}
  \gamma^{t-1}\mathbf 1_{\{s_t \in X\}}]$, matching FB and matching the implemented loss,
  with a footnote on the other variant.

## 1. Decisions taken (user-approved)

| Decision | Choice |
|---|---|
| Artifact | Math-first technical report; no intro/related/experiments prose |
| SM representation | Affine general form → bilinear as a *derived* special case under a named assumption |
| Rigor | Prove what's provable; label the rest as Conjecture with an explicit NOT PROVED |
| Indexing family | Fixed-index $\pi_u$, with transport coherence promoted to a defined, measurable assumption (A3) |
| Notation | Follow `RESEARCH_NOTE.md` + code: $\varphi$ = state basis, $\psi$ = successor-feature side. This **flips** the current `main.tex` usage. |

## 2. Document structure

| § | Content | Status |
|---|---|---|
| 1 | Setting, notation; the restricted-support problem restated as **unbounded concentrability of the training policy family** (sets up F2) | — |
| 2 | Behavior flow. **Lemma 2.1**: A1 ⇒ $G_\theta(s,\cdot)$ is a $C^1$-diffeomorphism and $\det J = \exp\int_0^1\operatorname{tr}\partial_u v\,dt > 0$ (Picard–Lindelöf + Liouville — yields invertibility *and* the log-det). **Lemma 2.2**: change of variables. A2 = exact flow, flagged as idealization | proved |
| 3 | Flow-indexed family. **Prop. 3.1** divergence isometry (F2) → **Cor. 3.2** mass conversion (old Prop. 1) → **Cor. 3.3** bounded concentrability for $\nu\ll p_0$ → **Cor. 3.4** density-adapted resolution, exact form $\lvert\det J\rvert = p_0(u)/\hat\mu_\theta(G(s,u)\mid s)$. **Def. 3.5** transport coherence via a mode partition; **A3**; **Remark 3.6**: A3 is needed for *statistical efficiency of the basis*, not for soundness — falsifiable by D2 | proved |
| 4 | SMs over the family. $M^{u\to u'}$, Bellman with diagonal closure. **A4** affine form $m = b(s,u,x) + \Phi(s,u,x)^\top w^{u'}$ (basis independent of the policy index). **A5** factorized basis $\Phi = A(s,u)\varphi(x)$ ⇒ **Prop. 4.3** bilinear $\psi^\top\varphi$ with $\psi = A^\top w^{u'} + \beta$, affine in $w^{u'}$. **Remark 4.4** + table: free-$\psi$ implementation drops affineness in $w^{u'}$, hence keeps closed-form $w$ but loses constrained-LP inference | proved |
| 5 | **Prop. 5.1** SF Bellman identity $\psi(s,u,u') = \mathbb E_{s'}[\varphi(s') + \gamma\psi(s',u',u')]$; **Cor. 5.2** the reward identity — settles both conjectures in the old `main.tex` §5 (both were right, and the independence is from $u$ *and* $u'$); **Cor. 5.3** latent Q-iteration | proved |
| 6 | Dataset latents. Exact preimage; **Lemma 6.1**: under A2, $E_\theta(s,a) \sim p_0$ when $(s,a)\sim\mathcal D$ — so D3's $\chi^2$ test is a direct test of A2. **Prop. 6.2** Laplace/EM posterior with the corrected covariance; **Remark 6.3**: the posterior is an isotropic ball of radius $\sim(2\alpha)^{-1/2}$ *in action space*, pulled back | proved |
| 7 | **Prop. 7.1 (in-sample TD)** stated *distributionally*: under A2 the queried $(s',a')$ law equals $\rho\otimes\mu$, concentrability $=1$; table vs FB ($\infty$) and PSM ($\sup 1/(\lvert\mathcal A\rvert\mu)$). **Prop. 7.2** contraction + fixed-point error. **Remark 7.3**: the contraction is standard; what is new is that $\epsilon$ is estimable without $\mathcal A$-concentrability | proved |
| 8 | Losses (Alg. 1), derived by expanding $\lVert m - \mathcal Tm\rVert^2_{L^2(\rho)}$; **Remark**: $\mathcal L_{\text{ortho}}$ is not a heuristic — it is what makes $w = \mathbb E_\rho[r\varphi]$ the least-squares solution | derived |
| 9 | Inference ladder, Alg. 2–4. **Prop. 9.1** GPI bound with $\epsilon$ split into flow / TD / reward-regression error. **Remark 9.2** adversarial-$u$: the $\arg\max$ must be confined to $U_\delta$, which is exactly Cor. 3.2's ball. **Prop. 9.3** Rung 2 dominates Rung 1 in the finite-$\Lambda$ limit | proved |
| 10 | Conjectures and open problems, each with a strategy and an explicit NOT PROVED: **C1** flow-error perturbation (and the honest obstruction — $D_f$ is not $W_2$-continuous, so the $W_2$ route needs a smoothed divergence); **C2** affine span (old Prop. 5); **C3** continuous-space in-sample bound; **C4** the compact-support/diffeo tension (F1); **C5** does A3 hold for CFM flows (empirical only). Plus the unaddressed **state-distribution shift** | not proved |
| A | Assumptions A1–A5 in one table, with which result needs which, and the falsifying diagnostic (D1–D6) for each | — |

## 3. Success criteria

- `pdflatex` compiles `PAPER/main.tex` clean (no errors; undefined-reference warnings resolved).
- Every numbered claim is either proved inline or carries an explicit NOT PROVED.
- Every assumption A1–A5 appears in the Appendix A table with a diagnostic.
- No occurrence of "in-support" used to mean the mass statement (F1).
- Notation matches the code: $\varphi$ = state basis, $\psi$ = SF side.
- The `\section{Problem}`-era typos and the unfinished "we require that" sentence are gone.

## 4. Risks

- **Over-claiming via F2's caveat.** Mitigation: separate propositions for the training
  and deployment regimes; no shared wording.
- **Assumption creep** — closing a proof by quietly strengthening A1–A5. Mitigation:
  Appendix A table makes every dependency visible in one place.
- **Divergence from the note.** The note stays as-is; this spec records the three points
  of disagreement (F1–F3) so the note can be reconciled in a later pass.

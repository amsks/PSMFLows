# PSMFlows: Behavior Foundation Models from Restricted-Support Data via Flow-Indexed Policy Families

**Research note — v1 (2026-07-20).** The precise formulation of the project, the claims we
can defend, the theory we should prove, and the experimental program. Supersedes the
sketch in `PAPER/main.tex` (kept for history); positions against the literature as of
July 2026.

---

## 0. One-paragraph thesis

Zero-shot RL methods (FB, PSM) promise a single pretrained model that produces a policy
for *any* reward, but their training loop optimizes and evaluates actions the dataset
never contains — FB's actor does TD3-style maximization over the full action space, and
PSM's proto codebook policies take *uniform random* actions. On realistic,
narrow-support offline data both therefore learn successor measures of policies that
cannot be executed and backups that are hallucinated. **PSMFlows** replaces the policy
space itself: a behavior-cloned conditional flow $G_\theta(s,u)$ turns latent noise
$u\sim p_0$ into dataset-supported actions, and we define the policy family *in latent
space* — every policy the representation ever sees, evaluates, or improves over decodes
through the flow and is therefore supported. Successor measures are learned over this
family in PSM's affine form; zero-shot inference is generalized policy improvement
(GPI) over the flow-indexed continuum, optionally sharpened by fully in-sample latent
Q-iteration. The support constraint $a \in \operatorname{supp}\mu_D(\cdot|s)$ — a
different, unknown set at every state — becomes the single, global, geometry-free
constraint $u \in \operatorname{typ}(p_0)$.

---

## 1. Setting and notation

MDP $(\mathcal S, \mathcal A, P, \gamma)$, $\mathcal A \subseteq \mathbb R^{d_a}$,
reward unspecified at train time. Fixed offline dataset
$\mathcal D = \{(s_i, a_i, s_i')\}_{i=1}^N$ from unknown behavior policy
$\mu(a|s)$, with state marginal $\rho$.

For a policy $\pi$, the **successor measure**

$$
M^\pi(s, a, X) \;=\; \sum_{t\ge 0} \gamma^t \,\Pr(s_t \in X \mid s_0 = s,\, a_0 = a,\, \pi),
\qquad X \subseteq \mathcal S,
$$

so $Q_r^\pi(s,a) = \int r(x)\, M^\pi(s,a,\mathrm dx)$ for any reward $r$.

**The restricted-support problem.** Every zero-shot method must, during training,
evaluate its measure/critic at actions $a'$ at successor states. If $a' \notin
\operatorname{supp}\mu(\cdot|s')$, the TD target is evaluated where the model has no
data — the classic extrapolation error of offline RL, but now baked into the
*representation*, poisoning every downstream task at once. Concretely:

- **FB** ([Touati & Ollivier 2021; Touati et al. 2023]): actor performs unconstrained
  $\max_a F(s,a,z)^\top z$; the measure $F^\top B$ is trained with bootstrapped actions
  from that actor.
- **PSM** ([Agarwal et al. 2024, arXiv:2411.19418]): the basis $\varphi$ is trained on the
  measures of hash-codebook policies $\pi_z(s) = \mathrm{UniformSample}(z + \mathrm{hash}(s))$
  — *uniform over the action cube*, i.e. maximally out-of-support on any narrow dataset.

Both work when data has broad coverage (ExORL-style) and degrade when it does not
(documented for FB by Jeen et al., NeurIPS 2024, arXiv:2309.15178 — OOD-action value
overestimation on small/homogeneous data, failing "across all tasks simultaneously",
fixed there by *conservatism penalties*; we instead fix the *policy space*).

---

## 2. The flow-indexed policy family

### 2.1 Behavior flow

Train conditional flow matching on $\mathcal D$: with $u_0 \sim p_0 = \mathcal N(0, I_{d_a})$,
$u_1 = a$, $u_t = (1-t)u_0 + t u_1$,

$$
\mathcal L_{\text{flow}}(\theta) = \mathbb E_{(s,a)\sim\mathcal D,\, u_0, \, t}
\big\| v_\theta(s, t, u_t) - (a - u_0) \big\|_2^2 .
$$

Define the **decoder** $G_\theta(s, u)$ = solution at $t{=}1$ of
$\dot u_t = v_\theta(s, t, u_t)$, $u_{t=0} = u$. When flow matching is exact,
$G_\theta(s,\cdot)_\# p_0 = \mu(\cdot|s)$, and for each $s$ the map
$G_\theta(s,\cdot): \mathbb R^{d_a} \to \mathbb R^{d_a}$ is a **diffeomorphism** (ODE
flows are invertible; $d_u = d_a$). Write $E_\theta(s,a) = G_\theta(s,\cdot)^{-1}(a)$
(backward ODE integration) and $J(s,u) = \partial G_\theta/\partial u$.

### 2.2 The three nested policy classes

1. **Fixed-index policies** $\ \pi_u(s) = G_\theta(s, u)$, one deterministic Markov
   policy per $u \in \mathbb R^{d_a}$. This is the *indexing family* over which the
   representation is learned. Interpretation: $u$ is a consistent "mode selector" — the
   flow's transport structure tends to map a fixed $u$ to corresponding modes of
   $\mu(\cdot|s)$ across states (an empirical property we will measure, not assume).
2. **Latent-policy class** $\ \Pi_{\text{lat}} = \{\, \pi_\nu : \pi_\nu(\cdot|s) =
   G_\theta(s,\cdot)_\# \nu(\cdot|s), \ \nu(\cdot|s) \in \Delta(\mathbb R^{d_a}) \,\}$ —
   re-sample $u$ every step from a state-conditioned latent policy $\nu$. **Claim
   (Prop. 2): this class is exactly the set of policies with
   $\operatorname{supp}\pi(\cdot|s) \subseteq \operatorname{supp}\mu(\cdot|s)$** (all
   "well-covered" policies), by change of variables
   $\nu(u|s) = \pi(G_\theta(s,u)|s)\,|\det J(s,u)|$. So nothing is lost: the latent
   parameterization is a *lossless reparameterization* of the well-covered class, and
   $\nu = p_0$ recovers the behavior policy itself.
3. **Downstream/greedy policies**: state-wise selection $u^\*(s) = \arg\max_u Q(s,u)$ —
   a member of $\Pi_{\text{lat}}$ with $\nu(\cdot|s) = \delta_{u^\*(s)}$.

### 2.3 Why the latent space is the right constraint set (the geometry claim)

Change of variables: the pushforward density is
$\mu(a|s) = p_0(E_\theta(s,a)) \,/\, |\det J(s, E_\theta(s,a))|$. Two consequences:

- **(Support conversion — Prop. 1.)** For any measurable $U \subseteq \mathbb R^{d_a}$,
  $G_\theta(s,U)$ has behavior-probability mass exactly $p_0(U)$. Constraining
  $u \in \{\|u\|^2 \le \chi^2_{d_a}(1-\delta)\}$ (a *fixed, state-independent ball*)
  constrains actions to a set of conditional behavior mass $\ge 1-\delta$ at **every**
  state simultaneously. The state-dependent, unknown-geometry support constraint
  becomes a single Euclidean ball.
- **(Density-adapted resolution.)** Where $\mu(\cdot|s)$ is dense, $|\det J|$ is small:
  a unit step in $u$ moves the action little. Search/optimization over $u$ at fixed
  resolution is automatically fine-grained inside data modes and coarse across
  low-density gaps — an implicit trust region in behavior density, obtained for free.

This is the precise version of "the flow converts the difficult constraint into an easy
one," and it is what pure conservatism penalties (CQL-style, VC-FB) approximate by
loss shaping. **Positioning note:** the two closest latent-space precursors (LAPO,
DSRL) both rely on *empirical* latent clipping and explicitly lack a support theorem
(LAPO: "performance is significantly worse when we do not limit the latent values";
DSRL appendix: atypical noise "may be unreliable", clipped to $[-b,b]^d$). Prop. 1 with
the flow-error perturbation term would be the **first formal support statement in this
line of work** — worth doing carefully, including the honest failure mode (flow error
$\varepsilon_s$ is itself largest exactly where data is thin).

*Honesty clause:* per-state action support does **not** imply trajectory/state coverage
— $\pi_u$ can still drift to states $\rho$ barely covers. That limitation is shared with
every offline method (it is the state-distribution-shift half of offline RL); what we
remove is the *action extrapolation* half, which is the half TD backups amplify. Our
experiments must include a state-coverage diagnostic, not just claims.

---

## 3. Successor measures over the flow family

### 3.1 Representation

Following PSM's affine decomposition $M^\pi = b + P^\top w^\pi$ (exact for finite MDPs;
[Agarwal et al. 2024]), we learn densities w.r.t. $\rho$ in factored form. For the
fixed-index family, define

$$
m(s, u, u', x)\;:=\;\frac{M^{u\to u'}(s, \mathrm dx)}{\rho(\mathrm dx)}
\;\approx\; \psi(s, u, u')^\top \varphi(x),
$$

where $M^{u\to u'}(s,\cdot)$ is the occupancy from $s$ when the first action is
$G_\theta(s,u)$ and the continuation policy is $\pi_{u'}$. ($\psi$ plays the role of
PSM's $\psi/w$-side and FB's $F$; $\varphi$ is the shared basis / FB's $B$. Code:
`sf_psi`/`proto_psi` and `phi` in `agents/psm.py`.)

**Bellman equation** (occupancy counting $s_0$):

$$
M^{u\to u'}(s, X) = \mathbf 1_{\{s\in X\}} + \gamma\, \mathbb E_{s' \sim P(\cdot|s, G_\theta(s,u))}\,
M^{u'\to u'}(s', X).
$$

Note the recursion **closes on the diagonal**: only $M^{u'\to u'}$ appears on the
right. The two-argument object is the Q-like one-step extension of the diagonal
V-like object, exactly as $(Q^\pi, V^\pi)$.

### 3.2 The training oracle: dataset transitions carry their own latents

The TD update needs triples $(s, u, s')$ with $s' \sim P(\cdot|s, G_\theta(s,u))$. We
only have $(s, a, s')$ — so we need $u$ with $G_\theta(s,u) = a$: the **preimage**
$u = E_\theta(s,a)$, which for a flow is a *single point*, computable by backward ODE
(implemented: `agents/fql.py:_get_preimage_and_jacobian`). Because the trained flow is
imperfect and we want robustness, we use the $\varepsilon$-relaxed preimage posterior

$$
q(u \mid s, a) \;\propto\; p_0(u)\, \exp\!\big(-\alpha \|G_\theta(s,u) - a\|^2\big),
$$

approximated by the existing EM Gaussian-mixture machinery
(`compute_full_proposal_distribution_em`, `utils/flow_inversion.py`), precomputed
offline per transition (`tools/precompute_preimages.py`). As $\alpha\to\infty$ this
concentrates on $E_\theta(s,a)$ with covariance $\tfrac{1}{\alpha^2}(J^\top J)^{-1}$ —
the mixture is not a hack but the Laplace approximation of the exact posterior, and its
spread *is* the local inverse-Jacobian geometry.

**Critical diagnostic (D3 below): prior-typicality of inverted latents.** If the flow
underfits, $E_\theta(s,a)$ lands in the tail of $p_0$ and train-time latents mismatch
test-time samples $u \sim p_0$. Metric: distribution of $\|E_\theta(s,a)\|^2$ vs
$\chi^2_{d_a}$; round-trip error $\|G_\theta(s, E_\theta(s,a)) - a\|$ (implemented:
`tools/validate_flow_inversion.py`).

### 3.3 Losses

With batch $\{(s_i, a_i, s_i', u_i)\}$, $u_i \sim q(\cdot|s_i,a_i)$, index draws
$u'_j \sim p_0$, and the PSM/FB norm-squared TD loss over $x^- = s_j'$ (off-diagonal)
and $x^+ = s_i'$ (diagonal):

$$
\mathcal L_{\text{SM}} = \mathbb E\Big[\big(\psi(s_i,u_i,u') ^\top \varphi(s_j')
- \gamma\, \bar\psi(s_i', \tilde u_i', u')^\top \bar\varphi(s_j')\big)^2\Big]
\;-\; 2\, \mathbb E\big[\psi(s_i, u_i, u')^\top \varphi(s_i')\big]
\;+\; \lambda_{\perp}\, \mathcal L_{\text{ortho}}(\varphi),
$$

where $\tilde u_i' $ realizes the continuation policy at $s_i'$: for the fixed-index
family, $\tilde u' = u'$ itself and the bootstrapped action is $G_\theta(s_i', u')$ —
**in-support at $s'$ by construction**. This is the load-bearing sentence of the whole
method: *every action the TD backup ever evaluates is a flow decode.* Compare PSM,
whose analogous bootstrap uses the uniform hash action.

This directly replaces PSM's proto branch: the codebook $\{\pi_z\}_{z\text{ binary}}$
becomes $\{\pi_{u'}\}_{u'\sim p_0}$ — a *continuous, dataset-consistent* codebook. The
`sf` branch (task-conditioned policies) is subsumed: $u'$ **is** the policy index; no
separate task-policy family is trained at pretrain time, which is exactly the
task/policy disentanglement claim (§5).

### 3.4 What the two-argument $\psi$ buys: the identity from `main.tex` §5, resolved

Define successor features $\psi(s,u,u') = \int \varphi(x)\, M^{u\to u'}(s,\mathrm dx)$.
The Bellman equation above gives, for **every** $u'$:

$$
\psi(s, u, u') - \gamma\, \mathbb E_{s'}\,\psi(s', u', u') = \varphi(s).
$$

So the conjecture in the draft is **true and is nothing but the SF Bellman identity**:
$(\psi(s,u,u') - \gamma \psi(s',u',u'))^\top w \approx \varphi(s)^\top w = \hat r_w(s)$,
independent of $u'$ (up to TD noise). Consequences:

1. Reward inference stays PSM's closed form: $w = \mathbb E_{\mathcal D}[r(s)\varphi(s)]$
   (least squares on the basis; `infer_z`).
2. The latent Q-function for task $w$ obeys **standard Q-iteration in latent space**:
   $Q^\*_w(s,u) = \hat r_w(s) + \gamma \max_{u'} Q^\*_w(s', u')$ trained *only* on
   dataset transitions with their inverted $u$ — a fully in-sample (SARSA-support) DP:
   the max is over $u'$, and any $u'$ decodes to a supported action at $s'$. No
   OOD-action query exists anywhere in the loop.

### 3.5 Zero-shot inference: a three-rung ladder (each rung a paper contribution)

- **Rung 1 — Flow-GPI (no test-time training).**
  $\pi^{\text{GPI}}_w(s) = G_\theta\big(s, \arg\max_{u} \max_{u'} \psi(s,u,u')^\top w\big)$,
  maxima approximated by $K$ samples $u, u' \sim p_0$ (+ a few gradient steps in $u$).
  This is **generalized policy improvement over a continuum of behavior-consistent
  policies**: the classic GPI bound gives
  $Q^{\pi^{\text{GPI}}} \ge \sup_{u'} Q^{\pi_{u'}} - \tfrac{2\epsilon}{1-\gamma}$ with
  $\epsilon$ the SF/reward-regression error — *and every policy in the sup is
  executable.* USFA did GPI over goal-indexed families; ours is over the
  dataset's own behavioral continuum.
- **Rung 2 — Latent Q-iteration (test-time DP, still zero new environment samples).**
  Initialize $Q_w(s,u) \leftarrow \max_{u'}\psi(s,u,u')^\top w$, then run the in-sample
  Q-iteration of §3.4 on the offline data for a small budget. Strictly improves on
  one-step GPI toward the best policy in $\Pi_{\text{lat}}$ (state-wise resampling
  closure), while remaining in-support.
- **Rung 3 — Amortized latent actor.** Distill $\arg\max_u Q_w(s,u)$ into
  $u = \pi_\eta(s, w)$ (or a small conditional flow over $u$) for deployment-speed
  action selection across all tasks — the BFM interface.

### 3.6 Theory targets (what we should actually prove)

- **Prop. 1 (support conversion).** Exact-flow case: statements of §2.3, plus a
  perturbation bound: if $W_2(G_\theta(s,\cdot)_\#p_0,\ \mu(\cdot|s)) \le \varepsilon_s$,
  decoded actions are within $\varepsilon_s$-Wasserstein of supported ones (support
  holds approximately, degrading gracefully with flow error).
- **Prop. 2 (losslessness).** $\Pi_{\text{lat}}$ = all well-covered policies (change of
  variables). Corollary: the optimal well-covered policy for any $r$ is realized by
  some $\nu^\*(u|s)$, and Rung-2 Q-iteration converges to it in the tabular-latent limit.
- **Prop. 3 (in-sample TD).** The PSMFlow TD operator only evaluates $(s', a')$ pairs
  with $a' = G_\theta(s',u')$: under exact flow, it is a contraction on the data
  distribution with no out-of-support queries; error bounds then follow the in-sample
  offline RL analyses (IQL / In-Sample Softmax style) rather than requiring uniform
  concentrability over $\mathcal A$. **This is the theorem that separates us from FB/PSM:
  their operators require action-space concentrability; ours requires only
  state-coverage.**
- **Prop. 4 (GPI over the continuum).** Instantiate the GPI bound with $\epsilon$ split
  into (flow error) + (SF TD error) + (reward regression error), all measurable.
- **(Stretch) Prop. 5 (affine span).** PSM's affine-space result restricted to
  $\Pi_{\text{lat}}$: the measures of the fixed-index family affinely span the measures
  of all well-covered policies under stated conditions — justifying learning the basis
  on fixed-$u$ policies only. (Even a finite-MDP version suffices.)

---

## 4. What exists in the codebase (grounding)

| Piece | Status |
|---|---|
| PSM (JAX, parity-audited vs reference) | `agents/psm.py`, parity story closed (seed variance + ceiling; see HANDOFF) |
| FB (JAX, bit-exact port) | `agents/fb.py`, 13/13 parity tests |
| FQL (flow BC + one-step distill) | `agents/fql.py` — reuse its vector field as $G_\theta$ |
| Exact preimage + Jacobian (backward ODE) | `agents/fql.py:_get_preimage_and_jacobian` |
| Preimage posterior (Laplace + IS + EM mixture) | `compute_full_proposal_distribution[_em]`, `utils/flow_inversion.py` |
| Offline preimage precompute → augmented dataset | `tools/precompute_preimages.py` |
| Inversion validation (round-trip, ESS) | `tools/validate_flow_inversion.py` |
| Flow steering (project action → manifold) | `utils/flow_steering.py` |
| OGBench envs + eval harness | `envs/`, `main.py`, `utils/evaluation.py` |

The new agent (`agents/psmflow.py`) is PSM's skeleton with: proto branch → flow-indexed
branch ($u'$ replaces the hash codebook; bootstrap action $G_\theta(s',u')$), dataset
latents from the preimage pipeline, GPI/latent-Q inference replacing `infer_z`+actor
rollout at eval. The flow $G_\theta$ is pretrained (FQL's BC flow, frozen), so the
representation trains on a *stationary* policy family — a stability advantage over FB's
moving actor that we should measure explicitly.

---

## 5. Experimental program

**Environments.** OGBench: `pointmaze-{medium,large}-navigate`, `antmaze-{medium,large}-{navigate,stitch}`,
`cube-single/double-play`, each with the 5 singletask rewards for zero-shot eval
(train the representation reward-free once; infer $w$ per task from relabeled rewards —
the harness already does this for PSM).

**The support axis (the headline experiment).** For each env, a coverage ladder:
1. full `play`/`navigate` data (broad);
2. `stitch` (short fragments — poor trajectory coverage, good action support);
3. mode-restricted: filter the dataset by trajectory clusters / directions (drop k of m
   k-means modes of $(s,a)$) — narrow *action* support, our target regime;
4. `-noisy` variants (action noise — tests preimage robustness);
5. expert-only narrow data (single-mode).

**Hypothesis:** FB and PSM degrade sharply from (1)→(3,5) (their actor/codebook queries
leave support); PSMFlows degrades gracefully; FQL (single-task, task-specific training)
upper-bounds each task.

**Baselines** (grouped by what they test):
- *Representation, wrong policy space:* PSM, FB (both in-repo, parity-verified).
- *Right problem, penalty fix:* **VC-FB / MC-FB** (Jeen et al., arXiv:2309.15178;
  official code at `enjeeneer/zero-shot-rl`) — the must-beat: does fixing the policy
  space beat fixing the loss? BREEZE (arXiv:2510.15382) cite + compare if code exists.
- *Mechanism, no representation:* DSRL-style per-task latent-noise SAC/Q-iteration
  trained per reward (no zero-shot; upper-bounds what the latent space alone gives) —
  cheap to run since it is our Rung-2 machinery with a known reward.
- *Per-task oracles:* FQL (in-repo); IDQL-style sample-and-rank with a per-task critic
  (ablation: is the SM doing work beyond re-ranking behavior samples?).
- HILP if time permits.

**Staged validation (kills the idea early if it's wrong):**
- **D1 Flow fidelity:** MMD / k-means-mode histograms, flow samples vs data actions,
  per env (user's WP0). Gate: modes reproduced, no mass off-support.
- **D2 Index consistency:** for fixed $u$, roll $\pi_u$; measure (a) outcome diversity
  across $u$ (entropy of final-state clusters), (b) mode-consistency of $G(s,u)$ across
  states. Gate: distinct $u$ → distinct, coherent behaviors (the family is non-trivial).
- **D3 Inversion health:** round-trip error, ESS, $\chi^2$-typicality of
  $E_\theta(s,a)$ (§3.2). Gate: typicality holds; else the flow needs capacity/steps.
- **D4 Latent-space control sanity (no SM yet):** single-task in-sample latent
  Q-iteration (§3.4 with known $r$) vs FQL on 2 envs. Gate: within ~10% of FQL —
  proves the latent space supports DP before we invest in the representation.
- **D5 Zero-shot:** full matrix (envs × coverage ladder × {PSMFlows rungs 1–3, PSM, FB,
  ablations}), 5+ seeds (reference curves show extreme seed variance; HANDOFF).
- **D6 State-coverage honesty check:** occupancy divergence of deployed policies from
  $\rho$ (the caveat of §2.3), reported alongside returns.

**Ablations.** Point-preimage vs EM posterior; $K$ (GPI sample budget); rung 1 vs 2 vs 3;
frozen vs finetuned flow; $d_z$; $u'$-codebook vs PSM hash codebook *holding everything
else fixed* (the crispest single-variable comparison in the paper); one-step distilled
vs multi-step ODE decode.

---

## 6. Related-work positioning (deep-research verified, 2026-07-20)

Adversarially verified sweep (25/25 claims confirmed, 0 refuted; 22 primary sources).
**Bottom line: the intersection is unoccupied.** No verified work learns successor
measures / FB representations / any zero-shot BFM over a policy family indexed by the
latent noise of a behavior-cloned generative model. The two ingredient lines are each
mature — which is exactly what makes the combination credible rather than exotic.

### 6.1 Latent/noise-space policy optimization (the mechanism line)

| Work | What it does | Delta vs PSMFlows |
|---|---|---|
| **PLAS** (Zhou et al., CoRL 2020, arXiv:2011.07213) | Policy in CVAE latent space so support is "naturally satisfied" | Earliest precursor; single-task, single-reward, CVAE not flow, no SM/zero-shot |
| **LAPO** (Chen et al., NeurIPS 2022) | Latent policy $\pi(z\|s)$ over CVAE latents via TD3 | Single-task; **explicitly defers multi-task to future work**. Cautionary: latents must be clipped or performance collapses |
| **DSRL** (Wagenmaker et al., CoRL 2025, arXiv:2506.15799) | **Closest mechanism competitor.** SAC over the noise space of a frozen BC diffusion policy, black-box, incl. steering π₀ | Single-task, *online*, reward-supervised adaptation. No SM/FB/zero-shot, no reward-free continuum. Its in-support property is informal (noise clipped to $[-b,b]^d$, appendix admits atypical noise "may be unreliable") |
| **FQL** (Park et al., ICML 2025, arXiv:2502.02538) | Flow BC + one-step distillation, RL in *action* space | Deliberately avoids noise-space optimization ("rather than directly guiding an iterative flow policy…"); single-task |
| **IDQL** (Hansen-Estruch et al., arXiv:2304.10573) | Critic-weighted resampling of diffusion-BC samples | Sample-and-rank, no latent optimization; single-task |
| **QPILOTS** (arXiv:2606.14801, June 2026) | Test-time Q-gradient steering of frozen flow policies (incl. frozen π0.5 VLA on LIBERO) | Positions itself *against* DSRL-style noise-space RL; ordinary Q-ensembles; zero SM/FB/zero-shot content |
| Q-learning w/ adjoint matching (arXiv:2601.14234), LPS (arXiv:2603.05296) | Recent single-task noise-space/steering variants | Same axis: single-task, no representation over the family |

**Takeaways for us.** (i) The noise-space *interface* is established prior art — we must
cite DSRL prominently and claim the *representation over the family*, not the interface.
(ii) Both LAPO and DSRL found that latents need explicit bounding — so **Prop. 1 done
properly (typical-set statement + flow-error perturbation bound) would be the first
formal version of the support argument in this entire line**, upgrading community
folklore into a theorem. (iii) QPILOTS/DSRL steering frozen π₀-class VLAs confirms the
VLA extension path (§7 of the project, KISSKI experiments) is timely.

### 6.2 Zero-shot RL / BFMs (the representation line)

| Work | Policy continuum indexed by | Restricted support? |
|---|---|---|
| **FB** (Touati & Ollivier 2021; Touati et al. 2023) | Task latent $z$ (actor argmax family) | No — degrades on narrow data (below) |
| **PSM** (Agarwal et al., ICML 2025, arXiv:2411.19418) | Task $w$ + binary hash codebook for the basis | **No — its limitations section literally hands us the problem**: "An interesting future direction would be to study the impact of dataset coverage on zero-shot RL performance." Experiments were broad-coverage (uniform gridworld, RND/ExORL). Full-text: zero occurrences of diffusion/generative/noise |
| **VC-FB / MC-FB** (Jeen et al., NeurIPS 2024, arXiv:2309.15178) | FB task latent $z$ | **Yes — the canonical prior work on our motivation**: documents FB's OOD-value overestimation on small/homogeneous data (failure "across all tasks simultaneously"); fixes with value/measure *conservatism penalties*. We fix the *policy space* instead — this is our must-beat baseline and clearest contrast |
| **FB-CPR / Meta Motivo** (arXiv:2504.11054, ICLR 2025) | FB latent $z$, discriminator-regularized toward a behavior dataset | Adjacent but different regime: *online* unsupervised RL with observation-only data; a soft behavioral prior, not a hard offline support constraint. Cite carefully, don't claim it as offline prior work |
| **HILP** (Park et al., ICML 2024, arXiv:2402.15567) | Unit directions in a learned Hilbert space, plain IQL | Latent = movement direction, not generative noise |
| **ZOL** (arXiv:2602.01962, 2026) | FB-style | Diagnoses FB task-inference collapse under limited/biased coverage — corroborates our motivation from the inference side |
| Fast BFM adaptation (arXiv:2504.07896), FB-AWARE (arXiv:2412.04368) | Search/encode in the *task* latent $z$ of a pretrained BFM | Optimization over a BFM latent exists — but the latent is the task index, never the generative-noise index |
| **BREEZE** (arXiv:2510.15382) | **Nearest near-miss**: FB + behavioral regularization + task-conditioned *diffusion* policy extraction | Policy family still indexed by FB task $z$; diffusion noise is sampled/marginalized, never the policy index. **Action item: read in full before submission** (surfaced via verifier evidence, not a 3-vote claim) |

### 6.3 The verified gap claim

> Latent-noise policy spaces are used for **single-task** extraction/steering
> (PLAS, LAPO, DSRL, QPILOTS); successor-measure/BFM continua are indexed by **task
> latents** over unconstrained actor families (FB, PSM, HILP, FB-CPR, BREEZE); the
> restricted-support failure of zero-shot RL is documented and currently treated with
> **conservatism penalties** (VC-FB/MC-FB) or **soft imitation priors** (FB-CPR).
> **No work makes the generative model's noise variable the policy index of a
> successor-measure representation** — i.e., builds the zero-shot layer on the
> supported-policy continuum itself. That intersection is PSMFlows.

Three-way novelty triangle to draw in the paper: DSRL (mechanism, no representation) ×
PSM/FB (representation, wrong policy space) × VC-FB (right problem, penalty-based fix).

### 6.4 Known holes in the sweep (do not over-claim; re-run before submission)

The verifier produced **no confirmed claims** on: flow-inversion literature (incl. the
arXiv 2605.10821 method cited in the draft — read directly), Diffusion-QL/SRPO,
noise-conditioned critics, USFA, distributional SM / γ-models / generative occupancy
models, and SF-based VLA fine-tuning. These are *unresearched, not absent*. Negative
gap claims decay monthly (QPILOTS appeared June 2026) — re-verify at submission time.

---

## 7. Risks and open questions

1. **Fixed-$u$ policies may be weird.** Nothing forces $G(s,u)$ to select semantically
   coherent modes across states; if $\pi_u$ behaviors are incoherent, the SM basis is
   built on a degenerate family. D2 measures this; mitigation: learn a light
   re-indexing (e.g., condition the family on a low-dim $z$ with an InfoMax term) —
   design change, keep in back pocket.
2. **Preimage typicality failure** (D3): flow underfit → inverted latents atypical →
   train/test mismatch in $u$. Mitigations: bigger flow, more ODE steps, posterior
   $\alpha$ tuning, latent whitening.
3. **GPI max over samples is a soft max**: with $K$ samples the improvement bound
   weakens; gradient refinement in $u$ may exploit $\psi$ errors (adversarial-$u$
   effect) — measure Q-gap vs decoded return; clamp to typical set.
4. **Compute**: preimage precompute is per-transition ODE + EM (~1M transitions/env) —
   already designed as offline one-shot (`precompute_preimages.py`); budget it.
5. **PSM inheritance risk:** reference PSM itself is extremely seed-noisy (HANDOFF:
   0.2–0.9 peaks). Every claim needs ≥5 seeds and full curves, or reviewers can't
   distinguish us from variance.

---

## 8. Contribution statement (draft for the paper)

1. **Formulation:** zero-shot RL over restricted-support data as successor-measure
   learning over a flow-indexed, provably lossless reparameterization of the
   well-covered policy class (Props. 1–2).
2. **Algorithm (PSMFlows):** in-sample successor-measure TD over the flow family
   (Prop. 3), with a three-rung zero-shot inference ladder (flow-GPI → in-sample latent
   Q-iteration → amortized actor) and its GPI guarantee (Prop. 4).
3. **Systems insight:** dataset transitions carry their own latents via flow inversion;
   the $\varepsilon$-preimage posterior is the Laplace/EM object already implemented.
4. **Evidence:** a coverage-ladder benchmark on OGBench showing FB/PSM's failure mode is
   the policy space, not the amount of data — and that fixing the space (not adding
   penalties) recovers zero-shot control.

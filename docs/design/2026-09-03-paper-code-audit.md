# Paper ↔ code audit — 2026-09-03

Read-only audit of `agents/psmflow.py` and its dependencies against the two write-ups.
Extends `docs/COMPENDIUM.md` §2–§3 (which is verified correct where it overlaps); it does
not repeat the parts of §3.1–§3.5 that still hold.

## 0. Sources of truth, and a housekeeping problem

**`PAPER/main.tex` is not on disk.** `PAPER/` was untracked at `2007e65` (2026-08-29,
"chore: untrack PAPER/ from the repo — maintained outside git for now"). The on-disk
`PAPER/ICLR/` tree is a *newer, untracked* set of files (`content/method.tex`,
`content/preliminaries.tex` exist on disk but in no commit). The formal report was
recovered for this audit from `git show 2007e65^:PAPER/main.tex` (1346 lines) and every
`main.tex` line number below refers to that blob. **Action item: re-add PAPER/, or at least
pin the report's commit, before anyone else audits against it.**

Three documents, in decreasing age:

| tag | file | character |
|---|---|---|
| **REPORT §1–§9** | `PAPER/main.tex` @ `2007e65^`, `\section{...}` up to `sec:inference` | the formal, theorem-carrying construction: `ψ(s,u,u')`, `u'~p₀`, three-rung ladder, **no actor** |
| **REPORT §10** | same file, `\section{LatentFlowPSM}` (l.1063–1164) | "the algorithm that is trained and evaluated" — already documents the actor, ensemble, κ-pessimism, `u_clip=3`, mix_ratio |
| **ICLR** | `PAPER/ICLR/content/{preliminaries,method}.tex` (on disk, untracked) | newest and *least* finished: 30-line `method.tex`, back to `φ(s,u₀,u₀')`, a "latent noise DSRL style critic", **no actor**, no ensemble, no clipping |

**Where the two drafts disagree** (this matters for §2 below):

1. **Policy index.** ICLR `method.tex:8` — "`φ(s,u_0,u'_0)^Tψ(s^+)`. Note that `φ` takes two
   `u` as an input: one as the current state action, and one that is the index of the
   future rollout." REPORT §5–§8 agrees (`ψ(s,u,u')`). REPORT §10 `Objects` (l.1091)
   disagrees with both: "The policy index is the task vector `w`."
2. **Actor.** ICLR has none — it says "sample initial noise to maximize `Q^*`"
   (`method.tex:27`), i.e. Rung 1/2. REPORT §1–§9 has none in the training loop (Rung 3 is
   an optional post-hoc distillation). REPORT §10 makes the actor *load-bearing in the
   backup*.
3. **The bootstrap.** REPORT eq. `loss-empirical` bootstraps `ψ̄(s',u',u')` with `u'~p₀`;
   REPORT eq. `latent-sm-loss` bootstraps `ψ̄(s',π_η(s',w,ε),w)`. The two boxed equations
   in the same document contradict each other, and REPORT §10 says so explicitly
   ("Identical to (loss-empirical) **except for the bootstrap action**", l.1108).
4. **Latent set.** REPORT §1–§9 confines search to `U_δ = {‖u‖² ≤ χ²_{d_a}(1−δ)}`
   (Cor. `mass`, Rem. `adversarial-u`); REPORT §10 uses the L∞ box `‖u‖_∞ ≤ 3`. Different
   sets; only the χ² ball carries the `1−δ` behaviour-mass guarantee.
5. **ICLR's simplification claim.** `method.tex:21–25` conjectures the one-step term is
   `u'_0`-independent, reducing to plain Q-learning. REPORT settles this as
   Cor. `reward-identity` (l.649) — correct, and it holds exactly.

Below, "REPORT" without qualification means §1–§9 (the theorem-carrying part); "§10" is
LatentFlowPSM.

---

## 1. Symbol / equation / algorithm table

Code line numbers are current on `feat/inversion-integration` @ `1116712`.

### 1.1 Stage A — the behaviour flow

| write-up | LaTeX | code | status |
|---|---|---|---|
| Flow-matching objective | `\mathcal L_{\mathrm{flow}}(\theta)=\E[\|v_\theta(s,t,u_t)-(a-u_0)\|_2^2]` (main.tex:199; ICLR `preliminaries.tex:76`) | `agents/fql.py:338-346` (`x_0~N(0,I)`, `x_1=actions`, `t~U[0,1]`, `x_t=(1-t)x_0+tx_1`, `vel=x_1-x_0`, `bc_flow_loss=mean((pred-vel)**2)`) | **faithful** |
| `u_t=(1-t)u_0+tu_1` | main.tex:196 | `agents/fql.py:341` | faithful |
| Freeze `θ` after Stage A (Alg. `pretrain` l.1 / §10 `Objects` "never updated afterwards") | main.tex:938, 1073 | `psmflow.py:98-99` (`flow_vf`/`flow_onestep` are plain param trees, not `TrainState`s — no optimizer exists for them); loaded at `psmflow.py:618-622`, `_load_flow_params` `psmflow.py:677-708` | **faithful** |
| Decoder `G_θ(s,u)` = ODE at `t=1` (Def. `decoder`, eq. `ode`) | main.tex:207-213 | `psmflow.py:476-486`, `else` branch (Euler, `flow_decode_steps`) | faithful *only* under `gpi_decode=ode` |
| — | — | `psmflow.py:478-479` one-step distilled decoder | **IMPLEMENTED DIFFERENTLY — see §3** |
| One-step distillation term (`alpha*distill_loss`) | **not in any draft** | `agents/fql.py:348-353,360` | **extra machinery** (Stage A only) |

### 1.2 Stage B — inversion

| write-up | LaTeX | code | status |
|---|---|---|---|
| Point preimage `u* = E_θ(s,a)` by backward integration | Def. `decoder`; §`sec:preimage` "there is exactly one, namely `u=\Eth(s,a)`" (main.tex:723) | `agents/fql.py:23-64` — implicit-Euler backward loop at `:47-56`, forward Jacobian `jax.jacfwd` at `:58-64`. Driven by `tools/precompute_preimages.py:159` → `utils/flow_inversion.py:254-313` | **faithful** (with a documented ~0.0013% divergence-to-NaN rate, `fql.py:29-38`) |
| ε-relaxed posterior `q_\alpha(u\mid s,a)\propto p_0(u)\exp(-\alpha\|\Gth(s,u)-a\|^2)` | eq. `posterior`, main.tex:747-750 | `agents/fql.py:208-209` `log_energy = -alpha*sum((actions-action)**2) - 0.5*prior_scale*sum(samples**2)`; IS variant `fql.py:123-128` | **faithful since the 2026-08-14 prior fix + the 2026-08-31 squared-norm fix.** Code generalises the paper with `prior_scale` (paper fixes it at 1); default `configs/inversion/default.yaml:89` `prior_scale: 1.0` = the paper's target exactly |
| Laplace covariance `\Sigma_\alpha=(I+2\alpha J^\top J)^{-1}` | Prop. `laplace`, eq. `laplace`, main.tex:757-762 | `agents/fql.py:95` `cov_eigvals = clip(1.0/(2.0*alpha*eigvals + prior_scale), 0.01, 1.0)` | **faithful** at `prior_scale=1`. The `clip(0.01, 1.0)` floor/ceiling is not in the paper (the ceiling is vacuous at `prior_scale=1`, `fql.py:93-94`) |
| Laplace mean `m_\alpha=2\alpha\Sigma_\alpha J^\top J u^\star` | eq. `laplace` | **NOT IMPLEMENTED** — `fql.py:97` returns `x_0` (`=u*`) as the mean, i.e. the `α→∞` limit `m_α→u*`, and refines it by EM instead | deviation, benign: the paper itself calls EM "a multi-modal refinement of this Laplace object" (Rem. `laplace-geometry`, main.tex:786) |
| "fit `q_α` via Prop. laplace (**EM refinement optional**)" | Alg. `pretrain` l.2, main.tex:940 | `agents/fql.py:145-295` (`compute_full_proposal_distribution_em`), 10 EM steps, `n_components=num_clusters` | faithful. **Note `configs/inversion/default.yaml:1` `num_clusters: 1`** — the shipped "mixture" is a *single* Gaussian, so the multi-modal refinement the paper describes has never actually been run with `K>1` |
| `u_i ~ q_\alpha(\cdot\mid s_i,a_i)` used in the loss | eq. `loss-empirical`, main.tex:917; §10 `Dataset latents` (l.1096) softens to "a latent `u_i` with `G(s_i,u_i)≈a_i`" | **Both arms exist**: mixture draw `utils/datasets.py:181-185` → `utils/flow_inversion.py:121-171`; point `utils/datasets.py:177-178`. Selected by `agent.use_point_preimage` (`main.py:213`, `configs/agent/psmflow.yaml:106`) | **IMPLEMENTED DIFFERENTLY in practice**: every shipped result uses `use_point_preimage=true` (COMPENDIUM §3.5/§4.9, and the CLAUDE.md launch line), i.e. `q_α = δ_{u*}`, not the ε-relaxed posterior |
| Preimage validity / repair | **not in the paper** | `utils/flow_inversion.py:31-105` | extra machinery, necessary (see §3) |

### 1.3 The objects

| write-up | LaTeX | code | status |
|---|---|---|---|
| `M^{\pi_{u'}}(s,\Gth(s,u),\cdot)` — the two-argument successor measure | Def. `Muu`, main.tex:512-520 | represented implicitly by `M = psi(obs, idx, u) @ phi(goal).T`, `psmflow.py:118` | see index row below |
| Bellman / diagonal closure `M^{u\to u'}(s,X)=\E_{s'}[\1+\gamma M^{u'\to u'}(s',X)]` | Lem. `bellman`, eq. `bellman` | `psmflow.py:126` + `agents/psm.py:60-64` | faithful **in shape**; which latent occupies `u'` is the deviation |
| Affine form `m = b + \Phi^\top w^{u'}` | Ass. `affine`, eq. `affine` | **NOT IMPLEMENTED** — no bias `b`, no policy-independent `Φ`, no LP over `w` | deviation **the paper itself declares**: Rem. `tradeoff` (main.tex:601-627) tabulates "free ψ (implemented)" with "constrained (LP) inference over `w`: **no**" |
| Bilinear `m(s,u,u',x)=\psi(s,u,u')^\top\varphi(x)` | Prop. `bilinear`, eq. `bilinear` | `psmflow.py:118` | faithful |
| `\varphi(x)` basis | Def. `sf` | `utils/psm_networks.py:52-71` `PhiMap`, `psmflow.py:79` | faithful (final `sqrt(d)·x/‖x‖` norm, `psm_networks.py:45-49`, is not in the paper) |
| `\psi(s,u,u')` — index slot carries a **policy latent** | Def. `sf`, main.tex:634-637; ICLR `method.tex:8` | `utils/psm_networks.py:139-155` `PsiMap(obs, z, action)`; call site `psmflow.py:118` passes `idx = self._index(sampled)` (`psmflow.py:135-143`) | **IMPLEMENTED DIFFERENTLY by default** — `policy_index='task_vector'` puts `w` in the index slot (`psmflow.py:143`, `configs/agent/psmflow.yaml:51`), which is §10 `Objects` (l.1091) but contradicts REPORT §5–§9 and ICLR. `policy_index='latent'` restores the paper's `ψ(s,u,u')` |
| `\varphi` orthonormality `\E_\rho[\varphi\varphi^\top]=I_d` | Cor. `reward-inference`, main.tex:672 | `agents/psm.py:67-71` `ortho_loss` — batch Gram form `0.5·Σ_offdiag((φφ^T)_{ij})²/|off| − mean‖φ_i‖²`, which is `0.5·‖E[φφ^T]−I‖_F² − d/2` in expectation | **faithful up to a factor of ½** and the choice of off-diagonal pairs as the independent-sample estimator. Applied to `φ(next_obs)`, not `φ(obs)` (`psmflow.py:117,130`) — consistent with `x~ρ` |
| `\lambda_\perp` | eq. `loss-empirical` | `ortho_coef=1000.0`, `configs/agent/psmflow.yaml:7` | faithful (value not specified by the paper) |

### 1.4 The measure loss

REPORT eq. `loss-empirical` (main.tex:919-930):
```
L_SM = (1/B²)Σ_{i,j}(ψ(s_i,u_i,u')ᵀφ(s'_j) − γ ψ̄(s'_i,u',u')ᵀφ̄(s'_j))² − (2/B)Σ_i ψ(s_i,u_i,u')ᵀφ(s'_i) + λ_⊥ L_ortho
```
§10 eq. `latent-sm-loss` (main.tex:1110-1122), identical but `u_i^{+}=\pi_\eta(s_i',w_i,\epsilon_i)` and index `w_i`.

| term | code | status |
|---|---|---|
| online `ψ(s_i,u_i,·)ᵀφ(s'_j)` grid | `psmflow.py:117-118` | faithful |
| target `γ ψ̄(s'_i,·,·)ᵀφ̄(s'_j)` | `psmflow.py:119,126`; `γ` applied in `agents/psm.py:61` `diff = M - discount*target_M` | faithful |
| **which latent is bootstrapped** | `psmflow.py:173-174` `u_next = u_clip * actor(next_obs, task_w, noise)` | **IMPLEMENTED DIFFERENTLY (default)** — the *actor's* latent (§10), not `u'~p₀` (REPORT). `backup_explore_frac=1.0` (`psmflow.py:179-184`, yaml:94) or `policy_index=latent` (`psmflow.py:185-190`) restore the paper's version |
| negative/positive split | `agents/psm.py:60-64`: `offdiag = 0.5·Σ((diff·off)²)/|off|`, `diag = −mean(diag(diff))·B` | **IMPLEMENTED DIFFERENTLY (cosmetically)**: the diagonal `i=j` is *excluded* from the squared term (FB convention) rather than included as `(1/B²)Σ_{i,j}`; the positive term carries coefficient `B` rather than `2/B`, and includes a `−γψ̄ᵀφ̄` diagonal piece that is constant under `stop_gradient`. Same gradient direction, different scale — no config restores the literal `2/B` |
| stop-gradient on the target | `psmflow.py:129` `jax.lax.stop_gradient(target_M)` | faithful |
| target nets, Polyak | `\bar\psi,\bar\varphi` "updated by Polyak averaging" (§10, main.tex:1123) | `psmflow.py:390-391` + `agents/psm.py:101-103`; `tau=0.01`, yaml:6 | faithful |
| **discount `γ`** | `\gamma\in[0,1)`, never numbered | `discount: 0.98`, `configs/agent/psmflow.yaml:5` | faithful (unspecified) |
| ensemble of size `P`, target = "mean minus κ standard deviations" | §10 `Objects`/l.1123 | `psmflow.py:128` `target_M = tmean - pessimism_penalty*tunc`; `agents/psm.py:74-79` `targets_uncertainty` returns the **mean absolute pairwise difference**, not the std | in the paper (§10 only); the dispersion statistic differs. At `P=2, κ=0.5` this is *exactly* min-Q (yaml:9-12) |

### 1.5 Task vectors and readout

| write-up | LaTeX | code | status |
|---|---|---|---|
| `w_i`: Gaussian w.p. `1-p`, else `\varphi(s_j')` permuted, both projected to the sphere of radius `\sqrt d`; **no gradient** | §10 `Task vectors`, main.tex:1101-1104 | `psmflow.py:155-161`; `project_z` `agents/psm.py:88-92` (`sqrt(d)·z/‖z‖`); `jax.lax.stop_gradient` at `:158` | **faithful — and this is in the paper (§10 only), contrary to the audit brief's premise.** `mix_ratio=0.5` = `p`, yaml:29 |
| `w=\E_\D[r(x)\varphi(x)]` closed-form inference | Cor. `reward-inference`; §10 `Deployment`, main.tex:1160 | `psmflow.py:454-457` `infer_z`; hook `psmflow.py:466-474` `infer_eval_z`; called from `main.py:330-338` | **faithful**, plus two unpapered eval knobs: `eval_relabel_size=10000` (`main.py:335`) and `eval_reward_shift=1.0` (`main.py:337`) |
| `Q_w(s,u)=\psi(s,u,w)^\top w` | §10 `Objects`, main.tex:1085 | `psmflow.py:244` (actor), `:510`/`:518` (GPI) | faithful |

### 1.6 Inference / deployment

| write-up | LaTeX | code | status |
|---|---|---|---|
| **Rung 1** flow-GPI: draw `u'_1..K, u_1..K ~ p₀`; `argmax_{i,j} ψ(s,u_i,u'_j)ᵀw`; return `G(s,u_î)` | Alg. `gpi`, main.tex:967-979 | `psmflow.py:499-513` — **verbatim**, but only under `policy_index='latent'`. Under the default index, `:514-521` is the degenerate `K`-candidate version with `w` in the index slot | faithful under `policy_index=latent`; degenerate otherwise |
| Rung 1 line 3: "optionally refine `u_î` by a few gradient steps … **projected onto** `U_δ`" | main.tex:975-976 | **NOT IMPLEMENTED** | missing (declared optional) |
| `U_δ = \{u:\|u\|^2\le\chi^2_{d_a}(1-\delta)\}` (χ² ball) | Cor. `mass`, Rem. `adversarial-u`, main.tex:395-410, 1025-1029 | `jnp.clip(..., -u_clip, u_clip)` — an **L∞ box** (`psmflow.py:151,168-169,181-182,233,350-351,504-505,514`) | **IMPLEMENTED DIFFERENTLY.** §10 `Objects` (l.1075) sanctions the box; REPORT §1–§9 does not, and only the χ² ball carries Cor. `mass`'s state-uniform `1−δ` mass guarantee |
| **Rung 2** in-sample latent Q-iteration | §`sec:inference`, Prop. `rung2`, main.tex:1031-1053 | **NOT IMPLEMENTED.** No latent value-iteration loop exists; `tools/latent_q_sanity.py` is a Rung-1 eval, not Rung 2 | missing |
| **Rung 3** amortised latent actor `\pi_\eta(s,w)` distilled from `argmax_u Q_w(s,u)` | §`sec:inference` l.1055-1060 | `psmflow.py:210-251` `flow_actor_loss` | **IMPLEMENTED DIFFERENTLY**: REPORT distils Rung-3 from a *converged* `Q_w`, post hoc, "adds no guarantee". Code trains it **jointly and inside the backup** (§10 Alg. `latentflowpsm` l.4-5). §10's version is what the code does |
| Actor loss `-\bar Q/|\bar Q| + \lambda_{bc}\|u^a-\tilde u\|^2 + \|v_\xi - (u_i-u_{i,0})\|^2` | eq. `latent-actor-loss`, main.tex:1128-1136 | `psmflow.py:239` (CFM), `:247` (normalised `−Q`), `:248` (distill), `:249` sum | **faithful**, one nit: the normaliser is `mean|Q_s|` over ensemble members (`:247`) where the paper writes `|mean Q|` |
| "ψ receives no gradient from the actor loss; `G_θ` receives none from either" | main.tex:1137 | `psmflow.py:242-244` (psi read at stored params, outside `argnums`); `flow_vf` has no optimizer | **faithful** |
| Actor updated at the **pre-update** ψ | §10 Alg. l.5, main.tex:1147 | `psmflow.py:393-405` comment + `new.actor`/`new.actor_vf` on `self.psi.params` read inside `flow_actor_loss` | faithful |
| Deployment `a = G_\theta(s,\pi_\eta(s,w,\epsilon))` | §10 `Deployment`, main.tex:1162 | `psmflow.py:559-563` + `:568` | faithful |

### 1.7 Assumptions and propositions (no code equivalent)

| statement | main.tex | code / diagnostic |
|---|---|---|
| Ass. `lipschitz` (regularity of `v_θ`) | :204 | none. `utils/networks.py` `ActorVectorField` is an unconstrained MLP (`layer_norm=False`, yaml:100) |
| Lem. `diffeo` / eq. `liouville` | :216 | `tools/diag_flow_jacobian.py` measures `dG/du` conditioning (closest) |
| Ass. `exact` (`μ̂ = μ`) | :268 | `tools/validate_flow_inversion.py` typicality (χ² on `‖E(s,a)‖²`, gate ≥0.95) — the paper's own named test (Lem. `typicality`, :727); also `tools/diag_flow_fit.py`, `tools/diag_generated_pair_support.py` |
| Prop. `isometry` / Cor. `mass` / Cor. `conc` | :328, :383, :412 | **nothing measures any of these** |
| Def./Ass. `coherence` (ε-coherence of `G_θ`) | :469, :484 | **not measured as defined.** `tools/eval_fixed_u_rollouts.py` (D2) is the closest: within-`u` vs across-`u` rollout consistency, not a mode-partition mislabel rate |
| Ass. `affine`, Ass. `factorised` | :548, :577 | not implemented, not checked |
| Prop. `sf-identity`, Cor. `reward-identity` | :639, :649 | not checked (would be a cheap residual probe) |
| Prop. `insample` (`u'⊥s'` ⇒ `C=1`) | :807 | **hypothesis violated by default** (§2.1); no diagnostic measures `C` |
| Prop. `contraction` | :837 | not checked |
| Prop. `gpi` (`ε`-accuracy of `ψᵀw` vs `Q^{π_{u'}}`) | :981 | `tools/calibration_check.py` measures exactly this ranking premise; `tools/diag_policy_ranking.py`, `tools/diag_latent_ranking_oracle.py` |

---

## 2. Discrepancy list, ranked by how much each could change results

### 2.1 The bootstrap latent is the actor's, not `u'~p₀`  — **top-ranked, already run, no effect**

- **Paper (REPORT).** eq. `loss-empirical`: `γ ψ̄(s'_i, u', u')ᵀφ̄(s'_j)` with "`u'~p₀` a fresh
  index per batch element" (main.tex:917), and Prop. `insample` requires `s'~ρ` and `u'~p₀`
  **independently** — that independence is the entire `C=1` claim (main.tex:807-820).
  ICLR agrees (`max_{u'_0}` over prior draws, `method.tex:18`).
- **Code.** `psmflow.py:173-174`: `u_next = u_clip * self.actor(batch["next_observations"], task_w, noise)`.
  A deterministic function of `s'`. `psmflow.py:29-33` states this honestly in the module
  docstring.
- **Flag.** `agent.backup_explore_frac=1.0` (`psmflow.py:179-184`) replaces every bootstrap
  latent with a clipped prior draw; `agent.policy_index=latent` (`psmflow.py:185-190`) does
  it structurally.
- **Already run?** **Yes — Arm A.** COMPENDIUM §4.5: `0.171 ± 0.113` (seeds 0.220/0.164/0.130)
  vs its matched control `0.220 ± 0.037`. *"Satisfying Prop. insample's hypothesis … changes
  nothing."* Also note §10 sanctions the code's version, so this is a REPORT-vs-§10 conflict,
  not a code bug.

### 2.2 ψ's index slot carries `w`, not a policy latent `u'` — **already run, refuted**

- **Paper.** REPORT Def. `sf`: `\psi(s,u,u')`; ICLR `method.tex:8` is emphatic
  ("`φ` takes two `u` as an input"). §10 `Objects` l.1091 contradicts them: "The policy index
  is the task vector `w`."
- **Code.** `psmflow.py:143` `return sampled.u_index if self.config["policy_index"] == "latent" else sampled.task_w`,
  default `task_vector` (yaml:51). Consequence, correctly recorded in COMPENDIUM §3.2: the
  agent is **FB with a latent action space**, not PSM.
- **Flag.** `agent.policy_index=latent` (+ `train_actor=false acting=gpi`, needed at **eval as
  well as training** — psi's slot width changes).
- **Already run?** **Yes — Arm B.** COMPENDIUM §4.5: `0.083 ± 0.191` (0.006/0.160/0.084) vs
  matched gpi control `0.054 ± 0.032`, against a BC control of `0.068`. *"The writeup's
  construction is refuted on its own terms on cube."* Caveat carried in §4.5: Arm B moves the
  index **and** removes the actor together, so it does not isolate the index.

### 2.3 Stage-B ships the **point** preimage, not `q_α` — **material, never A/B'd at 500 ep**

- **Paper.** eq. `loss-empirical` draws `u_i ~ q_α(·|s_i,a_i)`; the whole of §`sec:preimage`
  and Prop. `laplace` exist to build that distribution; ICLR's introduction calls
  "identify a distribution over latent noise actions" a **core contribution**
  (`introduction.tex:19-24`).
- **Code.** Both arms exist (`utils/datasets.py:177-185`). Every shipped number used
  `agent.use_point_preimage=true` (COMPENDIUM §3.5, §4.9; CLAUDE.md's own launch line), i.e.
  `q_α → δ_{u*}`. Compounding it, `configs/inversion/default.yaml:1` sets `num_clusters: 1`,
  so even the mixture arm is a single Gaussian — the multi-modal EM the paper describes has
  never been run.
- **Flag.** `agent.use_point_preimage=false` (yaml:106, the config default; overridden by the
  documented launch command).
- **Already run?** Partially: COMPENDIUM §4.6 E4b measured *mixture* checkpoints and found
  ranking Spearman 0.054 vs 0.10 for the point arm — "mixture training does not create ranking
  signal" — but there is **no 500-episode, 3-seed mixture-vs-point success comparison** in
  §4.1. This is the largest paper claim with the weakest empirical settlement.

### 2.4 `U_δ` is an L∞ box, not the χ² ball — **unrun, cheap, and it is what the guarantee is about**

- **Paper.** Cor. `mass` (main.tex:383-397) and Rem. `adversarial-u` (main.tex:1019-1029):
  confining the search to `U_δ = {‖u‖² ≤ χ²_{d_a}(1−δ)}` "is *not* a numerical convenience —
  it is what converts Cor. `mass` into a guarantee about the deployed action."
- **Code.** `jnp.clip(u, -3.0, 3.0)` everywhere (`psmflow.py:151,168,181,233,350,504,514`).
  For `d_a=8` (cube), the box has volume `6^8` while `χ²_8(0.95)=15.5` gives radius 3.94 —
  the box admits points at `‖u‖=8.5`, far into the `p₀` tail, in exactly the corner directions
  an adversarial argmax prefers.
- **Flag.** **None.** `u_clip` is a scalar box half-width; there is no norm-ball projection
  anywhere in the codebase.
- **Already run?** No. §10 `Objects` sanctions the box, so this is a REPORT-vs-§10 conflict
  again, but it is the one where the theory says the difference is load-bearing.

### 2.5 Rung 2 does not exist — **the paper's own "dominates Rung 1" claim is untested**

- **Paper.** Prop. `rung2` (main.tex:1038-1051) proves Rung 2 attains Rung 1's target
  *without* the `2ε/(1−γ)` GPI slack. ICLR `method.tex:15-25` treats latent Q-learning as
  **the** policy-inference mechanism.
- **Code.** **NOT IMPLEMENTED.** The code jumps from Rung 1 (`gpi_select`) to a Rung-3-shaped
  actor trained jointly. `tools/latent_q_sanity.py` is a Rung-1 oracle-`w` eval despite the name.
- **Flag.** None. **Already run?** No.
- Given COMPENDIUM §4.6 E4a (decode-then-score is dead as a deployment scheme even with a
  proven ranker), Rung 2 is the only untried rung whose failure mode differs from Rung 1's
  winner's curse.

### 2.6 One-step distilled decoder at deployment — **measured, small, in the code's favour**

- **Paper.** `G_θ` is the ODE solution (eq. `ode`). No draft mentions a distilled decoder.
- **Code.** `psmflow.py:478-479` + `gpi_decode: onestep` (yaml:111) is the default and was
  used by every shipped run (COMPENDIUM §3.4).
- **Flag.** `agent.gpi_decode=ode agent.flow_decode_steps=100`.
- **Already run?** **Yes.** COMPENDIUM §4.4 E2, 5 seeds × 500 ep, paired: ODE−onestep =
  **−0.038 ± 0.027** for the actor arm, −0.010 ± 0.016 for gpi. The exact decoder is a small
  consistent *loss*. Note the awkward corollary: Stage B inverts `G_100`, so the "exact" latent
  is exact for a decoder the agent never uses.

### 2.7 Actor trained jointly rather than distilled post hoc — REPORT vs §10

REPORT Rung 3 is a *post-hoc* regression onto `argmax_u Q_w(s,u)` that "adds no guarantee";
the code (`psmflow.py:400-405`, §10 Alg. l.5) trains it in the same loop, and its output feeds
the backup (2.1). No flag isolates "same actor, distilled at the end". Not run.

### 2.8 Affine PSM structure (`b`, `Φ`, LP over `w`) absent

The paper concedes this in Rem. `tradeoff` (main.tex:601-627) as "a deliberate trade, not an
oversight". `agents/affine_psm.py` exists in the registry as a separate agent. Not a psmflow
discrepancy; noted so nobody re-derives it.

### 2.9 Loss-scale details

`(1/B²)Σ_{i,j}` vs off-diagonal-only with a `0.5` factor; `−(2/B)` vs `−B`; `‖·‖_F²` ortho vs
its `½·` Gram estimator (`agents/psm.py:60-71`). Same minimiser, different effective
`λ_⊥`/learning-rate pairing. No flag. Unlikely to matter, but it means `ortho_coef=1000` is not
comparable to a `λ_⊥` anyone reads off the paper.

---

## 3. Machinery in the code that the write-ups do not mention

"In §10" below means the REPORT's LatentFlowPSM section *does* describe it, so it is not
actually extra — the brief assumed several of these were undocumented and they are not.

| machinery | code | in a draft? | load-bearing for the reported numbers? |
|---|---|---|---|
| Latent actor + CFM anchor + distillation | `psmflow.py:210-251` | **yes, §10** eq. `latent-actor-loss`; absent from REPORT §1–§9 and ICLR | **Yes.** `acting=actor` scores 0.220 ± 0.037 vs `gpi` 0.054 ± 0.032 on the same checkpoints (COMPENDIUM §4.4). The actor *is* the result |
| `mix_ratio` task-vector sampling | `psmflow.py:155-161` | **yes, §10 `Task vectors`** | Yes — inherited from PSM/FB; never ablated here |
| `u_clip = 3` | `psmflow.py`, `configs/agent/psmflow.yaml:108` | **yes, §10 `Objects`** ("`‖u‖_∞ ≤ c`, we use `c=3`"); conflicts with REPORT's χ² ball | Yes — `psmflow.py:148-150` records that an unclipped `u_data` feeds the online branch inputs the tanh-bounded target branch can never produce |
| Ensemble `P=2` + κ-pessimism in the TD target and the actor | `psmflow.py:128,245-246`; `agents/psm.py:74-79` | **yes, §10** ("ensemble of size `P`", "mean minus `κ` standard deviations"); **absent from REPORT §1–§9 and from ICLR** — and ICLR's `main.tex:` prewriting §4 explicitly sells "get rid of additional loss regularization terms … tuned to the correct level of pessimism" | Yes, and **awkwardly so**: yaml:9-12 records that without target pessimism `actor_q` oscillated 450↔6500 on pointmaze. A method whose pitch is "no pessimism knob" ships exact min-Q |
| One-step distilled decoder | `psmflow.py:478-479` | **no** | Yes (default path), but worth +0.038 at most — §2.6 |
| `backup_explore_frac` | `psmflow.py:179-184` | no (it is a *knob toward* the REPORT) | No — default 0.0, and Arm A at 1.0 changed nothing |
| `action_critic` branch (`psi_a`, `Q_a`, FB graft, λ-rank) | `psmflow.py:268-334,303-334,529-548` | **no** | No for the headline PSMFlow row; **yes for the 10%-data row**, the one place the project beats everything (COMPENDIUM §4.2: hybrid 0.238 vs FB 0.030). At 100% data the residual's contribution is **−0.065** |
| `residual` head, `residual_eps=0.05` | `psmflow.py:254-266,363-382` | **no** — and it breaks the paper's central claim, since `a = clip(G(s,u) + ε·δ)` is *not* a flow decode | Same as above: sign-flipping, dataset-size dependent |
| `preimage_valid` / `repair_invalid_preimages` | `utils/flow_inversion.py:31-105` | no | Yes, mechanically — a single NaN latent NaNs the batch (`fql.py:29-38`) |
| Sidecar/checkpoint pairing guard | `main.py:85-156` | no | Yes, operationally |
| `eval_reward_shift=1.0`, `eval_relabel_size=10000` | `main.py:335-338` | no | Yes — `w = E[(r+1)φ]` is not `E[rφ]`; harmless only because `φ` is sphere-normalised and the shift is task-independent, but it is an unpapered choice |
| `psm_norm` on `φ` output | `utils/psm_networks.py:45-49,69-70` | no | Yes — interacts directly with the ortho term the paper *does* specify |

---

## 4. Claims and assumptions nothing checks

| unchecked claim | main.tex | nearest existing diagnostic | gap |
|---|---|---|---|
| **Prop. `insample`: `u'⊥s'` ⇒ `C(π_tr)=1`** | :807 | none | The headline separation from FB/PSM. Nothing measures the bootstrap-action distribution against `μ̂`. A two-sample test of `{G(s'_i,u'_i)}` vs `{a_i}` at matched states would settle it in an afternoon; under the default path it would **fail by construction** |
| **Def./Ass. `coherence` (ε-coherence)** | :469, :484 | `tools/eval_fixed_u_rollouts.py` (within-`u` vs across-`u` rollout distance), `tools/diag_policy_differentiation.py` | Neither builds the mode partition `{C_k(s)}` or the labelling `κ`, so neither produces the `ε` of eq. `coherence`. The paper calls this "falsifiable **before** any representation is trained" (Rem. `coherence-role`) — it never was. HANDOFF 2026-08-05 measured the fixed-`u` family non-goal-covering, which is the symptom `ε → (m−1)/m` predicts |
| **Ass. `exact` (`μ̂=μ`)** | :268 | `tools/validate_flow_inversion.py` typicality (χ², gate ≥0.95) — **the paper's own test**, Lem. `typicality` :727; supported by `diag_flow_fit.py`, `diag_generated_pair_support.py` | Best-covered assumption. But Conj. `compact` (:1213) says it is *unattainable* in the narrow-data regime the paper targets, and nothing quantifies the residual |
| **Ass. `affine` / `factorised`** | :548, :577 | none | Not implemented, so not checkable in `psmflow`; would require the `affine_psm` agent |
| **Prop. `isometry` / Cor. `mass` / Cor. `conc`** | :328, :383, :412 | none | Cor. `mass` is directly testable: decode `u ~ p₀` restricted to `U_δ` and compare the decoded-action distribution to the data at matched states. `tools/diag_action_coverage.py` is adjacent but answers a different question |
| **Cor. `resolution`: `|det J| = p₀(u)/μ̂(G(s,u)|s)`** | :443 | `tools/diag_flow_jacobian.py` computes `dG/du` spectra | The tool measures conditioning, never tests the identity |
| **Prop. `sf-identity` / Cor. `reward-identity`** | :639, :649 | none | A one-line residual probe (`‖ψ(s,u,u') − φ(s') − γψ(s',u',u')‖`) would give `ε_TD` empirically — the quantity Rem. `contraction-novelty` (:865) claims is estimable from data alone, which is the paper's stated advantage over FB |
| **Prop. `gpi`'s `ε` hypothesis** | :981 | `tools/calibration_check.py`, `diag_policy_ranking.py`, `diag_latent_ranking_oracle.py` | **Well covered, and it fails**: D1 Spearman 0.10, p=0.78, 0.9% value spread over policies spanning 4.7× in success (COMPENDIUM §4.7). Prop. `gpi`'s bound is vacuous at the measured `ε` |
| Ass. `lipschitz` | :204 | none | `layer_norm: false` on the flow trunk (yaml:100); no spectral norm, no bound |
| State-distribution shift (stated open) | :1247 | none | Acknowledged unaddressed by the paper |

---

## 5. Verdict — have we done everything the paper outlines?

**No, and the honest answer has three parts.**

*Faithful.* Stage A is the paper's flow-matching objective verbatim (`fql.py:338-346`) and
the decoder is genuinely frozen — no optimizer exists for it, and neither loss reaches it.
Stage B is the paper's ε-relaxed posterior with the correct Laplace covariance
(`fql.py:95,208-209`) since the 08-14 prior and 08-31 squared-norm fixes; the `prior_scale=1`
default *is* eq. `posterior`. Closed-form reward inference `w = E[rφ]` with sphere projection,
the orthonormality term that makes it least-squares-correct, the bilinear
`m = ψᵀφ`, the contrastive measure loss with its positive/negative split, Polyak targets, the
task-vector sampler, and `Q = ψᵀw` are all as written. `gpi_select` under
`policy_index=latent` is Alg. Rung 1 line for line. Everything the code does that the audit
brief flagged as suspicious — mix_ratio, u_clip, the ensemble, the pessimism, the actor and
its CFM anchor — **is in the write-up**, in §10 `LatentFlowPSM`, which was written to describe
the implementation.

*Deviates by design, with the paper's knowledge.* The free `ψ` in place of the affine
`b + Φᵀw` (Rem. `tradeoff` tabulates the trade and the forfeited LP inference). The
task-vector index slot and the actor's bootstrap latent — §10 states both, and both
contradict §1–§9's `ψ(s,u,u')` with `u'~p₀`, which is the version every theorem is proved
about. The document is therefore internally inconsistent, and the code implements the half
without theorems: **Prop. `insample`'s C=1, the paper's headline separation from FB and PSM,
does not apply to any run in `docs/tables/results.md`.** That is the sharpest way to state
the finding, and `psmflow.py:29-33` already says it.

*Missing.* Rung 2 (in-sample latent Q-iteration) is not implemented at all, so Prop. `rung2`'s
"dominates Rung 1" is untested and ICLR's *only* stated inference mechanism has no code.
`U_δ` is an L∞ box, not the χ² ball the mass guarantee is stated for, and no flag restores it.
Rung 1's projected gradient refinement is absent. Nothing measures ε-coherence as defined,
the concentrability coefficient, Cor. `mass`, or the SF Bellman residual `ε_TD` — the last
being the very quantity the paper claims is estimable from data alone as its advantage over
FB. And the deployed decoder is a distilled one-step net the paper never mentions.

*Where the honesty bites hardest.* The two deviations that were switchable have been run and
neither rescues the method: Arm A `0.171 ± 0.113` against a `0.220 ± 0.037` control, Arm B
`0.083 ± 0.191` against a `0.068` BC floor (COMPENDIUM §4.5). So "we did not implement the
paper" is **not** an explanation for PSMFlow's `0.220` versus FB's `0.721` — the paper's own
construction was implemented, run for 3 seeds × 500k steps, and scored at BC level. The three
things genuinely not yet tried are Rung 2, the χ² ball in place of the box, and a
500-episode mixture-vs-point comparison; of those, only the mixture arm speaks to a claim the
ICLR draft calls a core contribution.

---

### Housekeeping

1. Re-add `PAPER/` to the repo or pin the report's commit — this audit had to reconstruct the
   source of truth from `2007e65^`.
2. `PAPER/ICLR/content/method.tex` is 30 lines and predates every settled result in
   COMPENDIUM §4. Its Q-iteration equations are Rung 2, which does not exist in code.
3. `configs/agent/psmflow.yaml:106` says `use_point_preimage: false` while CLAUDE.md's
   documented launch line passes `true`. The config default is not what anything ran.

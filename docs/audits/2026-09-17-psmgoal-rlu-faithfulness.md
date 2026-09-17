# psmgoal ↔ RLU faithfulness audit (2026-09-17)

Read-only audit of the committed `agents/psmgoal.py` port of CalCharles/RLU's Proto Successor
Measure (PSM, arXiv 2411.19418) onto the frozen flow's latents. No code was changed.

## What was read

- Ours: `agents/psmgoal.py`, `utils/psm_networks.py` (`RLUMeasure`, `PolicyCoefficient`,
  `TripleMultiplier`, `TanhGaussianLatentActor`, `psm_norm`/`_L2`, `_triple_trunk`),
  `configs/agent/psmgoal.yaml`, `utils/psm_proto.py`, and the psmgoal branches of `main.py`
  (lines 74-97, 231-241, 377-385) and `tools/eval_checkpoint.py` (lines 371-376).
- RLU: `agent/psm.py` (continuous PSMAgent — the closest match to our signature),
  `agent/discrete_psm.py` (tabular PSM — the source of the seed table and the L2 coefficient),
  `agent/fb_modules.py` (`mlp`, `_L2`, `_nl`).
- Paper: arXiv 2411.19418 v2 HTML.

## Paper equation numbers (verified against v2 HTML)

The code and design comments cite "Eq. 5/6" for the successor-measure Bellman recursion. That
citation does not match the version I read.

| Claim in code | What the paper (v2) actually numbers it |
|---|---|
| successor-measure Bellman = "Eq. 5/6" | Bellman-flow constraint is **Eq. 2**; ∑ M = (1−γ)·1[s=s+,a=a+] + γ·∑P·M |
| basis parametrization M = φᵀw + b | **Corollary 4.2** (not an equation-5 span); φ, b independent of π |
| constrained coefficient program = "Eq. 10" | **Eq. 10** confirmed: max_{λ≥0} min_w −Φ·w·r − ∑_{s,a} λ(s,a)·min(Φw+b, 0) |

The "Eq. 5/6" label is a documentation error, not a code error. It appears in
`agents/psmgoal.py:176`, `utils/psm_networks.py:563`, and the design doc. The maths the code
implements matches Eq. 2 / Corollary 4.2 / Eq. 10; only the reference tag is wrong. Caveat:
arXiv numbering differs between versions; I could only read v2.

One substantive point from Eq. 10: the multiplier there is **λ(s,a)** — indexed by state-action
(the "row form"). Our code uses a per-triple `l(s,u,g)`. That is a deliberate departure from
Eq. 10 as written, and it matches RLU's own continuous implementation (`psm.py`), which also
uses a per-triple multiplier rather than the row form printed in the paper. See §3(b).

---

## 1. Component-by-component

`file:line` is our code; "RLU" is the reference file:line it ports.

| # | Component | Ours | RLU | Paper | Verdict | Note |
|---|---|---|---|---|---|---|
| 1 | φ(s,u,g), b(s,u,g) net | `psm_networks.py:549-567` (`RLUMeasure`), trunk `461-470` | `psm.py:167-196` (`PSM`; `mlp_phi` 174, `mlp_b` 178, `forward` 191) | Cor. 4.2 | DEVIATION | Same φᵀw+b head; but RLU uses two **separate** MLP trunks for φ and b, we share one trunk with two heads; our first activation is ntanh (LayerNorm+tanh), `psm.py` uses relu-only. |
| 2 | w(z) coefficient + √z sphere | `psm_networks.py:570-591` (`PolicyCoefficient`, L2 at 591) | `discrete_psm.py:298-300` (`…,"L2"`); `psm.py:233-235` (no L2); `_L2` `fb_modules.py:33-40` | Cor. 4.2 | FAITHFUL | Matches `discrete_psm` L2 sphere √d·x/‖x‖. `psm.py` continuous does NOT L2 the training coefficient; we follow the discrete form (this is what makes the scale-anchor argument hold — §3a). |
| 3 | basis loss: off-diag squared TD | `psmgoal.py:181-183` | `discrete_psm.py:431-432`; `psm.py:350-351` | Eq. 2 | FAITHFUL | 0.5·mean over off-diagonal of (M − γ·target_M)². Mesh = all (row i)×(goal j). |
| 4 | basis loss: −(1−γ) diagonal pull | `psmgoal.py:184` | `discrete_psm.py:433`; `psm.py:352` | Eq. 2 | FAITHFUL | −(1−γ)·mean(diag(M)); occupancy pull toward own next state. |
| 5 | state×goal mesh | `psmgoal.py:133-146` (`mesh_M`) | `discrete_psm.py:374-384`; `psm.py:313-318` | — | FAITHFUL | goals = batch next_obs; RLU uses `next_goal` (= next_obs when goal_space is None, as on cube). |
| 6 | Polyak targets | `psmgoal.py:200-203`; init `114-115` | `discrete_psm.py:619-622`; `psm.py:478-481` | — | FAITHFUL | τ=0.01 on both basis and coefficient targets; matches soft_update of psm and w. |
| 7 | fixed z-indexed bootstrap policy | `psmgoal.py:149-160`; `psm_proto.py` | `discrete_psm.py:137-171` (`SamplingSeedActor`; max_seed 144, seed_long 162, gather 169) | — | DEVIATION (intended) | Seed arithmetic (powers reversed, +20000, keyed on dataset row) copied verbatim; the action emitted is a **clipped-normal latent decoded by the frozen flow**, not a table action. This is the LatentFlowPSM adaptation. |
| 8 | bootstrap target uses target nets | `psmgoal.py:174-175` (`M_bar`, stop_grad) | `discrete_psm.py:391-416` | Eq. 2 | FAITHFUL | φ_target·w_target + b_target at (next_obs, u_next, goals). |
| 9 | inference objective | `psmgoal.py:247` (`_obj_and_constraint`) | `psm.py:541` (`_infer_step_gc`) | Eq. 10 | DEVIATION | RLU obj = −φ_g·w_inf (no b). Ours = mean(φ·w + b). b does not depend on w, so the w-gradient is identical; only the logged value differs. Benign (see §2). |
| 10 | inference constraint sample | `psmgoal.py:248-255` | `psm.py:539,545-546` | Eq. 10 | DEVIATION | RLU permutes the **goal** against fixed (s,a). Ours permutes **(s,u)** against the fixed goal set. Both build random off-support triples; the distribution differs. Worth-watching. |
| 11 | per-triple multiplier l(s,u,g) | `psm_networks.py:528-537` (`TripleMultiplier`); use `psmgoal.py:251-255` | `psm.py:500-501` (lmult out-dim 1, "soft") | Eq. 10 uses λ(s,a) | FAITHFUL to RLU | Softplus scalar per triple. Matches `psm.py` (not `discrete_psm`, whose lmult is action-dim = row form). §3b. |
| 12 | dual ascent on multiplier | `psmgoal.py:280-291` | `psm.py:559-565` | Eq. 10 | FAITHFUL | Minimize mean(cons·l) with cons detached ⇒ raises l where cons=φw+b<0. §3b. |
| 13 | w sphere renorm each step | `psmgoal.py:236-239` (`_project`), applied `277` | `discrete_psm.py:662,698`; `psm.py` normalizes w_inf **only inside** the einsum / not at all in `_infer_step_gc` | Eq. 10 (w on √d sphere) | DEVIATION | We project the actual parameter onto √d every step. `discrete_psm` normalizes on-the-fly and once at the end; `psm.py` continuous GC-inference does neither. Cleaner but not literal. Benign. |
| 14 | acting = gpi argmax | `psmgoal.py:326-332` (`select_latent`), `318-324` (`goal_Q`) | none in `psm.py` (acts via ddpg_actor); `discrete_psm.py:794-801` argmax over discrete actions | — | DEVIATION (our addition) | argmax of goal-averaged Q over `gpi_num_u` clipped prior latents — the continuous analog of the tabular argmax. |
| 15 | acting = distill | `psmgoal.py:346-369` (`distill_actor`), `339-341` | `psm.py:569-603` (`update_actor`, `distill_actor_ddpg`) | — | DEVIATION | Q-maximizer with temp·logp − Q kept; but DDPG→`TanhGaussianLatentActor`, default temp=0 (no entropy), no flow-BC anchor, no SAC α. §2. |
| 16 | eval goal-set inference | `psmgoal.py:372-400` (`infer_eval_goals`); `main.py:377-385`; `eval_checkpoint.py:371-376` | `discrete_psm.py:882-917` (`inference`, `psm_set`) | Eq. 10 | DEVIATION | Goal set = k_goals rewarding next states (faithful). But the (s,u) the Lagrangian runs on use **prior-drawn latents u**, not dataset preimages/actions. §2. |

---

## 2. Deviations, each judged

**D1 — Shared φ/b trunk vs RLU's two separate MLPs.** `RLUMeasure` (`psm_networks.py:564-566`)
runs one `_triple_trunk` and reads φ and b from two linear heads. RLU `PSM`
(`psm.py:174-193`) builds `mlp_phi` and `mlp_b` as fully independent 3-hidden-layer trunks.
Also our default depth is `hidden_layers=2` vs RLU's three hidden layers plus the feature
layer. Judgment: **worth-watching.** Sharing couples the bias b to the basis φ; if b absorbs
scale that φ then cannot, the two heads can fight. Fewer parameters than RLU. Not obviously
wrong, but it is a real architectural change from the reference.

**D2 — First activation ntanh vs irelu.** Our `_triple_trunk` and `PhiMap` open with
Dense→LayerNorm→tanh; RLU's continuous `PSM` uses relu throughout (`psm.py:174-176`), while
RLU's `discrete_psm` `PSM` does open with ntanh (`discrete_psm.py:244`). Judgment: **benign.**
LayerNorm-tanh is a standard input stabilizer and is used across the RLU codebase (Actor,
BackwardMap, ForwardMap all open with ntanh); the continuous PSM is the exception, not the rule.

**D3 — Single critic.** `RLUMeasure` is a single, non-ensembled net. This is **FAITHFUL**, not a
deviation: RLU's `PSM` is itself a single critic (only FB uses twin F1/F2). No min-over-ensemble
pessimism exists in RLU PSM either. Judgment: **benign / faithful.** Recorded because the task
flagged it; there is nothing to reconcile.

**D4 — Inference (s,u) from prior-drawn latents, not dataset preimages.** `infer_eval_goals`
(`psmgoal.py:389-395`) samples real states s from the relabel batch and draws u from the clipped
flow prior. RLU runs the Lagrangian on dataset (s, a) (`psm.py:512-518`). The stated rationale
(`psmgoal.py:386-388`): the frozen flow decodes the prior to the data-action distribution, so
E_{(s,u)}[φ·w] matches RLU's dataset objective in expectation and both eval entry points stay
identical without a preimage load. Judgment: **worth-watching.** The equality holds in
expectation over u but the per-triple constraint set is different — the non-negativity is being
enforced on prior-typical latents rather than on the exact recorded actions. If the flow's
support and the dataset action support diverge, the constraint is enforced on the wrong triples.
Cheaper and simpler; the risk is that the constraint no longer pins the recorded transitions.

**D5 — Objective includes b.** Item 9. RLU's objective is −φ_g·w (no bias); ours is
mean(φ·w + b). Since b(s,u,g) has no w-dependence, ∂/∂w is identical and the w update is
unchanged. Judgment: **benign.** Only the logged `obj` scalar differs.

**D6 — Constraint permutes (s,u) not goals.** Item 10. Judgment: **worth-watching.** Same intent
(random off-support triples), different sampling law. On a goal set of size k with a batch of n,
RLU makes n random (s,a,goal) pairs by permuting the goal; we make n pairs by permuting (s,u)
against every one of the k goals (k·n_sub constraint terms). The multiplier network sees a
different input distribution than RLU's. Unlikely to break, but it is not the reference's
negative-sampling scheme.

**D7 — w projected every step vs RLU's normalize-inside/final.** Item 13. Judgment: **benign.**
Projecting the parameter each step and normalizing inside the objective each step are equivalent
up to the Adam moment buffers seeing a projected vs unprojected iterate; both keep ‖w‖=√d in the
objective. `psm.py`'s continuous GC path does not renormalize at all, so if anything our version
is closer to the discrete reference's intent than `psm.py` is.

**D8 — eps inside sqrt vs torch F.normalize.** `PolicyCoefficient` (`psm_networks.py:591`) uses
`√d · h / sqrt(Σh² + 1e-8)`; RLU `_L2` uses `√d · F.normalize(h)` = `√d · h / max(‖h‖, 1e-12)`.
At h=0 both return 0; ours has a finite Jacobian there, torch's is a defined-to-0 subgradient.
For nonzero h the 1e-8-inside vs 1e-12-clamp difference is numerically negligible. Note `psm_norm`
(`psm_networks.py:49-53`, used for φ elsewhere and for `_project`) uses the max-clamp 1e-12 and so
matches torch exactly; only `PolicyCoefficient` uses eps-inside. Judgment: **benign** (arguably a
safety improvement, since the all-zero policy code z=0 gives enc(z)=0).

**D9 — distill actor is a bare Q-maximizer.** Item 15. `distill_actor` minimizes temp·logp − Q
with a reparameterized `TanhGaussianLatentActor` draw. RLU's `update_actor` (`psm.py:569-592`) is
the same shape but with a DDPG SquashedNormal actor and temp=1. Our default `actor_temp=0.0`
removes the entropy term entirely, and there is no flow-BC anchor and no SAC dual-α. Judgment:
**worth-watching.** With temp=0 and no BC anchor, nothing keeps the distilled actor's latents in
the flow's support; it can drift to off-manifold latents that maximize a critic evaluated where
it was never trained. This is the same failure mode `TanhGaussianLatentActor.prior_init`
(`psm_networks.py:193-200`) documents for the DSRL line. `acting=distill` is off by default
(`acting=gpi`), which sidesteps it, but the distill path as written is the least faithful and
riskiest component.

**D10 — config values.** Closest RLU reference is `psm.py` (continuous); `discrete_psm` supplied
the L2 coefficient and the seed table.

| Knob | RLU `psm.py` | RLU `discrete_psm` | Ours | Judgment |
|---|---|---|---|---|
| basis dim d_dim | 50 | 50 | z_dim=128 | worth-watching (2.5× capacity) |
| policy-code width | z_dim=50 | z_dim=16 | max_log_seed=8 (2⁸=256 policies) | worth-watching (much smaller proto family) |
| batch_size | 32 (mesh 1024) | 16 (mesh 256) | 256 (mesh 65536) | benign (more negatives/step, more compute) |
| basis hidden_dim | 1024 | 1024 | measure 512 | worth-watching (narrower) |
| coef/mult hidden | 1024 / 1024 | 1024 / 256 | 256 / 256 | benign (mult matches discrete's 256) |
| lr (basis) | 1e-4 | 1e-4 | 1e-4 | faithful |
| lr coefficient | lr·lr_coef=1e-4 | 1e-4 | 1e-4 | faithful |
| lr_w (inference) | 1e-4 (uses `lr`) | 3e-4 | 3e-4 | matches discrete |
| lr_l (multiplier) | 1e-4 (uses `lr`) | 3e-4 | 3e-4 | matches discrete |
| tau | 0.01 | 0.01 | 0.01 | faithful |
| num_inference_steps | 5120 | 20000 | 20000 | matches discrete |
| inf_coeff | 5.0 | 5.0 | 5.0 | faithful |
| num_actor_steps | 10000 | — | 10000 | faithful |
| actor temp | 1.0 | — | 0.0 | deviation (D9) |

Our config is a coherent blend: basis loss, mesh and coefficient follow `discrete_psm`; the
per-triple multiplier and the actor distillation follow `psm.py` (continuous). The one config
value that changes behaviour rather than capacity is `actor_temp=0.0` (D9); it only matters under
`acting=distill`.

---

## 3. The two claims that matter

**(a) The basis loss cannot blow up without a scale anchor.** Confirmed; the code realizes the
RLU mechanism. `RLUMeasure` deliberately drops the two anchors that `_TripleTower` carries: no
`psm_norm` on φ and no `tanh` bound on b (`psm_networks.py:565-566`). What keeps the measure
finite instead:

1. The off-diagonal term regresses M(s,u,g) to a **stopped, discounted** target,
   γ·target_M with γ=0.98 < 1 (`psmgoal.py:175,182`). target_M is `stop_gradient` and computed
   at the Polyak-lagged params, so per step it is a fixed, bounded regression target. Because the
   target is γ times a same-scale quantity, the TD map is a contraction; its fixed point is the
   finite successor measure M = (1−γ)·1[·] + γ·P·M (Eq. 2), not a growing one.
2. The coefficient is pinned to the sphere ‖w‖=√z_dim every forward pass
   (`PolicyCoefficient`, `psm_networks.py:591`), so the only way to inflate |M| = |φ·w + b| is to
   grow ‖φ‖ or |b|.
3. The −(1−γ)·mean(diag M) term is linear in the diagonal and, alone, would reward growing φ
   without bound. It does not, because the same φ(s_i,u_i,·) sets the off-diagonal entries too:
   raising the diagonal raises the off-diagonals, which the squared contraction term penalizes.
   The equilibrium is the finite Bellman-flow fixed point.

So the fixed coefficient norm plus the γ-contraction to a bootstrapped target replace the
head-level anchor. This is exactly why the contrastive `_TripleTower` needs its `psm_norm`+`tanh`
(its −2·M(s,u,s′) term is unbounded below, `psm_networks.py:478-482`) and the RLU TD form does
not. The code matches the reference in dropping the anchor. One caveat specific to our port: the
guarantee leans on the coefficient being norm-fixed during **training**, which we take from
`discrete_psm` (item 2); RLU's continuous `psm.py` does not L2 its training coefficient, so the
argument is actually stronger for our code than for `psm.py` itself.

**(b) The multiplier is genuinely per-triple, and dual ascent raises it on violated triples.**
Confirmed on both halves.

- Per-triple: `TripleMultiplier` (`psm_networks.py:528-537`) is a scalar softplus head on the
  concatenated triple [s, u, g]; in `_obj_and_constraint` (`psmgoal.py:251-255`) it is evaluated
  at every (permuted sample × goal) pair and reshaped to (n, k). This matches RLU `psm.py`'s
  `lmult` (out-dim 1 over [obs, action, goal], `psm.py:500-501,545`). It is **not** the row form:
  `discrete_psm`'s lmult has out-dim = action_dim (`discrete_psm.py:639-640`), and paper Eq. 10 as
  printed uses λ(s,a). Ours follows the continuous per-triple form.
- Direction: the w-step penalty is `pen = −mean(cons · stop_grad(l))` with
  cons = φ·w + b (`psmgoal.py:270`), so descending −obj+pen pushes cons upward wherever l>0
  (enforces non-negativity). The multiplier step minimizes `mean(cons_sg · l)` with cons detached
  (`psmgoal.py:283-287`); ∂/∂(l-params) of cons·l is cons, so gradient descent moves l opposite to
  cons: it **increases l where cons = φw+b < 0** (violated) and decreases it where cons ≥ 0, with
  softplus flooring l at 0. This is RLU's dual ascent (`psm.py:559-565`) sign-for-sign. The logged
  `viol_frac`/`viol_mean` (`psmgoal.py:293-294`) track the violated set directly.

---

## 4. Planned-but-unimplemented goal-conditioned modifications (design-level)

These live in `utils/psm_networks.py:453-525` (`_TripleTower`, `TripleMeasure`,
`GoalCoefficient`) and the design doc but are not wired into `agents/psmgoal.py`. The committed
agent uses `RLUMeasure` + `PolicyCoefficient` + the Lagrangian inference; the goal-conditioned
head below is scaffolding.

**Amortized w\*(g) = h(g)/‖h(g)‖ head trained by J(θ|g).** `GoalCoefficient`
(`psm_networks.py:513-525`) is the intended amortized coefficient: a goal encoder projected to the
√z_dim sphere, meant to be trained by the same Eq. 10 objective J(θ|g) (maximize E φ·w\*(g)) with
the non-negativity penalty, so inference over goals is a forward pass instead of a 20000-step
optimization. Faithfulness: this is the FB `BackwardMap` pattern (`fb_modules.py:213-232`: MLP →
√z·F.normalize) reused as a reward/goal encoder, which is standard. **Risk, as flagged:** if the
non-negativity penalty is dropped from J(θ|g), the objective collapses to max E φ·w\*(g), i.e.
w\*(g) becomes plain FB f(g) — the amortized head then learns the FB forward map with none of the
PSM constraint, and the "PSM" reduces to FB. The constraint is the only thing that distinguishes
the amortized PSM coefficient from FB; it must be kept in J(θ|g).

**Conditioning the DSRL actor on w\*(g).** Planned: feed w\*(g) as the actor's z input so the
policy is explicitly goal-conditioned rather than re-distilled per goal. Faithfulness: this is how
FB/DSRL condition a policy on a task vector (`TanhGaussianLatentActor.__call__(obs, z)` already
takes a z slot, `psm_networks.py:208-226`); mechanically sound. Risk: inherits D9 — without a
flow-BC anchor the conditioned actor can still leave the flow's support; goal-conditioning changes
which latent is optimal, not whether it is in-support.

**Full SAC + flow-BC DSRL actor.** Planned replacement for the bare Q-maximizer (D9): add the SAC
dual-α entropy term and a flow-BC anchor pulling the actor toward the prior/decoded data actions.
Faithfulness: this is the repo's own DSRL line (the `prior_init`, `layer_norm`, `LogAlpha`
machinery in `psm_networks.py:171-258` already exists for it), not RLU's DDPG distill. It departs
from RLU but toward the repo's established, more stable actor; the risk is the opposite of D9
(over-anchoring to BC can cap performance at the BC control, the failure the DSRL notes in MEMORY
record). Reasonable, but it is a repo choice, not an RLU port.

**OGBench/FB on-trajectory + off-trajectory goal mixture.** Planned: build the training/eval goal
distribution as a mixture of goals reached later on the same trajectory (on-trajectory / hindsight)
and random dataset goals (off-trajectory), per FB and OGBench practice. Faithfulness: matches FB's
hindsight+random z-sampling and OGBench's goal-sampling convention. The committed agent does not do
this — it takes the goal set purely from rewarding relabel next states (`psmgoal.py:378-384`) and
uses the batch's own next states as basis-training goals (`psmgoal.py:170`), with no hindsight
future-goal sampling. Risk: without on-trajectory goals the basis is trained only on one-step next
states as goals, so long-horizon reachability is learned only through bootstrapping; adding the
mixture is the standard fix and low-risk, but it changes the goal distribution the basis sees and
would need re-tuning.

---

## 5. Verdict

The committed `agents/psmgoal.py` is a faithful replica of RLU's continuous Proto Successor
Measure on the numerically load-bearing parts: the basis loss (off-diagonal squared TD to a
discounted stopped target plus the −(1−γ) diagonal pull), the √z-sphere coefficient, the Polyak
targets, the fixed seed-indexed bootstrap policy (verbatim seed arithmetic, adapted to emit a
flow latent), the per-triple softplus multiplier, and the Eq. 10 dual ascent — all match RLU
sign-for-sign, and both verification claims hold. The single critic is faithful (RLU PSM is also
single). The departures are a mix of blends (basis/mesh from `discrete_psm`, multiplier/actor from
`psm.py`), one documentation error (the "Eq. 5/6" tag should read Eq. 2 / Corollary 4.2 / Eq. 10),
and a few real behavioural changes. Top risks, in order: (1) the `acting=distill` actor is a bare
Q-maximizer with temp=0, no entropy and no flow-BC anchor, so it can chase off-manifold latents —
mitigated only because the default is `acting=gpi`; (2) inference enforces the non-negativity
constraint on prior-drawn latents rather than the recorded preimages, which pins the wrong triples
if flow support and data-action support diverge; (3) the shared φ/b trunk and the narrower/larger
config knobs (hidden 512 vs 1024, z_dim 128 vs 50, 2⁸ vs 2¹⁶ proto policies) change capacity from
the reference; and (4) for the unbuilt amortized head, dropping the constraint from J(θ|g) would
silently reduce PSM to FB.

# Affine (full) PSM — implementation spec (affine successor measure + constrained goal inference)

**Date:** 2026-07-22 · **Branch:** `feat/psm-integration` · **Design source:**
RLU reference `controllable_agent/url_benchmark/agent/{psm.py,discrete_psm.py}`
(continuous + discrete full PSM), cross-checked against `PAPER/RESEARCH_NOTE.md` §3
(affine decomposition, Prop. 5 "affine span"). User-approved design decisions:
**Path A** (faithful affine factorization), **new agent** (existing `psm.py` untouched),
**goal-conditioned** task interface, `full` + `zero_shot` inference modes, purpose =
**foundation for `psmflow`** (the affine representation + constrained-inference machinery
that the flow-indexed variant borrows conceptually).

## 0. Motivation and what exists today

The current `agents/psm.py` is a parity-audited JAX port of the **bilinear / FB-style**
successor measure $M(s,a,x) = \psi(s,z,a)^\top \varphi(x)$ with **closed-form** reward
inference $w = \mathbb E_{\mathcal D}[r\,\varphi]$. That is the *zero-shot* PSM. It does
**not** implement the paper's *general* affine form
$M^\pi(s,a,x) = \Phi(s,a,x)^\top w + b(s,a,x)$ or the constrained-optimization inference
that defines the "full" PSM. Those are implemented only in RLU and were scoped as
"Phase 2" in `2026-07-17-psm-refactor-and-paper-faithful-design.md` (deviations D1/D2).

**Why the affine form is a different network, not a bias bolt-on.** RLU's constrained-LP
inference works *only because $M$ is linear in the task coordinate $w$* ($\Phi, b$ fixed →
maximize-reward-at-goal subject to $\Phi w + b \ge 0$ is a genuine program in $w$). The
bilinear form is not linear in the policy code, so it admits only closed-form inference.
"Full PSM with the affine transformation" therefore means **adopting RLU's affine
factorization**, which is architecturally distinct from the current agent.

## 1. Goal and success criteria

Build a reward-free **affine successor-measure** representation
$M(s,a,x) \approx \Phi(s,a,x)^\top w + b(s,a,x)$ over the hash-codebook policy family
$\pi_z$, trained by the PSM contrastive TD loss, with two inference modes selectable by
config:

- **`full`** — constrained goal-conditioned inference (dual gradient descent / hinge)
  solving $w_{\text{inf}}$, then a distilled continuous actor for action selection.
- **`zero_shot`** — closed-form goal code (no test-time optimization), same actor path.

**Success gates:**

- **Unit gate:** all TDD tests green (§7), including the affine-in-$w$ property and a
  monotone drop in mean constraint violation during `full` inference on a fixture.
- **Smoke gate:** `affine_psm` runs end-to-end through `main.py` on a goal-conditioned
  OGBench env (start: `pointmaze-medium-navigate-v0`, then `cube-single-play-v0`) for
  both modes, logging to wandb without shape/NaN errors.
- **Sanity gate (not a parity claim):** on `pointmaze-medium`, `full` inference drives
  mean constraint violation $\to 0$ and produces non-trivial goal-reaching success
  (> chance); `zero_shot` runs and is no worse than a random-$w$ control. Numeric parity
  with RLU is explicitly **not** a goal.

## 2. Architecture overview

Two artifacts, one training stage + one inference stage:

```
Stage A: representation (reward-free)          Stage B: per-task inference (eval time)
agents/affine_psm.py                           infer_w_goal (constrained)  ── full
hash codebook π_z + contrastive TD             OR closed-form goal code    ── zero_shot
→ trains Φ, b (AffineMeasureNet) and w(z)      → w_inf → distill actor → evaluate()
```

The existing bilinear `agents/psm.py` is **not modified**. The new agent reuses the
repo's `flax.struct.PyTreeNode` + one-`TrainState`-per-network convention, the hash
codebook proto sampler, the contrastive/ortho loss helpers, and the actor infrastructure.

### 2.1 Networks (`utils/psm_networks.py`, new classes)

- **`AffineMeasureNet(s, a, x) → (phi ∈ R^d, b ∈ R)`** — MLP on `concat[s, a, x]` with a
  shared trunk and two heads (`phi_fc → R^d`, `b_fc → R^1`), mirroring RLU `PSM`
  (`psm.py:167-196`). `x` is the measure argument (a future state / goal). Ensembled
  (size 2) to reuse the target-uncertainty machinery, matching `PsiMap`; the ensemble is
  min-reduced only where RLU uses a single net (document the deviation inline).
- **`w`: MLP `z → R^d`** (training-time task coordinates for codebook code `z`) with a
  soft-updated `w_target` (RLU `psm.py:233-241`).
- **`w_inf`: free `R^d` parameter**, re-initialized and solved at inference
  (RLU `psm.py:485-505`).
- **`lmult`: MLP `concat[s,a,x] → R^1` (softplus head)** — learned Lagrange multiplier
  for dual gradient descent, created only when `inference.use_dgd` (RLU `psm.py:499-505`).
- **Codebook proto policy:** reuse the existing hash-codebook sampler from `psm.py`
  (`proto`, `proto_sample`) — the same uniform-action $\pi_z$ as RLU `SamplingSeedActor`.
- **Actor:** reuse the existing `PSMActor` (DDPGBC) path for distillation; a `flow`
  actor is out of scope for v1.

Naming: `x` is the measure/goal argument throughout (avoid the `goal`/`obs` collision in
RLU where `goal` is the measure argument and `obs` the current state).

### 2.2 Measure and Q

$$
M(s,a,x) = \Phi(s,a,x)^\top w + b(s,a,x), \qquad
Q(s,a;x) = \Phi(s,a,x)^\top w_{\text{inf}} + b(s,a,x).
$$

`w(z)` supplies $w$ during training (per codebook code); `w_inf` supplies $w$ at
inference. Both are $d$-dimensional. No implicit reshape to `action_dim` (RLU's discrete
`action_dim × d` head is a discrete-action artifact; continuous control uses a plain
$d$-vector `phi` and scalar `b`, as in RLU continuous `psm.py`).

## 3. Training (`update`, jitted)

Faithful to RLU `update_psm` (`psm.py:301-405`):

1. Sample a batch $\{(s_i, a_i, s_i')\}$ and a **single** binary codebook code $z$ for
   the batch (RLU samples one $z$ and repeats it, `psm.py:431-432`); get the continuation
   actions $a'_i = \pi_z(s_i')$ from the hash codebook sampler.
2. Form the $(s_i,\, x_j = s_j')$ mesh (all pairs) for the contrastive diagonal /
   off-diagonal split.
3. Predictions and targets:
   $$
   M_{ij} = \Phi(s_i, a_i, s_j')^\top w(z) + b(s_i, a_i, s_j'), \quad
   \bar M_{ij} = \bar\Phi(s_i', a_i', s_j')^\top \bar w(z) + \bar b(s_i', a_i', s_j').
   $$
4. Loss (RLU `psm.py:351-353`):
   $$
   \mathcal L = \tfrac12\,\big\|(M - \gamma \bar M)\odot \text{offdiag}\big\|^2_{\text{mean}}
   \;-\; (1-\gamma)\,\overline{\mathrm{diag}(M)} \;+\; \lambda_\perp \mathcal L_{\text{ortho}}(\Phi).
   $$
   The ortho term is **optional** (RLU comments it out; keep it behind
   `ortho_coef`, default `0.0` to match RLU, exposed for ablation).
5. One optimizer step over `{AffineMeasureNet, w}` (joint, RLU `psm.py:254-256`);
   Polyak-update `AffineMeasureNet_target` and `w_target` at rate `fb_target_tau` (0.01).

Single `TrainState` per network (`measure`, `w`), matching the repo's post-refactor
convention. `total_loss` returns the same named-loss dict style as `psm.py` for the
validation logging path in `main.py`.

## 4. Inference modes

Config block `inference:` with `mode: full | zero_shot` (plus the knobs below). Both
modes end by producing `w_inf` and then selecting actions via a distilled actor.

### 4.1 `full` — constrained goal-conditioned inference

Port of RLU `infer_w_goal` + `_infer_step_gc` (`psm.py:507-567`). Given a goal state
$g$, iterate for `num_inference_steps` over dataset batches:

- $\Phi_g, b_g = \text{measure}(s, a, g)$; $\Phi_\perp, b_\perp = \text{measure}(s, a,
  \tilde x)$ for permuted future states $\tilde x$ (validity anchors).
- **Objective** (maximize measure at goal): $\text{obj} = -\overline{\Phi_g^\top w_{\text{inf}}}$.
- **Constraints** ($M \ge 0$ off-goal):
  - `use_dgd`: $-\overline{(\Phi_\perp^\top w_{\text{inf}} + b_\perp)\,\lambda(s,a,\tilde x)}$,
    then an ascent step on $\lambda$ (dual gradient descent, `psm.py:559-565`).
  - else (hinge): $-\overline{\min(\Phi_\perp^\top w_{\text{inf}} + b_\perp,\,0)\cdot \text{inf\_coeff}}$.
- Adam step on `w_inf` w.r.t. `obj + constraints`.
- After the loop, normalize $w_{\text{inf}} \leftarrow \sqrt d \cdot w_{\text{inf}}/\|w_{\text{inf}}\|$
  (RLU discrete `discrete_psm.py:662`; RLU continuous omits it — we adopt the normalize
  for scale stability and note the deviation).

### 4.2 `zero_shot` — closed-form goal code (no test-time optimization)

The measure argument is the goal itself, so a valid measure that concentrates on $g$ is
obtained in one shot without the LP. v1 definition: set
$w_{\text{inf}} = \sqrt d \cdot \hat w/\|\hat w\|$ with
$\hat w = \mathbb E_{(s,a)\sim\mathcal D}\big[\Phi(s,a,g)\big]$ — the mean basis response
toward the goal, the affine-network analogue of FB's `get_goal_meta` ($z = B(g)$).
This preserves the "no per-task optimization" character of the current agent as a
selectable mode. (Reward-based closed form $w=\mathbb E[r\varphi]$ is **not** revived
here; it belongs to the bilinear `psm.py` and to the deferred reward interface, §8.)

### 4.3 Actor distillation (shared by both modes)

Port of RLU `distill_actor_ddpg` / `update_actor` (`psm.py:569-603`). After `w_inf` is
fixed, train the `PSMActor` for `num_actor_inference_steps` to maximize
$Q(s,\pi(s); g) = \Phi(s,\pi(s),g)^\top w_{\text{inf}} + b(s,\pi(s),g)$ over dataset
states (entropy-regularized DDPG objective). `act(s)` returns the actor mean. Distillation
is required because the continuous $\arg\max_a Q$ has no closed form (unlike discrete
`argmax`).

## 5. Config (`configs/agent/affine_psm.yaml`)

New file (do not edit `psm.yaml`). Keys, RLU-sourced defaults:
`agent_name: affine_psm`, `d_dim: 50`, `z_dim: 50`, `hidden_dim: 1024`, `feature_dim:
512`, `lr: 1e-4`, `lr_coef: 1.0`, `fb_target_tau: 0.01`, `ortho_coef: 0.0`,
`batch_size` (per repo default), and an `inference:` block:
`mode: full`, `use_dgd: true`, `inf_coeff: 5.0`, `num_inference_steps: 5120`,
`lr_w: 1e-4`, `num_actor_inference_steps: 10000`, `norm_w: true`. Actor sub-block reuses
`psm.yaml`'s DDPGBC actor knobs. Registered in `agents/__init__.py` as
`affine_psm=AffinePSMAgent`.

## 6. Integration (`main.py`, `utils/evaluation.py`)

Goal-conditioned eval is the only new harness work. Gated on the agent exposing an
`infer_w_goal(dataset, goal)` / `distill_actor(dataset, goal)` interface, so the existing
reward-based `infer_eval_z` branch (used by `psm.py`) is untouched.

- **Goal extraction:** OGBench goal-conditioned envs expose the goal observation via
  `env.reset(options=...)` / `info['goal']`. Add a small helper in `utils/evaluation.py`
  that, for goal-conditioned agents, resets the eval env per episode/task and returns the
  goal observation to condition the agent on.
- **Eval branch in `main.py`:** if `hasattr(agent, 'infer_w_goal')`, for each eval task:
  (1) get the goal $g$ from the env; (2) `agent2 = agent.infer_w_goal(train_dataset, g)`
  (solves `w_inf`); (3) `agent2 = agent2.distill_actor(train_dataset, g)`; (4) run
  `evaluate` with `agent2`. Re-seed the eval env like the existing path for reproducible
  success.
- **`evaluate`:** no signature change. Because `w_inf` and the goal are already baked
  into the distilled actor, `sample_actions(obs)` selects goal-directed actions exactly
  as today. The only `evaluate` addition is optional per-task goal reset when the env is
  goal-conditioned (via the goal-extraction helper above).

Deviation from RLU: RLU solves `w_inf` once per goal and mutates the agent in place; the
JAX agent is a frozen pytree, so `infer_w_goal` / `distill_actor` return **new** agent
copies (functional style), consistent with `infer_eval_z` in `psm.py`.

## 7. Testing (TDD, `tests/`)

Write tests first, one per behavior:

1. **Shapes:** `AffineMeasureNet(s,a,x)` returns `phi:(B,d)`, `b:(B,1)`; ensemble axis
   handled.
2. **Affine-in-$w$ property:** at fixed `(s,a,x)`,
   $M(w_1) - M(w_2) = \Phi^\top(w_1 - w_2)$ within tol (the property the LP relies on).
3. **Training step:** one `update` on a fixed fixture decreases `psm_loss`; targets are
   stop-gradient'd; `w` and `measure` both receive gradient.
4. **Constrained inference:** over `full` inference steps on a fixture, mean constraint
   violation $\overline{\min(\Phi_\perp^\top w_{\text{inf}} + b_\perp, 0)}$ is
   non-increasing and ends near 0; `use_dgd=true` and hinge both tested.
5. **Zero-shot mode:** returns finite, $\sqrt d$-normalized `w_inf`; deterministic given
   inputs.
6. **Actor distillation:** distilled actor's mean $Q$ increases over distillation steps.
7. **Smoke E2E:** `affine_psm` trains for a handful of steps and runs one goal-conditioned
   eval episode on a tiny env through `main.py` plumbing, both modes, no NaNs.

## 8. Scope / non-goals (YAGNI)

- **In:** faithful goal-conditioned affine PSM (`AffineMeasureNet`, `w(z)`, contrastive
  TD training), `full` (DGD + hinge) and `zero_shot` inference modes, DDPGBC actor
  distillation, goal-conditioned OGBench eval integration, TDD suite.
- **Out (deferred, later specs):**
  - **Reward-based *full* inference** (the reward-weighted constrained objective) — RLU
    leaves it `NotImplementedError`; v2.
  - The **`psmflow`** flow-indexed agent itself — separate plan
    (`2026-07-20-psmflow-v1.md`); this spec only builds the affine substrate it borrows.
  - **Flow actor** for distillation — reuse DDPGBC only in v1.
  - **`pos_neg`** goal inference (RLU `_infer_step_pos_neg`).
  - **Numeric/bit parity** with RLU — the user chose "works on OGBench as a mode."
  - Encoder / pixel observations (the agent asserts `encoder is None`, as `psm.py` does).

## 9. Open risks

1. **Goal-conditioned eval plumbing** is the least-tested surface; OGBench goal APIs vary
   by env family (`pointmaze`/`antmaze` vs `cube`). Mitigation: land `pointmaze-medium`
   first, add `cube` once the goal-extraction helper is proven.
2. **Inference cost:** `num_inference_steps=5120` + `num_actor_inference_steps=10000`
   *per eval task* is expensive. Mitigation: expose both as config; use small values in
   the smoke gate; profile before the full sanity run.
3. **Ensemble vs single net:** RLU uses a single measure net; our ensembled port must not
   silently change the loss. Mitigation: default to reproducing RLU's single-net
   reduction and cover the choice in a test.
4. **`zero_shot` definition** ($\hat w = \mathbb E[\Phi(s,a,g)]$) is a design choice, not
   an RLU port (RLU's zero-shot is a *separate* FB agent). Flagged for user review.

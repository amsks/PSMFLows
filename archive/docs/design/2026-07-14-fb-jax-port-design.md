# FB (Forward–Backward) Agent — JAX/Flax Port Design

**Date:** 2026-07-14
**Branch:** `feat/psm-integration`
**Reference:** PyTorch `/u/amsks/git/Factored-FB` @ `feat/psm-integration`
**Target env:** `cube-single-play-singletask-v0`

## Goal

Port the Forward–Backward (FB) agent from the PyTorch `Factored-FB` reference into
this JAX/Flax repo, following the **exact same porting protocol used for PSM**:
mirror the reference structure, achieve **bit-exact torch parity** via a fixture +
equivalence tests, then validate the training curve against the reference FB
benchmark. FB reuses ~80% of the existing PSM JAX scaffolding.

## Scope decisions (locked)

1. **Verification bar: full bit-exact torch parity** (like PSM) — export a torch
   fixture, add `torch_to_flax` loaders, and assert per-module and per-step
   numeric equivalence, plus smoke tests.
2. **Actors: both `td3` and `flow`, flow default** (mirrors PSM's `ddpgbc|flow`).
   The bit-exact fixture/equivalence tests target the **flow** path (the cube
   parity path, `fb_flowbc`); `td3` gets smoke coverage.
3. **Variant surface: cube-default path only** — measure critic + `perm` goal
   mode + learned `B` + `left_encoder` + td3/flow actor. **Not ported:** iql
   critic, traj/HER goal mode (`future_gamma` buffer sampling), `fixed_b`,
   `goal_cond` (V1), `onestep` SARSA, coverage/interaction reweighting.

## Background: FB vs the PSM we already ported

FB is essentially **PSM with the proto branch removed**:

| Piece | PSM (ours, ported) | FB (this port) |
|---|---|---|
| Basis map | `phi = PhiMap(obs)`, L2-normed | `B = BackwardMap(next_obs)`, L2-normed (√z_dim) |
| Successor(s) | **two**: `sf_psi` (policy next-action) + `psm_psi` (proto next-action) | **one**: `F(left_enc(obs), z, action)` |
| z during training | Gaussian mixed 50/50 with `phi(goal)` (SF) + binary z (proto) | Gaussian mixed 50/50 with `B(next_obs)` |
| Proto sampler / binary z | yes | **dropped** |
| Extra trunk | none (`phi_input='s'`, Identity) | **`left_encoder`** (obs feature trunk feeding F; has target) |
| Backward passes / step | 3 (psm, sf, actor) | **2** (fb, actor) |
| Target nets | phi, sf_psi, psm_psi (tau 0.01) | forward, backward, left_encoder (tau 0.005); **actor has none** |
| Measure/ortho loss math | off-diag/diag split + ortho | **identical skeleton** |
| Eval z-inference | `project(mean(r·phi))` | `project(mean(r·B))` — identical shape |
| Buffer `index` key | required (proto) | **not used** |

Because FB uses no `batch["index"]` and exposes `infer_eval_z`, **`main.py` needs
no FB-specific branch** — it drives FB through the generic path, and the
already-committed seeded eval applies unchanged.

## 1. Networks — `utils/fb_networks.py` (new)

Reuses shared primitives imported from `utils/psm_networks.py`:
`_ORTH1` (orthogonal gain 1), `_ORTH_RELU` (orthogonal √2), `ensemblize`,
`_simple_embedding`, `psm_norm`/`Norm` (√d L2), `truncated_sample`,
`NoiseConditionedActor`, `FlowVectorField`, `PSMActor`.

### `ForwardMap` (F) — ensembled, num_parallel=2
Mirrors reference `nn_models.py:ForwardMap` and is structurally PSM's `PsiMap`
with **continuous** z:
- `embed_z = simple_embedding([obs_feat, z])`, `embed_sa = simple_embedding([obs_feat, action])`
  — each: `Dense→LayerNorm→Tanh`, then `(embedding_layers-2)×[Dense→ReLU]`, final
  `Dense(hidden//2)→ReLU`; the two embeddings concat to `hidden`.
- trunk: `hidden_layers × [Dense→ReLU]` then `Dense(z_dim)`.
- ensembled via `ensemblize(..., num_parallel)`, output `[P, B, z_dim]`.
- `obs_feat` is `left_encoder(obs)` (dim `L_dim`), **not** raw obs.
- init `_ORTH_RELU` on the parallel towers.

### `BackwardMap` (B) — single copy
Mirrors reference `nn_models.py:BackwardMap` (= PSM `PhiMap` shape + `Norm`):
- `Dense→LayerNorm→Tanh`, `(hidden_layers-1)×[Dense→ReLU]`, `Dense(z_dim)`,
  then `Norm` (L2-normalize to √z_dim) when `norm=True` (default).
- consumes a single obs stream (no z, no action). Output `[B, z_dim]`.
- init `_ORTH1`.

### `left_encoder`
A `BackwardMap` instance (cube cfg: hidden 512, hidden_layers 4, norm true) that
produces `L_dim`=50 features consumed by `F`. Distinct params from `B`. Has a
**target copy** soft-updated with `f_target_tau`.

### Actor
- **flow** (default, cube): reuse `NoiseConditionedActor` (inputs `(obs, z, noise)`
  → `tanh` action) + `FlowVectorField` (`(obs, action, t)` → velocity, erf-GELU).
  Consumes **raw obs** (`actor_encode_obs=false`).
- **td3**: reuse `PSMActor` pattern (`(obs, z)` → `tanh(policy)`), sampled via
  `truncated_sample(mu, actor_std, noise, clip=stddev_clip)`.
- Actor has **no target net**.

## 2. Agent — `agents/fb.py` (new)

`FBAgent(flax.struct.PyTreeNode)` mirroring `PSMAgent` field-for-field minus proto:
```
rng, params, opt_states, z_eval,
config = nonpytree_field(),   # FrozenDict(plain, hashable) for jit static aux
nets   = nonpytree_field(),   # _HashableDict of nn.Module defs
txs    = nonpytree_field(),   # _HashableDict of optax transforms
```
Reuse `_HashableDict`, `_plain_config`, `nonpytree_field`, `_step`, `_soft`,
`get_targets_uncertainty`, and the off-diag/diag + ortho helpers from the PSM
module (extract to a shared location or import from `agents.psm` as needed).

**params keys:** `forward, backward, left_encoder, actor` (+`actor_vf` for flow),
and targets `target_forward, target_backward, target_left_encoder`.

**opt_states / txs keys (per-net `optax.adam`):** `forward=adam(lr_f)`,
`left_encoder=adam(lr_f)`, `backward=adam(lr_b)`, `actor=adam(lr_actor)`,
`actor_vf=adam(lr_actor_vf)` (flow). Separate per-net adam is numerically
identical to the reference's grouped F+left_encoder optimizer (Adam moments are
per-parameter).

### `create(cls, seed, ex_observations, ex_actions, config)`
1. `rng = PRNGKey(seed)`, split for each net.
2. Assert `config.get("encoder") is None`.
3. Build `nets`: `forward=ForwardMap(...)`, `backward=BackwardMap(...)`,
   `left_encoder=BackwardMap(...)`, actor by `actor.type`.
4. Init params with example inputs (`ex_z = zeros((B, z_dim))`, `ex_obs_feat`
   from a `left_encoder.init`+apply, `ex_noise` for flow).
5. Targets = deep copy of `forward/backward/left_encoder`.
6. Optimizers + `opt_states`.
7. `_plain_config` + inject `ob_dims/action_dim/actor_type/...`, wrap in
   `FrozenDict`. Return the agent with `z_eval=zeros((z_dim,))`.

### `update(self, batch)` — `@jax.jit`
`split rng → inj = _draw_injection(batch, rng) → apply_update(batch, inj) → replace(rng=...)`.

### `_draw_injection(batch, rng)` (draw ALL randomness outside the loss)
- Gaussian `z = project_z(normal((B, z_dim)), norm_z)`.
- 50/50 mix: `mask = uniform((B,1)) < train_goal_ratio`; `perm = permutation(B)`;
  `goal_emb = backward(next_obs)[perm]` (current backward params, treated const in
  loss); `z = where(mask, project_z(goal_emb), z)`.
- flow: `flow_x0, flow_t, flow_noise`, actor `noise`; td3: actor/next noise.
- Mirrors PSM `_draw_injection` so per-step equivalence is deterministic.

### `apply_update(batch, inj)` — un-jitted numeric core (for tests) + jitted path
**Stage 1 — FB loss** (grads on `forward, left_encoder, backward`):
```
obs_feat       = left_encoder(obs)
Fs             = forward(obs_feat, z, action)                 # [P,B,z]
B_next         = backward(next_obs)                           # [B,z]
next_obs_feat  = target_left_encoder(next_obs)                # no grad
next_action    = actor sample (flow/td3) at next_obs, z       # injected noise
tFs            = target_forward(next_obs_feat, z, next_action)
tB             = target_backward(next_obs)
tM             = get_targets_uncertainty(tFs @ tB.T, fb_pessimism_penalty)   # [B,B]
Ms             = Fs @ B_next.T                                # [P,B,B]
diff           = Ms - discount * tM
fb_offdiag     = 0.5 * (diff * off_diag).pow(2).sum() / off_diag_sum
fb_diag        = -diagonal(diff).mean() * num_parallel
Cov            = B_next @ B_next.T
ortho          = 0.5*(Cov*off_diag).pow(2).sum()/off_diag_sum - Cov.diag().mean()
fb_loss        = fb_offdiag + fb_diag + ortho_coef * ortho
```
Step `forward, left_encoder, backward`; `discount = γ·(1-terminated)` per row
(reference forces terminated=False → always γ, matching our PSM masks fix).

**Stage 2 — actor loss** (grads on `actor` (+`actor_vf`)):
```
obs_feat  = left_encoder(obs)                                 # no grad
a         = actor sample (flow: NoiseConditionedActor; td3: TruncatedNormal)
Qs        = (forward(obs_feat, z, a) * z).sum(-1)             # [P,B]
Q         = get_targets_uncertainty(Qs, actor_pessimism_penalty)
actor_loss = -Q.mean()
# flow: actor_loss = -Q.mean()/|Q|.detach() + bc_coeff*distill_bc + bc_flow_loss
# td3 : actor_loss = -Q.mean()/|Qs|.mean().detach() + bc_coeff*mse(a, action)
```
Step `actor` (+`actor_vf`). Reuse the PSM `_flow_actor_fn` flow-matching + one-step
distillation verbatim.

**Stage 3 — soft-update targets:** `target_forward, target_backward,
target_left_encoder` via `_soft(·, f_target_tau / b_target_tau)`. Actor: none.

### Eval / acting
- `infer_z(next_observations, rewards)`: `project(mean(r·B(next_obs)))` (same shape
  as PSM `infer_z`, using `backward` instead of `phi`).
- `infer_eval_z`: `self.replace(z_eval=infer_z(...))`.
- `sample_actions`: flow one-step (`NoiseConditionedActor`) or td3 mean, exactly as
  PSM. Picked up by the generic eval loop.

### `get_config()`
Importable default mirroring `configs/agent/fb.yaml`.

## 3. Config — `configs/agent/fb.yaml` (new)

```yaml
agent_name: fb
batch_size: 256
z_dim: 50
L_dim: 50
num_parallel: 2
discount: 0.99
f_target_tau: 0.005
b_target_tau: 0.005
ortho_coef: 1.0
train_goal_ratio: 0.5        # fraction of z replaced by B(next_obs)
fb_pessimism_penalty: 0.0
actor_pessimism_penalty: 0.0
actor_std: 0.2
stddev_clip: 0.3
norm_z: true
actor_encode_obs: false      # actor consumes raw obs, F consumes left_enc(obs)
weight_decay: 0.0

lr_f: 1.0e-4
lr_b: 1.0e-4
lr_actor: 1.0e-4

forward:
  hidden_dim: 512
  hidden_layers: 2
  embedding_layers: 2
backward:
  hidden_dim: 512
  hidden_layers: 4
  norm: true
left_encoder:
  hidden_dim: 512
  hidden_layers: 4
  norm: true
actor:
  type: flow                 # td3 | flow
  hidden_dim: 512
  hidden_layers: 2
  embedding_layers: 2
  bc_coeff: 3.0
  flow_steps: 10
  lr_actor_vf: 3.0e-4
  flow_actor_hidden_dim: 512
  flow_actor_hidden_layers: 2
  flow_actor_embedding_layers: 2
  flow_vf_hidden_dim: 512
  flow_vf_hidden_layers: 4

ob_dims: null
action_dim: null
encoder: null
```
Selected via `agent=fb` on the Hydra CLI.

## 4. Registration — `agents/__init__.py`
Add `from agents.fb import FBAgent` and `fb=FBAgent` to the `agents` dict (key must
equal `agent_name`).

## 5. Verification (bit-exact, mirrors PSM)

- **`tools/export_fb_fixture.py`** (torch, `/var/local/amsks/ffb-venv`): build the
  reference FB flow agent at a **small config** (z_dim=8, tiny hidden, num_parallel=2),
  `manual_seed` per stage, dump `tests/fixtures/fb_reference.npz` with namespaces
  `w__<key>` (all params incl. targets), `in__<name>` (fixed batch + injected
  randomness: Gaussian z, mask, perm, flow/actor noise), `out__<name>` (forward
  outputs + branch losses, no opt step), `grad__<net>__<key>`, and
  `step_in__<i>__<name>` / `step__<i>__<key>` for a 10-step trace.
- **`utils/torch_to_flax.py`**: add `load_forward_params`, `load_backward_params`,
  `load_left_encoder_params`, `load_flow_actor`, `load_flow_vf` (same key-remap
  convention: torch `Linear.weight [out,in]` → flax `kernel [in,out]`; DenseParallel
  `[P,in,out]` untransposed; LayerNorm `weight/bias` → `scale/bias`).
- **`tests/test_fb_networks_equiv.py`**: x64; per-module `.apply` vs `out__` at
  atol 1e-10 (`forward`, `backward`, `left_encoder`, flow actor, flow vf).
- **`tests/test_fb_agent_equiv.py`**: pure-helper checks + `compute_static` (static
  branch losses/outputs at atol 1e-10) + 10-step `apply_update` vs `step__{i}__{k}`
  at atol 1e-8, using a small equiv config.
- **`tests/test_fb_smoke.py`**: hydra-compose `configs/agent/fb.yaml`, registration,
  update finiteness (flow + td3), `sample_actions`/`infer_eval_z`, arch-regression
  Dense-kernel count.
- **Parity run:** after tests pass, train `cube-single-play-singletask-v0` and
  compare the seeded-eval curve to the reference FB benchmark (`amsks/factored-fb`;
  see the `reference-fb-flow-benchmarks` memory).

## 6. Files

**New:** `agents/fb.py`, `utils/fb_networks.py`, `configs/agent/fb.yaml`,
`tools/export_fb_fixture.py`, `tests/test_fb_networks_equiv.py`,
`tests/test_fb_agent_equiv.py`, `tests/test_fb_smoke.py`.
**Edited:** `agents/__init__.py`, `utils/torch_to_flax.py`.
**Possibly refactored:** extract shared helpers (`_HashableDict`, `_plain_config`,
`get_targets_uncertainty`, off-diag/diag + ortho, `_step`/`_soft`, `_flow_actor_fn`)
from `agents/psm.py` into a shared module if importing from `agents.psm` proves
awkward — decided during implementation, kept minimal and non-breaking to PSM.

## Non-goals / risks

- **Non-goals:** the unported variant switches (§ scope decision 3); visual/pixel
  encoders; online fine-tuning specifics beyond what `main.py` already handles.
- **Risk — cross-framework RNG:** as with PSM, bit-exact parity holds only for
  *injected* randomness; a from-scratch run cannot match torch draw-for-draw. The
  fixture/equiv tests control for this by injecting identical randomness. Final
  parity is judged on the **curve** vs the reference benchmark, not draw-parity.
- **Risk — left_encoder faithfulness:** the cube FB config wires `left_encoder` as
  a real `BackwardMap` trunk with a target; the port must include it (not Identity).
- **Risk — shared-helper extraction** from `agents/psm.py` must not regress the
  existing PSM equivalence tests; keep PSM's imports working.

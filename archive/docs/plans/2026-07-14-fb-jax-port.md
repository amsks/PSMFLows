# FB (Forward–Backward) Agent JAX Port — Implementation Plan

> **For implementers:** Execute this plan task-by-task; steps use checkbox list syntax for tracking.

**Goal:** Port the PyTorch `Factored-FB` Forward–Backward agent into this JAX/Flax repo with bit-exact torch parity, mirroring the existing PSM port.

**Architecture:** FB is PSM minus the proto branch: one Forward map `F(left_enc(obs), z, action)`, one Backward map `B(next_obs)` (the measure basis and z-source), a `left_encoder` trunk, and a td3/flow actor. Measure `M = F·Bᵀ` trained with the same off-diag/diag + ortho loss we already have. Two backward passes/step (FB, actor); targets on forward/backward/left_encoder (none on actor). It reuses ~80% of the PSM scaffolding and needs no `main.py` change.

**Tech Stack:** JAX, Flax (`flax.struct.PyTreeNode`, `@nn.compact`), Optax, ml_collections, Hydra/OmegaConf, pytest. Torch (in `/var/local/amsks/ffb-venv`) only for the fixture export tool.

## Global Constraints

- **Reference:** `/u/amsks/git/Factored-FB` @ `feat/psm-integration`. Do not modify it.
- **Target env:** `cube-single-play-singletask-v0`.
- **FB-native cube config values (verbatim):** `z_dim=50`, `L_dim=50`, `batch_size=256`, `discount=0.99`, `num_parallel=2`, `ortho_coef=1.0`, `train_goal_ratio=0.5`, `fb_pessimism_penalty=0.0`, `actor_pessimism_penalty=0.0`, `actor_std=0.2`, `stddev_clip=0.3`, `f_target_tau=0.005`, `b_target_tau=0.005`, `lr_f=1e-4`, `lr_b=1e-4`, `lr_actor=1e-4`, `lr_actor_vf=3e-4`, `bc_coeff=3.0`, `flow_steps=10`, `actor_encode_obs=false`, `norm_z=true`, `weight_decay=0.0`.
- **Actors:** `actor.type ∈ {td3, flow}`, default `flow`. Bit-exact fixture/equiv tests target the **flow** path; `td3` gets smoke coverage only.
- **Not ported (must stay absent):** iql critic, traj/HER goal mode (`future_gamma`), `fixed_b`, `goal_cond` (V1), `onestep` SARSA, coverage/interaction reweighting.
- **Python:** always `.venv/bin/python` (system `python` is 2.7). Torch tool uses `/var/local/amsks/ffb-venv/bin/python`.
- **Commits:** follow existing history conventions.
- **Determinism rule (parity):** all stochastic inputs are drawn in `_draw_injection` and passed into the loss as constants — never sampled inside a loss/grad function. This is what makes per-step equivalence testable.
- **Torch→Flax key convention:** torch `Linear.weight [out,in]` → flax `Dense.kernel [in,out]` (transpose); `DenseParallel.weight [P,in,out]` → flax vmapped kernel `[P,in,out]` (no transpose); torch `LayerNorm.{weight,bias}` → flax `{scale,bias}`.

**Confirmed by source read (2026-07-14, during execution):**
- `left_encoder` IS a real `BackwardMap` on cube (`train.py:195-198`: `BackwardArchiConfig(hidden_dim=512, hidden_layers=4, norm=True)`); F consumes `left_encoder(obs)` (L2-normed, dim L_dim=50), NOT raw obs. Include it with a target.
- `_fb_z(z) = z` when `onestep=False` (agent.py:408-412) → **no z-zeroing** in the cube path (F gets the real z). The flow_bc comment about "zero-z" only applies to onestep.
- `Fs` = `hidden_layers × [Dense→ReLU]` **plus** a final `Dense(z_dim)` (nn_models.py:223-227) → cube `hidden_layers=2` gives **3** trunk linears (`fs_0/fs_2/fs_4`), not 2. Generalize `_ForwardTower` to loop `hidden_layers`+output.
- Every dense inside `ForwardMap` is a `DenseParallel` → `weight_init` gives it `_ORTH_RELU` (√2); plain `nn.Linear` (BackwardMap/left_encoder/actor/vf) gets `_ORTH1` (gain 1) (nn_models.py:61-77).
- FB cube penalties are both 0 (`fb_pessimism_penalty=actor_pessimism_penalty=0`) → `get_targets_uncertainty` returns the plain parallel-mean; implement full uncert anyway for faithfulness.
- Optimizers (setup_training agent.py:305-321): `forward_optimizer=Adam(forward_map+left_encoder+fw_encoder, lr_f)`, `backward_optimizer=Adam(backward_map+bw_encoder, lr_b)`, `actor_optimizer=Adam(actor, lr_actor)`, `actor_vf_optimizer=Adam(actor_vf, lr_actor_vf)`. Encoders are Identity (no params). Per-net JAX adam is identical.
- Injected randomness set (per update): `z_gauss, mix_mask, perm, next_actor_noise` (update_fb next-action), `flow_x0, flow_t, actor_noise` (update_actor; the flow start = `actor_noise`, reused for compute_flow_actions). No separate `flow_noise`.
- `goal = bw_encoder(next_obs) = next_obs` (Identity); both online B and target B in the measure loss are on `next_obs` (aligned, not permuted). The perm only affects the mixed z.
- flowbc `sample_action_from_norm_obs` (flow_bc/agent.py:57-61): `_actor(obs, z, randn)` — used inside update_fb for next_action.

**Reference file:line index (from the codebase map):**
- FB TD loss `agents/fb/agent.py:566-682`; helpers `fb_successor_terms` `:67-92`, `ortho_cov` `:95-104`, `get_targets_uncertainty` `:730-745`.
- Actor (TD3) `agent.py:696-728`; (FlowBC) `agents/fb/flow_bc/agent.py:63-131`.
- z-sampling `agent.py:347-376`; reward inference `agents/fb/model.py:162-177`; project_z `model.py:151-154`.
- Update order `agent.py:464-557`; soft-update `nn_models.py:85-87`; taus/opt `agent.py:299-345`.
- Nets: ForwardMap `nn_models.py:209-240`, BackwardMap `172-197`, Actor `251-273`, NoiseConditionedActor `287-307`, VectorField `425-442`, `simple_embedding` `200-206`, init `61-77`.
- Model assembly `agents/fb/model.py:37-118`; FlowBC model `agents/fb/flow_bc/model.py`.
- Config `configs/agent/fb.yaml`, `fb_flowbc.yaml`, `configs/domain/cube_single.yaml`.

**Our-repo mirror points:**
- Agent scaffold `agents/psm.py` (`_HashableDict` `:26-36`, `nonpytree_field` import `:19`, `create` `:365-475`, `update` `:315-324`, `_draw_injection` `:269-313`, `apply_update` `:231-267`, `compute_static` `:209-229`, `_stages`/losses `:92-203`, `_step`/`_soft` `:529-535`, `total_loss` `:339-353`, `infer_z`/`infer_eval_z` `:355-363`, `sample_actions` `:326-337`, `get_config` `:478-516`).
- Networks `utils/psm_networks.py` (`_ORTH1`/`_ORTH_RELU` `:21-22`, `psm_norm` `:39-43`, `_simple_embedding` `:152-159`, `PhiMap` `:46-65`, `_PsiTower` `:68-100`, `PsiMap` `:133-149`, `PSMActor` `:103-130`, `NoiseConditionedActor` `:162-185`, `FlowVectorField` `:188-205`; `ensemblize` `utils/networks.py:13-23`).
- Converter `utils/torch_to_flax.py` (`dense_params` `:11-12`, `layernorm_params` `:15-16`, `edense_params` `:19-22`, `eln_params` `:25-27`, `load_psi_params` `:30-43`, `load_actor_params` `:46-59`, `load_phi_params` `:62-71`).
- Fixture tool `tools/export_psm_fixture.py` (torch; `savez` `:203`). Tests `tests/test_psm_networks_equiv.py`, `tests/test_psm_agent_equiv.py`, `tests/test_psm_smoke.py`; fixture `tests/fixtures/psm_reference.npz`.
- Registration `agents/__init__.py` (import `:4`, dict `:8-15`). Main loop `main.py:86-94` (generic create/update), eval `:191-212`.

---

## File Structure

**New files:**
- `tools/export_fb_fixture.py` — torch: dump `tests/fixtures/fb_reference.npz` (params, injected randomness, forward outputs, per-step trace).
- `tests/fixtures/fb_reference.npz` — generated artifact (committed; ~1 MB).
- `utils/fb_networks.py` — `ForwardMap`, `BackwardMap` Flax modules (reuse shared primitives from `psm_networks`).
- `agents/fb.py` — `FBAgent` (create/update/losses/eval/tests hooks).
- `configs/agent/fb.yaml` — FB agent config group.
- `tests/test_fb_networks_equiv.py` — per-module bit-exact vs fixture.
- `tests/test_fb_agent_equiv.py` — static + 10-step bit-exact vs fixture.
- `tests/test_fb_smoke.py` — registration, update finiteness (flow+td3), acting/eval, arch regression.

**Modified files:**
- `utils/torch_to_flax.py` — add FB loaders.
- `agents/__init__.py` — register `fb=FBAgent`.

---

## Task 1: Torch fixture export tool + generate `fb_reference.npz`

Produces the bit-exact ground truth every later parity test checks against. Mirror `tools/export_psm_fixture.py` exactly in structure; swap PSM's 3-branch update for FB's 2-branch update and drop the proto branch.

**Files:**
- Create: `tools/export_fb_fixture.py`
- Create (generated): `tests/fixtures/fb_reference.npz`

**Interfaces:**
- Produces: an `.npz` with these key namespaces (consumed by Tasks 2 & 4):
  - `w__<torch_key>` — every param incl. targets: `forward.*`, `backward.*`, `left_encoder.*`, `_actor.*` (NoiseConditionedActor), `_actor_vf.*` (VectorField), and `target_forward.*`, `target_backward.*`, `target_left_encoder.*`.
  - `in__<name>` — the fixed batch (`observations, actions, next_observations, rewards, terminals`) and all injected randomness for **one** update: `z_gauss` `[B,z]`, `mix_mask` `[B,1]` (bool), `perm` `[B]` (int), `flow_x0` `[B,adim]`, `flow_t` `[B,1]`, `flow_noise` `[B,adim]`, `actor_noise` `[B,adim]`, `next_actor_noise` `[B,adim]`.
  - `out__<name>` — pre-optimizer-step forward outputs and scalar losses: `F` `[P,B,z]`, `B` `[B,z]`, `left_enc` `[B,L]`, `M` `[P,B,B]`, `fb_loss`, `fb_offdiag`, `fb_diag`, `ortho_loss`, `actor_loss`, `bc_flow_loss`, `Q` `[B]`.
  - `grad__<net>__<param_key>` — grads for `forward`, `backward`, `left_encoder`, `actor`, `actor_vf` at the fixed inputs.
  - `step_in__<i>__<name>` and `step__<i>__<param_key>` for `i=0..9` — a 10-step trace: fresh injected randomness per step and the resulting params after each `agent.update`.
- **Small config** (so tests are fast and x64-stable): `z_dim=8`, `L_dim=8`, `num_parallel=2`, forward `hidden_dim=32/hidden_layers=2/embedding_layers=2`, backward & left_encoder `hidden_dim=32/hidden_layers=4/norm=true`, actor `hidden_dim=32/hidden_layers=2/embedding_layers=2`, actor_vf `hidden_dim=32/hidden_layers=4`, `batch_size B=16`, `action_dim=5` (cube), `discount=0.99`, taus `0.005`, `ortho_coef=1.0`, `train_goal_ratio=0.5`, `bc_coeff=3.0`, `flow_steps=10`, `actor.type=flow`.

- [ ] **Step 1: Copy the PSM tool as the starting template**

Run: `cp tools/export_psm_fixture.py tools/export_fb_fixture.py`
Then edit it to import the FB reference agent instead of PSM. Study these reference locations before editing: `agents/fb/flow_bc/agent.py:21` (`FBFlowBCAgent`), `agents/fb/flow_bc/model.py:21` (`FBFlowBCModel`), the update entry `agents/fb/agent.py:464-557`, the FB loss `:566-682`, the flow actor loss `agents/fb/flow_bc/agent.py:63-131`.

- [ ] **Step 2: Build the reference FB flow agent at the small config and dump `w__` params**

Instantiate `FBFlowBCModel`/`FBFlowBCAgent` with the small config above using the `ffb-venv` torch. Call `weight_init` (`nn_models.py:61-77`) then deep-copy targets (`agents/fb/model.py:120-124`). Serialize every named parameter to `w__<name>` (float64) exactly as `export_psm_fixture.py` does for PSM. Names must match the torch module attribute paths so the flax loaders (Task 2) can remap them.

- [ ] **Step 3: Fix one batch + injected randomness; record `in__` and `out__`**

Use `torch.manual_seed` per stage (mirror the PSM tool). Build `B=16` fixed obs/action/next_obs/rewards/terminals. Draw and record: `z_gauss`, `mix_mask`, `perm`, `flow_x0`, `flow_t`, `flow_noise`, `actor_noise`, `next_actor_noise`. Compute `z = where(mix_mask, project_z(backward(next_obs)[perm]), z_gauss)` (reference `sample_mixed_z` `agent.py:347-376`, but with the injected draws substituted for internal sampling). Run the FB loss forward (`agent.py:566-682`) and the flow-actor loss (`flow_bc/agent.py:63-131`) WITHOUT stepping optimizers; record `out__F/B/left_enc/M/fb_loss/fb_offdiag/fb_diag/ortho_loss/actor_loss/bc_flow_loss/Q` and `grad__*` (call `.backward()` on each loss with retained graph, read `.grad`).

- [ ] **Step 4: Record the 10-step trace**

For `i in range(10)`: draw fresh injected randomness → `step_in__{i}__*`; run one full `agent.update` (FB step then actor step then soft-update) with that injected randomness; snapshot all params → `step__{i}__<name>`. Match the injection substitution used in Step 3.

- [ ] **Step 5: Save and sanity-check the npz**

`np.savez(tests/fixtures/fb_reference.npz, **all_arrays)` (mirror `export_psm_fixture.py:203`). Add a tiny loader check at the bottom of the tool that reasserts the key set and shapes.

Run: `/var/local/amsks/ffb-venv/bin/python tools/export_fb_fixture.py`
Expected: prints saved path; `tests/fixtures/fb_reference.npz` exists (~1 MB).

- [ ] **Step 6: Write and run a schema test**

Create `tests/test_fb_smoke.py` with only this test for now:

```python
import numpy as np


def test_fb_fixture_schema():
    fix = np.load("tests/fixtures/fb_reference.npz")
    keys = set(fix.keys())
    for k in ["in__observations", "in__z_gauss", "in__mix_mask", "in__perm",
              "out__F", "out__B", "out__M", "out__fb_loss", "out__actor_loss"]:
        assert k in keys, f"missing {k}"
    for net in ["forward", "backward", "left_encoder"]:
        assert any(k.startswith(f"w__{net}") for k in keys), f"no params for {net}"
    for net in ["forward", "backward", "left_encoder"]:
        assert f"w__target_{net}" in "\n".join(keys) or any(
            k.startswith(f"w__target_{net}") for k in keys), f"no target for {net}"
    for i in range(10):
        assert any(k.startswith(f"step__{i}__") for k in keys), f"no step {i}"
    assert fix["out__F"].ndim == 3 and fix["out__F"].shape[0] == 2  # [P,B,z]
```

Run: `.venv/bin/python -m pytest tests/test_fb_smoke.py::test_fb_fixture_schema -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add tools/export_fb_fixture.py tests/fixtures/fb_reference.npz tests/test_fb_smoke.py
git commit -m "FB port: torch fixture export tool + fb_reference.npz"
```

---

## Task 2: FB networks + torch→flax loaders + per-module bit-exact tests

**Files:**
- Create: `utils/fb_networks.py`
- Modify: `utils/torch_to_flax.py` (append FB loaders)
- Create: `tests/test_fb_networks_equiv.py`

**Interfaces:**
- Consumes: `w__*`/`out__*` from `fb_reference.npz`; shared primitives from `utils/psm_networks.py` (`_ORTH1`, `_ORTH_RELU`, `psm_norm`, `_simple_embedding`, `NoiseConditionedActor`, `FlowVectorField`) and `ensemblize` from `utils/networks.py`.
- Produces (used by Tasks 3–5):
  - `ForwardMap(z_dim, hidden_dim, hidden_layers, embedding_layers, num_parallel).__call__(obs_feat, z, action) -> [P,B,z_dim]`
  - `BackwardMap(z_dim, hidden_dim, hidden_layers, norm=True).__call__(obs) -> [B,z_dim]`
  - loaders `load_forward_params(fix, prefix="forward")`, `load_backward_params(fix, prefix="backward")`, `load_left_encoder_params(fix, prefix="left_encoder")`, `load_flow_actor(fix, prefix="_actor")`, `load_flow_vf(fix, prefix="_actor_vf")` returning flax param pytrees.

- [ ] **Step 1: Write the failing per-module test**

Create `tests/test_fb_networks_equiv.py`:

```python
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import jax.numpy as jnp
from utils.fb_networks import ForwardMap, BackwardMap
from utils.torch_to_flax import (load_forward_params, load_backward_params,
                                 load_left_encoder_params)

FIX = np.load("tests/fixtures/fb_reference.npz")
Z, L, P = 8, 8, 2


def _obs():   return jnp.asarray(FIX["in__observations"], jnp.float64)
def _act():   return jnp.asarray(FIX["in__actions"], jnp.float64)
def _z():     return jnp.asarray(FIX["in__z_gauss"], jnp.float64)


def test_backward_equiv():
    m = BackwardMap(z_dim=Z, hidden_dim=32, hidden_layers=4, norm=True)
    p = load_backward_params(FIX)
    out = m.apply({"params": p}, _obs())
    assert np.allclose(out, FIX["out__B"], atol=1e-10)


def test_left_encoder_equiv():
    m = BackwardMap(z_dim=L, hidden_dim=32, hidden_layers=4, norm=True)
    p = load_left_encoder_params(FIX)
    out = m.apply({"params": p}, _obs())
    assert np.allclose(out, FIX["out__left_enc"], atol=1e-10)


def test_forward_equiv():
    m = ForwardMap(z_dim=Z, hidden_dim=32, hidden_layers=2,
                   embedding_layers=2, num_parallel=P)
    p = load_forward_params(FIX)
    obs_feat = jnp.asarray(FIX["out__left_enc"], jnp.float64)  # F consumes left_enc(obs)
    out = m.apply({"params": p}, obs_feat, _z(), _act())
    assert np.allclose(out, FIX["out__F"], atol=1e-10)
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_fb_networks_equiv.py -v`
Expected: FAIL (`ModuleNotFoundError: utils.fb_networks` / loaders undefined).

- [ ] **Step 3: Implement `utils/fb_networks.py`**

Mirror `utils/psm_networks.py` module style. `BackwardMap` = `PhiMap` shape with a trailing `psm_norm` gated by `norm`; `ForwardMap` = `PsiMap`/`_PsiTower` shape (two `_simple_embedding`s over `[obs_feat,z]` and `[obs_feat,action]`, concat → `hidden_layers×[Dense→relu]` → `Dense(z_dim)`, ensembled). Use `_ORTH_RELU` on the parallel towers and `_ORTH1` on `BackwardMap`. Name the `_PsiTower`-equivalent submodules explicitly (`embed_z_0/embed_z_ln/embed_z_3`, `embed_sa_0/embed_sa_ln/embed_sa_3`, `fs_0`, `fs_2`) so the loader mapping is unambiguous — copy the naming from `psm_networks.py:68-100`.

```python
import math
import flax.linen as nn
import jax.numpy as jnp
from utils.psm_networks import _ORTH1, _ORTH_RELU, psm_norm, _simple_embedding
from utils.networks import ensemblize


class BackwardMap(nn.Module):
    z_dim: int
    hidden_dim: int
    hidden_layers: int = 4
    norm: bool = True

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim, kernel_init=_ORTH1)(x)
        x = nn.LayerNorm(epsilon=1e-5)(x)
        x = jnp.tanh(x)
        for _ in range(self.hidden_layers - 1):
            x = nn.relu(nn.Dense(self.hidden_dim, kernel_init=_ORTH1)(x))
        x = nn.Dense(self.z_dim, kernel_init=_ORTH1)(x)
        return psm_norm(x) if self.norm else x


class _ForwardTower(nn.Module):
    z_dim: int
    hidden_dim: int
    hidden_layers: int = 2
    embedding_layers: int = 2

    def setup(self):
        # explicit names to mirror _PsiTower for unambiguous torch->flax mapping
        self.embed_z_0 = nn.Dense(self.hidden_dim, kernel_init=_ORTH_RELU)
        self.embed_z_ln = nn.LayerNorm(epsilon=1e-5)
        self.embed_z_3 = nn.Dense(self.hidden_dim // 2, kernel_init=_ORTH_RELU)
        self.embed_sa_0 = nn.Dense(self.hidden_dim, kernel_init=_ORTH_RELU)
        self.embed_sa_ln = nn.LayerNorm(epsilon=1e-5)
        self.embed_sa_3 = nn.Dense(self.hidden_dim // 2, kernel_init=_ORTH_RELU)
        self.fs_0 = nn.Dense(self.hidden_dim, kernel_init=_ORTH_RELU)
        self.fs_2 = nn.Dense(self.z_dim, kernel_init=_ORTH_RELU)

    def __call__(self, obs_feat, z, action):
        ez = jnp.tanh(self.embed_z_ln(self.embed_z_0(jnp.concatenate([obs_feat, z], -1))))
        ez = nn.relu(self.embed_z_3(ez))
        esa = jnp.tanh(self.embed_sa_ln(self.embed_sa_0(jnp.concatenate([obs_feat, action], -1))))
        esa = nn.relu(self.embed_sa_3(esa))
        h = nn.relu(self.fs_0(jnp.concatenate([esa, ez], -1)))
        return self.fs_2(h)


class ForwardMap(nn.Module):
    z_dim: int
    hidden_dim: int
    num_parallel: int = 2
    hidden_layers: int = 2
    embedding_layers: int = 2

    @nn.compact
    def __call__(self, obs_feat, z, action):
        tower = ensemblize(_ForwardTower, self.num_parallel, in_axes=None)(
            z_dim=self.z_dim, hidden_dim=self.hidden_dim,
            hidden_layers=self.hidden_layers, embedding_layers=self.embedding_layers,
            name="tower")
        return tower(obs_feat, z, action)
```

Note: this assumes cube's `embedding_layers=2` and `hidden_layers=2` (one trunk Dense `fs_0` then output `fs_2`), matching `simple_embedding` with `hidden_layers==2` (single `Dense(hidden//2)→ReLU` embedding). If the reference `Fs` has >1 trunk hidden layer, add the extra `fs_*` Denses and reflect them in the loader — verify against `nn_models.py:209-240` during implementation and adjust both module and loader together.

- [ ] **Step 4: Implement the FB loaders in `utils/torch_to_flax.py`**

Append, reusing `dense_params`, `layernorm_params`, `edense_params`, `eln_params`. `load_backward_params`/`load_left_encoder_params` map the torch `net` Sequential indices to flax compact auto-names (`Dense_0, LayerNorm_0, Dense_1..`) — copy the index scheme from `load_phi_params` (`torch_to_flax.py:62-71`) but for `hidden_layers=4` (more `Dense_*`). `load_forward_params` returns `{"tower": {...}}` mapping the ensembled `[P,in,out]` DenseParallel weights via `edense_params`/`eln_params` to the `_ForwardTower` setup names — copy the scheme from `load_psi_params` (`:30-43`). Add `load_flow_actor`/`load_flow_vf` mirroring the eventual PSM flow loaders (if absent in PSM, derive from `NoiseConditionedActor`/`FlowVectorField` setup names: `NoiseConditionedActor` dual `_simple_embedding` + trunk + tanh head; `FlowVectorField` `net.{0,2,4,6,8}`).

- [ ] **Step 5: Run the per-module tests**

Run: `.venv/bin/python -m pytest tests/test_fb_networks_equiv.py -v`
Expected: PASS (`test_backward_equiv`, `test_left_encoder_equiv`, `test_forward_equiv`). If a mismatch appears, it is almost always a loader index or a transpose — fix the loader/module together; do not loosen `atol`.

- [ ] **Step 6: Add flow actor/vf equivalence tests and run**

Append to `tests/test_fb_networks_equiv.py`:

```python
from utils.psm_networks import NoiseConditionedActor, FlowVectorField
from utils.torch_to_flax import load_flow_actor, load_flow_vf


def test_flow_actor_equiv():
    m = NoiseConditionedActor(action_dim=5, hidden_dim=32, hidden_layers=2, embedding_layers=2)
    p = load_flow_actor(FIX)
    out = m.apply({"params": p}, _obs(), _z(), jnp.asarray(FIX["in__actor_noise"], jnp.float64))
    assert np.allclose(out, FIX["out__actor_mu"], atol=1e-10)


def test_flow_vf_equiv():
    m = FlowVectorField(action_dim=5, hidden_dim=32, hidden_layers=4)
    p = load_flow_vf(FIX)
    out = m.apply({"params": p}, _obs(), _act(), jnp.asarray(FIX["in__flow_t"], jnp.float64))
    assert np.allclose(out, FIX["out__vf"], atol=1e-10)
```

(Task 1 must also emit `out__actor_mu` and `out__vf` — if missing, add them to `export_fb_fixture.py` Step 3 and regenerate.)

Run: `.venv/bin/python -m pytest tests/test_fb_networks_equiv.py -v`
Expected: PASS (all 5).

- [ ] **Step 7: Commit**

```bash
git add utils/fb_networks.py utils/torch_to_flax.py tests/test_fb_networks_equiv.py \
        tools/export_fb_fixture.py tests/fixtures/fb_reference.npz
git commit -m "FB port: F/B/left_encoder + flow nets with bit-exact torch parity"
```

---

## Task 3: Config, registration, and `FBAgent.create`

**Files:**
- Create: `configs/agent/fb.yaml`
- Modify: `agents/__init__.py`
- Create: `agents/fb.py` (create + skeleton only; update comes in Task 4)
- Modify: `tests/test_fb_smoke.py`

**Interfaces:**
- Consumes: `ForwardMap`/`BackwardMap` (Task 2), `NoiseConditionedActor`/`FlowVectorField`/`PSMActor` (psm_networks), `_HashableDict`/`nonpytree_field`/`_plain_config`/`project_z` (from `agents/psm.py` or a shared module).
- Produces: `FBAgent.create(seed, ex_observations, ex_actions, config) -> FBAgent` with `params` keys `forward/backward/left_encoder/actor(/actor_vf)` + `target_forward/target_backward/target_left_encoder`; `opt_states`/`txs` keyed `forward/left_encoder/backward/actor(/actor_vf)`; `z_eval` `[z_dim]`. Registered as `agents["fb"]`.

- [ ] **Step 1: Write `configs/agent/fb.yaml`** (exact values from Global Constraints)

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
train_goal_ratio: 0.5
fb_pessimism_penalty: 0.0
actor_pessimism_penalty: 0.0
actor_std: 0.2
stddev_clip: 0.3
norm_z: true
actor_encode_obs: false
weight_decay: 0.0
lr_f: 1.0e-4
lr_b: 1.0e-4
lr_actor: 1.0e-4
forward: {hidden_dim: 512, hidden_layers: 2, embedding_layers: 2}
backward: {hidden_dim: 512, hidden_layers: 4, norm: true}
left_encoder: {hidden_dim: 512, hidden_layers: 4, norm: true}
actor:
  type: flow
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

- [ ] **Step 2: Write the failing registration+create smoke test**

Add to `tests/test_fb_smoke.py`:

```python
import ml_collections
import numpy as np
from hydra import compose, initialize
from omegaconf import OmegaConf
from agents import agents


def _cfg():
    with initialize(version_base="1.3", config_path="../configs/agent"):
        c = compose(config_name="fb")
    return ml_collections.ConfigDict(OmegaConf.to_container(c, resolve=True))


def _ex(n=4, obs=19, act=5):
    rng = np.random.default_rng(0)
    return (rng.standard_normal((n, obs)).astype(np.float32),
            np.clip(rng.standard_normal((n, act)), -1, 1).astype(np.float32))


def test_fb_registered_and_creates():
    assert "fb" in agents
    cfg = _cfg(); cfg["ob_dims"] = (19,); cfg["action_dim"] = 5
    o, a = _ex()
    agent = agents["fb"].create(0, o, a, cfg)
    for k in ["forward", "backward", "left_encoder", "actor", "actor_vf",
              "target_forward", "target_backward", "target_left_encoder"]:
        assert k in agent.params, f"missing param {k}"
    assert agent.z_eval.shape == (cfg["z_dim"],)
```

Run: `.venv/bin/python -m pytest tests/test_fb_smoke.py::test_fb_registered_and_creates -v`
Expected: FAIL (`KeyError: 'fb'`).

- [ ] **Step 3: Implement `agents/fb.py` create + skeleton**

Mirror `agents/psm.py:365-475`. Struct fields: `rng, params, opt_states, z_eval, config=nonpytree_field(), nets=nonpytree_field(), txs=nonpytree_field()` (NO `proto`). In `create`: split rng per net; assert `config.get("encoder") is None`; build `nets` (`forward=ForwardMap(...)`, `backward=BackwardMap(z_dim, backward.hidden_dim, backward.hidden_layers, backward.norm)`, `left_encoder=BackwardMap(L_dim, left_encoder.hidden_dim, left_encoder.hidden_layers, left_encoder.norm)`, actor by `actor.type`: flow → `NoiseConditionedActor`+`FlowVectorField`, td3 → `PSMActor`); init params (compute `ex_obs_feat = left_encoder.apply(...)` to init `forward`); deep-copy targets for forward/backward/left_encoder; build per-net `optax.adam` (`forward/left_encoder=lr_f`, `backward=lr_b`, `actor=lr_actor`, `actor_vf=lr_actor_vf`) and `opt_states`; `_plain_config` + inject `ob_dims/action_dim/actor_type`; return the agent with `z_eval=zeros((z_dim,))`. Reuse `_HashableDict`/`nonpytree_field`/`_plain_config`/`_step`/`_soft`/`get_targets_uncertainty`/`project_z` by importing from `agents.psm` (extract to a shared module only if an import cycle appears; keep PSM imports working per Global Constraints).

- [ ] **Step 4: Register the agent** — edit `agents/__init__.py`: add `from agents.fb import FBAgent` near `:4` and `fb=FBAgent` into the `agents=dict(...)` at `:8-15`.

- [ ] **Step 5: Run the smoke test**

Run: `.venv/bin/python -m pytest tests/test_fb_smoke.py::test_fb_registered_and_creates -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add configs/agent/fb.yaml agents/__init__.py agents/fb.py tests/test_fb_smoke.py
git commit -m "FB port: config, registration, and FBAgent.create"
```

---

## Task 4: `FBAgent.update` (FB loss + actor loss + soft update) with bit-exact parity

**Files:**
- Modify: `agents/fb.py` (add `_draw_injection`, `_fb_loss`, `_actor_loss`, `apply_update`, `compute_static`, `update`, `total_loss`)
- Create: `tests/test_fb_agent_equiv.py`

**Interfaces:**
- Consumes: nets/params from Task 3; injected-randomness keys from `fb_reference.npz` (`in__*`, `step_in__*`).
- Produces: `update(self, batch) -> (FBAgent, info)` (`@jax.jit`), `apply_update(self, batch, inj) -> (FBAgent, info)` (un-jitted), `compute_static(self, batch, inj) -> info` (un-jitted static losses/outputs).

- [ ] **Step 1: Write the failing static-equivalence test**

Create `tests/test_fb_agent_equiv.py`:

```python
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import jax.numpy as jnp
import ml_collections
from agents.fb import FBAgent
from utils.torch_to_flax import (load_forward_params, load_backward_params,
                                 load_left_encoder_params, load_flow_actor, load_flow_vf)

FIX = np.load("tests/fixtures/fb_reference.npz")

EQUIV = ml_collections.ConfigDict(dict(
    agent_name="fb", batch_size=16, z_dim=8, L_dim=8, num_parallel=2, discount=0.99,
    f_target_tau=0.005, b_target_tau=0.005, ortho_coef=1.0, train_goal_ratio=0.5,
    fb_pessimism_penalty=0.0, actor_pessimism_penalty=0.0, actor_std=0.2,
    stddev_clip=0.3, norm_z=True, actor_encode_obs=False, weight_decay=0.0,
    lr_f=1e-4, lr_b=1e-4, lr_actor=1e-4,
    forward=dict(hidden_dim=32, hidden_layers=2, embedding_layers=2),
    backward=dict(hidden_dim=32, hidden_layers=4, norm=True),
    left_encoder=dict(hidden_dim=32, hidden_layers=4, norm=True),
    actor=dict(type="flow", hidden_dim=32, hidden_layers=2, embedding_layers=2,
               bc_coeff=3.0, flow_steps=10, lr_actor_vf=3e-4,
               flow_actor_hidden_dim=32, flow_actor_hidden_layers=2,
               flow_actor_embedding_layers=2, flow_vf_hidden_dim=32,
               flow_vf_hidden_layers=4),
    ob_dims=(19,), action_dim=5, encoder=None))


def _batch():
    return {k: jnp.asarray(FIX[f"in__{k}"], jnp.float64) for k in
            ["observations", "actions", "next_observations", "rewards", "terminals"]}


def _inj():
    return {k: jnp.asarray(FIX[f"in__{k}"]) for k in
            ["z_gauss", "mix_mask", "perm", "flow_x0", "flow_t", "flow_noise",
             "actor_noise", "next_actor_noise"]}


def _mapped_agent():
    o = jnp.asarray(FIX["in__observations"], jnp.float64)
    a = jnp.asarray(FIX["in__actions"], jnp.float64)
    agent = FBAgent.create(0, o, a, EQUIV)
    p = dict(agent.params)
    p["forward"] = load_forward_params(FIX); p["backward"] = load_backward_params(FIX)
    p["left_encoder"] = load_left_encoder_params(FIX)
    p["actor"] = load_flow_actor(FIX); p["actor_vf"] = load_flow_vf(FIX)
    for t in ["forward", "backward", "left_encoder"]:
        p[f"target_{t}"] = load_forward_params(FIX) if t == "forward" else (
            load_backward_params(FIX) if t == "backward" else load_left_encoder_params(FIX))
    return agent.replace(params=p)


def test_fb_static_equiv():
    agent = _mapped_agent()
    info = agent.compute_static(_batch(), _inj())
    assert np.allclose(info["fb_loss"], FIX["out__fb_loss"], atol=1e-10)
    assert np.allclose(info["actor_loss"], FIX["out__actor_loss"], atol=1e-10)
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_fb_agent_equiv.py::test_fb_static_equiv -v`
Expected: FAIL (`compute_static` undefined).

- [ ] **Step 3: Implement `_draw_injection`, losses, `compute_static`**

Mirror `agents/psm.py:269-313` (injection) and `:92-203`/`:209-229` (loss stages / static). Injection (used by `update`): draw `z_gauss=project_z(normal((B,z_dim)))`, `mix_mask=uniform((B,1))<train_goal_ratio`, `perm=permutation(B)`, `flow_x0/flow_t/flow_noise/actor_noise/next_actor_noise`; compute `z = where(mix_mask, project_z(backward(next_obs)[perm]), z_gauss)` using current `backward` params. `compute_static`/`apply_update` accept `inj` so tests inject the fixture's draws instead. FB loss (reference `agent.py:566-682`):

```python
obs_feat  = apply(left_encoder, obs)
Fs        = apply(forward, obs_feat, z, action)                 # [P,B,z]
Bn        = apply(backward, next_obs)                           # [B,z]
noff      = next_obs feat via target_left_encoder
na        = actor next-action at next_obs (injected next_actor_noise)
tFs       = apply(target_forward, noff, z, na)
tBn       = apply(target_backward, next_obs)
tM        = get_targets_uncertainty(einsum('pbz,cz->pbc', tFs, tBn), fb_pessimism_penalty)  # [B,B]
Ms        = einsum('pbz,cz->pbc', Fs, Bn)                       # [P,B,B]
diff      = Ms - (discount*(1-terminals))[:,None] * tM
off       = 1 - eye(B)
fb_off    = 0.5 * (diff*off**2).sum() / off.sum()
fb_diag   = -jnp.diagonal(diff, axis1=1, axis2=2).mean() * num_parallel
Cov       = Bn @ Bn.T
ortho     = 0.5*((Cov*off)**2).sum()/off.sum() - jnp.diagonal(Cov).mean()
fb_loss   = fb_off + fb_diag + ortho_coef*ortho
```

Actor loss (flow): reuse the PSM `_flow_actor_fn` verbatim but with `Q = get_targets_uncertainty((apply(forward, obs_feat_nograd, z, a)*z).sum(-1), actor_pessimism_penalty)`. `compute_static` returns `{fb_loss, fb_offdiag, fb_diag, ortho_loss, actor_loss, bc_flow_loss, ...}` without stepping optimizers.

- [ ] **Step 4: Run static equivalence**

Run: `.venv/bin/python -m pytest tests/test_fb_agent_equiv.py::test_fb_static_equiv -v`
Expected: PASS (atol 1e-10).

- [ ] **Step 5: Implement `apply_update` + `update` + `total_loss`; add per-step test**

`apply_update(batch, inj)`: stage 1 `value_and_grad(fb_loss)` over `{forward,left_encoder,backward}` → `_step` each; stage 2 `value_and_grad(actor_loss)` over `{actor(,actor_vf)}` → `_step`; stage 3 `_soft` targets (`forward/left_encoder`→`f_target_tau`, `backward`→`b_target_tau`). `update` = split rng → `_draw_injection` → `apply_update` → `replace(rng=...)`, `@jax.jit`. `total_loss(batch, grad_params=None)` returns summed branch losses for validation logging. Add:

```python
def test_fb_perstep_equiv():
    agent = _mapped_agent()
    for i in range(10):
        inj = {k: jnp.asarray(FIX[f"step_in__{i}__{k}"]) for k in
               ["z_gauss", "mix_mask", "perm", "flow_x0", "flow_t", "flow_noise",
                "actor_noise", "next_actor_noise"]}
        batch = {k: jnp.asarray(FIX[f"step_in__{i}__{k}"], jnp.float64) for k in
                 ["observations", "actions", "next_observations", "rewards", "terminals"]}
        agent, _ = agent.apply_update(batch, inj)
        for net in ["forward", "backward", "left_encoder"]:
            ref = FIX[f"step__{i}__{net}.net.0.weight"] if False else None  # see note
    # compare a representative param each step
        got = np.asarray(agent.params["backward"]["Dense_0"]["kernel"])
        exp = FIX[f"step__{i}__backward.net.0.weight"].T
        assert np.allclose(got, exp, atol=1e-8), f"step {i} backward mismatch"
```

(Adjust the exact compared key to whatever `export_fb_fixture.py` names; the point is a per-step atol-1e-8 check on a stepped param for `i=0..9`.)

Run: `.venv/bin/python -m pytest tests/test_fb_agent_equiv.py -v`
Expected: PASS (`test_fb_static_equiv`, `test_fb_perstep_equiv`).

- [ ] **Step 6: Commit**

```bash
git add agents/fb.py tests/test_fb_agent_equiv.py
git commit -m "FB port: update (FB + actor + soft-update) bit-exact to torch (atol 1e-8)"
```

---

## Task 5: Eval/acting, td3 actor, and full smoke suite

**Files:**
- Modify: `agents/fb.py` (`infer_z`, `infer_eval_z`, `sample_actions`, `get_config`, td3 actor loss branch)
- Modify: `tests/test_fb_smoke.py`

**Interfaces:**
- Produces: `infer_z(next_observations, rewards) -> [z_dim]`, `infer_eval_z(next_observations, rewards) -> FBAgent`, `sample_actions(observations, seed=None, temperature=1.0) -> actions`.

- [ ] **Step 1: Write failing acting/eval + td3 smoke tests**

Add to `tests/test_fb_smoke.py`:

```python
import jax


def test_fb_update_finite_flow():
    cfg = _cfg(); cfg["ob_dims"] = (19,); cfg["action_dim"] = 5
    o, a = _ex(64)
    agent = agents["fb"].create(0, o, a, cfg)
    batch = dict(observations=o, actions=a, next_observations=o,
                 rewards=np.zeros((64,), np.float32), terminals=np.zeros((64,), np.float32),
                 masks=np.ones((64,), np.float32))
    agent, info = agent.update(batch)
    assert np.isfinite(info["fb_loss"]) and np.isfinite(info["actor_loss"])


def test_fb_update_finite_td3():
    with initialize(version_base="1.3", config_path="../configs/agent"):
        c = compose(config_name="fb", overrides=["actor.type=td3"])
    cfg = ml_collections.ConfigDict(OmegaConf.to_container(c, resolve=True))
    cfg["ob_dims"] = (19,); cfg["action_dim"] = 5
    o, a = _ex(64)
    agent = agents["fb"].create(0, o, a, cfg)
    assert "actor_vf" not in agent.params
    batch = dict(observations=o, actions=a, next_observations=o,
                 rewards=np.zeros((64,), np.float32), terminals=np.zeros((64,), np.float32),
                 masks=np.ones((64,), np.float32))
    agent, info = agent.update(batch)
    assert np.isfinite(info["actor_loss"])


def test_fb_infer_eval_z_and_act():
    cfg = _cfg(); cfg["ob_dims"] = (19,); cfg["action_dim"] = 5
    o, a = _ex(64)
    agent = agents["fb"].create(0, o, a, cfg)
    ea = agent.infer_eval_z(o, np.ones((64,), np.float32))
    assert ea.z_eval.shape == (cfg["z_dim"],)
    act = ea.sample_actions(observations=o[0], seed=jax.random.PRNGKey(0), temperature=0)
    act = np.asarray(act)
    assert act.shape == (5,) and np.all(np.abs(act) <= 1.0 + 1e-5)
```

Run: `.venv/bin/python -m pytest tests/test_fb_smoke.py -k "finite or infer" -v`
Expected: FAIL (methods/branch missing).

- [ ] **Step 2: Implement `infer_z`/`infer_eval_z`/`sample_actions`/td3 branch/`get_config`**

`infer_z`: `phi = apply(backward, next_observations); z = (rewards.reshape(1,-1) @ phi).reshape(-1)/phi.shape[0]; return project_z(z, norm_z)` (mirror `psm.py:355-358` with `backward`). `infer_eval_z` = `self.replace(z_eval=infer_z(...))`. `sample_actions` mirror `psm.py:326-337` (flow one-step `NoiseConditionedActor`; td3 → actor mean). td3 actor-loss branch: `a = truncated_sample(mu, actor_std, actor_noise, clip=stddev_clip)`, `actor_loss = -Q.mean()/jnp.abs(Qs).mean() + bc_coeff*mse(a, action)`. `get_config()` returning the yaml defaults via `ml_collections.ConfigDict`.

- [ ] **Step 3: Run the acting/eval + td3 tests**

Run: `.venv/bin/python -m pytest tests/test_fb_smoke.py -k "finite or infer" -v`
Expected: PASS.

- [ ] **Step 4: Add an architecture-regression test and run the whole FB suite**

Add to `tests/test_fb_smoke.py`:

```python
def test_fb_arch_regression():
    cfg = _cfg(); cfg["ob_dims"] = (19,); cfg["action_dim"] = 5
    o, a = _ex()
    agent = agents["fb"].create(0, o, a, cfg)
    # backward: 4 hidden_layers -> Dense_0..Dense_4 = 5 Dense kernels
    n_bwd = sum(1 for k in agent.params["backward"] if k.startswith("Dense"))
    assert n_bwd == 5
    assert "target_forward" in agent.params and "target_backward" in agent.params
```

Run: `.venv/bin/python -m pytest tests/test_fb_smoke.py tests/test_fb_networks_equiv.py tests/test_fb_agent_equiv.py -v`
Expected: PASS (all FB tests).

- [ ] **Step 5: Run the full repo suite (guard against regressions)**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all PSM + FB tests PASS; the 4 pre-existing `test_flow_inversion.py`/`test_flow_steering.py` failures remain (unrelated — do not fix here).

- [ ] **Step 6: Commit**

```bash
git add agents/fb.py tests/test_fb_smoke.py
git commit -m "FB port: eval/acting, td3 actor, full smoke suite"
```

---

## Task 6: Parity-run launcher + docs

**Files:**
- Create: `scripts/launch_fb_cube.sh`
- Modify: `docs/HANDOFF.md`

**Interfaces:** none (ops/docs).

- [ ] **Step 1: Write the launcher** — `scripts/launch_fb_cube.sh` mirroring `scripts/launch_psm_cube.sh`, but `agent=fb`, `env_name=cube-single-play-singletask-v0`, one seed per GPU, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.30`, `save_interval=100000`, seeded eval (already default). Include the `source .venv/bin/activate` and `CUDA_VISIBLE_DEVICES` pattern.

- [ ] **Step 2: Dry-run the config resolves** (no training)

Run: `.venv/bin/python -c "from hydra import compose, initialize; from omegaconf import OmegaConf; initialize(version_base='1.3', config_path='configs'); print('ok')"`
Then: `.venv/bin/python main.py agent=fb env_name=cube-single-play-singletask-v0 offline_steps=2 online_steps=0 eval_interval=0 save_interval=0 wandb_project=PSMFLows run_group=fb_smoke seed=0 2>&1 | tail -5`
Expected: 2 update steps run without error (a real 2-step training tick).

- [ ] **Step 3: Document in HANDOFF** — add an "FB port" section to `docs/HANDOFF.md`: what was ported, the parity-test guarantee, the config, and the next step (a cube-single parity run vs `amsks/factored-fb` FB benchmark using the seeded eval; see the `reference-fb-flow-benchmarks` memory).

- [ ] **Step 4: Commit**

```bash
git add scripts/launch_fb_cube.sh docs/HANDOFF.md
git commit -m "FB port: cube launcher + handoff notes"
```

---

## Self-Review (completed)

**Spec coverage:** §1 networks → Task 2; §2 agent/update → Tasks 3–4; §3 config → Task 3; §4 registration → Task 3; §5 verification (fixture/loaders/equiv/smoke) → Tasks 1,2,4,5; eval/acting → Task 5; parity run → Task 6. All spec sections mapped.

**Placeholders:** the two intentional "verify against reference and adjust module+loader together" notes (Task 2 Step 3, Task 4 Step 5 key names) are unavoidable for a byte-faithful port where exact torch Sequential indexing must be confirmed against `nn_models.py` at implementation time; they are bounded (adjust a named index, keep atol) not open-ended. No `TODO`/`TBD`/"add error handling" placeholders remain.

**Type consistency:** `ForwardMap`/`BackwardMap` signatures, loader names (`load_forward_params`/`load_backward_params`/`load_left_encoder_params`/`load_flow_actor`/`load_flow_vf`), param keys (`forward/backward/left_encoder/actor/actor_vf` + `target_*`), and opt keys are consistent across Tasks 2–5. The `_ForwardTower` submodule names match between the module (Task 2 Step 3) and the loader scheme (Task 2 Step 4).

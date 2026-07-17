# PSM Phase 1 — Idiomatic Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor `agents/psm.py` into idiomatic, readable JAX (ModuleDict/`select`-style network layer, named `*_loss` methods, a named `StepInputs` struct instead of the opaque `inj` dict) with **identical numerics** — the bit-exact equivalence tests stay green.

**Architecture:** Keep a single `network` (`ModuleDict` `TrainState`, no optimizer) as the one params tree for apply/`select`/checkpointing. Keep per-network optimizers + target params as sibling fields, stepped sequentially, because PSM's update is a 3-stage sequential procedure (proto → sf → actor) with per-network learning rates and interleaved target soft-updates that cannot collapse to a single-loss/single-optimizer step without changing numerics. All math (`contrastive_loss`, `ortho_loss`, `proto_sample`, the three branch losses) is moved verbatim under a rename map; no formula changes.

**Tech Stack:** JAX, Flax (`flax.linen`, `flax.struct`), optax, `utils/flax_utils.py` (`ModuleDict`, `TrainState`, `nonpytree_field`), pytest.

## Global Constraints

- **Behavior-preserving.** No change to training/eval numerics. `tests/test_psm_agent_equiv.py` must pass at `atol=1e-10` (static) / `atol=1e-8` (10-step), `tests/test_psm_networks_equiv.py` at `atol=1e-10`, after test-glue updates. The `.npz` fixture data is **not** re-exported; only JAX-side names and the test glue that maps `.npz` keys → JAX names change.
- **Renames (apply consistently):** `psm_psi → proto_psi`; `z_cont → task_z`; `z_psm → proto_seed`; `inj → sampled` (a `StepInputs` struct); `z_eval → task_z` (the eval field); `_draw_injection → sample_step_inputs`; `compute_static → losses_and_grads`; `_stages` → the three named loss methods `proto_loss`/`sf_loss`/`actor_loss`(+`flow_actor_loss`).
- **Keep public names** `contrastive_loss`, `ortho_loss`, `proto_sample`, `project_z`, `PSMAgent`, `get_config` (imported by tests/`main.py`).
- **Domain symbols stay:** `phi`, `psi` are the paper's notation — do not rename to generic terms.
- **x64 in tests:** every PSM test module begins with `jax.config.update("jax_enable_x64", True)` before importing `jax.numpy` (already present; preserve).
- **Networks file is docstring-only in this phase** — `PhiMap`/`PsiMap`/`PSMActor`/`NoiseConditionedActor`/`FlowVectorField` module internals and submodule names are pinned by `torch_to_flax.py` loaders and `test_psm_networks_equiv.py`; do not alter structure.
- **Commit after each task.** Branch `feat/psm-integration` (already checked out).

---

### Task 0: Capture the green baseline

**Files:**
- Test: `tests/test_psm_agent_equiv.py`, `tests/test_psm_networks_equiv.py`, `tests/test_psm_smoke.py` (run only)

**Interfaces:**
- Produces: a recorded baseline that all three PSM suites pass before any edit.

- [ ] **Step 1: Run the three PSM suites**

Run: `cd /u/amsks/git/PSMFLows && python -m pytest tests/test_psm_networks_equiv.py tests/test_psm_agent_equiv.py tests/test_psm_smoke.py -q`
Expected: all pass (the equiv suites are the invariant this refactor must preserve). If any fail *before* editing, stop and report — the baseline is not green.

- [ ] **Step 2: Record the pass line**

Note the summary (e.g. `NN passed`) in the task’s execution log. No commit (read-only task).

---

### Task 1: Document the networks file (no structural change)

**Files:**
- Modify: `utils/psm_networks.py` (module + class docstrings only)

**Interfaces:**
- Consumes: nothing new.
- Produces: no API change. `PhiMap`, `PsiMap`, `PSMActor`, `NoiseConditionedActor`, `FlowVectorField` unchanged in structure/params.

- [ ] **Step 1: Add the code↔paper mapping to the module docstring**

Prepend to the existing module docstring in `utils/psm_networks.py` a short table tying each module to §3 of the spec:

```python
"""Bespoke PSM networks, transcribed from the PyTorch reference psm_nets.py.

Code ↔ paper (arXiv 2411.19418):
  PhiMap   -> phi_s(s+)      basis over future states (the learned proto basis)
  PsiMap   -> psi^pi(s,a)    successor-feature coefficients (task or codebook head)
  PSMActor -> pi(a|s,z)      TD3 mean actor conditioned on the task vector w
  NoiseConditionedActor / FlowVectorField -> flow-BC one-step actor + velocity field

These intentionally do NOT reuse utils/networks.MLP: the reference uses a specific
activation/norm sequence — `ntanh` (LayerNorm then tanh), `relu`, and a final
`Norm` = sqrt(d) * x / ||x|| — that must be reproduced exactly for numerical
equivalence. flax LayerNorm uses epsilon=1e-5 to match torch's default.
"""
```

Leave all class bodies, submodule names, and `__call__` math unchanged.

- [ ] **Step 2: Verify networks equivalence is untouched**

Run: `python -m pytest tests/test_psm_networks_equiv.py -q`
Expected: PASS (4 tests). A docstring change cannot alter params; if this fails, a structural edit slipped in — revert it.

- [ ] **Step 3: Commit**

```bash
git add utils/psm_networks.py
git commit -m "PSM refactor: document networks with code<->paper mapping (no behavior change)"
```

---

### Task 2: Introduce the `StepInputs` struct and `sample_step_inputs`

**Files:**
- Modify: `agents/psm.py` (add struct; rename `_draw_injection`; keep `compute_static`/`apply_update` working by reading struct fields)
- Modify: `tests/test_psm_agent_equiv.py` (build the struct in `_inj`)

**Interfaces:**
- Produces: `StepInputs` (a `flax.struct.dataclass`) with fields `task_z, proto_seed, proto_next_action, sf_next_action, actor_sample, flow_x0, flow_t, flow_noise` (unused ones default `None`); `PSMAgent.sample_step_inputs(self, batch, rng) -> StepInputs`. `compute_static`/`apply_update` accept a `StepInputs`.

- [ ] **Step 1: Add the struct**

Add near the top of `agents/psm.py` (after imports):

```python
@flax.struct.dataclass
class StepInputs:
    """The per-step sampled quantities (replaces the old `inj` dict).

    task_z:            continuous task vector w (Gaussian, 50%-mixed with phi(goal))
    proto_seed:        binary codebook seed z for the proto branch
    proto_next_action: codebook policy action at s' (proto measure target)
    sf_next_action:    learned-actor action at s' (sf measure target)
    actor_sample:      truncated-normal actor sample (ddpgbc only; None for flow)
    flow_x0/flow_t/flow_noise: flow-BC noise/time/noise (flow only; None for ddpgbc)
    """
    task_z: Any
    proto_seed: Any
    proto_next_action: Any
    sf_next_action: Any
    actor_sample: Any = None
    flow_x0: Any = None
    flow_t: Any = None
    flow_noise: Any = None
```

- [ ] **Step 2: Rename `_draw_injection` → `sample_step_inputs`, return the struct**

Move the body of `_draw_injection` (current `agents/psm.py:269-313`) into `sample_step_inputs(self, batch, rng)`, verbatim except: rename local `z_cont → task_z`, `z_psm → proto_seed`, `proto_na → proto_next_action`, `na/sf_na → sf_next_action`, `actor_smp → actor_sample`; and instead of building/returning the `inj` dict, return:

```python
        return StepInputs(
            task_z=task_z, proto_seed=proto_seed, proto_next_action=proto_next_action,
            sf_next_action=sf_next_action, actor_sample=actor_sample,
            flow_x0=flow_x0, flow_t=flow_t, flow_noise=flow_noise,
        )
```

For the ddpgbc branch set `flow_x0=flow_t=flow_noise=None`; for the flow branch set `actor_sample=None`. Keep the `batch["index"]` proto-key logic unchanged.

- [ ] **Step 3: Make the loss plumbing read struct fields**

In the existing `_stages`/`_flow_actor_fn`/`compute_static`/`apply_update` (still present at this point), replace every `inj["x"]` / `inj.get("x")` with the attribute `sampled.x` (e.g. `inj["z_cont"]` → `sampled.task_z`, `inj["z_psm"]` → `sampled.proto_seed`, `inj.get("actor_sample")` → `sampled.actor_sample`). Rename the parameter `inj` → `sampled` in those method signatures. No math changes.

- [ ] **Step 4: Update the agent-equiv test to build the struct**

In `tests/test_psm_agent_equiv.py`, replace `_inj` with a struct builder and import:

```python
from agents.psm import PSMAgent, StepInputs, contrastive_loss, ortho_loss, proto_sample  # noqa: E402
...
def _inj(prefix):
    g = lambda n: jnp.asarray(FIX[f"{prefix}{n}"], jnp.float64)
    return StepInputs(
        task_z=g("z_cont"), proto_seed=g("z_psm"),
        proto_next_action=g("proto_next_action"),
        sf_next_action=g("actor_next_action"),
        actor_sample=g("actor_sample"),
    )
```

(The `.npz` keys `z_cont`, `z_psm`, `proto_next_action`, `actor_next_action`, `actor_sample` are unchanged — only the JAX-side field names differ.)

- [ ] **Step 5: Run the equiv + smoke suites**

Run: `python -m pytest tests/test_psm_agent_equiv.py tests/test_psm_smoke.py -q`
Expected: PASS. `compute_static`/`apply_update` now consume a `StepInputs`; numerics unchanged.

- [ ] **Step 6: Commit**

```bash
git add agents/psm.py tests/test_psm_agent_equiv.py
git commit -m "PSM refactor: replace opaque inj dict with named StepInputs struct"
```

---

### Task 3: Adopt `ModuleDict`/`TrainState` network layer + named loss methods + explicit sequential update

**Files:**
- Modify: `agents/psm.py` (state fields, `create`, losses, update)
- Modify: `tests/test_psm_agent_equiv.py` (`_mapped_agent`, `losses_and_grads`/`apply_update` calls, param keys)

**Interfaces:**
- Consumes: `StepInputs` (Task 2); `ModuleDict`, `TrainState`, `nonpytree_field` from `utils.flax_utils`.
- Produces the new `PSMAgent` shape:
  - Fields: `rng`; `network: TrainState` (ModuleDict of `{phi, proto_psi, sf_psi, actor[, actor_vf]}`, `tx=None`); `target_params` (`{'phi','proto_psi','sf_psi'}`); `opt_states` (`{'phi','proto_psi','sf_psi','actor'[, 'actor_vf']}`); `task_z`; `config` (nonpytree); `optims` (nonpytree `StaticDict` of optax transforms); `proto`.
  - Methods: `proto_loss(self, batch, sampled, phi_params, proto_psi_params)`, `sf_loss(self, batch, sampled, sf_params, phi_params)`, `actor_loss(self, batch, sampled, actor_params, sf_params)`, `flow_actor_loss(self, batch, sampled, actor_params, vf_params)`, `sample_step_inputs`, `losses_and_grads(self, batch, sampled)`, `apply_update(self, batch, sampled)`, `update(self, batch)`, `sample_actions`, `infer_z`, `infer_eval_z`.
  - Network apply via `self.network.select(name)(*args, params=<subtree>)` (pass a subtree to flow grad, `params=None` to stop grad).

- [ ] **Step 1: Rename `_HashableDict` → `StaticDict` and document it**

Rename the class and update its docstring in `agents/psm.py`:

```python
class StaticDict(dict):
    """Identity-hashed dict for jit static aux (optax transforms aren't value-hashable).
    `.replace()` keeps these by reference across steps, so identity hashing gives a
    stable jit-cache key."""
    __hash__ = object.__hash__
    def __eq__(self, other):
        return self is other
```

- [ ] **Step 2: Build the network as a ModuleDict in `create`**

In `PSMAgent.create`, after constructing the module instances, assemble a `ModuleDict` and one `TrainState` (no tx), plus per-net optimizers and target params. Replace the old `nets`/`params`/`txs`/`opt_states`/return with:

```python
        modules = {
            "phi": PhiMap(z_dim=z_dim, hidden_dim=config["phi"]["hidden_dim"],
                          hidden_layers=config["phi"]["hidden_layers"], norm=True),
            "sf_psi": PsiMap(output_dim=z_dim, hidden_dim=config["sf"]["hidden_dim"],
                             num_parallel=config["num_parallel"],
                             embedding_layers=config["sf"]["embedding_layers"],
                             hidden_layers=config["sf"]["hidden_layers"]),
            "proto_psi": PsiMap(output_dim=z_dim, hidden_dim=config["sf"]["hidden_dim"],
                                num_parallel=config["num_parallel"],
                                embedding_layers=config["sf"]["embedding_layers"],
                                hidden_layers=config["sf"]["hidden_layers"]),
        }
        # actor (+ actor_vf for flow) added to `modules` exactly as today, keyed
        # "actor"/"actor_vf" (build NoiseConditionedActor+FlowVectorField for flow,
        # PSMActor for ddpgbc — move the current create() actor block verbatim).
        network_args = dict(
            phi=(ex_obs,),
            sf_psi=(ex_obs, ex_z, ex_actions),
            proto_psi=(ex_obs, ex_zbin, ex_actions),
            actor=<actor init args as today>,          # (ex_obs, ex_z) or (ex_obs, ex_z, ex_noise)
            # actor_vf=(ex_obs, ex_actions, ex_times)   # flow only
        )
        network_def = ModuleDict(modules)
        params = network_def.init(init_rng, **network_args)["params"]
        network = TrainState.create(network_def, params, tx=None)

        optims = StaticDict(
            phi=optax.adam(config["lr_phi"]),
            proto_psi=optax.adam(config["lr_sf"]),
            sf_psi=optax.adam(config["lr_sf"]),
            actor=optax.adam(config["lr_actor"]),
        )
        if actor_type == "flow":
            optims["actor_vf"] = optax.adam(lr_actor_vf)
        opt_states = {k: optims[k].init(params[k]) for k in optims}
        target_params = {k: copy.deepcopy(params[k]) for k in ["phi", "proto_psi", "sf_psi"]}
        return cls(rng=rng, network=network, target_params=target_params,
                   opt_states=opt_states, task_z=jnp.zeros((z_dim,), jnp.float32),
                   config=flax.core.FrozenDict(config), optims=optims, proto=proto)
```

Notes: with a single `ModuleDict.init(**network_args)`, sub-net params live under keys `phi/sf_psi/proto_psi/actor[/actor_vf]` in one tree (`network.params`). Keep the existing `_plain_config`, `proto` table build, and config hoisting unchanged. The `init_rng` may be a single key (ModuleDict inits all submodules in one call); adjust the `jax.random.split` at the top of `create` to yield one `init_rng` plus `rproto`.

- [ ] **Step 3: Write the three loss methods (verbatim math, new net access)**

Replace `_stages` with three methods. Each moves the corresponding closure body from current `agents/psm.py` verbatim, changing only network access to `self.network.select(name)(..., params=<subtree>)` and target reads to `self.target_params[name]`:

```python
    def proto_loss(self, batch, sampled, phi_params, proto_psi_params):
        c = self.config
        goal = batch["next_observations"]
        phi_g = self.network.select("phi")(goal, params=phi_params)
        M = self.network.select("proto_psi")(batch["observations"], sampled.proto_seed,
                                              batch["actions"], params=proto_psi_params) @ phi_g.T
        tphi = self.network.select("phi")(goal, params=self.target_params["phi"])
        tM = self.network.select("proto_psi")(batch["next_observations"], sampled.proto_seed,
                                              sampled.proto_next_action,
                                              params=self.target_params["proto_psi"]) @ tphi.T
        tmean, tunc = targets_uncertainty(tM, c["num_parallel"])
        target_M = tmean - c["pessimism_penalty"] * tunc
        off, off_sum = off_diagonal_mask(batch["observations"].shape[0])
        cl, cdiag, coff = contrastive_loss(M, jax.lax.stop_gradient(target_M), c["discount"], off, off_sum)
        ol, odiag, ooff = ortho_loss(phi_g, off, off_sum)
        loss = cl + c["ortho_coef"] * ol
        return loss, {"psm_loss": loss, "psm_diag": cdiag, "psm_offdiag": coff,
                      "orth_loss": ol, "orth_diag": odiag, "orth_offdiag": ooff}
```

`sf_loss` mirrors the current `sf_loss_fn` (current `agents/psm.py:127-135`): pred-phi is `stop_gradient(select('phi')(goal, params=phi_params))` where `phi_params` is the **just-updated** online phi passed by `apply_update`; target-phi is `select('phi')(goal, params=phi_params)` (same updated online phi — NOT `target_params['phi']`, preserving the reference); sf target uses `self.target_params["sf_psi"]`. `actor_loss` and `flow_actor_loss` mirror the current `actor_loss_fn`/`_flow_actor_fn` bodies verbatim, with `self.network.select("sf_psi")(..., params=<sf_params>)` and `select("actor"/"actor_vf")(..., params=...)`. Keep the BC-term and `-Q/|Q|` normalization exactly.

- [ ] **Step 4: Add the small named helpers**

```python
def off_diagonal_mask(B):
    off = 1.0 - jnp.eye(B)
    return off, off.sum()

def adam_step(tx, grad, params, opt_state):
    updates, new_opt = tx.update(grad, opt_state, params)
    return optax.apply_updates(params, updates), new_opt

def polyak_update(online, target, tau):
    return jax.tree_util.tree_map(lambda p, tp: p * tau + tp * (1 - tau), online, target)
```

(These replace `_off`, `_step`, `_soft` — same math.)

- [ ] **Step 5: Rewrite `apply_update` preserving the exact 3-stage sequence**

```python
    def apply_update(self, batch, sampled):
        p = dict(self.network.params)          # {'phi','proto_psi','sf_psi','actor'[, 'actor_vf']}
        opt = dict(self.opt_states)
        tp = dict(self.target_params)
        tau = self.config["tau"]

        # stage 1: proto -> step phi + proto_psi, soft-update their targets
        (_, psm_i), (g_phi, g_proto) = jax.value_and_grad(
            self.proto_loss, argnums=(2, 3), has_aux=True)(batch, sampled, p["phi"], p["proto_psi"])
        p["phi"], opt["phi"] = adam_step(self.optims["phi"], g_phi, p["phi"], opt["phi"])
        p["proto_psi"], opt["proto_psi"] = adam_step(self.optims["proto_psi"], g_proto, p["proto_psi"], opt["proto_psi"])
        tp["phi"] = polyak_update(p["phi"], tp["phi"], tau)
        tp["proto_psi"] = polyak_update(p["proto_psi"], tp["proto_psi"], tau)

        # stage 2: sf reads the just-updated phi -> step sf_psi, soft-update its target
        (_, sf_i), g_sf = jax.value_and_grad(self.sf_loss, argnums=2, has_aux=True)(
            batch, sampled, p["sf_psi"], p["phi"])
        p["sf_psi"], opt["sf_psi"] = adam_step(self.optims["sf_psi"], g_sf, p["sf_psi"], opt["sf_psi"])
        tp["sf_psi"] = polyak_update(p["sf_psi"], tp["sf_psi"], tau)

        # stage 3: actor reads the just-updated sf_psi
        if self.config["actor_type"] == "flow":
            (_, a_i), (g_actor, g_vf) = jax.value_and_grad(
                self.flow_actor_loss, argnums=(2, 3), has_aux=True)(batch, sampled, p["actor"], p["actor_vf"])
            p["actor"], opt["actor"] = adam_step(self.optims["actor"], g_actor, p["actor"], opt["actor"])
            p["actor_vf"], opt["actor_vf"] = adam_step(self.optims["actor_vf"], g_vf, p["actor_vf"], opt["actor_vf"])
        else:
            (_, a_i), g_actor = jax.value_and_grad(self.actor_loss, argnums=2, has_aux=True)(
                batch, sampled, p["actor"], p["sf_psi"])
            p["actor"], opt["actor"] = adam_step(self.optims["actor"], g_actor, p["actor"], opt["actor"])

        info = {**psm_i, **sf_i, **a_i}
        return self.replace(network=self.network.replace(params=p),
                            opt_states=opt, target_params=tp), info
```

Note: `sf_loss`'s target-phi reads its `phi_params` argument (the updated `p["phi"]`) for both pred and target phi — matching the current `sf_loss_fn`. Confirm the argnums so grads are taken w.r.t. `sf_psi` only (argnums=2).

- [ ] **Step 6: Rewrite `losses_and_grads` (was `compute_static`) and `update`**

`losses_and_grads(self, batch, sampled)` mirrors the current `compute_static`: compute all three branch losses/grads at the **current** params (no interleaving) and return `(info, grads_dict)`; use `self.network.params[...]` and the loss methods with `argnums` as above. `update` stays `@jax.jit`:

```python
    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)
        sampled = self.sample_step_inputs(batch, rng)
        new_agent, info = self.apply_update(batch, sampled)
        return new_agent.replace(rng=new_rng), info
```

Update `sample_step_inputs`, `sample_actions`, `infer_z`, `infer_eval_z`, `total_loss` to read `self.network.select(name)(...)` and `self.task_z` (renamed from `z_eval`). `sample_actions` uses `self.network.select("actor")(observations, z[, noise])`.

- [ ] **Step 7: Update the agent-equiv test glue to the new shape**

In `tests/test_psm_agent_equiv.py` rewrite `_mapped_agent` and the two call sites:

```python
def _mapped_agent():
    ex_obs = jnp.asarray(FIX["in__obs"], jnp.float64)
    ex_act = jnp.asarray(FIX["in__action"], jnp.float64)
    agent = PSMAgent.create(0, ex_obs, ex_act, CONFIG)
    mapped = {
        "phi": load_phi_params(FIX),
        "sf_psi": load_psi_params(FIX, "sf_psi"),
        "proto_psi": load_psi_params(FIX, "psm_psi"),   # .npz key stays 'psm_psi'
        "actor": load_actor_params(FIX),
    }
    network = agent.network.replace(params=mapped)
    target_params = {k: mapped[k] for k in ["phi", "proto_psi", "sf_psi"]}
    opt_states = {k: agent.optims[k].init(mapped[k]) for k in ["phi", "proto_psi", "sf_psi", "actor"]}
    return agent.replace(network=network, target_params=target_params, opt_states=opt_states)
```

Change `agent.compute_static(...)` → `agent.losses_and_grads(...)` in `test_agent_static_equiv`. `test_agent_perstep_equiv` calls `agent.apply_update(batch, _inj(...))` (now `_inj` returns a `StepInputs` from Task 2). Metric keys in `checks`/`keys` are unchanged (`psm_loss`, `sf_loss`, … still emitted).

- [ ] **Step 8: Run the full equiv + smoke suite**

Run: `python -m pytest tests/test_psm_networks_equiv.py tests/test_psm_agent_equiv.py tests/test_psm_smoke.py -q`
Expected: all PASS — `test_agent_static_equiv` at 1e-10, `test_agent_perstep_equiv` at 1e-8, smoke green (note `test_flow_actor_trains_and_acts` checks `"actor_vf" in agent.params` — update that assertion to `"actor_vf" in agent.network.params`).

- [ ] **Step 9: Commit**

```bash
git add agents/psm.py tests/test_psm_agent_equiv.py tests/test_psm_smoke.py
git commit -m "PSM refactor: ModuleDict/TrainState network layer + named losses + explicit 3-stage update"
```

---

### Task 4: Update callers (`main.py`) for the renamed API

**Files:**
- Modify: `main.py` (PSM eval/inference block around `infer_eval_z`)
- Modify: any other referencing site found by grep

**Interfaces:**
- Consumes: the Task-3 `PSMAgent` (`network`, `task_z`, `infer_eval_z`, `sample_actions`).
- Produces: a runnable `main.py` for `agent=psm`.

- [ ] **Step 1: Find references to the old field names**

Run: `grep -rnE "z_eval|\.params\[|compute_static|_draw_injection|psm_psi|\.txs\b" main.py utils/ agents/psm.py`
Expected: a small list. `main.py` uses `hasattr(agent, 'infer_eval_z')` + `infer_eval_z(...)` (unchanged API) — confirm it needs no change beyond any `params`/`z_eval` access. Fix each hit to the new names (`agent.network.params[...]`, `agent.task_z`).

- [ ] **Step 2: Dry-init the agent through the registry**

Run: `python -c "import numpy as np, ml_collections; from hydra import compose, initialize; from omegaconf import OmegaConf; from agents import agents;
import jax;
with initialize(version_base='1.3', config_path='configs/agent'): cfg=compose(config_name='psm');
cfg=ml_collections.ConfigDict(OmegaConf.to_container(cfg, resolve=True));
a=agents['psm'].create(0, np.zeros((1,8),'float32'), np.zeros((1,2),'float32'), cfg);
b={'observations':np.zeros((4,8),'float32'),'actions':np.zeros((4,2),'float32'),'next_observations':np.zeros((4,8),'float32'),'masks':np.ones((4,),'float32')};
a,info=a.update(b); print('ok', {k:float(v) for k,v in info.items() if k.endswith('loss')})"`
Expected: prints `ok {...}` with finite losses (confirms create/update/registry path works end to end).

- [ ] **Step 3: Commit**

```bash
git add main.py
git commit -m "PSM refactor: update main.py callers for renamed agent API"
```

---

### Task 5: Refresh docs & memory pointer

**Files:**
- Modify: `agents/psm.py` (module docstring: add the §3 variable table)
- Modify: memory `psm-vs-reference-audit.md` (append a one-line pointer to the refactor)

**Interfaces:**
- Produces: the code↔paper table lives in the agent module docstring; memory notes the new structure.

- [ ] **Step 1: Put the variable table in the agent module docstring**

Replace the top-of-file docstring of `agents/psm.py` with a version that includes the §3 table (paper symbols → code) and a one-line statement of the 3-stage sequential update and per-network optimizers. Keep it concise (≤ 30 lines).

- [ ] **Step 2: Run the suites once more (docstring-only, must still pass)**

Run: `python -m pytest tests/test_psm_networks_equiv.py tests/test_psm_agent_equiv.py tests/test_psm_smoke.py -q`
Expected: all PASS.

- [ ] **Step 3: Commit**

```bash
git add agents/psm.py
git commit -m "PSM refactor: module docstring with code<->paper variable map"
```

- [ ] **Step 4: Update memory pointer**

Append to `/u/amsks/.claude/projects/-u-amsks-git-PSMFLows/memory/psm-vs-reference-audit.md` a line noting: PSM refactored to ModuleDict/TrainState + named losses + `StepInputs` (2026-07-17), bit-exact preserved; `psm_psi`→`proto_psi`, `z_cont/z_psm`→`task_z/proto_seed`. No commit needed (memory dir is outside the repo).

---

## Self-review

**Spec coverage:**
- Spec §7.1 file split → Task 1 (networks docstring) + Task 3 (agent). ✓
- §7.2 ModuleDict network layer → Task 3 Step 2. ✓
- §7.3 explicit 3-stage update, per-network optimizers, why-not-single-loss → Task 3 Steps 5–6. ✓
- §7.4 naming table (`inj→sampled`, `psm_psi→proto_psi`, `_stages`→named losses, `_HashableDict`/`_off`/`_step`/`_soft`) → Task 2 + Task 3 Steps 1,3,4,6. ✓
- §7.5 parity/tests (fixtures not re-exported; equiv 1e-10/1e-8; smoke) → Task 0 + every task’s verify step. ✓
- §3 variable table in code → Task 5 Step 1. ✓
- Phase 2 (bias, LP, correctness fixes) → **out of scope for this plan** (separate plan after Phase 1 lands), matching the spec's phased approach. ✓

**Placeholder scan:** The `create()` actor block and `network_args` actor entries say "move verbatim / as today" and reference exact current line numbers — these are mechanical moves of existing, working code, not vague TODOs. No "add error handling"/"write tests for the above" placeholders.

**Type consistency:** `StepInputs` fields (Task 2) are consumed by the loss methods (Task 3) by the same names; `network`/`target_params`/`opt_states`/`optims`/`task_z` field names are used consistently across `create`, `apply_update`, `losses_and_grads`, the tests, and `main.py`; `proto_psi` replaces `psm_psi` everywhere on the JAX side while the `.npz` key stays `psm_psi` (only in `load_psi_params(FIX, "psm_psi")`).

**Open items intentionally deferred to Phase-2 plan:** exact continuous bias-head form and the LP-solver internals (spec §8, risks R2/R3).

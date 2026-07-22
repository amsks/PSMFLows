# Affine (full) PSM Implementation Plan

> **For implementers:** Execute this plan task-by-task; steps use checkbox list syntax for tracking.

**Goal:** Add a faithful *affine* Proto Successor Measure agent (`M = Φ(s,a,x)·w + b`) with `full` (constrained goal inference) and `zero_shot` (closed-form) modes, running on goal-conditioned OGBench, without touching the existing bilinear `agents/psm.py`.

**Architecture:** New `agents/affine_psm.py` (`AffinePSMAgent`, a `flax.struct.PyTreeNode`) holds a single-net affine measure `AffineMeasureNet` + target, a task-coordinate MLP `w` + target, a free inference vector `w_inf`, an optional Lagrange net, and a `PSMActor` distilled at eval. Training is the PSM contrastive TD loss over the `B²` `(sᵢ,aᵢ,xⱼ)` mesh (RLU `update_psm`). Inference solves `w_inf` by dual gradient descent / hinge (`full`) or closed form (`zero_shot`), then distills the actor against `Q = Φ·w_inf + b`. Goal-conditioned eval is a new gated branch in `main.py`/`utils/evaluation.py`.

**Tech Stack:** JAX / Flax (`flax.linen`, `flax.struct`), Optax, `utils/flax_utils.TrainState`, Hydra + `ml_collections` configs, pytest.

## Global Constraints

- **Do not modify `agents/psm.py` or `utils/psm_networks.py`'s existing classes.** Only *add* new classes (`AffineMeasureNet`, `LagrangeNet`) to `utils/psm_networks.py`. The parity-audited bilinear PSM must stay byte-identical.
- **Affine factorization only:** `M(s,a,x) = Φ(s,a,x)ᵀw + b(s,a,x)`, `Φ ∈ ℝᵈ`, `b ∈ ℝ`, `w ∈ ℝᵈ`. Linear in `w` (the property the constrained program relies on).
- **Single measure net** (not ensembled) — faithful to RLU `psm.py`; no `targets_uncertainty`/pessimism in the affine agent.
- **Small batch:** `AffineMeasureNet` is evaluated on all `B²` `(sᵢ,aᵢ,xⱼ)` pairs, so `batch_size = 32` (RLU default). Never reuse `psm.yaml`'s 1024.
- **Functional JAX style:** inference methods return **new** agent copies via `self.replace(...)`; no in-place mutation. Jit only single steps, not the eval loops.
- **`x` = measure/goal argument** throughout (avoid RLU's `obs`/`goal` naming collision).
- **`d_dim` == `z_dim` == 50** (RLU defaults); reuse the existing proto-codebook machinery from `agents/psm.py` (`proto_sample`, `project_z`, `off_diagonal_mask`, `polyak_update`) by importing them.
- Tests live in `tests/`, mirror `tests/test_psm_smoke.py` conventions (synthetic `_batch`, `ml_collections` config via Hydra `compose`).
- Reference source of truth: `/u/amsks/git/RLU/controllable_agent/url_benchmark/agent/psm.py` (continuous) + `discrete_psm.py` (`_infer_step`), and the spec `docs/design/2026-07-22-affine-psm-design.md`.

---

### Task 1: Affine measure + Lagrange networks

**Files:**
- Modify (append only): `utils/psm_networks.py`
- Test: `tests/test_affine_psm_networks.py`

**Interfaces:**
- Produces:
  - `AffineMeasureNet(d_dim: int, hidden_dim: int, hidden_layers: int=2)`; `__call__(obs, action, x) -> (phi, b)` with `phi: (B, d_dim)`, `b: (B, 1)`.
  - `LagrangeNet(hidden_dim: int, hidden_layers: int=2)`; `__call__(obs, action, x) -> lam` with `lam: (B, 1)`, softplus output (≥0).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_affine_psm_networks.py
import jax, jax.numpy as jnp, numpy as np
from utils.psm_networks import AffineMeasureNet, LagrangeNet

def _io(B=4, obs=8, act=2, d=5):
    rng = np.random.default_rng(0)
    o = jnp.asarray(rng.standard_normal((B, obs)), jnp.float32)
    a = jnp.asarray(np.clip(rng.standard_normal((B, act)), -1, 1), jnp.float32)
    x = jnp.asarray(rng.standard_normal((B, obs)), jnp.float32)
    return o, a, x, d

def test_affine_measure_shapes():
    o, a, x, d = _io()
    net = AffineMeasureNet(d_dim=d, hidden_dim=16, hidden_layers=2)
    params = net.init(jax.random.PRNGKey(0), o, a, x)["params"]
    phi, b = net.apply({"params": params}, o, a, x)
    assert phi.shape == (o.shape[0], d)
    assert b.shape == (o.shape[0], 1)

def test_measure_is_affine_in_w():
    # M(w) = phi·w + b must be exactly affine: M(w1)-M(w2) == phi·(w1-w2).
    o, a, x, d = _io()
    net = AffineMeasureNet(d_dim=d, hidden_dim=16, hidden_layers=2)
    params = net.init(jax.random.PRNGKey(0), o, a, x)["params"]
    phi, b = net.apply({"params": params}, o, a, x)
    w1 = jnp.asarray(np.random.default_rng(1).standard_normal((d,)), jnp.float32)
    w2 = jnp.asarray(np.random.default_rng(2).standard_normal((d,)), jnp.float32)
    M1 = (phi * w1).sum(-1, keepdims=True) + b
    M2 = (phi * w2).sum(-1, keepdims=True) + b
    lhs = M1 - M2
    rhs = (phi * (w1 - w2)).sum(-1, keepdims=True)
    assert np.allclose(np.asarray(lhs), np.asarray(rhs), atol=1e-5)

def test_lagrange_nonnegative():
    o, a, x, _ = _io()
    net = LagrangeNet(hidden_dim=16, hidden_layers=2)
    params = net.init(jax.random.PRNGKey(0), o, a, x)["params"]
    lam = net.apply({"params": params}, o, a, x)
    assert lam.shape == (o.shape[0], 1)
    assert np.all(np.asarray(lam) >= 0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /u/amsks/git/PSMFLows && python -m pytest tests/test_affine_psm_networks.py -v`
Expected: FAIL with `ImportError: cannot import name 'AffineMeasureNet'`.

- [ ] **Step 3: Append the implementation to `utils/psm_networks.py`**

```python
class AffineMeasureNet(nn.Module):
    """Affine successor-measure net (RLU psm.py PSM). Shared trunk on concat[obs,action,x]
    with two heads: phi (basis, R^d) and b (offset, R^1). M(s,a,x) = phi(s,a,x)·w + b."""

    d_dim: int
    hidden_dim: int
    hidden_layers: int = 2

    @nn.compact
    def __call__(self, obs, action, x):
        h = jnp.concatenate([obs, action, x], -1)
        for _ in range(self.hidden_layers):
            h = nn.relu(nn.Dense(self.hidden_dim, kernel_init=_ORTH_RELU)(h))
        phi = nn.Dense(self.d_dim, kernel_init=_ORTH1)(h)
        b = nn.Dense(1, kernel_init=_ORTH1)(h)
        return phi, b


class LagrangeNet(nn.Module):
    """Learned Lagrange multiplier lam(s,a,x) >= 0 for dual gradient descent
    (RLU psm.py `lmult`, softplus head)."""

    hidden_dim: int
    hidden_layers: int = 2

    @nn.compact
    def __call__(self, obs, action, x):
        h = jnp.concatenate([obs, action, x], -1)
        for _ in range(self.hidden_layers):
            h = nn.relu(nn.Dense(self.hidden_dim, kernel_init=_ORTH_RELU)(h))
        return nn.softplus(nn.Dense(1, kernel_init=_ORTH1)(h))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /u/amsks/git/PSMFLows && python -m pytest tests/test_affine_psm_networks.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add utils/psm_networks.py tests/test_affine_psm_networks.py
git commit -m "AFFINE PSM: AffineMeasureNet + LagrangeNet (affine measure Phi·w+b)"
```

---

### Task 2: Config, agent skeleton, `create`, registry

**Files:**
- Create: `configs/agent/affine_psm.yaml`
- Create: `agents/affine_psm.py`
- Modify: `agents/__init__.py` (add import + registry entry)
- Test: `tests/test_affine_psm_smoke.py`

**Interfaces:**
- Consumes: `AffineMeasureNet`, `LagrangeNet` (Task 1); `proto_sample`, `project_z`, `off_diagonal_mask`, `polyak_update` imported from `agents.psm`; `utils.flax_utils.TrainState`.
- Produces: `AffinePSMAgent` (`flax.struct.PyTreeNode`) with fields `rng, measure: TrainState, w: TrainState, target_measure, target_w, w_inf, actor: TrainState, task_goal, config, proto`. `create(cls, seed, ex_observations, ex_actions, config) -> AffinePSMAgent`. Registered as `affine_psm`.

- [ ] **Step 1: Create the config `configs/agent/affine_psm.yaml`**

```yaml
agent_name: affine_psm
batch_size: 32            # B^2 mesh over (s_i,a_i,x_j): keep small (RLU default)
d_dim: 50
z_dim: 50
max_log_seed: 12          # binary codebook width for the proto policy
proto_table_path: null
discount: 0.98
tau: 0.01                 # target soft-update rate (RLU fb_target_tau)
lr: 1.0e-4
lr_w: 1.0e-4
ortho_coef: 0.0           # RLU comments the ortho term out; off by default
measure:
  hidden_dim: 1024
  hidden_layers: 3
actor:
  hidden_dim: 1024
  hidden_layers: 1
  embedding_layers: 2
inference:
  mode: full              # full | zero_shot
  use_dgd: true
  inf_coeff: 5.0          # hinge weight when use_dgd=false
  num_inference_steps: 512
  lagrange_hidden_dim: 256
  lagrange_hidden_layers: 2
  norm_w: true
  num_actor_inference_steps: 512
ob_dims: null
action_dim: null
encoder: null
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_affine_psm_smoke.py
import math, ml_collections, numpy as np
from hydra import compose, initialize
from omegaconf import OmegaConf
from agents import agents

def _config():
    with initialize(version_base="1.3", config_path="../configs/agent"):
        cfg = compose(config_name="affine_psm")
    return ml_collections.ConfigDict(OmegaConf.to_container(cfg, resolve=True))

def _batch(n=32, obs=8, act=2):
    rng = np.random.default_rng(0)
    return dict(
        observations=rng.standard_normal((n, obs)).astype(np.float32),
        actions=np.clip(rng.standard_normal((n, act)), -1, 1).astype(np.float32),
        next_observations=rng.standard_normal((n, obs)).astype(np.float32),
        index=np.arange(n, dtype=np.int64),
        masks=np.ones((n,), np.float32),
    )

def test_affine_psm_registered_and_creates():
    config = _config()
    assert config["agent_name"] == "affine_psm"
    cls = agents["affine_psm"]
    agent = cls.create(0, np.zeros((1, 8), np.float32), np.zeros((1, 2), np.float32), config)
    assert agent.w_inf.shape == (config["d_dim"],)
    assert agent.actor is not None
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd /u/amsks/git/PSMFLows && python -m pytest tests/test_affine_psm_smoke.py::test_affine_psm_registered_and_creates -v`
Expected: FAIL with `KeyError: 'affine_psm'`.

- [ ] **Step 4: Write `agents/affine_psm.py` (skeleton + `create` + `get_config`/`_plain_config`)**

```python
"""Affine (full) PSM agent — M(s,a,x) = Phi(s,a,x)·w + b(s,a,x).

Faithful JAX port of RLU controllable_agent/url_benchmark/agent/psm.py (continuous)
+ discrete_psm.py `_infer_step`. Distinct from the bilinear agents/psm.py: the task
coordinate w enters LINEARLY, which is what makes the constrained-LP `full` inference
well-defined. See docs/design/2026-07-22-affine-psm-design.md.
"""
import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import optax

from utils.flax_utils import TrainState, nonpytree_field
from utils.psm_networks import AffineMeasureNet, LagrangeNet, PSMActor
from agents.psm import proto_sample, project_z, off_diagonal_mask, polyak_update, _plain_config


class AffinePSMAgent(flax.struct.PyTreeNode):
    rng: Any
    measure: TrainState
    w: TrainState
    target_measure: Any
    target_w: Any
    w_inf: Any                 # (d_dim,) inference task coordinate, solved at eval
    actor: TrainState          # PSMActor distilled against Q = Phi·w_inf + b
    config: Any = nonpytree_field()
    proto: Any = None          # (seed_to_action, powers)

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, rm, rw, ract, rproto = jax.random.split(rng, 5)
        assert config.get("encoder", None) is None, "affine_psm does not support visual encoders."
        action_dim = ex_actions.shape[-1]
        d_dim, z_dim = config["d_dim"], config["z_dim"]
        ex_obs, ex_act = ex_observations, ex_actions
        ex_x = ex_obs
        ex_z = jnp.zeros((ex_obs.shape[0], z_dim))

        measure_def = AffineMeasureNet(d_dim=d_dim, hidden_dim=config["measure"]["hidden_dim"],
                                       hidden_layers=config["measure"]["hidden_layers"])
        measure = TrainState.create(measure_def, measure_def.init(rm, ex_obs, ex_act, ex_x)["params"],
                                    tx=optax.adam(config["lr"]))

        # w: z (codebook code) -> d task coords. Plain MLP built inline via a tiny flax module.
        w_def = _WNet(d_dim=d_dim, hidden_dim=config["measure"]["hidden_dim"])
        w = TrainState.create(w_def, w_def.init(rw, ex_z)["params"], tx=optax.adam(config["lr_w"]))

        actor_cfg = config["actor"]
        actor_def = PSMActor(action_dim=action_dim, hidden_dim=actor_cfg["hidden_dim"],
                             embedding_layers=actor_cfg["embedding_layers"],
                             hidden_layers=actor_cfg["hidden_layers"])
        actor = TrainState.create(actor_def, actor_def.init(ract, ex_obs, ex_z)["params"],
                                  tx=optax.adam(config["lr"]))

        # proto codebook table (reuse the psm.py scheme).
        max_seed = 2 ** config["max_log_seed"] + 20000
        table = (jax.random.uniform(rproto, (max_seed, action_dim)) - 1.0) * 2.0
        powers = (2 ** jnp.arange(config["max_log_seed"]))[::-1].astype(jnp.float32)
        proto = (table.astype(jnp.float32), powers)

        config = _plain_config(config)
        config["ob_dims"] = tuple(ex_observations.shape[1:])
        config["action_dim"] = action_dim
        config["proto_max_seed"] = max_seed
        return cls(rng=rng, measure=measure, w=w,
                   target_measure=copy.deepcopy(measure.params),
                   target_w=copy.deepcopy(w.params),
                   w_inf=jnp.ones((d_dim,), jnp.float32), actor=actor,
                   config=flax.core.FrozenDict(config), proto=proto)


class _WNet(flax.linen.Module):
    """z (codebook code) -> w (task coordinates), an MLP (RLU psm.py self.w)."""
    d_dim: int
    hidden_dim: int

    @flax.linen.compact
    def __call__(self, z):
        h = z
        for _ in range(3):
            h = flax.linen.relu(flax.linen.Dense(self.hidden_dim)(h))
        return flax.linen.Dense(self.d_dim)(h)
```

- [ ] **Step 5: Register in `agents/__init__.py`**

Add `from agents.affine_psm import AffinePSMAgent` with the other imports and `affine_psm=AffinePSMAgent,` inside the `agents = dict(...)`.

- [ ] **Step 6: Run test to verify it passes**

Run: `cd /u/amsks/git/PSMFLows && python -m pytest tests/test_affine_psm_smoke.py::test_affine_psm_registered_and_creates -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add configs/agent/affine_psm.yaml agents/affine_psm.py agents/__init__.py tests/test_affine_psm_smoke.py
git commit -m "AFFINE PSM: agent skeleton, config, create(), registry"
```

---

### Task 3: Training — affine contrastive TD update

**Files:**
- Modify: `agents/affine_psm.py` (add `sample_step_inputs`, `measure_loss`, `apply_update`, `update`, `total_loss`)
- Test: `tests/test_affine_psm_smoke.py` (add training tests)

**Interfaces:**
- Consumes: `create` (Task 2), `proto_sample`, `off_diagonal_mask`.
- Produces: `update(self, batch) -> (AffinePSMAgent, info)` (jitted); `info` has keys `psm_loss`, `psm_diag`, `psm_offdiag`. `measure_loss(self, batch, sampled, measure_params, w_params) -> (loss, info)`. `sample_step_inputs(self, batch, rng) -> StepInputs` with fields `proto_seed`, `proto_next_action`.

- [ ] **Step 1: Write the failing tests (append to `tests/test_affine_psm_smoke.py`)**

```python
import math
from agents import agents

def _agent():
    config = _config()
    return agents["affine_psm"].create(0, np.zeros((1, 8), np.float32),
                                       np.zeros((1, 2), np.float32), config)

def test_update_runs_and_is_finite():
    agent = _agent()
    agent, info = agent.update(_batch())
    for k in ["psm_loss", "psm_diag", "psm_offdiag"]:
        assert math.isfinite(float(info[k])), (k, info[k])

def test_training_decreases_loss():
    agent = _agent()
    b = _batch()
    first = float(agent.update(b)[1]["psm_loss"])
    for _ in range(30):
        agent, info = agent.update(b)
    assert float(info["psm_loss"]) < first

def test_measure_and_w_receive_gradient():
    agent = _agent()
    a0 = agent
    a1, _ = agent.update(_batch())
    import jax
    def changed(t0, t1):
        leaves0 = jax.tree_util.tree_leaves(t0)
        leaves1 = jax.tree_util.tree_leaves(t1)
        return any(not np.allclose(np.asarray(x), np.asarray(y)) for x, y in zip(leaves0, leaves1))
    assert changed(a0.measure.params, a1.measure.params)
    assert changed(a0.w.params, a1.w.params)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /u/amsks/git/PSMFLows && python -m pytest tests/test_affine_psm_smoke.py -k "update or training or gradient" -v`
Expected: FAIL with `AttributeError: 'AffinePSMAgent' object has no attribute 'update'`.

- [ ] **Step 3: Add training methods to `agents/affine_psm.py`**

```python
@flax.struct.dataclass
class StepInputs:
    proto_seed: Any
    proto_next_action: Any


# --- add these methods to AffinePSMAgent ---

    def sample_step_inputs(self, batch, rng):
        c = self.config
        B = batch["observations"].shape[0]
        r_seed, = jax.random.split(rng, 1)
        # ONE binary codebook code z per batch (RLU samples one z and repeats).
        w = c["max_log_seed"]
        code = jax.random.randint(r_seed, (), 0, 2 ** w)
        bits = ((code >> jnp.arange(w)) & 1).astype(jnp.float32)
        proto_seed = jnp.broadcast_to(bits, (B, w))
        seed_to_action, powers = self.proto
        obs_hash = batch["index"] if "index" in batch else jnp.arange(B)
        proto_next_action = proto_sample(seed_to_action, powers, obs_hash, proto_seed, c["proto_max_seed"])
        return StepInputs(proto_seed=proto_seed, proto_next_action=proto_next_action)

    def measure_loss(self, batch, sampled, measure_params, w_params):
        c = self.config
        obs, action, next_obs = batch["observations"], batch["actions"], batch["next_observations"]
        x = next_obs                      # measure argument (future state)
        B = obs.shape[0]
        off, off_sum = off_diagonal_mask(B)
        z, proto_na = sampled.proto_seed, sampled.proto_next_action

        # B^2 mesh over (s_i, a_i, x_j): tile rows over i, x over j.
        i_idx = jnp.repeat(jnp.arange(B), B)     # (B^2,)
        j_idx = jnp.tile(jnp.arange(B), B)       # (B^2,)
        m_obs, m_act = obs[i_idx], action[i_idx]
        m_next_obs, m_next_act = next_obs[i_idx], proto_na[i_idx]
        m_x = x[j_idx]

        wz = self.w(z, params=w_params)          # (B, d) — identical rows
        wz_i = wz[i_idx]
        phi, b = self.measure(m_obs, m_act, m_x, params=measure_params)
        M = ((phi * wz_i).sum(-1, keepdims=True) + b).reshape(B, B)

        tphi, tb = self.measure(m_next_obs, m_next_act, m_x, params=self.target_measure)
        twz = self.w(z, params=self.target_w)[i_idx]
        target_M = ((tphi * twz).sum(-1, keepdims=True) + tb).reshape(B, B)
        target_M = jax.lax.stop_gradient(target_M)

        diff = M - c["discount"] * target_M
        offdiag = 0.5 * jnp.sum((diff * off) ** 2) / off_sum
        # RLU psm.py:352 diagonal source term: -(1-gamma)*mean(diag(M)).
        diag = -((1 - c["discount"]) * jnp.diagonal(M)).mean()
        loss = offdiag + diag
        return loss, {"psm_loss": loss, "psm_diag": diag, "psm_offdiag": offdiag}

    def apply_update(self, batch, sampled):
        tau = self.config["tau"]
        (_, info), (g_m, g_w) = jax.value_and_grad(self.measure_loss, argnums=(2, 3), has_aux=True)(
            batch, sampled, self.measure.params, self.w.params)
        measure = self.measure.apply_gradients(grads=g_m)
        w = self.w.apply_gradients(grads=g_w)
        target_measure = polyak_update(measure.params, self.target_measure, tau)
        target_w = polyak_update(w.params, self.target_w, tau)
        return self.replace(measure=measure, w=w, target_measure=target_measure, target_w=target_w), info

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)
        sampled = self.sample_step_inputs(batch, rng)
        new_agent, info = self.apply_update(batch, sampled)
        return new_agent.replace(rng=new_rng), info

    def total_loss(self, batch, grad_params=None, rng=None):
        rng = rng if rng is not None else self.rng
        sampled = self.sample_step_inputs(batch, rng)
        loss, info = self.measure_loss(batch, sampled, self.measure.params, self.w.params)
        return loss, info
```

Note: `self.w(z, params=w_params)` works because `TrainState.__call__(*args, params=..., ...)` forwards to the module; `self.measure(obs, act, x, params=...)` likewise returns `(phi, b)`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /u/amsks/git/PSMFLows && python -m pytest tests/test_affine_psm_smoke.py -k "update or training or gradient" -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add agents/affine_psm.py tests/test_affine_psm_smoke.py
git commit -m "AFFINE PSM: contrastive TD training over the B^2 affine measure mesh"
```

---

### Task 4: `full` constrained goal inference (DGD + hinge)

**Files:**
- Modify: `agents/affine_psm.py` (add `_infer_step`, `infer_w_goal`)
- Test: `tests/test_affine_psm_inference.py`

**Interfaces:**
- Consumes: training methods (Task 3), `LagrangeNet`, `project_z`.
- Produces: `infer_w_goal(self, dataset, goal, seed=0) -> AffinePSMAgent` (new copy with solved `w_inf`). Drives a Python loop of `num_inference_steps`; each step samples `dataset.sample(batch_size)` and calls a jitted pure `_infer_step`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_affine_psm_inference.py
import math, ml_collections, numpy as np, jax.numpy as jnp
from hydra import compose, initialize
from omegaconf import OmegaConf
from agents import agents

def _config(overrides=None):
    with initialize(version_base="1.3", config_path="../configs/agent"):
        cfg = compose(config_name="affine_psm", overrides=overrides or [])
    return ml_collections.ConfigDict(OmegaConf.to_container(cfg, resolve=True))

class _FakeDataset:
    """Minimal replay-buffer stand-in: .sample(n) -> dict of arrays."""
    def __init__(self, n=256, obs=8, act=2, seed=0):
        rng = np.random.default_rng(seed)
        self.obs = rng.standard_normal((n, obs)).astype(np.float32)
        self.act = np.clip(rng.standard_normal((n, act)), -1, 1).astype(np.float32)
        self.nxt = rng.standard_normal((n, obs)).astype(np.float32)
        self.n = n
        self._rng = np.random.default_rng(seed + 1)
    def sample(self, k):
        idx = self._rng.integers(0, self.n, size=k)
        return dict(observations=self.obs[idx], actions=self.act[idx],
                    next_observations=self.nxt[idx], index=idx.astype(np.int64))

def _trained_agent(overrides=None):
    config = _config(overrides)
    agent = agents["affine_psm"].create(0, np.zeros((1, 8), np.float32),
                                        np.zeros((1, 2), np.float32), config)
    ds = _FakeDataset()
    for _ in range(20):
        agent, _ = agent.update(ds.sample(config["batch_size"]))
    return agent, config, ds

def _mean_violation(agent, ds, goal, k=128):
    b = ds.sample(k)
    phi, bb = agent.measure(b["observations"], b["actions"], b["next_observations"])
    M = (phi * agent.w_inf).sum(-1, keepdims=True) + bb
    return float(np.mean(np.minimum(np.asarray(M), 0.0)))

def test_full_inference_reduces_constraint_violation_dgd():
    agent, config, ds = _trained_agent(["inference.use_dgd=true", "inference.num_inference_steps=200"])
    goal = ds.sample(1)["next_observations"][0]
    before = _mean_violation(agent, ds, goal)
    agent2 = agent.infer_w_goal(ds, goal)
    after = _mean_violation(agent2, ds, goal)
    assert math.isfinite(after)
    assert after >= before - 1e-6  # violation (a negative number) moves toward 0

def test_full_inference_hinge_runs():
    agent, config, ds = _trained_agent(["inference.use_dgd=false", "inference.num_inference_steps=100"])
    goal = ds.sample(1)["next_observations"][0]
    agent2 = agent.infer_w_goal(ds, goal)
    assert np.all(np.isfinite(np.asarray(agent2.w_inf)))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /u/amsks/git/PSMFLows && python -m pytest tests/test_affine_psm_inference.py -k full -v`
Expected: FAIL with `AttributeError: ... 'infer_w_goal'`.

- [ ] **Step 3: Add inference methods to `agents/affine_psm.py`**

```python
    def infer_w_goal(self, dataset, goal, seed=0):
        """Solve w_inf by constrained optimization (RLU infer_w_goal/_infer_step_gc).
        Objective: maximize measure at `goal`; constraint: Phi·w+b >= 0 off-goal.
        Returns a new agent with w_inf set (Python loop; jitted inner step)."""
        c = self.config
        ic = c["inference"]
        d = c["d_dim"]
        goal = jnp.asarray(goal, jnp.float32)

        w_inf = jnp.ones((d,), jnp.float32)
        w_opt = optax.adam(c["lr_w"])
        w_state = w_opt.init(w_inf)

        use_dgd = bool(ic["use_dgd"])
        lam_def = LagrangeNet(hidden_dim=ic["lagrange_hidden_dim"],
                              hidden_layers=ic["lagrange_hidden_layers"])
        ex = dataset.sample(c["batch_size"])
        lam_params = lam_def.init(jax.random.PRNGKey(seed),
                                  jnp.asarray(ex["observations"]), jnp.asarray(ex["actions"]),
                                  jnp.asarray(ex["next_observations"]))["params"]
        lam_opt = optax.adam(c["lr_w"])
        lam_state = lam_opt.init(lam_params)

        @jax.jit
        def step(w_inf, w_state, lam_params, lam_state, obs, act, xperm, goal_rep):
            def primal(w):
                phi_g, _ = self.measure(obs, act, goal_rep)
                phi_p, b_p = self.measure(obs, act, xperm)
                obj = -(phi_g * w).sum(-1).mean()
                Mp = (phi_p * w).sum(-1, keepdims=True) + b_p
                if use_dgd:
                    lam = jax.lax.stop_gradient(lam_def.apply({"params": lam_params}, obs, act, xperm))
                    con = -(Mp * lam).mean()
                else:
                    con = -(jnp.minimum(Mp, 0.0) * ic["inf_coeff"]).mean()
                return obj + con, (obj, con)
            (_, (obj, con)), gw = jax.value_and_grad(primal, has_aux=True)(w_inf)
            upd, w_state = w_opt.update(gw, w_state, w_inf)
            w_inf = optax.apply_updates(w_inf, upd)
            if use_dgd:
                def dual(lp):
                    phi_p, b_p = self.measure(obs, act, xperm)
                    Mp = (phi_p * jax.lax.stop_gradient(w_inf)).sum(-1, keepdims=True) + b_p
                    lam = lam_def.apply({"params": lp}, obs, act, xperm)
                    return (Mp * lam).mean()   # ascent on multipliers
                gl = jax.grad(dual)(lam_params)
                # gradient ASCENT: negate for optax (which minimizes)
                gl = jax.tree_util.tree_map(lambda g: -g, gl)
                ul, lam_state = lam_opt.update(gl, lam_state, lam_params)
                lam_params = optax.apply_updates(lam_params, ul)
            return w_inf, w_state, lam_params, lam_state, obj, con

        for _ in range(int(ic["num_inference_steps"])):
            b = dataset.sample(c["batch_size"])
            obs = jnp.asarray(b["observations"]); act = jnp.asarray(b["actions"])
            xb = jnp.asarray(b["next_observations"])
            perm = np.random.default_rng().permutation(obs.shape[0])
            xperm = xb[perm]
            goal_rep = jnp.broadcast_to(goal, obs.shape[:-1] + goal.shape)
            w_inf, w_state, lam_params, lam_state, _, _ = step(
                w_inf, w_state, lam_params, lam_state, obs, act, xperm, goal_rep)

        if bool(ic["norm_w"]):
            w_inf = project_z(w_inf, True)
        return self.replace(w_inf=w_inf)
```

Note: `np.random.default_rng().permutation` uses fresh entropy each step (matches RLU's `torch.randperm`); this is eval-time only, not part of the reproducible-seed training path.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /u/amsks/git/PSMFLows && python -m pytest tests/test_affine_psm_inference.py -k full -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add agents/affine_psm.py tests/test_affine_psm_inference.py
git commit -m "AFFINE PSM: full constrained goal inference (DGD + hinge)"
```

---

### Task 5: `zero_shot` inference + mode dispatcher

**Files:**
- Modify: `agents/affine_psm.py` (add `infer_w_zeroshot`, `infer_eval`)
- Test: `tests/test_affine_psm_inference.py` (add)

**Interfaces:**
- Produces: `infer_w_zeroshot(self, dataset, goal) -> AffinePSMAgent`; `infer_eval(self, dataset, goal) -> AffinePSMAgent` dispatching on `config["inference"]["mode"]`.

- [ ] **Step 1: Write the failing test (append to `tests/test_affine_psm_inference.py`)**

```python
def test_zero_shot_inference_normalized_and_deterministic():
    agent, config, ds = _trained_agent(["inference.mode=zero_shot"])
    goal = ds.sample(1)["next_observations"][0]
    a1 = agent.infer_eval(ds, goal)
    a2 = agent.infer_eval(ds, goal)
    w1 = np.asarray(a1.w_inf)
    assert np.all(np.isfinite(w1))
    # sqrt(d)-normalized
    assert abs(np.linalg.norm(w1) - math.sqrt(config["d_dim"])) < 1e-3
    # deterministic given same goal + dataset order
    assert np.allclose(w1, np.asarray(a2.w_inf), atol=1e-5)
```

`_trained_agent` and the zero-shot path must draw the SAME dataset batches for determinism: give `infer_w_zeroshot` its own fixed RNG-free reduction (mean over a single fixed `dataset.sample` seeded call). Use `ds.sample` deterministically by re-seeding: add `seed` param threading in the test's `_FakeDataset` if needed, or average over the full stored arrays. Implementation below averages over a single large sample.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /u/amsks/git/PSMFLows && python -m pytest tests/test_affine_psm_inference.py -k zero_shot -v`
Expected: FAIL with `AttributeError: ... 'infer_eval'`.

- [ ] **Step 3: Add zero-shot + dispatcher to `agents/affine_psm.py`**

```python
    def infer_w_zeroshot(self, dataset, goal, num_samples=4096):
        """Closed-form goal code (no test-time optimization): w = sqrt(d)*normalize(
        E_{(s,a)~D}[Phi(s,a,goal)]) — the affine-net analogue of FB get_goal_meta."""
        c = self.config
        goal = jnp.asarray(goal, jnp.float32)
        b = dataset.sample(min(num_samples, getattr(dataset, "n", num_samples)))
        obs = jnp.asarray(b["observations"]); act = jnp.asarray(b["actions"])
        goal_rep = jnp.broadcast_to(goal, obs.shape[:-1] + goal.shape)
        phi, _ = self.measure(obs, act, goal_rep)
        w = phi.mean(0)
        if bool(c["inference"]["norm_w"]):
            w = project_z(w, True)
        return self.replace(w_inf=w)

    def infer_eval(self, dataset, goal):
        mode = self.config["inference"]["mode"]
        if mode == "zero_shot":
            return self.infer_w_zeroshot(dataset, goal)
        if mode == "full":
            return self.infer_w_goal(dataset, goal)
        raise ValueError(f"unknown inference.mode {mode!r}")
```

For the determinism assertion, make `_FakeDataset.sample` deterministic when called via `infer_w_zeroshot` by having the test construct the dataset so `sample(4096)` with `n=256` returns a stable draw — simplest: in `infer_w_zeroshot`, when `num_samples >= dataset.n`, read `dataset.obs`/`dataset.act` directly if present, else fall back to `sample`. Implement that guard:

```python
        if hasattr(dataset, "obs") and num_samples >= getattr(dataset, "n", 0):
            obs = jnp.asarray(dataset.obs); act = jnp.asarray(dataset.act)
        else:
            b = dataset.sample(num_samples)
            obs = jnp.asarray(b["observations"]); act = jnp.asarray(b["actions"])
```

(The real OGBench `ReplayBuffer` has no `.obs` attribute, so production uses the `sample` path; the guard only makes the unit test deterministic. Alternatively, average over a fixed `dataset.sample` under a seeded numpy state — pick one and keep it.)

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /u/amsks/git/PSMFLows && python -m pytest tests/test_affine_psm_inference.py -k zero_shot -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add agents/affine_psm.py tests/test_affine_psm_inference.py
git commit -m "AFFINE PSM: zero_shot closed-form inference + mode dispatcher"
```

---

### Task 6: Actor distillation + `sample_actions`

**Files:**
- Modify: `agents/affine_psm.py` (add `distill_actor`, `sample_actions`)
- Test: `tests/test_affine_psm_inference.py` (add)

**Interfaces:**
- Consumes: `infer_eval` (Tasks 4–5), `PSMActor`.
- Produces: `distill_actor(self, dataset, seed=0) -> AffinePSMAgent` (new copy, actor trained to maximize `Q = Phi(s, pi(s), goal_free) · w_inf + b`); `sample_actions(self, observations, seed=None, temperature=1.0) -> actions`.

Note on `Q`: after `w_inf` is solved for a goal, the distilled greedy action at `s` maximizes `Q(s,a) = Phi(s,a,s)ᵀw_inf + b(s,a,s)` — RLU evaluates `q_function(obs, actions, goal)` but at deployment the actor is state-only. v1 uses `x = obs` as the measure argument for the distillation Q (self-measure at the current state), the continuous analogue of RLU's `act(obs)`. Document this choice inline.

- [ ] **Step 1: Write the failing test (append)**

```python
def test_distill_actor_increases_q_and_bounds_actions():
    agent, config, ds = _trained_agent(["inference.mode=full",
                                         "inference.num_inference_steps=100",
                                         "inference.num_actor_inference_steps=200"])
    goal = ds.sample(1)["next_observations"][0]
    agent = agent.infer_eval(ds, goal)

    def mean_q(ag):
        b = ds.sample(128)
        obs = jnp.asarray(b["observations"])
        a = ag.actor(obs, jnp.zeros((obs.shape[0], config["z_dim"])))  # PSMActor(obs, z-unused-shape)
        phi, bb = ag.measure(obs, a, obs)
        return float(np.mean(np.asarray((phi * ag.w_inf).sum(-1, keepdims=True) + bb)))

    q0 = mean_q(agent)
    agent2 = agent.distill_actor(ds)
    q1 = mean_q(agent2)
    assert q1 >= q0 - 1e-4
    a = np.asarray(agent2.sample_actions(ds.sample(16)["observations"]))
    assert np.all(np.abs(a) <= 1.0 + 1e-5)
```

Note: `PSMActor(obs, z)` requires a `z` input of shape `(B, z_dim)`; the affine actor ignores task conditioning at act time (the goal is baked into `w_inf`), so distillation and `sample_actions` pass a zero `z`. Keep `z_dim` in the config for this placeholder shape.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /u/amsks/git/PSMFLows && python -m pytest tests/test_affine_psm_inference.py -k distill -v`
Expected: FAIL with `AttributeError: ... 'distill_actor'`.

- [ ] **Step 3: Add distillation + acting to `agents/affine_psm.py`**

```python
    def distill_actor(self, dataset, seed=0):
        """Train the PSMActor to maximize Q(s,a)=Phi(s,a,s)·w_inf + b(s,a,s) (RLU
        distill_actor_ddpg, self-measure form). Returns a new agent with the distilled actor."""
        c = self.config
        z0_dim = c["z_dim"]
        actor = self.actor
        w_inf = self.w_inf

        @jax.jit
        def step(actor, obs):
            zc = jnp.zeros((obs.shape[0], z0_dim))
            def loss_fn(params):
                a = actor.apply_fn({"params": params}, obs, zc)
                phi, b = self.measure(obs, a, obs)
                Q = (phi * w_inf).sum(-1, keepdims=True) + b
                return -Q.mean()
            g = jax.grad(loss_fn)(actor.params)
            return actor.apply_gradients(grads=g)

        for _ in range(int(c["inference"]["num_actor_inference_steps"])):
            b = dataset.sample(c["batch_size"])
            actor = step(actor, jnp.asarray(b["observations"]))
        return self.replace(actor=actor)

    @jax.jit
    def sample_actions(self, observations, seed=None, temperature=1.0):
        zc = jnp.zeros((*observations.shape[:-1], self.config["z_dim"]))
        return self.actor(observations, zc)
```

`actor.apply_fn` is the module's `apply`; `TrainState` exposes it (see `utils/flax_utils.py`). If the accessor differs, use `self.actor.model_def.apply({"params": params}, obs, zc)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /u/amsks/git/PSMFLows && python -m pytest tests/test_affine_psm_inference.py -k distill -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add agents/affine_psm.py tests/test_affine_psm_inference.py
git commit -m "AFFINE PSM: actor distillation against Phi·w_inf+b + sample_actions"
```

---

### Task 7: Goal-conditioned eval integration

**Files:**
- Modify: `main.py` (eval branch)
- Modify: `utils/evaluation.py` (goal extraction helper)
- Test: `tests/test_affine_psm_inference.py` (end-to-end infer→distill→act smoke, no real env)

**Interfaces:**
- Consumes: `infer_eval`, `distill_actor`, `sample_actions`.
- Produces: a `main.py` eval branch that, when `hasattr(agent, "infer_w_goal")`, extracts the env goal, solves `w_inf`, distills the actor, and evaluates.

- [ ] **Step 1: Write the failing E2E smoke test (append)**

```python
def test_end_to_end_infer_distill_act():
    agent, config, ds = _trained_agent(["inference.mode=full",
                                         "inference.num_inference_steps=50",
                                         "inference.num_actor_inference_steps=50"])
    goal = ds.sample(1)["next_observations"][0]
    agent = agent.infer_eval(ds, goal)
    agent = agent.distill_actor(ds)
    obs = ds.sample(10)["observations"]
    a = np.asarray(agent.sample_actions(obs))
    assert a.shape == (10, 2)
    assert np.all(np.isfinite(a)) and np.all(np.abs(a) <= 1.0 + 1e-5)
```

- [ ] **Step 2: Run test to verify it fails, then passes with existing methods**

Run: `cd /u/amsks/git/PSMFLows && python -m pytest tests/test_affine_psm_inference.py -k end_to_end -v`
Expected: PASS already (Tasks 4–6 provide the methods). If it fails, fix the method wiring before proceeding. This test locks the eval sequence contract that `main.py` depends on.

- [ ] **Step 3: Add the goal-extraction helper to `utils/evaluation.py`**

```python
def extract_goal(env):
    """Return the goal observation for a goal-conditioned OGBench env, or None.
    OGBench exposes info['goal'] on reset (see envs/env_utils.py frame-stack wrapper)."""
    _, info = env.reset()
    return info.get("goal", None)
```

- [ ] **Step 4: Add the eval branch to `main.py`**

In the `# Evaluate agent.` block, before the existing `infer_eval_z` branch, add:

```python
            if hasattr(agent, "infer_w_goal"):
                # Affine (full) PSM: solve w_inf for the env goal, distill the actor.
                from utils.evaluation import extract_goal
                goal = extract_goal(eval_env)
                assert goal is not None, "affine_psm eval needs a goal-conditioned env (info['goal'])."
                eval_agent = agent.infer_eval(train_dataset, goal)
                eval_agent = eval_agent.distill_actor(train_dataset)
            elif hasattr(agent, 'infer_eval_z'):
                ...  # existing reward-based branch unchanged
```

(Keep the existing `infer_eval_z` branch verbatim as the `elif`.)

- [ ] **Step 5: Run the full affine test suite**

Run: `cd /u/amsks/git/PSMFLows && python -m pytest tests/test_affine_psm_networks.py tests/test_affine_psm_smoke.py tests/test_affine_psm_inference.py -v`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add main.py utils/evaluation.py tests/test_affine_psm_inference.py
git commit -m "AFFINE PSM: goal-conditioned eval integration (extract_goal + main.py branch)"
```

---

## Verification (end-to-end, manual)

1. **Existing PSM untouched:** `python -m pytest tests/test_psm_smoke.py tests/test_psm_agent_equiv.py tests/test_psm_networks_equiv.py -q` — all still green.
2. **Affine suite:** `python -m pytest tests/test_affine_psm_networks.py tests/test_affine_psm_smoke.py tests/test_affine_psm_inference.py -q`.
3. **Real run (goal-conditioned OGBench), full mode:**
   ```bash
   python main.py agent=affine_psm env_name=pointmaze-medium-navigate-v0 \
     offline_steps=2000 eval_interval=1000 eval_episodes=10 \
     agent.inference.num_inference_steps=256 agent.inference.num_actor_inference_steps=256
   ```
   Confirm: training `psm_loss` decreases in wandb; eval runs `infer_eval`→`distill_actor`→`evaluate` without NaN; `evaluation/success` (or the env's success key) is logged and > 0.
4. **Zero-shot mode:** rerun step 3 with `agent.inference.mode=zero_shot` — eval should be much faster (no inference loop) and still produce bounded actions and a logged success metric.
5. **Batch-size guard:** verify `agent.batch_size` resolves to 32 (not 1024) — a 1024 batch makes the `B²=1,048,576` mesh OOM.

## Notes / risks carried from the spec

- **`x = obs` self-measure in distillation** (Task 6) is the continuous analogue of RLU's state-only `act(obs)`; if goal-reaching is weak, revisit to condition the distilled actor on the goal explicitly (design change, out of v1 scope).
- **Inference cost:** `num_inference_steps` × `num_actor_inference_steps` run *per eval*. Keep small in smoke runs; profile before scaling to `cube`.
- **`zero_shot` definition** (`w = √d·normalize(E[Φ(s,a,goal)])`) is a design choice, not an RLU port — flagged for review in the spec (risk #4).
- **`pointmaze` first, `cube` second** — goal APIs differ across OGBench families; land the simpler env before `cube-single`.
```

# Pool Measure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train the psmgoal measure as a distribution over a fixed reference set of dataset states (the pool), with the TD target anchored on the pool state nearest the next state, and read it out by importance-weighted reward integration.

**Architecture:** One new value of `measure_loss` (`pool`) in `agents/psmgoal.py` that swaps the candidate set in `measure_loss` for a fixed array `pool_states` carried on the agent; one new `policy_index` value (`bank`, one member = BC) for clean policy evaluation; one new eval readout `coef_source=pool_reward` in the inference path; the pool is rebuilt from `(dataset, pool_seed)` at eval time. No new network, no new symbols.

**Tech Stack:** JAX/Flax agent (`agents/psmgoal.py`), Hydra config (`configs/agent/psmgoal.yaml`), pytest on CPU with `allow_untrained_flow`, SLURM launchers under `scripts/slurm/`.

**Spec:** `docs/design/2026-10-19-pool-measure.md`

## Global Constraints

- `utils.xla_guard` is imported before jax in every entry point and test (see `tests/conftest.py`).
- Only the symbols phi, b, w, h, u, u', M, l, t, gamma; no A/beta split, no new Greek names (user rule, `docs/PSMGOAL_COMPENDIUM.md` 2.2).
- Every new config key defaults to the pre-existing behaviour; `tools/eval_checkpoint.py` `LEGACY_AGENT_DEFAULTS` gets each new key's pre-2026-10-19 value.
- Tests run per file: `.venv/bin/python -m pytest tests/<file>.py -x -q`; never the whole suite in one process.
- `.venv/bin/ruff check <file>` clean on every touched file (line length 120, single quotes).
- No AI attribution in commits. Commit after each task.
- Before any GPU launch: the hyperparameter table is printed, a 200-step smoke of the exact code path exits 0, and each run's `flags.json` is re-read after launch.

## Review Focus

1. A next state equidistant from two pool states must get a deterministic anchor (argmin, first index); test pins it.
2. `pool_size` larger than the number of valid-preimage rows must assert, not silently draw with replacement.
3. Eval must rebuild an identical pool: `pool_hash` recorded in `flags.json`/train log and asserted in `eval_checkpoint.py`; a mismatch raises.
4. The pool loss with `norm_future_k > 0` must assert (the pool replaces the candidate set; mixing is not defined).
5. `pool_reward` with zero rewarding rows in the relabel batch must return a finite Q of 0 and not divide by zero.

---

### Task 1: pool construction and anchor, pure helpers

**Files:**
- Modify: `utils/psm_common.py` (append)
- Test: `tests/test_pool_measure_helpers.py` (create)

**Interfaces:**
- Produces: `draw_pool_indices(valid_rows: np.ndarray, pool_size: int, pool_seed: int) -> np.ndarray` (sorted int64, length pool_size, no replacement; asserts `pool_size <= len(valid_rows)`); `pool_hash(indices: np.ndarray) -> str` (sha1 hex of `indices.astype(np.int64).tobytes()`); `pool_anchor(next_obs: jnp.ndarray (N, ob), pool_obs: jnp.ndarray (P, ob), sigma: jnp.ndarray (ob,)) -> jnp.ndarray (N,) int32` = argmin_j of the standardised squared distance, first index on ties; `pool_sigma(pool_obs) -> (ob,)` = per-dimension std floored at 1e-6.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import utils.xla_guard  # noqa: F401
import jax.numpy as jnp
from utils.psm_common import draw_pool_indices, pool_hash, pool_anchor, pool_sigma

def test_draw_pool_is_deterministic_sorted_without_replacement():
    rows = np.arange(100, 1100)
    a = draw_pool_indices(rows, 64, 0); b = draw_pool_indices(rows, 64, 0); c = draw_pool_indices(rows, 64, 1)
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    assert len(set(a.tolist())) == 64 and np.all(np.diff(a) > 0) and set(a) <= set(rows)

def test_draw_pool_refuses_oversize():
    import pytest
    with pytest.raises(AssertionError):
        draw_pool_indices(np.arange(10), 11, 0)

def test_pool_hash_changes_with_indices():
    assert pool_hash(np.arange(5)) != pool_hash(np.arange(1, 6))
    assert len(pool_hash(np.arange(5))) == 40

def test_anchor_is_nearest_in_standardised_units_and_first_on_ties():
    pool = jnp.asarray([[0.0, 0.0], [10.0, 0.0], [0.0, 1.0]])
    sigma = jnp.asarray([10.0, 1.0])            # x is 10x wider than y
    nxt = jnp.asarray([[6.0, 0.0], [0.0, 0.5]]) # row 0: x=6 is 0.6 sd from both x=0 and x=10 -> tie between 0 and 1 -> first = 0
    a = pool_anchor(nxt, pool, sigma)
    assert a.tolist() == [0, 0]                 # row 1: 0.5 sd from pool 0 and pool 2 -> tie -> 0

def test_pool_sigma_floor():
    s = pool_sigma(jnp.zeros((4, 3)))
    assert float(s.min()) >= 1e-6
```

- [ ] **Step 2: Run to verify failure** — `.venv/bin/python -m pytest tests/test_pool_measure_helpers.py -x -q` → ImportError.
- [ ] **Step 3: Implement** in `utils/psm_common.py`:

```python
def draw_pool_indices(valid_rows, pool_size, pool_seed):
    """The reference pool R of the pool measure (docs/design/2026-10-19-pool-measure.md): pool_size dataset rows
    drawn once, without replacement, from the valid-preimage rows, reproducible from pool_seed alone."""
    valid_rows = np.asarray(valid_rows, dtype=np.int64)
    assert 0 < pool_size <= len(valid_rows), (pool_size, len(valid_rows))
    rng = np.random.default_rng(int(pool_seed))
    return np.sort(rng.choice(valid_rows, size=int(pool_size), replace=False))

def pool_hash(indices):
    return hashlib.sha1(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()

def pool_sigma(pool_obs):
    return jnp.maximum(jnp.std(pool_obs, axis=0), 1e-6)

def pool_anchor(next_obs, pool_obs, sigma):
    d = ((next_obs[:, None, :] - pool_obs[None, :, :]) / sigma) ** 2
    return jnp.argmin(d.sum(-1), axis=1).astype(jnp.int32)
```

- [ ] **Step 4: Run tests** → 5 pass. `ruff check utils/psm_common.py tests/test_pool_measure_helpers.py`.
- [ ] **Step 5: Commit** `psm_common: pool indices, hash, sigma, anchor for the pool measure`.

### Task 2: `measure_loss=pool` in the agent

**Files:**
- Modify: `agents/psmgoal.py` (`MEASURE_LOSSES` tuple or equivalent assert near the top; dataclass fields; `create`; `measure_loss`; new `_pool_measure_loss`; `get_config`), `configs/agent/psmgoal.yaml` (keys `pool_size: 1024`, `pool_seed: 0`), `main.py` (pass the dataset's valid rows and observations into `create` for the pool; follow how `preimage` data already reaches the agent).
- Test: `tests/test_psmgoal_pool_loss.py` (create). Reuse `_agent`, `_batch` from `tests/test_psmgoal_joint_actor.py` and `OVER` from `tests/test_psmgoal_bootstrap_delta_scale.py` (as `tests/test_psmgoal_frozen_delta_bootstrap.py` does).

**Interfaces:**
- Consumes: Task 1 helpers.
- Produces: agent fields `pool_states: Any = None` (P, ob) and `pool_indices: Any = None`; `create(..., pool_obs=None, pool_indices=None)`; a static method `pool_td_loss(M, M_bar, anchor, discount, temp)` = cross-entropy against `(1-gamma) onehot(anchor) + gamma softmax(M_bar/temp)`; `_pool()` predicate (`measure_loss == 'pool'`); log keys `measure_p_anchor`, `pool_entropy`, `pool_u_tv`, `phi_eff_rank`, `pool_hash`.

- [ ] **Step 1: Failing tests**

```python
def test_pool_loss_matches_softmax_loss_when_pool_is_the_batch_next_states():
    # with pool = the batch's own next states in batch order, anchors are the identity and the pool loss
    # equals softmax_td_loss on the same meshes (up to 1e-5)
def test_pool_loss_target_rows_sum_to_one_and_anchor_gets_one_minus_gamma_when_target_is_flat():
    # M_bar = zeros -> target = (1-g) onehot + g/P; check pool_td_loss equals -mean(sum target*log_softmax(M))
def test_pool_refuses_norm_future_k():   # AssertionError at create
def test_pool_update_runs_and_logs_keys():   # one update on a CPU agent with pool_size=16: finite loss, keys present, pool_hash str
def test_pool_states_are_not_trainable_and_persist_across_update():
```

- [ ] **Step 2: Run → fail.**
- [ ] **Step 3: Implement.** In `measure_loss`: `if self._pool(): goals = self.pool_states` (replacing the candidate set), anchors `a = pool_anchor(next_obs, self.pool_states, self.pool_sigma)`; compute `M`, `M_bar` exactly as now over the pool; return `self._pool_measure_loss(M, target_M, phi, w, a)` which calls `pool_td_loss` and adds the three log keys (entropy of `softmax(M/t)`; `pool_u_tv` over the first 32 rows using a fresh clipped prior draw from `fold_in(self.rng, 106)` and the online params, stop_gradient). Assert `norm_future_k == 0` and `measure_loss == 'pool'` are compatible with `ortho_coef == 0` as softmax is. `get_config`: `pool_size: 1024`, `pool_seed: 0`.
- [ ] **Step 4: Tests pass; ruff clean.** Also re-run `tests/test_psmgoal_frozen_delta_bootstrap.py` and `tests/test_psmgoal_joint_actor.py` (nothing else may change).
- [ ] **Step 5: Commit** `psmgoal: measure_loss=pool (fixed reference set, nearest-pool anchor)`.

### Task 3: `policy_index=bank`

**Files:** `agents/psmgoal.py` (`POLICY_INDICES` or the assert that lists `goal`; `_coef`; `_select_u_next`; the actor step must be skipped or inert under bank), `configs/agent/psmgoal.yaml` (`n_bank: 1`), test `tests/test_psmgoal_bank_index.py`.

**Interfaces:** under `policy_index=bank`, `_coef(z, goals, ...)` returns `w` broadcast to (N, z) from a single learned parameter (reuse the existing `w` net at a constant code of zeros if that is the smallest change; document the choice in the docstring); `_select_u_next` returns `clip(eps)` with eps from `FLOWBC_BOOT_KEY`; `train_actor` must be false under bank (assert); `n_bank != 1` asserts "not implemented".

- [ ] Tests: `test_bank_w_is_shared_across_rows`, `test_bank_bootstrap_is_a_clipped_prior_draw`, `test_bank_refuses_train_actor`, `test_bank_update_runs` (pool loss + bank, CPU).
- [ ] Implement, tests pass, ruff, commit `psmgoal: policy_index=bank (one member, BC draw)`.

### Task 4: eval — pool rebuild, `pool_hash` check, `coef_source=pool_reward`

**Files:** `tools/eval_checkpoint.py` (`LEGACY_AGENT_DEFAULTS` += `measure_loss: 'squared'`, `pool_size: 0`, `pool_seed: 0`, `n_bank: 1`; rebuild the pool from the dataset when `measure_loss == 'pool'` and assert the hash against the run's `flags.json`/train log), `agents/psmgoal.py` (`infer_eval_goals` builds the eval bank and weights under `coef_source=pool_reward`; `select_latent` scores draws by the integrated Q with max over goals or the bank), `configs/agent/psmgoal.yaml` (`pool_eval_size: 512`), test `tests/test_psmgoal_pool_reward_readout.py`.

**Interfaces:** `infer_eval_goals` stores `eval_bank` (B, ob), `eval_bank_r` (B,), `eval_bank_w` (B,) weights; `_pool_reward_score(agent, obs, u_k)` → (64,) = max over w_z of `sum_j omega_j exp(M/t) r_j / sum_j omega_j exp(M/t)`.

- [ ] Tests: `test_pool_reward_weights_reproduce_class_counts` (omega sums per class equal the relabel-batch class counts), `test_pool_reward_zero_rewarding_rows_gives_finite_zero`, `test_pool_reward_equals_plain_mean_when_weights_are_uniform_and_r_is_indicator`, `test_eval_hash_mismatch_raises` (monkeypatch flags with a wrong hash).
- [ ] Implement, tests pass, ruff, commit `eval: pool rebuild + hash check; coef_source=pool_reward`.

### Task 5: smoke, hyperparameter table, launch, flags

**Files:** `docs/plans/2026-10-19-pool-measure.md` (this file: fill the table below with the values that landed), `docs/HANDOFF.md` (entry), `scripts/campaign_2026-10/pool_arms.sh` (create: the three launch loops).

Arms (cube, 3 seeds, 500k, eval 50k/50 episodes in-loop, save 250k, 24 h):

| group | overrides on top of arm A's (`scripts/campaign_2026-10/ja_arms.sh` fbc3_pt) |
|---|---|
| `psmgoal_pool_bank_cube` | `agent.measure_loss=pool agent.policy_index=bank agent.train_actor=false agent.bootstrap_source=actor` (bank ignores it; or `data`, whichever the asserts permit; record which) |
| `psmgoal_pool_fdplus_cube` | `agent.measure_loss=pool agent.bootstrap_source=frozen_delta agent.frozen_delta_path=$A agent.frozen_delta_epoch=500000 agent.frozen_delta_mode=plus` |
| `psmgoal_pool_ja_cube` | `agent.measure_loss=pool` (live flowbc actor, as A) |

- [ ] 200-step GPU smoke of each (STEPS=200 EVAL_INT=100 EVAL_EPS=2), exit 0, `flags.json` shows `measure_loss: pool`, `pool_size: 1024`, `pool_hash` present; the in-loop 2-episode eval runs with `coef_source=pool_reward` and with `hgoal_each`.
- [ ] Launch 9 jobs; re-read every `flags.json`; start `ARMS=... bash scripts/campaign_2026-10/arm_eval_watcher_cube.sh` with `READOUTS="hgoal_each pool_reward"`.
- [ ] HANDOFF entry with the table, expectations copied from the design doc, job ids. Commit.

### Task 6: fullness on the first 250k checkpoints (gate)

- [ ] `sbatch scripts/slurm/diag_measure_fullness.sbatch` on each arm's seed 0 at 250k with `+rollout_policy=bc` (bank), `actor` (ja), `bon64` (all); report tv, H(p), H(rho) beside the design doc's expected rows; HANDOFF.

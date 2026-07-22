# PSMFlow v1 — implementation spec (flow-indexed successor measures + GPI eval)

**Date:** 2026-07-20 · **Branch:** `feat/psm-integration` · **Design source:**
`PAPER/RESEARCH_NOTE.md` (commit d330c7b; user-approved). This spec covers **Phase A**:
behavior-flow pretraining, preimage pipeline hardening, the new `psmflow` agent with
Rung-1 (flow-GPI) inference, diagnostics D1–D4, and first runs on standard-coverage
data. **Out of scope (later specs):** Rung 2 (test-time latent Q-iteration), Rung 3
(amortized latent actor), the coverage-ladder dataset tooling, the VC-FB/MC-FB baseline
port, and all paper/theory writing.

## 1. Goal and success criteria

Build a reward-free representation $\psi(s,u',u)^\top\varphi(x) \approx
M^{u\to u'}(s,\cdot)/\rho$ over the policy family $\pi_{u'}(s) = G_\theta(s,u')$ of a
frozen behavior flow, with dataset latents $u$ obtained by flow inversion, and zero-shot
eval via flow-GPI. Success gates:

- **D3 gate** (before agent work): round-trip $\|G(s,E(s,a))-a\|_2 <$ 0.1 (action range
  $[-1,1]$), mean ESS > 20/100, and $\|E(s,a)\|^2$ within the central 99% $\chi^2_{d_a}$
  band for ≥ 95% of transitions, on pointmaze-medium and cube-single.
- **D4 gate**: rung-1 GPI with an *oracle* reward (true task reward → $w$ via
  `infer_z`) reaches ≥ 50% of FQL's single-task success on pointmaze-medium before we
  scale to cube.
- **End state**: `psmflow` runs through `main.py` end-to-end on
  `cube-single-play-singletask-v0` with the standard PSM eval protocol (z-shift,
  seeded eval), logging to wandb, with parity-style unit tests green.

## 2. Architecture overview

Three stages, three artifacts:

```
Stage A: behavior flow            Stage B: preimages                Stage C: representation
FQL(bc_only=true), reward-free    tools/precompute_preimages.py     agents/psmflow.py
→ ckpt (bc_flow + onestep_flow)   ckpt + dataset → preimages.npz    flow ckpt + preimages → ψ,φ
                                                                    eval: infer_z → GPI → decode
```

### 2.1 Stage A — behavior flow (`agents/fql.py`, minimal change)

- Add config flag `bc_only: bool = False` to `FQLAgent.get_config()` and
  `configs/agent/fql.yaml`.
- When `bc_only`, `total_loss` = `bc_flow_loss + alpha * distill_loss` only: skip
  `critic_loss` (rewards unused → reward-free pretrain) and skip the actor `q_loss`
  term. Critic networks are still created (keeps `create`/checkpoint shape identical)
  but receive zero gradient; `target_update('critic')` still runs (harmless no-op
  semantics). Logged info keys keep their names so dashboards don't break.
- Trained via existing `main.py` with `eval_interval=0` (no meaningful eval for a BC
  flow) or small eval as sanity; `flow_steps=100` engineering default. New launcher
  `scripts/launch_flowbc.sh ENV [GPU] [STEPS]` mirroring `launch_psm_cube.sh`.
- **Deliberate choice:** the flow is FROZEN after Stage A. The representation trains
  against a stationary policy family (contrast with FB's moving actor). Fine-tuning the
  flow in-loop is a later ablation, not v1.

### 2.2 Stage B — preimage pipeline (`tools/precompute_preimages.py`, hardened)

Current tool creates a *fresh* FQL agent (docstring admits it's smoke-only). Changes:

- Load the Stage-A checkpoint: build FQLAgent at dataset dims, then
  `utils.flax_utils.restore_agent(agent, cfg.restore_path, cfg.restore_epoch)`.
  Assert `cfg.restore_path` is set unless `inversion.allow_untrained=true` (smoke).
- Batched EM over the full dataset ALREADY EXISTS
  (`utils/flow_inversion.py:augment_dataset_with_preimage_distribution` — jit+vmap,
  chunked, tqdm). Keep its `noise_preimage_{mean,cov,weights}` in-dataset schema.
- Extend the augmented npz with (aligned to dataset row order):
  - `noise_preimage_point (N, d_a)` — exact backward-ODE preimage $E_\theta(s,a)$
    (enables the point-vs-mixture ablation),
  - `preimage_ess (N,)`, `preimage_roundtrip (N,)` — health scalars for D3 reporting
    (ess is already computed by the EM scan; roundtrip = decode(point) error),
  - `preimage_meta.json` sidecar: env_name, flow ckpt path/epoch, flow_steps,
    inversion config.
- `utils/datasets.py` ALREADY has `return_preimage_noise`: when set, `sample()` draws
  one latent per row from its mixture (`batch['noise_preimage']`) and the next row's
  (`batch['next_noise_preimage']`, end-of-trajectory caveat noted in code). Reuse
  as-is; `u_data := batch['noise_preimage']`. Point-preimage mode: when the agent
  config sets `use_point_preimage`, the dataset instead copies
  `noise_preimage_point` into `batch['noise_preimage']` (new flag
  `preimage_point_mode` on Dataset, set from main.py).

### 2.3 Stage C — the `psmflow` agent (`agents/psmflow.py`, new)

Skeleton copied from `agents/psm.py` (per-network TrainState, named StepInputs, 
3-stage → now 2-stage update), with these deltas:

**Networks.**
- `phi`: `PhiMap` unchanged (basis over future states, `z_dim=128`, ortho loss kept).
- `psi`: existing `PsiMap(obs, z, action)` reused with **z-slot := $u'$ (policy index,
  $d_a$-dim)** and **action-slot := $u$ (current latent, $d_a$-dim)**; `output_dim=z_dim`,
  `num_parallel=2` ensemble + pessimism kept. One branch only — PSM's proto/sf split
  collapses: the flow family IS the codebook, and tasks enter only at inference.
- Frozen flow: plain (non-TrainState) param pytrees `flow_vf_params` (multi-step
  decode/BC field) and `flow_onestep_params` (fast decode), stored like PSM's `proto`
  leaf. Loaded in `create()` from config `flow_ckpt_path`/`flow_ckpt_epoch` by
  instantiating a throwaway FQLAgent and extracting
  `modules_actor_bc_flow` / `modules_actor_onestep_flow` subtrees. Module defs
  (`ActorVectorField`) stored as static aux. `assert flow_ckpt_path` — no silent
  fresh-flow fallback outside tests.

**StepInputs** (replaces PSM's): `u_data (B, d_a)` — dataset latent drawn per row from
the preimage mixture (or `point` when `use_point_preimage=true`); `u_index (B, d_a)` —
policy-index draw.

**Index sampling** (`sample_step_inputs`): `u_index` = with prob `index_mix_ratio`
(default 0.5) the `u_data` of a permuted batch row (behavior-biased indices, analog of
PSM/FB z-mixing), else $u' \sim \mathcal N(0, I)$; finally clipped to
`u_clip` (default 3.0) — the typical-set constraint of Note §2.3, and the knob the
LAPO/DSRL literature says matters.

**Loss** (single measure branch + ortho, reusing `contrastive_loss`/`ortho_loss`/
`targets_uncertainty` verbatim from psm.py — move them to a shared module):

```
goal   = next_obs                                         # phi_input='s' convention kept
M      = psi(obs,      z=u_index, a=u_data)  @ phi(goal).T                # online params
tM     = target_psi(next_obs, z=u_index, a=u_index) @ target_phi(goal).T  # target params;
target = mean(tM over ensemble) - pessimism * unc         # continuation latent IS the index
L      = contrastive_loss(M, sg(target), discount) + ortho_coef * ortho_loss(phi(goal))
```

(`a=u_data` in the online term; the target's action-slot takes `u_index` because the
continuation policy at $s'$ is $\pi_{u'}$ itself.) No decoded action appears in
training — the dynamics enter only through the dataset transition $(s,a,s')$ whose
latent is $u_{\text{data}}$. This is the in-sample property (Note Prop. 3) made literal.

**Update**: single-stage `apply_update` — step phi+psi jointly on the measure loss,
then polyak both targets (tau=0.01). No actor at v1. `update` jitted as in PSM.

**Inference (Rung 1 — flow-GPI).**
- `infer_z(next_obs, rewards)` → $w = \mathbb E[r\,\varphi]$: copied from PSM
  (identical formula, `norm_z` projection kept); `infer_eval_z` keeps its name so
  `main.py`'s generic z-inference hook (main.py:192) works unchanged.
- `sample_actions(obs, seed)`: draw `gpi_num_u` (64) candidates $u_k \sim \mathcal N$
  clipped to `u_clip`, and `gpi_num_uprime` (16) indices $u'_j$ (fresh per step);
  score $Q_{kj} = \psi(s, u'_j, u_k)^\top w$ aggregated over the ensemble as
  `mean - actor_pessimism_penalty * unc`; take $\max_j$ then $\arg\max_k$; decode
  $a = $ one-step flow$(s, u^*)$ (`gpi_decode='onestep'`, ablation `'ode'` uses the
  multi-step Euler rollout), clip to $[-1,1]$. All jitted; K×M ψ evals per env step —
  batch as one `(K·M)` forward.

**Config** `configs/agent/psmflow.yaml` (+ importable `get_config()`): inherits PSM's
winners — `z_dim=128, ortho_coef=1000, lr_phi=1e-5, lr_sf=1e-4, discount=0.98,
batch_size=1024, num_parallel=2, tau=0.01, norm_z=true, pessimism_penalty=0.0,
actor_pessimism_penalty=0.5` — plus new: `flow_ckpt_path, flow_ckpt_epoch,
preimage_path, use_point_preimage=false, index_mix_ratio=0.5, u_clip=3.0,
gpi_num_u=64, gpi_num_uprime=16, gpi_decode=onestep, flow_steps=10`.

**Wiring**: register in `agents/__init__.py`; `main.py` gets one branch next to the
PSM one: `if agent_name == 'psmflow':` load the augmented dataset
(`load_augmented_dataset(config['preimage_path'])` → `Dataset.create`) in place of the
raw one, set `dataset.return_preimage_noise = True` (and `preimage_point_mode` from the
agent config). Everything else (eval z-shift/relabel, seeded eval, save/restore) is
inherited.

### 2.4 Diagnostics (new tools, each a runnable script + wandb-free stdout report)

- `tools/validate_flow_fidelity.py` (**D1**): flow samples vs dataset actions at
  matched states — per-state-cluster MMD (RBF), k-means (k=16) mode histograms
  (dataset vs flow), fraction of flow samples farther than ε from any dataset action
  at k-NN states. Gate feeds Stage A sign-off.
- `tools/eval_fixed_u_rollouts.py` (**D2**): roll $\pi_u$ for a grid/sample of fixed
  $u$ in the real env; report final-state cluster entropy across $u$ (diversity) and
  per-$u$ trajectory self-consistency across episodes. Needs env access only, no rewards.
- **D3** = existing `tools/validate_flow_inversion.py` + new $\chi^2$-typicality line +
  now loads the Stage-A ckpt (same restore change as Stage B).
- `tools/latent_q_sanity.py` (**D4**): oracle-reward GPI — build the trained psmflow
  agent, set $w$ from *true* task rewards via `infer_z`, evaluate; compare to the
  in-repo FQL benchmark number for the same env. (Full latent Q-*iteration* is Rung 2,
  deferred; this checks the representation + GPI pipeline alone.)

## 3. Testing

Property/parity-style tests mirroring the repo's conventions (`tests/`):

1. `test_psmflow_networks.py`: shapes/dtypes; psi z/action slot wiring (u' vs u not
   swapped — assert asymmetry by probing with distinct inputs); frozen-flow decode
   equals FQLAgent.compute_flow_actions on transplanted params (guards the subtree
   extraction, the converter-bug lesson from the transplant work).
2. `test_psmflow_agent.py`: `update` runs jitted; flow param leaves bit-identical
   before/after update (frozen); loss decreases on a fixed synthetic batch over 50 steps; `u_clip` and
   `index_mix_ratio` respected (statistical check on StepInputs).
3. `test_psmflow_groundtruth.py`: **tiny-MDP fixed-point test.** Deterministic 5-state
   1-D chain with identity decoder injected ($G(s,u)=\mathrm{clip}(u)$, so $\pi_u$ =
   "always action u"): analytic $M^{u\to u'}$ is computable in closed form; train on an
   enumerated dataset and check $\psi^\top\varphi$ against the analytic occupancy
   within loose tol (0.1 rel). This is the only test that validates the *math*, not
   just the plumbing.
4. `test_psmflow_smoke.py`: 3-step end-to-end through `main.py` on pointmaze with a
   fresh (untrained) flow + `inversion.allow_untrained` preimages (plumbing only),
   exit 0.
5. Existing FQL tests still pass with `bc_only` defaulted off; one added case for
   `bc_only=true` loss composition.

## 4. Error handling / gotchas (from the parity investigations)

- Fail loudly on: missing `flow_ckpt_path`/`preimage_path` (outside tests), preimage
  row-count ≠ dataset size, preimage metadata env/ckpt mismatch, `encoder` set
  (state-only at v1, like PSM).
- Preimage sampling must key on the **global buffer row index** (the `return_index`
  lesson — batch-position keying is inconsistent across resamples); `return_preimage`
  gathers by the same row ids as the batch.
- One seed per GPU (HANDOFF infra lesson); `XLA_PYTHON_CLIENT_MEM_FRACTION` overridable.
- ≥ 3 seeds for any reported number; reference-grade claims need 5+ (PSM seed noise).

## 5. Run plan (Phase A exit)

1. Stage A flow on `pointmaze-medium-navigate-singletask-v0` + `cube-single-play-…`;
   D1 gate.
2. Stage B preimages both envs; D3 gate.
3. D2 rollout report (informational, feeds the note's §7 risk 1).
4. Train psmflow both envs (500k, 3 seeds); D4 oracle-GPI gate on pointmaze.
5. Zero-shot eval vs in-repo PSM and FB on the same envs/steps/seeds; compare with
   `scripts/compare_multiseed.py`-style tooling (extend to psmflow groups).

Coverage-ladder stress tests and VC-FB comparisons are the *next* spec, once Phase A
gates pass.

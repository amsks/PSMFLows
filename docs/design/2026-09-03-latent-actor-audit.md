# Latent (DSRL-style) actor audit — `agents/psmflow.py`

Date: 2026-09-03 · Branch `feat/inversion-integration` · Read-only audit, no code changed.

**Question.** DSRL (Wagenmaker et al. 2025) treats a frozen generative policy's input noise
as the action: a latent actor `pi(s) -> u` maximizes a critic over noise, and the deployed
action is `G(s, pi(s))`. PSMFlow instantiates that with `Q_W(s,u) := psi(s,w,u)^T w`.
Is that actor *correctly implemented*, and is it *actually doing anything* on the PSM
substrate?

**Verdict up front.** The implementation is **correct** — every gradient path, range,
ordering and pessimism claim in the code's own comments checks out numerically, and the
frozen flow is genuinely frozen. The actor is **not meaningfully utilized**: on a trained
500k checkpoint it is a 96%-correlated pass-through of its own input noise, its deployed
action is 99.4% invariant to the task vector `w`, and it sits at the **51st percentile** of
the prior-Q distribution its own critic defines. PSMFlow-with-actor is, to measurement,
behaviour cloning with an 8% perturbation. Two real bugs found, both latent (not hit by any
shipped config).

Probe scripts and JSON: `/tmp/claude-10025/-mnt-home-amohan-git-Austin-PSMFLows/28ad4e4b-a7a5-4c94-90a3-9708af3c15cc/scratchpad/`
(`probe1_gradpath.json`, `probe2_trained_antmaze.json`, `probe3_utilization.json`,
`probe4_cube500.json`, `probe5_udata_box.json`).
Trained checkpoint used: `/mnt/home/amohan/psm-data/exp/PSMFLows/psmflow_antmaze_mix_a26p5_20260902/sd000_s_2491447.0.20260902_174348`
@500000 (`acting=actor`, `train_actor=true`, `policy_index=task_vector`, `u_clip=3.0`,
`bc_coeff=1.0`, `action_critic.enabled=false` — i.e. the shipped "PSMFlow (zero-shot)" arm).

---

## 1. Gradient path — CLEAN

`flow_actor_loss` (psmflow.py:210-252) is differentiated in `apply_update` (psmflow.py:400-405)
w.r.t. `(actor_params, vf_params)` only.

| quantity (synthetic agent, probe1) | value |
|---|---|
| `d(actor_loss)/d(actor)` | **13.22** (nonzero — the DPG signal reaches the actor) |
| Q-term-only `d(q_loss)/d(actor)` | **3.04** |
| `dQ/du_a` at the actor's latent | 0.283 (psi's u-slot is a live input) |
| `d(actor_loss)/d(psi)` *if it were taken* | 13.42 |
| `d(actor_loss)/d(phi)` | **0.0** (exact) |
| `d(actor_loss)/d(flow_vf)` | **0.0** (exact) |
| `d(actor_loss)/d(flow_onestep)` | **0.0** (exact) |
| flow_vf / flow_onestep bit-identical after 5 updates | **True / True** |

Notes.
- The gradient **does** flow through psi's `u` input into the actor: that is what DSRL
  requires and it is present.
- psi receives **zero applied** gradient from the actor loss, but only because it is outside
  `argnums`, **not** because of a `stop_gradient`. The dependence is real (13.42). This is
  the same convention as `agents/psm.py:197-237` (which passes `sf_params` explicitly and
  differentiates only `(actor_params, vf_params)`), so it is a *convention*, not a defect —
  but it is one refactor away from a silent leak. There is a regression test for the
  symptom (`tests/test_psmflow_actor.py:35`), none for the mechanism.
- phi gets exactly zero because `task_w` is built under `stop_gradient` in
  `sample_step_inputs` (psmflow.py:157-161) and `sampled` is a concrete array by the time
  the actor loss runs.
- The frozen flow is never called in the actor loss at all (no decode in the training
  graph for this branch), matching DSRL's "steer the noise, never touch the generator".

## 2. Is the actor's `w` the critic's `w`? — YES

`sample_step_inputs` draws one `task_w` (psmflow.py:157-161: Gaussian mixed at
`mix_ratio=0.5` with `stop_gradient(phi(next_obs))[perm]`, both `project_z`'d).
`measure_loss` reads it via `_index` (psmflow.py:117,135-143) and `flow_actor_loss` reads
`sampled.task_w` (psmflow.py:213) — the **same object**, verified: `_index(s) is s.task_w`
is True on the default path (probe1).

At eval, `infer_eval_z` (psmflow.py:465-473) sets `task_z = project_z(E[r phi])`, and
`sample_actions` passes exactly that to the actor (psmflow.py:562-563). Probe3 confirms
`sample_actions(obs, seed)` reproduces a manual `decode(u_clip * actor(obs, task_z, N(0,I)))`
bit-for-bit. Norm matches training: `||task_z|| = 11.314 = sqrt(128)`, and training `w`
carries the same norm by construction (probe1: `task_w_norm_mean == sqrt(z_dim)`). No
stale/zero `w`, no projection mismatch.

## 3. Ranges and noise — consistent; but the actor is a noise pass-through

Ranges (all consistent, no bug):

| site | expression | file:line |
|---|---|---|
| training `u_a` | `u_clip * tanh(...)` | psmflow.py:241 |
| bootstrap `u_next` | `u_clip * tanh(...)` | psmflow.py:173 |
| eval `u_star` | `u_clip * tanh(...)` | psmflow.py:562 |
| `u_data` | `clip(preimage, ±u_clip)` | psmflow.py:151 |
| distill target | `clip(rollout, ±u_clip)` | psmflow.py:233 |
| gpi candidates | `clip(N(0,I), ±u_clip)` | psmflow.py:507,516 |

The noise input is `N(0, I)` at every site (training psmflow.py:174, 205-207; eval
psmflow.py:561). `u_data` clipping binds on **0.46%** (cube) / **0.38%** (antmaze) of entries
(probe5) — the box is not distorting the BC target.

**Does the actor use its noise?** Yes — overwhelmingly. On the trained checkpoint, at fixed
`(s, w=task_z)` over 128 noise draws at 64 states (probe2/probe3):

- `std(u_a)` over noise = **0.790**; over states (noise-averaged) = 0.216 → ratio **3.67**
- `corr(u_a, noise)` per dim = 0.951-0.968, **mean |corr| = 0.961**
- `std(u_a)` per dim 0.827 vs `std(noise)` 1.000

So `pi(s,w,n) ≈ 0.83 · n` — a **shrunk identity map on its own noise**. That is a valid
flowBC actor (the CFM anchor is doing its job) but it is not a *policy* in any useful sense.

**Does the actor use `w`?** Essentially not:

- `std(u_a)` over 128 realistic `w = phi(next_obs)` at fixed noise = **0.023**
  → 2.9% of the noise sensitivity
- decoded-action `std` over those `w` = **0.0053** against an action scale of 0.623 → **0.85%**
- `sample_actions` with `task_z` vs a **zero** `w`: mean |Δaction| = **0.0038** (0.6% of scale);
  vs a random sphere `w`: **0.0048**

The claim in the module docstring that "policy identity lives in the task vector `w`" is
**not realized in the deployed policy**. Zeroing the inferred task vector changes the action
by less than 1%. On the fresh cube agent at 500 steps the w-sensitivity is 12.7% (probe4),
so w-dependence *decays over training* rather than growing.

**Is the deployed action different from BC?** Barely. Decoding the actor's latent vs decoding
the *same noise* through the frozen flow directly (i.e. the BC control) differs by mean
|Δa| = **0.0505** against |a| = 0.623 → **8.1%**. Dispersion: actor decodes `std` 0.164 vs
prior decodes `std` 0.210 — 78% of the BC policy's own spread.

## 4. Ordering — as documented; matters little

- The actor loss is computed at the **PRE-update** psi: `apply_update` binds
  `self.flow_actor_loss` (psmflow.py:401), where `self` is the pre-measure-step agent, and
  applies the result to `new.actor` (psmflow.py:404). Verified: sign agreement between the
  applied Adam step and the pre-update-psi gradient is **1.000**, against **0.957** for the
  post-update-psi gradient; the two gradients differ by 14.4% in norm (probe1). This is
  PSM's no-interleave convention (`agents/psm.py:240-247`) and is a one-step staleness on a
  target that moves at `lr_sf=1e-4` — immaterial.
- The measure target bootstraps `u_next` from the **online** actor at `s'`
  (psmflow.py:173), evaluated inside `target_psi` (psmflow.py:126). There is **no target
  actor** — `PSMFlowAgent` has `target_phi/target_psi/target_psi_a/target_phi_a` and nothing
  else. That matches the reference family: `agents/psm.py:325,333-335` also bootstraps the
  online actor, and `agents/fb.py:6` states outright "targets on
  forward/backward/left_encoder (none on actor)". **Design deviation from TD3/DSRL, not a
  bug.**

## 5. The three actor terms — BC-dominated 5:1, CONFIRMED

`loss = q_loss + bc_coeff * distill + bc_flow_loss` (psmflow.py:249), with
`q_loss = -Q.mean() / stop_grad(|Qs|.mean())` (psmflow.py:247).

Gradient norms w.r.t. **actor params only** (`bc_flow_loss` has zero actor gradient):

| checkpoint | `w` | ‖∇ q_loss‖ | ‖∇ bc_coeff·distill‖ | ratio BC:Q |
|---|---|---|---|---|
| antmaze @500k (trained) | training mix | 0.0720 | 0.3425 | **4.76 : 1** |
| antmaze @500k (trained) | eval `task_z` | 0.0876 | 0.5121 | **5.85 : 1** |
| cube, fresh @500 steps | training mix | 0.3223 | 0.4889 | 1.52 : 1 |
| cube, fresh @500 steps | eval `task_z` | 0.4706 | 0.7919 | 1.68 : 1 |

**COMPENDIUM §4.7's "BC-dominated 5:1 (‖∇q‖ 0.168 vs ‖∇distill‖ 0.839)" is reproduced**, on a
different env and a different checkpoint, at 4.76:1 / 5.85:1. It is not a cube artifact and
not a measurement error. The early-training cube number (1.5:1) shows the domination
*grows* with training — consistent with §4.7's "Q is flat" and with the w-sensitivity decay
in §3.

**Clipped-target question (asked): non-issue.** The distill target is
`clip(rollout, ±u_clip)` (psmflow.py:233) while `u_a = u_clip·tanh` is open in `(-3,3)`.
The clip binds on **0.037%** of entries on the trained checkpoint (0.16% on the fresh cube
agent); `|target|max = 3.167` before clipping, target `std`/dim 0.892 against `u_data`
`std`/dim 1.033. There is no meaningful "distilled toward a clipped target while Q pushes
unclipped" tension.

**Q-flatness (D3), reproduced on this checkpoint** (probe2/probe3, 128 prior draws × 64 states):

- relative Q spread over prior draws = **1.74%** of |Q| (§4.7 cube: 1.1%)
- around the actor's own latents = 1.32%
- **actor percentile in the prior-Q distribution = 0.514** — the actor's latent is at the
  *median* of a random prior draw under its own critic (§4.7 cube: 44th percentile)
- `Q(actor) − Q(prior)` = **+0.12** prior-σ, while `max_K Q(prior) − Q(actor)` = **+2.44**
  prior-σ. The actor captures ~5% of the improvement its own critic can already see over
  128 random draws.

That is the utilization verdict in one line: the Q term is 1/5 of the gradient, and what it
buys is one-eighth of a standard deviation of a quantity that only varies by 1.7% anyway.

## 6. Pessimism — exact min, consistently applied where Q is read

`targets_uncertainty` (`agents/psm.py:74-79`) returns `(mean over axis 0, sum_ij|p_i-p_j| /
(P²−P))`. At `P=2`: `mean − 0.5·unc = (p0+p1)/2 − |p0−p1|/2 = min(p0,p1)`. Verified exactly
(probe1: `pess_equals_exact_min = True`), and the reduction axis is correct (`(P,B) -> (B,)`,
`(P,B,B) -> (B,B)`). Applied at `actor_pessimism_penalty=0.5` in `flow_actor_loss`
(psmflow.py:245-246) and identically in `gpi_select` (psmflow.py:511-512, 520-521).
With `acting=actor` no Q is evaluated at eval at all, so pessimism is training-only there —
correct by construction, not a gap.

## 7. Eval path — the actor really is what acts

- `sample_actions` (psmflow.py:551-566) with `acting=actor`: **one** `N(0,I)` draw, one
  `u_clip·tanh` latent, one `decode`. `gpi_decode` defaults to `"onestep"`
  (`configs/agent/psmflow.yaml:111`), so the decode is the distilled one-step flow
  (psmflow.py:476-480). Confirmed on the trained checkpoint: `gpi_decode == "onestep"` and
  `sample_actions` matches a manual actor→one-step decode bit-for-bit (probe3).
- `assert not (acting == "actor" and not train_actor)` (psmflow.py:557-559) guards the
  untrained-actor deployment; `tests/test_psmflow_policy_index.py:99` pins it.
- The live run `psmflow_antmaze_mix_a26p5_20260902/sd000`'s own `flags.json` records
  `acting: "actor"`, `train_actor: true`, `action_critic.enabled: false` — so the numbers
  reported as "PSMFlow (zero-shot)" are the **actor** acting, not gpi.
- **Gap (suspected, operational).** `tools/eval_checkpoint.py:67-71` builds the agent config
  from the Hydra `agent` group **plus CLI overrides only — it never reads the run's own
  `flags.json`**. Any run trained off-default (`policy_index=latent`, `train_actor=false`,
  a non-default `u_clip` or `acting`) must have those flags re-typed on the eval command or
  the eval measures a different policy. `scripts/eval500.sh:14-19` documents this for
  `latentrl`'s two flags and repeats none for `psmflow`. `restore_agent`
  (`utils/flax_utils.py:181-206`) replaces the tree wholesale and does not shape-check, so
  the failure mode is loud for Arm B (psi width changes) and **silent** for `u_clip`.

## 8. DSRL-specific gaps vs the reference

| gap | classification | note |
|---|---|---|
| No entropy / no stochastic policy head | **design deviation** | flowBC recipe (PSM/FQL lineage) is deterministic-given-noise; DSRL-SAC's entropy term is what `agents/latentrl.py` `critic_input=latent` is testing separately (HANDOFF 2026-09-03) |
| Deterministic given noise | **design deviation, and the live problem** | fine in principle, but measured: `corr(u_a, noise) = 0.961`, so "given noise" is the whole policy |
| BC anchor to Stage-B preimages (CFM + distill) | **design deviation** | DSRL's offline arm also regularizes to the noise-aliased data; here it is weighted `bc_coeff=1.0` and empirically outweighs Q 5:1 |
| Critic is an SF readout `psi^T w`, not a scalar TD critic | **design deviation — the root cause** | it is the whole point of the substrate, but D2 (best linear readout of phi explains ~13% of reward variance) + 1.7% Q spread means the readout carries almost no ranking signal. Not a code bug |
| No distillation from an action critic (DSRL-NA) | **irrelevant to correctness** | the `action_critic` branch is the codebase's attempt at that axis; disabled in the reported arm and on record as failing its gate (psmflow.py:305) |
| `u_clip = 3.0` vs DSRL's ~0.5-1.5 offline | **design deviation, probably harmless** | tested independently: `latentrl` at `u_clip=1.0` vs `3.0` did **not** raise `q_spread_rel` (HANDOFF 2026-09-03). Also, `u_data` exceeds ±3 on only 0.4% of entries, so a tighter box would start truncating real preimages |
| No target actor | **design deviation** | matches `agents/psm.py` and `agents/fb.py`; see §4 |

## 9. Bugs

### Confirmed

1. **`flow_actor_loss` hardcodes `w` in psi's index slot instead of `self._index(sampled)`**
   — psmflow.py:244 (`self.psi(obs, w, u_a)`) vs `_index` at psmflow.py:135-143.
   Under `policy_index="latent"` psi's index slot is `d_a`-wide, so passing the `z_dim`-wide
   `w` is both semantically wrong (the actor would be maximizing a head indexed by the wrong
   object) and a shape error. Reproduced (probe1):
   `policy_index=latent, train_actor=True` →
   `ScopeParamShapeError: For parameter "kernel" in "/vmap(tower)/embed_z_0", the given
   initializer is expected to generate shape (22, 1024), but the existing parameter it
   received has shape (8, 1024)`.
   **Nothing asserts against this combination.** `configs/agent/psmflow.yaml:52-54` says
   `train_actor=false` is "only valid with acting=gpi" but no config or `create()` assert
   forbids `policy_index=latent` **with** `train_actor=true`. Severity: low (fails loudly at
   the first update, and no shipped config hits it), but it is a real hole in Arm B's
   guard rail and the right fix is one line (`self._index(sampled)` + an assert in
   `create`).

2. **RNG key reused as both a sample key and a split parent** — psmflow.py:173-180.
   `r_next` is consumed by `jax.random.normal(r_next, ...)` for the bootstrap noise and then,
   when `backup_explore_frac > 0`, re-consumed by `jax.random.split(r_next)` to make
   `r_expl, r_emask`. JAX's threefry makes these practically independent so no statistical
   harm is measurable, but it is the documented anti-pattern and the comment above it
   ("keys are drawn inside the branch, so at frac=0 the random stream is unchanged") only
   justifies the *placement*, not the *reuse*. Severity: cosmetic. Not exercised by the
   shipped default (`backup_explore_frac=0.0`).

### Suspected / not-bugs, worth recording

3. **psi is protected from actor gradients by `argnums`, not by `stop_gradient`** —
   psmflow.py:244, 401-403. The mathematical dependence is live
   (‖d(actor_loss)/d(psi)‖ = 13.42 if taken). Correct today; brittle under refactor.
   `tests/test_psmflow_actor.py:35` tests the symptom (a `bc_coeff` change must leave the
   psi step byte-identical), which would catch a leak — so this is adequately fenced.

4. **`eval_checkpoint.py` does not read the run's `flags.json`** — see §7. Operational, not
   a code defect, but it is the one path where a wrong number can be produced silently.

5. **Checked and clean, no bug:** `targets_uncertainty` axes on both `(P,B)` and `(P,B,B)`;
   `Qs = psi(...) * w` broadcasting `(P,B,z)*(B,z)`; `stop_gradient` on the distill target
   (psmflow.py:248), the Q normalizer (psmflow.py:247), `goal_w` (psmflow.py:157) and the TD
   target (psmflow.py:129); the `flow_noise` reuse as both actor noise and rollout start
   (psmflow.py:241,248) is the FQL/PSM distillation convention (`agents/psm.py:223,231`), not
   a leak; no silently-defaulted config key on the actor path (`config["actor"]`,
   `config["train_actor"]`, `config["acting"]`, `config["u_clip"]` are all indexed, not
   `.get`-ed).

### Existing test coverage

`tests/test_psmflow_actor.py` (actor+vf train every step, finite losses, no leak into
psi/phi, flow frozen, acting=actor emits a clipped latent), `tests/test_psmflow_agent.py:120`
(the backup really is actor-coupled), `tests/test_psmflow_backup_explore.py` (frac=0 is the
actor's latent exactly), `tests/test_psmflow_policy_index.py` (Arm B; acting=actor with
train_actor=false is refused). **Not covered:** the `policy_index=latent` × `train_actor=true`
hole (bug 1); that eval's `task_z` reaches the actor (only that `sample_actions` runs);
anything about the *magnitude* of the Q term relative to the BC term.

---

## Verdict

**Correctly implemented: yes.** The DSRL mechanism is faithfully wired — a latent actor
trained by `-Q_W(s, pi(s,w,n))` with the gradient flowing through the critic's `u` input,
the flow frozen and absent from the training graph, the same `w` in critic and actor, the
same box and the same noise distribution at train and eval time, exact-min pessimism, and
the deployed action a genuine flow decode of the actor's latent. Two latent bugs, neither
reachable from any shipped or reported configuration.

**Properly utilized: no.** On the substrate as built, the DSRL half of the algorithm is
nearly inert:

- the value gradient is outweighed **~5:1** by the BC/distill anchor (0.072 vs 0.343),
  reproducing COMPENDIUM §4.7 on an independent env and checkpoint;
- the actor is a **0.961-correlated** pass-through of its own input noise;
- the deployed action is **0.85%** sensitive to the task vector — zeroing the inferred `w`
  moves it by 0.6% — so the *zero-shot* claim is not carried by the actor at all;
- the actor's latent sits at the **51st percentile** of the prior-Q distribution, gaining
  +0.12σ over a random draw where best-of-128 prior draws gains +2.44σ.

The cause is upstream of the actor, and this audit does not contradict the standing reading:
`Q_W = psi^T w` has ~1.7% relative spread over the latents the decoder can produce (§5), so
there is nothing for a correctly-implemented DSRL actor to climb. Raising `bc_coeff` down or
the Q weight up would only amplify a 1.7%-relief signal — COMPENDIUM already records the
`bc_coeff` sweep as "not indicated" for exactly this reason, and the measurements here
support that. The open question remains the critic, not the actor.

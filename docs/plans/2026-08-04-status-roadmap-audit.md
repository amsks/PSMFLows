# PSMFlows — where we stand, the OGBench roadmap, and the code audit

Date: 2026-08-04 · Branch: `feat/psm-integration` · Machine: `midi-01`
Companion docs: `docs/HANDOFF.md` (session log), `docs/design/2026-07-20-psmflow-v1-design.md`
(spec), `PAPER/main.tex` (formal write-up), `PAPER/RESEARCH_NOTE.md` (positioning + experiment
program).

---

## 1. Where we stand

**The method.** A behavior-cloned conditional flow `G(s,u)` maps latent noise to
dataset-like actions; policies are indexed by latents `u'` instead of raw actions; a
successor-measure representation `m(s,u,u',x) = psi(s,u,u')^T phi(x)` is trained over that
flow-indexed family (Stage C), with dataset transitions carrying their own latents via flow
inversion (Stage B). Inference is Rung-1 flow-GPI: `w = E[r*phi]`, argmax over sampled
`(u, u')`, one-step flow decode. The formal claim structure lives in `PAPER/main.tex`
(concentrability C=1 for the training distribution, the divergence isometry Prop. 3.3, the
in-sample TD Prop. 7.2); Rungs 2–3 and the coverage-ladder benchmark are the next specs.

**Pipeline status (Phase A):**

| Stage / gate | Status |
|---|---|
| Stage A — behavior flow (FQL `bc_only`, 500k) | **DONE** both envs: cube `bcflow_cube_single_20260726_135032`, pointmaze `bcflow_pointmaze-medium-navigate_20260729_142219` |
| D1 — flow fidelity | **PASSED** both envs (cube margin ~1000x on MMD over random control; pointmaze only 13x — weakly certified because d_a=2) |
| Stage B — preimages | **RECOMPUTE IN FLIGHT** at alpha=20, N=200 for both envs (tmux `watch_cube` / `watch_pointmaze`, ETA ~12:30 on 08-04, outputs `/data-local/amsks/PSMFLows/preimages_{cube_single,pointmaze_medium}_a20_n200.npz`). The old cube npz (alpha=1, N=100) is superseded. |
| D3 — inversion gate | **PENDING** on the new npz. History: failed at alpha=1 (ESS ~7); root-caused to proposal/target width mismatch; see §1.1 for today's twist. |
| D2 — fixed-u rollouts | **RUN BUT LOST** (08-03): the tool prints JSON to stdout only and stdout wasn't captured. Must re-run with output persisted. |
| Stage C — psmflow training | **NOT STARTED** (only a 2k-step plumbing smoke on 08-03). |
| D4 — oracle-GPI gate | Denominator now exists: FQL on pointmaze-medium, 3 seeds x 500k, mean peak **0.687 ± 0.081 sd** → gate = **peak ≥ 0.34** (50-episode evals on the 50k grid). |
| Zero-shot vs PSM/FB | Not started; `scripts/compare_multiseed.py` does not support psmflow groups yet. |

**Prior workstreams:** bilinear-PSM parity hunt **CLOSED** (no code bug; seed variance + a
500k budget ceiling — reference peaks 0.8–0.9 arrive at 650k–1350k). FB JAX port
**bit-exact** (13/13 fixture tests). Affine PSM cube push **PARKED** (flowBC actor did not
lift it off the floor; open leads = rectangular measure mesh, split x-/(s,a)-branch).

### 1.1 The ESS gate story, corrected twice — and the measurement bug found today

- 07-29: "alpha=1 is ~20x too low; alpha=20 clears the gate (ESS 21.8)".
- 08-03 (uncommitted): "that table used the broken `preimage_ess` (reported best-value on
  total failure). Re-measured: **no alpha clears 20** (saturates ~18.4); what clears it is
  `num_samples` (linear scaling). At alpha=20, N=200: ESS 49.3 cube / 82.0 pointmaze."
- **08-04 (this audit): the D3 tool gated on the mean ESS over the whole EM trace**, while
  Stage B stores (and the figure plots) only the final EM iterate — `ess` from
  `compute_full_proposal_distribution_em` is shaped `(n_steps,)` per row and the tool
  averaged both axes. Since ESS improves across EM iterations, every number the tool printed
  **understates** the stored posterior's ESS. Fixed: the gate now uses `ess[:, -1]` and also
  reports `mean_ess_trace` for comparability with older printouts. The alpha-sweep table in
  `configs/inversion/default.yaml` and the N=100-vs-200 decision were measured through the
  understating statistic. **Re-measured today at final-step ESS (cube ckpt, 256 rows,
  seed 0): alpha=20, N=100 gives mean ESS 21.7 (trace-mean 17.7) — the gate passes at
  N=100.** So "no alpha reaches 20" was the artifact; alpha=20 was right all along and N
  buys margin, not the pass. The running N=200 recompute is kept (comfortable headroom
  over a marginal 21.7); the config comment now carries the correction.

### 1.2 Uncommitted working tree

`configs/inversion/default.yaml` (alpha 20, N 200 + new sweep table),
`tools/validate_flow_inversion.py` (reads `cfg.inversion` instead of hardcoded alpha=1.0 /
3 components — a real fix; plus today's final-step ESS fix), `docs/reference_benchmarks.md`
(FQL D4 denominator section). Coherent, but commit together with: (a) the sweep-table
re-statement above, (b) a HANDOFF entry (its top slide still tells the superseded
"alpha fixes it" story), (c) a fix or annotation for `tools/plot_preimage_intuition.py`,
whose panel titles ("alpha=1 as run", "raising alpha is what clears the gate") are now
stale on both counts.

---

## 2. Roadmap

### Phase A exit (this week) — operator-driven, order matters

1. **When the precomputes land (~12:30 today):** run the fixed D3 on both npz. Gate =
   roundtrip < 0.1, chi2 typicality ≥ 0.95, mean final-step ESS > 20. Expect a comfortable
   pass (the trace-mean already read 49.3 / 82.0).
2. **Commit the inversion change** + HANDOFF update + figure fix (one commit, one story).
3. **Re-run D2** on both Stage-A ckpts with output persisted (`... | tee <file>.json`, or
   after adding the `--out`/`report_out` flag from §4). D2 is what confirms `u` indexes
   *distinct* policies — it feeds RESEARCH_NOTE §7 risk 1 and paper assumption A3, and it
   is currently the only gate with zero recorded numbers.
4. **Stage C:** `launch_psmflow.sh`, 3 seeds x 500k, both envs, one seed per GPU, pointing
   at the new npz + matching Stage-A ckpt. Before launching, add the npz↔checkpoint
   pairing assert (§5 open item 1) — it is exactly this launch that a silent mismatch
   would waste.
5. **D4 gate** on pointmaze: `tools/latent_q_sanity.py` (now seeded) ≥ **0.34** peak.
   If it fails, the debugging order is: D2 result (is the family degenerate?) → oracle-w
   vs inferred-w gap (representation vs reward-inference) → GPI budget K sweep →
   pessimism=0.
6. **Zero-shot comparison** vs in-repo PSM and FB, same envs/steps/seeds, via the new
   comparison script (§4.1). This is the first headline-able number.

### Phase B (next 2–4 weeks) — the paper's actual claim

7. **Coverage ladder** (RESEARCH_NOTE §5): full play → stitch → mode-restricted (drop k of
   m k-means modes of (s,a) — *the target regime*) → noisy → expert-only. Needs a small
   dataset-filtering tool (`tools/make_coverage_ladder.py`). Hypothesis to test: FB/PSM
   degrade sharply, PSMFlows degrades gracefully, FQL per-task upper-bounds.
8. **VC-FB / MC-FB baseline** (`enjeeneer/zero-shot-rl` port or reimplementation) — the
   must-beat: "does fixing the policy space beat fixing the loss?"
9. **Method-hygiene ablations** (each is also a theory-code gap, §5.3): pessimism penalty
   = 0 (the shipped agent quietly carries an ensemble-uncertainty penalty that undercuts
   the "we don't need conservatism" positioning); `index_mix_ratio` = 0 (paper assumes
   `u' ~ p0`; code mixes 50% permuted dataset preimages); one-step vs multi-step ODE
   decode; u'-codebook vs PSM hash codebook (main.tex Rem. 8.1 calls this "the ablation
   to run first"); point-preimage vs EM posterior.
10. **Antmaze + cube-double** expansion once pointmaze/cube-single numbers exist.

### Phase C — paper

11. Reconcile RESEARCH_NOTE with main.tex (retracted "losslessness" still in the
    contributions list; SM convention t≥0 vs t≥1; Laplace covariance formula mismatch),
    port related work + experiments into main.tex, attack Conjecture 1 (flow-error
    perturbation of the isometry) and Conjecture 4 (compact-support tension — the target
    regime is exactly where A2 is hardest).

### Success criteria — what "it works" means, concretely

- **Gate-level (Phase A):** D1–D4 all pass; specifically D4 ≥ 50% of FQL (0.34 peak on
  pointmaze-medium).
- **Method-level (first claim):** on full-coverage OGBench (pointmaze-medium, cube-single),
  psmflow zero-shot success is **within noise of PSM and FB** (mean ± 95% CI across ≥3
  seeds, overlapping intervals) — i.e. the flow reparameterization costs nothing when
  coverage is good.
- **Headline (Phase B claim):** on mode-restricted / limited-support data, psmflow
  **beats PSM and FB with non-overlapping 95% CIs** while staying within a stated fraction
  of per-task FQL; degradation across ladder rungs is measurably flatter than FB/PSM's.
- **Reporting protocol (standing):** ≥3 seeds always, 5+ for reference-grade claims; 50
  eval episodes on the 50k grid; report mean ± 95% CI across seeds plus peak-over-grid;
  never single-seed reads; compare only within shared step windows.

---

## 3. Benchmarking & baselines summary

| Baseline | Role | Status |
|---|---|---|
| FQL (per-task, reward-driven) | topline / D4 denominator | DONE pointmaze (peak 0.687); needs cube run |
| PSM (bilinear, in-repo JAX) | zero-shot baseline, same family | parity-verified vs reference |
| FB (in-repo JAX) | zero-shot baseline | bit-exact port; cube reference: 5-task mean peak 0.63 |
| VC-FB / MC-FB | the must-beat (conservatism fix) | Phase B, not ported |
| Flow BC (act with u~p0 through Stage-A flow) | floor — is the SM adding anything over the flow itself? | trivial to run, should be in every table |
| DSRL-style latent-noise SAC (per task) | upper bound on the latent space alone | Phase B, optional |

Also run the "is the SM doing work" control from main.tex §10: a per-task critic ranking
the same K decoded actions (IDQL-style). The representation earns its cost only if it beats
this.

---

## 4. Analysis & visualization scripts (why does the policy succeed/fail)

**Exists:** D1–D4 tools, `tools/plot_preimage_intuition.py` (preimage basin + alpha sweep
figure), `scripts/compare_{multiseed,fb_multiseed,protoxplant}.py` (PSM/FB only).

**To build, in priority order:**

1. **`report_out` flag on all four D-tools** (P0, ~30 min): write the JSON report next to
   the checkpoint (`<ckpt_dir>/d3_report.json` etc.). The D2 results were already lost once
   to stdout; gates must leave artifacts. (Interim: always `| tee`.)
2. **`scripts/compare_zeroshot.py`** (P0): step-aligned multi-seed table for
   psmflow/psm/fb groups — mean ± 95% CI across seeds per step, peak-over-grid per seed,
   shared-window comparison; reads `eval.csv` from run groups. Replaces the hardwired
   reference-JSON logic of `compare_multiseed.py` for the new method.
3. **`tools/viz_latent_value.py`** — the "why" plot for pointmaze (d_a=2, so u-space is
   directly plottable): heatmap of `max_u' psi(s,u',u)^T w` over a u-grid at chosen states,
   overlaid with (a) dataset action preimages at that state, (b) the GPI-selected u, (c)
   the decoded action arrows in the maze frame. Shows in one figure whether GPI selects
   inside the data manifold and whether the value landscape is smooth in u.
4. **`tools/viz_policy_rollouts.py`** — eval rollouts rendered in the env with per-step
   annotations: chosen u (colored), predicted `psi^T w`, and for pointmaze the trajectory
   overlaid on the maze vs dataset trajectories. Success/failure case gallery.
5. **`tools/calibration_check.py`** — predicted `psi(s,u,u')^T w` vs realized discounted
   return for the K GPI candidates (main.tex Rem. 9.2: widening gap with K is the
   adversarial-u failure signature). This is the single most diagnostic plot for "the
   representation lies to GPI".
6. **`tools/analyze_preimages.py`** — npz-level report generalizing
   plot_preimage_intuition: ESS / roundtrip / typicality / validity histograms, Jacobian
   singular-value spectra, and (pointmaze) spatial maps of ESS over the maze — where in
   state space is inversion weakest, and does it correlate with policy failure locations.
7. **Cluster-occupancy check** (from the project brief): k-means the dataset (s,a); compare
   cluster occupancy of dataset vs flow samples vs the *deployed GPI policy* — D1 covers
   the first two; adding the deployed policy closes the loop "the policy stays on the data
   modes".

---

## 5. Code audit (2026-08-04) — full pass over agents/, utils/, tools/, scripts/, tests/, configs/

**Overall: unusually healthy for a mid-experiment research codebase.** Failure modes are
documented with measurements at the exact line they bite; silent-failure invariants are
pinned by dedicated tests (108 test functions); every previously-claimed fix was verified
present (ESS 0.0 fallback, 1% repair ceiling, XLA guard, `return_index` now includes
affine_psm). The math has been audited repeatedly and held; the residual risk is in the
**seams between stages** (checkpoint ↔ npz ↔ decode-discretization coherence lives in
humans and shell scripts) and in **`FQLAgent` having become the dumping ground for the
inversion math**.

### 5.1 Confirmed bugs — FIXED this session

| # | What | Where |
|---|---|---|
| 1 | D3 gate averaged ESS over the whole EM trace instead of the final iterate (understates; gate stat differed from what Stage B stores) | `tools/validate_flow_inversion.py` |
| 2 | D4 gate tool's 10k relabel batch was unseeded (global np.random) — the gate number changed run to run | `tools/latent_q_sanity.py` |
| 3 | Two launchers ran bare `python` (= Python 2.7 on midi-01) | `scripts/launch_{fb,psm}_cube.sh` |
| 4 | `infer_w_goal` permutation drew from OS entropy — identical seed + weights gave different eval success | `agents/affine_psm.py:372` |

### 5.2 Confirmed defects — OPEN (prioritized)

**P0 — before/with the next launches:**
1. **No npz↔checkpoint pairing guard**: `main.py` checks only row count; a preimage npz
   from a different Stage-A ckpt / alpha vintage loads silently and Stage C trains on
   mismatched latents with no symptom. The `.meta.json` sidecar exists precisely for this —
   read it and assert `restore_path`/`flow_steps`/`alpha` against the agent config.
   Do this before the first real Stage-C launch.
2. HANDOFF + intuition-figure staleness must ship with the pending inversion commit (§1.2).

**P1:**
3. `next_noise_preimage` is dead weight: sampled (with per-row Cholesky) on **every**
   psmflow batch, consumed by nothing — free training-loop speedup to remove; also its
   `idx+1` crosses episode boundaries unmasked, and the `repair_invalid_preimages`
   docstring cites this unused pairing as its rationale.
4. `tools/validate_flow_fidelity.py` hardcodes `act[:1024]` (and `[:512]` bandwidth) in
   `mmd_rbf`, ignoring `preimage_limit` above 1024.
5. `utils/flow_steering.py`: `mode='sample'` ignores its own `rng` for the mixture draw
   (global np.random), and uses bare `vmap` over machinery documented to NaN un-jitted on
   GPU (latent — WP2 groundwork, CPU-tested only).
6. `eval_fixed_u_rollouts.py` (D2): `across` includes self-pairs and within-u pairs —
   deflates `across`, inflates the consistency ratio. Fix before D2 becomes a recorded gate.
7. `tools/export_psm_fixture.py` hardcodes a macOS path (dead on this machine);
   `scripts/transplant_eval.py` is self-declared stale — port or delete both.
8. `compare_multiseed.py` has no psmflow support and none of the compare scripts report
   95% CIs (standing convention) — superseded by §4.2's new script.
9. `gpi_decode=ode` integrates at `flow_decode_steps=10` against a 100-step-trained flow
   with no assert/comment — the exact discretization-mismatch class the repo treats as a
   bug everywhere else. Assert or document at the knob.
10. `FQLAgent.sample_actions` crashes on its declared default `seed=None`; dead
    `noise_seed` split. Latent (all callers pass a seed).

**P2:** launcher drift (MUJOCO_GL / MEM_FRAC overridability / seed-per-GPU guard /
eval_episodes 10-vs-50 / group-name hyphen-vs-underscore styles); `fql.get_config()`
`flow_steps=100` vs yaml 10 with no sync test (psmflow has one); stale "~7/100" ESS
reference comment in `utils/flow_inversion.py:171`; D3 spec says "ESS > 20/100" (a ratio)
— restate the gate as ESS/N or justify absolute-20 at N=200.

### 5.3 Theory ↔ code gaps (from the PAPER cross-check — ablate or document, don't silently ship)

1. **Pessimism penalties exist in the shipped agent** (`targets_uncertainty` on the
   measure target, `actor_pessimism_penalty` in `gpi_select`) but appear in neither paper —
   and the paper's whole positioning is "fix the policy space, not the loss". Ablate at 0
   or write the paragraph.
2. **`u'` is not drawn from `p0`**: `index_mix_ratio=0.5` mixes permuted dataset
   preimages. Prop. 7.2's C=1 hypothesis is `u' ~ p0` independent. Ablate at 0.
3. **Box clip (`u_clip=3`) is not the paper's chi-square ball `U_delta`** — the exact
   1−delta mass statement doesn't transfer.
4. **The deployed decoder is the one-step distillation**, not the ODE map the theorems
   (diffeomorphism, mass conversion, isometry) are about; plus a final `clip(a,-1,1)`.
   Currently only listed as an ablation — it is a soundness caveat.
5. **Preimage energy is un-squared L2** (`exp(-alpha·||·||)`) vs the paper's squared form
   (Eq. 18), and the implemented Laplace covariance drops the prior precision + clips
   eigenvalues — Prop. 6.2 describes a different object.
6. Argument-order trap: the code's `psi(s, u_idx, u)` is the *reverse* of the paper's
   `psi(s, u, u')`. Internally consistent; document at the network definition.
7. RESEARCH_NOTE still claims the retracted "losslessness Prop. 2" in its contributions;
   SM time-convention and Laplace formula differ between the two documents; the
   implementation follows main.tex (t≥1). Reconcile before circulating either.

### 5.4 Test-coverage gaps (ranked)

1. The four D-tools have zero tests — and they carry the spec gates (today's trace-mean
   bug is the proof). A test asserting the D3 gate stat equals mean of final-step ESS is
   ~10 lines.
2. `sample_preimage_noise` (component pick, cdf guard, Cholesky path) untested.
3. main.py psmflow wiring (`return_preimage_noise`, npz size-mismatch assert) untested.
4. Mixture-mode `next_noise_preimage` row alignment untested (moot if item 5.2.3 removes it).
5. Reward-shift *direction* (the +1 shift pointing w at goals) only partially covered.

---

## 6. Rewrite plan — making the code intuitive (no big-bang; sequenced, test-protected)

The codebase's biggest comprehension hazards, in the order a new collaborator hits them:

1. **Extract the inversion suite out of `FQLAgent`** (~200 lines of preimage math inside
   an RL agent that never calls it in training; tools reach into `_private` methods).
   Move `_get_preimage_and_jacobian`, `_get_predistribution_proposal`,
   `compute_full_proposal_distribution[_em]` to `utils/flow_preimage.py` as free functions
   taking `(flow_apply_fn, ...)`; keep thin delegating methods one release. Then split the
   113-line EM god-function into `_draw_mixture_samples` / `_is_weights` / `_m_step` so the
   war-story comments sit next to the step they describe. Equivalence/inversion tests
   protect the move.
2. **One `euler_decode(vf_apply, obs, noise, steps)` helper** — the identical Euler loop
   exists 4x (fql, psm, affine_psm, psmflow); the discretization-mismatch bug class
   becomes greppable.
3. **Shared `measure_td_loss` core** — `psmflow.measure_loss` is a near-verbatim copy of
   `psm.proto_loss` differing only in slot arguments; sharing it keeps the two agents' TD
   semantics from drifting.
4. **Per-agent `REQUIRED_BATCH_KEYS` classattr** replacing main.py's growing agent-name
   if-ladder for dataset flags (would have structurally prevented the affine_psm
   `return_index` bug).
5. **Naming/consistency sweep** (minutes each): `psmflow` logs `psm_loss` = contrastive
   only while `psm` logs contrastive+ortho under the same name (dashboards silently compare
   different quantities); three different `StepInputs` dataclasses share one name; magic
   numbers in the inversion (the fixed 5 implicit-Euler sweeps — which IS the
   divergence-tail knob — the (0.01, 1.0) Laplace eigenvalue clip, 1e-6/1e-12 floors) get
   names; `AffineMeasureNet` docstring says "shared trunk" but the code builds two trunks
   (the docstring is the stale half); trim the unused preimage arrays from the psmflow
   batch payload.

Sequencing: 1–3 are half-day moves each, best done in the Stage-C training window (they
don't touch training semantics; the fixture tests pin them). 4–5 are opportunistic.

---

## 7. Session log (what this audit changed)

- Fixed: D3 final-step ESS gating (+ `mean_ess_trace` for old-number comparability), D4
  relabel seeding, `.venv` python in the fb/psm cube launchers, seeded `infer_w_goal`
  permutation.
- Re-measured the alpha sweep question at final-step ESS on the cube Stage-A ckpt
  (result recorded in `docs/HANDOFF.md`, 2026-08-04 entry).
- No commits made; working tree additionally contains the pre-existing inversion/benchmark
  changes from 08-03.

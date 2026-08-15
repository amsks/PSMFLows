# Priority stack: implementation + runs (2026-08-14, post-audit)

Context: the five-agent audit (HP-vs-reference, PSM survey, core-code, pipeline/tools,
ideas-verification) found (a) one deep structural deviation — φ is trained by a
pessimistic, actor-bootstrapped, proto-branch-less loss unlike every working
implementation, and w is inferred through that φ — and (b) no bug invalidating any
published number, but three paper-bound claims needing repair. The proven mechanism is
the bounded value-directed residual (per-task 0.22 → 0.91). The hybrid (zero-shot,
0.302 sd1 / 0.048 sd0, curve rising at 500k) is the flagship and currently runs with a
healthy action-critic steered through the diseased basis. LIBERO is explicitly OUT of
scope for this plan.

Ground rules unchanged: `.venv/bin/python`, one training seed/GPU, evals share at
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.30`, named tmux + tracked watcher discipline (never
end a turn without a live tracked watcher or the report), report_out JSONs to
`/data-local/amsks/PSMFLows/logs/`, 500-ep Wilson or seed mean ± 95% CI, BC control
0.068 quoted, document every launch in this file's addendum at launch time, commit
nothing without the user.

## P0 — protect what is already claimed (tool fixes; ~1 day CPU/eval, DO FIRST)

These repair audit findings that touch paper-bound numbers. All are small.

1. **Radius sweep redo** (`tools/diag_action_coverage.py:151`): the published
   "tripling the latent radius doesn't move coverage" was vacuous — `clip(N(0,1), ±r)`
   never samples wider latents. Replace with SCALED draws (u = r/3 · N(0,1), clipped at
   ±r) for r ∈ {3, 4.5, 6}, rerun on the fs100 flow, and update the verdict. Then fix
   `PAPER/ICLR/note.tex` accordingly: if radius still does nothing, the claim stands
   with a corrected method; if wider latents DO recover coverage, that is a new finding
   — flag to the user immediately before touching the note's argument.
2. **Seed the paper eval path** (`tools/eval_checkpoint.py` + `utils/evaluation.py:73`):
   seed np.random and the relabel batch from `cfg.seed`; fix the false docstring claim.
   Do NOT re-run old evals (statistically valid; Wilson CIs honest) — new evals are
   reproducible from here on. One protocol sentence queued for the paper.
3. **Geometry harmonization**: one shared helper for k-NN neighborhoods (standardized
   obs, k=32) used by `diag_action_coverage`, `diag_action_distance`,
   `diag_generated_pair_support`; recompute the coverage ratio + C3 ceiling under the
   standardized geometry so the support-curve figure quotes ONE protocol. Keep the raw
   numbers in the JSONs for continuity, labeled.
4. **The missing 2a Q-gap probe**: Q(s, a_FQL(s)) − Q(s, a_data-NN(s)) at dataset
   states, on the 3 FQL checkpoints — completes the pre-registered calibration verdict
   the paper cites. Fold into `diag_fql_calibration.py`.
5. **Calibration tool penalty fix** (`diag_fql_calibration.py:73-74`): `q_unc` is half
   the agent's `targets_uncertainty` and uses the wrong knob; fix both, note in the
   JSON that prior collapse verdicts UNDERSTATED underestimation (direction unchanged).
6. **Pairing-guard subset identity** (`main.py:81`, `precompute_preimages.py:150`):
   write `dataset_fraction`/`dataset_fraction_seed` into the npz sidecar; guard
   compares them, plus an `np.array_equal` spot-check on observations (first+last 1k
   rows is enough). Closes the silent wrong-subset trap.
7. **Defuse the ψ_a pessimism landmine** (`agents/psmflow.py:241-242`): the per-feature
   vector penalty is sign-indefinite in Q-space. Either implement scalar-Q pessimism
   (penalize ψ_aᵀw uncertainty, not per-feature SF) or rename the knob
   `UNSAFE_feature_pessimism` with a docstring warning. Must land before any P2
   pessimism ablation.
8. Two cheap latents from the audit while in there: `latentrl` smoke tests (residual
   inertness at ε=0, target-critic usage — it produced the ceiling number and has zero
   coverage); PSD jitter + zero-weight guard in the training-path mixture sampler
   (`utils/datasets.py:150`) matching the fidelity tool's.

## P1 — hybrid consolidation (runs; launch alongside P0 wherever GPUs allow)

1. **Hybrid seeds 2–4 at 1M steps** (`psmflow_hybrid` config unchanged — pessimism 0.0
   ψ_a branch, ε=0.05), tmux `hybrid_1M_sd{2,3,4}`. Rationale: 2 seeds is not a claim;
   the curve was still rising at 500k; every reference recipe trains 1M. Also EXTEND
   sd1 (the 0.302 seed) to 1M from its 500k checkpoint if restore-and-continue is
   supported; else include seed 1 in the fresh 1M set.
2. **λ-rank eval variant** (eval-time only, existing checkpoints): score K=32 decoded
   candidates by Q_a, execute argmax, NO residual — one 500-ep eval per hybrid seed.
   Separates "Q_a ranks better" from "Q_a pushes better." ~2h total.
3. **hpmatch 500-ep evals** (still owed): finals of both seeds → closes the audit-doc
   loop on whether the HP bundle matters at all.
4. In-loop metric to add before the 1M launches (small code): log ψ_aᵀw spread over
   decoded candidates per eval — the live "is the action critic awake" signal, so we
   stop discovering flat critics post-hoc.

## P2 — the FB-graft arm (the audit's highest-expected-value new training)

Build: **unshare the basis for the action branch.** ψ_a gets its own backward map
B_a(s') trained by ψ_a's own TD loss (FB recipe verbatim from `agents/fb.py`: γ 0.99,
τ 0.005, ortho as FB runs it, no pessimism), and **w_a is inferred from B_a**
(z = E[r·B_a]/norm) for the residual/Q_a path; ψ_u/φ/w_u stay untouched for the latent
actor. Acting: unchanged residual mechanism but δ and Q_a driven by (ψ_a, w_a).
This severs the hybrid's dependence on the diseased φ — the compass gets FB's proven
representation while actions stay decoder-anchored with a measured budget.

- Tests first: disabled ⇒ byte-identical (same pattern as the action_critic tests);
  B_a receives no gradient from ψ_u paths and vice versa.
- Runs: 2 seeds × 1M, tmux `hybrid_fbgraft_sd{0,1}`.
- Gate: 500-ep ≥ 0.45 on either seed ⇒ scale to 5 seeds and this becomes the paper's
  flagship agent. Below the current hybrid ⇒ the shared-φ theory of the gap is wrong —
  report, don't iterate blindly.
- Fallback arm (only if the graft underperforms AND capacity exists): the
  reference-faithful proto-branch port for φ (measure-pessimism 0, φ trained by the
  codebook branch only, detached in the w-branch, bc_coeff 3.0) — bigger build,
  pre-specify in an addendum before starting.

## P3 — finish the fraction story

1. psmflow_frac50 completes automatically (chain running); fold into
   `table_dataset_fraction.json` and refresh the table.
2. **Hybrid data-efficiency cells**: hybrid (current config) on the 10% and 50% subsets,
   1 seed each, 500k (same subset seed 0; the frac flows + preimages exist). The
   question with actual upside: does the action critic rescue the low-data regime where
   old PSMFlows collapsed to the BC floor (0.067)? tmux `hybrid_frac{10,50}_sd0`.
3. The 2a Q-gap probe (P0.4) plus existing calibration completes the pre-registered
   Idea-2 verdict for the note.

## P4 — paper sync (after P0–P1 results exist)

- note.tex: radius-claim correction (P0.1's outcome), eval-protocol sentence (P0.2),
  geometry-protocol note (P0.3), the hybrid section updated with multi-seed 1M numbers
  and the λ-rank ablation, fraction table refresh. HANDOFF entry for the audit-fleet
  verdicts (five reports, one structural finding, three claim repairs) — this is
  currently in no dated doc.
- The audit reports themselves: save the five agents' findings under
  `docs/audits/2026-08-14-fleet/` (one md each, verbatim) so they outlive this session.

## GPU schedule sketch (3 GPUs assumed)

```
Day 1: P0 (CPU/eval) · P1.1 hybrid_1M sd2/sd3 on two GPUs · P1.3 hpmatch evals shared
Day 2: P1.1 sd4 + sd1-extend · P2 build+tests · P1.2 λ-rank evals shared
Day 3: P2 fbgraft sd0/sd1 (2 GPUs) · P3.2 hybrid_frac10 (1 GPU) · P0 note fixes
Day 4: P2 gate call · P3 table · P4 paper sync
```

## Acceptance checklist

- [ ] P0.1 radius sweep redone with scaled draws; note.tex claim corrected; any
      surprise flagged before editing the argument
- [ ] P0.2 evals seeded; P0.3 one geometry protocol in the support-curve figure
- [ ] P0.4 Q-gap probe JSON; P0.5 calibration fix + understated-verdict note
- [ ] P0.6 sidecar records fraction/seed + content spot-check guard
- [ ] P0.7 ψ_a pessimism knob fixed or loudly renamed
- [ ] P1: 4–5 hybrid seeds at 1M with 500-ep evals; λ-rank ablation table; hpmatch
      closed with 500-ep numbers
- [ ] P2: graft tests green before launch; gate applied; no blind iteration
- [ ] P3: fraction table complete incl. psmflow@50% and hybrid cells
- [ ] P4: note + HANDOFF synced; audit reports archived under docs/audits/
- [ ] Every launch documented at launch time; tmux server killed at the end

---

## Addendum — launches and results (append at launch time)

### 2026-08-14 00:57 — P1.1 hybrid 1M, seeds 2 and 3
tmux `hybrid_1M_sd2` (GPU0), `hybrid_1M_sd3` (GPU3). `agent=psmflow
agent.action_critic.enabled=true agent.residual_eps=0.05`, flow
`bcflow_cube_single_20260726_135032/sd000`, point preimages, `offline_steps=1000000`,
eval every 50k @50 episodes, `run_group=psmflow_hybrid_1M`. ETA ~20 h each
(13.5 it/s). Logs `logs/hybrid_1M_sd{2,3}.log`.

### 2026-08-14 00:57 — P0.1 radius sweep with SCALED draws (tmux `p0radius`, done 01:11)
`logs/diag_action_coverage_fs100_scaledradius.json`, fs100 flow. **The published claim
does not survive the corrected method.** Coverage ratio (decoded action sd / behaviour
conditional sd), raw-observation geometry as published:

| decode | r=3.0 | r=4.5 | r=6.0 |
|---|---|---|---|
| onestep (deployed) | 0.632 | 0.986 | 1.407 |
| ODE | 0.638 | 1.026 | NaN (decode diverges) |

Radius gain +0.776, i.e. **widening the latent prior DOES recover coverage** — r=4.5
reaches the aleatoric ceiling (0.936) and r=6 over-disperses. The old "tripling the radius
does not move coverage" compared three near-identical clipped samples and was vacuous.

BUT the killer statistic moves the WRONG way: min‖G(s,u) − a_FQL(s)‖ normalised by action
scale goes 0.187 → 0.234 → 0.279 as r grows (same 512 draws spread thinner), while C1 is
unchanged (a_FQL sits 0.577 from the data's own k-NN actions, still off-support). So wider
latents buy SPREAD, not proximity to the working policy's actions: the reachability
conclusion stands, its stated reason does not. **Flagged to the user; note.tex argument NOT
edited pending their call.**

### 2026-08-14 01:18 — P0 eval/diagnostic chain (tmux `p0evals`, GPU1 @ mem 0.20)
`scratchpad/chain_p0evals.sh`, sequential: (1) coverage re-run under the harmonised
standardized geometry → `logs/diag_action_coverage_fs100_stdgeom.json` (P0.1 final + P0.3);
(2) `diag_fql_calibration` with the P0.5 penalty fix and the new P0.4 2a Q-gap probe on the
three FQL checkpoints `fqlbaseline_cube_a300_20260810/sd00{0,1,2}` →
`logs/diag_fql_calib_qgap_sd00{0,1,2}.json`; (3) **P1.3** hpmatch 500-episode evals of
`psmflow_hpmatch_20260813/sd00{0,1}` @500k → `logs/eval500_hpmatch_sd00{0,1}.json`.

### 2026-08-14 01:33 — P0.3 result: the geometry choice changes numbers, not conclusions
`logs/diag_action_coverage_fs100_stdgeom.json`. Under the shared standardized protocol
(k=32, z-scored observations) against the published raw-observation geometry:

| quantity | standardized | raw (published) |
|---|---|---|
| deployed coverage (onestep, r=3) | 0.688 | 0.632 |
| coverage at r=4.5 / r=6 | 1.076 / 1.533 | 0.986 / 1.407 |
| C1: ‖a_FQL − nearest k-NN action‖ / scale | 0.537 | 0.577 |
| C3 aleatoric ceiling | 1.196 (interleaved split) | 0.936 (contiguous) |

Both verdicts survive verbatim: FQL is off-support (0.537 vs our best decode 0.187, so the
capacity arm stays cancelled) and the radius genuinely widens coverage while pushing the
decode FURTHER from a_FQL. The support-curve figure can now quote one protocol.

### 2026-08-14 01:37 — P0.4/P0.5 + P1.2 (tmux `p0b`, GPU1, queued behind `p0evals`)
The first calibration attempt inside `p0evals` died instantly on a hydra override
(`calib_episodes` is not a base-config key, needs `+calib_episodes`); no other step was
affected. `scratchpad/chain_p0b.sh` waits for the `p0evals` sentinel, then reruns the three
FQL calibrations with the Q-gap probe, then runs **P1.2**: `eval_rank_k=32` λ-rank evals of
the two 500k hybrid seeds → `logs/eval500_hybrid_lambdarank_sd00{0,1}.json`.

### 2026-08-14 01:20 — P1.1 queue behind the running seeds (tmux `queue_gpu0`, `queue_gpu3`)
`scratchpad/queue_after_hybrid.sh` waits on each training PID, then on that GPU: 500-episode
eval of the finished 1M run, then **GPU0 → fresh seed 4 at 1M**, **GPU3 → sd1-extend**
(restore `psmflow_hybrid_20260813/sd001` @500k, +500k steps, `run_group=psmflow_hybrid_1M_ext`;
its own step counter restarts, read the curve with a +500k offset) followed by its 500-ep
eval. No GPU is free before then: GPU2 belongs to another user (24 GB, `idutta`).

### 2026-08-14 — P0 code landed (tests green: 49 passed across the touched suites)
- **P0.1** `diag_action_coverage`: scaled draws `u = (r/3)·N(0,1)` clipped at ±r.
- **P0.2** `utils/evaluation.evaluate` now pins the ACTION-noise key to `seed` (it was
  `np.random.randint` regardless of the seed); `tools/eval_checkpoint` seeds the global
  numpy stream so the relabel batch is reproducible; the false "reproduces the in-loop
  number exactly" docstring corrected. New regression tests in `tests/test_eval_seeding.py`.
  Old evals stand as published (statistically valid); reproducibility applies from here.
  Note for the multi-seed table: sd2/sd3 were launched before this landed, so their IN-LOOP
  eval curves draw action noise from entropy while sd4's are pinned — a different noise
  realisation, not a bias, and every 500-episode final in the table is produced by the fixed
  tool. Same for the ε-jitter now added to the mixture sampler (point preimages are used by
  every run in flight, so it does not touch them at all).
- **P0.3** new `utils/geometry.NeighbourIndex` — one protocol (per-dim standardized
  observations, cKDTree, k=32, seeded index, exact self-exclusion) now used by
  `diag_action_coverage`, `diag_action_distance`, `diag_generated_pair_support`; every
  report embeds `geometry`. Coverage keeps the raw-geometry numbers under `*_raw_geometry`.
  C3's split-half is interleaved (distance-balanced) under the sorted shared geometry.
  `tests/test_geometry.py`.
- **P0.4** 2a Q-gap probe folded into `diag_fql_calibration`: Q(s, a_agent) − Q(s, a_data)
  at 512 dataset states, plus the best in-sample k-NN competitor, the fraction of agent
  actions beyond the data's own p95 match distance, and the gap-vs-distance rank correlation.
- **P0.5** the pessimistic value now uses the ACTOR's pessimism knob and the agent's own
  `targets_uncertainty` (|q0−q1| at P=2, was |q0−q1|/2 with the target knob). Both errors
  shrank the penalty, so **prior collapse verdicts understated** the over-estimation;
  directions unchanged. Recorded in every new JSON as `penalty_note`.
- **P0.6** `precompute_preimages` writes `dataset_fraction`/`dataset_fraction_seed` into the
  `.meta.json`; `main.py` asserts they match and additionally compares the first and last
  1k observation rows of the npz against the env dataset (verified to pass on the live
  full-dataset npz). Sidecars written before today fall back to the content check.
- **P0.7** ψ_a pessimism is now **scalar-Q**: blend weight λ∈[0,1] between the target
  ensemble mean and its least-task-valued member, replacing a per-feature penalty that was
  sign-indefinite in Q-space (it RAISED Q wherever w was negative). Bit-identical at λ=0,
  which is every run to date. Pinned by two tests.
- **P0.8** `sample_preimage_noise` (the TRAINING-path mixture sampler) gets the PSD jitter
  the fidelity tool already had, plus symmetrisation, an eigen-clip fallback, and a
  zero-weight-row guard; new `tests/test_latentrl_smoke.py` covers residual inertness at
  ε=0, the eps budget, target-critic usage/polyak, and flow freezing.
- **P1.4** in-loop `ac_q_spread{,_rel}` / `ac_q_range_rel` — Q_a spread over decoded
  candidates, logged every step the branch is enabled, so a flat action critic shows up
  live instead of post-hoc. (sd2/sd3 predate it; sd4 and the graft arm will have it.)
- **P1.2** λ-rank acting implemented as eval-only `agent.action_critic.eval_rank_k=K`:
  rank K decoded candidates by Q_a, execute the argmax, no residual.
- Enabling: `utils/flax_utils.restore_agent` now tolerates checkpoints written before a
  field existed (the hpmatch checkpoints have no psi_a/residual and could not otherwise be
  loaded at all); extra fields in a checkpoint remain a hard error.

### 2026-08-14 01:56 — P1.3 CLOSED: the HP bundle buys nothing
500-episode evals of `psmflow_hpmatch_20260813` @500k: sd0 **0.276** [0.239, 0.317],
sd1 **0.192** [0.160, 0.229]. Plain PSMFlow over 5 seeds is **0.236 ± 0.071** (mean ± 95%
CI: 0.318/0.236/0.176/0.260/0.188). Both hpmatch seeds sit inside that interval, so
matching the FB reference's hyper-parameter bundle does not move Stage-C. The audit-doc
loop on "does the HP bundle matter at all" is closed: no.

### 2026-08-14 01:59 — ⚠ P0.4 SECOND FLAG: the C1 off-support verdict lacks its baseline
First Q-gap report (`logs/diag_fql_calib_qgap_sd000.json`, FQL sd0, 512 dataset states,
shared geometry):

- Q(s, a_FQL) − Q(s, a_data) = **+0.570** (+1.6% of |Q_data|); the critic prefers its own
  action on 68% of states — a small, consistent preference, not an over-estimation spiral.
- Against the best in-sample k-NN action the gap is **−2.31**: the critic rates some
  neighbouring dataset action ABOVE FQL's own action, so FQL is not the critic-argmax over
  the local data.
- **The support number that matters:** FQL's actions sit 0.522 (normalised) from the local
  action cloud while the data's own self-match p95 is **1.030** — only **4.9%** of FQL's
  actions are beyond what real data routinely is from its own neighbours.

That last line undercuts C1 as stated. C1 called FQL off-support by comparing its 0.537
against OUR DECODE's 0.187 with a hard-coded 0.6 factor and no data-matching-itself
calibration; with the calibration, the honest reading flips to: **our decodes are
anomalously tight (0.19 ≪ the data's own typical self-match), while FQL's actions are
ordinary for this dataset.** The conclusion "the task needs actions the behaviour
distribution does not contain, so capacity/retrain is CANCELLED" is a paper-bound claim
resting on the uncalibrated comparison. Flagged, not edited. `diag_action_coverage` now
computes that baseline (`c1_fql_offsupport.data_self_match_baseline`) and a rerun is queued
(tmux `p0c`, after `p0b`) → `logs/diag_action_coverage_fs100_c1calib.json`.
Also note P0.5's fix does not move the FQL numbers (both its pessimism knobs are 0.0); the
understated-penalty caveat applies to the latentrl collapse reports, which ran at 0.5.

Seed 1 (02:00) reproduces all of it — gap +0.550 (+1.5%), vs k-NN best −2.42, preference on
63% of states, d_agent 0.513 against the same p95 1.030, 4.3% off-support — so the flag is
not one checkpoint's quirk. (Rollout bias differs by seed as before: +7.71 vs +2.06.)

### 2026-08-14 02:40 — C1 calibrated on its own states: FQL is NOT off-support
`logs/diag_action_coverage_fs100_c1calib.json`, same 64 states and same statistic C1 has
always used, now with the data-matches-itself baseline alongside:

| quantity (normalised by mean|a|) | value |
|---|---|
| a_FQL → nearest k-NN action (the C1 number) | 0.537 (mean over states 0.560) |
| **real data → its own neighbours' actions (self excluded)** | **0.582 mean, p95 1.165** |
| fraction of a_FQL beyond that p95 | **6.3%** |
| our best decode → nearest k-NN action | 0.187 |

FQL's actions are as far from their neighbourhood as REAL DATA ACTIONS ARE (0.560 vs
0.582). The dataset simply is that dispersed locally. So "the task needs actions the
behaviour distribution does not contain" is not what these numbers say. What they do say is
the mirror image: **our decode at 0.187 is ~3× TIGHTER than the data's own self-match — the
flow decodes into an over-concentrated set.** That also explains the P0.1 radius result
(scaling the prior takes coverage from 0.69 to ~1.0, i.e. up to the data's own dispersion)
and it means the capacity/retrain arm was cancelled on a comparison that had no yardstick.

Both flags therefore converge on ONE reframing: the binding constraint looks like decoder
concentration, not missing data support. The `capacity_arm: cancelled` string in the JSON is
left exactly as it was — the verdict rule is unchanged and unedited, pending the user.

### 2026-08-14 02:25 — P1.2 answered: the gain is the PUSH, not the ranking
500-episode λ-rank evals (K=32 decoded candidates scored by Q_a, argmax executed, no
residual) against the same checkpoints' deployed residual acting:

| seed | λ-rank (rank only) | deployed (draw + residual) | BC control |
|---|---|---|---|
| sd0 | **0.004** [0.001, 0.015] | 0.048 [0.033, 0.070] | 0.068 |
| sd1 | **0.162** [0.132, 0.197] | 0.302 [0.263, 0.344] | 0.068 |

Non-overlapping intervals on sd1 and an order-of-magnitude drop on sd0. Since the ranked
candidates ARE actor draws, an uninformative Q_a would land near the actor's own number;
landing well below it says the ordering is not merely uninformative on sd0. Verdict: the
hybrid's gain comes from the residual PUSH, not from Q_a's ability to rank — which is also
the arm that the (still unlaunched) FB graft is designed to improve, since a ranking this
weak is what a diseased basis would produce.

### 2026-08-14 03:02 — ⚠ THIRD FLAG: the decode-only control beats the flagship on average
The third arm landed (`eval_rank_k=1`: one actor draw, decoded, **no residual, no
selection** — i.e. plain PSMFlow acting out of a hybrid-trained checkpoint):

| seed | decode-only | λ-rank (K=32) | **deployed (draw + residual)** |
|---|---|---|---|
| sd0 | **0.266** [0.229, 0.306] | 0.004 [0.001, 0.015] | 0.048 [0.033, 0.070] |
| sd1 | **0.200** [0.167, 0.237] | 0.162 [0.132, 0.197] | **0.302** [0.263, 0.344] |
| mean of 2 | **0.233** | 0.083 | 0.175 |

Three things follow, and none of them are what the current write-up says:

1. **The residual is a coin flip, not a mechanism.** On sd1 it adds +0.10 (CIs disjoint); on
   sd0 it COSTS 0.22 (0.266 → 0.048, CIs disjoint). Across the two seeds the deployed
   hybrid mean (0.175) is BELOW the decode-only mean (0.233). The flagship 0.302 is one seed
   where the coin landed heads. The proven-mechanism claim comes from the per-task W4 result
   (0.22 → 0.91, latentrl, true task reward) — it does not transfer to the zero-shot setting
   on this evidence.
2. **Q_a's ranking is actively harmful**, not merely uninformative: λ-rank is below
   decode-only on BOTH seeds, i.e. selecting by ψ_aᵀw is worse than not selecting at all.
3. Decode-only (0.266 / 0.200) sits right on plain PSMFlow's 0.236 ± 0.071 — consistent
   with the action branch contributing nothing positive through the shared basis, which is
   precisely the hypothesis the P2 FB graft tests. The graft is now the most informative
   thing on the schedule, not a nice-to-have.

Consequence for the runs in flight: a single-mode number at 1M is uninterpretable, so
**every 1M checkpoint is now evaluated both ways** — tmux `queue_dec1M` waits for
`params_1000000.pkl` on sd2/sd3 and runs the decode-only eval →
`logs/eval500_hybrid_1M_decodeonly_sd00{2,3}.json`. The trainings themselves are untouched.

### Test-suite note: the whole-suite-in-one-process run is unsound, and it is not new code
Running all 177 tests in ONE process dies partway through, and the crash site MOVES with
whatever is deselected — three runs, three different victims, two different signals:

1. full suite → SIGABRT in `test_psmflow_action_critic::test_graft_trains_its_own_basis…`
   (inside XLA's `backend_compile_and_load`, no error text);
2. same, that test deselected → SIGSEGV in `test_psmflow_actor::test_actor_trains_every_update`;
3. same, ALL of today's new tests excluded → SIGSEGV in
   `test_psmflow_groundtruth::test_chain_q_prefers_goalward_latent`.

Run (3) is the discriminating one: with nothing of today's in it, it still dies. So this is
in-process accumulation in the CPU XLA compiler, not something today's code introduced. (A
clean `HEAD` export passed 146/1-skipped, but that export has none of the untracked test
files, so it ran 147 tests against 177 — less compile pressure, and NOT evidence either
way. Recording that here because it is the tempting wrong conclusion.) The host is not out
of memory (190 GB free at crash time). Per-module runs are the sound way to run this suite.

**Module-per-process sweep (the number to quote): all 29 modules green — 177 passed, 1
skipped, 0 failed.** Slowest: `test_affine_psm_inference` 5m54s, `test_psmflow_action_critic`
1m37s, `test_psmflow_groundtruth` 1m19s, `test_psmflow_agent` 1m07s; everything else under a
minute. Recommended invocation until someone fixes the in-process compiler death:
`for f in tests/test_*.py; do .venv/bin/python -m pytest "$f" -q; done` (or pytest-forked).

### 2026-08-14 01:39 — P3.2 hybrid data-efficiency cells queued (tmux `queue_hybfrac`, GPU1)
`scratchpad/queue_hybrid_frac.sh` waits for the frac50 chain's sentinel, then runs the
hybrid (action critic + ε=0.05) at 500k on the 10% and then the 50% subset, each against
its own Stage-A flow and preimages, each followed by a 500-episode eval →
`logs/eval500_hybrid_frac{10,50}_sd0.json`. The question: does the action critic rescue the
low-data regime where PSMFlow collapsed to the BC floor (0.067)?

### 2026-08-14 — P2 FB graft BUILT (code + tests; no run yet, no GPU is free)
`agent.action_critic.fb_graft=true` gives the action branch its own basis: a second
`PhiMap` **B_a** (BackwardMap is PhiMap in this codebase) with its own optimizer and
target, trained jointly with ψ_a by the FB measure loss (`contrastive_loss` +
`ortho_coef=1000` on B_a, no pessimism, `b_tau=0.005`), its own task vector w_a sampled
against B_a exactly as FB samples z against B, and **w_a inferred from B_a** at eval
(`infer_z_a`). ψ_u/φ/w_u and the latent actor are untouched — only δ and Q_a move to
(ψ_a, w_a). Six tests pin it: byte-identity with the graft off (params, rng, and
`task_w_a == task_w`), B_a and ψ_a train while the shared branches take exactly the step
they take without the graft, **both-directions gradient isolation** (∂graft/∂φ ≡ 0 and
∂measure/∂B_a ≡ 0), w_a ≠ w under the graft and w_a ≡ w without it, and the ε budget still
holding at acting time.

**01:48 — the graft runs are queued, not blocked**: tmux `queue_graft0`/`queue_graft3`
wait for the sd4 / sd1-extend chains to end and then start
`scratchpad/fbgraft_run.sh` in sessions named `hybrid_fbgraft_sd{0,1}` (GPU0 seed 0,
GPU3 seed 1): `agent.action_critic.fb_graft=true` on the hybrid recipe, 1M steps,
`run_group=psmflow_fbgraft_1M`, each followed by its 500-episode eval →
`logs/eval500_fbgraft_sd{0,1}.json`. Gate stays manual: 500-ep ≥ 0.45 on either seed ⇒
scale to 5 seeds; below the current hybrid ⇒ the shared-φ theory of the gap is wrong,
report rather than iterate. Ordering matches the plan's own schedule (sd4/sd1-ext Day 2,
graft Day 3); GPU2 is another user's, so nothing starts sooner.

### 2026-08-14 12:00–13:00 — P1.1 at 1M, both acting modes: the residual is net NEGATIVE
Paired on identical checkpoints (deployed = actor draw + residual; decode-only =
`eval_rank_k=1`, same draw, no residual):

| seed | steps | deployed | decode-only | residual effect |
|---|---|---|---|---|
| sd0 | 500k | 0.048 | 0.266 | **−0.218** |
| sd1 | 500k | **0.302** | 0.200 | +0.102 |
| sd2 | 1M | 0.134 | 0.260 | **−0.126** |
| sd3 | 1M | 0.162 | 0.180 | −0.018 |
| sd1-ext | 1M | 0.284 | — | — |

- **Deployed, 4 seeds: 0.162 ± 0.168. Decode-only, same 4 checkpoints: 0.226 ± 0.068.
  Plain PSMFlow, 5 seeds: 0.236 ± 0.071.** The action branch's contribution in zero-shot is
  ≤ 0, and its own no-residual control is statistically indistinguishable from plain
  PSMFlow — which is what decode-only IS.
- Mean paired residual effect **−0.065**, helping on exactly one seed of four.
- **1M did not pay off.** The premise for the 1M launches was "the curve was still rising at
  500k": fresh 1M seeds land at 0.134/0.162, below sd1's 500k 0.302, and extending sd1
  itself 500k → 1M moved 0.302 → 0.284 (overlapping CIs). Between-seed spread dominates
  training length, and the deployed arm's ±0.168 interval is the headline problem.

### 2026-08-14 12:44 — P3.2 the one place the hybrid wins: 10% data
`eval500_hybrid_frac10_sd0` = **0.238** [0.203, 0.277] against plain PSMFlow at 10%
(0.068 / 0.066, two seeds) and the BC control (0.070) — a 3.5× gap with disjoint intervals.
Also P3.1: `psmflow_frac50` = 0.052 [0.036, 0.075], i.e. plain PSMFlow collapses to the BC
floor at BOTH 10% and 50% while reaching 0.236 on the full set (FQL: 0.408 at 10%, 0.976 at
50%). So the low-data collapse is real and the hybrid appears to rescue it.
**The control came back and it credits the residual.** Decode-only on the same fraction
checkpoints: **0.072** [0.053, 0.098] at 10% and **0.070** [0.051, 0.096] at 50% — the BC
floor, exactly where plain PSMFlow sits (0.068 / 0.052). So in the low-data regime the
residual carries the entire result (0.238 and 0.228 deployed), which is the mirror image of
the full-data regime where it costs 0.065 on average:

| data | deployed | decode-only | plain PSMFlow | residual effect |
|---|---|---|---|---|
| 10% | **0.238** | 0.072 | 0.068 | **+0.166** |
| 50% | **0.228** | 0.070 | 0.052 | **+0.158** |
| 100% | 0.162 ± 0.168 | 0.226 ± 0.068 | 0.236 ± 0.071 | −0.065 |

At 10% the hybrid is also the best zero-shot method on the board by a wide margin — FB
collapses there (0.030, below the BC control's 0.070) while the hybrid holds 0.238. One seed
per fraction cell, so this needs replication before it is a claim, but it is the first
setting where the action branch earns its place, and it suggests the residual matters
precisely when the behaviour flow is weakest.

### 2026-08-14 19:30 — P2 GATE CALL: the FB graft FAILS, and the shared-φ theory with it
`hybrid_fbgraft_sd1` finished 1M. Both acting modes, 500 episodes:

| | deployed | decode-only |
|---|---|---|
| **FB graft, sd1** | **0.064** [0.046, 0.089] | 0.174 [0.143, 0.210] |
| hybrid (same recipe, shared basis) | 0.162 ± 0.168 (4 seeds) | 0.226 ± 0.068 |
| BC control | 0.068 | — |

**Gate was 500-ep ≥ 0.45 on either seed. Deployed reads 0.064** — a factor of seven below
the gate, at the BC control, and below the current hybrid. The pre-registered rule for this
outcome is explicit: *"Below the current hybrid ⇒ the shared-φ theory of the gap is wrong —
report, don't iterate blindly."* So: no 5-seed scale-up, no graft tuning.

What it rules out: the action critic's weakness was blamed on being steered through a φ
trained by a pessimistic, actor-bootstrapped, proto-branch-less loss. Give ψ_a its own
FB-recipe basis B_a and its own w_a — the audit's highest-expected-value fix — and the
residual gets WORSE (0.174 → 0.064 within this run; the hybrid's own residual effect is
−0.065 on average). The basis is not what is wrong with the action branch.

The decode-only number (0.174) sits inside the hybrid's own decode-only seed spread
(0.180–0.266), which is the expected null: the graft touches only ψ_a/B_a/w_a and the
residual, so the latent actor should be untouched — and measurably is. That is a useful
positive control on the implementation: the graft did what it claimed, and what it claimed
did not help.

sd0 is left running (≈5 h out, GPU0 idle otherwise) purely so the negative result has n=2.

### 2026-08-14 — user's three preimage-pipeline claims: all three confirmed
**(1) The full inversion utility is not used.** `configs/inversion/default.yaml` sets
`num_clusters: 1`, so `compute_full_proposal_distribution_em` runs at `n_components=1`:
responsibilities, per-component covariances and mixture weights all collapse to a single
importance-weighted Gaussian, and `noise_preimage_weights` is a constant 1.0 in every file
we own (confirmed in each npz sidecar). The non-EM variant
`compute_full_proposal_distribution` is reachable only from `utils/flow_steering.py`
(`mode='mean'`) — nothing in the training or preimage path calls it.

**(2) Everything runs on point preimages, so the distributions are dead weight.** The yaml
default is `use_point_preimage: false`, but every launch command passes
`agent.use_point_preimage=true` (verified in the flags.json of hybrid_1M sd2, hybrid sd1,
hpmatch sd0), and `configs/agent/latentrl.yaml` defaults to true. So the training path reads
only `noise_preimage_point`; the stored mean/cov/weights arrays are carried and never read.
(This is also why the P0.8 PSD-jitter fix in `sample_preimage_noise` cannot affect any run
in flight — that sampler is off the point-preimage path entirely.)

**(3) The inversion target has no prior term — confirmed, and it is the root cause of the
mixture pathology.** The target was π(u) ∝ exp(−α‖G(s,u) − a‖) with no N(0, I) factor, so
it is FLAT in every direction the decoder cannot see and the EM fit has no finite answer.
Evidence, in order of directness:

- the code's own comment already recorded the runaway: max covariance eigenvalue
  1 → 6 → 34 → 305 → 2281 → 6095 over 8 EM steps;
- in the SHIPPED npz files: cube 82–84% of rows carry a fitted per-dim variance wider than
  the prior (max 9.8, |μ| to 15.4); pointmaze mean per-dim variance 9.8, p99 66.8, max
  3.8e4, |μ| to 205.8;
- the control that isolates it: the exact backward-ODE point preimages in the SAME files
  are prior-like (|u| mean 0.85, p99 2.7, 0.5% beyond u_clip=3). The inverter is fine; the
  target was wrong.

Fixed: `inversion.prior_scale` (default **1.0**) puts log N(u; 0, I) into both targets;
`0.0` reproduces legacy files exactly and is recorded in the sidecar. On the REAL cube
Stage-A flow, 512 rows, α=20, N=200:

| target | \|μ\| mean | per-dim var mean | frac var > 1 | final ESS (mean / median) |
|---|---|---|---|---|
| legacy (`prior_scale=0`) | 0.900 | 1.449 | **82.2%** | 87.0 / 94.7 |
| posterior (`prior_scale=1`) | 0.424 | 0.493 | **0.0%** | **118.6 / 140.6** |

The fit now lives inside the prior AND the estimator gets better by the project's own D3
gate metric — ESS 87 → 119 out of 200, because a proper posterior is far easier to
importance-sample than a flat likelihood ridge. Tests: `prior_scale=0` is bit-reproducible;
at α=0 (flat likelihood, so the target IS the prior) the legacy path gives per-dim variance
167 with means at |u| = 85 while the fixed path gives 0.79 and 0.43.

**Should the pipeline then USE mixtures? Measured: no, for cube.**
`tools/diag_preimage_posterior_width.py` (new) draws K=32 latents from the fitted posterior,
decodes them all, and reports how far apart they are in u against how faithful their decodes
are in a — with prior draws (0.320) and the point round-trip (1.2e-4) as calibrations.
Sweeping the temperature separates "the decoder has flat directions" from "the width is
just α":

| target | per-dim var | u sd | dist → point | decode err | / action scale | faithful | far AND faithful |
|---|---|---|---|---|---|---|---|
| α=20, no prior (legacy) | 1.474 | 1.081 | 2.80 | (NaN rows) | — | 0.2% | 0.2% |
| α=20, prior | 0.490 | 0.659 | 1.94 | 0.165 | 0.53 | 0.1% | 0.1% |
| α=100, prior | 0.093 | 0.256 | 0.72 | 0.045 | 0.145 | 4.8% | 1.3% |
| α=500, prior | 0.020 | 0.078 | 0.28 | 0.009 | 0.029 | **94.4%** | 6.3% |

Width, distance-to-point and decode error all shrink together as α rises, and samples only
become faithful once the posterior has collapsed ONTO the point inverse. That is a
temperature effect, not a preimage set: the observed widths track 1/(ασ) with σ ≈ 0.07, the
documented median Jacobian singular value. The decoder is a near-uniform CONTRACTION
(σ ≈ 0.07 in every direction), not a many-to-one collapse — which reconciles the two threads
that made mixtures look attractive: coverage is 3× too tight because the map contracts, and
the preimage is nonetheless a point because a contraction can be injective. Decode error
grows smoothly and linearly with distance in u (0.28 → 0.009, 0.72 → 0.045, 1.94 → 0.165;
slope ≈ σ), i.e. there is no direction the decoder cannot see.

Two practical consequences: (i) **keep point preimages** — at the only temperature where the
mixture is faithful (α=500) its sd is 0.078, a negligible blur around the point; (ii) the
historical mixture arm was training on latents whose decodes sat **53% of an action scale**
from the recorded action, so its "in-sample anchor" was not in sample — that arm was
handicapped by the missing prior AND by α=20 being far too soft for a d_a=5 flow.
Caveat: measured on the cube Stage-A flow. σ and therefore this whole conclusion is a
property of the trained flow; pointmaze (d_a=2, the catastrophic legacy fits) and antmaze
(d_a=8, documented ESS failure at α=20) should be re-measured before generalising. The tool
takes minutes per env.

Consequences of the prior fix, none of which touch a current number:
- every result to date uses point preimages, which this bug cannot reach;
- the historical **point-vs-mixture ablation was handicapped** — its mixture arm sampled
  from these runaway fits. That does not reopen the 08-05 Rung-1 root cause (structural,
  settled), but "point beats mixture" was never a fair comparison;
- regenerating the npz files is NOT recommended yet: ~19.5 h per 1M rows at N=200, and
  nothing in flight reads the mixture. Do it when a mixture-mode arm is actually planned;
- `flow_steering` (both modes) now steers through the posterior by default — its two test
  modules stay green (23 passed).

### Not doable from this session
P4's "archive the five fleet audit reports verbatim under `docs/audits/2026-08-14-fleet/`":
those reports only ever existed in the previous session's context and are not on disk
(`docs/audits/` holds just the 08-12 HP diff). They need to be re-pasted or regenerated.

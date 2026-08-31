---
marp: true
title: PSMFlows — Session Handoff
theme: default
paginate: true
---

<!-- _class: lead -->

# PSMFlows — Session Handoff

Current work: **PSMFlow v1** (`docs/plans/2026-07-20-psmflow-v1.md`) — **Tasks 1–8 code
complete**; the pipeline is now operator-driven (GPU runs + gates), with one open method
call on the D3 ESS gate. The **affine PSM** cube push (07-22 → 07-26, below) is **PARKED**,
not closed: the 07-26 entry's Findings 3 and 4 (rectangular measure mesh, split
x-/(s,a)-branch) are still the open leads if we return to it. The older bilinear-PSM parity
hunt (2026-07-07 → 07-15) is **CLOSED** — see the 07-13 entry and `PAPER/RESEARCH_NOTE.md`
§4: no code bug, the gap was seed variance + a training-budget ceiling.

Branch: `feat/inversion-integration` · Machine: `midi-01` (UT CS)
Date: **2026-08-30** (latest) · prior: 2026-08-29, 2026-08-14, 2026-08-13, 2026-08-12, 2026-08-10, 2026-08-05, 08-04, 07-29, 07-28, 07-26, 07-15, 07-13, 07-07

---

<!-- _class: lead -->

## 2026-08-30 — the agent we have been running is FB, not PSM; the preimage is exact for a decoder we never use

Two things came out of reading the ICLR draft against the code, and one port followed.

### The draft describes a method we are not running

`PAPER/ICLR` was re-added this session (skeleton: prewriting form + intro + preliminaries
+ a working-notes method section; `experiments.tex` and `related-work.tex` are empty, the
abstract is template text). Audited claim by claim against the run record:

- **Basis policies.** The draft defines `Pi = {G(s, u_0) | u_0 ~ p_0}` — fix `u_0` and hold
  it. That family was measured non-goal-covering on 08-05 (0/233 fixed-u policies reach the
  pointmaze goal). Everything that works uses per-step latent selection.
- **Preimage sets.** The draft's second contribution is identifying the DISTRIBUTION of
  latents decoding to the same action and augmenting the dataset with it. Against that: the
  width diagnostic on all three environments says the fitted mixture is a blurred point (far
  in `u` AND faithful in `a` peaks at 3.4% pointmaze, 0.7% antmaze). For that claim the cube
  is the open case — the 08-29 sweep found coverage@64 0.990 inside budget with near-flat
  Jacobian directions, which is the opposite reading. The Jacobian probe settles it.
- **No tuned regularizer.** The pitch is that the latent parameterization removes the
  pessimism/BC knobs. In practice the per-task result depends on a tuned residual budget
  (eps=0.05 -> 0.905 +/- 0.020 over 3 seeds; eps=0 -> 0.142 +/- 0.025 over 2) and the
  collapse forensics traced failure to a pessimism spiral.
- **Not in the draft at all:** the hybrid action critic + residual, which carries the only
  limited-coverage evidence we have (10% data: deployed 0.238 vs FB 0.030).

### `psmflow` is FB with a latent action space

`agents/psm.py` — the one pinned to the torch reference by weight transplant
(`tests/test_psm_agent_equiv.py`, 5 passed) — has two measure branches: `proto_loss` learns
the basis against a hash codebook of policies, and `sf_loss` fits the task head on a FROZEN
basis with ONLINE phi in the target ("matching the reference").

`agents/psmflow.py:96 measure_loss` has one branch: `M = psi(s,w,u) @ phi(s')^T`, ortho on
phi, phi trained by that same contrastive term, target from `target_psi` and `target_phi`,
`w` Gaussian mixed with `phi(next_obs[perm])`, `w = E[r phi]`. That is `agents/fb.py:49
_fb_loss_fn` line for line with F->psi, B->phi, a->u. No codebook basis, no affine `b`, no
constrained-LP inference.

This was a deliberate call — `docs/design/2026-07-20-psmflow-v1-design.md:89`: "One branch
only — PSM's proto/sf split collapses: the flow family IS the codebook." That held under
Rung-1, where the fixed-`u` family WAS the codebook. The 08-05 redesign moved policy
identity to `w` and made the bootstrap the actor's latent, which removed the codebook;
nothing re-derived the argument afterwards. The code is reference-faithful — to FB.

### The preimage is exact for a decoder the agent never uses

New measurement, 4096 rows of the cube HPO npz, same Stage-A checkpoint:

| latent | decode distance to `a`, ODE-100 | one-step decoder |
|---|---:|---:|
| stored point preimage | **0.00012** | **0.0886** |
| stored mixture mean | 0.1153 | 0.1495 |
| prior draw N(0, I) | — | 0.2855 |

Mean `||a||` = 0.875. The inversion solves `G_100(s,u) = a` and `precompute_preimages.py`
asserts `inversion.n_initial_steps == agent.flow_steps` to keep it exact — but every psmflow
run sets `gpi_decode: onestep`, so acting and training go through the distilled one-step net.
Under that decoder the "exact" latent misses by 0.0886, ~10% of action scale and comparable
to the mixture width the HPO spent 50 trials tuning (cube winner decode_mix 0.196). Still
3.2x better than a prior draw, so the latent carries real information — but the point arm is
NOT the noise-free control the point-vs-mixture ablation treats it as. Fix: invert at the
one-step decoder, or act at `flow_decode_steps=100`.

Two smaller integration findings:

- `utils/datasets.py:141-161` computes `next_noise_preimage` (u_0') on every batch and NO
  agent reads it — a Rung-1 leftover. In mixture mode it doubles the per-step mixture
  sampling (numpy Cholesky over B rows) on the training path. The `idx+1` pairing rule, its
  "incorrect at the end of a trajectory" warning, and the guard it forced into `main.py`
  all exist for that dead field.
- `preimage_valid` is computed and diverged rows are repaired (mixture -> prior, point -> 0)
  but the mask never excludes them from the loss: a point-arm row reset to `u=0` decodes to
  `G(s,0)` and trains as a legitimate pair. cube 13/1M, antmaze 881/1M.

What is solid: the pairing guards (env, ckpt realpath, epoch, `dataset_fraction` + seed,
`prior_scale > 0` whenever the mixture is read, first/last-1k row content check), `u_clip`
keeping online and target inputs on one support, per-visit mixture resampling, and `infer_z`
identical to the reference-verified `psm.py`.

### Ported: PSM into the latent action space (single latent)

`agents/latent_affine_psm.py` (commit `db96e48`). `LatentAffinePSMAgent(AffinePSMAgent)`
inherits the whole PSM — affine `M(s,u,x) = Phi(s,u,x).w + b(s,u,x)`, `WNet`, the codebook
basis branch, the constrained-LP `infer_w_goal` and closed-form `infer_w_zeroshot`, the
amortized actor — and substitutes the action slot for the flow's latent. Four overrides:
`_slot` (the clipped `noise_preimage`), `_slot_scale` (`u_clip`), `_emit` (decode through
the frozen flow — the only place an action exists), and `_codebook_table` (codebook policies
are latents drawn from the flow's own N(0, I) prior, still keyed on the row hash so `pi_z`
varies with the state and is NOT the fixed-u family).

`agents/affine_psm.py` gained those four seams and routes all nine action-slot reads through
them; its defaults leave behaviour unchanged, and all five affine test modules still pass
(smoke 6, networks 3, inference 5, flow 9, factored 8). New `tests/test_latent_affine_psm.py`
(7 passed) includes the load-bearing one: negating `batch['actions']` leaves the measure loss
bit-identical while shifting `noise_preimage` moves it. End-to-end smoke on the real cube npz
ran 200 steps plus LP inference, actor distillation and eval.

Open on the port: no HP tuning (it inherits affine PSM's raw-action cube values);
`inference.mode` defaults to `full`, i.e. a 5120-step LP at every eval, untested in latent
space against `zero_shot`; and no BC control yet, without which no number is quotable.

### The Jacobian probe refutes the 08-29 structural claim

`tools/diag_flow_jacobian.py` (new): singular spectrum of dG(s, .)/du at each transition's
OWN preimage — the point the inversion solved for, so the numbers describe the region the
mixture is fitted in. 2048 rows per environment, `flow_steps=100`, reports at
`/data-local/amsks/PSMFLows/logs/jacobian_{cube,antmaze}.json`.

| | cube (d_a 5, mean ||a|| 0.87) | antmaze (d_a 8, mean ||a|| 1.99) |
|---|---:|---:|
| sigma pooled, median | 0.083 | 0.201 |
| sigma_min, median | 0.068 | 0.050 |
| sigma_max, median | 0.117 | 0.325 |
| condition number, median | 1.79 | 6.67 |
| free radius in u for a 0.05 action move | 0.73 | 1.00 |
| rows dropped (inverse diverged) | 0 / 2048 | 5 / 2048 |

The 08-29 entry read cube's 0.068 as evidence of near-flat directions and inferred antmaze
must be near-bijective. **Both halves are wrong.** Cube's condition number is 1.79 and its
spectrum runs 0.068-0.117: a near-ISOTROPIC contraction by ~0.1 with no null direction —
the same object the 08-14 width diagnostic described ("a near-uniform contraction sigma
~ 0.07, which is injective"). And antmaze's smallest singular value is SMALLER than cube's
with a LARGER free radius, so locally antmaze has more latent slack per unit of action
change, not less.

The cube/antmaze split therefore is NOT local injectivity, and "cube has a preimage set,
antmaze has a point" should not be written down. Coverage@64 is a global property — where
prior draws land relative to the data — and antmaze differs on two axes this probe cannot
reach: 8 action dims instead of 5, and actions of twice the norm, so a fixed decode budget
is spread thinner. Whatever explains antmaze's 0.115 coverage and ESS 7/128 is still open;
the Jacobian is now excluded, and lead 2 of the 08-29 entry is closed negative.

### In flight

- **Cube mixture arm**, 3 seeds, `psmflow_mixture_hpo_20260829`, tmux `mix_sd0/1/2`: the
  first runs to use the HPO npz with `use_point_preimage=false`. Point arms of the old and
  new npz are bit-identical, so the 5-seed 0.236 +/- 0.071 is an exact control. Prediction on
  record: no better than 0.236, because mixture draws sit 1.38 in `u` from the point inverse.
- **Antmaze precompute**, queued behind them (tmux `pre_antmaze_queued`) at the best trial of
  the 50 (alpha 11.55, prior_scale 0.489, n_steps 5): ~13.5 h. Run to be safe, but its own
  sweep says coverage@64 0.115 and decode_mix 0.263 vs decode_pt 0.0002, against a working
  point-arm number of 0.220 +/- 0.005 (3 seeds, BC control 0.090).

### Also this session

- `PAPER/` untracked from `main`, `feat/psm-integration` and `feat/inversion-integration`
  (tip removal only; history keeps it), then the ICLR skeleton re-added — it builds clean
  with latexmk, 6 pages, one undefined citation (`wagenmaker`).
- `utils/log_utils.py`: wandb 0.29 rejects `start_method` / `_disable_stats` through
  pydantic, so every run died in `setup_wandb` before training. Commit `d634b63`.

---

<!-- _class: lead -->

## 2026-08-29 — inversion HPO: cube solved and precomputed; antmaze structurally closed to width tuning

Branch `feat/inversion-integration` (= main + Claas's `tuning_inversion` merge + HPO work).
Three unpushed commits: `643d4d7` (hypersweeper+SMAC harness), `75fa57d` (typicality hinge
+ per-trial json fix), `7d635a4` (jittered Cholesky for near-singular EM covariances —
required or every antmaze trial crashes).

**What was built.** `tools/hpo_preimage_inversion.py` — single-trial SMAC target derived
from Claas's `tools/tune_preimage_inversion.py`: inverts one fixed neighbourhood batch
(same batch + draw seed across trials), returns
`cost = −coverage@k64 + 10·max(0, decode_mix − budget) + 10·max(0, E‖u‖²/d_a − 1.1)`.
Budget rule: 0.1 per action dim (cube 0.22, antmaze 0.28), config keys
`hpo_decode_budget` / `hpo_typicality_ratio`. Sweeper config `configs/hpo_preimage.yaml`
(SMAC BlackBox BO, 50 trials; alpha log [2,100], prior_scale [0.3,1.0], n_steps [5,20];
num_samples pinned 128, num_clusters 1). ~65 s/trial cube, ~127 s antmaze.
The typicality hinge exists because the first cube sweep's incumbents cheated: prior_scale
0.30 bought coverage 0.999 at E‖u‖² ≈ 8 vs expected 5 (mixture leaves the prior).

**Cube result (sweep dir `/data-local/amsks/PSMFLows/hpo/cube_20260828_000356`, log
`cube_launch.log`, one JSON line per trial).** Feasible region comfortably non-empty:
43/50 trials inside budget, 17 also typical. Winner after typicality filter:
**alpha 14.1, prior_scale 0.69, n_steps 14** — coverage@64 0.990, decode 0.196,
E‖u‖² 5.06, ESS 58/128. Beats the hand grid (0.96 @ 0.227, over budget).
Full 1M-row precompute at exactly that setting is **DONE**:
`/var/local/amsks/exp/PSMFLows/preimages_cube_hpo_a14p1_ps0p69_ns14_N128.npz` (+ meta),
13/1M diverged rows reset to prior (flagged in `preimage_valid`).
**Next pending step: recovery tests on that npz** (`tools/validate_decode_recovery.py`,
`tools/validate_dynamics_recovery.py`, or `scripts/run_recovery_tests.sh`), then it can
replace the shipped cube npz for mixture-mode (`use_point_preimage=false`) runs.

**Antmaze result (sweep dir `.../antmaze_20260828_020557`, log `antmaze_launch.log`) —
the important one.** Ckpt: only **antmaze-medium** exists
(`bcflow_antmaze-medium-navigate_20260805_014546/sd000` @ 500000; no -large ckpt),
env `antmaze-medium-navigate-singletask-v0`, d_a=8. Over 50 BO trials:
best coverage inside budget **0.115**; best coverage of ANY trial, budget ignored,
**0.24**; ESS 6–7/128 across every setting. Conclusion, now with search-based evidence
rather than a grid: **no width setting makes the antmaze mixture usable — the conflict is
structural.** The cube flow has near-flat Jacobian directions (median singular value
0.068), so a genuine wide preimage region exists and tuning finds its width; the antmaze
flow is near-bijective, the preimage is a point, and widening it is label noise by
definition. **[SUPERSEDED 2026-08-30: the Jacobian probe refutes this. Cube's 0.068 is its
sigma_MIN; the spectrum is 0.068-0.117, condition number 1.79, i.e. an isotropic contraction
with no flat direction, and antmaze's sigma_min is SMALLER at 0.050. The conclusion that no
width setting works on antmaze stands on the sweep; the structural explanation offered for
it does not.]** This refutes the "re-run with a re-tuned alpha" hope in
`scripts/upload_preimages_hf.py` and explains why antmaze runs use
`use_point_preimage=true` (mixture is the default elsewhere — `psmflow.yaml:95`).

**Open leads, in order.**
1. Recovery tests on the new cube npz (above) — blocks adopting it.
2. Jacobian probe: singular spectrum of dG/du at data points, antmaze vs cube. Cube's
   0.068 is on record; if antmaze's is ~10× larger, the structural claim is quantified.
3. The actual fix direction: retrain the antmaze flow so preimage SETS exist — extra
   latent dims beyond d_a, or a decoder noise floor / entropy regularizer — then re-run
   the same sweep. Tuning the inversion harder is ruled out; don't respend there.
4. Housekeeping: stray `hpo_trial.json` + `smac3_output/` in repo root are pre-fix sweep
   droppings (safe to delete / gitignore); the three commits above are unpushed.

---

<!-- _class: lead -->

## 2026-08-14 — the residual is DATA-DEPENDENT: it carries the low-data regime and costs on full data

Full plan, every launch and every number: `docs/plans/2026-08-14-priority-stack.md`.
Tables regenerate from the eval JSONs with `tools/make_tables.py`
(`docs/tables/results.md`, `PAPER/ICLR/tables/*.tex`).

### The result that reorganises the story

Every hybrid checkpoint is now evaluated in BOTH acting modes — deployed (actor draw +
ε-residual) and decode-only (`agent.action_critic.eval_rank_k=1`: same draw, no residual,
no selection). The residual's sign flips with dataset size:

| data | deployed | decode-only | plain PSMFlow | residual effect |
|---|---|---|---|---|
| 10% | **0.238** | 0.072 | 0.068 | **+0.166** |
| 50% | **0.228** | 0.070 | 0.052 | **+0.158** |
| 100% | 0.162 ± 0.168 (4 seeds) | 0.226 ± 0.068 | 0.236 ± 0.071 | **−0.065** |

At 100% the residual helps on one seed of four (+0.102 on sd1) and hurts on the rest
(−0.218, −0.126, −0.018), so the flagship 0.302 was the seed where the coin landed heads;
the deployed arm's ±0.168 interval is the real headline. At 10–50% the opposite holds and
the residual carries everything — decode-only sits exactly on the BC floor. At 10% the
hybrid is also the best zero-shot method measured (0.238) while **FB collapses to 0.030**,
below the BC control. One seed per fraction cell: replicate before claiming.

Also: λ-rank acting (K=32 decoded candidates ranked by Q_a, no residual) is WORSE than
decode-only on both seeds (0.004 / 0.162), so Q_a's ordering over reachable actions is not
merely uninformative. 1M did not pay off (fresh 1M seeds 0.134/0.162; extending sd1
500k→1M moved 0.302→0.284). HP-matching to FB buys nothing (0.276 / 0.192 vs 0.236 ± 0.071).

### Two paper-bound claims failed their re-check

1. **The radius claim was vacuous** — `clip(N(0,1), ±r)` never samples wider latents. With
   scaled draws, coverage rises 0.63 → 0.99 → 1.41 at 1/1.5/2×, i.e. it DOES reach the
   split-half reference, while the distance to FQL's action gets WORSE (0.19 → 0.23 → 0.28).
   `note.tex` corrected in both places (body + figure caption).
2. **C1's "FQL is off-support" had no yardstick.** Measured against the data-matches-itself
   baseline it was missing: a_FQL sits 0.537 from its k-NN actions while real data sits
   0.582 from its own (p95 1.165), and only 6.3% of FQL's actions exceed that p95. FQL is as
   data-like as the data; what is anomalous is our decode at 0.187, ~3× TIGHTER. The
   capacity/retrain arm was cancelled on the uncalibrated comparison. Verdict string left
   unchanged in the tool, baseline now recorded beside it — this one still needs a call.

### The preimage pipeline had a real bug (user-flagged, confirmed)

The inversion target was π(u) ∝ exp(−α‖G(s,u)−a‖) with **no N(0,I) prior factor**, so it is
flat wherever the decoder is insensitive and the EM fit runs away (covariance eigenvalue
1→6→34→305→2281→6095 over 8 steps; in the shipped npz files 82–84% of cube rows and 50% of
pointmaze rows are fitted WIDER than the prior, worst case 3.8e4, means to |μ|=206). The
exact point preimages in the same files are prior-like (|u| mean 0.85), which identifies the
target rather than the inverter. Fixed via `inversion.prior_scale` (default 1.0; 0.0
reproduces legacy files) — on the real cube flow this puts 0% of rows outside the prior and
lifts ESS 87 → 119 of 200.

**But mixtures still are not worth switching to.** Sweeping α (`tools/diag_preimage_posterior_width.py`)
shows the posterior's width IS the temperature, not decoder degeneracy: width, distance to
the point inverse and decode error all shrink together (α=20/100/500 → per-dim var
0.49/0.093/0.020, decode error 0.165/0.045/0.009), and samples only become faithful once the
fit has collapsed onto the point inverse. The decoder is a near-uniform CONTRACTION
(σ ≈ 0.07), which is injective — hence a point preimage — while still compressing the
reachable action set 3× below the data's dispersion. Historical mixture arms trained on
latents decoding 53% of an action scale away, so that ablation was never fair.

### Everything else that landed

- eval reproducibility: `evaluate` pinned the env's init RNG but drew ACTION noise from OS
  entropy regardless of `seed`; now pinned, relabel batch seeded, false docstring fixed.
- one k-NN protocol (`utils/geometry.NeighbourIndex`, standardized obs, k=32, exact
  self-exclusion) shared by the three support probes; raw-geometry numbers kept, labeled.
- ψ_a pessimism is now scalar-Q (blend toward the least-task-valued ensemble member);
  the old per-feature penalty was sign-indefinite in Q-space and RAISED Q where w < 0.
  Bit-identical at λ=0, i.e. every run to date.
- 2a Q-gap probe + calibration penalty fix (actor knob, |q0−q1| not |q0−q1|/2): prior
  collapse verdicts UNDERSTATED the over-estimation; directions unchanged.
- pairing guard now records `dataset_fraction`/seed in the npz sidecar and spot-checks the
  first/last 1k observation rows; `restore_agent` tolerates checkpoints predating a field
  (the hpmatch checkpoints could not otherwise be loaded at all).
- eval reports are self-identifying (`acting_mode`, `action_critic`, fraction, flow/npz
  paths): three acting modes per checkpoint were distinguishable only by FILENAME.
- **P2 FB-graft FAILED its gate (sd1, 1M): deployed 0.064 [0.046, 0.089] against a 0.45
  gate — at the BC control, below the hybrid.** ψ_a got its own backward map B_a trained by
  the FB measure loss and its own w_a inferred from B_a; the residual got WORSE
  (decode-only 0.174 → deployed 0.064). Its decode-only number sits inside the hybrid's own
  decode-only spread, confirming the graft left the latent actor alone as designed — so the
  implementation did what it claimed and the claim did not help. Pre-registered rule
  applied: shared-φ was NOT the explanation for the weak action critic; report, do not
  iterate. sd0 still running only to give the negative result n=2.
- **Run the test suite module-per-process** — all 29 modules pass (177 passed, 1 skipped),
  but the whole suite in ONE process dies in the XLA CPU compiler at a moving victim
  (pre-existing; reproduced with every one of today's tests excluded).

---

<!-- _class: lead -->

## 2026-08-13 — the residual dial WORKS (0.90–0.91 at 500 ep, 3 seeds); the collapse is a PESSIMISM spiral, not optimism

The W4/l1stab arc, verified end-to-end this session. All artifacts in
`/data-local/amsks/PSMFLows/logs/`.

### Peak checkpoints hold at 500 episodes

`eval500_l1stab_*_peak*.json`, summary `eval500_l1stab_peaks_summary.json`:
ε=0.05 → **0.910 / 0.910 / 0.896** (sd2@75k, sd3@50k, sd5@50k); ε=0.1 → 0.888 / 0.850.
Anchors: ε=0 final 0.140/0.144 · BC 0.068 · FB 0.716–0.730 · FQL 0.949. A 5%-of-action-
scale residual budget recovers ~95% of the per-task ceiling. The w4 finals table
(`eval500_w4_res*_final.json`): ε=0.1 sd0 survivor **0.820** [0.784, 0.851]; everything
else collapsed (0.000–0.144).

### The collapse mechanism is pessimism-driven UNDERestimation

`diag_calibration_collapse.json` (new `tools/diag_fql_calibration.py`), same-run
peak-vs-collapsed pairs (res0.05 sd2 75k/250k, sd5 50k/300k): at the peak Q is mildly
optimistic (bias **+6.6/+8.8**, Spearman 0.14–0.25); at collapse Q has crashed −49 →
−105/−112 while realized return only fell −56 → −87 — bias **−18.7/−25.0**. Exact-min
pessimism (0.5 backup + actor) compounds as ensemble disagreement grows off-data. The
in-flight `l1_pess0_e010` wave (pessimism 0.0, ε=0.1, seeds 2–4) is the mechanism test.

### The winning policies barely leave the data (regime verdict)

`dist_res*.json` (new `tools/diag_action_distance.py`): executed-action distance to
k=32 NN dataset actions, vs the data-matching-itself baseline (median 0.520, p95 1.036).
ε=0.05 peak policies: median **0.504–0.509** — AT data-noise distance — with only
10–12% of steps beyond the baseline p95. ε=0.1: median 0.710, 23% beyond. Pre-registered
reading: improvement arrives while actions stay behavior-close ⇒ **decoder undercoverage
(Hypothesis A) was the binding cost**; pushing further off-data buys nothing more. NB
this also recalibrates C1: real data actions sit ~0.52 from their own neighbours, so
FQL's 0.577 is near data-noise level under the matched statistic — the "off the data"
framing in note.tex needs this baseline context.

### Also this session

- **Inversion re-cert PASSES** (`audit_inversion_recert.json`): ESS 81.3/94.6/7.6,
  roundtrip 1e-4/1e-4/2e-4, invalid 13/0/881 — all matching the HF card.
- Peer bar: PSM cube 3-seed 500-ep spread **0.056 / 0.156 / 0.532** — quote the spread,
  never a mean (n=3 CI is wider than the range).
- Audit 0a/0b (commit `1df9594`): FB run-config clean, 0.72 stands; HP shortlist top
  suspect is pessimism 0.5-vs-0.0 — now directly implicated by the collapse forensics.
- T1/T2 flow tests (`diag_preimage_sampling_*.json`, `diag_generated_pair_support_*.json`)
  — see those JSONs for the mixture-arm decode fidelity and the generated-pair support
  verdicts.
- New tools: `diag_fql_calibration.py`, `diag_action_distance.py`,
  `diag_preimage_sampling_fidelity.py`, `diag_generated_pair_support.py`.

NEXT: read pess0 (does removing pessimism stop the collapse?); if yes, the stabilized
recipe (ε=0.05, pessimism tuned down, or early-stopping on the calibration bias signal)
is the port target for zero-shot via the w-conditioned action critic (Idea 1).

---

<!-- _class: lead -->

## 2026-08-12 (L0) — the ordering survives, but it is 1.7x and not 3x

`logs/diag_action_coverage_cube_robust.json`, `tools/diag_geometry_robustness.py`.

C1 compared a minimum over **512** decodes against a minimum over **32** neighbours. That
is a combinatorial advantage, not a geometric one. Matching the candidate counts:

| | k=16 | k=32 | k=64 |
|---|---:|---:|---:|
| d_flow, 512 candidates (C1's number) | 0.183 | 0.183 | 0.183 |
| d_flow, matched to k | 0.370 | **0.345** | 0.296 |
| d_data over k neighbours | 0.649 | **0.577** | 0.521 |
| ratio (matched) | 1.75 | **1.67** | 1.76 |

**The ordering survives in all six geometry x k settings (raw and per-dim standardized
observations), ratio 1.64-1.80 — but the magnitude roughly halves, from ~3x to ~1.7x.**
Quote the matched pair (0.577 vs 0.345 at k=32), never C1's 512-vs-32 pair.

The conditioning asymmetry that matching CANNOT remove: d_flow conditions on exactly s,
d_data on a neighbourhood of s. So the defensible claim is "the decoder interpolates at
least as close to the task-optimal action as nearby data does", not a statement about the
behaviour conditional at s.

Everything downstream of C1 stands: the winning policy is still off-support, the capacity
arm stays cancelled, W3 stays retired. Only the number changes.

Two smaller reads: coverage is mildly k-dependent (0.38-0.48) and its split-half reference
moves with it (0.92-1.11), so coverage must always be quoted relative to that reference.
And the coverage statistic is numerically unstable at k=16 under standardized observations
(one cell read 364 — an action dim with near-zero sd in a 16-neighbour set); use k>=32 for
coverage, the distance statistics are unaffected.

note.tex updated: the abstract's "3x" is now "1.7x", Hypothesis B carries the matched
numbers and the residual asymmetry, and the provisional-geometry caveat is resolved.

---

<!-- _class: lead -->

## 2026-08-12 — the interface ceiling is a SUPPORT fact, not a decoder fact

Chain: P0 Branch B → differentiation probe → P2 ceiling → W1 coverage → W3 gate failure →
C1. Reports in `logs/diag_action_coverage_cube{,_fs100,_c123}.json`.

### C1 — the reframe. Read this before quoting any interface number.

| cube, 64 states, normalized by action scale | |
|---|---:|
| a_FQL → the dataset's OWN 32-NN actions | **0.577** |
| a_FQL → our best decode (512 candidates) | **0.187** |

**FQL's actions sit 0.577 from the dataset's own nearest neighbours while our decoder gets
within 0.187 of them — the flow interpolates closer to the winning actions than the
empirical data itself does.** Therefore 0.187 was never a decoder failure, and no retrain
could have closed it: the actions FQL uses are not in the distribution being cloned.
Capacity/retrain arm **cancelled** (automatically, by the pre-committed rule); **W3
retired**.

This is easy to garble later, so state it in this order: (1) the winning policy is
off-support, (2) by 3x more than our decode gap, (3) hence the ceiling is a property of
the data + support constraint, not of the flow.

### W1 / W3 / C2 / C3, briefly

- **W1:** coverage 0.415–0.418 across all six decode × radius settings on the deployed
  flow. ODE gain +0.002, radius gain +0.001 — doubling the latent radius changes nothing,
  so the decoder saturates rather than being clipped too tightly. W2 arms A and B both
  eliminated without training either.
- **W3:** retraining Stage-A at `flow_steps=100` lifted coverage 0.42 → **0.63** — the
  07-29 caveat about the fs10 checkpoint was correct — but left the FQL-action distance at
  0.187, unmoved to three decimals. Gate (<0.10) failed; the flow was NOT inverted.
- **C2:** residual spreads evenly across all 5 action dims (max 23%) ⇒ blanket ε.
- **C3:** aleatoric coverage ceiling 0.936, so fs100 is at **67% of achievable**, fs10 at
  44%. Coverage was being judged against the wrong reference (1.0).
- Tool debt closed: the ODE step count now reads the flow's `flags.json` (fixing the
  `gpi_decode=ode` step mismatch for all envs), and the knob-sweep verdict is scoped to
  the flow under test — the fs100 report had reprinted the fs10 recommendation verbatim.

### Peer bar, 500 episodes — PSM does NOT survive its in-loop numbers

PSM cube in-loop finals were 0.38 / 0.02 / 0.58; at 500 episodes sd0 reads **0.156**
[0.127, 0.190] and sd1 **0.056** [0.039, 0.080] (sd2 pending). Do not quote the in-loop
numbers. FB at **0.716–0.730** remains the real peer bar; BC control 0.068.

### In flight

W4 residual sweep on the instrument: ε ∈ {0, 0.05, 0.1, 0.2, 0.4}, 2 seeds each, tmux
`w4_q0` / `w4_q1`. The W4 critic scores executed ACTIONS, not latents — a residual head
gets no gradient from a latent critic — so ε=0 is its own anchor arm and the P2 numbers
are NOT its left endpoint. Gate: any arm ≥ 0.40 at 500 episodes ⇒ port to LatentFlowPSM.
Per-arm off-support budget (executed-action distance to k-NN dataset actions) is the
x-axis of the headline tradeoff figure, not a side stat.

---

<!-- _class: lead -->

## 2026-08-10 (evening) — P0 says BRANCH B: FB's basis is WORSE than ours, and it still wins

`docs/plans/2026-08-10-psm-fix-roadmap.md` P0, the deciding probe. Report:
`logs/p0_fb_basis_probe.json`. New tool `tools/diag_fb_basis_probe.py`.

| cube, 500k | linear reward read-out (ridge R^2, held out) | Q relief (spread / |Q|) | 500-ep success |
|---|---:|---:|---:|
| LatentFlowPSM (our phi) | **0.129** | 0.011 | 0.236 |
| FB (3 seeds) | **0.069** | 0.023-0.031 | **0.716 / 0.730 / 0.716** |

**FB's backward map carries LESS reward signal than our phi -- at z_dim=50 against our 128,
so it is worse per-dimension too -- and FB still scores 3x our success.** The premise
behind P1 (copy FB's measure objective so the basis carries the task) is therefore not
supported: the property we were going to import is one FB does not have either.

FB's Q relief is 2-3x ours (2.3-3.1% vs 1.1%) but nowhere near the >=5% Branch-A bar, so
that is not the explanation on its own either.

**Consequence, per the plan's pre-registered decision rule: P1a (the loss swap) is NOT
indicated; LatentFB (P3) is promoted to the primary path, and the PSM-internal work
narrows to P1b (backup exploration + actor unshackling).** The differentiator has to be
the improvement loop -- FB's actor optimizes F^T z from step one against a critic trained
on ITS OWN visitation -- not the reward read-out of the basis.

Caveat worth carrying into the paper: R^2 of a linear read-out on a 2%-sparse reward is a
weak instrument in absolute terms for both methods. The comparison is meaningful because
it is like-for-like; the absolute levels are not evidence that either basis is "good".

### Peer baselines, 500 episodes

| cube-single | success | 95% CI |
|---|---:|---|
| BC control | 0.068 | [0.049, 0.093] |
| LatentFlowPSM (5 seeds pooled) | 0.236 | [0.219, 0.253] |
| **FB sd0/1/2** | **0.716 / 0.730 / 0.716** | [0.675,0.754] / [0.689,0.767] / [0.675,0.754] |
| FQL per-task | 0.949 | [0.936, 0.960] |

FB is a ZERO-SHOT peer, not a per-task topline: same env, dataset, budget, flow actor, no
reward at train time. This is the bar, and we are far from it. Antmaze LatentFlowPSM is now
3 seeds (0.22 / 0.26 / 0.22 in-loop @500k) vs its 0.090 control.

### In flight

- `abl_be_sd{0,1}`: backup exploration `backup_explore_frac=0.5`, cube, 2 seeds (P1b).
  Implemented config-gated with a test pinning that the default 0.0 path is BIT-IDENTICAL
  (the extra RNG keys are split inside the branch, so published runs stay reproducible).
- PSM cube sd1 finishing, sd2 queued; 500-ep PSM evals after.

---

<!-- _class: lead -->

## 2026-08-10 (later) — diagnosis: the gain is in-support latent BC + pessimism, not value improvement

Ran §B of `docs/plans/2026-08-10-critic-diagnosis-and-baselines.md`. Three diagnostics,
three verdicts, and they agree. Reports: `logs/d1_policy_ranking.json`,
`logs/d2_task_projection_{cube,antmaze}.json`, `logs/d3_q_landscape_cube.json`.

### D1 — H0 REJECTED. The critic cannot rank policies either.

Ten cube policies spanning 0.068–0.320 measured success (5 final seeds, sd2 at
50k/150k/250k/400k, and the BC prior), all scored by ONE frozen representation (sd0) at 64
seeded start states. **Spearman 0.10 (permutation p = 0.78).** Predicted values span
−1816 to −1832 — a **0.9% spread** — while realised success varies 4.7×. So panel (c) of F4
was not a confounded diagnostic: the critic is uninformative at the ranking job too.

### D2 — H1 RELOCATED. Inference is fine; the basis is weak.

Closed-form w = E[r·φ] achieves R² **0.129** on held-out cube transitions against a ridge
topline of **0.127** on the same features — the estimator extracts everything φ contains,
so task inference is NOT lossy. But the topline itself is low: the best linear read-out of
φ explains ~13% of reward variance. Antmaze is the same (0.135 / 0.134), so **no
cube-vs-antmaze asymmetry** — H5 (horizon myopia) gets no support from this. w_inf is
stable across disjoint relabel batches (min pairwise cosine 0.93). Fix direction, if this
is pursued: a φ-grounding auxiliary, not a better estimator.

### D3 — H2 SUPPORTED, decisively. H3 not supported.

At 64 states, Q over 512 prior latent draws has relative spread **0.011 of |Q|** — flat.
Around the actor's own latent it is flatter still (0.0038). Two details that settle the
story:
- **The actor sits at the 44th percentile of the prior-Q distribution** (median 38th). If
  Q meant anything the actor would be near the top; it is below the middle.
- **Actor gradient is BC-dominated 5:1**: ‖∇q-term‖ 0.168 vs ‖∇distill‖ 0.839.
  (bc_flow reads 0.0 w.r.t. actor params by construction — it only touches the vf.)
- **Dispersion does NOT collapse** over training (‖u‖² 4.66 → 4.54 across 50k–500k), so
  H3's co-collapse signature is absent. The flatness is not a shrinking-support artifact.

### What this means

The honest reading: **LatentFlowPSM's 3.5× over BC is explained by in-support latent
behaviour cloning plus ensemble pessimism, not by value-driven improvement.** Every piece
fits — flat curves after 50k, chance-level calibration, GPI *below* the BC control (argmax
over a flat noisy Q is worse than sampling), and an actor whose gradient is 5:1 imitation.

Per the plan's decision tree §E this is the "flat-Q" branch, and its own C-tier gating says
bc_coeff↓ runs only if "D3 shows bc-term dominance **AND** Q has relief". Q has no relief
(1.1%), so **the bc_coeff sweep is not indicated** — do not run it reflexively. The
remaining candidate is backup exploration (mix p0 latents into the bootstrap), which
targets the flatness itself rather than the actor's weighting.

### In flight

Baselines from §D, three tmux queues (`base_q0/1/3`, ~2.5–4 h): PSM cube ×3 seeds
(`psm_cube_flow_20260810`), FB cube ×3 (`fb_cube_ortho1000_20260810`, ortho_coef=1000 —
the reference value, repo default is 1.0), antmaze LatentFlowPSM seeds 1,2. R3 is the one
that matters for framing: if action-space PSM matches 0.236, the headline narrows to the
support/coverage story.

---

<!-- _class: lead -->

## 2026-08-10 session — LatentFlowPSM measured end to end; ICLR figures F1-F4

Executed `docs/plans/2026-08-10-iclr-figures.md`. Everything below is a 500-episode
evaluation with a Wilson interval, quoted beside the per-step-prior BC control from the
same frozen flow. Reports in `/data-local/amsks/PSMFLows/logs/eval500_*.json`; every
figure has a sidecar JSON in `PAPER/ICLR/figures/data/`.

### Headline: the actor works, the value function still does not rank

| cube-single, 500 ep | success | Wilson 95% |
|---|---:|---|
| BC control (frozen flow, per-step prior) | 0.068 | [0.049, 0.093] |
| flow-GPI over fixed u (Rung 1) | 0.015 | 50-ep in-loop |
| LatentFlowPSM `acting=gpi` | 0.055 | [0.047, 0.064] |
| **LatentFlowPSM `acting=actor`** | **0.236** | [0.219, 0.253] |
| FQL per-task (alpha=300) | 0.949 | [0.936, 0.960] |

Across 5 seeds the actor is **0.236 ± 0.071** (mean ± 95% CI); per-seed 0.318 / 0.236 /
0.176 / 0.260 / 0.188. Antmaze: LatentFlowPSM **0.222** [0.188, 0.261] vs BC **0.090**
[0.068, 0.118]. Pointmaze BC control is **0.002** — the env is near-unreachable from the
prior, which is why pointmaze zeros never discriminated between methods.

**Three findings, in order of how much they should change what we do next.**

1. **The F1 prediction landed to three decimals.** Fixed-u flow-GPI scores 0.015 against a
   measured family ceiling of 2/128 = 0.0156. GPI was extracting exactly what the family
   contained — the 08-05 root cause is now quantitative, not argued.
2. **`acting=gpi` (0.055) is BELOW the BC control (0.068)**, non-overlapping intervals, on
   the *same* representations the actor uses. So psi is a usable training signal for the
   actor but not a reliable ranker of latents. The Finding-3 weakness survived the
   redesign; it moved rather than disappeared. This is the open problem now.
3. **FQL reaches 0.949 with alpha=300** — the FQL reference's per-env value; the repo
   default of 10 would have understated the topline badly. LatentFlowPSM recovers ~25% of
   per-task performance, and its curve is flat from 50k, so it is converged, not
   budget-limited.

### What was run

- **WP0**, 17 evals: cube sd0-4 x {actor, gpi}, antmaze sd0, BC controls for all three
  envs. `scripts/eval500.sh` wraps the invocation (psmflow and `bc` modes).
- **WP1**, cube FQL 3 seeds, `fqlbaseline_cube_a300_20260810`. **34 min per seed**, not the
  3-4 h the plan budgeted — the tqdm ETA during JIT warmup is what mislead the estimate.
- **D1 re-runs**, 6 (3 envs x trained/random). Pointmaze reproduced the historical numbers
  exactly (MMD 3.0e-4 vs 4.0e-3).
- **Figures** `tools/fig_{reachability,flow_fit,actor_comparison,policy_anatomy}.py` +
  `tools/figstyle.py` (Okabe-Ito order, run through a colorblind validator).
  `tools/viz_policy_rollouts.py` records per-step (state, actor latent, action, reward)
  and the initial-state predicted score.

### Corrections to the plan, from the data

- Antmaze family ceiling is **5/64 = 8%**, not the ~4% the plan cites. Figures compute it
  from the JSON rather than quoting prose.
- The Rung-1 GPI bar **cannot** get a 500-episode rerun: that checkpoint predates the
  latent-PSM redesign, so its parameter tree will not load in the current agent. F3 uses
  its in-loop 50-episode late window and labels the bar as such.
- Calibration needs **AUC against binary success**, not Spearman on returns: every failed
  episode returns the identical value, so most of the sample is one tie group and
  arbitrary tie-breaking manufactures a correlation out of array order.

### Finding 5: the measure does not rank, measured a second way

`tools/viz_policy_rollouts.py`, 60 episodes x 3 cube seeds + 40 antmaze:
- **Support claim holds, with room to spare.** Actor latents mean ||u||^2 = 3.87 (cube,
  d_a=5) and 5.64 (antmaze, d_a=8) — narrower than the prior AND narrower than the dataset
  latents (5.59 / 8.49), 0% on the clip boundary. Every executed action is a decode of a
  typical latent.
- **The value does not predict the outcome.** Predicted psi(s,w,u)^T w at s0 separates
  success from failure with AUC **0.588 [0.480, 0.694]** on cube (180 ep, 33 successes) and
  **0.354 [0.168, 0.567]** on antmaze. Both cover chance. Bootstrap CI, seeded.

So the GPI-below-BC result is not an artifact of the argmax: the critic genuinely carries
little outcome information. The unexplained part is why DPG on it still produces a policy
3.5x better than its behavior prior. That is the next thing to isolate.

### Open

- Why does GPI underperform the prior? If psi cannot rank latents at a state, the
  representation is not doing the job the method claims for it, and the actor is
  succeeding for a reason we have not isolated.
- Antmaze is 1 seed. Cube seed spread (0.176-0.318) is wide enough that 5 seeds pins
  "beats BC" but not the level.
- Antmaze mixture-arm preimages remain unusable (ESS 7.6, 6% > 20); alpha was tuned at
  d_a=5. Point arm is what every run uses.

---

<!-- _class: lead -->

## 2026-08-05 session — ROOT CAUSE: the fixed-u family is structurally non-goal-covering; Rung-1 is dead on navigate data

All Stage-C variants (pointpre x2, mixture, pointpre1M, mix0 audit — 11 runs) read **0.0
success** on pointmaze task1; cube Stage-C floors at 0.02–0.06. D1–D3 pass. This session
root-caused it. **It is not a bug anywhere — the policy family itself has no goal-reaching
member, so even a perfect psi gives GPI a flat landscape.** Three measurements:

### 1. Exhaustive reachability: 0 of 233 latents ever reach the goal

`tools/latent_reachability.py` (new). d_a=2 ⇒ a 13x13 grid covers the ENTIRE box
[-3,3]^2 — no sampling escape — plus 32 dataset preimages + 32 preimages of actual
goal-reaching transitions, each rolled 2 full 1000-step episodes (kills the 200-step
confound in `calibration_check`). **num_success_any = 0.** 75% of latents never get
closer than ~23.5 (maze is ~30 across); best is 1.32 at the saturated corner u≈[2,2.5].
Goal-transition preimages do NO better than random ones — a preimage decodes to its
action only in its own state. Report:
`/data-local/amsks/PSMFLows/logs/latent_reachability_pointmaze.json` + `latent_reachability.png`.

### 2. The expert's route is latent WHITE NOISE — no fixed u encodes a route

From the npz alone: within-episode preimage variance / marginal variance = **0.99**
(1.0 = zero episode-level identity); lag-1 autocorr 0.27, ~0 by lag 50; goal-reaching
segments identical (0.987). The BC flow factorizes behavior as (state → conditional,
u → quantile); the expert's direction choice is driven by a goal that is NOT in the
observation (obs = xy only), so that variance is forced into u independently each step.
Routes exist only as latent *sequences*; Rung-1 assumed u is a persistent policy index,
but Stage A/B construct it as per-step noise. **D2's "pass" measured coherence+diversity,
not usefulness — it could not see this.**

### 3. What the family actually contains: orbiters and constant headings

`tools/viz_fixed_u_field.py` (new): quiver of a = G(xy, u) over the maze per fixed u +
rollout overlay (`fixed_u_fields.png`). Two degenerate regimes and nothing else:
- **Typical u** (prior bulk, e.g. [0,0]): state-dependent field that follows corridors
  but with per-cell arbitrary direction choices → circulation → the rollout ORBITS near
  start (path length 164, net displacement 1.4). heading circ_var 0.83.
- **Saturated u** (box corners): tanh saturation ⇒ near-constant heading everywhere
  (circ_var 0.01–0.03) → wall-sliding diagonal marches. The up-right one gets to 1.32
  from the goal and slides past — it cannot turn, by construction.
Per-cell coverage is FINE: angle circ_var across 512 draws at fixed cells = 0.45–0.78 —
every direction stays available per state. The deficiency is purely temporal.

### Why Stage C then behaves exactly as observed

`agents/psmflow.py` TD target (line ~79) bootstraps `psi(next_obs, u_idx, u_idx)` — the
continuation latent IS the index, faithful to the fixed-u semantics. Since no pi_u
reaches the goal, true V^{pi_u}(task) is the same "never" value for every u → psi^T w is
flat (Q spread ~10% of mean, `viz_latent_value.png`), gpi argmax is noise and drifts to
the saturated corners (the only occupancy-distinct members), calibration reads
predicted 85–212 / realized all-0. Every symptom follows from the one cause.

### Where this leaves the method (decision needed)

The paper's support constraint (every considered action is a flow decode, C=1) survives;
the "fixed noise = policy index" premise does not. Candidate directions, in rough order:
1. **Latent-space PSM (Rung-3 completed properly):** keep the frozen flow; make the
   index a task vector w conditioning a per-step latent actor u(s,w) (already built,
   08-04), and change the psi backup to bootstrap the ACTOR's latent at s'
   (psi(s', ·, u(s',w))) instead of u_idx — policy improvement in latent space, still
   in-support, per-cell coverage measured sufficient. Aligns with in-sample TD Prop 7.2.
2. **Trajectory-level latent in Stage A** (skill-VAE/OPAL-style G(s,u;w_traj), invert w
   per trajectory) — makes fixed-w routes exist by construction; biggest retool.
3. Report as a negative-result finding for flow-noise-indexed families + pivot envs
   where behavior modes ARE single-step-consistent (cube floor says probably not).

### In flight / artifacts

- tmux `audit_psm_pm` (PSM baseline, pointmaze task1): 0.0 through 200k, ETA ~5h — the
  control for whether an RL-optimized family navigates pointmaze at all.
- tmux `audit_psmflow_mix0`: finished flat 0.0 (as the root cause predicts).
- New tools: `tools/latent_reachability.py`, `tools/viz_fixed_u_field.py` (uncommitted).

---

<!-- _class: lead -->

## 2026-08-04 session — full audit + roadmap; the D3 ESS statistic was wrong (again), and it changes the story back

Deliverable: **`docs/plans/2026-08-04-status-roadmap-audit.md`** — status, OGBench roadmap
(gates, baselines, success criteria, viz/analysis scripts), full code audit, rewrite plan.
Read that first; this slide records only what changed on disk.

### The D3 gate was averaging ESS over the whole EM trace, not the final iterate

`compute_full_proposal_distribution_em` returns `ess` shaped `(n_steps,)` per row (one per
EM iteration); after vmap the D3 tool took `mean(ess)` over BOTH axes. Stage B stores — and
`precompute_preimages.py` persists, and the intuition figure plots — only `ess[:, -1]`, and
ESS improves across EM iterations, so **every number the tool printed understated the
stored posterior**. Fixed: the gate now uses final-step ESS and also prints
`mean_ess_trace` for comparability with pre-08-04 printouts.

**Re-measured (cube Stage-A ckpt, 256 rows, seed 0, alpha=20, N=100): final-step mean ESS
21.7, trace-mean 17.7 — the gate PASSES at N=100.** So the 08-03 working-tree conclusion
"no alpha reaches 20; num_samples is what clears the gate" was an artifact of the trace
statistic; alpha=20 was the right call and N buys *margin* (linear), not the pass.
`configs/inversion/default.yaml`'s comment block now carries the correction. The in-flight
N=200 recomputes (`watch_cube` / `watch_pointmaze` tmux, ETA ~12:30 08-04) are NOT wasted —
they give comfortable headroom over a marginal 21.7 — let them finish, then run the fixed
D3 on both npz.

### Also fixed this session (audit P0s)

- `tools/latent_q_sanity.py` (D4): the 10k relabel batch was **unseeded** (global
  np.random) — the gate number changed run to run. Now seeded off `cfg.seed`, same pattern
  as D3. D4 is a spec gate; its number must be a function of the checkpoint.
- `scripts/launch_fb_cube.sh` / `launch_psm_cube.sh` ran bare `python` (= 2.7 on midi-01).
  Now `$REPO/.venv/bin/python`.
- `agents/affine_psm.py` `infer_w_goal`: the constraint-set permutation used
  `np.random.default_rng()` (OS entropy) — identical seed + weights gave different `w_inf`
  and eval success. Now seeded off the `seed` argument.

### Later same session — D2 recovered (PASSES both envs), preimage analyzer landed

- **D-tools now persist their reports** (`utils/log_utils.py:write_report`; `report_out`
  config key, default `<hydra run dir>/<tool>.json`) — the class of loss that ate the
  08-03 D2 numbers is closed. D2's `across` metric also fixed: it averaged over self- and
  within-u pairs, deflating `across` and flattering the ratio.
- **D2 re-run on both Stage-A ckpts** (reports in `/data-local/amsks/PSMFLows/logs/`):
  consistency ratio **0.51 cube / 0.53 pointmaze** (within-u final-state distance ≈ half
  of across-u). Fixed `u` is reproducible AND distinct — the policy family is real, which
  answers note §7 risk 1 and gives A3 its first favorable evidence.
- **`tools/analyze_preimages.py`** (new): full-npz report (ESS/roundtrip/typicality
  distributions, posterior geometry vs prior and u_clip, correlations) + figure, JSON next
  to the npz. On the OLD alpha=1 cube npz: mean ESS 9.8, **39% of posterior means outside
  the u_clip box**, 85% of posteriors wider than the prior — quantifies what the alpha=20
  recompute restores. Run it on the new npz when they land.
- **New finding: 8 rows/1M have finite-but-astronomical point preimages** (||u*||² up to
  1e59; trimmed-mean typicality is healthy at 5.57 vs expected 5). The finiteness-based
  `preimage_valid` does NOT catch these. P1: extend `compute_preimage_validity` with a
  typicality bound (e.g. ||u*||² > 100 ⇒ invalid) — do it before point-mode training is
  ever used; EM-mode Stage C is unaffected (their mixtures are finite).

### Later still — amortized flowBC LATENT actor (Rung 3), off by default

`agents/psmflow.py` now carries the fb_flowbc actor recipe transposed to latent space
(`actor.enabled`, default **false** — v1 inference stays flow-GPI, and the pending Stage-C
launches are unaffected):

- `actor_vf` = FlowVectorField, CFM toward the dataset **preimage latents** (the
  latent-space behavior distribution — N(0,I) only under a perfect flow, so worth
  learning); `actor` = NoiseConditionedActor(s, w, noise) → tanh·u_clip.
- Actor loss mirrors `psm.flow_actor_loss`: −Q/|Q| + bc_coeff·distill + bc_flow_loss,
  with Q = the **diagonal** score ψ(s,u_a,u_a)ᵀw (value of committing to index u_a),
  same ensemble pessimism as gpi_select. ψ/φ take no actor gradients; flow frozen.
- w per batch = PSM's sample_mixed_z (task_mix_ratio phi(next_obs[perm]) vs random unit z).
- `acting: gpi | actor` switches deployment; both decode through the frozen flow, so
  actions stay data-like either way. Knobs in `configs/agent/psmflow.yaml` under `actor:`;
  reference cube used bc_coeff **3.0** (our default 1.0 — sweep when the ablation runs).
- Tests: `tests/test_psmflow_actor.py` — disabled default is a no-op; enabled trains only
  (actor, actor_vf) with psi deltas byte-identical to the disabled run; acting=actor
  emits a boxed latent and a valid action.

### Top open items before Stage C (full list + priorities in the roadmap doc)

1. **npz↔checkpoint pairing guard** in `main.py` — row count is the only check today; a
   wrong-vintage npz trains silently on mismatched latents. The `.meta.json` sidecar
   exists for exactly this. Do it before the first real Stage-C launch.
2. **Persist diagnostic reports** — the 08-03 D2 results were lost because the tools print
   JSON to stdout only. Add a `report_out` (or always `| tee`). D2 must be re-run.
3. Commit the 08-03 + 08-04 working tree as one story (inversion config + fixed D3 tool +
   benchmarks doc + this slide); fix `tools/plot_preimage_intuition.py`'s now-stale panel
   titles ("alpha=1 as run", "raising alpha is what clears the gate") when regenerating.

---

<!-- _class: lead -->

## 2026-07-29 (later) — the D3 ESS gate is an `alpha` problem, not a sample-count problem

Root-caused the failing gate by auditing the stored cube preimages. **`inversion.alpha=1.0`
is too low by ~20x**, and that single knob accounts for the failure. Chain of measurement:

**1. The flow map is shallow in `u`.** Singular values of J = d(action)/d(noise) at the
preimage, 4096 cube rows: sigma_max median **0.117**, sigma_min median **0.068**.

**2. So the Laplace proposal carries no information.** cov = (1/alpha^2)(J^T J)^{-1} with
sigma ~ 0.1 gives eigenvalues ~100, and they are clipped to `[0.01, 1.0]` — for **99.8%** of
rows *every* eigenvalue is clipped, leaving the proposal exactly the N(0, I) prior. The
"local Laplace covariance" is adaptive in name only.

**3. And the target is ~10x broader than that proposal.** With
`||G(s,u) - a|| ~ sigma*||u - u*||`, the target `exp(-alpha*||G(s,u)-a||)` has width
`1/(alpha*sigma) ~ 10` at alpha=1. A prior-width proposal against a 10x-wider target is
exactly the regime that collapses ESS — and it explains the EM's measured 4x-per-iteration
covariance growth as it chases a genuinely broad target.

**4. Raising alpha fixes every symptom monotonically** (512 cube rows, all else as configured):

| alpha | mean ESS | frac ESS>20 | post. width / prior | ‖mu‖ | ‖G(s,mu)-a‖ |
|---|---|---|---|---|---|
| **1.0** (as run) | 9.99 | 0.115 | 4.124 | 3.945 | 0.323 |
| 5.0 | 15.35 | 0.301 | 2.653 | 2.770 | 0.166 |
| 10.0 | 20.31 | 0.404 | 1.977 | 2.575 | 0.104 |
| **20.0** | **21.81** | 0.438 | **0.939** | 2.602 | 0.055 |
| 50.0 | 21.79 | 0.436 | 0.292 | 2.534 | 0.022 |

At alpha=20 the gate passes (mean ESS > 20), the posterior becomes *narrower* than its prior
for the first time — i.e. starts behaving like a posterior — ‖mu‖ lands near the chi_5
typical radius of 2.13, and the mean's round-trip improves 6x. ESS saturates by 20; past it
the posterior over-tightens for nothing. **This is the opposite of the 07-28 note's
suggestion to lower alpha**; flattening the target raises ESS only by discarding information,
and is measurably not the mechanism here. Table is recorded in
`configs/inversion/default.yaml`; the value is left at 1.0 pending the recompute call
(~10 h / 1M rows, and it invalidates the existing npz).

Two hypotheses checked and **rejected** — worth recording so they are not re-run:
- *More samples.* ESS tracks posterior geometry, not sampling noise: corr(ESS, cov trace)
  **+0.33**, corr(ESS, ‖mu‖) **−0.31**, but corr(ESS, ‖a‖) **−0.003** and mean ESS is flat
  at 9.6–9.8 across 0/1/2/3/4 clipped action dims. `num_samples` buys sqrt(N) against a
  mismatch it cannot remove.
- *The flow collapsed to a deterministic policy.* It has not. Varying `u` over the whole
  typical set moves the action with per-dim sd **0.105**, vs **0.270** for the behaviour data
  across the 32 nearest states — **39%**, narrower than the data but nowhere near degenerate.
  The policy family pi_u is real, so `psi(s,u',u)` has something to discriminate. (The small
  Jacobian invited the opposite conclusion; the nonlinear map over ‖u‖<=3 is what saves it.)

### The 13 NaN rows: root cause found, and it is NOT the EM

`_get_preimage_and_jacobian`'s **implicit-Euler inverse diverges** — a fixed 5 sweeps with no
convergence check. Traced step by step on row 32009: the preimage, Jacobian, J^T J and the
whole proposal are already NaN *before EM step 0*. Deterministic: 13/13 poisoned rows
reproduce across 5 independent rng seeds, 0/13 finite controls ever do. So a NaN mixture in
an npz means the inverse diverged, not that the EM misbehaved — the earlier attribution to
`fql.py`'s masking was wrong.

### Fixed this session

- **`preimage_ess` reported 100/100 on total failure** (`fql.py`, both EM and non-EM
  variants). When no sample is usable the uniform fallback makes every weight
  `1/num_samples`, so `1/sum(w^2)` evaluates to `num_samples` — the metric's *best* value.
  All 13 NaN rows scored a perfect 100, so **D3 gated on a metric that was blind to exactly
  its worst cases**. Now 0.0, with a test that pins it.
- **NaN latents could reach Stage C** (`utils/flow_inversion.py`, `main.py`,
  `tools/precompute_preimages.py`). New `preimage_valid` mask + `repair_invalid_preimages`:
  invalid rows are reset to the N(0, I) prior — the honest posterior under no information —
  counted, warned about, and **aborted above a 1% ceiling** so this can never quietly paper
  over a broken inversion. Rows are *not dropped*: `Dataset.sample` pairs each transition
  with row `idx+1`, so deleting rows would silently re-pair across the gap. Applied at load
  as well as at write, so the existing cube npz (which predates the key) is covered.

Audit notes: all 999,987 finite stored covariances are Cholesky-able in float32 (0 negative
eigenvalues), so `sample_preimage_noise` is safe; its unseeded global-`np.random` default is
fine because `main.py:76` seeds it from `cfg.seed`. `utils/flow_steering.py` discards ESS and
never checks finiteness — same exposure, but it is WP2 groundwork reachable only from its own
test, so latent. `tools/validate_flow_fidelity.py`'s `mmd_rbf` ignores `preimage_limit`
(hardcoded `act[:1024]`), so that one metric is always a 1024-sample estimate.

### D1 — Stage A signs off, but pointmaze is weakly certified

| config | MMD (RBF) | mode-hist TV | off-support @0.2 |
|---|---|---|---|
| pointmaze trained @100 | 3.0e-4 | 0.038 | 0.314 |
| pointmaze RANDOM control | 3.97e-3 | 0.111 | 0.522 |
| cube trained @100 | 1.5e-4 | 0.016 | 0.473 |
| cube trained @10 | 2.7e-4 | 0.019 | 0.350 |
| cube RANDOM control | 1.46e-1 | 0.507 | 0.9995 |

Both trained flows beat their controls (the spec sets no numeric threshold). **Cube's margin
is ~1000x on MMD; pointmaze's is 13x, and its random control is nearly competent** — with
`d_a=2` a near-identity flow already reproduces the action marginal, the same reason ESS is
not comparable across envs. A pointmaze D1 pass certifies much less than a cube one, which
matters because pointmaze carries the D4 gate. off-support is a real ~31% for pointmaze, not
a subsample artifact (0.350 @1024 → 0.329 @2048 → 0.314 @4096 → 0.316 @8192). Cube's *local*
fidelity is worse at flow_steps=100 (0.473) than at 10 (0.350) while global MMD/TV improve —
and 100 is the setting the preimages ran at.

### Stage A on pointmaze — DONE

500k steps, 1h27m, `flow_steps=100` (inversion-safe from the start, unlike the cube ckpt):
`/var/local/amsks/exp/PSMFLows/bcflow_pointmaze-medium-navigate_20260729_142219/sd000_20260729_142225`
wandb `qcmsr46t`. Note `utils/log_utils.py:73` `mkdtemp()`s the wandb dir and so **ignores
`WANDB_DIR`** repo-wide; offline runs land in `/tmp` and must be copied out before syncing.

### NEXT

1. **Decide alpha** (recommend 20) → recompute cube preimages → D3 gate should pass.
2. Stage B on pointmaze at the same alpha → D3 there.
3. **D2** on either ckpt — independent of all the above, and it is what confirms `u` indexes
   *distinct* policies rather than merely non-degenerate ones.
4. Then Stage C, D4, zero-shot vs PSM/FB.

---

<!-- _class: lead -->

## 2026-07-29 session — PSMFlow v1 Tasks 3–8 done; cube preimages landed; D3 ESS gate FAILS

Reference docs: note `PAPER/RESEARCH_NOTE.md` · spec `docs/design/2026-07-20-psmflow-v1-design.md`
· plan `docs/plans/2026-07-20-psmflow-v1.md`.

| Task | What landed | Commit |
|---|---|---|
| 3 | `PSMFlowAgent` core — flow-indexed measure loss, frozen-flow load, jitted update | `0c4f2ba` |
| 4 | Flow-GPI inference (`infer_z`, `gpi_select`, frozen-flow decode) | `c444a16` |
| 5 | Chain-MDP ground-truth test (GPI ranks goalward latents) | `b463b87` |
| 6 | Agent registered, `configs/agent/psmflow.yaml`, `main.py` wiring + smoke | `76310c8` |
| 7 | Diagnostics D1–D4 | `1f0fd42` |
| 8 | Stage launchers + this slide | *this commit* |

Suite: **102 passed, 1 skipped.**

---

## The three-stage pipeline — one command each

```bash
# Stage A — reward-free behaviour flow (FQL bc_only). One seed per GPU.
SEEDS="0" bash scripts/pretrain_behavior_flow.sh 1 500000 cube-single-play-singletask-v0

# Stage B — invert the flow over the dataset (MUST be flow_steps>=100)
.venv/bin/python tools/precompute_preimages.py agent=fql \
    env_name=cube-single-play-singletask-v0 \
    restore_path='/var/local/amsks/exp/PSMFLows/bcflow_*/sd000_*' restore_epoch=500000 \
    agent.flow_steps=100 preimage_out=/var/local/amsks/exp/PSMFLows/preimages_cube_single.npz

# Stage C — psmflow representation training
FLOW_CKPT='/var/local/amsks/exp/PSMFLows/bcflow_cube_single_20260726_135032/sd000_*' \
FLOW_EPOCH=500000 PREIMAGES=/var/local/amsks/exp/PSMFLows/preimages_cube_single.npz \
SEEDS="0" bash scripts/launch_psmflow.sh cube-single-play-singletask-v0 1 500000
```

**Stage A is `pretrain_behavior_flow.sh`, not the plan's `launch_flowbc.sh`** — Task 1 had
already created it under that name and two tool docstrings cite the path, so Task 8
extended it (multi-seed, `FLOW_STEPS`/`SAVE_INT`) instead of landing a second launcher for
the same stage. Both scripts guard their required inputs and exit 1 before JIT.

---

## Diagnostics D1–D4 (Task 7) — all `stdout` JSON, no pytest

| | Tool | Asks |
|---|---|---|
| D1 | `tools/validate_flow_fidelity.py` | does the flow reproduce the dataset's action distribution (RBF-MMD, mode-hist TV, off-support frac) |
| D2 | `tools/eval_fixed_u_rollouts.py` | is a fixed `u` reproducible AND distinct (within-u vs across-u final-state distance) |
| D3 | `tools/validate_flow_inversion.py` | typicality / round-trip / ESS — carries the spec gate |
| D4 | `tools/latent_q_sanity.py` | oracle-reward flow-GPI, isolating representation+GPI from reward inference. Gate: ≥50% of FQL on pointmaze-medium |

**D3 is now seeded — and that settles the 07-28 open question.** `Dataset.sample` drew from
global `np.random`, so every invocation scored a *different batch*; the earlier
"KS 0.095 → 0.061" reading was that, not a changed flow. Pinned at `seed=0`, cube gives
p = **3.4e-9** @100 and **6.3e-5** @200 — **A2 IS rejected at both**. This supersedes the
07-28 slide's "A2 is not rejected at 5% @200" caution. D3 also built its agent from
`fql.get_config()` rather than the Hydra agent group (same defect fixed in
`tools/precompute_preimages.py` in `03bfc71`); it now uses `cfg.agent`.

---

## OPEN — the D3 ESS gate FAILS on the real cube flow

| `flow_steps` | roundtrip | χ² band | mean ESS | gate |
|---|---|---|---|---|
| 100 | 1.2e-4 | 0.9873 | **7.16** | **FAIL** (needs > 20) |
| 200 | 7.5e-5 | 0.9912 | **7.14** | **FAIL** (needs > 20) |

Round-trip and typicality pass comfortably; only ESS fails. **Not a regression** — ESS was
~7 at every `flow_steps`, including settings that never NaN'd. It is the EM's actual
operating point on cube: `num_clusters=1` fits a 5×5 covariance (15 free parameters) from
~7 effective samples, and `min_ess=1.0` means some transitions put all posterior mass on a
single draw. Knobs: `inversion.num_samples` (linear cost) and `inversion.alpha` (flatter
target ⇒ higher ESS, less sharp posterior).

**ESS is not comparable across environments** — untrained pointmaze reads 16.9, *higher*
than the trained cube flow, because `d_a` is 2 not 5 and a near-identity flow gives a flat
posterior. Do not gate on the cross-env comparison.

**The cube preimages were computed at this ESS.**
`/var/local/amsks/exp/PSMFLows/preimages_cube_single.npz` — 319 MB, n=1,000,000, finished
07-29 00:47, from the 500k Stage-A ckpt at `flow_steps=100` (`.meta.json` sidecar records
the full inversion config). **The decision to make before Stage C: is 20 the right
threshold, or does this get recomputed at larger `num_samples`?** That is a method call,
not a code one. A recompute is ~10 h wall-clock at last night's rate.

---

## NEXT — Phase A exit (spec §5), operator-driven

1. Stage A flow on **pointmaze-medium** (cube ckpt exists) → **D1 gate** both envs.
2. **Resolve the ESS question**, then Stage B preimages for pointmaze (+ cube recompute if
   that is the call) → **D3 gate**.
3. **D2** rollout report — informational, feeds note §7 risk 1 (does `u` index distinct
   behaviours at all).
4. Train psmflow both envs, 500k, 3 seeds → **D4 oracle-GPI gate on pointmaze**.
5. Zero-shot eval vs in-repo PSM and FB, same envs/steps/seeds; extend
   `scripts/compare_multiseed.py` to psmflow groups.

Coverage-ladder stress tests and VC-FB comparisons are the *next* spec, once these gates pass.

Caveat carried forward: the landed cube Stage-A ckpt was trained at the FQL default
`flow_steps=10` and is inverted by overriding to 100 at precompute time. Stage A now
*trains* at 100 by default (safe only because `utils/xla_guard.py` disables the autotuner —
see the 07-28 slide), so a re-trained flow will not be step-identical to the current one.

---

<!-- _class: lead -->

## 2026-07-28 session — PSMFlow v1 Tasks 1–2; two numerics bugs cleared

Workstream switched to the **PSMFlow v1 plan**. Execution order is 1→2→…→8; GPU work
(Stage A → D1/D3 gates → preimages → psmflow 3 seeds → D4 → zero-shot vs PSM/FB) is
operator-driven after Task 8.

### Task 1 — DONE (`290b951`)

FQL `bc_only`: reward-free behaviour-flow pretraining. Strips the reward terms from the
actor loss, leaving `bc_flow_loss + alpha*distill_loss`; the critic branch is skipped and
`rewards`/`masks` are never read (the pretraining dataset has neither). Param tree is
unchanged so checkpoints restore into a default-shaped FQL agent in Tasks 2/3.
**Stage-A cube checkpoint landed at 500k:**
`/var/local/amsks/exp/PSMFLows/bcflow_cube_single_20260726_135032/sd000_20260726_135037`.

### The XLA autotuner miscompiles the flow integration (`d2b3ec1`)

D3's round-trip of 2.26 and NaN ESS were **both artifacts**. Past a threshold unroll
length, XLA:GPU's autotuner picks a wrong kernel for `compute_flow_actions`: 84.5% of
outputs pinned at the ±1 clip jitted vs 4.5% un-jitted, reproducibly, across processes;
CPU correct in every form. Onset is between `flow_steps` 30 and 100, so **training at
`flow_steps=10` was never affected and the Stage-A checkpoint is sound** — but the plan's
Stage-A launcher specifies `flow_steps=100` and WOULD have trained on a corrupted
distillation target. Fix: `utils/xla_guard.py` sets `--xla_gpu_autotune_level=0` **before
jax initialises** (it is read at XLA init; setting it later is a silent no-op), imported
ahead of jax by `main.py`, the GPU tools, and `tests/conftest.py`.
**The trigger needs TRAINED weights** — random weights agree to 3.5e-5, which is why a
self-contained numerics test would pass vacuously (it is opt-in behind `PSMFLOWS_FLOW_CKPT`;
the committed test pins the import ORDER instead).

### The residual NaN ESS — fixed this session

Not "the EM proposal is broken at `flow_steps>=100`". One unguarded NaN with a wide blast
radius, from two compounding causes:

1. **The BC flow genuinely diverges from tail proposal samples.**
   `_get_predistribution_proposal` clips the Laplace cov eigenvalues to `[0.01, 1.0]` and
   on cube it **saturates at the upper bound**, so the EM effectively samples `~N(x_0, I)`
   — as wide as the prior. ~0.02% land at `||u||~6.8` (mean 3.1) and integrate
   `6.5 -> 2.3e10 -> inf -> NaN` by step ~95. **`flow_steps` 10 and 30 under-resolve the
   blow-up and stay finite** — that, not any instability of fine discretization, is why it
   only appeared at >=100.
2. **One NaN killed the whole row.** The 07-28 guards floored `log_q` but never checked
   `log_energy`, and softmax over a vector containing one NaN is NaN in EVERY position.
   Cascade: 17% of rows NaN at EM step 0 -> 69% at step 1 -> **100% by step 5**.

Fix: mask on both grounds (`ok = q_ok & isfinite(log_energy)`) -> logit `-inf` (weight
exactly 0), uniform fallback if nothing survives. Masked samples keep responsibility
`1/K` so `gamma = resp * weight` stays finite — **`0 * NaN` would not**. Same guard applied
to the non-EM `compute_full_proposal_distribution`, which had identical exposure.
`test_em_ess_survives_a_diverging_flow_sample` injects one NaN action (random weights do
not diverge on their own) and asserts the sample is EXCLUDED, not merely tolerated
(`ESS <= 23` of 24). **77 passed, 1 skipped.**

### D3 gate — Stage-A cube checkpoint, 1024 transitions, guard active

| `flow_steps` | roundtrip | KS (p) | mean ESS | min ESS |
|---|---|---|---|---|
| 10 | **NaN** | 0.51 (0) | 6.88 | 1.0 |
| 30 | 3.4e-4 | 0.217 (6e-43) | 6.96 | 1.0 |
| 100 | 1.2e-4 | 0.061 (9e-4) | 7.13 | 1.0 |
| 200 | 7.6e-5 | 0.039 (**0.087**) | 7.13 | 1.0 |

- **Preimage precompute must run at `flow_steps>=100`.** At 10 the implicit-Euler fixed
  point (5 sweeps) diverges outright; 10 is the *training* default and is NOT safe for
  inversion.
- **mean ESS ~7/100 at EVERY step count**, including the regimes that never NaN'd ⇒ that is
  the EM's normal operating value, not damage from the divergence and not something the fix
  changed. Low, and `min_ess=1.0` means some rows put all mass on one sample — a
  proposal-quality question worth a threshold on the Task-2 `preimage_ess` health scalar,
  but not a blocker.
- **CAUTION — D3 is unseeded.** `Dataset.sample` draws from global `np.random`, so
  typicality moves batch-to-batch: these numbers differ from `d2b3ec1`'s (KS 0.095->0.061
  @100, 0.068->0.039 @200) because it is a **different batch**, NOT because anything
  improved. A2 is not rejected at 5% @200 *on this batch only* — do not read that as A2
  now holding until D3 takes a seed. Worth fixing before D3 is used as a gate.

### NEXT — Task 2 (preimage pipeline hardening)

The ESS blocker is cleared, so Task 2 is unblocked. Per the plan: checkpoint restore +
Hydra agent cfg + meta sidecar in `tools/precompute_preimages.py`; `noise_preimage_point`,
`preimage_roundtrip`, `preimage_ess` npz keys; `preimage_point_mode` on `Dataset`;
`allow_untrained: false` in `configs/inversion/default.yaml`; `tests/test_preimage_pipeline.py`.
**No preimage `.npz` exists yet.** Run the precompute at `flow_steps>=100`.

---

<!-- _class: lead -->

## 2026-07-26 session — flowBC actor for affine PSM; reference-HP audit

**Question:** affine PSM sits at floor on `cube-single-play-singletask-v0`. Are we
using the flowBC actor the reference uses for cube?

### Finding 1 — we were not (now fixed)

`../Factored-FB` README is explicit: `fb_flowbc` (flow-matching VF + distilled noise
actor) for **cube/scene/puzzle**; the deterministic actor only for antmaze/locomotion.
The proven cube recipe whose `ortho_coef=1000` our config copied
(`psm_state_orthohi__ortho_coef1000`) launches with `agent=psm_flowbc`. We had inherited
the hyperparameter and dropped the actor it was tuned around. Mechanism: cube actions are
multimodal, and a tanh mean fit by MSE-to-data regresses to the **mean of the modes**,
which is not a valid action.

- **Ported** the flow path into `agents/affine_psm.py` behind `actor.type: ddpgbc | flow`
  (mirrors `agents/psm.py`): `flow_actor_loss` = CFM velocity field `v(s,x_t,t)` + one-step
  `NoiseConditionedActor` distilled from its 10-step Euler rollout; new `actor_vf`
  TrainState; dispatch in `apply_update`/`total_loss`/`sample_actions`. Q is unchanged —
  still the frozen-measure goal Q `Φ(s,a,g)·w_g + b(s,a,g)`, factored into `_goal_task_coord`
  / `_goal_q` so both actor branches share it.
- Config defaults from the reference `psm_flowbc.yaml` (512×2 actor, 512×4 VF,
  `flow_steps=10`, `lr_actor_vf=3e-4`, `bc_coeff` 1.0 → **3.0** = the cube value).
- `tests/test_affine_psm_flow.py` (9 tests) + one pre-existing test rerouted through
  `sample_actions` (it called `agent.actor(obs, w)` with the ddpgbc arity). **27/27 green.**

**Result: the actor was NOT the binding constraint.** 3 seeds × 500k, 50 eval episodes
(`affine_flow_500k_20260726_022644`): peaks **0.08 / 0.10 / 0.04**, no trend. At a matched
500k the ddpgbc baselines were 0.00–0.10, so flow is **not worse** — but it does not fix it.

### Finding 2 — affine PSM is running RLU's DMC-scale HPs, not the cube recipe

`batch_size=32`, `d_dim=z_dim=50`, `lr=1e-4` all come verbatim from
`RLU/controllable_agent/url_benchmark/agent/psm.py` (a DMC/gridworld codebase), not from
anything tuned on OGBench cube. Deviation table vs `../Factored-FB/configs/agent/psm.yaml`:

| HP | reference (cube) | affine_psm | matchable? |
|---|---|---|---|
| `batch_size` | 1024 | 32 | **NO — see Finding 3** |
| basis dim | 128 | 50 | free |
| basis LR | **1e-5** (sweep winner; `psm.yaml` says "NOT 1e-4") | 1e-4 | **NO — see Finding 4** |
| `max_log_seed` | 16 | 12 | free |
| `target_tau` / `discount` / `ortho_coef` / `lr_actor` | 0.01 / 0.98 / 1000 / 1e-4 | same | already matched |

### Finding 3 — batch_size=1024 is architecturally blocked (measured)

The bilinear PSM gets B² free: `M = ψ(s,z,a) @ φ(g)ᵀ` is an outer product of two `B×d`
matrices, so B=1024 costs **1024 network evals**. The affine net takes `x` *inside* the
network, so B=1024 costs **1024² = 1,048,576 evals** of the 1024×3 measure MLP.
**Probed on GPU: it OOMs** — a single backward fusion alone requested 4.02 GiB and failed
at `XLA_PYTHON_CLIENT_MEM_FRACTION=0.45`. This is the architecture, not a missing knob.

*Options if we want the reference's effective batch:* a **rectangular mesh** — decouple
source rows from measure-argument columns (`B_src=1024 × B_x=64` ≈ 65k pairs, feasible)
so the source-side gradient noise and the contrastive negative count are restored
independently; or a square B=256 (same 65k pairs). Both need a change to `measure_loss`,
which currently builds `i_idx=repeat(arange(B),B)` / `j_idx=tile(arange(B),B)`.

### Finding 4 — the reference's two-timescale split has no landing spot in affine PSM

The reference does **not** run actor-vs-critic two-timescale: `lr_sf = lr_actor = 1e-4`.
Its split is **basis vs everything else** — `lr_phi=1e-5`, 10× slower, marked the sweep
winner. We already match on the actor axis (`lr_actor == lr == 1e-4`; the `lr_actor` knob
from `0a82050` defaults to equal, and the one probe — `affine_tt_1e4` vs `affine_tt_3e5` —
was inconclusive at 10 eval episodes).

But `AffineMeasureNet` (`utils/psm_networks.py:236`) is a **single trunk on
`concat[obs,action,x]`** with φ/b heads: the basis shares every weight with the successor
side, so **there is no parameter group to slow down**. Restoring the reference's structure
means splitting it into separate x- and (s,a)-branches so the x-branch carries its own
optimizer.

### IN FLIGHT (launched 03:41, ~45 min, 2 seeds/GPU on 0,1,3)

| group | vs the 500k flow runs |
|---|---|
| `affine_refdims_20260726_034122` | `d_dim`/`z_dim` 50→**128**, `max_log_seed` 12→**16** |
| `affine_refdims_slowbasis_20260726_034122` | same + `lr` 1e-4→**1e-5** |

Both flow actor, `zero_shot` inference, 3 seeds, 500k, 50 eval episodes. refdims vs the
flow runs isolates the dims; slowbasis vs refdims isolates the representation timescale.
**Caveat on slowbasis:** `lr=1e-5` slows the *whole* measure net (basis + offset +
successor side), whereas the reference keeps ψ at 1e-4. If it helps ⇒ the Finding-4
refactor is worth doing properly. If it hurts, suspect plain underfitting at 500k before
concluding the timescale idea is wrong.

### Corrections to earlier reads (don't re-derive these)

- The ddpgbc baselines do **not** top out at 0.2 once. Reading only the last CSV rows is
  misleading — `affine_psm_amortized` hits 0.20 @200k, 0.20 @600k, **0.30 @800k**; the good
  numbers all sit **past 500k**. Those were 10-episode evals (each 0.1 = one episode).
- Reported `psm_loss` for affine PSM is dominated by a constant: with sqrt(d)-normalized Φ
  the ortho diagonal term is inert at ≈ −d, so `ortho_coef=1000` contributes a ≈ −50000
  offset. **Read `orth_offdiag` (decorrelation) and `psm_offdiag`, not `psm_loss`.**

### Known issue, NOT fixed (flagged, outside this session's scope)

`main.py:80` sets `dataset.return_index = True` only for `agent_name == 'psm'`, so
`affine_psm` falls back to `jnp.arange(B)` for the codebook hash — a transition's proto
action changes with **where it lands in the batch**. `agents/psm.py:315-319` documents this
as incorrect. Worth fixing before trusting any conclusion about the measure.

### Runnable

`bash scripts/launch_affine_psm_cube.sh [GPUS] [STEPS] [WANDB_MODE]` (new) — comma-separated
GPUs, **one seed per GPU**, `SEEDS=`/`ACTOR=`/`EXTRA=`/`EVAL_INT=` env overrides, uses
`.venv/bin/python` explicitly. Call it twice with different `GROUP` to stack 2 configs/GPU.

---

<!-- _class: lead -->

## 2026-07-15 session — FB agent ported to JAX (bit-exact)

Ported the PyTorch **Forward–Backward (FB)** agent (`../Factored-FB` `agents/fb/*`)
to JAX/Flax, mirroring the PSM port protocol. **Spec** `docs/design/
2026-07-14-fb-jax-port-design.md`, **plan** `docs/plans/2026-07-14-fb-jax-port.md`.

- **Scope:** cube-default `fb_flowbc` path — Forward map `F(left_enc(obs),z,a)`,
  Backward map `B(next_obs)` (measure basis + z-source), a **left_encoder** trunk
  (with target), td3/flow actor. Measure `M=F·Bᵀ`, off-diag/diag + ortho loss.
  2 backward passes (FB, actor); targets on forward/backward/left_encoder, none on
  actor; taus 0.005. z mixed 50/50 with `B(next_obs[perm])`. **Not** ported:
  iql critic, traj goal-mode, fixed_b, goal_cond, onestep, reweight.
- **Files:** `agents/fb.py`, `utils/fb_networks.py` (ForwardMap/BackwardMap/FBTd3Actor;
  reuses NoiseConditionedActor/FlowVectorField), `configs/agent/fb.yaml`
  (z_dim=50, batch=256, disc=0.99), registered `fb` in `agents/__init__.py`.
  `main.py` needs NO FB branch (no `index`; `infer_eval_z` picked up generically;
  seeded eval applies).
- **Bit-exact parity (like PSM):** `tools/export_fb_fixture.py` → `tests/fixtures/
  fb_reference.npz`; `tests/test_fb_{networks,agent,smoke}_equiv.py`. Per-module
  atol 1e-10, 10-step `apply_update` atol 1e-8. **13/13 FB tests pass.** Also the
  first bit-exact check of our shared NoiseConditionedActor/FlowVectorField.
- **Fixture gotchas (cost a long debug):** (1) torch `.numpy()` **aliases** memory —
  per-step param snapshots must `.copy()` or they all show the final trained weights;
  (2) the reference torch build's in-place first Adam step lands ~10× too large under
  some construction orders, so the K-step trace uses a **manual optax-matching Adam**.
- **Runnable:** `bash scripts/launch_fb_cube.sh [GPU] [STEPS]` (one seed/GPU,
  save_interval=100k, seeded eval, eval_episodes=10). Verified 3-step end-to-end
  through `main.py` (exit 0). **NEXT:** cube-single parity run vs the reference FB
  benchmark (wandb `amsks/factored-fb`; see [[reference-fb-flow-benchmarks]]).

---

<!-- _class: lead -->

## 2026-07-13 session — fresh from-scratch parity runs

**Goal:** measure how much our JAX PSM+flow recovers vs the reference on
from-scratch runs (not transplant), 3 seeds, 500k.

- Launched seeds **0/1/2**, flow, `ortho_coef=1000`, `z_dim=128`, `lr_phi=1e-5`,
  eval@50k / 50 ep. Group `psm_recover500k_flow_ortho1000_20260713_160709`.
- **NEW, CONTRASTS with prior finding:** **seed 2 recovers the reference
  in-window.** seed2 @500k = **0.42** vs ref **0.60** (−0.18); but
  **mean(0–500k) ours 0.238 vs ref 0.240 — a wash.** We LEAD early (50k/100k the
  ref is still 0.00), ref leads the 250–300k bump. Peak ours 0.44@350k vs ref's
  own in-window 0.50. This is NOT the old "plateau at ~0.05–0.11" story.
- **Seeds 0 & 1 weaker so far** (peaks 0.18 / 0.22 at 450k) — but the reference
  for THOSE seeds is also weak early (s0 ref 0.30 by 300k). ⇒ **high seed
  variance dominates**; single-seed reads are noisy.
- **THE remaining gap is the CEILING, not the trajectory.** The reference keeps
  climbing past 500k: peaks **0.80 @750k**, holds **0.5–0.8 to 1.5M**. At our
  500k cap the −0.18 is real but small; the reference's *ceiling* only appears
  with 2–3× more training we hadn't run.
- **⇒ Launched seed 2 to 1M** on GPU 1 (solo) to test the ceiling.
  Group `psm_recover1M_flow_ortho1000_s2_20260713_174415`. **At 250k already
  0.60** (ref 0.50 @250k). ETA ~2.5–3h from 17:44.
- **Infra lesson:** 2 seeds sharing one GPU run ~2× slower (seeds 0/1 on GPU3
  took ~2.5h; seed 2 solo on GPU1 did 500k in ~1h23m). **One seed per GPU** for
  even wall-clock. `XLA_PYTHON_CLIENT_MEM_FRACTION=0.30` per proc.

### Runs (2026-07-13)

| group | seeds | steps | GPU | status |
|---|---|---|---|---|
| `psm_recover500k_flow_ortho1000_20260713_160709` | 0,1 | 500k | 3 (shared) | running (~450k) |
| ″ | 2 | 500k | 1 | **done** @500k=0.42 |
| `psm_recover1M_flow_ortho1000_s2_20260713_174415` | 2 | **1M** | 1 (solo) | running (~250k, 0.60) |

Compare: `.venv/bin/python scripts/compare_multiseed.py [GROUP]` (defaults to
`/var/local/amsks/exp/multiseed_group.txt`; the 1M group is in
`recover1M_s2_group.txt`). Reference cache has seeds 0,1,2,3,4,5,7.

### Next (2026-07-13 → next session)

1. **Read the 1M seed-2 result** — does it reach the ref's 0.7–0.8 ceiling? If
   yes ⇒ we DO have parity, just needed budget; the "systematic gap" was a
   500k-cap artifact + seed variance. If it plateaus ~0.4 ⇒ real ceiling gap.
2. Let seeds 0/1 finish 500k; re-run `compare_multiseed.py` for the 3-seed table.
3. If ceiling confirmed, consider 1M runs for seeds 0/1 too (one-per-GPU) to
   get a real 3-seed peak distribution vs the reference's spread.
4. GPUs 1 & 3 usable; 0 & 2 were busy/full (other users) this session.

---

## TL;DR — the headline finding

- Our JAX PSM+flow **task2 success plateaus at peak ~0.20–0.32** and stays near the floor.
- The **code-matched reference climbs to 0.30–0.60 by 500k and peaks 0.80–0.90** (at 650k–1350k).
- This is a **real, systematic difference** (all 3 seeds fail uniformly), **NOT noise**.
- It is **NOT** in the eval/acting path, the loss math, masks, the proto table, or obs-norm
  (all audited/matched). **It is in TRAINING/init** — our update yields weaker weights.
- **CONFIRMED by transplant-eval:** reference weights (seed5 @100k) score **0.58** in OUR eval
  (task2); our own weights top ~0.25 in the same eval. Our eval is faithful; **training is where
  our weights end up worse.**
- **UPDATE (round 9): init scheme ALSO faithful.** After 9 rounds, EVERY code path is faithful
  (per-step math, training-gen, acting, eval, init scheme, config). **No code bug found.**
- Remaining diffs are **pure cross-framework RNG** (init draws + minibatch order). The reference
  is **hugely noisy** (same seed5: 0.20 wandb vs 0.52 fresh @100k). Effect size is uncertain
  with 3 seeds×1 run — may be a low draw, not a bug.

---

## How we got here (this session)

1. Started from an "eval@100k red flag". First pass concluded *faithful* — but that was an
   **improper comparison** (wrong reference algo for s5/10/7 + mid-run vs 1.5M peak).
2. Corrected: pulled the **code-matched PSM-orthohi** reference curves (wandb
   `amsks/factored-fb` `psm_state_orthohi__ortho_coef1000__lr_phi1e-5__s{0,5,7}`), aligned
   step-for-step. At matched steps ≤140k we tracked it — but the reference **blooms late**.
3. User (correctly) pushed: a faithful port must reproduce the reference's tight 0.8–0.9
   **peak distribution**. It doesn't.
4. Transplanted the reference **proto behavior table** to kill the last RNG diff → **gap did
   NOT close**. Confirmed the difference is real.

---

## What is RULED OUT (audited faithful / matched)

- **Per-step update math** — `tests/test_psm_agent_equiv.py` transplants reference weights,
  runs 10 optimizer steps on injected data, matches to **atol 1e-8**. Network+objective+HPs
  are bit-identical *given identical inputs*.
- **Eval z-inference** — formula-identical (`infer_z` == ref `reward_inference`); reward
  source matches (our dataset 2.08% nonzero == ref relabel ~2%).
- **Eval task identity** — `cube-single-play-singletask-v0` has `cur_task_id=2`; we compared
  task2-to-task2 all along.
- **Acting path** — `sample_actions` + actor nets faithful (one-step flow sample, tanh/clip,
  z-proj all match; multi-step Euler rollout is only a distill target, never acted).
- **masks** (always-γ), **obs normalization** (Identity for state, both sides), **proto table
  distribution** (`(rand-1)*2 ∈ [-2,0)`), now transplanted verbatim.

---

## KEY INSIGHT — why "losses match" proved nothing

- `actor_loss ≈ -Q/|Q|` and `train/q` is the **critic's own** estimate — both read ~-0.77 /
  ~600 **whether or not the policy reaches goals**. `bc_flow_loss` only measures fit to the
  behavior dist. **None of the scalar losses measure task success.**
- The equiv test **injects** the batch + all latent/action samples, so it verifies the update
  *given inputs* but **never how those inputs are GENERATED**.
- ⇒ The bug must live in **training-signal generation** (batch sampling, `z_cont` mixing,
  next-action injection, target updates) — invisible to both the losses and the equiv test.

---

## Two deep audits — BOTH came back "faithful"

- **Acting/eval-path audit: FAITHFUL.** `sample_actions` + actor nets match; eval task is
  task2 both sides; z-inference reward matches.
- **Training-generation audit: FAITHFUL.** batch sampling, `z_cont` mix, SF/proto next-action,
  target updates, per-stage optimizers all match. Only nits: `z_cont` mixed half uses
  pre-proto-step phi (ref uses post-step) — one `lr_phi=1e-5` step, negligible; proto table
  (now transplantable).
- **Config verified matching:** our runs used **z_dim=128, ortho_coef=1000, actor=flow**
  (checked `flags.json`). The audit's `z_dim=50` worry was stale-handoff text, not real.

⇒ **No code/config bug found on either side.** Remaining suspects: **init SCHEME** (orthogonal
gain/draw — audits didn't deep-check; all 3 seeds fail uniformly = systematic, argues against
pure basin luck) OR genuine cross-framework init/data-order basin variance (weakened by 3/3
uniform failure).

## STILL IN FLIGHT — the discriminator

**Torch reference checkpoint dump** — `seed5`, **ortho_coef=1000** (gotcha: psm_flowbc default
is 1.0!), GPU 1, `/var/local/amsks/exp/ref_ckpt_s5_300k`, saving `step_{100,200,300}k.pt`.
**Transplant-eval:** load ref weights into our JAX eval →
- ref-weights score **high** ⇒ bug is in **our training/init** (not eval).
- score **low** ⇒ bug is in **our eval** (both audits say unlikely).
To separate init-vs-training: load ref *early* weights into our JAX and TRAIN — climbs ⇒ init.

---

## Transplant-eval — READY to build (checkpoint landed)

`step_100000.pt` (+200k,300k) saved in `/var/local/amsks/exp/ref_ckpt_s5_300k/`. state_dict
naming **matches the fixture** convention → prefix keys with `w__` and reuse existing loaders.

- **phi / sf_psi / psm_psi:** reuse `load_phi_params` / `load_psi_params` as-is. Keys:
  `phi.net.{0,1,3,5}`, `{sf,psm}_psi.{embed_z,embed_sa}.{0,1,3}` + `Fs.{0,2}` (ensembled, P=2,
  shapes `[2,in,out]`). targets present too (`target_*`).
- **NEW converters needed** (torch_to_flax is ddpgbc-only):
  - `_actor` = **NoiseConditionedActor** (18 params): `embed_z.{0,1,3}`, `embed_s.{0,1,3}`,
    + noise/policy layers → our `utils/psm_networks.py:NoiseConditionedActor` tree.
  - `_actor_vf` = **FlowVectorField** (10 params): `net.{0,2,4,6,8}` (5 Linears, GELU between)
    → our `FlowVectorField` tree.
- Harness: build `PSMAgent` with these params (config z_dim=128, ortho1000, actor=flow), run
  `infer_eval_z` + `evaluate` on task2. **Ref weights should score ~0.20 (100k)** in a faithful
  eval — matches this run's own eval (seed5 task2 0.08@50k, climbing). Low ⇒ eval bug; ok ⇒
  training/init bug. **A buggy converter gives a false low — validate by matching phi/Q outputs
  to the torch model on a shared batch first.**

---

## Code changes this session

- **Committed & pushed** `31ed43d` "PSM: reference-parity fixes (eval z-inference + masks) +
  audit tooling": masks always-γ (`agents/psm.py`), eval z-shift/relabel (`main.py`,
  `config.yaml`), `eval_interval 100k→20k`, `docs/`, `scripts/launch_psm_cube.sh`.
  **NB: commits follow existing history conventions.**
- **UNCOMMITTED** (working tree): `proto_table_path` hook — `agents/psm.py` `create()` loads a
  transplanted table when `agent.proto_table_path` set; `configs/agent/psm.yaml` declares it
  (`null` default). 15/15 PSM tests still pass.

---

## Runs (this session)

| group / run | what | status |
|---|---|---|
| `psm_maskfix_flow_ortho1000_20260707_164832` | s0/5/10/7, 500k, eval+masks fix | **killed** (superseded) |
| `psm_protoxplant_flow_ortho1000_20260707_184559` | s0/5/7, 500k, +proto transplant | **done** — gap NOT closed |
| `/var/local/amsks/exp/ref_ckpt_s5_300k` (torch) | ref s5 ortho1000, ckpt dump | **running** (GPU 1) |

Late-window (350–500k) task2 mean, protoxplant: ours s0/5/7 = **.06/.11/.05** vs ref **.08/.35/.33**.

---

## Reference curves (code-matched PSM-orthohi, task2)

Cached: `/var/local/amsks/exp/ref_orthohi_task2_curves.json` (seeds 0/5/7, every 50k to 1.5M).

| seed | @100k | @300k | @500k | peak |
|------|------:|------:|------:|-----:|
| 0 | 0.10 | 0.20 | 0.00 | 0.80 @1350k |
| 5 | 0.20 | 0.60 | 0.30 | 0.90 @650k |
| 7 | 0.00 | 0.40 | 0.10 | 0.80 @900k |

Reference is **very noisy** but clearly trends up; ours does not. `s10` has **no** code-matched
reference (orthohi set is seeds 0–9) — dropped it.

---

## Next steps (priority order) — DONE: transplant-eval + init audit (both clean)

All 9 rounds of code audit are clean. The question is no longer "where's the code bug" but
"is our training genuinely worse, or a low RNG draw." Two ways to settle it:

1. **DECISIVE: continue-training from ref weights in OUR loop.** Load ref 100k weights (score
   0.58 in our eval) into a trainable `PSMAgent` + fresh opt_states, train 100k more with our
   `update`, eval. **DEGRADES toward ~0.2 ⇒ our training dynamics actively hurt (real bug);
   PRESERVES/climbs ⇒ it was init-draw/basin luck.** (Extend `scripts/transplant_eval.py`:
   add opt_states init + our training loop over the dataset.)
2. **Characterize distributions:** run s0/5/7 (+more seeds) as **multiple replicates** and
   compare to the reference's own spread — the reference bounces 0.20–0.52 for the same seed.
3. Commit the `proto_table_path` hook (+ converter/harness) once settled.
4. Peak parity ultimately needs **1.5M** runs (user capped at 500k this round).

---

## Environment gotchas (unchanged + new)

- **`python` = system Python 2.7** on this box. ALWAYS use `.venv/bin/python` (ours) or
  `/var/local/amsks/ffb-venv/bin/python` (torch). The launcher needs `source .venv/bin/activate`.
- **midi-01 = HTCondor, no Slurm.** Run directly (nohup/tmux + `CUDA_VISIBLE_DEVICES`).
- Home NFS quota tiny → all bulk data in `/var/local/amsks/`. GPUs shared — check `nvidia-smi`.
- Torch stdout is **block-buffered** to a file (looks "stuck"; it's flushing in bursts).
- **Reference ortho_coef override is bare `ortho_coef=1000`** (`@package _global_`), NOT
  `agent.*`; psm_flowbc default is 1.0. The old repro `ref_psm_cube_s0_100k` was ortho=1.0.

---

<!-- _class: lead -->

## Pointers

- Agent: `agents/psm.py` · Networks: `utils/psm_networks.py` · Converter: `utils/torch_to_flax.py`
- Config: `configs/agent/psm.yaml`, `configs/config.yaml` · Launcher: `scripts/launch_psm_cube.sh`
- Reference (PyTorch): `/u/amsks/git/Factored-FB` (`agents/psm/*`, `nn_models.py`), torch venv
  `/var/local/amsks/ffb-venv`, data `/dev/shm/factored-fb/datasets`
- Investigation tools (now in repo): `scripts/transplant_eval.py` (load ref torch weights →
  our JAX eval), `scripts/compare_protoxplant.py` (step-aligned curve vs cached ref)
- Cached ref curves: `/var/local/amsks/exp/ref_orthohi_task2_curves.json`
- **Memory: `psm-vs-reference-audit` (rounds 1–6 — the full investigation trail)**

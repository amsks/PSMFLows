# Audits + three ideas (2026-08-12, evening)

Order is deliberate: **Phase 0 audits are blocking** — none of the three ideas launches
until the audit table exists and round-trip is re-certified, because all three build on
the PSM/FB/inversion code being trustworthy and HP-comparable. The L1 residual sweep
(`latrl_res*`, 2026-08-12 boost-levers plan) KEEPS RUNNING throughout; do not kill it.
L2/L3 of that plan are paused in favor of this one unless GPUs are otherwise idle.

Ground rules: §0 of `2026-08-10-iclr-figures.md` (venv python, named tmux + kill-server,
one training seed/GPU, 500-ep Wilson / seed mean ± 95% CI, BC control quoted, JSON
reports to `/data-local/amsks/PSMFLows/logs/`, pre-registered interpretations).

## Phase 0 — audits (blocking; ~1 day, no training)

### 0a. PSM-vs-FB hyperparameter audit

Produce `docs/audits/2026-08-12-psmflow-fb-hp-diff.md`: a full diff table of every HP
between (`agents/fb.py` + `configs/agent/fb.yaml`, as run in `fb_cube_ortho1000_20260810`
— read the run's flags.json, not just the yaml) and (`agents/psmflow.py` +
`configs/agent/psmflow.yaml`, as run in `psmflow_latentpsm_cube_021456`). Columns:
HP · FB value · psmflow value · matched? · classification (intentional / unintentional /
needs-ablation). Known suspects to check explicitly:

- **Two-timescale learning**: FB/PSM reference runs the basis 10x slower
  (`lr_phi=1e-5` vs everything else 1e-4) — verify BOTH agents actually implement the
  split (which parameter groups each LR touches), not just declare it.
- **Orthonormality**: ortho loss form and coefficient (our 1000 vs FB-as-run), and
  WHERE it applies (φ vs B; normalized how). Verify the loss term is active in the
  losses actually computed (read the jitted loss fns, not the config).
- **discount** (ours 0.98 vs FB 0.99), **target tau** (0.01 vs 0.005), **z_dim**
  (128 vs 50), **z mixing** (our mix_ratio 0.5 φ(next_obs[perm]) vs FB's 50/50
  B(next_obs[perm])), **batch size**, **pessimism** (our 0.5 exact-min vs FB's
  handling), **actor bc_coeff** (ours 1.0 vs reference cube 3.0).
- Output ends with a ranked shortlist: which mismatches plausibly matter, each with a
  one-line rationale. **One bundled "HP-matched" psmflow arm** (all shortlisted values
  set to FB's, 2 seeds × 500k, tmux `psmflow_hpmatch_sd{0,1}`) launches AFTER the
  table is reviewed — a single bundle, not a per-HP sweep.

### 0b. FB run-config audit

Confirm the FB 0.72 runs used the reference cube recipe (fb_flowbc actor, correct
ortho override — recall the historical gotcha: reference ortho override is global, not
`agent.*`). One page appended to the same audit doc. If the FB runs turn out
mis-configured, flag to the user IMMEDIATELY — every comparison in the note quotes them.

### 0c. Inversion re-certification

Re-run `tools/analyze_preimages.py` on all three published npz; assert in a JSON report:
per-row round-trip distribution (mean ~1e-4, p99, max), `preimage_valid` counts
(13 cube / 0 pointmaze / 881 antmaze expected), typicality trimmed-mean, and the
npz↔flow pairing guard passing. Any regression from the published card numbers is a
stop-and-report. Output `logs/audit_inversion_recert.json`.

## Phase A — Idea 2: does FQL overestimate? (IMMEDIATE — start alongside Phase 0; 2a needs no training)

Motivation: our support argument leans on FQL's off-data actions being *genuinely good*.
The offline-RL null hypothesis is that FQL's critic overestimates off-data actions and
succeeds despite (or through) it. Test both directions.

### 2a. Calibration on existing checkpoints (no training, ~hours)

New tool `tools/diag_fql_calibration.py` on the 3 existing FQL cube checkpoints:
- At ~64 eval start states: Q(s0, a_FQL(s0)) vs realized discounted return over full
  rollouts (per-seed; report bias = mean(Q − G_realized), ratio, and Spearman across
  states).
- Q-gap probe: Q(s, a_FQL(s)) − Q(s, a_data-NN(s)) at dataset states — how much of
  FQL's claimed advantage lives exactly in the off-data direction.
- Output `logs/diag_fql_calibration_cube.json` with a verdict:
  **calibrated** (bias small, positive Spearman) ⇒ the off-data actions are real
  improvements and our support-cost story stands as-is;
  **overestimated** (large positive bias) ⇒ FQL's 0.95 partially rides extrapolation
  error — say so in the note (it does NOT rescue our 0.22, per-task latent RL used the
  same rewards — but it reframes the FB/FQL toplines).

### 2b. Dataset-reduction arms (training; 1 seed per arm first)

Episode-level random subsets of the cube dataset at {10%, 25%, 100%} (seeded,
`utils` loader change with a `dataset_fraction` config key; keep episode boundaries
intact). Train FQL 500k on each, tmux `fql_frac{10,25,100}_sd0`, then 500-ep eval +
the off-support distance of its chosen actions (reuse coverage-probe machinery).
Pre-registered readings:
- FQL degrades sharply with less data AND its Q-vs-realized bias grows ⇒ its off-data
  advantage is data-coverage-fragile (overestimation-flavored).
- FQL degrades gracefully with calibration intact ⇒ the off-data actions are robustly
  learnable; the support constraint's cost is real at every data scale.

### 2c (cheap add-on, 2 arms): the constraint's revenge test

On the 10% subset only: BC flow control + latentrl ε=0 instrument (1 seed each). If the
constrained method loses LESS than FQL loses going 100%→10%, we get the classic
pessimism story empirically: support constraints pay off when data is thin. That panel
would complete the tradeoff narrative (constraint costs at full data, protects at low
data). tmux `frac10_bc`, `frac10_latrl`.

## Phase B — Idea 1: two critics, hybrid steering (build after Phase 0 review)

Train BOTH successor measures sharing φ/w-space: the existing latent one ψ_u(s,w,u) and
an action-space one ψ_a(s,w,a) (FB/PSM-style, TD on dataset actions — reuse `agents/fb.py`
forward-map machinery so we inherit the audited recipe). Acting variants, pre-registered:

- **B-A (rank hybrid, no new action freedom):** sample K=32 latents → decode →
  score candidates with λ·Q_a(s,w,a) + (1−λ)·Q_u(s,w,u), execute argmax.
  λ ∈ {0, 0.5, 1}. Tests whether the action-space critic ranks decoded candidates
  better than the latent critic (it sees real action geometry; Q_u cannot).
- **B-B (steer hybrid = the residual, better informed):** B-A's winner + the bounded
  action-space gradient step: a = a* + ε·clip(∇_a Q_a, unit)·|a-scale|, ε from L1's
  best arm. This is L1's δ replaced by the explicit critic gradient — no learned
  residual net, so it composes with the zero-shot stack directly (Q_a is w-conditioned,
  still reward-free at train time).

Runs: joint training 2 seeds × 500k (tmux `hybrid_sd{0,1}`), then acting variants are
EVAL-TIME switches (cheap: one 500-ep eval per variant per seed). Gate to celebrate:
any variant ≥ 0.35 (clear of the 0.22 ceiling with CI margin). This is the idea most
likely to produce a zero-shot number that breaks the ceiling, because it adds the
value-directed step while keeping every component reward-free.

## Phase C — Idea 3: LIBERO / better behavior data (scoped feasibility FIRST)

Hypothesis: the interface ceiling is partly a PLAY-data property (rest-dominated,
winning actions off-support). Demonstration-quality data (LIBERO) should have higher
conditional coverage and a smaller off-support gap — possibly no ceiling at all.

**C-1 Feasibility audit (≤1 day, hard stop):** can LIBERO give us state-based obs +
continuous actions compatible with our pipeline (no image encoders)? Deliver a one-page
report: dataset format, obs modality options, action space, dataset size, integration
cost estimate. **If images-only or integration >2 days: STOP and report to the user —
do not start building.**

**C-2 (only if C-1 passes):** train the BC flow on one LIBERO suite → run D1 + the
coverage probe + the FQL-gap analog (a reference policy if one exists, else skip that
stat) → report whether demonstration data shows the same two-layer bottleneck. No RL
training until those probes are read.

## Sequencing

```
Day 1: Phase 0 (0a/0b/0c parallel) · 2a calibration · L1 keeps running
Day 2: review audit shortlist → launch hpmatch arm · 2b/2c arms · Phase B build
Day 3: Phase B runs + eval-time variants · C-1 feasibility · 2b readouts
Day 4: syntheses: audit doc, Idea-2 verdict, hybrid gate call, C-1 go/no-go to user
```

## Acceptance checklist

- [ ] HP diff table with per-row classification + ranked shortlist; hpmatch arm only
      after review; FB run-config confirmed or flagged
- [ ] Inversion re-cert JSON matches published card numbers (or stop-and-report)
- [ ] 2a calibration verdict with pre-registered interpretation; 2b/2c arms with
      off-support distance logged; no conclusions from in-loop evals
- [ ] Hybrid: both critics share φ; acting variants as eval-time switches; ≥0.35 gate
      before any porting/claims
- [ ] C-1 report delivered BEFORE any LIBERO build; hard stop honored
- [ ] L1 sweep untouched and its regime plot still delivered; tmux server killed

## Launched (2026-08-13, three-track execution)

- **A1 hpmatch**: `psmflow_hpmatch_20260813`, seeds 0/1, GPUs 0/1. Bundle as per the
  audit doc MINUS the sf 512x2 shape (PsiMap hard-asserts embedding_layers=2/
  hidden_layers=1 — a code change, out of scope for a launch arm; phi 512x4 IS applied).
- **A2 FQL calibration**: done — `logs/diag_fql_calibration_cube.json`. Verdict:
  calibrated with mild optimism (+2 to +8 on |G|~50, ratio 0.83–0.96); FQL 0.95 is not
  riding overestimation. Spearman ~0 is uninformative (92–95% episode success leaves no
  variance to rank). Tool patched to reach FQL's critic via network.select.
- **B1 dataset_fraction**: `envs/env_utils.py:subsample_episodes` (episode-level, seeded,
  pairing-safe) + config keys + main.py/precompute_preimages pass-through;
  `tests/test_dataset_fraction.py` (3 tests green).
- **B2 Idea-1 action branch**: `agents/psmflow.py` — psi_a(s,w,a) successor features
  over EXECUTED actions (shared phi, vector TD, gamma 0.99, tau 0.005, **pessimism 0.0
  by design** per the collapse forensics) + eps-bounded w-conditioned residual head;
  acting applies the residual when enabled. `tests/test_psmflow_action_critic.py`
  (3 tests green; shared branches byte-identical when disabled AND across the switch).
- **C3 hybrid runs**: `psmflow_hybrid_20260813` seeds 0/1 QUEUED behind hpmatch on
  GPUs 0/1 (tmux hybrid_sd{0,1} wait-loops), action_critic.enabled=true,
  residual_eps=0.05, save/eval interval 25k.
- **2b/2c chain** (GPU 3, tmux chain_frac10, sequential): FQL-10% (running) → its
  500-ep eval → 10% BC flow (fs100) → 500-ep BC eval → 10% preimages (alpha=20 N=200)
  → latentrl eps=0 → 500-ep eval. 2b's 25% arm DROPPED for now (3 GPUs; Idea 1
  preempts, per plan).
- **2c' zero-shot PSMFlow @10% (user priority addition, 08-13)**: `psmflow_frac10_20260813`
  seeds 0/1 QUEUED (tmux psmflow_frac10_sd{0,1}): LatentFlowPSM recipe on the SAME 10%
  episode subset (fraction seed 0), flow = bcflow_frac10, preimages = frac10 npz
  (pairing-guarded). sd0 runs on GPU 3 after the 2c chain completes; sd1 on GPU 0 after
  hybrid_sd1's slot frees. 500-ep evals -> eval500_psmflow_frac10_sd{0,1}.json, quoted
  against PSMFlow@100%, FQL@10%, BC@10%, instrument@10%.
- **frac50 chain (user addition 2, 08-13 evening)**: tmux chain_frac50, GPU 1, sequential:
  fql_frac50 -> 500-ep eval -> bcflow_frac50 (fs100) -> eval -> frac50 preimages
  (alpha=20 N=200, ~500k rows, overnight) -> psmflow_frac50 zero-shot -> eval. Same
  fraction-seed (0) as frac10. Final deliverable: logs/table_dataset_fraction.json
  ({FQL, PSMFlow, BC, instrument} x {10%, 50%, 100%}).
- **eval queue (08-13 evening)**: tmux evalq_20260813, GPU 1 shared: 500-ep finals for
  hybrid sd0/sd1 (gate check on sd1's in-loop 0.46) and hpmatch sd0/sd1.

# Getting the PSM to work — fix roadmap (2026-08-10, evening)

Context (from `2026-08-10-critic-diagnosis-and-baselines.md` results, commit `6f53b68`):
the LatentFlowPSM critic is dead — φ's best linear reward readout is R²≈0.13 (ridge
topline = closed-form, so inference is innocent), Q relief over u is 1.1% of |Q|, the
actor is 84%-driven by its BC terms, and D1 shows the critic cannot rank policies
(Spearman 0.10, p=0.78). Meanwhile **FB on the same env/budget/dataset reads 0.62–0.82**
(`fb_cube_ortho1000_20260810`, in-loop 50-ep @500k). So the machinery class works; our
PSM-style measure learning is what's broken.

**Priority set by the user: fix the PSM itself.** LatentFB (transplanting our latent
actor into FB) is the FALLBACK, triggered only if the evidence says the problem is not
in the PSM measure objective, or if two PSM fix rounds fail their gates.

Ground rules: identical to the previous two plans (§0 of `2026-08-10-iclr-figures.md`):
`.venv/bin/python`, named tmux + kill-server after, one training seed/GPU, 500-ep Wilson
or seed mean ± 95% CI, BC control quoted next to every Stage-C number, every tool writes
`report_out` JSON to `/data-local/amsks/PSMFLows/logs/`.

## Sequencing (decided 2026-08-10 after P0 returned Branch B)

P0 said Branch B, which pre-commits to promoting LatentFB. That promotion is HELD pending
two cheaper reads, because Branch B's inference ("the improvement loop is the
differentiator, so transplant our actor into FB's loop") carries an untested assumption:
that the latent action space can support a working improvement loop **at all**.

1. **P2 first.** It tests exactly that assumption in one night with no new architecture.
   Lands high (~0.7) ⇒ the latent space is innocent, build LatentFB with confidence, and
   P2's critic additionally calibrates what an alive latent critic's Q relief looks like —
   a number we do not currently have. Caps low (~0.3) ⇒ LatentFB would have capped there
   too, and the bottleneck is the frozen flow / latent action space, which is a different
   fix and a different paper sentence.
2. **Then read P1b** (backup exploration, already running). If it wakes the critic we stay
   inside PSM, which is the stated priority, and LatentFB stays parked.
3. **Then commit** to LatentFB or not.

Building P3 before P2 risks spending the build on a ceiling that a GPU-night already in
the plan as unconditional could have measured.

## P0 — the deciding probe: does FB's B carry the reward? (no training, ~1 h, DO FIRST)

Extend `tools/diag_task_projection.py` to accept `agent=fb` and read features from the
backward map B(s') of the trained FB checkpoints
(`/data-local/amsks/PSMFLows/exp/PSMFLows/fb_cube_ortho1000_20260810/sd00{0,1,2}_*`,
epoch 500000). Same protocol as the psmflow run: closed-form w = E[r·B], ridge topline,
held-out R², all three seeds. Also add the D3-style relief probe for FB: spread of
Q_FB(s,a,z) = F(s,z,a)ᵀz over ~512 dataset/actor actions at fixed (s,z), relative to
|Q| — measured on the same 64 states.

Output: `logs/p0_fb_basis_probe.json` with a `verdict` field. Decision:

- **Branch A — B reads R² ≫ 0.13 (say ≥ 0.4) and FB's Q has real relief (≥ 5%):**
  the load-bearing property is confirmed and OUR MEASURE OBJECTIVE IS THE DEFECT.
  Proceed to P1 (fix the PSM's φ/ψ training). LatentFB stays parked.
- **Branch B — B ALSO reads ~0.13 yet FB scores 0.72:** linear reward readout is NOT
  the lever; FB wins through its improvement loop (actor optimizes Fᵀz from step one).
  Then the PSM-internal fix is P1b (backup exploration + actor unshackling) rather than
  the loss swap, and **LatentFB is promoted** to the primary path (P3) because the
  working loop, not the representation, is what we lack.
- Ambiguous middle (B reads 0.2–0.4): run P1 anyway but gate it hard (see gates).

## P1 — fix the PSM measure learning (primary path, Branch A)

### P1a — TD measure loss for ψ/φ (the loss swap)

Add `measure_objective: contrastive | td` to `configs/agent/psmflow.yaml`
(default stays `contrastive` so nothing regresses silently). `td` implements the
FB-style objective transposed to our objects — ψ(s, w, u) as the F-analog (latent-action
-conditioned forward features), φ(s') as the B-analog:

- loss = E_{(s,u,s'), s_rand}[ (ψ(s,w,u)ᵀφ(s_rand) − γ ψ̄(s',w,u')ᵀφ̄(s_rand))² ]
  − 2·E[ψ(s,w,u)ᵀφ(s')], with u' = the actor's latent at s' (keep our backup), targets
  on both ψ and φ, stop-grad as in `agents/fb.py` — REUSE its loss code, do not
  re-derive. Keep `ortho_coef` as a separate additive regularizer but sweep it DOWN
  (1000 → 100 → 0 arms; FB does not carry our ortho pressure and its B works).
- γ: 0.98 → **0.99** in the same change (FB's value; our 50-step horizon is myopic for
  cube/antmaze episode lengths). Applies to both objectives.
- z_dim: keep 128, but note FB works at 50 — do not sweep this yet.

Tests: loss-shape test on synthetic batch; `measure_objective=contrastive` run is
byte-identical to current agent (same guard pattern as the actor.enabled tests).

### P1b — break the actor–critic co-collapse (bundled into every P1 run, no extra arms)

- `backup_explore_frac: 0.2` — fraction of TD bootstrap latents drawn u' ~ p0 (clipped)
  instead of the actor's output, so ψ is trained off the actor's mode. Config-gated,
  default 0.0.
- Keep bc_coeff=1.0 and pessimism 0.5 UNCHANGED in round 1 — one variable class at a
  time; the actor knobs only move in round 2 if the critic comes alive but the actor
  ignores it (D3's actor-percentile stat is the tell).

### P1 round-1 runs (cube only, the env with the FB reference)

| arm | overrides | seeds | tmux |
|---|---|---|---|
| td-ortho100 | `measure_objective=td ortho_coef=100 discount=0.99 backup_explore_frac=0.2` | 0,1 | `psmfix_td_o100_sd{0,1}` |
| td-ortho0 | same, `ortho_coef=0` | 0,1 | `psmfix_td_o0_sd{0,1}` |
| control: contrastive+bundle | `measure_objective=contrastive discount=0.99 backup_explore_frac=0.2` | 0,1 | `psmfix_ctr_sd{0,1}` |

6 runs × 500k, in-loop 50-ep evals. The control arm isolates whether the cheap knobs
alone move anything (if THEY do, the loss swap wasn't the fix and the story changes).

### P1 gates — measured on round-1 checkpoints BEFORE any celebration or scaling

Run the existing diagnostic trio on the best arm (they are all built):

1. `diag_task_projection`: held-out R² ≥ **0.5 × (FB's B R² from P0)** — the basis now
   carries the task signal.
2. `diag_q_landscape`: Q relative spread over prior draws **inside FB's measured band
   (2.3–3.1%)** — the original ≥5% bar was a guess that FB itself fails, so it cannot be
   the definition of an alive critic. Actor percentile in the Q distribution ≥ 0.7 (it
   exploits whatever relief exists).
3. `diag_policy_ranking` (judge = new ckpt, same 10-policy set): Spearman ≥ **0.6**.
   **This is the load-bearing gate.** Relief is necessary but not sufficient — a critic
   with spread that still cannot order policies is no more usable than a flat one, and
   R^2 of a linear read-out has already been shown (P0) not to track success at all.
4. Only if 1–3 pass: 500-ep evals, quote vs BC 0.068 and FB's 500-ep numbers (which
   the leftover-evals WP below produces).

Pass ⇒ scale the winning arm to 5 seeds + antmaze, refresh figures, and the paper's
story becomes "diagnosed the failure, fixed the objective, closed the gap to X".
Fail after round 1 ⇒ ONE round-2 iteration (actor knobs per D3's tell, or ortho/z_dim)
with the same gates. Fail round 2 ⇒ **stop fixing PSM, promote P3 (LatentFB)** — the
decision is pre-committed here so we don't sink the deadline into a dead objective.

## P1c — φ-grounding auxiliary (only if P0 lands Branch A but P1a fails its gates)

HILP-flavored temporal-distance grounding: auxiliary loss shaping φ so that
‖φ(s) − φ(s')‖ tracks reachability (in-trajectory time offsets as targets, reward-free).
Spec'd deliberately thin — do not build this until the loss swap has actually failed,
and write a short design note first (it interacts with the ortho term).

## P2 — per-task latent control (DSRL-style ceiling; unconditional, parallel, 1 GPU)

The preimage npz already makes the dataset a LATENT-action dataset (u* per transition).
Run offline per-task RL in latent space: critic Q(s,u) trained by TD on the dataset's
task reward over (s, u*, s', u*'), actor = latent policy max Q + BC-to-u*, actions
deployed through the frozen flow decode. Implement as a thin variant inside
`agents/psmflow.py` or a small new agent reusing FQL's critic machinery — whichever is
less code; it needs NO ψ/φ/w. Cube, 2 seeds × 500k, tmux `latsac_cube_sd{0,1}`,
500-ep evals.

Reading: near FQL (0.95) ⇒ the latent space + frozen flow are innocent; all blame on
zero-shot representation (sharpens the paper's diagnosis AND bounds P1's headroom).
Caps well below FQL ⇒ the support constraint itself costs performance — the paper must
say so, and P1's realistic target is this number, not FQL's.

## P3 — LatentFB (FALLBACK; build only on Branch B or after P1 round-2 failure)

Transplant the latent actor into the working FB agent: keep F/B training verbatim
(`agents/fb.py`, γ=0.99, their z-mixing), actor emits u, action = frozen-flow decode
(reuse psmflow's actor + decode path). New agent `fb_latent` registered alongside; cube
2 seeds first, gates 2–4 from P1 apply. Rationale for deprioritizing: the user's goal
is a working PSM; this path abandons the PSM measure loss rather than fixing it, so it
runs only when the evidence says that loss is not fixable in-deadline.

## Leftover mechanical work (from the previous plan — finish regardless, fills idle GPUs)

- PSM cube: let sd1 finish, launch sd2, then 500-ep evals ×3 (`psm_cube_flow_20260810`).
- 500-ep evals: FB cube ×3, antmaze latentpsm sd1/sd2 (trained, unevaluated).
- Refresh `fig_actor_comparison` (FB + PSM bars, 3-seed antmaze) and the F4 calibration
  panel with the D1 policy-ranking result; HANDOFF entry for the diagnosis verdicts +
  this roadmap.

## Schedule (GPU-nights)

```
Night 1: P0 probe (CPU/1 GPU, 1 h) → branch decision · leftover evals · P2 build
Night 2: P1 round-1 (6 runs, 3–6 GPUs) · P2 runs (1 GPU)
Day 3:   P1 gates on round-1 ckpts → scale winner / round 2 / promote P3
```

## Acceptance checklist

- [ ] `p0_fb_basis_probe.json` with verdict + branch called in HANDOFF
- [ ] `measure_objective=td`, `backup_explore_frac`, γ=0.99 implemented, config-gated,
      contrastive default byte-identical (test pinned)
- [ ] Round-1: 6 runs complete; gates 1–3 measured and reported as JSON before any
      500-ep eval is launched
- [ ] P2 latent per-task control: 2 seeds + 500-ep evals, number quoted vs FQL 0.949
      and BC 0.068
- [ ] Pre-committed stop rule honored: ≤2 PSM fix rounds before P3
- [ ] Leftover evals + figure refresh done; tmux server killed

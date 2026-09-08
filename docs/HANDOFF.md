---
marp: true
title: PSMFlows — Lab Record
theme: default
paginate: true
---

<!-- _class: lead -->

# PSMFlows — Lab Record

Current work: **PSMFlow v1** (`docs/plans/2026-07-20-psmflow-v1.md`) — **Tasks 1–8 code
complete**; the pipeline is now operator-driven (GPU runs + gates), with one open method
call on the D3 ESS gate. The **affine PSM** cube push (07-22 → 07-26, below) is **PARKED**,
not closed: the 07-26 entry's Findings 3 and 4 (rectangular measure mesh, split
x-/(s,a)-branch) are still the open leads if we return to it. The older bilinear-PSM parity
hunt (2026-07-07 → 07-15) is **CLOSED** — see the 07-13 entry and `PAPER/RESEARCH_NOTE.md`
§4: no code bug, the gap was seed variance + a training-budget ceiling.

Branch: `feat/inversion-integration` · Machine: `kisski` (GWDG, SLURM/H100) · prior: `midi-01` (UT CS)
Date: **2026-09-08** (latest) · prior: 2026-09-07, 2026-09-06, 2026-09-05, 2026-09-04, 2026-09-03, 2026-09-01, 2026-08-31, 2026-08-30, 2026-08-29, 2026-08-14, 2026-08-13, 2026-08-12, 2026-08-10, 2026-08-05, 08-04, 07-29, 07-28, 07-26, 07-15, 07-13, 07-07

---

<!-- _class: lead -->

## 2026-09-08 — the measure-loss verdict, the K sweep, and the first faithful DSRL arm

---

### 1. Measure-loss audit: verdict filled

`docs/design/2026-09-08-measure-loss-audit.md` §8. All nine pre-registered arms landed.
Growth refitted on a **common 10k-150k window** (the §7 numbers were 50k-500k and must not
be read against these). antmaze `medium-navigate`, gamma=0.995:

| arm | kappa_measure | steps/decade | log10 |Q| growth |
|---|---|---|---|
| `mla_g995_pess10` | 1.0 | 1.5e4 | 4.15 |
| baseline | 0.5 | 5.18 / 4.25 / 3.84e4 | 1.25-1.78 |
| `mla_g995_pess025` | 0.25 | 8.0e4 | 0.73 |
| `mla_g995_pess0` | 0.0 | 1.08e5 / 1.95e5 | 0.45 / 0.33 |

**H-PESS confirmed** -- monotone in `pessimism_penalty` across four dose levels spanning
13x, against 35% baseline seed scatter. Orthonormality delays only (`orel` within scatter
seed-for-seed). **The pre-registered structural fix is REFUTED on cube**: kappa_measure=0
scores 0.136 vs 0.415, while antmaze's 0.334 vs 0.252 is inside overlapping CIs. Default
stays at 0.5.

**`psi_bound` is settled negative** -- best growth number measured anywhere (11x baseline)
and the worst policy (early success 0.096-0.256 vs baseline 0.288-0.344). Growth rate is
not a proxy for policy health.

**Ensemble spread carries no support information**: prior/in-support disagreement ratio is
1.00 across eight checkpoints. `kappa*spread` is a uniform one-signed shift, not a
support-aware penalty.

---

### 2. Dead seeds are a SELECTION failure

`$PSM_DATA/logs/diag_deadseeds/`. Over the candidate roster, Spearman(Q, MC return) is
**0.06-0.16 in every arm including healthy seeds**, with a negative 10th percentile --
on a tenth of states the ranking is inverted. Meanwhile **79% of decoded prior candidates
succeed under MC**, identically in every seed. The flow offers a good action nearly
everywhere; the measure cannot pick it.

---

### 3. `gpi_num_u` sweep (new, `scripts/slurm/launch_gpi_num_u_sweep.sh`)

500 episodes, frozen checkpoints, only `agent.gpi_num_u` overridden. Positive control
(K=8 on the dead seed) reproduced 0.134 against the 0.140 on record before the rest ran.

| checkpoint | K=4 | K=8 | K=16 | K=32 | K=64 | best |
|---|---|---|---|---|---|---|
| cube healthy sd0 | 0.274 | 0.488 | **0.514** | 0.452 | 0.312 | K=16 |
| cube healthy sd1 | 0.132 | 0.244 | 0.334 | 0.340 | **0.374** | K=64 |
| cube dead (`mpess0`) | **0.218** | 0.134 | 0.078 | 0.012 | 0.004 | K=4 |
| antmaze g99 sd2 | 0.350 | 0.394 | 0.458 | 0.482 | **0.510** | K=64 |

**The optimum is per-seed, not just per-domain, and there is no single right K.** The dead
seed collapses **54x** from K=4 to K=64 -- max over a critic it cannot rank, i.e. max over
noise. Healthy cube sd0 is an inverted U peaking at 16 and losing a third by 64. Healthy
cube sd1 and antmaze are monotone increasing out to 64.

Shared-K comparison (one hyperparameter, no per-seed picking): on the two healthy cube
seeds K=16 gives 0.424 against K=64's 0.343. But antmaze prefers 64 (0.510 vs 0.458).
So cube and antmaze want different K, which is what the literature reports. **Two seeds per
env; cube sd2 was not swept. Do not change the default on this alone.**

Note antmaze here is monotone INCREASING, the opposite of Diffusion-DICE's antmaze result.
Its seed is the healthiest we have, which fits the pattern that K-tolerance tracks critic
quality rather than the domain.

Literature agrees there is no consensus: published N spans 1-400, tuned per task, and
FQL's own IDQL/IFQL rejection-sampling baselines use **32 for every OGBench task including
cube-single-play**. Diffusion-DICE (arXiv:2407.20109) Table 4 shows an 82.0 -> 51.8
monotone collapse on antmaze-umaze-diverse over K=1..16. The argmax's optimization
pressure is `log N - (N-1)/N` nats, exact for continuous latents: K=64 is 3.16 nats.

---

### 4. What we called DSRL was not DSRL

External review, verified against the code. Two findings:

**`dsrl_na` is not DSRL-NA.** DSRL-NA is a dual-critic scheme: an ACTION-space critic Q_A
trained on real transitions, distilled into the latent critic, exploiting the fact that
many latents decode to the same action. Ours regresses the actor onto the argmax of 16
prior draws scored by the GPI rule -- candidate-argmax behaviour distillation, the
SfBC/IDQL family. **Renamed `gpi_distill`**, with `dsrl_na` kept as a read-time alias so
old flags.json files still restore. `psi_a(s, w, a)` in the action branch already IS the
action-space critic real NA would need.

**The default arm structurally cannot do SAC.** Under `policy_index=latent`,
`sample_step_inputs` computes the actor's bootstrap latent and then overwrites it with the
prior draw u'. psi is the successor measure of the constant-latent policy pi_{u'}, which
is what makes GPI well-posed -- and means the backup never contains the actor's action. An
actor trained against it is doing policy improvement against a value function that is not
Q^pi for any pi it converges to. The assertion chain closes the escape hatch:
`gpi_distill` -> `index_panel>0` -> `policy_index=latent`.

**`psi_form=affine` and a faithful DSRL arm are mutually exclusive.** Now stated in the
module docstring; the config space does not enforce it.

**The paper is unaffected.** `table_headline.tex`'s DSRL-SAC rows (0.183, 0.127) come from
the archived `latentrl` agent, whose backup does use the actor's latent
(`archive/agents/latentrl.py:176,182`). Only the psmflow ablations were mislabelled, and
no published number depends on them.

**One review claim did not hold.** The BC anchor was said to cancel the steering at
`bc_coeff=1.0`. Every shipped DSRL run used `bc_coeff=0.0` (14 `gpi_distill` + 18
`dsrl_sac`); the `bc_coeff=1.0` runs are the 33 older ddpg-era ones. So the plateau
happened with no BC pull, and the actor's 1.70-vs-2.13 contraction is regression-to-mean
onto a re-randomised target, not a constraint.

---

### 5. Code changes (all default-off; published numbers unchanged)

- `actor_mode=gpi_distill`, `dsrl_na` aliased on read (`ACTOR_MODE_ALIASES`).
- `MEASURE_DEFAULTS` now also backfills `index_panel`, `expectile_mu`,
  `mask_invalid_preimages`. (The earlier gap: `psi_form`/`index_agg`/`gpi_select`/
  `index_clip` were guarded with `.get(legacy)` in `create` but read unguarded at runtime;
  both `index_clip` tests were failing before this landed.)
- **Mode-based eval.** `sample_actions` ignored its `temperature` argument, so the DSRL
  arms were evaluated stochastically while `utils.evaluation` passes `eval_temperature=0`.
  Now returns `u_clip * tanh(mu)` at temperature 0, which is how SAC is normally evaluated.
  **Every DSRL number on record was a stochastic-policy eval.**
- **`mask_invalid_preimages`** (default false). `preimage_valid` was recorded by
  `repair_invalid_preimages` and never consulted, so diverged rows trained the measure on
  (u, a) pairs where u does not decode to a. `contrastive_loss` gained an optional
  `row_weight`; verified to reproduce the published loss bit for bit at weight 1.
  **Scale check: cube has 13 diverged rows in 1,000,000 (0.0013%)**, so on this dataset the
  flag is correctness hygiene and will not move a number. It matters only if an inversion
  is ever run at settings that diverge more.
- **`actor.prior_init`** (default false). Default init gives per-dim latent std **1.88** at
  u_clip=3 against the flow's prior 1.0, so the policy started outside the support the flow
  was fitted on. `prior_init` zeroes the mu head and biases log_std to log(0.3).
- `flow_steering.py`: jitted both `vmap`s (bare vmap over the proposal reproducibly returns
  all-NaN on GPU), seeded the mixture draw from the passed rng instead of global numpy
  state, and clamped it to `u_clip`.
- Corrected the `u_data` clip comment: its justification does not hold under
  `policy_index=latent`, where the bootstrap is a prior draw.

---

### 6. RUNNING: the first faithful DSRL arm

`scripts/slurm/launch_dsrl_faithful.sh`, cube, 3 seeds x 2 boxes, 500k.

    actor_mode=dsrl_sac policy_index=task_vector psi_form=free index_agg=max
    actor.index_panel=0 train_actor=true acting=actor
    actor.bc_coeff=0 actor.prior_init=true

`policy_index=task_vector` is the one index under which `u_next` survives into the backup,
so this is the first arm where the latent MDP is genuinely on-policy. It gives up
Prop. `bilinear` by necessity.

Two boxes so the structural change and the box change are separable: **u_clip=1.5 and 3.0**.
DSRL recommends `b_W` in **[0.5, 1.5] for offline** and [1, 3] for online; every run in this
repo has used 3.0, the top of the *online* range (150 runs at 3.0, 2 at 1.0). Their decoder
also trains with `randn_clip_value: 3`, so 3.0 sits at the edge of trained support with no
margin.

**Read it against the BC control (0.072), not against actor-free GPI.** The question is
whether steering works at all on this substrate, not whether it beats GPI.

#### RESULT (landed same night, 500 episodes)

| arm | sd0 250k/500k | sd1 250k/500k | sd2 250k/500k | pooled |
|---|---|---|---|---|
| u_clip=1.5 | 0.100 / 0.018 | 0.174 / 0.298 | 0.162 / 0.064 | **0.136 +/- 0.079** |
| u_clip=3.0 | 0.102 / 0.116 | 0.250 / 0.026 | 0.274 / 0.076 | **0.141 +/- 0.079** |

against BC **0.072**, `gpi_distill` **0.307 +/- 0.091**, actor-free GPI **0.415 +/- 0.083**.

**Fixing the backup did not rescue steering.** The structural criticism was right -- under
policy_index=latent the actor's action never enters the Bellman target, so `dsrl_sac` was
not SAC -- but making the latent MDP genuinely on-policy yields ~0.14, *below* the
mislabelled `gpi_distill` and a third of actor-free GPI, with a CI whose lower edge is
under the BC control. The backup was not the binding constraint.

**No box effect.** The in-loop 50-episode read suggested u_clip=1.5 beat 3.0 by 2.2x
(0.184 vs 0.085) and that was reported as empirical support for DSRL's offline
b_W in [0.5, 1.5]. It does not survive 500 episodes: 0.136 vs 0.141, indistinguishable.
**Recorded so the claim is not revived from the in-loop curves** -- and as one more instance
of why the 50-episode evals are not reportable.

**Training past 250k hurts this arm.** u_clip=3.0 pools 0.209 at 250k against 0.073 at
500k; per-seed swings are 0.250 -> 0.026 and 0.100 -> 0.018. n=3, but it matches the
instability seen everywhere else.

**Interpretation.** Every thread this session converges on the same place: the critic
cannot rank (Spearman 0.06-0.16, negative 10th percentile) while 79% of decoded prior
candidates succeed. An actor climbing that critic has nothing to climb, on-policy backup or
not. The ranking objective, not the backup and not the box, is the binding constraint.

Verified directly from `ajwagen/dsrl` configs: `target_ent: 0.0` is held constant across
`action_magnitude` 1.0 / 1.5 / 2.5, so it is NOT tied to a box size (an earlier claim in
this record that it was b=1.5-specific is withdrawn). Our log-prob is computed in the
squashed-but-unscaled space, so our entropy target is box-invariant and comparable.

---

### 7. RUNNING: the stability campaign -- oscillation is the blocker, not ranking

`docs/design/2026-09-08-oscillation-stability.md` (pre-registered); scored by `tools/stability_ladder.py`. Every cube seed already
reaches 0.6-0.7 and none of them holds it: per-seed maxima **0.704 / 0.620 / 0.596**, mean
of maxima **0.640**, against a pooled 300-500k figure of **0.415**. Seed 0 traverses
0.532 -> 0.286 -> 0.704 -> 0.282 -> 0.272 -> 0.086 at 500 episodes, where sampling noise is
+/-0.04. The capability is present; the consistency is not.

Correlating every logged training metric against 500-episode success over 28
(seed, checkpoint) pairs: **`orth_offdiag` is the strongest predictor at -0.375** (tighter
orthonormality, better policy) and **`psm_loss` is +0.079, i.e. nothing.** Standing warning:
on this method loss diagnostics are not a proxy for policy quality -- the same lesson
`psi_bound` taught.

The ortho settled-negative does NOT cover this. It was measured against loss-growth rate at
gamma=0.995; `ortho_coef=1e4/1e5`, `lr_sf=1e-5` and `tau=1e-3` have never been scored on
cube success at gamma=0.98.

Four arms, 3 seeds each, checkpoints every 50k, control = existing `affine_strict_cube`:
`tau1e3_cube`, `oc1e4_cube`, `lrsf1e5_cube`, `oc1e4_lrsf1e5_cube`.

**Scored on stability, not the mean** -- per-seed swing and mean step over the 300-500k
ladder, reported beside the pooled mean. An arm at 0.45 flat beats an arm averaging 0.70
and 0.09. Control swing to beat: 0.618 / 0.222 / 0.254.

Also corrected here: at a good checkpoint the critic is worth ~8x over random
(argmax 0.704, mean 0.734, small_ball 0.670 vs every critic-free rule at ~0.08). The
"critic picks worse than random" result in 8.4 came from the MC probe's held-fixed-latent
regime, which is not what GPI deploys.

---

### 8. Next

1. **Read the faithful-DSRL result against BC.** If it does not clear 0.072 the problem is
   upstream of steering.
2. **Finish the K sweep** -- cube sd2, and antmaze K=64. Then decide on `gpi_num_u`.
3. **The ranking problem is the central question.** "Good Rankers, Bad Objectives"
   (arXiv:2607.27422) studies our exact head -- a bilinear contrastive critic scoring
   best-of-K -- and finds, parameter-matched, contrastive training ranks at Kendall
   tau ~0.40 against Bellman training's ~0.70. Failure attributed to the OBJECTIVE, not
   the bilinear form; contrastive constrains the angle and leaves norm growth off-support
   unconstrained. Their fix is two heads: contrastive for retrieval, a **Bellman-trained**
   scalar for selection. Note `q_dist` is expectile-distilled from `psi^T w`, so it
   inherits the contrastive ranking rather than fixing it.
4. Real DSRL-NA on `psi_a` if a DSRL comparator is wanted -- DSRL-NA, LPS
   (arXiv:2603.05296) and QPILOTS all independently moved scoring into action space.
5. `kappa=0.25` on cube 0.98 + antmaze 0.99 -- still the only rate benefit not shown to
   cost cube.

---

<!-- _class: lead -->

## 2026-09-07 — actor ablation: DSRL-NA gets 74 % of the actor-free ceiling, and it is not the support

---

### The table (`docs/tables/actor_ablation.md`, generated)

500-episode evals, `EVAL_WORKERS=1`, pooled over 300k–500k × 3 seeds, mean ± 95 % CI.
Substrate identical across arms (`psi_form=affine policy_index=latent index_agg=max
u_clip=3.0`, same frozen flow and preimages); only the actor and acting rule differ.

| env | arm | late mean | n | BC |
|---|---|---|---|---|
| cube | actor-free (GPI) | **0.415 ± 0.083** | 15 | 0.072 |
| cube | **`dsrl_na`** | **0.307 ± 0.091** | 15 | 0.072 |
| cube | ddpg latent actor (prior arm) | 0.146 | 2 | 0.072 |
| cube | `dsrl_sac` | 0.108 ± 0.053 | 15 | 0.072 |
| cube | **`prior_shrunk` control** | **0.059 [0.048, 0.072]** | 1500 ep | 0.072 |
| antmaze γ=0.98 | actor-free | 0.067 ± 0.036 | 7 | 0.072 |
| antmaze γ=0.98 | `dsrl_sac` | **0.000** | 3 | 0.072 |
| antmaze γ=0.99 | actor-free | 0.252 ± 0.100 | 15 | 0.072 |
| pointmaze | actor-free | 0.000 | 15 | 0.002 |
| pointmaze | `dsrl_sac` | **0.000** | 3 | 0.002 |

---

### What was wrong with the old latent actor

Audit: `docs/design/2026-09-06-dsrl-actor-audit.md`. The shipped actor climbed
`psi(s,u,u')^T w` at **one** random policy index per batch element, while acting takes
`max_{u'}` over 64. Measured on `affine_actor_cube` @500k with the new
`tools/diag_actor_grad_terms.py`: **cos(∇q_single, ∇q_panel_max) = −0.49 / −0.20** — the
actor's gradient was *anti-correlated* with the objective GPI maximises. BC domination on
the affine critic is 2.8–3.6 : 1 (milder than the free-psi 5 : 1 of 09-03).

The critic is untouched by the actor under `policy_index=latent` (`u_next = u_index`), so
`affine_actor` is the actor-free critic with a different acting rule bolted on — 0.415 vs
0.146 is a clean measurement of the acting rule alone.

Fix: `actor.index_panel=K` makes the actor climb the same panel max, nearly free under the
affine head (A(s,u) does not see u'). New `actor_mode: ddpg | dsrl_sac | dsrl_na`.

---

### The two decisive controls

**`prior_shrunk`** (new `gpi_select` mode): one prior draw scaled to the NA actor's own
operating radius (`u_norm` 1.72 vs prior median 2.13), reading neither `psi` nor `task_z`.
Its checkpoint-invariance cell returned **28/500 at both 300k and 500k, byte-identical**,
validating it as critic-free. It scores **0.059, at BC** → `dsrl_na`'s 0.307 is genuine
state-dependent behaviour, **not** "shrink your latents". Pre-registered prediction held.

**Ranking probe** (`tools/diag_actor_vs_gpi.py`, 256 states × 64-candidate roster):
`dsrl_na` sits at roster percentile **0.614** (top-quartile fraction 0.369 vs 0.250 chance);
the ddpg actor at **0.533** (0.248 — chance). The ordering matches success (0.307 > 0.146),
so the distillation is real but **weak**. Both actors are *farther* from the roster argmax
than from a typical candidate (ratio 1.22 / 1.25) — neither approaches it in latent space.
The specified Spearman metric is **confounded and uninformative**: ≈0.97 vs the roster max
*and* ≈0.99 vs the roster mean, for both arms, i.e. it measures per-state Q scale.

---

### Two negatives, both pre-registered

**`dsrl_na`'s regression plateaus.** Distance explained vs the independence floor
`(‖u_a‖²+‖u*‖²)/d_a`: cube **6.6–7.2 %**, antmaze 7.5–10.4 %, pointmaze 13.9–18.3 % — nine
seeds, three envs, flat from 25k to 500k. The 25 % gate is missed everywhere. So the arm
scores 0.307 **without** distilling the argmax and **without** riding the support.
What it tracks is unidentified and is the most interesting object here.

**`dsrl_sac` is 0.000 on both mazes** (0/1500 each) and 0.108 on cube. Cause: `actor_u_norm`
is **~2.0× the prior radius on all three envs** (d_a = 2, 5, 8) and `actor_q` diverges
(pointmaze sd1: **2.9e7**). `target_entropy=0` is DSRL's value for a box of **b=1.5**; ours
is `u_clip=3.0`. **These rows measure the port, not DSRL-SAC.** A corrected arm
(`affine_dsrl_sac_te_cube`, `target_entropy=−d_a·log(u_clip/1.5)`, `u_clip` unchanged so the
critic's training distribution is identical) is running — SLURM 2492091–2492093.

---

### Pre-registration scorecard

| hypothesis | outcome |
|---|---|
| H-A level (`dsrl_na` cube ≥ 0.25) | **met** (0.307) |
| H-A stability (std < 0.12) | **refuted** (0.180; per-seed 0.205/0.237/0.479) |
| H-B (`dsrl_sac` at/near BC) | **met**, and worse than registered |
| `prior_shrunk` at BC (0.05–0.12) | **met** (0.059) |
| `dsrl_na` NA-gate ≥ 25 % explained | **failed** (6.6–7.2 %) |

**Discipline note.** Two over-readings were made *while these numbers landed* and are
recorded in the design doc: "striking" after 2 of 15 cells (the next three were 0.09–0.12),
and "every seed peaks at 300k and collapses" after 2 of 3 seeds (seed 2 rises monotonically
to its best value at 500k). Both were generalisations from n = 2. Only the pooled row is
quotable.

---

### Incident: 12 evals killed by a py/yaml skew

The `actor.entropy` guard landed in `agents/psmflow.py` a few minutes **before** the key
landed in `configs/agent/psmflow.yaml`. Twelve queued evals of *older* checkpoints died at
`create()` with `KeyError: 'entropy'` (SLURM 2491907, 2491909–2491916, 2491919–2491921);
they need resubmitting, nothing else was lost. Separately, 2491881–2491889 (exit 126) and
2491924 (exit 2) failed on `scripts/eval500.sh` shell bugs *after* writing their reports —
different owner, results usable.

Root cause generalised: **a restored checkpoint's `flags.json` is older than the code
restoring it, so every `actor` key added from now on must be optional at read time.** Fix:
`fill_actor_defaults()` is now the first statement of `create()`. Regression test
`tests/test_psmflow_config_compat.py` (9 cases) drives three *verbatim archived*
`flags.json` fixtures through the eval tool's own `merge_run_config` + `create`, and asserts
**yaml ↔ `get_config()` leaf-key parity** — the one assertion that would have caught it
(negative control run: deleting `entropy` from the yaml fails it).

---

### Artifacts

- `docs/design/2026-09-06-dsrl-actor-audit.md` — audit, pre-registrations, all findings.
- `docs/tables/actor_ablation.md` — generated; `ACTOR_ABLATION` appended to `make_tables.py`.
- New: `tools/diag_actor_grad_terms.py`, `tools/diag_actor_vs_gpi.py`,
  `utils/psm_networks.py::{TanhGaussianLatentActor, tanh_gaussian_sample, LogAlpha}`,
  `gpi_select=prior_shrunk`, `tests/test_psmflow_{dsrl_actor,config_compat}.py`.
- Runs: `affine_dsrl_{na,sac}_{cube,antmaze,pointmaze}` 2491932–2491949 (six plain-NA maze
  runs cancelled at 30–40k, data salvaged); g99 arms 2492040–2492045; corrected SAC
  2492091–2492093; probe 2492090.
- Tests: 138 passed / 2 skipped over 14 modules, module-per-process.

---

<!-- _class: lead -->

## 2026-09-06 — three-seed affine ladder + baselines

---

## Affine strict, cube: **a third seed does not settle it** — the ladder is
## 0.086 … 0.704 over 30 measurements and the swing is within-run, on every seed

Seed 2 of `agent=psmflow` (repo default = paper-strict affine: `psi_form=affine
policy_index=latent train_actor=false acting=gpi`) was trained to 500k on both envs with a
config verified byte-equal to sd000's `flags.json` (only `seed`, `run_group` and the three
eval-only `gpi_select` keys added 09-05 differ), and the cube ladder was completed for all
three seeds: **every 50k checkpoint from 50k to 500k, 500 episodes each, 30 cells, no gaps.**

Table: `docs/tables/affine_cube_ladder.md` (generated).
Figure + series: `docs/figures/2026-09-06-affine-cube-ladder.{png,json}`
(generator `tools/fig_affine_cube_ladder.py`; the 09-05 two-seed figure is left in place).

---

## The full cube ladder (500 episodes per cell, Wilson 95%)

| epoch | sd0 | sd1 | sd2 | mean ± std |
|---|---|---|---|---|
| 50k | 0.192 [0.160, 0.229] | 0.412 [0.370, 0.456] | 0.224 [0.190, 0.263] | 0.276 ± 0.119 |
| 100k | 0.078 [0.058, 0.105] | 0.330 [0.290, 0.372] | 0.178 [0.147, 0.214] | 0.195 ± 0.127 |
| 150k | 0.088 [0.066, 0.116] | 0.268 [0.231, 0.308] | **0.596** [0.552, 0.638] | 0.317 ± 0.258 |
| 200k | 0.264 [0.227, 0.304] | 0.298 [0.260, 0.340] | 0.466 [0.423, 0.510] | 0.343 ± 0.108 |
| 250k | 0.532 [0.488, 0.575] | **0.620** [0.577, 0.661] | 0.246 [0.210, 0.286] | 0.466 ± 0.196 |
| 300k | 0.286 [0.248, 0.327] | 0.532 [0.488, 0.575] | 0.360 [0.319, 0.403] | 0.393 ± 0.126 |
| 350k | **0.704** [0.663, 0.742] | 0.494 [0.450, 0.538] | 0.350 [0.309, 0.393] | 0.516 ± 0.178 |
| 400k | 0.282 [0.244, 0.323] | 0.346 [0.306, 0.389] | 0.546 [0.502, 0.589] | 0.391 ± 0.138 |
| 450k | 0.272 [0.235, 0.313] | 0.568 [0.524, 0.611] | 0.564 [0.520, 0.607] | 0.468 ± 0.170 |
| 500k | 0.086 [0.065, 0.114] | 0.548 [0.504, 0.591] | 0.292 [0.254, 0.333] | 0.309 ± 0.232 |

| pooled window | n (ckpt × seed) | mean | std | 95% CI | min | max |
|---|---|---|---|---|---|---|
| **300k–500k** | 15 | **0.415** | 0.164 | ± 0.083 | 0.086 | 0.704 |
| **250k–500k** | 18 | **0.424** | 0.164 | ± 0.076 | 0.086 | 0.704 |
| BC control | — | 0.072 | — | — | — | — |

---

## What the third seed changed, and what it did not

**Did not change the headline.** The 250k–500k pooled mean moved 0.424 → 0.424 (the interval
tightened, ± 0.141 → ± 0.076, purely from n 10 → 18). The 09-05 claim stands verbatim: this
arm is ~0.42 as a *mean over late checkpoints*, ~6× BC, and **no single checkpoint of it is
reproducible to ± 0.1**.

**Settled the "is sd0 the broken one?" question: no.** Seed 2 oscillates like seed 0, not
like seed 1. Per-seed spread over the ten checkpoints (min → max, largest jump between
*adjacent* 50k checkpoints):

| seed | min | max | range | sd | largest adjacent jump | mean ≥300k |
|---|---|---|---|---|---|---|
| 0 | 0.078 | 0.704 | 0.626 | 0.202 | **0.422** (350k→400k) | 0.326 |
| 1 | 0.268 | 0.620 | 0.352 | 0.126 | 0.322 (200k→250k) | 0.498 |
| 2 | 0.178 | 0.596 | 0.418 | 0.152 | **0.418** (100k→150k) | 0.422 |

Seed 2's 0.178 → **0.596** → 0.466 → 0.246 over four consecutive checkpoints is the same
±0.4 within-run swing sd0 shows, with Wilson intervals ±0.04 wide — disjoint by a wide
margin, so it is training-time non-stationarity on **two of three seeds**. Seed 1 is the
outlier for being *calm*, not seed 0 for being wild. Its own late mean (0.498) is the best
of the three, so "pick the stable seed" and "pick the good seed" happen to coincide here —
which is exactly the trap the pooled row exists to avoid.

**No checkpoint-selection rule is available from the logs.** The in-loop 50-episode eval
tracks the 500-episode ladder only loosely (middle panel of the figure), and
`w_enc_spread` — the policy-encoder collapse diagnostic — decays smoothly to the same place
on all three seeds (peak ~1.31 → 0.36 / 0.25 / 0.31 at 500k) while success swings ±0.4
around it. The onset differs (it first crosses 0.6 at 190k / 215k / **305k** for sd0 / sd1 /
sd2) but the ordering does not match the success ordering: sd2 collapses *latest* and is
neither the best nor the worst arm. The encoder collapse is real and monotone; it does not
predict which checkpoint, or which seed, is good.

---

## Antmaze, seed 2: still nothing

| epoch | sd0 | sd1 | sd2 |
|---|---|---|---|
| 50k | 0.112 [0.087, 0.143] | 0.000 [0.000, 0.008] | — |
| 250k | — | 0.178 [0.147, 0.214] | — |
| 300k | — | — | 0.088 [0.066, 0.116] |
| 350k | 0.082 [0.061, 0.109] | — | — |
| 400k | 0.056 [0.039, 0.080] | 0.134 [0.107, 0.167] | — |
| 500k | 0.002 [0.000, 0.011] | 0.096 [0.073, 0.125] | **0.008** [0.003, 0.020] |

Seed 2 was evaluated at 500k and at its single best in-loop checkpoint (300k, in-loop 0.40).
**500k = 0.008, i.e. the third seed also decays to the floor**; 2 of 3 seeds now end below
the BC control (0.072). The late-checkpoint mean (≥250k, 8 measurements) is **0.081 ± 0.050**
against BC 0.072 — the interval still contains BC, so antmaze remains **not a result**.

The in-loop eval is worse than useless here: 300k read **0.40 over 50 episodes** and
**0.088 over 500**. Any checkpoint picked off `eval.csv` on this env is picking noise.

---

## Provenance / caveats for the two entries above

* Every cell is `tools/eval_checkpoint.py`, 500 episodes, agent config inherited from the
  run's own `flags.json` (no arm flags on the CLI). SLURM 2491687-94 (sd0/sd1 gap fill),
  2491730-2491798 (sd2 cube ladder), 2491799-2491800 (sd2 antmaze).
* Training: cube sd2 02:41:50, antmaze sd2 04:05:51, matching sd0/sd1 to a few minutes.
* The sd0/sd1 **100k and 250k** JSONs carry `acting_mode: "decode(actor latent)"`. That is
  the pre-09-05 **label** bug only — `acting: gpi` in the same JSON, and the shipped
  `gpi_select=argmax` code path is unchanged by the 09-05 ablation commit (the ablations
  branch out before it). The numbers are comparable; the string is not.
* `tools/make_tables.py` now carries one row per cube checkpoint 50k–500k plus a
  300k–500k pooled row, and an antmaze @300k row; the `sd?` globs already span three seeds.


---

## Baseline — **plain FB with the actor's BC term off: 0.000 on cube, every seed,
## every checkpoint; the same critic with BC on clears the BC control**

Full write-up, hyperparameters and caveats: `docs/tables/fb_nobc_cube_ladder.md`.
Repro: `bash scripts/baselines/fb_nobc.sh {table|train|eval}`.

**Where the code came from.** `https://github.com/LUH-AI/Factored-FB.git`, branch
`density-fb`, commit `b62dc9d5e73f282924c29d0ab64d1f889b43532e`, checked out at
`/mnt/home/amohan/git/Austin/Factored-FB`. That branch is simply where the newest code
sits; the agent run is the **plain Forward-Backward critic** `fb`
(`impls/critics/fb.py`), not the density variant. Nothing from PSMFlows was used — this
agent trains F, B, the left encoder and the actor from scratch in one 500k run, so
`$PSM_DATA/flow/cube-single-play` never entered it.

**"BC anchoring off" = `--actor ddpgbc --override actor.alpha=0.0`.** `DDPGBCActor.loss`
is `q_loss + bc_loss` with `q_loss = -q.mean() / sg(|q|.mean() + 1e-6)` and
`bc_loss = -(alpha * dist.log_prob(dataset_action)).mean()`. `alpha=0` zeroes the anchor
exactly and keeps the `|q|` normaliser (it is gated by the separate `q_normalize`, default
true), so the actor ascends `Q = <F(le(s), a, z), z>` and nothing else. `alpha` is the only
BC-ish knob on this actor, so there is no second reading to hedge against. Picking `ddpgbc`
over the repo's canonical `flowbc` is also what makes the arm flow-free: `flowbc` carries
`bc_coeff: 3.0` and distils a flow-matching policy. This is the repo's own idiom —
`scripts/launch_exorl.sh` pins exactly these two flags for its no-BC arms. The runs' logs
show `bc_loss: 0.0` throughout and their `config.json` records `actor.alpha = 0.0`.

**The ladder** (500 episodes per cell, OGBench task_id=1, i.e. what PSMFlows calls
`cube-single-play-singletask-v0`; reports at
`$PSM_DATA/logs/eval500_fb_nobc_cube_{epoch}k_sd{S}.json`):

| epoch | 50k | 100k | 150k | 200k | 250k | 300k | 350k | 400k | 450k | 500k |
|---|---|---|---|---|---|---|---|---|---|---|
| seed 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| seed 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| seed 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

**Late mean (300k–500k, 15 measurements): 0.000 ± 0.000.** Same on the affine ladder's
250k–500k window (18 measurements). Wilson 95% on each 0/500 cell is [0.000, 0.008] — 15000
evaluation episodes without a single success, not a noisy zero.

Against the two numbers that matter: **BC control 0.072**, **affine psi strict, cube, late
mean 0.424**. The baseline is below the control, not merely below the method.

**Two controls, and they are the actual result.** Both seed 0, same critic, same data, same
500-episode protocol:

| arm | flags | 300k | 400k | 500k |
|---|---|---|---|---|
| **BC ON** — the repo's canonical cube-single pairing | `--actor flowbc` (`bc_coeff 3.0`) | **0.110** [0.086, 0.141] | **0.084** [0.063, 0.112] | **0.108** [0.084, 0.138] |
| BC OFF + strong orthonormaliser | `--actor ddpgbc actor.alpha=0.0 ortho_coef=1000` | 0.000 | 0.000 | 0.000 |

1. **The behaviour anchor is load-bearing and it is the only thing that is.** Same FB
   critic, same budget: with the anchor the arm clears the BC control at all three
   checkpoints (0.110 / 0.084 / 0.108 vs 0.072); without it, zero.
2. **`ortho_coef` is not the confound.** The archived PSMFlows FB reached cube 0.721 with
   `ortho_coef=1000` while this repo ships the reference-native 1.0, so the obvious
   objection was that the baseline was mis-tuned rather than BC-starved. Re-run at
   `ortho_coef=1000` with BC still off: still 0/500 at 300k, 400k and 500k.
3. The raw-action PSM baseline in this same entry reached 0.000 by deleting the same kind of
   anchor from a different agent in a different codebase. Two independent confirmations that
   "successor measure + pure Q ascent in raw action space" does not stand up on cube.

**Hyperparameters** (merged config, identical across seeds): `batch_size 256 · z_dim 50 ·
L_dim 50 · num_parallel 2 · discount 0.99 · f/b_target_tau 0.005 · ortho_coef 1.0 ·
train_goal_ratio 0.5 · fb_pessimism_penalty 0.0 · q_loss_coef 0.0 ·
actor_pessimism_penalty 0.0 · norm_z true · lr_f/lr_b 1e-4 · lr_actor 3e-4 ·
forward {512, 2, emb 2} · backward {512, 4, norm} · left_encoder {512, 4, norm} ·
actor {[512,512,512], alpha 0.0, const_std true, q_normalize true}`. 500k steps, ckpt every
50k, one H100 per seed, 41–45 min per run.

**Caveats.** (a) This is the *shipped* FB arm with its BC coefficient zeroed, not a tuned
pure-Q FB: the four Touati-parity knobs the repo exposes (`actor.q_normalize=false`,
`boot_noise_std/clip`, `actor_pessimism_penalty=0.5`, `q_loss_coef=1/z_dim`) all default off
and none were tried. The claim is not "no pure-Q FB can work". (b) Eval conditions FB on the
goal-Dirac `z = project_z(B(goal))`, this repo's default for `fb` on cube — *more*
information than PSMFlows hands `psmflow`, so the arm is not being under-sold. The
reward-inference route is wired and smoke-tested in `scripts/baselines/fb_eval500.py`
(`--relabel_infer`) but an arm at 0/500 under easier conditioning cannot be rescued by
harder. (c) `actor_loss` sits at exactly −1.0 all run — what `-q.mean()/sg(|q|.mean())`
degenerates to once `q` has a consistent sign; the gradient is live, the logged scalar is
not informative. (d) One seed each for the two controls. (e) The repo has no recorded FB
cube-single number to check against; `results/table.csv` carries `fb` rows for ManiSkill and
Metaworld only.

**If we port this FB back into PSMFlows** (to replace `archive/agents/fb.py`): the two
implementations are the *same maths*. `impls/critics/fb.py::stage_loss` and
`archive/agents/fb.py::_fb_loss_fn` share the off-diagonal squared TD residual, the
`-mean(diag)*P` term, the `0.5*sum((cov*off)^2)/off_sum - mean(diag(cov))` orthonormaliser,
the `train_goal_ratio=0.5` hindsight/sphere z-mix, and `infer_cond` = `(r^T B)/N` then
`project_z` — and the two yamls agree on every shared key. What the newer one adds is (i)
knobs, all defaulting to the old behaviour: `q_loss_coef` (the reference's auxiliary Q loss,
eq. 42), `left_encoder.identity` (Touati parity, F reads raw obs), `boot_noise_std/clip`
(target-policy smoothing), `q_normalize`, `motivo_archi`; (ii) a **goal-conditioned eval
route** (`map_eval_cond`: `z = project_z(B(goal))`) that PSMFlows' version has no equivalent
of — PSMFlows only ever infers `z` from rewards; and (iii) the critic×actor factoring, which
buys the Gaussian `ddpgbc` policy as an alternative to PSMFlows' in-agent `actor.type:
flow|td3` switch. A port does **not** require adopting the factoring: it is (1) re-register
`fb` in `agents/__init__.py` + restore `configs/agent/fb.yaml`, (2) copy the five knobs
across, (3) add `ddpgbc` as a third `actor.type`, (4) optionally add the goal-Dirac eval
path. Roughly a day. Because every new knob defaults to the pre-existing behaviour, the
archived equivalence fixture (`archive/tests/fixtures/fb_reference.npz`,
`test_fb_agent_equiv.py`) should still pass unchanged — which is the cheapest available
check that the port did not silently change the loss.


---

## Baseline — **PSM in raw action space, no BC anchoring: 0.000 on cube, on every seed,
## at every checkpoint**

Full write-up, hyperparameters and caveats: `docs/tables/psm_raw_nobc_cube_ladder.md`.
Repro: `scripts/baselines/psm_raw_nobc.sh`.

**What was run.** The two archived successor-measure agents, in RAW action space (dataset
action in the measure's middle slot — no behaviour flow, no preimages, no decode) with the
BC anchor deleted (`agent.actor.bc_coeff=0.0`, so the actor objective is the bare
`-Q.mean()`):

* **Arm A** `archive/agents/affine_psm.py` — the raw-action twin of the shipped agent
  (affine measure `M = Phi(s,a,x)·w + b`, factored `Phi = A(s,a) phi_x(x)`, goal-conditioned
  LP inference). **Preferred**, because the project's algorithm is affine.
* **Arm B** `archive/agents/psm.py` — the bilinear PSM of arXiv 2411.19418,
  `M = psi(s,z,a)ᵀ phi(x)`, the peer baseline of record.

`actor.type=ddpgbc` on both, so no flow-matching BC loss remains anywhere in the objective.
Pessimism kept exactly as each agent has it and reported: Arm A has **none** (only √d
normalisation, `b_scale=10`, `ortho_coef=1000`, `tau=0.01`); Arm B keeps
`actor_pessimism_penalty=0.5` over `num_parallel=2` heads (and `pessimism_penalty=0.0` on
the target, its own default). Nothing was added back.

3 seeds × 500k offline steps × 500-episode evals at every 50k, cube-single-play.

| epoch | Arm A sd0/sd1/sd2 | Arm B sd0/sd1/sd2 |
|---|---|---|
| 50k | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 |
| 100k | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.002 |
| 150k | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 |
| 200k | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 |
| 250k | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 |
| 300k | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 |
| 350k | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 |
| 400k | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 |
| 450k | 0.000 / 0.002 / 0.000 | 0.000 / 0.000 / 0.000 |
| 500k | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 |

**Late mean (300k–500k, n = 15 checkpoint×seed measurements each, 500 episodes each):
Arm A 0.000, Arm B 0.000.** Pooled over the whole sweep each arm scored **1 success in
15 000 episodes** (Wilson 95% upper bound 0.0004). Against **BC 0.072** and the affine
LatentFlowPSM's **0.424 ± 0.141** (late-checkpoint mean, ≥250k), the baseline clears
neither — it does not clear zero. The in-loop 50-episode eval read 0.00 at all 11 points of
all 6 runs, so nothing was missed between checkpoints.

---

## Why it is zero: the actor leaves the manifold, exactly as pre-registered

The representation trains fine on both arms (`psm_loss`, `orth_loss` converge normally).
What runs away is the greedy actor's own Q:

| arm | seed | Q @5k | @50k | @100k | @250k | @500k |
|---|---|---|---|---|---|---|
| A | 0 | 60 | 249 | 110 | 16 | 10 |
| A | 1 | 142 | **1810** | 388 | 20 | 67 |
| A | 2 | 60 | 15 | 23 | 22 | 28 |
| B | 0 | 406 | 1440 | 1820 | 1900 | **1970** |
| B | 1 | 407 | 1290 | 1970 | 1950 | **2340** |
| B | 2 | 372 | 1440 | 1790 | 1920 | **2080** |

Arm B's Q is `psi(s,z,a)ᵀz` with `norm_z=true`, so `|z| = √128 = 11.3` and Q ≈ 2000 means a
`psi` norm of ~175 at the action the actor picked — a value the contrastive loss never sees
on data, reached by 100k and held to the end. Two ensemble heads with
`actor_pessimism_penalty=0.5` do not stop it: they agree about an action neither was
trained on. Arm A's factored measure is *unnormalised* (only `phi_x` is √k-normalised, so
`Phi = A·phi_x` is free), and its Q spikes the same way early.

**This is the control the anchor exists for, and it is a clean negative.** It bounds the
*pair* (raw actions + no anchor), not either alone: the separating cell — raw actions
**with** `bc_coeff=3.0` — is the archived `affine_psm` cube push, PARKED 2026-07-26 with no
500-episode ladder. Not a tuned baseline either: archived cube hyperparameters, one flag
changed, no sweep.

---

## Infra: running an archived agent without un-archiving it

`archive/` stays frozen — nothing moved, nothing edited, and `main.py`,
`agents/__init__.py`, `agents/psmflow.py`, `agents/fql.py`, `utils/psm_networks.py` and
`tools/eval_checkpoint.py` are untouched. New, all under `scripts/baselines/`:

* `_archive.py` — imports `archive.agents.*` as a namespace package and registers a hydra
  `SearchPathPlugin` so `agent=affine_psm` resolves to `archive/configs/agent/`.
* `run_archived.py` — `main.py` for archived agents. Restores the two seams the live entry
  point dropped: `dataset.return_index = True` (the proto sampler keys `pi_z` on the global
  buffer row index, not batch position) and the goal-conditioned eval branch
  (`infer_w_goal` -> `infer_eval`), both verbatim from pre-archive `main.py` (`db96e48`).
* `eval_archived.py` — 500-episode eval; **reuses** `tools/eval_checkpoint.py`'s
  `merge_run_config` / `_cli_agent_keys` / `wilson` unchanged, so the agent config is
  inherited from the run's own `flags.json` (verified on every job:
  `actor.bc_coeff: config 3.0 -> run 0.0`).
* `train_archived.sbatch`, `eval_archived.sbatch`, `psm_raw_nobc.sh`,
  `psm_raw_nobc_table.py` (regenerates the ladder from the JSONs).

Deleting `scripts/baselines/` reverts the repo exactly. Discipline log: 200-step smokes of
both arms (2491695/6) and of the eval path (2491712/3) before launch; `flags.json` re-read
after launch on all 6 runs; expected failure mode stated before results; every number from
a persisted JSON.

---

## Antmaze H1 — **pinning the GPI policy index does not rescue antmaze; it destroys it**
## (11 of 12 cells significantly BELOW behaviour cloning, 6 of them exactly 0/500)

Pre-registration: `docs/design/2026-09-06-antmaze-failure-tests.md` §2, written and
committed to before any job was submitted; §5 is the results section and §2 was not edited
after they returned.

**The hypothesis.** Affine paper-strict is null on antmaze (late-ckpt mean 0.081 ± 0.050 vs
BC 0.072, episodes timing out at 1000 steps) while the arms with a *persistent* policy — the
affine latent-actor arm — score ~0.21 on the same env, flow and checkpoints. H1: the acting
rule redraws `K=64` policy indices `u'` **every step** and maximises over them, so the
executed behaviour is a different policy at every step of a 1000-step maze episode, and that
temporal incoherence is the failure.

**The test.** Eval-time only, no retraining, no code changes: `agent.gpi_select=fixed_index`
pins `u'` to `PRNGKey(agent.gpi_index_seed)` for the whole evaluation while the inner
`argmax_u` over 64 action latents still runs per step. Three checkpoints x four index seeds,
500 episodes each, config inherited from each run's own `flags.json` with only
`gpi_select`/`gpi_index_seed` typed on top (confirmed in each JSON's `agent_config_source`).
SLURM 2491818-2491829, all COMPLETED, 11 min each — **64x cheaper than the shipped rule**,
because `n_idx=1` collapses the `K x K = 4096` pair scan to `K = 64`.

---

## H1 results: 500 episodes per cell, Wilson 95%, BC control 0.072

| checkpoint | argmax ref | index seed 0 | 1 | 2 | 3 |
|---|---|---|---|---|---|
| sd1 @250k | 0.178 | 0.002 [0.000, 0.011] | **0.142** [0.114, 0.175] | 0.000 [0.000, 0.008] | 0.000 [0.000, 0.008] |
| sd2 @300k | 0.088 | 0.020 [0.011, 0.036] | 0.026 [0.015, 0.044] | 0.004 [0.001, 0.015] | 0.000 [0.000, 0.008] |
| sd0 @350k | 0.082 | 0.000 [0.000, 0.008] | 0.000 [0.000, 0.008] | 0.000 [0.000, 0.008] | 0.000 [0.000, 0.008] |

Pooled over the 12 cells: **97 / 6000 = 0.016**, against BC 36/500 = 0.072 —
z = -8.47, **p = 2.4e-17**. Every cell except sd1/is1 is individually below BC at
p <= 8.7e-05. Reference points: BC **0.072**, affine strict late-ckpt mean **0.081 ± 0.050**,
affine **latent-actor** arm (persistent policy) **~0.21** (0.224/0.206 @500k, 0.218/0.224
@50k, 0.176/0.216 @100k).

**Verdict: H1 refuted, in the opposite direction.** The pre-registered not-H1 threshold was
"every cell <= 0.10"; the cells are an order of magnitude below it. The single exception,
sd1 @250k index seed 1 at 0.142, is not a lift: it sits on the one checkpoint that already
scores **0.178** with the shipped per-step argmax (-0.036, p = 0.12, no significant change),
and that same checkpoint's other three index draws return 0.002 / 0.000 / 0.000. §2
pre-registered exactly this caveat — a move confined to the best-of-eight checkpoint is
about the checkpoint, not the rule. The checkpoints the hypothesis needed (argmax at the
floor, 0.082 / 0.088) return 0.000 on seven of their eight cells.

---

## What H1 changes

1. **It reproduces the 09-05 cube `fixed_index` finding on a second env and task family.**
   Cube: 0.704 -> 0.000 (index seed 0) / 0.180 (seed 1). Antmaze: 0.082-0.178 -> 0.000-0.142.
   The per-step max over 64 fresh policy indices is **load-bearing on both**, and it behaves
   the same on a 200-step manipulation task and a 1000-step navigation task. It is an
   optimism device averaged over the index lottery every step, not a policy choice — and
   removing it is worse than behaviour cloning, not merely equal to it.
2. **Temporal incoherence is not why antmaze fails.** The per-step reselection is the only
   thing holding this arm *at* BC rather than at zero.
3. **The latent-actor arm's ~0.21 is not "persistence".** A pinned index is maximally
   persistent and pools to 0.016. Whatever the actor arm has comes from the amortized actor
   being *trained* — a distillation over the index panel with its own learning signal. That
   makes the actor arm, not the strict arm, the interesting antmaze object.
4. **The index lottery is enormous on antmaze.** Four draws of `u'` on one checkpoint give
   0.142 / 0.002 / 0.000 / 0.000. Conditioned on a fixed index, `psi`'s preference over
   action latents is worth between -0.072 and +0.070 against BC depending on the draw: the
   learned measure has no index-independent notion of a good action.

Reports: `$PSM_DATA/logs/eval500_antmaze_fixedidx_sd{S}_{epoch}k_is{I}.json`, registered as
12 separate `ablation` rows in `tools/make_tables.py` (they never pool — a mean over the
four index seeds of one checkpoint would hide the entire finding).

---

## Antmaze H2 (pre-registration) — is `discount=0.98` (~50-step horizon) too short
## for a 1000-step maze?  [results in the next two slides]

Pre-registered in the same note, §3. Six retrains of the repo-default affine-strict antmaze
agent with **`agent.discount=0.99`** and **`0.995`** (horizons 100 and 200 vs the default's
50), 3 seeds each, 500k steps, `save_interval=50000`. SLURM 2491831-2491836, launched
2026-09-06 22:18, ~4 h expected. Groups `affine_strict_antmaze_g99` /
`affine_strict_antmaze_g995`.

Pre-registered expectation: under H2 the in-loop eval leaves the floor by ~250k and the
500-episode success at 500k is >= 0.15 for `gamma=0.995`, with `gamma=0.99` intermediate,
**monotone in the horizon**; under not-H2 all six stay in 0.00-0.10 like `gamma=0.98`. An
anticipated failure mode was stated in advance: a longer discount can *destabilise* the TD
backup, and `gamma=0.995` collapsing earlier than `gamma=0.98` is a plausible outcome.
The in-loop 50-episode eval over-reads badly on antmaze (sd2 @300k read **0.40** in-loop and
**0.088** at 500 episodes) and is used only to pick which checkpoint gets a 500-episode eval.

---

## Antmaze H2 — **CONFIRMED: the discount was the blocker.** gamma=0.99 pools to
## **0.294 [0.224, 0.364]** over a 30-cell ladder against gamma=0.98's **0.076**

All six retrains COMPLETED (SLURM 2491831-2491836, 3 h 53 m - 4 h 10 m each); every
`flags.json` re-read after launch differs from `affine_strict_antmaze/sd001` in `discount`
alone. The ladder is **51 500-episode cells** — gamma=0.99 at every 50k checkpoint 50k-500k
(30) and gamma=0.995 at 50k-300k plus a 500k endpoint (21) — all on the serial eval path
(`EVAL_WORKERS=1`), the one the 12 H1 cells and every historical antmaze number used. Full
per-cell table: `docs/tables/affine_antmaze_discount_ladder.md`. Figure:
`docs/figures/2026-09-07-affine-antmaze-discount-ladder.{png,json}`.

| arm | horizon | window | n | mean | 95% CI | min | max |
|---|---|---|---|---|---|---|---|
| gamma=0.98 (default) | 50 | all measured | 10 | 0.076 | ± 0.043 | 0.000 | 0.178 |
| **gamma=0.99** | 100 | **all measured** | 30 | **0.294** | **± 0.070** | 0.004 | 0.624 |
| gamma=0.99 | 100 | 50k-250k | 15 | 0.336 | ± 0.097 | 0.050 | 0.624 |
| gamma=0.99 | 100 | 300k-500k | 15 | 0.252 | ± 0.100 | 0.004 | 0.590 |
| gamma=0.995 | 200 | all measured | 21 | 0.214 | ± 0.092 | 0.002 | 0.678 |
| gamma=0.995 | 200 | 50k-250k | 15 | 0.296 | ± 0.102 | 0.074 | 0.678 |
| gamma=0.995 | 200 | **300k-500k** | 6 | **0.010** | ± 0.008 | 0.002 | 0.020 |

BC control **0.072**; the affine latent-actor arm — previously the only antmaze arm off the
floor — **~0.21**. gamma=0.99's interval is **disjoint from gamma=0.98's**, it is ~3.9x BC,
and it is the first antmaze arm whose *late* window stands on its own (300k-500k = 0.252 ±
0.100, n=15). **Antmaze-medium is not structurally closed to this agent.** The 50-step
effective horizon of gamma=0.98 was the obstruction, and the failure it produced — episodes
running to the 1000-step timeout with success pinned at BC — is exactly what a value
function blind past 50 steps does on a maze whose goal is hundreds of steps away.

---

## H2: the pre-registered signature is wrong on BOTH halves, and that is the useful part

§3 pre-registered "in-loop leaves the floor by ~250k **and** 500-ep success at 500k >= 0.15
for gamma=0.995, **monotone in the horizon**".

* **Not monotone.** gamma=0.995 beats gamma=0.99 only at 50k-100k (0.492/0.519 vs
  0.377/0.305). Pooled over any window it does not: 50k-250k 0.296 vs 0.336, all-measured
  0.214 vs 0.294. Horizon 100 is enough; horizon 200 is past the useful point.
* **gamma=0.995 collapses, now dated at 500 episodes.** Its 300k-500k window is **0.010 ±
  0.008 over 6 cells**, every one between 0.002 and 0.020 — *below the BC control* — on all
  three seeds at both checkpoints (300k: .008/.020/.018; 500k: .006/.004/.002). This was the
  destabilisation mode named in advance, and it is **the most reproducible behaviour this
  agent has ever shown**: three of three seeds, 200k steps apart, same near-zero value.
* **A 500k-only eval would have got this wrong in both directions** — reading gamma=0.995 as
  null (0.004) and gamma=0.99 as marginal (0.170 ± 0.227, its *worst* checkpoint). The
  originally specified "500k plus best in-loop" protocol was inadequate; the 50k ladder is
  what makes the result readable. Nothing here may be quoted at a single checkpoint.

**Seed spread is still the dominant term**, as on cube: within gamma=0.99, seed 2 runs
0.43-0.62 across the whole ladder while seed 1 wanders 0.004-0.414, and the across-seed std
is 0.14-0.28 at every epoch.

**Unregistered calibration finding: the in-loop 50-ep eval is well calibrated here.**
gamma=0.995 in-loop at 100k read .32/.56/.78 against 500-episode .332/.546/.678, and its
in-loop 0.000 from 300k is confirmed at 0.002-0.020 over 500 episodes. The 0.40 -> 0.088
over-read recorded for gamma=0.98 is a property of an agent **sitting at the floor** — a
50-episode sample of a near-zero rate is almost all noise — not a property of antmaze.

---

## H2 provenance and what remains

* Eval: **51 COMPLETED jobs** — 2491922-2491923, 2491961-2491964, 2491984-2492016,
  2492023-2492025, 2492030-2492038. All 51 JSONs verified programmatically: 500 episodes,
  `restore_epoch` matching the filename's epoch token, `num_workers=1`.
* 2491912/2491913 died at startup on another agent's in-flight `agents/psmflow.py` change
  (`KeyError: 'entropy'`), wrote no JSON, and were resubmitted as 2491922/2491923. No other
  eval failed.
* `tools/fig_affine_antmaze_discount_ladder.py` is idempotent and partial-safe: it reads
  whatever JSONs exist, renders the rest as gaps, and prints a coverage line
  (`gamma=0.99: 30, gamma=0.995: 21` = complete). Re-run it plus `tools/make_tables.py`,
  which carries 19 generated H2 rows (one per discount x checkpoint, seeds pooled, never
  pooled across checkpoints — gamma=0.995 wins early and collapses late, so an early mean
  and a late mean describe different agents).

**What remains.** (1) Make `discount=0.99` the antmaze setting and decide whether it becomes
the repo default — a one-line config change with a 30-cell three-seed ladder behind it.
(2) Cube and pointmaze at gamma=0.99, in flight with another agent
(`docs/design/2026-09-07-discount-sweep.md`); cube should be unaffected and pointmaze should
stay at zero for the COMPENDIUM 4.11 reason, and those two pre-registered negatives are what
would confirm the mechanism is *horizon* rather than "a bigger gamma helps everything".
(3) The gamma=0.995 collapse is an unusually clean object — `w_enc_spread` and the psi-spread
diagnostics run straight through it (third panel of the figure) and whatever explains it may
also explain the cube arm's ±0.4 within-run swings. (4) The gamma=0.98 antmaze ladder still
has 20 of 30 cells missing; filling it is 20 evals of already-trained checkpoints and would
make the three-arm comparison exact rather than pooled-over-what-exists. (5) Combined with
H1, the antmaze story is now **"the critic was horizon-starved"**, not "the acting rule was
temporally incoherent".

Discipline log: pre-registration written before submission; a 200-step
smoke of the exact training path (SLURM 2491830) whose `flags.json` differed from
`affine_strict_antmaze/sd001` **only** in `discount` (plus the three eval-only `gpi_select`
keys added 09-05, which training never reads); all six runs' `flags.json` re-read after
launch and confirmed to carry 0.99 / 0.995; the H1 eval path verified from job 2491818's log
(`CLI overrides kept: ... gpi_index_seed, gpi_select`) before reading any number; every
number from a persisted JSON.

---

## H3 (in flight) — **does the discount horizon matter beyond antmaze?** cube and
## pointmaze retrained at `gamma=0.99`, with opposite predictions registered for the two

Pre-registration: `docs/design/2026-09-07-discount-sweep.md`, written and committed to
before any job was submitted. Nothing in `agents/` or `utils/` is touched; the only change
is `agent.discount`.

**Why.** The 09-06 antmaze H2 test (section above) is still running, but two things are
already visible. `gamma=0.99` reads **0.534** and `gamma=0.995` reads **0.678** at 100k on
seed 2 over 500 episodes, against the `gamma=0.98` arm's late-checkpoint mean **0.081 ±
0.050** and BC **0.072** — a 6-8x move on the env this project had written off. And
`gamma=0.995` then **dies**: all three of its seeds read exactly 0.00 in-loop at 300k and
350k while all three `gamma=0.99` seeds hold (0.28/0.24/0.64 and 0.28/0.12/0.46). H2's §3
named that destabilisation as an anticipated failure mode before the runs started, and it
fired. `gamma=0.99` is therefore the value carried forward: it is the one that both moved
and survived.

**The test.** Six retrains of the repo-default paper-strict affine agent (`agent=psmflow`),
3 seeds x 2 envs, 500k steps, `save_interval=50000`, groups `affine_strict_cube_g99` and
`affine_strict_pointmaze_g99`. SLURM **2492017-2492019** (cube sd0/1/2) and
**2492020-2492022** (pointmaze sd0/1/2), each gated `--dependency=afterok` on its env's
200-step smoke (**2491981** cube, **2491982** pointmaze) so no training job can start unless
the smoke of the exact code path exited 0. Everything else is byte-identical to
`affine_strict_cube` / `affine_strict_pointmaze`: same frozen flow, same preimage npz, same
budget, same eval protocol.

---

**The two envs were chosen because they predict opposite things** — a sweep that expects the
same outcome everywhere tests nothing.

* **Cube: no change.** Cube episodes are 200 steps and the reward is reachable well inside
  `gamma=0.98`'s ~50-step horizon, so unlike antmaze (goal 200-400 steps away,
  `0.98^300 ≈ 2e-3`) cube is not horizon-starved. Registered point prediction for the
  300k-500k late mean: **0.42**, interval **0.33-0.50**, i.e. within ±0.09 of the
  `gamma=0.98` value **0.415 ± 0.083**. Registered separately and more falsifiably: the
  **oscillation does not shrink** — at least two of three seeds will still show a
  >= 0.30 jump between adjacent 50k checkpoints and a >= 0.35 range over the ten. Nothing in
  the diagnosed mechanism (the `w_enc_spread` encoder collapse, the per-step index lottery
  of 09-06 H1) is a function of `gamma`, so a *stabilised* ladder would contradict the 09-06
  diagnosis and is recorded here as the outcome I consider least likely.
* **Pointmaze: 0.000 everywhere, and the discount cannot change it.** COMPENDIUM §4.11 is a
  settled negative with a mechanism upstream of anything `gamma` touches: a 13x13 grid tiles
  the whole `d_a=2` latent box, and **0 of 233 enumerated latents reach the goal** in two
  full 1000-step rollouts each; the expert's route is latent white noise (within-episode
  preimage variance / marginal = **0.99**) because the goal is not in the observation, so
  routes exist only as latent *sequences*. `discount` changes which member of the policy
  family `psi` prefers; it does not add a member to a family that has none. Registered
  prediction: **0.000 on all 30 cells** (15000 episodes), anything <= 0.004 counted as
  consistent since BC itself is 0.002. The one channel that is not a flat no is stated in
  the note: the deployed rule redraws 64 indices per step, so it executes a *sequence*, and
  §4.11's probe used a fixed `u` — if per-step stitching were possible, a 50-step horizon in
  a ~200-step maze is exactly where it would be invisible. **Any cell >= 0.05** would reopen
  §4.11 for per-step-reselecting acting rules; cells in 0.004-0.05 are floor noise and will
  be reported as noise, not as a partial rescue.

---

**Failure modes named in advance.** (1) The `gamma=0.995`-style late collapse — in-loop
success pinned at *exactly* 0.00 on all three seeds from some epoch on; monitored per 50k in
each run's `eval.csv`, and it does **not** abort the ladder, because the in-loop number is
the thing this project has repeatedly caught over-reading (antmaze sd2 @300k: 0.40 in-loop,
0.088 over 500). (2) `training/w_enc_spread` crossing 0.6 markedly earlier than the
`gamma=0.98` runs' 190k / 215k / 305k. (3) TD divergence in `psm_loss` / `psi_q_spread` /
`psi_q_range_rel`. (4) Reading the answer off the in-loop eval at all — listed as a failure
mode because it is a process failure this project has committed before.

---

**Eval plan.** 500 episodes at **every** 50k checkpoint for **all** seeds — 30 cells per env,
60 jobs — with `EVAL_WORKERS=1`. That is deliberate: `scripts/eval500.sh`'s parallel path
seeds worker `w` as `seed*N + w` and therefore draws a *different* sample of 500 episode
inits, so it agrees with an existing number only within the Wilson interval. Every
`gamma=0.98` ladder this sweep is compared against was produced serially, and the comparison
is uniform only at N=1. Reports
`$PSM_DATA/logs/eval500_affine{N}k_strict_{cube|pointmaze}_g99_sd{S}.json`, disjoint from
every existing basename. Table `docs/tables/affine_discount_ladder.md` and figure
`docs/figures/2026-09-07-affine-discount-ladder.{png,json}` (generator
`tools/fig_affine_discount_ladder.py`, which carries both discounts for both envs so the
object plotted is always the pair); 22 append-only rows in `tools/make_tables.py`, the
`_g99` infix keeping them from pooling with the `gamma=0.98` rows, whose basenames carry no
discount token.

Discipline log: hyperparameter table printed in the note's §6 before submission; 200-step
smokes of the exact training path submitted per env (2491981 / 2491982, both COMPLETED
0:0 in ~2 min) and made a hard `afterok` gate on all six trainings; both smoke
`flags.json` diffed against their `gamma=0.98` references before the gate released, which
caught real config drift since the antmaze `g99` runs 14 h earlier (`agent.actor_mode`,
the `agent.actor.*` DSRL keys, `eval_workers`) and established it is inert for this arm --
all of it is actor-only code gated on `train_actor=false`, or an eval-time knob whose
default is the serial path (note §8.0); expected outcomes *and* expected failures registered before any number
arrived; and all six runs' own `flags.json` re-read after launch (§8.1) — **identical
across seeds within each env**, and differing from their `gamma=0.98` references in
**`agent.discount` alone**. SLURM 2492017-2492022 started 07:49-08:10.

---

## Every cube number this project has ever reported is OGBench **task 2**.
## Re-evaluated on all five: 0.284 avg vs BC 0.111 — the zero-shot claim survives, unevenly

`cube-single-play-singletask-v0` carries no task token, and `cube_env.py:359` defaults
`reward_task_id` to **2**; `utils/evaluation.py:86,93` resets with `env.reset()` and never
passes `options={'task_id': ...}`, so the bare id has always been one task out of five.
Stated plainly, for the record:

- **every cube number in this file, in `docs/tables/results.md`, in the ladder tables and in
  `PAPER/ICLR/tables/table_headline.tex` — headline, arms, ablations — is task 2 alone;**
- **every antmaze and pointmaze number is task 1** (`locomaze/maze.py:348` defaults mazes to 1).

Nothing had to be retrained to fix that. `agents/psmflow.py` reads `batch['rewards']` **only**
in `infer_z`/`infer_z_a`/`infer_eval_z` (`:609-628`) — the eval-time closed-form task vector —
so the frozen flow, phi, psi and the latent index are reward-free.
`ogbench/utils.py:199-202` loads the **same** `cube-single-play-v0.npz` for every task id and
only calls `relabel_dataset`, which reads `env.unwrapped._reward_task_id`. So changing
`env_name` to `cube-single-play-singletask-task{K}-v0` gives each task its own dataset reward
column, its own inferred `w`, and its own success criterion, against one unchanged
checkpoint. `tools/eval_checkpoint.py` loads no preimage npz and never runs `main.py`'s
pairing guard, so the env-name change trips nothing (verified: env ids
`cube-single-singletask-task{1..5}-v0` are all registered, `_reward_task_id` reads 1/2/3/5 as
expected, and the bare id reads 2).

---

## The five-task table (500 episodes per cell, 3 seeds x 5 checkpoints 300k-500k)

64 SLURM jobs: 4 tasks x 5 checkpoints x 3 seeds on the finished `affine_strict_cube` runs,
plus one BC control per task. No arm flags on any eval line — the agent config came from each
run's own `flags.json` (`policy_index=latent acting=gpi train_actor=false psi_form=affine`,
confirmed in every report JSON).

| task | n (ckpt x seed) | late mean 300k-500k ± 95% CI | min … max | BC control (500 ep) | ratio |
|---|---|---|---|---|---|
| task 1 | 15 | 0.314 ± 0.123 | 0.000 … 0.832 | 0.152 [0.123, 0.186] | 2.1x |
| **task 2** (the default — all earlier numbers) | 15 | **0.415 ± 0.083** | 0.086 … 0.704 | 0.072 [0.052, 0.098] | 5.8x |
| task 3 | 15 | 0.480 ± 0.074 | 0.190 … 0.730 | 0.230 [0.195, 0.269] | 2.1x |
| task 4 | 15 | 0.100 ± 0.034 | 0.038 … 0.294 | 0.086 [0.065, 0.114] | 1.2x |
| task 5 | 15 | 0.113 ± 0.045 | 0.002 … 0.296 | 0.014 [0.007, 0.029] | 8.1x |
| **5-task average** | 75 cells | **0.284** (± 0.152 across task means) | — | **0.111** | **2.6x** |

Table: `docs/tables/affine_cube_multitask.md` (generated) and a section in
`docs/tables/results.md`. Figure + series:
`docs/figures/2026-09-06-affine-cube-multitask.{png,json}`
(generator `tools/fig_affine_multitask.py`).

---

## Per-checkpoint mean across the three seeds

| epoch | task 1 | task 2 | task 3 | task 4 | task 5 |
|---|---|---|---|---|---|
| 300k | 0.507 | 0.393 | 0.339 | 0.087 | 0.213 |
| 350k | 0.258 | 0.516 | 0.410 | 0.165 | 0.121 |
| 400k | 0.375 | 0.391 | 0.505 | 0.091 | 0.091 |
| 450k | 0.233 | 0.468 | 0.532 | 0.097 | 0.054 |
| 500k | 0.195 | 0.309 | 0.616 | 0.061 | 0.087 |
| BC | 0.152 | 0.072 | 0.230 | 0.086 | 0.014 |

Per-seed late means (5 checkpoints each), showing the 09-05/09-06 oscillation is **not**
task-2-specific and is not a fixed per-seed ranking either:

| task | sd0 | sd1 | sd2 |
|---|---|---|---|
| 1 | 0.416 | **0.074** | 0.451 |
| 2 | 0.326 | 0.498 | 0.422 |
| 3 | 0.464 | 0.530 | 0.446 |
| 4 | 0.103 | 0.130 | 0.068 |
| 5 | 0.126 | 0.048 | 0.164 |

---

## What this settles, and what it does not

**The zero-shot claim survives, and it is no longer a one-task claim.** One representation per
seed, five reward functions inferred in closed form, all beating or matching their own BC
control: **0.284 vs 0.111 averaged over the five, 2.6x.** Every task is at or above BC; three
of five (2, 3, 5) are clearly above it, task 1 marginally so (CI lower bound 0.191 vs the BC
Wilson upper bound 0.186).

**Task 2 was a lucky draw for the headline, and the paper must stop quoting it alone.** Its
5.8x is the second-best *ratio* but rests on the weakest BC control of the five (0.072). Read
by absolute success the arm is best on task 3 (0.480) and task 2 (0.415), and lands near
0.10 on tasks 4 and 5.

**Task 4 is the honest null.** 0.100 ± 0.034 against BC 0.086 [0.065, 0.114] — indistinguishable.
The method neither helps nor hurts there. This is the first cube task on which the arm has no
effect, and it is worth understanding before the next environment: whatever makes task 4 hard
is *not* the inversion (the same latents serve all five tasks) and *not* the representation
(same phi/psi), so it is the reward-inference step or the task's own difficulty.

**Task 5 is where the method earns its claim most cleanly.** 0.113 vs a BC floor of 0.014 —
8.1x — a task the behaviour flow essentially cannot do at all, done eight times as often by
selecting inside the same flow's latent space. Small absolute number, but the cleanest
separation from the control in the whole project.

**The oscillation is a property of the arm, not of the task.** The per-seed table shows the
same non-convergence found on 09-05/09-06: seed 1 is best on task 2 (0.498) and worst on task
1 (0.074); no seed dominates. Task 3 is the exception — monotone 0.339 → 0.616 across the five
checkpoints on all three seeds, the only clean learning curve any cube task has produced.

**Consequence for the roadmap.** `docs/plans/2026-09-06-env-roadmap.md` §6.0 costed this at
~0 GPU-hours of training and predicted it would multiply the evidence base by five. It did:
75 cells instead of 15, on artifacts that already existed. Every future eval — including the
next environment — should report all five tasks, and `scripts/eval500.sh` now takes an
`ENV_NAME` override for exactly that.

---

## Tooling changed with this result

- `scripts/eval500.sh` / `scripts/slurm/eval500.sbatch`: `ENV_NAME` overrides the ENVKEY's
  default env id (this is the whole multi-task mechanism); the sbatch also lets a preset
  `EVAL_LOGS` through, so BC controls land in `$PSM_DATA/evals` beside `bc_cube.json`.
- `tools/eval_checkpoint.py`: the report now records **`train_seed`**, read from the restored
  run's own `flags.json`. `seed` was and remains the *eval* seed (0 on every eval500 line ever
  run), so until now a JSON carried no machine-readable trace of which training seed produced
  it — only `restore_path` and the filename. Covered by four new cases in
  `tests/test_eval_checkpoint_flags.py` (18 pass).
- `tools/fig_affine_multitask.py` (new) and a `multitask_section` in `tools/make_tables.py`;
  `docs/tables/results.md`'s header now states the task-2 scope of every other row.

Discipline log: full hyperparameter table printed before submission; a 5-episode task-1 smoke
(SLURM 2491837) confirmed the env id, the inherited arm config and the report's `env` field
before the 64 jobs went in; all 60 + 4 JSONs re-validated after the fact for env id, epoch,
episode count, `train_seed` and arm flags (0 mismatches); every number here from a persisted
JSON. Nine jobs report SLURM state FAILED with exit 126 *after* writing a complete report and
printing the final success line — a teardown artifact, not a lost eval; all 60 cells are
present and were validated.

---

## Pointmaze — **the third published env now has a complete `gamma=0.98` ladder:
## 0.000 in all 30 cells, 15,000 episodes, zero successes** (and it is horizon-confounded)

Table: `docs/tables/affine_pointmaze_ladder.md` (generated). Rows registered in
`tools/make_tables.py`; `docs/tables/results.md` regenerated.

The third environment with published Stage-A/B artifacts was brought up end to end and run
at the repo default (`agent=psmflow`, no flags = paper-strict affine). Three seeds to 500k
(SLURM 2491814/15/16, **4h00 / 3h55 / 4h00**), `save_interval=50000`, then the full
500-episode ladder at every 50k checkpoint on every seed — **30 of 30 cells, no gaps**.

| epoch | sd0 | sd1 | sd2 | mean ± std |
|---|---|---|---|---|
| 50k … 500k, all ten rungs | 0.000 [0.000, 0.008] | 0.000 [0.000, 0.008] | 0.000 [0.000, 0.008] | 0.000 ± 0.000 |

| pooled | n | mean | 95% CI |
|---|---|---|---|
| 300k–500k | 15 | **0.000** | ± 0.000 |
| all rungs 50k–500k | 30 | **0.000** | ± 0.000 |
| **BC control** (frozen flow, per-step prior) | 500 ep | **0.002** (1/500) | Wilson [0.0004, 0.0112] |

The in-loop 50-episode evals agree: `evaluation/success` is `0.0` at **every** logged point
on all three seeds, step 1 through 500k. No figure was produced — 30 identical zeros against
33 identical in-loop zeros plot as a flat line on the axis and would dress a structural null
as a measurement.

Note the control: pointmaze BC is **0.002**, not antmaze's 0.072. This env offers almost no
bar to clear, and "the agent is below BC" would be over-reading a one-episode difference.
Both the method and its control are at zero here.

---

## The pointmaze null is CONFOUNDED by the same discount that manufactured antmaze's

This ladder ran at `discount=0.98` — a ~50-step effective horizon — on a maze with a
**1000-step** episode limit. The H2 section above then showed that this exact setting, not
the environment, is what pinned antmaze at the BC floor (`gamma=0.995` → **0.519 ± 0.175**).
So the correct statement is **"the affine arm scores 0.000 on pointmaze at `gamma=0.98`"**,
not "pointmaze is closed to this method".

COMPENDIUM §4.11 remains the standing prior — 0 of 233 enumerated latents reach the goal;
within-episode preimage variance / marginal = 0.99, so routes exist only as latent
*sequences* — and it was pre-registered here as the expected outcome before results arrived.
But it does not settle this table, for two reasons. The reachability probe rolled **fixed-`u`**
policies while this arm acts by `gpi`, re-drawing `u` from 64 candidates every step (a
strictly larger family); and whether the critic can *steer* that per-step choice toward a
goal hundreds of steps away is exactly what the discount controls. The coherence mechanism
and the horizon mechanism are not alternatives, and a `gamma=0.98` run cannot separate them.

**This ladder is therefore the baseline for H3's `affine_strict_pointmaze_g99` arm**
(SLURM 2492020-2492022, pre-registered in `docs/design/2026-09-07-discount-sweep.md` with the
prediction that pointmaze stays at 0.000). Read the two together. A `gamma=0.995` pointmaze
arm was *not* launched: H2 showed 0.995 destabilises (all three antmaze seeds at exactly 0.00
in-loop from 300k), so 0.99 is the value worth spending GPUs on, and H3 already has it.

---

## Infra — the OGBench dataset host is dead; a HF mirror is the working route

`pointmaze-medium-navigate-v0.npz` was missing locally and **could not be downloaded from
upstream**. `https://rail.eecs.berkeley.edu/datasets/ogbench/<name>.npz` now 302s to
`https://iris.eecs.berkeley.edu/datasets/ogbench/<name>.npz`, which returns a **404**
WordPress page. This hits *every* OGBench dataset, including `cube-single-play-v0.npz` which
we already hold — it is an upstream move, not a proxy problem (the no-proxy route fails TLS at
the same host, and ogbench 1.2.1 on PyPI plus GitHub master still hardcode the dead URL, so no
version bump or compute-node retry helps).

**What worked:** the HF mirror `ryanhoangt/ogbench_data`, which carries the standard
`<env>-v0.npz` / `-val.npz` at repo root.

```bash
.venv/bin/python - <<'EOF'
import os, shutil
from huggingface_hub import hf_hub_download
dst = os.path.expanduser('~/.ogbench/data')
for f in ['<env>-v0.npz', '<env>-v0-val.npz']:
    shutil.copyfile(hf_hub_download('ryanhoangt/ogbench_data', f, repo_type='dataset'),
                    os.path.join(dst, f))
EOF
```

Verified against the preimage npz (`obs[0]`, `act[0]` match to the pipeline's action clip),
and `main.py`'s row-count + first/last-1000-row guard would refuse a wrong copy anyway. The
mirror also holds `scene-play`, `cube-double/triple/quadruple-play` and `puzzle-3x3/4x4-play`,
so the outage blocks no environment on the roadmap. Also worth knowing:
**`OGBENCH_DATASET_DIR` is a no-op** with ogbench 1.1.0 — `ogbench/utils.py:10` hardcodes
`~/.ogbench/data` and `envs/env_utils.py:139-140` never passes `dataset_dir`; it works here
only because `$HOME/.ogbench/data` is the same path.

Preimages came from `scripts/hf_preimages.py pull --name pointmaze-medium-navigate --dest
$PSM_DATA --with-flow` (52 MB, 1M rows, sidecar `restore_path` repaired automatically).

---

## Provenance / caveats (pointmaze)

* `flags.json` re-read after launch on all three seeds: byte-identical to
  `affine_strict_cube/sd000` except `env_name`, `run_group`, the flow/preimage paths and the
  three eval-only `gpi_select` keys added 09-05.
* Ten ladder cells (100k sd2, 150k/200k/250k all seeds; SLURM 2491907-2491921) died at
  startup in 20-41 s with `KeyError: 'entropy'` — an `assert a_cfg["entropy"]` against a key
  these runs' `flags.json` predates, since `eval_checkpoint.py` inherits the run's own config.
  Fixed in `agents/psmflow.py` by another agent (`_actor_opt` fallback + `fill_actor_defaults`
  as the first statement of `create`); `tests/test_psmflow_config_compat.py` passes 9/9. All
  ten were resubmitted (2491965-2491974) and completed. **They wrote no JSON, so no partial
  result entered the table** — the ladder was regenerated only once all 30 files existed.
* Every cell is `tools/eval_checkpoint.py`, 500 episodes, agent config from the run's own
  `flags.json`, no arm flags on the CLI. The generator asserts `restore_epoch` and
  `num_episodes` per file, so a mislabelled JSON cannot silently populate a cell.
* Eval throughput on this arm: ~11 s/episode serial (measured in the 200-step smoke), i.e.
  ~1.5 h per 500-episode cell; later cells used the new `EVAL_WORKERS=4` default.

---

<!-- _class: lead -->

## 2026-09-05 — the affine strict arm **oscillates**: cube swings 0.086 ↔ 0.704
## between adjacent 50k checkpoints, and no logged diagnostic predicts which

The 09-04 entry quoted this arm at 250k (0.532 / 0.620) because that was the latest
checkpoint the runs had reached. All eight runs have now finished 500k and the full
checkpoint ladder has been evaluated at 500 episodes. **The 250k number was not the arm
converging — it was one draw from a wide, non-monotone distribution over checkpoints.**
The headline claim of 09-04 does not survive as stated; the arm is still the best thing on
cube, but only as a *mean over late checkpoints*, and it is dead on antmaze.

Figures + machine-readable series: `docs/figures/2026-09-05-affine-cube-collapse.{png,json}`,
`docs/figures/2026-09-05-affine-antmaze-sweep.{png,json}`.

---

## Repo reorg: done

The 09-04 reorganisation is complete and is the state of the tree: `agent=psmflow` defaults
to the paper-strict affine arm (`psi_form=affine policy_index=latent train_actor=false
acting=gpi`), and every agent other than `psmflow` and `fql` now lives under `archive/`.
See the **2026-09-04 (evening)** entry below for what was built, the test evidence and the
design note (`docs/design/2026-09-04-affine-psi.md`). The only `agents/` change here
is the eval-time `gpi_select` ablation switch (`_gpi_select_ablation`), added for
the ablation batch below and left as a documented switch, not a new default.

---

## Every 500k number, 500 episodes, both envs, both arms

| arm | env | sd0 | sd1 | mean ± 95% CI (t, n=2) |
|---|---|---|---|---|
| affine **strict** (`train_actor=false acting=gpi`) | cube | 0.086 | 0.548 | 0.317 ± 2.935 |
| affine **actor** (`train_actor=true acting=actor`) | cube | 0.162 | 0.130 | 0.146 ± 0.203 |
| affine **strict** | antmaze | 0.002 | 0.096 | 0.049 ± 0.597 |
| affine **actor** | antmaze | 0.224 | 0.206 | 0.215 ± 0.114 |

Comparators (all 500k): BC control **cube 0.072**, **antmaze 0.072**; free-psi Arm B cube
0.083 ± 0.191 (3 seeds); the old latent-actor agent cube 0.230 ± 0.051, antmaze
0.213 ± 0.140; FB cube 0.721.

**At 500k, on the terminal checkpoint alone, the strict arm beats nothing.** Cube strict
0.317 is inside the old actor agent's interval and its two seeds (0.086, 0.548) do not
overlap each other; antmaze strict 0.049 is *below* the BC control; the affine actor arm
matches the old actor agent on antmaze (0.215 vs 0.213) and is worse on cube (0.146 vs
0.230). Every n=2 t-interval here is worthless and is printed only for form.

---

## Cube, affine strict: the full checkpoint sweep (500 episodes each)

| epoch | sd0 | sd1 |
|---|---|---|
| 100k | 0.078 [0.058, 0.105] | 0.330 [0.290, 0.372] |
| 250k | **0.532** [0.488, 0.575] | **0.620** [0.577, 0.661] |
| 300k | 0.286 [0.248, 0.327] | — |
| 350k | **0.704** [0.663, 0.742] | — |
| 400k | 0.282 [0.244, 0.323] | 0.346 [0.306, 0.389] |
| 450k | 0.272 [0.235, 0.313] | 0.568 [0.524, 0.611] |
| 500k | 0.086 [0.065, 0.114] | 0.548 [0.504, 0.591] |

Ten late (≥250k) checkpoint×seed measurements, 500 episodes each:
**mean 0.424, sd 0.196, 95% CI ± 0.141, min 0.086, max 0.704.**

sd0 alone goes 0.532 → 0.286 → **0.704** → 0.282 → 0.272 → 0.086 on consecutive 50k
checkpoints of one run. Each of those is a 500-episode measurement whose Wilson interval is
±0.04 wide, so **the swing is real training-time non-stationarity, not evaluation noise** —
adjacent intervals are disjoint by a wide margin. sd1 swings less (0.620 … 0.346 … 0.568 …
0.548) but is not monotone either.

---

## Antmaze, affine strict: the same non-stationarity, but around the BC control

Four new 500-episode evals (the two highest un-evaluated in-loop checkpoints
per seed; 50k and 500k were already done). SLURM 2491639-42, ~93 min each at ~11 s/episode.

| epoch | sd0 | sd1 |
|---|---|---|
| 50k | 0.112 [0.087, 0.143] | 0.000 [0.000, 0.008] |
| 250k | — | **0.178** [0.147, 0.214] |
| 350k | 0.082 [0.061, 0.109] | — |
| 400k | 0.056 [0.039, 0.080] | 0.134 [0.107, 0.167] |
| 500k | **0.002** [0.000, 0.011] | 0.096 [0.073, 0.125] |

Late-checkpoint mean (≥250k, 6 measurements): **0.091 ± 0.064**, against BC 0.072, the
affine *actor* arm's 0.215 and the old actor agent's 0.213.

**Not flat-low, and not the cube pattern either.** The two seeds move in *opposite*
directions: sd0 decays monotonically to zero (0.112 → 0.082 → 0.056 → 0.002) while sd1
rises off the floor and then decays (0.000 → 0.178 → 0.134 → 0.096). Every antmaze
checkpoint sits within ±0.11 of the BC control, so on this env the arm is **not a result at
all** — its late mean's interval contains BC. The in-loop 50-episode eval is useless here
(0.00 at 8 of 11 points on sd0 while 500 episodes read 0.002-0.112).

**The encoder does not collapse on antmaze.** `w_enc_spread` *rises* monotonically to ≈1.24
and stays there for 400k steps — the opposite of cube's 1.30 → 0.25 decay — and the arm is
still bad. Together with cube — where the spread collapses monotonically while
success swings up and down four times — that settles it: `w_enc_spread` moves monotonically
in whichever direction the env dictates and success does not follow it in either env.

`docs/figures/2026-09-05-affine-antmaze-sweep.{png,json}`.

---

## What the oscillation is not: no logged scalar predicts it

Every quantity this arm logs was paired with the 12 cube 500-episode measurements
(checkpoint × seed) and correlated against success. **Nothing predicts it.**

| logged series | Pearson vs 500ep success | Spearman |
|---|---|---|
| `w_enc_spread` (encoder pairwise distance) | −0.386 | −0.231 |
| `psi_q_index_spread_rel` | +0.017 | +0.266 |
| `psi_q_spread_rel` | +0.032 | +0.203 |
| `psi_q_range_rel` | −0.003 | +0.210 |
| `psi_q_spread` (absolute) | −0.072 | +0.084 |
| `psm_loss` (measure TD) | +0.106 | +0.322 |
| `orth_loss` | −0.398 | −0.385 |
| *in-loop 50-episode eval at the same step* | **+0.916** | **+0.916** |

Read that last row carefully. The in-loop 50-episode eval — the thing this project's
reporting rules forbid quoting — **tracks the 500-episode number at r = 0.92 on this arm**
(12 paired points, Pearson = Spearman = 0.916). That is not a licence to quote it; it is
evidence about *where* the variance lives. The 500-episode CI is ±0.04 and the
checkpoint-to-checkpoint swing is ±0.3, so the noise is in the **policy at that
checkpoint**, not in the measurement, and 50 episodes already see most of it.

Ruled out by the table: encoder collapse (`w_enc_spread` decays smoothly and monotonically
on cube and *rises* monotonically on antmaze, while success does neither); the
orthonormality constraint (`orth_loss` pinned at ≈−64 from 10k on, flat to three digits in
every run); a diverging measure TD (`psm_loss` grows ~10x and spikes constantly, unaligned
with the dips); and any index/readout scale artefact (all four `psi_q_*` statistics,
|Spearman| ≤ 0.27). **A monotone training trace with a non-monotone eval trace means the
failure is in the acting rule or in what psi has learned, not in a scalar the loss sees.**

---

## Two probes settled it

### Selection diagnostic — `tools/diag_gpi_selection.py`, `docs/design/2026-09-05-gpi-selection-diag.md`

State-matched probe: the same 256 dataset states and the same 64×64 candidate roster scored
by four checkpoints of `affine_strict_cube/sd000` (250k / 350k / 400k / 500k).

- **The two pre-registered failure modes are not it.** Selected-`u` norm 2.83-2.95 (roster
  2.13), top1−top2 gap 0.31-0.43 sd, selected-`u` clip fraction ~1.2% vs the roster's 0.28%
  — **identical at the good and bad checkpoints**. Edge-seeking and flatness are permanent
  properties of the rule, present at 0.704 too.
- **The ranking is re-randomised.** The per-`u` GPI score the argmax consumes correlates
  only **rho 0.21-0.36 between any pair of checkpoints**, including 250k vs 350k (0.31), the
  two that both work. Top-5 overlap 0.20-0.28; argmax agreement 11-19%.
- **Target-critic selection is not a stabiliser.** Online vs target rank the same roster at
  **rho 0.92-0.96** within a checkpoint (`tau=0.01` has long converged), and the target
  drifts across checkpoints identically (0.23-0.40). Dropping pessimism changes the ranking
  even less (0.93-0.95).
- **The index/action asymmetry.** psi's spread over `u'` is **~30x** its spread over `u`
  (`index_matters` ~5-13k against a per-`u` spread of a few hundred): the inner argmax over
  `u`, the one that picks the action, runs on the weakest part of the signal.
- **MC ground truth** (fixed-`u` rollouts near rewarding transitions): Q-rank vs return-rank
  Spearman +0.10 / +0.15 / +0.13 / +0.10, while the checkpoint-free baseline `−||u||` scores
  **+0.28**. MC regret of the Q-argmax 16.8 / 22.8 / 23.0 / 32.4 against 19.1 for a random
  pick — three of four checkpoints select *worse than chance* on this proxy. Averaging the
  four rankings lifts Spearman 0.121 → 0.166 and regret 23.75 → 19.2: checkpoint averaging
  removes the harm, it does not make the selector good.

---

### Eval-time GPI ablations — `docs/design/2026-09-05-gpi-ablations.md`

Eleven 500-episode evals, one run (`affine_strict_cube/sd000`), acting rule swapped at eval
time with nothing retrained. `eval500_gpiabl_*.json`, now registered in `make_tables.py`.

| arm | rule | @350k | @500k |
|---|---|---|---|
| G | `argmax` (shipped) | **0.704** (exact reproduction, 352/500) | 0.086 |
| E | `mean` — ensemble mean, no pessimism | 0.734 (p=0.29, tie) | — |
| C | `small_ball` | 0.670 (p=0.25) | **0.046** (p=0.011, *worse*) |
| D | `soft_topm` m=8 | 0.612 (p=0.002, *worse*) | 0.062 (p=0.15) |
| A1 | `max_norm`, critic-free | 0.078 | — |
| A2 | `top_quartile_random`, critic-free | 0.078 | — |
| B | one prior draw, K=1 | 0.090 | — |
| F1 | `fixed_index` seed 0 | **0.000** | — |
| F2 | `fixed_index` seed 1 | 0.180 | — |

Three results, in order of importance:

1. **The critic carries the whole gap, and the norm bias carries none of it.** Both
   critic-free controls that reproduce the deployed rule's *selection statistics* land at
   **0.078**, indistinguishable from BC (p=0.72) — `top_quartile_random` even overshoots the
   deployed |u|=2.8 at 3.01. One prior draw gives 0.090. So the large-norm story is a true
   description of the argmax and **dead as a performance explanation**.
2. **The per-step max over fresh policy indices is the mechanism.** Pinning `u'` to one
   prior draw for a whole eval — same critic, same 64 `u` draws, same argmax over `u` —
   gives **0.000** and **0.180** depending only on which `u'` was drawn. Seed 0 is
   *significantly below BC* (0/500, p=1e-9): with the index pinned, ranking `u` by
   `psi(s,u,u')ᵀw` selects actively harmful actions. The `K×K` pair scan is load-bearing.
3. **No eval-time intervention recovers 500k.** Five rules leave it at or below BC. The
   0.704 → 0.086 collapse is in the learned `psi`, not in the rule that reads it — so the
   fix is training-side or aggregation-side, not acting-side.

---

## Reporting rule (binding, effective now)

> **No single checkpoint of the affine strict arm may be quoted.** Not the best, not the
> last, not 250k, not 350k. Any number reported for this arm is the **mean over late
> checkpoints (≥250k) and seeds**, with the spread stated.

| env | late-ckpt mean (≥250k) | n (ckpt×seed) | range | BC control |
|---|---|---|---|---|
| cube | **0.424 ± 0.141** | 10 | 0.086 - 0.704 | 0.072 |
| antmaze | **0.091 ± 0.064** | 6 | 0.002 - 0.178 | 0.072 |

On cube the arm is still a real result — the only PSMFlow configuration above 0.4, against
Arm B's 0.083, the old actor agent's 0.230 and FB's 0.721. On antmaze its interval contains
BC and it is not a result at all. The 09-04 headline "0.532 / 0.620 @250k" is **superseded**
and must not be re-quoted on its own; likewise 0.704 and 0.086 are the same agent, the same
rule and the same norm bias.

`tools/make_tables.py` now enforces the shape: one row per (arm, env, **checkpoint**), plus
one explicitly labelled `late-ckpt mean` row per env, plus the GPI ablation block. The four
rows that previously globbed `eval500_affine_strict_*_sd?.json` under a "500k" label were in
fact reading the **100k** JSONs; that is fixed. Both BC controls now resolve (they live in
`$PSM_DATA/evals/` on this cluster, not `logs/`), and `_load` de-duplicates an eval reached
through two names so a 1-seed control cannot print as 2.

---

## Open items

**1. `index_agg=expectile` — the best-motivated next experiment.** Both probes point at the
same object: the deployed 0.704 is an **optimistic per-step max over 64 samples of a learned
function** on the axis where psi has ~30x more spread, which is exactly the structure E4a
indicted. `index_agg=expectile` (already in the tree, `configs/agent/psmflow.yaml:86-88`)
distils an upper expectile of `psi(s,u',u)ᵀw` over `u' ~ p0` into a z-free scalar head and
takes no argmax over samples at acting time. Run it **on the affine agent**
(`psi_form=affine policy_index=latent train_actor=false acting=gpi index_agg=expectile`),
The ablations note's §7.4 names the cheap companions: a **larger `index_panel`** (shipped
16; the yaml records `index_panel: 1` as the InFOM-like setting) and a **`gpi_num_u` sweep
on the index axis alone**. Neither note proposes specific sweep values — pick them, print
the table before launch. Judge the result on the **late-checkpoint mean and its spread**,
never a peak.

**2. Aggregation, not sharpening.** Checkpoint averaging is the only intervention measured
to help (MC Spearman 0.121 → 0.166, regret 23.75 → 19.2). A param EMA of psi is the way to
turn "report the mean over checkpoints" into an *agent*. Within-checkpoint softening
(`soft_topm`) is a measured loss and is not the same medicine.

**3. Closed — do not re-run.** Target-critic selection (rho 0.92-0.96 with the deployed
rule, a no-op), ensemble mean instead of min (0.734 vs 0.704, p=0.29, a tie),
`small_ball` (holds 350k, hurts 500k) and `max_norm`/`top_quartile` (BC-level). `lr_sf`
lowering is *not* yet tested and remains open, but note the drift is in the ranking, not in
the loss scale.

**4. DSRL-NA actor.** The actor arms are stable and capped (cube 0.108 / 0.113 / 0.146 at
100k / 250k / 500k — a total span of 0.04 against the strict arm's 0.62); the strict arm is
unstable and high. A noise-aliased/DSRL-NA-style latent actor is the standing proposal for
getting the strict arm's ceiling with the actor arm's smoothness. Not started.

**5. Antmaze needs a reason, not another seed.** The arm sits at BC on antmaze at every
checkpoint while `w_enc_spread` behaves in the opposite direction to cube. Nothing here
explains that; the affine *actor* arm reaches 0.215 there, so the substrate is fine
and the strict/GPI path is what fails.

---

<!-- _class: lead -->

## 2026-09-04 (evening) — the affine measure head psi = A(s,u)ᵀ w(u′) + beta(s,u):
## cube strict 0.532 / 0.620 at 250k, against Arm B's 0.083 at 500k

Design + pre-registration: `docs/design/2026-09-04-affine-psi.md`, written **before** launch,
expectations included. Rem. `tradeoff` records that the shipped agent adopts Prop.
`bilinear`'s form but **not** the affineness of psi in the policy coordinate — `w^{u′}` is
absorbed into a free network. The affine form is implemented explicitly below. The paper
asserts `w^{u′}` exists (Assumption `affine`) and gives **no formula** for it, so the encoder
`u′ -> w(u′)` is **our design choice**, stated as such in the design note.

### What was built

| what | where |
|---|---|
| `AffinePsiMap` — `psi(s,u,u′) = A(s,u)ᵀ w(u′) + beta(s,u)`, drop-in for `PsiMap` | `utils/psm_networks.py:439-500` |
| `_AffinePsiTower` — `(s,u)` trunk, two heads (`A`: `h -> z_dim·d_w`, `beta`: `h -> z_dim`), **no `u′` input at all** | `utils/psm_networks.py:403-436` |
| `encode_index` / `sa_terms` — the halves exposed so affineness and collapse are measurable | `utils/psm_networks.py:481-495` |
| `psi_form: free \| affine` + `affine:{w_dim,encoder_hidden,encoder_layers,norm_w}` | `configs/agent/psmflow.yaml:52-65`, `agents/psmflow.py:899-906` |
| head selection + the `policy_index=latent` guard | `agents/psmflow.py:699-726` |
| `index_spread` — `psi_q_spread_rel`, `psi_q_range_rel`, `psi_q_index_spread_rel`, `w_enc_spread` | `agents/psmflow.py:298-349`, called at `:512-516` |
| `RESTORE_EPOCH` (default 500000) so arms can be evaluated at a common earlier checkpoint | `scripts/eval500.sh:29,83,90` |
| eval-JSON registration, 4 rows (not run) | `tools/make_tables.py:92-105` |

**Design choices.** ONE encoder shared across the `num_parallel` ensemble — `w^{u′}` is a
property of the policy, not of a critic member, so the ensemble disagrees only through
`(A, beta)`, which is what `pessimism_penalty` measures. `w(u′)` on the **unit sphere** (not
`psm_norm`'s `sqrt(d)`): `(A,w)` is identified only up to `(cA, w/c)`; unit norm pins it,
keeps psi's scale independent of `d_w`, and makes collapse read as small *pairwise distance*
at fixed radius. `A` and `beta` never see `u′` — Assumption `affine` enforced by
construction, not by a penalty. `d_w = 128 = z_dim`.

---

### Tests — full suite clean

`tests/test_psmflow_affine.py`, **11 tests**: default path **byte-identical** to
`psi_form=free` (post-update params compared leaf by leaf, plus identical
`psm_loss`/`orth_loss`/`actor_loss`, and the shipped info dict gains no key); psi **exactly
affine** in `w(u′)` — `psi(u′₁) − psi(u′₂) == Aᵀ(w(u′₁) − w(u′₂))` and `psi == Aᵀw + beta`,
both to 1e-4; `A`, `beta` index-free; `w(u′)` on the unit sphere and not constant; finite
step in **both** arms; `psi_form=affine` + `policy_index=task_vector` refused; unknown
`psi_form` refused; `sample_actions` in both acting modes; affine checkpoint structurally
distinct from the free one.

Whole suite, module-per-process: **248 passed, 3 skipped, 0 failed** (39 modules).
`ruff` on the touched files adds one finding (`C408` on the new `affine=dict(...)`), the
same kind as the 8 sibling config entries; it removes one (`I001` in `psm_networks.py`).

**Restore path checked on GPU, not just in unit tests.** `tools/eval_checkpoint.py` was run
against an affine checkpoint with **none** of `psi_form / policy_index / train_actor /
acting` on the CLI: `agent_config_source.inherited` came back
`['acting','policy_index','psi_form','train_actor']` from the run's own `flags.json`
(`logs/affine_restorecheck.json`). The reported evals below pass those four explicitly, so
their `agent_config_source` shows `flags_json` set and the four keys under `cli_overrides`.

---

### Runs — 8 jobs, both envs, seeds 0/1, launched at 500k

`agent=psmflow`, `psi_form=affine`, `policy_index=latent`, `use_point_preimage=true`,
`u_clip=3.0`, `offline_steps=500000`, `eval_interval=50000`, `eval_episodes=50` (in-loop,
never quoted), `save_interval=50000`; `affine_strict_*` = `train_actor=false acting=gpi`,
`affine_actor_*` = `train_actor=true acting=actor`. Full HP table printed pre-launch;
200-step GPU smokes run for both arms; all 8 `flags.json` re-read after launch and confirmed.

**The runs did NOT reach 500k inside the compute window.** The affine head's `A` output
(`1024 -> 16384`) makes a step 24-36 ms under 8-way contention against the free head's ~7 ms,
and the strict arm's antmaze GPI eval costs **10.1 s/episode** (84 min per 500-episode eval).
Everything below is therefore quoted at the latest checkpoint each arm reached and could be
evaluated at — **100k for all four arms, plus 250k for cube** — never at 500k. Every number
is a fresh 500-episode `tools/eval_checkpoint.py` run; no in-loop 50-episode value is quoted
anywhere in this entry. The 8 training jobs are still running toward 500k.

---

### 500-episode results (per-seed Wilson; comparators are 500k numbers)

**Headline — the affine strict arm on cube, at 250k, is the best zero-shot PSMFlow number
on record and both seeds agree.**

| arm | epoch | sd0 | sd1 | mean ± 95% CI (t, n=2) | pooled (1000 ep) |
|---|---|---|---|---|---|
| **affine strict, cube** | **250k** | **0.532** [0.488, 0.575] | **0.620** [0.577, 0.661] | **0.576 ± 0.559** | 0.576 [0.545, 0.606] |
| affine strict, cube | 100k | 0.078 [0.058, 0.105] | 0.330 [0.290, 0.372] | 0.204 ± 1.601 | 0.204 [0.180, 0.230] |
| affine actor, cube | **250k** | 0.130 [0.103, 0.162] | 0.096 [0.073, 0.125] | 0.113 ± 0.216 | 0.113 [0.094, 0.135] |
| affine actor, cube | 100k | 0.110 [0.085, 0.140] | 0.106 [0.082, 0.136] | 0.108 ± 0.025 | 0.108 [0.090, 0.129] |
| affine actor, antmaze | 100k | 0.176 [0.145, 0.212] | 0.216 [0.182, 0.254] | 0.196 ± 0.254 | 0.196 [0.173, 0.222] |
| affine actor, antmaze | 50k | 0.218 [0.184, 0.256] | 0.224 [0.190, 0.263] | 0.221 ± 0.038 | 0.221 [0.196, 0.248] |
| affine strict, antmaze | 50k | 0.112 [0.087, 0.143] | 0.000 [0.000, 0.008] | 0.056 ± 0.712 | 0.056 [0.043, 0.072] |
| affine strict, antmaze | 100k | *eval running (>75 min each)* | | | |

Comparators, all at **500k** and all with more training than anything above:

| comparator (cube) | | comparator (antmaze) | |
|---|---|---|---|
| Arm B — free psi, strict, `acting=gpi` | 0.083 ± 0.191 (3) | PSMFlow (point, actor) | 0.213 ± 0.140 |
| gpi-matched control (point arm, gpi) | 0.054 ± 0.032 (5) | BC per-step prior | 0.072 |
| PSMFlow actor (point) | 0.230 ± 0.051 (5) | | |
| BC per-step prior | 0.072 | | |
| FB (zero-shot, raw actions) | 0.721 ± 0.020 | | |

**Read the strict-cube row against Arm B.** Arm B is the *same algorithm with a free psi*:
`policy_index=latent`, `train_actor=false`, `acting=gpi`. Its three recorded seeds are
0.006 / 0.160 / 0.084 and its matched gpi control is 0.054. The affine head at **half the
training** returns 0.532 / 0.620 — every one of the four Wilson intervals involved is
disjoint from every Arm B seed's, and the pooled 0.576 [0.545, 0.606] sits **7x** its
comparator. It also beats the actor arm (0.230) and is the only PSMFlow configuration that
has ever come within reach of FB's 0.721 on cube. The n=2 t-interval (±0.559) is worthless
and is quoted only for form; the evidence here is the **agreement of two independent seeds
at 500 episodes each**, which Arm B never had.

**And it climbs.** The same two runs read 0.078 / 0.330 at 100k and 0.532 / 0.620 at 250k.
The 09-03 entry's warning — the gpi arm "climbs late… do not judge this arm before ~400k" —
holds here in the affine arm's favour: 250k is still early, and the runs continue to 500k.

**The actor arms do not show the effect — and cube actor was measured at the SAME 250k
checkpoint, so this is not a training-budget artifact.** cube actor 0.108 at 100k and 0.113
at 250k (flat), antmaze actor 0.196 at 100k / 0.221 at 50k — all level with their free-psi
comparators, while cube strict went 0.204 → 0.576 over the same interval. On this evidence the
affine head helps the arm that *selects by ranking psi* (gpi) and does nothing for the arm
that delegates selection to an amortized actor. That is exactly where a better-conditioned
psi should show up, and it is the first time in this project's record that the strict,
paper-faithful arm has beaten the actor arm.

### Diagnostics — the mechanism the head was built to change

At the **100k** checkpoint (`w_enc_spread`, `psi_q_index_spread_rel`, `psi_q_spread_rel`):

| run | sd | `w_enc_spread` | `psi_q_index_spread_rel` | `psi_q_spread_rel` |
|---|---|---|---|---|
| affine strict cube | 0 / 1 | 1.200 / 1.307 | **0.627 / 0.559** | 0.033 / 0.043 |
| affine actor cube | 0 / 1 | 1.232 / 1.304 | **0.612 / 0.586** | 0.035 / 0.039 |
| affine strict antmaze | 0 / 1 | 1.152 / 1.169 | **0.599 / 0.287** | 0.061 / 0.020 |
| affine actor antmaze | 0 / 1 | 1.153 / 1.170 | **0.574 / 0.292** | 0.056 / 0.019 |

At **250k**, cube strict reads `w_enc_spread` 0.453 / 0.414, `psi_q_index_spread_rel`
**0.818 / 0.877**, `psi_q_spread_rel` 0.045 / 0.034.

`psi_q_index_spread_rel` is the relative spread of `Q` across prior policy indices `u′` —
the quantity GPI's outer argmax consumes. It reads **29-88%** against the **~1%** that every
free-psi measurement in this project has reported (D1 0.9%, D3 1.1%, Arm B 0.9%, E4b 0.86%,
DSRL-SAC 1.1-1.5%), and it **rises** with training (0.56-0.63 at 100k → 0.82-0.88 at 250k)
alongside the success climb. Spread across prior *action* latents (`psi_q_spread_rel`, the
inner argmax's quantity) is 2-6% — above the 1% band, but by a factor of a few, not fifty.

`w_enc_spread` starts at 0.72-1.11 (5k), peaks at 1.15-1.31 (100k) against the ~1.414 a
random unit-sphere sample gives, and on **cube** falls to 0.41-0.45 by 250k while
`psi_q_index_spread_rel` keeps rising — the encoder is *concentrating* its directions and
`A` is growing to compensate, not collapsing (a collapsed encoder reads ~0 and would take
the index spread with it). Antmaze holds ~1.19 at 160k. **Watch this number**: if it reaches
~0 while the index spread falls with it, the affine structure has degenerated back to a
`u′`-free psi, and that is the diagnostic that would say so.

---

### Verdict against the pre-registration

Pre-registered (design note §6): **success** = `psi_q_index_spread_rel` >> 1% **and**
strict > Arm B (0.083); **failure** = the same ~1% band and BC-level strict.

- **Success fires on both halves, for cube strict.** Index spread 29-88% against ~1%, and
  0.532 / 0.620 against Arm B's 0.083 ± 0.191 — at half the training, with two agreeing
  seeds. The stated failure branch ("the affine form is not the missing piece either") is
  **refuted**.
- The hypothesis was that the affine bottleneck acts as a *regularizer on the policy slot*.
  The diagnostic says the mechanism is real and specific: psi stops being flat across `u′`,
  which is precisely the faculty Arm B lacked and E1/E4a indicted (ranking).
- **Scope, honestly.** Two seeds, at 250k not 500k, on **one environment and one arm**. The
  actor arms show nothing (cube 0.108 → 0.113, antmaze 0.196, level with their comparators).
  And **antmaze strict at 50k is 0.112 / 0.000, i.e. below its BC control** — at a fifth of
  cube-strict's 250k budget, on the env whose latent geometry the 09-03 S1 probe already
  found less navigable (ratio 0.79 vs 0.91). Its 100k evals were still running when the
  compute window closed. Nothing here says the affine head fixes antmaze, and nothing says the
  cube result survives to 500k.
- Also unchanged: this is still below FB's 0.721 and far below FQL's 0.949.

**Next, in order.**
1. Let the 8 jobs reach 500k and run `scripts/eval500.sh` at `RESTORE_EPOCH=500000` for all
   8. The four `eval500_affine_{strict,actor}_<env>_sd<k>.json` names are already registered
   in `tools/make_tables.py` (not yet run).
2. Collect the pending antmaze-strict evals (50k and 100k, submitted, 84 min each).
3. A **third seed** on cube strict — 0.576 pooled over two seeds is strong, but this project
   has been burned by two-seed cube numbers before (09-03, seeds spanning 0.090-0.374).
4. The obvious ablation now: `psi_form=affine` with `acting=actor` is already run and flat,
   so the next cut is `d_w` (128 → 32/512) to test whether the *bottleneck width* is the
   active ingredient, and `norm_w=false` to test whether the sphere constraint is.

**Housekeeping.** The working tree carried a **pre-existing, uncommitted deletion** of 20
comment lines in `agents/psmflow.py` (the `q_dist` / action-branch / FB-graft field
docstrings added in `6e166d7`) before this work touched the file. It is not part of it and shows up in `git diff` as a deletion — restore it before committing. **DONE** in
the reorg below: restored verbatim, since all three switches they document still exist.

---

<!-- _class: lead -->

### 2026-09-04 (late) — the affine agent is now THE agent; everything else moved to `archive/`

Consequence of the number above, not a new experiment. **No results in this section.**

**The default flipped.** `agent=psmflow` with no flags is now the paper-strict affine
LatentFlowPSM. The four arm flags and one dataset flag that the 0.532 / 0.620 runs passed on
the command line are now the config:

| key | was | is | source |
|---|---|---|---|
| `psi_form` | `free` | **`affine`** | `affine_strict_cube_sd{0,1}` flags.json |
| `policy_index` | `task_vector` | **`latent`** | ditto |
| `train_actor` | `true` | **`false`** | ditto |
| `acting` | `actor` | **`gpi`** | ditto |
| `use_point_preimage` | `false` | **`true`** | ditto; `main.py` refuses a corrected-target npz with `false`, so the old default was unusable anyway |

Everything else is byte-identical to those runs' `flags.json` (`z_dim` 128, `num_parallel` 2,
`discount` 0.98, `tau` 0.01, `ortho_coef` 1000, both pessimism penalties 0.5, `lr_phi` 1e-5,
`lr_sf` 1e-4, `sf` 1024x1, `phi` 256x2, `affine` `{w_dim:128, encoder_hidden:256,
encoder_layers:2, norm_w:true}`, `index_agg` max, `u_clip` 3.0, `gpi_num_u` 64).
`configs/agent/psmflow.yaml` and `get_config()` both moved, and both now name the free psi
and the task-vector index as *ablations*.

**Back-compat for old checkpoints.** Flipping a default silently rewrites what
`tools/eval_checkpoint.py` builds for a run whose `flags.json` predates the key — a pre-09-04
psmflow checkpoint has no `psi_form` at all, so it would have been rebuilt as an affine head
(and, with its own `policy_index=task_vector` inherited on top, would have tripped
`create()`'s guard). `LEGACY_AGENT_DEFAULTS` in that file back-fills the **pre-2026-09-04**
value for any of the five keys the run does not carry, records it in
`agent_config_source.legacy_defaults`, and a typed CLI override still wins. Three tests pin it.

**`archive/`.** Every agent except `psmflow` and `fql` moved there with `git mv`, with its
yaml, its tests and the tools/scripts/plans built on it: `psm`, `affine_psm`,
`latent_affine_psm`, `latentrl`, `fb`, `ifql`, `iql`, `rebrac`, `sac`. Also archived: the
diagnostics for hypotheses the compendium records as settled negative (fixed-`u` family,
interface ceiling, flow Jacobian, mixture-preimage width, D1–D3 free-head critic forensics),
the dropped ICLR F1–F4 figure scripts, and the superseded plan queue. `archive/README.md` is
a one-line-each index; `pyproject.toml` now excludes `archive` from ruff and pins
`testpaths = ["tests"]`. **`fb` was archived rather than kept**: `tools/make_tables.py` is
pure JSON I/O, so its FB rows (cube 0.721) keep printing from the eval JSONs already on disk;
only *re-running* FB needs the agent back.

**Two edits the move forced.** (1) `agents/psmflow.py` imported seven pure helpers from
`agents/psm.py`, so they are now `utils/psm_common.py` and `archive/agents/psm.py`
re-exports them. (2) The latent actor's network and loss are **kept, not archived** — it is
the DSRL-style comparator and the substrate for the planned DSRL-NA distillation of the GPI
argmax; it is simply off by default, and `tests/test_psmflow_actor.py` now builds it
explicitly through a `_legacy_agent()` helper.

**Tests rewritten, not deleted.** Six modules asserted the OLD default was the default.
Each now pins the new default explicitly and keeps the old arm as a named ablation through
`_legacy_agent()` (`test_psmflow_agent`, `_actor`, `_policy_index`, `_backup_explore`,
`_affine`, `_groundtruth`). `test_psmflow_groundtruth.py` gained a **new** chain-MDP pin for
the default arm: with no actor to interrogate, its ground truth is `gpi_select` itself —
the (u_i, u′_j) search must return a goal-ward latent at every non-goal state, which it does.

### Verification (all run after the move)

| check | result |
|---|---|
| `python -c "from agents import agents"` | `['fql', 'psmflow']` |
| all nine archived agents still import from `archive.agents.*` | 9/9 OK |
| full suite, module-per-process (26 modules) | **176 passed, 3 skipped, 0 failed** |
| `ruff check .` | 145 findings, from 357 at HEAD under the same config (archive excluded). The only *added* finding in the live tree is the `affine=dict(...)` `C408` already noted above |
| 200-step GPU smoke, `main.py agent=psmflow` with **no** agent flags (sbatch 2491622, kisski-inference) | COMPLETED in 1:53. Its `flags.json` agent block diffs against `affine_strict_cube_sd0`'s: **NONE** — the defaults reproduce the 0.532/0.620 configuration exactly |
| `tools/eval_checkpoint.py` on `affine_strict_cube_sd0@250000`, 5 episodes, **no** arm flags (sbatch 2491624) | restored and ran; provenance printed *"run config already matches this checkout's defaults"*, CLI overrides only the three path flags. 1/5 solved (n=5) |

`squeue` before: 7 training jobs running, 0 pending. After: 6 running (`affine_strict_cube_sd0`
reached 500k on its own in the meantime), 0 pending — **no job was ever pending while the
tree was mid-move**, so nothing could import a half-moved tree.

---

<!-- _class: lead -->

## 2026-09-04 — S1: the latent value landscape IS navigable; S2 built, and its first run was invalid

Executes `docs/plans/2026-09-03-latent-coherence-and-infom.md` (added here; it was written
but never committed). S1 is the plan's gate and it **passed navigable**, which per the
plan's own decision rule authorises S2. S2 is implemented and re-running; its first launch
measured a bug and is withdrawn — see below.

### S1 — `tools/diag_latent_smoothness.py`, four panels, all navigable

Forward passes only: the frozen Stage-A flow for `decode` and a frozen FQL expert for the
oracle score. No Stage-C checkpoint, no `psi`, so the answer holds for every Stage-C arm at
once. 64 on-path states x 512 unclipped `N(0,I)` latents, `v = -||a - a*||`.

| env | decoder | `knn_r2_a` (control) | `knn_r2_u` | basin | disp_u | corr da/du | best-of-512 |
|---|---|---|---|---|---|---|---|
| cube | onestep | 0.850 | **0.769** | 0.848 | 0.627 | 0.850 | 0.049 |
| cube | ODE-100 | 0.796 | **0.726** | 0.880 | 0.579 | 0.880 | 0.060 |
| antmaze | onestep | 0.746 | **0.587** | 0.714 | 0.728 | 0.774 | 0.256 |
| antmaze | ODE-100 | 0.744 | **0.595** | 0.727 | 0.741 | 0.793 | 0.313 |

Pre-registered: navigable needs `knn_r2_u` > 0.5, basin > 0.4, dispersion < 0.8, corr > 0.7;
scrambled is `knn_r2_u` < 0.05. **All four statistics clear on all four panels**, at 12-15x
the scrambled threshold.

**Reading. The geometry is benign.** The ~1% flatness that D1 (0.9%), D3 (1.1%), Arm B
(0.9%), E4b (0.86%) and DSRL-SAC (1.1-1.5%) all report is NOT a property of the latent
space; it is a property of the learned critics. Live hypothesis #1 ("the landscape is
rough") dies by measurement rather than by another 3x500k run.

**Validity.** The harness reproduces E1 cold: mean best-of-512 oracle distance **0.060**
(ODE-100) against E1's recorded **0.062**, from a freshly trained expert on another machine.

**Caveat, stated because it missed its bar.** The calibration control `knn_r2_a` was
pre-registered > 0.9 and came in 0.74-0.85 on every panel. That is a finite-sample ceiling
(512 points in d_a 5 / 8), not a broken estimator -- the failure mode it exists to catch
(reading ~0 where it must read ~1) did not occur. A depressed ceiling drags every R^2 down,
so it makes the navigable verdict **conservative**. Ratio form: `knn_r2_u / knn_r2_a` =
**0.91** cube, **0.79** antmaze.

**Third independent sighting of the cube/antmaze split.** Antmaze is navigable but less so
(ratio 0.79 vs 0.91), and its best-of-512 sits at 0.313 against mean ||a|| 1.99 (15.7%)
where cube sits at 0.060 against 0.87 (6.9%) -- on a probe with no inversion, no preimage
and no alpha in it.

Reports: `$PSM_DATA/logs/d5_latent_smoothness_{cube,antmaze}_{onestep,ode}.json` (+ `_raw.npz`
with the `(u, a, v)` arrays so the statistics recompute without re-rolling).

### The oracle did not exist on this cluster, and had to be built

S1 needs a frozen FQL expert to roll the states AND to score. HF ships only the three
Stage-A BC flows; every local run dir is Stage-C. Trained here, 500k, 28 min each:

| env | recipe | 500-ep success | Wilson 95% |
|---|---|---|---|
| cube | `agent.alpha=300` (the recorded 0.949 recipe) | **0.966** (483/500) | [0.946, 0.979] |
| antmaze-medium | `agent.q_agg=min agent.alpha=10` | **0.980** (490/500) | [0.964, 0.989] |

Cube overlaps the recorded FQL 0.949 [0.936, 0.960], which is E1's readability condition.
The antmaze recipe is **adapted** from the FQL reference's antmaze-*large* line -- the
reference gives no antmaze-medium value. Incidentally this is the repo's **first per-task
topline on antmaze**: PSMFlow's 0.213-0.247 there is ~4x below its own ceiling, the same
shape as cube's 0.230 against 0.949.

### S2 — implicit GPI by expectile distillation (`index_agg=expectile`)

New in `agents/psmflow.py`: a scalar head `q_dist` fitted by upper-expectile regression onto
`psi(s, u', u)^T w` over `index_panel` prior draws `u'`, replacing the argmax at two call
sites (`gpi_select` drops the KxK pair scan; `flow_actor_loss` climbs the distilled head).
`measure_loss` untouched in v1 so any regression is attributable. Keys `index_agg`,
`expectile_mu`, `index_panel`, `q_dist` in `get_config()` and `configs/agent/psmflow.yaml`;
`create()` asserts `expectile` requires `policy_index=latent`. Static switch, keys folded
out of `rng` (108/109/110), so `index_agg=max` keeps a byte-identical stream -- pinned by
29 passed / 2 skipped across the three psmflow test modules including the golden fingerprints.

**Read the InFOM source before trusting the spec.** `agents/infom.py` at
github.com/chongyi-zheng/infom: the intention encoder conditions on
`(next_observations, next_actions)` -- S3's `p_e(z|s',a')` is **confirmed as written**. But
S2's stated premise is **wrong**: InFOM has **no z-conditioned -> z-free distillation**. Its
latents are prior draws, its target is a Monte-Carlo average
`target_q = 1/(1-gamma) * future_rewards.mean(axis=0)` over `num_flow_goals`, and the
expectile is IQL-shaped asymmetric L2 on the regression residual
(`weight = where(adv >= 0, expectile, 1 - expectile)`, default 0.9, task values 0.9-0.99).
So: **mu = 0.9 and ONE latent per update, not a panel.** `index_panel=16` is this repo's
lower-variance choice, `index_panel=1` is InFOM's analogue, and S2 should be described as
InFOM-inspired rather than a port. Also `latent_dim` defaults to 512, not the README's 128.

### The first S2 launch was invalid — `q_dist` could not see the task

**Withdrawn:** an earlier reading of `q_dist_spread_rel` = 0.7-1.2% at 115k steps as the
plan's pre-registered kill signal ("not above 5% by 50k => the argmax is not the
bottleneck, S2 is dead"). That measured a bug.

The plan specifies `q_dist(s, u)`. Its regression target `psi(s, u', u)^T w` is a function
of the task vector, and `sample_mixed_z` redraws `w` **per batch element every step**, so a
head without `w` can only fit the task-MARGINAL expectile -- and `gpi_select` then ranked
candidate latents at eval while blind to the `task_z` it was acting for. A task-blind head
also trivially shows little spread, so the kill statistic was contaminated too. InFOM can
omit the task because it adapts per-task with reward labels; a zero-shot method cannot.

Fixed to `q_dist(s, w, u)` at all five sites (init, actor loss, distillation loss, spread
diagnostic, `gpi_select` broadcasting `task_z`). Evidence it matters: on the identical
200-step smoke, `q_dist_loss` 23.3 -> 9.0 and `q_dist_pred` **+0.94 -> -1.77** against a
target of -6.83 -- the old head could not even get the sign right. Jobs 2491543/2491545
cancelled at 39 min rather than carried to 500k.

### Control arm result, and why it is not quotable yet

`index_agg=max` (= Arm B: `policy_index=latent train_actor=false acting=gpi`) never touches
`q_dist`, so these two runs are unaffected by the bug.

| seed | 500-ep success | Wilson 95% |
|---|---|---|
| 0 | **0.374** (187/500) | [0.333, 0.417] |
| 1 | **0.090** (45/500) | [0.068, 0.118] |

Seed 0 is the highest zero-shot cube number on file other than FB's 0.721 -- above the actor
arm (0.220 +/- 0.037) and cube point (0.230 +/- 0.051). **Do not quote it.** Two seeds
spanning 0.090-0.374 give a t-interval of +/-1.80, wider than the measurement, and Arm B's
own record (0.006 / 0.160 / 0.084, 0.083 +/- 0.191) is an arm whose seed spread already
dwarfs its mean. A third control seed is running.

Both seeds also **climb late** -- sd0 0.06 at 350k -> 0.26 -> 0.52 -> 0.40; sd1 0.06 -> 0.20
-> 0.20 -> 0.12. An in-loop reading taken at 300k called this arm flat at BC level and was
simply premature; do not judge this arm before ~400k.

### In flight

`s2_cube_bexp09` seeds 0/1/2 and `s2_cube_bmax` seed 2 (jobs 2491549-2491552), 500k each on
one H100. Expectile arms run ~2.6x slower than the control (the 16-draw panel per step).
Launched at 3 seeds, not the plan's 2: the control's measured spread makes n=2 unable to
separate the arms. Pre-registered: success = beating the BC control (0.072 on this cluster)
at 500 episodes with non-overlapping CIs; the `q_dist_spread_rel` > ~5% necessary condition
is now re-testable on a head that can actually see the task.

---

<!-- _class: lead -->

## 2026-09-03 — DSRL-SAC launched: a scalar critic over LATENTS, flow never called in training

New switch `agent.critic_input: action | latent` in `agents/latentrl.py`. `action` is the
default and is byte-for-byte the pre-existing computation (same rng splits, same shapes,
same logged values — pinned by golden numbers in `tests/test_latentrl_smoke.py`). `latent`
is the offline DSRL arm: `Q(s, u)`, TD anchored on `u_data = clip(noise_preimage, ±u_clip)`,
actor `−Q(s, u_a)/stopgrad(|Q|) + bc_alpha_latent·‖u_a − u_data‖²`, **no decode anywhere in
the training graph** (a test perturbs `flow_onestep` and requires the actor loss not to
move). Residual head must be inert (`residual_eps=0`, asserted at `create()`). Acting is
unchanged: actor latent → frozen one-step decode. Plan: `docs/plans/2026-09-03-latentrl-dsrl-sac.md`.

### Why this cell of the grid was empty

DSRL (Wagenmaker et al. 2025) steers a frozen generative policy through its input noise; the
offline variant needs noise aliasing, and Stage-B preimages ARE that aliasing. What we had
run instead: psmflow's actor steers latents against `ψᵀw` (flat — D3, 1.1% relative spread),
and `latentrl`'s critic scored **decoded actions** with the gradient through the decoder
(0.142 ± 0.025 at ε=0, 0.905 ± 0.020 at ε=0.05). Nobody had run *scalar latent critic +
latent actor + no decode in training*.

### Arms (SLURM, one H100 per seed — no tmux on this machine, job name is the handle)

| SLURM job (id) | group | `u_clip` | seed | run dir under `$PSM_DATA/exp/PSMFLows/` |
|---|---|---|---|---|
| `latsac_cube_uclip3_sd0` (2491460) | `latsac_cube_uclip3` | 3.0 | 0 | `latsac_cube_uclip3/sd000_s_2491460.0.20260903_021732` |
| `latsac_cube_uclip3_sd1` (2491461) | `latsac_cube_uclip3` | 3.0 | 1 | `latsac_cube_uclip3/sd001_s_2491461.0.20260903_021732` |
| `latsac_cube_uclip1_sd0` (2491462) | `latsac_cube_uclip1` | 1.0 | 0 | `latsac_cube_uclip1/sd000_s_2491462.0.20260903_021732` |
| `latsac_cube_uclip1_sd1` (2491463) | `latsac_cube_uclip1` | 1.0 | 1 | `latsac_cube_uclip1/sd001_s_2491463.0.20260903_021731` |

All four on `gpu002.kisski`, `--gres=gpu:1` each, so one seed per H100 and no sharing.
Logs `$PSM_DATA/logs/<job name>-<id>.out`. Submitted through
`scripts/slurm/train_psmflow.sbatch` with `AGENT=latentrl`.

All: `agent=latentrl agent.critic_input=latent agent.use_point_preimage=true`,
`bc_alpha_latent=10.0`, `residual_eps=0.0`, `alpha=10.0`, `lr=3e-4`, `batch_size=256`,
critic `[512]×4` + layer_norm, `num_parallel=2`, `discount=0.99`, `tau=0.005`,
`pessimism_penalty=0.5`, `actor_pessimism_penalty=0.5`, actor/residual `512×2` with 2
embedding layers, `offline_steps=500000 eval_interval=50000 eval_episodes=50`.
Frozen Stage-A flow `/mnt/home/amohan/psm-data/flow/cube-single-play` @ 500000 (run group
`bcflow_cube_single_20260726_135032` — the checkpoint every cube number decodes through);
preimages `/mnt/home/amohan/psm-data/preimages/cube-single-play.npz` (canonical, alpha=20,
`num_samples=200`, 13/1M invalid), pulled from HF with `--with-flow`.

### Expected outcomes, pre-registered

- **near FQL (0.949) / the residual result (0.905):** offline latent TD works with a scalar
  critic; the zero-shot loss is in the representation (D2: best linear readout of φ explains
  ~13% of reward variance).
- **near 0.14:** offline latent TD does not survive without an action-space critic; next
  step is DSRL-NA (`Q_A` over dataset actions, latent Q distilled over prior draws).

**Prediction on the new live diagnostic:** psmflow's measure head sits at ~1% relative Q
spread; if this scalar critic is also ~1%, the actor gradient will again be BC-dominated.
`q_spread` / `q_spread_rel` (16 clipped prior draws at ≤64 states, key folded out of `rng`
so the training stream is untouched) is now logged every `log_interval`.

**Early reading (in flight, NOT a result).** 400-step smoke `latsac_smoke2`: `q_spread_rel`
0.034 → 0.026 → 0.020 → 0.019 over steps 100–400. All four real arms at 25k steps:
`q_spread_rel` **0.0017–0.0037** (uclip3 sd0/sd1 0.0034/0.0017, uclip1 sd0/sd1
0.0037/0.0020), i.e. already **below** psmflow's 1.1%, with `critic_q_data` running away
downward (−75 to −81) exactly as the 08-13 calibration forensics describe. On the
pre-registered reading that is the 0.14 branch, and `u_clip=1.0` did **not** raise the
relative spread — so the "everything decodes to nearly the same action" explanation for
flatness is not what the tighter box tests it to be. By 165k–175k steps `q_spread_rel` has
risen to **0.006–0.012**, i.e. settling right on psmflow's ~1% band, and the 50-episode
in-loop evals are wandering 0.06–0.32 (±0.115 CI each — do not quote them). Wait for the
500-episode evals before concluding; the prediction is on record either way.

### RESULT — all four runs finished 500k, 500-episode evals in

All four SLURM jobs `COMPLETED 0:0` (22–24 min wall clock each on one H100);
`params_500000.pkl` present in every run dir. Evaluated one job per GPU through
`scripts/slurm/eval500.sbatch` → `scripts/eval500.sh latentrl cube`, with
`agent.critic_input=latent` and the training `u_clip` repeated; each was cross-checked
against the run's own `flags.json` before submitting and both flags are now recorded **in
the eval JSON itself** (`u_clip`, `critic_input` added to `tools/eval_checkpoint.py`'s
report), so the numbers below are verifiable, not asserted. Eval seed left at the config
default 0 for all four — the protocol every recorded eval500 number used
(`scripts/reeval_ode_e2.sh` never passes a seed), so the 500 episode inits are paired.

| arm | seed | success (500 ep) | Wilson 95% |
|---|---|---|---|
| `latsac_cube_uclip3` | 0 | **0.186** (93/500) | [0.154, 0.223] |
| `latsac_cube_uclip3` | 1 | **0.180** (90/500) | [0.149, 0.216] |
| `latsac_cube_uclip1` | 0 | **0.096** (48/500) | [0.073, 0.125] |
| `latsac_cube_uclip1` | 1 | **0.158** (79/500) | [0.129, 0.193] |

Per arm, mean ± 95% CI across the 2 seeds (t interval, the project convention; with n=2 the
t multiplier is 12.7, so the u_clip=1 interval is uninformative on its own):

| arm | mean ± 95% CI |
|---|---|
| `latsac_cube_uclip3` (u_clip=3) | **0.183 ± 0.038** |
| `latsac_cube_uclip1` (u_clip=1) | **0.127 ± 0.394** |

**BC control measured on this cluster** (`agent=fql agent.bc_only=true`, same frozen flow the arms decode through, 500 episodes, `$PSM_DATA/evals/bc_cube.json`): **0.072** (36/500) [0.052, 0.098] — reproducing midi-01's 0.068 [0.049, 0.093], so both arms genuinely clear the control on this machine's own measurement.

Comparators, from `docs/tables/results.md` (same env, same 500-episode protocol):
FQL per-task **0.949 ± 0.063**; latent RL per-task ε=0.05 **0.905 ± 0.020**; latent RL
per-task ε=0 pure decode **0.142 ± 0.025**; BC control (per-step prior through the same
frozen flow) **0.068 [0.049, 0.093]**.

**Verdict: the 0.14 branch, not the 0.9 branch.** Both arms land in the pure-decode band —
u_clip=3 at 0.183 ± 0.038 is a hair above the ε=0 latent-critic result (0.142 ± 0.025) and
squarely inside the PSMFlow zero-shot band (0.236 ± 0.071 / re-eval 0.220 ± 0.037); u_clip=1
at 0.127 ± 0.394 is indistinguishable from it. Nothing is within reach of 0.905 or 0.949.
Both beat the BC control (0.068) — the loop is not inert, it just does not improve past what
a decode-only policy already gets. So a *scalar critic over latents with no decode in the
training graph* does not recover the per-task ceiling: everything that ever worked here
worked by letting the critic see and move a raw action. Per the plan, the next step is
DSRL-NA (`Q_A` over dataset actions, latent Q distilled over prior draws).

**The Q-spread prediction was right.** Final in-loop diagnostics at step 500000 (`train.csv`,
not an eval number):

| arm | `q_spread_rel` | `critic_q_data` | `actor_q` | `bc_loss` |
|---|---|---|---|---|
| uclip3 sd0 | 0.0125 | −165.50 | −166.76 | 0.961 |
| uclip3 sd1 | 0.0150 | −177.05 | −177.73 | 0.892 |
| uclip1 sd0 | 0.0110 | −166.20 | −167.22 | 0.415 |
| uclip1 sd1 | 0.0153 | −191.72 | −192.92 | 0.426 |

`q_spread_rel` 1.1–1.5% is exactly psmflow's measure-head band (1.1%), i.e. the critic
separates clipped prior draws about as poorly as ψᵀw does, and `actor_q` sits within ~1 of
`critic_q_data` — the actor is not finding anything the data anchor does not already have.
`critic_q_data` at −165 to −192 (from −75 to −81 at 25k) is the same monotone downward drift
the 08-13 calibration forensics describe, so the scale the actor normalises by is set by
runaway pessimism rather than by real value differences. `u_clip=1.0` did not raise the
relative spread at 500k any more than it did at 25k, so the tighter typical-set box does not
rescue separability — the "everything decodes to nearly the same action" story is not what
is limiting this.

`tools/make_tables.py` gained the two rows (`eval500_latsac_uclip{3,1}_sd?.json`, patterned
on the existing latent-RL rows) and they resolve against `$PSM_DATA/logs` here, but the tool
was **not run**: this cluster holds none of the historical `eval500_*.json`, so
`make_tables.py --logs $PSM_DATA/logs` would rewrite `docs/tables/results.md` with `--` in
every other cell. Run it on midi-01, or after the four JSONs are copied to that logs dir.

### Also

- `scripts/eval500.sh` gained a `latentrl` mode and now **rejects unknown modes**: it
  previously fell through to `agent=psmflow` for any first argument other than `bc`, so
  `eval500.sh latentrl ...` would silently have evaluated the wrong agent.
- **`u_clip` must be repeated at eval.** Nothing outside the agent config hard-codes 3.0
  where it matters (`utils/flow_inversion.py` never clips; `tools/analyze_preimages.py`'s
  `U_CLIP_DEFAULT = 3.0` is a `--u_clip`-overridable diagnostic default; `psmflow.py:776` is
  a fallback `get_config`), but `tools/eval_checkpoint.py` rebuilds the agent from the hydra
  config, and the actor is `tanh * u_clip`. Evaluating the `u_clip=1.0` arm without
  `agent.u_clip=1.0` scales every action by 3x. Noted in `eval500.sh`'s header.
- `scripts/slurm/train_psmflow.sbatch` takes `AGENT`, `EVAL_EPS`, `LOG_INT` (defaults
  reproduce every earlier submit line).
- `scripts/eval500.sh`'s midi-01 paths are now env-overridable (`PSM_REPO`, `EVAL_LOGS`,
  `EXP_ROOT`, `FLOW_DIR`, `PRE_NPZ`, `OGBENCH_DATASET_DIR`); the defaults are unchanged, so
  every midi-01 invocation still works. It also fails fast if the flow dir or preimage npz
  is absent, and validates `MODE` before touching paths. New `scripts/slurm/eval500.sbatch`
  wraps it for the scheduler, the way `train_psmflow.sbatch` wraps `main.py`.
- `tools/eval_checkpoint.py` now records `u_clip` and `critic_input` in the report JSON, so
  a latentrl eval can be checked against the run's `flags.json` after the fact rather than
  trusted.
- Full suite module-per-process on this cluster: **37/37 modules, 223 passed, 3 skipped,
  zero failures** (`test_latentrl_smoke.py` 13 passed).
- On this machine `.venv` had neither pytest nor ruff; installed. `ruff check .` reports 354
  pre-existing findings repo-wide under ruff 0.16.5 (no `lint.select` in `pyproject.toml`,
  so the default rule set moved) — `agents/latentrl.py` is clean and the one finding in
  `tests/test_latentrl_smoke.py` (C408) is pre-existing at HEAD.
- `tests/test_decode_recovery.py` fails on this cluster with a pre-existing float32/float64
  carry mismatch in `agents/fql.py:51` (`implicit_euler_step`); unrelated to this change.

### Addendum (2026-09-03) — does the behaviour flow MEMORIZE cube? No: train ≈ val, everywhere

First run of `tools/diag_flow_fit.py` (untracked, written but never executed). SLURM
`flowfit` (2491473), one H100, 7 min: 2048 anchors x 512 unclipped N(0,I) latents, exact
ODE at `flow_steps=100` and the distilled one-step head, k_max=512, 250k-row standardized
k-NN index, eps swept in multiples of the median anchor-1NN state distance (1.029). Report:
`$PSM_DATA/logs/diag_flow_fit_cube.json`. Added a `min_mse` block to the tool (per-anchor
`min_j |G(s,u_j) - a|^2` and its min-over-the-whole-ball twin, with the data-to-data value
in the same ball as the scale); everything else is as written.

Lowest MSE per anchor, in units of `(mean|a|)^2 = 0.311^2` (ODE):

| split | own mean | median | p90 | ball@1x mean | data-self@1x mean | own / data-self |
|---|---|---|---|---|---|---|
| train (in Stage-A training set) | **0.0708** | 0.0437 | 0.1486 | 0.0782 | 0.699 | 0.101 |
| val (OGBench held-out split, 100k rows) | **0.0718** | 0.0452 | 0.1504 | 0.0615 | 0.619 | 0.116 |

val/train ratio **1.014** (ODE) and **1.003** (one-step) — the memorization line is flat.
The val split is genuinely held out: `main.py` only ever calls `train_dataset.sample()` for
gradients, `val_dataset` is logging-only, and OGBench ships it as a separate file.

- The flow reproduces a recorded action ~10x better than the nearest *other* dataset action
  in the same ball does (0.071 vs 0.70), so it is well inside the aleatoric noise floor of
  the behaviour conditional, and identically so on states it never saw.
- One-step decode is ~1.5x worse than ODE-100 on the same latents (own mean 0.105 vs 0.071)
  and splits the same way (1.003), so the deployed head loses fidelity but not generality.
- `null` (same ball population, wrong location) sits at 2.6-2.8 against `recall_ball`
  0.24-0.82: conditioning on s carries real information at every radius.
- Ceiling from the stored point preimage: **3.9e-4** — the exact inverse decodes ~180x
  tighter than the best of 512 prior draws, i.e. the residual is coverage of the prior, not
  capacity.
- `min_vs_n` at 1x: recall_ball falls 1.120 (N=1) -> 0.322 (N=512), still not flat, so any
  single min-over-N number is N-bound and must be quoted with N.

**Reading: the flow fits the conditional, it does not memorize.** Whatever is wrong
downstream (the 0.14/0.18 band above) is not a Stage-A overfitting problem.

### Addendum (2026-09-03, later) — the same diagnostic on antmaze: fits worse, memorizes a little

SLURM `flowfit-antmaze` (2491474), one H100, 7 min, through the new
`scripts/slurm/diag_flow_fit.sbatch`. Identical protocol to cube (2048 anchors x 512
unclipped `N(0,I)` latents, ODE-100 + one-step head, k_max=512, 250k-row standardized k-NN
index, the same 0.25x-3x eps sweep). Flow `$PSM_DATA/flow/antmaze-medium-navigate` @ 500000,
`d_a = 8`, `mean|a| = 0.6395`, median 1-NN state distance 1.789. Report:
`$PSM_DATA/logs/diag_flow_fit_antmaze.json`. **The diagnostic needs no preimages** — every
statistic samples `u ~ N(0, I)`; the npz is read only for the `point_preimage_ceiling` line,
so the 881/1M invalid antmaze rows are irrelevant here (the ceiling subsamples 256 rows).

Lowest MSE per anchor, in units of `(mean|a|)^2 = 0.6395^2` (ODE-100):

| split | own mean | median | p90 | ball@1x mean | data-self@1x mean | own / data-self |
|---|---|---|---|---|---|---|
| train | **0.2392** [0.229, 0.252] | 0.1809 | 0.4447 | 0.2602 | 1.125 | 0.213 |
| val (100k held-out rows) | **0.3047** [0.284, 0.330] | 0.2112 | 0.5300 | 0.2994 | 1.200 | 0.254 |

val/train ratio **1.274** (ODE), **1.131** (one-step) — non-trivially above 1, against
cube's 1.014/1.003. `recall_ball` (train, ODE) 0.461 / 0.675 / 1.091 / 1.168 at
0.5x / 1x / 2x / 3x with `null` 2.83 at 1x; one-step is within 0.005 of ODE at every radius
(0.463 / 0.676 / 1.080 / 1.156, null 2.82). Ceiling from the point preimage **3.8e-4**,
essentially cube's. `min_vs_n` at 1x falls 1.683 (N=1) -> 0.675 (N=512), still not flat.
Caveat: antmaze balls saturate the k_max=512 neighbour cap (62% of anchors at 2x, 99% at
3x; cube 55% at 3x), so the 2-3x radii are truncated balls, not full ones.

**Reading vs cube.** Antmaze fits the conditional *worse in absolute terms* — own min-MSE
0.239 vs 0.071, and 0.21 of the data's own local spread vs cube's 0.10 — and it is the one
env with a real, if small, train/val split (1.27x). Not memorization in the damaging sense:
a memorizing flow reproduces train anchors and fails held-out ones outright, and 1.27x on a
statistic whose own bootstrap CI is ±5% is a mild generalization gap on top of a fit that is
still ~4x tighter than the nearest other dataset action. The one-step head is the surprise
in the other direction: on cube it costs 1.49x on own min-MSE, on antmaze only 1.16x, and
its recall curves are indistinguishable from ODE-100 — the distillation gap is a cube
phenomenon, not a general one. pointmaze was **not** run: its Stage-A checkpoint pulled fine
from HF, but the OGBench `pointmaze-medium-navigate-v0` dataset is not on this cluster and
`rail.eecs.berkeley.edu` is unreachable through the proxy.

Figures (`tools/fig_diag_flow_fit.py`, reads every `$PSM_DATA/logs/diag_flow_fit_<env>.json`;
`.pdf` beside each `.png`), plus the generated table `docs/tables/flow_fit.md`:
`PAPER/ICLR/figures/fig_flow_fit_recall.png`,
`PAPER/ICLR/figures/fig_flow_fit_memorization.png`,
`PAPER/ICLR/figures/fig_flow_fit_min_vs_n.png`.

---

<!-- _class: lead -->

### Addendum (2026-09-03, later still) — three audit bugs fixed; point-vs-mixture measured at 500 episodes on cube AND antmaze

Acting on `docs/design/2026-09-03-latent-actor-audit.md` (bugs 1, 2, 4) and
`docs/design/2026-09-03-paper-code-audit.md` (discrepancy #3, the mixture `q_alpha`).
Nothing on the shipped code path changed — verified byte-identical, see below — so the
point arms are a straight reproduction check.

#### Fixes

| # | file:line | what | severity |
|---|---|---|---|
| 1 | `agents/psmflow.py:253` | `flow_actor_loss` read psi at the hardcoded task vector `w`; now `self._index(sampled)`. Under `policy_index=latent` that slot is `d_a` wide, so the old call both mis-typed the head and raised `ScopeParamShapeError` on the first update. | real, unreachable from any shipped config |
| 2 | `agents/psmflow.py:183` | `r_expl, r_emask` were split from `r_next` **after** `normal(r_next, ...)` had consumed it. Now `split(fold_in(rng, 107))`, the same convention as the `104`/`106` branches. | cosmetic (`backup_explore_frac=0.0` by default) |
| 3 | `tools/eval_checkpoint.py:69,106,177,263` | new `_cli_agent_keys()` + `merge_run_config()`: the **run's own `flags.json` supplies the agent config**, typed CLI overrides layered on top; provenance recorded in the report JSON as `agent_config_source`, and `policy_index`/`train_actor` added to the report. | operational — the one path that produced a wrong number **silently** |

Semantics of fix 1, decided rather than asserted: the actor stays **w-conditioned** (it is
the policy the deployed action comes from), the index slot carries **u'** exactly as
`measure_loss` and `gpi_select` read it, and the readout stays `Q = psi(s, u', u_a)^T w`.
That is coherent, so `create()` gained no assert — `policy_index=latent` × `train_actor=true`
now simply works. Fix 3 inherits **only keys the current config already has**, so a stale or
newer `flags.json` can change a value this checkout reads but cannot add or drop a field;
`u_clip`, `acting`, `train_actor`, `policy_index` and `critic_input` can no longer mismatch
in silence. Verified live: the eval of the 09-02 antmaze run printed
`agent config defaults from .../flags.json`, `CLI overrides kept: flow_ckpt_epoch,
flow_ckpt_path, preimage_path, use_point_preimage`, `(run config already matches this
checkout's defaults)`.

#### Tests

New: `tests/test_psmflow_policy_index.py:105,126` (Arm B with a trained actor takes a
finite step and the actor moves; psi is read at the policy index, and the old
hardcoded-`w` call is pinned as the shape error it was),
`tests/test_psmflow_backup_explore.py:74` (the explore keys are folded out of `rng`, not
split from the consumed `r_next`), and `tests/test_eval_checkpoint_flags.py` (11 cases:
flags.json supplies defaults, a typed override — flat or nested — still wins, the schema
stays this checkout's, a different `agent_name` is refused, and no/unreadable/agent-less
flags.json plus no `restore_path` are all no-ops, which is what keeps every recorded
eval500 number reproducible; plus an explicit pin that the `bc` control's semantics do not
move).

Both psmflow tests were confirmed to **fail** against the unpatched file before the fix.
The **default path is byte-identical**: a fixed-seed fingerprint of `sample_step_inputs`
(all seven draws, hex bits) and of a full `update`'s losses and post-step
phi/psi/actor/actor_vf parameters is unchanged before vs after — which is why the point
arms below are a reproduction and not a new measurement.

#### What the mixture arm can and cannot show

Audit row `paper-code-audit.md:75`: the write-up's loss draws `u_i ~ q_alpha(.|s_i,a_i)`,
the epsilon-relaxed posterior; **every shipped result instead used
`use_point_preimage=true`, i.e. `q_alpha = delta_{u*}`.** This is the first time the
mixture arm has been run to 500 episodes. Caveat that limits the ceiling of the result
(audit row `:74`): `configs/inversion/default.yaml` ships `num_clusters: 1`, so the
"mixture" is a **single** Gaussian — the multi-modal refinement the paper describes has
never been run at K>1. What this measures is "point estimate vs a Gaussian around it",
not "point vs a multi-modal posterior".

#### The mixture arm cannot use the published npz — finding, not a choice

`use_point_preimage=false` on `$PSM_DATA/preimages/{cube-single-play,antmaze-medium-navigate}.npz`
is **refused by `main.py:143-148`**: neither sidecar records `inversion.prior_scale`, and
absent means the pre-08-14 likelihood-only target, whose fits sit outside the prior the
latent actor samples from. The canonical npz files therefore support the point arm only.
The mixture arm runs instead on the HPO-corrected npz pulled from the HF dataset repo —
`cube-single-play-a20p6-ps0p69-ns12-N200` (alpha 20.57, prior_scale 0.691, n_steps 12) and
`antmaze-medium-navigate-a26p5-ps0p60-ns5-N200` (alpha 26.51, prior_scale 0.597, n_steps 5),
the 09-01 sweep winners. Both carry `noise_preimage_{mean,cov,weights}` at K=1, both
sidecars were repaired to this machine's flow dir by `hf_preimages.py pull --with-flow`.

That makes point-vs-mixture confounded with npz-vs-npz, so a **third arm** was added at seed
0: `pointps` = POINT preimages read from the SAME corrected npz. `mix` vs `pointps` is the
clean mixture ablation; `point` vs `pointps` isolates the inversion settings.

#### Arms — psmflow stage C, shipped defaults throughout

`batch_size=1024 z_dim=128 num_parallel=2 discount=0.98 tau=0.01 ortho_coef=1000
pessimism_penalty=actor_pessimism_penalty=0.5 mix_ratio=0.5 backup_explore_frac=0.0
acting=actor policy_index=task_vector train_actor=true u_clip=3.0 gpi_decode=onestep
action_critic.enabled=false actor.bc_coeff=1.0 lr_phi=1e-5 lr_sf=1e-4 lr_actor=1e-4
lr_actor_vf=3e-4`; `offline_steps=500000 online_steps=0 eval_interval=50000
eval_episodes=50 save_interval=250000`; flow `$PSM_DATA/flow/<env>` @ 500000.
Every run's own `flags.json` was re-read after launch and all seven fields confirmed.

#### Pre-registered expectations (written before any eval landed)

1. **cube point reproduces 0.220 ± 0.037.** No shipped-path code changed and the golden
   fingerprint is byte-identical, so anything outside that band is a machine/artifact
   difference, not the fixes.
2. **cube mix within noise of cube point.** E4b already measured the mixture-trained
   checkpoint's ranking Spearman and Q spread in the same band as the point arm — the
   mixture does not create ranking signal, so it should not move success.
3. **antmaze (both arms) lands near its BC control.** D2/D3 say `Q_W = psi^T w` carries
   almost no ranking signal, the 09-03 actor audit measured the deployed action as 0.85%
   sensitive to `w` and a 0.961-correlated pass-through of its own noise, and the 09-01
   HPO recorded antmaze coverage 0.036 at its own optimum. **Expected failure**: no
   separation from BC.
4. **`pointps` ≈ `point`.** The corrected inversion changes the BC anchor's target, not the
   critic; if it moves success materially, the standing "the actor is BC with an 8%
   perturbation" reading needs revisiting.

#### RESULT — all ten runs finished 500k; 500-episode evals

| env | arm | seed 0 (500 ep) | seed 1 (500 ep) | mean ± 95% CI (t, across seeds) | pooled |
|---|---|---|---|---|---|
| cube | point preimage, canonical npz | 117/500 = **0.234** [0.199, 0.273] | 113/500 = **0.226** [0.192, 0.265] | **0.230 ± 0.051** | 230/1000 = 0.230 [0.205, 0.257] |
| cube | mixture `q_alpha`, corrected npz | 94/500 = **0.188** [0.156, 0.225] | 100/500 = **0.200** [0.167, 0.237] | **0.194 ± 0.076** | 194/1000 = 0.194 [0.171, 0.220] |
| cube | point preimage, corrected npz *(control)* | 143/500 = **0.286** [0.248, 0.327] | 109/500 = **0.218** [0.184, 0.256] | **0.239 ± 0.102** (3 seeds) | 358/1500 = 0.239 [0.218, 0.261] |
| antmaze | point preimage, canonical npz | 112/500 = **0.224** [0.190, 0.263] | 101/500 = **0.202** [0.169, 0.239] | **0.213 ± 0.140** | 213/1000 = 0.213 [0.189, 0.239] |
| antmaze | mixture `q_alpha`, corrected npz | 101/500 = **0.202** [0.169, 0.239] | 105/500 = **0.210** [0.177, 0.248] | **0.206 ± 0.051** | 206/1000 = 0.206 [0.182, 0.232] |
| antmaze | point preimage, corrected npz *(control)* | 131/500 = **0.262** [0.225, 0.302] | 94/500 = **0.188** [0.156, 0.225] | **0.247 ± 0.131** (3 seeds) | 370/1500 = 0.247 [0.226, 0.269] |

**Seeds 1 and 2 of both `pointps` arms** (added 2026-09-03 later; SLURM `2491503`–`2491506`
for training, `2491507`–`2491510` for the evals; each run's `flags.json` is byte-identical to
its seed-0 sibling except for `seed`, verified after launch). Seed 2 has no column in the
table above, so all three seeds together:

| env | arm | seed 0 | seed 1 | seed 2 | mean ± 95% CI (t, n=3) | pooled |
|---|---|---|---|---|---|---|
| cube | point preimage, corrected npz | **0.286** (143/500) | **0.218** (109/500) | **0.212** (106/500) | **0.239 ± 0.102** | 358/1500 = 0.239 [0.218, 0.261] |
| antmaze | point preimage, corrected npz | **0.262** (131/500) | **0.188** (94/500) | **0.290** (145/500) | **0.247 ± 0.131** | 370/1500 = 0.247 [0.226, 0.269] |

Across-seed SD is 0.041 (cube) and 0.053 (antmaze) — this arm is the **noisiest** of the
three, and seed 0 was its best seed on both envs. Wall clock 50 min (cube) / 56 min
(antmaze) per 500k run, 4–9 min per 500-episode eval.

Comparators. **cube**: PSMFlow re-eval **0.220 ± 0.037**, FB **0.721 ± 0.020** (3 seeds,
`docs/tables/results.md:12`), BC control **0.072** [0.052, 0.098]. **antmaze**: PSMFlow
mixture 09-02, 3 seeds, **0.215 ± 0.041** (`$PSM_DATA/evals/psmflow_antmaze_sd00{0,1,2}.json`);
midi-01 doc-recorded 1-seed PSMFlow **0.222** [0.188, 0.261] and BC **0.090** [0.068, 0.118]
(`docs/HANDOFF.md:1137`); BC control re-measured on this cluster **0.072** [0.052, 0.098]
(`$PSM_DATA/logs/bc_control_antmaze.json`, 36/500). **There is no FB number on antmaze at
any episode count** — FB has only ever been run on cube, so the antmaze rows have no
zero-shot baseline to sit beside.

Two exact-count collisions were checked and are not artefacts: antmaze point sd1 and the
09-02 mixture sd0 are both 101/500 but their per-episode vectors differ in 162 places, and
the cube and antmaze BC controls are both 36/500 with different vectors. Wall clock:
cube 50 min, antmaze 56-59 min per 500k run on one H100 (10 concurrent).

#### Verdicts against the pre-registered expectations

1. **CONFIRMED.** cube point = 0.230 ± 0.051, pooled [0.205, 0.257] — inside the recorded
   0.220 ± 0.037 band. Stronger than that: the antmaze mixture arm reproduced the 09-02
   runs **bit-exactly** — 101/500 and 105/500 with **Hamming distance 0** on the
   per-episode vectors, from independently trained checkpoints on different nodes a day
   apart. Fixes 1 and 2 provably did not touch the shipped path.
2. **REFUTED.** The mixture is **worse**, not equal. Against the point arm on the SAME
   corrected npz, cube drops 0.286 [0.248, 0.327] → 0.194 [0.171, 0.220] — the Wilson
   intervals do not overlap, a real ~0.09 loss. Antmaze drops 0.262 [0.225, 0.302] →
   0.206 [0.182, 0.232], also non-overlapping. E4b showed the mixture does not create
   *ranking* signal; this shows it actively degrades the BC anchor. Read with the K=1
   caveat above: sampling a Gaussian around `u*` instead of `u*` injects noise into the
   CFM target and the TD anchor without adding modes.
3. **REFUTED.** Antmaze does **not** collapse to its BC control: 0.213 and 0.206 against
   0.072 [0.052, 0.098], ~3x and far outside the interval, the same separation cube shows
   (0.230 vs 0.072). Whatever the actor audit's "BC with an 8% perturbation" reading
   explains, it is not that the agent is indistinguishable from BC on success. It remains
   true that neither env comes near cube's FB 0.721.
4. **CONFIRMED at 3 seeds — and the 1-seed reading below it is WITHDRAWN.** With seeds 1
   and 2 in, `pointps` ≈ `point` on both envs, exactly as pre-registered: cube
   **0.239 ± 0.102** vs point **0.230 ± 0.051** (diff +0.009, Welch t=0.36); antmaze
   **0.247 ± 0.131** vs point **0.213 ± 0.140** (diff +0.034, Welch t=1.04). Every t-interval
   overlaps, and so do the pooled Wilson intervals (cube [0.218, 0.261] vs [0.205, 0.257];
   antmaze [0.226, 0.269] vs [0.189, 0.239]). **The provisional 1-seed claim that the
   HPO-corrected inversion buys ~0.05 does not survive replication**: seed 0 was the best
   seed of three on *both* envs (cube 0.286 vs 0.218/0.212, antmaze 0.262 vs 0.188/0.290),
   and the across-seed SD of this arm (0.041 / 0.053) is about the size of the effect that
   was read off it. Stage-B inversion quality is **not** shown to move Stage-C success, and
   the standing "the actor is BC with an 8% perturbation" reading does not need revisiting
   on this evidence. Do not quote the +0.05 anywhere — it was a one-seed artefact, and this
   is the second time in this entry that a single seed pointed the wrong way.

   *Consequence for verdict 2:* the mixture is still the worst arm, but with the correct
   3-seed `pointps` comparator the cube gap narrows from 0.286 → 0.194 to
   0.239 ± 0.102 → 0.194 ± 0.076 (Welch t=1.82) and the antmaze gap from 0.262 → 0.206 to
   0.247 ± 0.131 → 0.206 ± 0.051 (t=1.33) — **neither is significant across seeds** now,
   though the pooled Wilson intervals still separate on cube ([0.218, 0.261] vs
   [0.171, 0.220]) and now touch on antmaze ([0.226, 0.269] vs [0.182, 0.232]). Verdict 2's
   direction survives; its "the intervals do not overlap" phrasing was resting on the
   single lucky `pointps` seed and should be read as *point ≥ mixture, cube significant
   pooled, antmaze not*.

Net ordering at 3 seeds: **point/corrected ≈ point/canonical > mixture**, consistent
across both environments — the two point arms are indistinguishable and the paper's
`q_alpha` is the worst of the three (significantly so only on cube, pooled).

#### Artifacts

Run dirs under `$PSM_DATA/exp/PSMFLows/`: `psmflow_{cube,antmaze}_{point,mix}/sd00{0,1}_s_<jobid>.*`
(jobs 2491480-2491487) and `psmflow_{cube,antmaze}_pointps/sd00{0,1,2}_s_<jobid>.*`
(seed 0: 2491488-2491489; seeds 1-2: 2491503-2491506, evals 2491507-2491510).
Eval JSONs in `$PSM_DATA/logs/`: `eval500_psmflow_{cube,antmaze}_{point,mix,pointps}_sd<k>.json`,
`eval500_psmflow_antmaze_mix_20260902_sd0.json` (the fix-3 validation re-eval),
`bc_control_antmaze.json`. All registered in `tools/make_tables.py` (not run — its `LOGS`
default is still the dead midi-01 path, so it needs `--logs $PSM_DATA/logs`, and
`docs/tables/results.md` is cube-only until someone decides how to split the envs).

Tests: `.venv/bin/python -m pytest <module> -q` per module over all 38 test files, on a
compute node (the login node thrashes at load 95 and one module ran >15 min there):
**38/38 exit 0, 237 passed, 3 skipped, 0 failed** — the skips are the
`PSMFLOWS_STAGE_A_CKPT`-gated ones. `ruff check` on every touched file reports exactly the same findings as
the unmodified versions — 14 across `agents/psmflow.py` + `tools/eval_checkpoint.py`, 7 in
`tools/make_tables.py`, 2 across the two edited test modules, all pre-existing, and
`tests/test_eval_checkpoint_flags.py` clean: **zero new**.


---

<!-- _class: lead -->

## 2026-09-01 (later) — inversion HPO redone on the corrected target; cube regenerating, antmaze not

The 08-28 50-trial sweeps and every mixture npz they produced measured the OLD inversion
(un-squared likelihood, alpha^2 Laplace covariance). Both were redone on the corrected
target, same settings as 08-28 (`n_rows=256 sample_k=32 cov_k=16`, `flow_steps=100`,
`num_samples=128` pinned, alpha in [2,100] log, `prior_scale` in [0.3,1.0],
`n_steps` in [5,20]), 50 trials each. Logs: `logs/hpo_{cube,antmaze}_fixed.log`.

| env | alpha | prior_scale | n_steps | coverage@16 | decode_mix | ESS | E\|u\|^2 / d_a |
|---|---|---|---|---|---|---|---|
| cube | 25.26 | 0.677 | 17 | **0.982** | 0.2165 | 68.2 | 1.04 |
| antmaze | 26.51 | 0.597 | 5 | **0.036** | 0.2653 | 8.5 | 1.07 |

Three readings.

**Both incumbents sit exactly on the decode budget** (0.2165 vs 0.22; 0.2653 vs 0.28). The
objective maximizes coverage subject to a fidelity hinge, so the optimizer widens until the
hinge binds. That is a frontier position, not evidence of latent multiplicity: the exact
inverse of a diffeomorphism is unique, and the corrected alpha sweep reproduces the same
width-is-temperature collapse (cube cov_k16 0.945 at alpha=20 down to 0.003 at alpha=3200).

**Typicality is what the fix actually bought.** E\|u\|^2 / d_a is now 1.04 (cube) and 1.07
(antmaze) against the prior's 1.0. The shipped npz built under the old target sits at
3.92/5 = 0.78 -- measurably UNDER-dispersed. Lemma "typicality" says an exact flow's
preimages are prior-distributed; the corrected target nearly satisfies it, the old one did
not. This is the first mixture arm whose latents look like the ones the actor emits.

**Antmaze was not rescued.** Coverage 0.036 at its own optimum against cube's 0.982. The
corrected inversion does not close the cube/antmaze gap, consistent with the 08-30 Jacobian
probe having already excluded local geometry. Regenerating 1M antmaze rows at 3.6% coverage
would cost ~13 h for a file already measured useless as augmentation; **not run**.

**Cube is regenerating** at the runner-up alpha=20.57 / prior_scale=0.691 / n_steps=12
(cost -0.9805 vs the incumbent's -0.9819, coverage 0.981 vs 0.982, decode 0.2117, ESS 62) --
a statistical tie at ~23 h instead of ~33 h, since EM cost scales with n_steps.
`num_samples` 128 -> 200 for the real file (the HPO pins it during the width search so the
optimizer cannot buy coverage with compute). Output
`preimages_cube_single_fixed_a20p6_ps0p69_ns12_N200.npz`, tmux `pre-cube-fixed`, ETA
2026-09-02 ~19:00 CDT.

**Why this is worth running at all**, given the mixture arm's record: every mixture result
on file is confounded. Pre-08-14 arms had no prior factor; the 08-28 HPO and its npz had the
un-squared target and alpha^2 covariance; E4b's probe used that same npz. So the mixture has
never been tested with a proposal that has a Laplace mode at its own centre. The mechanism
under test is NOT "the preimage is a set" -- that is dead -- but whether a wider, typical,
faithful latent target regularizes psi. Against it stands the transition-label bias:
any u~ != u* pairs a perturbed action with the recorded s'.

**Gate, pre-registered.** Judge the resulting Stage-C runs on representation identifiability
before success: D1 policy-ranking Spearman (point arm 0.10, Arm B 0.079, old mixture 0.054),
D3 relative Q spread (1.1% / 0.9% / 0.86%), and the actor's percentile under Q (44th). If
those do not move, the mixture does not create ranking signal and the arm closes for good.

**Point preimages are unaffected and this was measured, not assumed.** `noise_preimage_point`
is the backward-ODE solution and takes no alpha or prior_scale; recomputing 4096 cube rows
with today's code reproduces the 08-04 npz **bit-identically** (max abs diff 0.000e+00). So
no shipped result, and neither E3 arm, is invalidated by the inversion bugs.

---

<!-- _class: lead -->

## 2026-09-01 — E3 landed: the paper's construction does not work on cube

Closes `docs/plans/2026-08-31-interface-fork-experiments.md`. Both arms ran 3 seeds ×
500k steps, evaluated at 500 episodes on the post-P0.2 pinned-stream harness. Table
rows regenerated from the eval JSONs; CIs are the repo's t95-across-seeds.

| arm | success | matched control |
|---|---|---|
| Arm A — `backup_explore_frac=1.0`, `acting=actor` | **0.171 ± 0.113** (3) | point arm, actor: 0.220 ± 0.037 (5) |
| Arm B — ψ(s, u, u′), no actor, `acting=gpi` | **0.083 ± 0.191** (3) | point arm, gpi: 0.054 ± 0.032 (5) |

Controls: BC per-step prior 0.068 [0.049, 0.093]; FB 0.721 ± 0.020; FQL 0.949 ± 0.063.
Each arm is quoted against the control that ACTS the same way — Arm B selects by latent
argmax, so the actor-arm 0.220 is not its comparison and using it would price the removal
of the actor as if it were the index change.

**Arm A is within noise of its control.** Making every bootstrap action a genuine p0
decode — the hypothesis Prop. "insample" needs for C=1, which no shipped run ever
satisfied — changes nothing. Pre-registered expectation held: a correct backup
distribution does not by itself create ranking signal.

**Arm B lands at BC level.** 0.083 ± 0.191 against a gpi control of 0.054 ± 0.032, with a
seed spread (0.006 / 0.160 / 0.084) far wider than any effect. It does not separate from
the 0.068 BC control, and is nowhere near the 0.45 bar or FB's 0.721. The pre-registered
reading fires: **the writeup's construction is refuted on its own terms on cube.**

What this does NOT isolate: Arm B changes the ψ index and drops the actor together, per
§8, so it does not price the policy-index idea alone. What it does establish is that the
algorithm as written does not work here, and that is coherent with E1 — Arm B's only means
of choosing an action is ranking latents by ψᵀw, and E1 showed ranking is the broken
faculty. The oracle reached 0.934 selecting from the same action set these agents fail in.

Taken with E1 and E2, the three experiments converge on one statement: the interface is
not the problem (oracle-aim 0.934), the decoder is not the problem (exact decode costs
−0.038 ± 0.027 paired), and neither of the paper's two deviations was load-bearing. The
open problem is a critic that can rank latents.

### Infrastructure, same day

`/var/local` (24 GB) hit 100% at 16:42 and killed all six runs mid-training with
`OSError: [Errno 28]`, losing ~3.5 h; they were relaunched on `/data-local` at 17:40 and
the numbers above are from those clean 500k runs. `scripts/launch_psmflow.sh` now takes
`STORE=`, defaulting to `/var/local` so older launch lines reproduce. 11.82 GiB of
finished July experiments were uploaded to `amsks/psmflows-checkpoint-archive`, verified
file-by-file, then deleted (`docs/reference/archived-checkpoints.md`).

Two mid-training `*_ep250000.json` evals of the killed runs exist in the logs. They are at
half the specified budget and `tools/make_tables.py` now globs `sd?` rather than `sd?*` so
they cannot be pooled into the arm rows.

---

<!-- _class: lead -->

## 2026-08-31 — oracle-aim: the reachable set is fine (0.934), the loss is ranking; paper-faithful arms launched

Plan and pre-registered readings: `docs/plans/2026-08-31-interface-fork-experiments.md`.
Everything below is cube unless stated. Table regeneration: commit `8e89d76`.

### The audit that set this up

A three-way audit of code vs the formal writeup (`git show 5249267:PAPER/main.tex`)
found the shipped agent replaced both hypotheses of Prop. "insample" (C=1): the
bootstrap latent is the actor's (`psmflow.py:139`), and the ψ index slot carries the
task vector, not a policy latent (`psmflow.py:106,111`). So every negative Stage-C
number to date tested latent-space FB, not the paper's construction. Three inversion
bugs also surfaced and are now FIXED (commits `313948e`, `3ef0afe`): the likelihood
norm was un-squared, the Laplace proposal used α²JᵀJ instead of the spec's 2αJᵀJ
(10× too narrow at α=20), and `preimage_valid` never reached sampling (u=0-repaired
rows trained on wrong pairs). Validation at matched fidelity: cube ESS 112.9 → 132.6,
antmaze ESS 8.8 → 18.8 with coverage 0.016 → 0.064 — the ESS collapse was partly this
bug, not purely d_a=8 geometry; antmaze mixture remains short of usable. Default alpha
retuned 20 → 50 with a frontier table in `configs/inversion/default.yaml`. All mixture
npz files and both 08-28 HPO sweeps predate the fix and measure the old target.

### E1 — oracle-aim rollout (`tools/diag_oracle_aim.py`, 500 ep, K=512, ODE-100)

Execute, at each step, the decoded prior latent closest to a frozen FQL expert's
action. **oracle-aim 0.934 [0.909, 0.953]** against the expert's own 0.960 [0.939,
0.974]; random-latent floor 0.086 (consistent with the BC control). Mean min-distance
to the expert action over K=512: 0.062. Pre-registered fork: ≥0.7 ⇒ **cannot aim** —
the flow's reachable action set contains near-expert behavior and the entire Stage-C
loss is latent *selection*. No flow retraining is indicated by this number; the target
is a critic that can rank.

### E2 — ODE re-eval of the 5 shipped checkpoints (post-P0.2 seeding, 500 ep × 5 seeds)

| arm | one-step | ODE-100 |
|---|---|---|
| actor | 0.220 ± 0.037 | 0.182 ± 0.019 |
| gpi | 0.054 ± 0.032 | 0.044 ± 0.043 |

Exact decode helps nowhere (slightly worse everywhere; even the random-latent floor
drops, 0.086 → 0.014). The one-step distilled decoder is exonerated as a loss source;
its 0.0886 preimage decode error stays a bookkeeping caveat only. The re-measured
one-step control (0.220 ± 0.037) supersedes the pre-seeding-fix 0.236 ± 0.071 for
comparisons; sd0's recorded 0.318 is not reproducible post-fix (0.240).

### E3 — paper-faithful arms (training today; results below when evals land)

Arm A: `backup_explore_frac=1.0` — the exact `u′~p₀` bootstrap, all else shipped
config, acting=actor, 3 seeds. Arm B: `policy_index=latent` — ψ(s, u, u′) with a
fresh prior policy-index latent, `train_actor=false`, acting=gpi, 3 seeds. Both point
arm, pessimism 0.5, flags verified from the runs' own flags.json. Pre-registered:
Arm A alone likely within noise of 0.22 (a correct backup does not create ranking
signal by itself); Arm B is the first number Prop. insample applies to — clear FB's
0.721 and the mechanism is demonstrated; land ~0.2 and the C=1 construction is
refuted on its own terms on a field where E1 proves a winning selection exists.

### Arm results at step 250k (training was torn down externally at ~16:40; 500k finals do not exist)

All six arm trainers stopped at ~16:40 with tmux sessions removed and no note; last
checkpoints are at 250k (armA sd0-2, armB sd0). armA sd2's params_250000.pkl was
mid-write at the kill and is truncated — unrecoverable, so Arm A is a 2-seed number.
armB sd1/sd2 died before their first checkpoint and were relaunched to 250k
(`psmflow_paperfaith_armB_relaunch_20260831`, slowed ~10x by GPU contention; their
evals follow separately). Everything below is
**at-step-250k, not final** — but the shipped agent's own in-loop curves were flat by
250k, so these are read as strong signal, weak proof.

| arm | 500-ep success | seeds |
|---|---|---|
| Arm A (u'~p0 bootstrap, actor acting) | 0.189 ± 0.089 (0.196 / 0.182) | 2 |
| Arm B (psi(s,u,u'), no actor, gpi K=64) | **0.006** [0.002, 0.018] | 1 |
| Arm B floor control: same path, gpi_num_u=1 (random pick) | 0.080 [0.041, 0.150] (100 ep) | 1 |

- **Arm A = the pre-registered null.** The correct backup distribution alone lands
  inside the shipped agent's noise band (control 0.220 ± 0.037). A right backup does
  not create ranking signal.
- **Arm B anti-selects.** The K=1 control proves the eval path sound (0.080 ≈ the
  0.086 random-latent floor); letting the critic pick among 64 candidates drops
  success 13x below random. Probes on the same checkpoint agree:
  - D3 (adapted for latent index, `d3_q_landscape_armB_ep250k.json`): Q relative
    spread over 512 prior draws **0.0092** of |Q| — flatter than the shipped agent's
    0.011. H2 again.
  - D1-analog (`d1a_latent_ranking_armB_ep250k.json`, new
    `tools/diag_latent_ranking_oracle.py`): Spearman of psi-ranking vs FQL-oracle
    distances over the same K=128 candidates: **0.079** mean (old D1: 0.10, chance).
    The critic's argmax sits 0.359 from the oracle action when 0.086 was available
    in the candidate set — worse than the median candidate (0.265).
- Reading, per the pre-registration: at 250k the C=1 construction shows **no ranking
  signal and active anti-selection** — the failure mode survives the paper-faithful
  bootstrap and index. Caveats before calling it refuted: half-trained, one seed,
  and the E2 finding that gpi acting under-performs even for the shipped agent.
  The honest verdict gate is the relaunched-seed evals plus (if pursued) a 500k
  retrain; but nothing in these numbers points at the bootstrap/index substitutions
  as the missing ingredient, and everything continues to point at the critic's
  inability to rank latents — now measured directly against an oracle over the very
  candidate set it deploys on.

### E4 (same evening) — the ranking failure is the deployment scheme, not our critics

**E4a, the known-good-ranker control** (`e4a_fql_critic_aim_cube_sd0.json`): the E1
harness with one change — candidates scored by the frozen FQL expert's OWN critic
(ensemble mean, its own reduction) instead of oracle distance. Success **0.032**
[0.020, 0.051] — below the one-step random floor (0.086). The nuance that explains
it: per-step Spearman vs the oracle ranking is bimodal (mean 0.285, median 0.346,
54% of steps above 0.3, p10 −0.33) — the expert's critic ranks *moderately well on
most steps*, but argmax over K=512 reliably lands on its most overestimated
candidate: picked action 0.426 from the expert vs 0.105 available. Winner's curse at
K=512. Pre-registered branch: **decode-then-score is dead as a deployment scheme;
even a proven ranker fails under per-step best-of-K GPI.** This exonerates ψᵀw as
the specific culprit — lambda-rank, FB-graft, Arm B, and the FQL critic all fail
the same way — and kills the "critic with a direct action pathway" fix on its own
pre-registration.

**E4b, mixture-checkpoint probes** (`d1a_latent_ranking_mixhpo_ep500k.json`,
`d3_q_landscape_mixhpo_ep500k.json`): the mixture-trained 500k checkpoint shows
ranking Spearman **0.054** and Q spread **0.86%** of |Q| — the same band as the
point arm (0.10 / 1.1%) and Arm B (0.079 / 0.9%). As pre-registered: mixture
training does not create ranking signal; the mixture arm stays closed.

**Where this leaves the fork:** E1 (oracle 0.934) + E4a (every learned critic
fails at K=512 argmax) means the remaining live directions are (a) actor-based
improvement in latent space — no argmax over a large candidate set, the actor
moves smoothly against the (weak but locally usable) critic gradient, which is
where the eps=0.05 residual's 0.96 peak also lives; and (b) small-K or
regularized selection (K where the winner's curse is weaker than the ranking
signal — the E4a Spearman distribution says an optimal K may exist and is small).
Both are specced next-step candidates, not started.

### Housekeeping from the 16:40 incident

The teardown cause was `/var/local` filling (killed six runs; since cleaned to
47%). Repo policy going forward: experiment STORE on `/data-local`. The armB
sd1/sd2 relaunches to /var/local died of the same disk exhaustion at ~100k with no
checkpoint and are DROPPED — Arm A/B seed replication rides on the parallel
full-500k b-runs (`psmflow_paperfaith_arm{A,B}_20260831b`, 3 seeds each on
/data-local, ~300k/500k as of this entry); their Arm B results supersede the
250k sd0 numbers above when they land.



---

<!-- _class: lead -->

## 2026-08-30 — the agent we have been running is FB, not PSM; the preimage is exact for a decoder we never use

Two things came out of reading the ICLR draft against the code, and one port followed.

### The draft describes a method we are not running

`PAPER/ICLR` was re-added (skeleton: prewriting form + intro + preliminaries
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

### Also

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

The W4/l1stab arc, verified end to end. All artifacts in
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

### Also

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

## 2026-08-10 — LatentFlowPSM measured end to end; ICLR figures F1-F4

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

## 2026-08-05 — ROOT CAUSE: the fixed-u family is structurally non-goal-covering; Rung-1 is dead on navigate data

All Stage-C variants (pointpre x2, mixture, pointpre1M, mix0 audit — 11 runs) read **0.0
success** on pointmaze task1; cube Stage-C floors at 0.02–0.06. D1–D3 pass. Root cause
below. **It is not a bug anywhere — the policy family itself has no goal-reaching
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

## 2026-08-04 — full audit + roadmap; the D3 ESS statistic was wrong (again), and it changes the story back

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

### Also fixed (audit P0s)

- `tools/latent_q_sanity.py` (D4): the 10k relabel batch was **unseeded** (global
  np.random) — the gate number changed run to run. Now seeded off `cfg.seed`, same pattern
  as D3. D4 is a spec gate; its number must be a function of the checkpoint.
- `scripts/launch_fb_cube.sh` / `launch_psm_cube.sh` ran bare `python` (= 2.7 on midi-01).
  Now `$REPO/.venv/bin/python`.
- `agents/affine_psm.py` `infer_w_goal`: the constraint-set permutation used
  `np.random.default_rng()` (OS entropy) — identical seed + weights gave different `w_inf`
  and eval success. Now seeded off the `seed` argument.

### Later the same day — D2 recovered (PASSES both envs), preimage analyzer landed

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

### Fixed

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

## 2026-07-29 — PSMFlow v1 Tasks 3–8 done; cube preimages landed; D3 ESS gate FAILS

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

## 2026-07-28 — PSMFlow v1 Tasks 1–2; two numerics bugs cleared

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

### The residual NaN ESS — fixed

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

## 2026-07-26 — flowBC actor for affine PSM; reference-HP audit

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

### Known issue, NOT fixed (flagged, out of scope here)

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

## 2026-07-15 — FB agent ported to JAX (bit-exact)

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

## 2026-07-13 — fresh from-scratch parity runs

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

### Next (from 2026-07-13)

1. **Read the 1M seed-2 result** — does it reach the ref's 0.7–0.8 ceiling? If
   yes ⇒ we DO have parity, just needed budget; the "systematic gap" was a
   500k-cap artifact + seed variance. If it plateaus ~0.4 ⇒ real ceiling gap.
2. Let seeds 0/1 finish 500k; re-run `compare_multiseed.py` for the 3-seed table.
3. If ceiling confirmed, consider 1M runs for seeds 0/1 too (one-per-GPU) to
   get a real 3-seed peak distribution vs the reference's spread.
4. GPUs 1 & 3 usable; 0 & 2 were busy/full (other users) at the time.

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

## How we got here

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

## Code changes

- **Committed & pushed** `31ed43d` "PSM: reference-parity fixes (eval z-inference + masks) +
  audit tooling": masks always-γ (`agents/psm.py`), eval z-shift/relabel (`main.py`,
  `config.yaml`), `eval_interval 100k→20k`, `docs/`, `scripts/launch_psm_cube.sh`.
  - **UNCOMMITTED** (working tree): `proto_table_path` hook — `agents/psm.py` `create()` loads a
  transplanted table when `agent.proto_table_path` set; `configs/agent/psm.yaml` declares it
  (`null` default). 15/15 PSM tests still pass.

---

## Runs

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

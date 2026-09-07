# Three specs: is the latent navigable, and can InFOM's tricks help

Date: 2026-09-03 · Branch: `feat/inversion-integration` · Machine: kisski (SLURM/H100)
Status: **specification only — nothing here has been run.**

Read `docs/COMPENDIUM.md` first. These three specs are ordered and **gated**: S1 is a
measurement that decides whether S2 and S3 are worth building. Do not run them in parallel
"to save time" — S1's outcome changes what S2 and S3 should be.

## The contradiction these specs exist to resolve

| fact | source |
|---|---|
| Near-expert actions exist in the decode family at nearly every step: oracle best-of-512 = **0.934** [0.909, 0.953], vs 0.086 random floor. Mean min-distance to the oracle action over K=512 is **0.062** against mean `‖a‖ = 0.875`. | §4.3 (E1) |
| Every *learned* function of `u` reads that relief as ~1% noise: D3 spread **0.011**, D1 0.9% (Spearman 0.10, p=0.78), Arm B 0.9%, E4b 0.86%, and the fresh DSRL-SAC scalar critic **1.1–1.5%**. | §4.6, §4.7, HANDOFF 2026-09-03 |
| Explicit best-of-K=512 argmax is dead even with a proven ranker: **0.032**, below the random floor, because argmax lands on the most overestimated candidate (picked 0.426 when 0.105 was available). Per-step Spearman vs oracle is bimodal: mean 0.285, median 0.346, 54% above 0.3, p10 −0.33. | §4.6 (E4a) |
| The preimage is a *faithful* coordinate, not an arbitrary one: point preimage decodes back to `a` with ODE-100 error **0.00012**; posterior width is the inversion temperature, not decoder degeneracy; Jacobian cond 1.79 (cube) / 6.67 (antmaze). | §4.8, §4.9 |
| But it carries no *policy*: within-episode preimage variance / marginal = **0.99**, lag-1 autocorr 0.27, ~0 by lag 50; 0/233 latents reach the pointmaze goal. The expert's direction is driven by a goal **not in the observation**, so that variance is forced into `u` independently each step. | §4.11 |
| A **0.05** action-space residual takes latentrl from 0.142 ± 0.025 to **0.905 ± 0.020**. A tiny smooth correction in `a` beats unrestricted freedom in `u` by 6×. | latentrl W4 arm |

Reading: good latents exist, but nothing learnable locates them. Either (a) the value
landscape over `u` at fixed `s` is rough — no smooth critic or actor can navigate it — or
(b) it is navigable and every attempt so far was destroyed by taking an argmax over
samples of a noisy learned function. **S1 distinguishes (a) from (b).**

---

# S1 — Is the value landscape over `u` navigable? (`tools/diag_latent_smoothness.py`)

The gate. Forward passes only, no training, no Stage-C checkpoint.

**Key property: this measures the FROZEN STAGE-A FLOW, not any agent.** The only learned
objects are the frozen flow (for `decode`) and a frozen FQL expert (for the oracle score).
So it is valid for every Stage-C arm simultaneously and cannot be invalidated by a
retrain.

### Implementation

New file, modelled on `tools/diag_latent_ranking_oracle.py` — copy its on-path state
harvest (roll the oracle, snapshot every 10 steps), its oracle loading (`+oracle_path`,
read `flags.json`, `restore_agent`), and its `report_out` handling. **Do not edit the
existing tool**; its outputs are already on record.

Differences from that tool:
- Build the psmflow agent from `agent.flow_ckpt_path` only. **Do not** call
  `restore_agent`, `infer_eval_z` or `agent.psi` — no Stage-C checkpoint is involved and
  `restore_path` must not be required.
- `N_STATES = 64`, `K = 512` (match E1's K, not the ranking tool's 128).
- Report for both decoders: `agent.gpi_decode=onestep` and `=ode` with
  `agent.flow_decode_steps=100`. The one-step head is what deploys; the ODE is the true map.

Per state `s`: draw `u_i = clip(N(0,I), ±u_clip)`, `i = 1..K`; decode `a_i = agent.decode(obs_b, u)`;
score `v_i = −‖a_i − a*‖` with `a*` the frozen FQL expert's action at `s` (identical to E1's
oracle). Then compute, per state, and aggregate mean/median across states:

1. **`knn_r2_u`** — leave-one-out k-NN regression of `v` on `u`, `k ∈ {5, 10, 20}`:
   `R² = 1 − Σ(v − v̂)² / Σ(v − v̄)²`. How much of the value is predictable from position
   in latent space at all. **This is the headline number.**
2. **`knn_r2_a`** — the same with neighbours in *action* space. `v` is by construction a
   smooth function of `a`, so this must come out ≈1; it is the estimator's calibration
   control. If it does not, the estimator is broken and nothing else in the report is
   readable.
3. **`basin_spearman`** — per state, Spearman between `v_i` and `−‖u_i − u_best‖` where
   `u_best = argmax_i v_i`. High = one basin an actor could descend into; ~0 = the good
   set is scattered.
4. **`topdecile_dispersion_u`** — mean pairwise `‖Δu‖` among the top 10% by `v`, divided by
   the same over a random 10%. ≈1.0 scattered, <1 clustered. Report the action-space twin.
5. **`local_lipschitz`** — over random pairs: median and p95 of `|Δv| / ‖Δu‖`, plus
   `‖Δa‖ / ‖Δu‖` (the decoder's own local expansion) and `corr(‖Δa‖, ‖Δu‖)`. The last one
   is the mechanism: a value-monotone transport preserves local geometry.

Persist everything through `report_out` (repo rule — a 2026-08-03 result was lost to
stdout-only output), and dump the raw `(u, a, v)` arrays to a sibling `.npz` so the
statistics can be recomputed without re-running the rollouts.

### Run

Both published envs, `$PSM_DATA` paths as in `scripts/eval500.sh`:

```
MUJOCO_GL=egl .venv/bin/python tools/diag_latent_smoothness.py agent=psmflow \
    env_name=cube-single-play-singletask-v0 \
    agent.flow_ckpt_path=$FLOW agent.flow_ckpt_epoch=500000 \
    agent.gpi_decode=ode agent.flow_decode_steps=100 \
    +oracle_path=$FQL_RUN +oracle_epoch=500000 \
    report_out=$PSM_DATA/logs/d5_latent_smoothness_cube_ode.json
```

Cost: ~10–20 min per (env, decoder) on one GPU. Four runs total.

### Pre-registered predictions

| statistic | "scrambled" (a) | "navigable" (b) |
|---|---|---|
| `knn_r2_u` (k=10) | < 0.05 | > 0.5 |
| `knn_r2_a` | > 0.9 either way (control) | same |
| `basin_spearman` | < 0.1 | > 0.4 |
| `topdecile_dispersion_u` | 0.95–1.05 | < 0.8 |
| `corr(‖Δa‖, ‖Δu‖)` | < 0.3 | > 0.7 |

### Decision rule

- **Scrambled** → the latent action space is not navigable by any smooth policy. Live
  hypothesis #1 dies by measurement instead of by another 3×500k-step run, and D1/D3/E4b/
  DSRL collapse into one explanation. S2 and S3 both still have to steer in `u`, so
  **neither is worth building as specified**; the lever moves to Stage A (a
  value-monotone transport — minibatch-OT CFM — which invalidates every preimage: 1 h
  Stage A + 4–19 h Stage B per env) or to the ε-residual arm as the actual method.
- **Navigable** → the argmax, not the geometry, is the killer. Build S2 (it is then the
  direct fix) and S3-a.
- **Intermediate (0.05–0.5)** → build S2 only; report the number and re-gate.

---

# S2 — Implicit GPI: expectile distillation instead of an argmax over samples

InFOM (arXiv 2506.08902) replaces `max_z Q(s,a,z)` with an **expectile regression** of a
single head onto `Q(s,a,z)`, `z ~ p(z)` — a soft max that never evaluates an argmax over
samples. Our `policy_index=latent` arm takes exactly that argmax over the index panel
(`gpi_select`, K×K pair scan), and E4a says argmax over samples of a learned function is
what kills us. This is the principled version of live hypothesis #2 ("small-K or
regularized selection", untested).

### Implementation — `agents/psmflow.py`

New config keys in `get_config()` beside `policy_index`:

```python
index_agg="max",        # max | expectile -- how psi's index slot is aggregated
expectile_mu=0.9,       # upper expectile for index_agg=expectile
index_panel=16,         # u' draws per state used to form the aggregate
q_dist=dict(hidden_dim=512, hidden_layers=1, embedding_layers=2, lr=3.0e-4),
```

Assert at `create()`: `index_agg="expectile"` requires `policy_index="latent"` — under
`task_vector` there is no index to aggregate. Follow the `critic_input` precedent in
`agents/latentrl.py`: back-fill both keys when restoring a `flags.json` written before
this change, so every existing checkpoint still loads.

Add a scalar head `q_dist(s, u)` trained by expectile regression against the
index-conditioned readout:

```
L(q_dist) = E_{(s,u)~batch, u'~p0}[ L²_μ( stopgrad(psi(s, u', u)ᵀw) − q_dist(s, u) ) ]
L²_μ(x)  = |μ − 1{x < 0}| · x²
```

`u'` is `index_panel` clipped prior draws per batch element. `μ > 0.5` fits the upper
expectile, so `μ → 1` approaches `max_{u'}`. `psi` enters under `stopgrad` — this head is
a distillation target, it must not push gradient back into the measure.

Three call sites, and **change only these**:
1. `gpi_select`: under `index_agg=expectile`, drop the K×K pair scan entirely and return
   `u_cand[argmax(q_dist(s, u_cand))]`. K× cheaper as a side effect.
2. `flow_actor_loss`: replace `Qs = (self.psi(obs, self._index(sampled), u_a) * w).sum(-1)`
   with `q_dist(obs, u_a)`. The actor then climbs the distilled soft-max rather than one
   random index draw.
3. `measure_loss`: **leave untouched in v1.** One change at a time — the backup keeps its
   current semantics so a regression is attributable.

Add `q_dist_spread_rel` to the eval-time logging, computed exactly like
`action_critic_spread` (`spread_candidates` latents at the same states, `std / mean|Q|`)
so the live signal is comparable to the `ac_q_spread_rel` and `q_spread_rel` numbers
already on record.

### Arms

Cube first. Baselines to quote beside every number: Arm B (`policy_index=latent
train_actor=false acting=gpi`, at BC level, 0.9% spread) and the BC control from the same
flow checkpoint (`agent=fql agent.bc_only=true`).

| arm | flags | seeds |
|---|---|---|
| B-max (control, on record) | `policy_index=latent train_actor=false acting=gpi` | 2 |
| B-exp μ=0.7 | `... index_agg=expectile expectile_mu=0.7` | 2 |
| B-exp μ=0.9 | `... expectile_mu=0.9` | 2 |
| B-exp μ=0.98 | `... expectile_mu=0.98` | 2 |

Start with μ=0.9 only (2 seeds, ~3 h each) and expand if `q_dist_spread_rel` moves.
`tools/eval_checkpoint.py` now inherits the run's own `flags.json`, so the eval line no
longer has to re-type these flags — but re-read the run's `flags.json` after launch anyway
and confirm the values landed.

### Pre-registered predictions

- **Necessary condition, visible within ~50k steps:** `q_dist_spread_rel` rises above
  ~5%. If it sits at the same 1% band as every previous critic, the flatness is not caused
  by the argmax, S2 is dead, and the run should be killed early rather than carried to
  500k.
- **Expected failure mode:** μ=0.98 reproduces the winner's curse (expectile → max) and
  lands at or below B-max. If μ=0.7 and μ=0.98 both fail while μ=0.9 works, that is a
  knife-edge result and needs a third seed before it goes in a table.
- Success = beating the BC control at 500 episodes with non-overlapping 95% CIs. Anything
  short of that is a negative, and goes in the settled-negative table.

---

# S3 — The intention latent

InFOM's encoder is `p_e(z | s', a')` — intention inferred from the *next* transition, on a
consistency assumption that consecutive transitions share an intention — trained by
`L_Flow + λ·D_KL(p_e(z|s',a') ‖ p(z))`. §4.11 diagnosed our failure as an unobserved goal
whose variance is forced into `u` per-step; that is precisely the variable this encoder is
built to recover. And `z` does not fight Prop. `isometry`: `u` must stay `N(0,I) ⊥ s`, and
`z` is a separate slot at a different timescale.

## S3-a — Premise test (`tools/diag_intention_premise.py`). Cheap. Do this before S3-b.

Does a temporally consistent latent carry information in *our* datasets? No flow, no
occupancy model, small MLPs.

Train jointly on the offline dataset:
- encoder `q(z | s', a')`, diagonal Gaussian, `d_z = 8`;
- a predictor of a *discounted-future* state `s_f` from `(s, a, z)` — sample the offset
  `k ~ Geom(1−γ)` with γ=0.99, plain Gaussian log-likelihood (deliberately not a flow);
- objective: prediction NLL + `λ·KL(q ‖ N(0,I))`, λ swept over {0.1, 1.0, 10.0}.

Report (all via `report_out`):
1. **`kl_per_transition`** at convergence. Collapse to ~0 means `z` carries nothing and
   InFOM's premise fails on our data — a publishable negative on its own, given §4.11.
2. **`predictive_gain`** — future-state NLL with inferred `z` vs with `z ~ prior` and vs
   `z = 0`. Δ ≈ 0 means the intention is not needed to explain the future.
3. **`z_autocorr(lag)`** for lag ∈ {1, 5, 10, 50}, within episodes. **The money statistic**,
   the direct contrast to §4.11's `u`: lag-1 0.27 and ~0 by lag 50. An intention that
   exists should hold autocorrelation out to lag 50.
4. **pointmaze only: `mi_z_goal`** — mutual information (or a simple regression R²)
   between `z` and the held-out goal coordinates. §4.11 named the unobserved goal as the
   hidden variable; if `z` recovers it, the premise is confirmed as directly as it can be.

Cost: minutes to ~1 h per env on one GPU. Run on all three published envs — pointmaze is
the one with the diagnosed hidden variable and is the most informative.

**Decision rule:** `z_autocorr(50) > 0.5` **and** `predictive_gain > 10%` → build S3-b.
Otherwise stop and record the negative.

## S3-b — The graft. Two variants; they are not the same research decision.

**S3-b-i, minimal graft — keeps our zero-shot claim.** Add `policy_index="intention"` as a
third setting of `_index()`: `z` from the S3-a encoder replaces the prior draw `u'` in
`psi(s, ·, u)`. Everything else in psmflow is unchanged — readout stays `psiᵀw`, closed-form
`w = E[rφ]` is preserved. Combined with S2's expectile aggregation this becomes *GPI over
intentions* instead of GPI over random latents. Cost: the S3-a encoder (already trained) +
a normal Stage C, ~3 h/seed. **This is the one to build first.**

**S3-b-ii, faithful InFOM — a baseline, not a fix.** The intention-conditioned flow
occupancy model `p(s_f | s, a, z)` by flow matching over discounted future states;
`Q̂(s,a,z) = 1/((1−γ)N) Σᵢ r(s_f^i)`; expectile distillation into `Q(s,a)`; actor distilled
against it. Note what this abandons: `psiᵀw` and the closed-form `w`. InFOM adapts *with
reward labels*; our headline is zero-shot. So this is the strongest published comparison in
our exact setting and the paper needs it in related work — but it is **not** a variant of
our method, and it must be reported as a baseline. Cost: ~Stage-A-scale flow per env plus
per-task adaptation.

Incidental benefit worth noting for both: `Q̂` as a Monte-Carlo average of `r(s_f)` has no
linear-in-`φ` readout, so it sidesteps D2's ceiling (best linear read-out of `φ` explains
~13% of reward variance; antmaze identical at 0.135/0.134).

### Verify against the paper before building S3

Read the method section from source. It was consulted here through a fetched HTML
rendering, not the PDF (this box has no `pdftotext`/poppler), and two details change the
implementation: **the encoder's exact conditioning (`s',a'` vs `s,a`)**, and **the
expectile parameter μ together with the number of `z` samples per update**.

---

# Rules for whoever runs this

Standing discipline for this repo:

- `import utils.xla_guard` **before jax** in every new entry point. XLA:GPU miscompiles the
  unrolled flow ODE past ~30 steps and silently returns actions pinned at the clip.
- Every diagnostic persists JSON through `report_out`. Never stdout-only.
- Before any launch: print the full hyperparameter table, smoke-test the exact code path
  for ~200 steps, and after launch re-read the run's own `flags.json` to confirm the values
  landed. On kisski this is `sbatch`, not tmux — `scripts/slurm/train_psmflow.sbatch`.
- 500 episodes (`scripts/eval500.sh`) for anything reported; in-loop 50-episode evals swing
  ±0.15 between consecutive points. Mean and 95% CI across seeds, never a peak or a best
  seed. Always quote the BC control from the same flow checkpoint.
- State expected outcomes — including expected failures — **before** results arrive. Each
  spec above has its predictions pre-registered; do not revise them after seeing numbers.
- `docs/tables/results.md` is generated by `tools/make_tables.py`, never hand-edited.
  Append a dated entry to `docs/HANDOFF.md` for any work that produced numbers.
- `.venv/bin/python -m pytest tests/ -x -q` and `.venv/bin/ruff check .` before committing.
  Note the suite aborts when run as one process (pre-existing, environmental, present on
  `origin` too) — run per-file if it bites.
- Nothing here touches Stage A or Stage B, so no preimage is invalidated. If that changes
  (S1's "scrambled" branch points at OT-CFM), every `.npz` and its sidecar must be
  regenerated: `docs/PREIMAGES.md`.

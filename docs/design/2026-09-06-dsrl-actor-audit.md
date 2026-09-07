# DSRL-style latent actor on the affine critic: audit, fix, and pre-registered ablation

Date: 2026-09-06 (written), launched 2026-09-07.
Subject: `agents/psmflow.py` under `psi_form=affine policy_index=latent train_actor=true acting=actor`
(the `affine_actor` arm), against the actor-free `affine_strict` default.
Reference: DSRL, Wagenmaker et al. 2025 — `ajwagen/dsrl` + `ajwagen/stable-baselines3-dsrl`
(`10e5d311`) and `nakamotoo/dsrl_pi0`, all three cloned and read for this audit.

## 0. Why

The repo default is actor-free: `affine_strict` scores cube **0.415** pooled over 300k–500k
× 3 seeds (n=15, 95% CI ±0.083) against a BC control of **0.072**, and antmaze **0.081 ±
0.050** — i.e. nothing. The 09-05 GPI ablations showed *where* the cube value comes from:
pinning the policy index `u'` to one prior draw (`gpi_select=fixed_index`) collapses 0.704
→ **0.000 / 0.180**, and the 09-06 antmaze H1 test reproduced it (pooled 97/6000 = 0.016 vs
BC 0.072, z = −8.47). **The per-step max over 64 fresh policy indices is the mechanism.**

The existing latent-actor arm on the same affine critic is flat: cube **0.162 / 0.130**,
antmaze **0.224 / 0.206** at 500k. The standing hypothesis (09-05 HANDOFF open item 4) is
that a DSRL-NA-style actor could get the strict arm's ceiling with the actor arm's
smoothness. This audit establishes what the current actor actually optimises before
building that.

---

## 1. Audit findings

### (a) Actor inputs/outputs, and what its Q is indexed by

`NoiseConditionedActor` (`utils/psm_networks.py:172-195`) is
`(obs, z, noise) -> tanh(MLP)`, and `flow_actor_loss` scales it:

```
agents/psmflow.py:263    u_a = u_clip * self.actor(obs, w, noise, params=actor_params)
```

so the map is **(s, w_task, noise) → u ∈ [-u_clip, u_clip]^{d_a}**, `u_clip = 3.0`.
The policy index `u'` is **not** an input to the actor.

The actor's Q is read at `agents/psmflow.py:274`:

```
Qs = (self.psi(obs, self._index(sampled), u_a) * w).sum(-1)      # (P, B)
```

and `_index` (`agents/psmflow.py:150-158`) returns `sampled.u_index` under
`policy_index=latent`, which `sample_step_inputs` draws at
`agents/psmflow.py:187-190` as **one fresh clipped `N(0,I)` draw per batch element**.

> **The actor climbs `Q(s, u, u')` at a single random policy index per batch element.**
> The acting rule it is compared against climbs `max_{u'} Q(s, u, u')` over 64 draws,
> per step. Those are different objectives — and the 09-05 selection diagnostic measured
> the index axis as carrying **~30x** more Q variation than the action axis, so the gap
> between `E_{u'}` and `max_{u'}` is not a detail.

Measured directly (§(b) below, cube `affine_actor` @500k, batch 1024): the cosine between
the single-draw actor gradient and the panel-max actor gradient is **−0.49 (sd0)** and
**−0.20 (sd1)**. The shipped actor gradient is *anti-correlated* with the gradient of the
objective GPI maximises. That is the answer to "is it optimising a lottery ticket": yes,
and the ticket is worse than random with respect to the deployed rule.

Under `index_agg=expectile` the actor instead climbs `q_dist(s, [w, u])`
(`agents/psmflow.py:266-270`), which *is* an amortized upper expectile over the index
distribution — the right shape. That arm has never been run with an actor.

### (b) Loss-term scale on the affine critic — measured, not asserted

New tool: **`tools/diag_actor_grad_terms.py`** (the 09-03 audit's equivalent lived only in
`/tmp` probe scripts and was not re-runnable). One batch of 1024, CPU, restored checkpoint,
gradients taken w.r.t. actor params only (`bc_flow_loss` has exactly zero actor gradient —
it trains `actor_vf` alone).

Reports: `$PSM_DATA/logs/diag_actor_grad_affine_cube_sd{0,1}_500k.json`.
Checkpoints: `affine_actor_cube/sd00{0,1}_*` @500000, `psi_form=affine`,
`policy_index=latent`, `index_agg=max`, `bc_coeff=1.0`.

| quantity | sd0 | sd1 |
|---|---|---|
| `‖∇_actor q_loss‖` (single index draw, **shipped**) | 0.1964 | 0.1343 |
| `‖∇_actor q_loss‖` (max over an index panel of 16) | 0.3812 | 0.2478 |
| `‖∇_actor bc_coeff·distill‖` (`bc_coeff=1.0`) | 0.5573 | 0.4761 |
| `‖∇_actor_vf bc_flow‖` | 0.9110 | 0.9009 |
| **ratio BC : Q (shipped)** | **2.84 : 1** | **3.55 : 1** |
| ratio BC : Q (panel max) | 1.46 : 1 | 1.92 : 1 |
| `cos(∇q_single, ∇q_panel_max)` | **−0.492** | **−0.199** |
| `cos(∇q_single, ∇distill)` | +0.253 | −0.187 |
| loss values: q_single / q_panel / distill / bc_flow | 0.656 / −1.575 / 0.117 / 1.359 | 0.626 / −0.902 / 0.113 / 1.366 |
| GPI best value vs actor's latent (64×64 pairs, 64 states) | 8479.8 vs 7973.7, **gap 27.7% of \|Q\|** | 2528.2 vs 2329.8, **gap 16.9%** |

Reading: BC domination is **real but milder on the affine critic (2.8–3.6:1) than the
09-03 audit found on the free-psi critic (4.8–5.9:1)**. Switching the actor's Q to the
panel max roughly doubles the value gradient's norm and halves the domination ratio,
without touching `bc_coeff`. And the actor's latent is 17–28% of `|Q|` short of what the
deployed GPI rule finds on the same batch — the actor is not merely under-driven, it is
being driven the wrong way.

### (c) Bootstrap: what the TD target's next latent is

`sample_step_inputs` computes the actor's latent at `s'`
(`agents/psmflow.py:181-183`) but under `policy_index=latent` **overwrites it**:

```
agents/psmflow.py:199-204   if c["policy_index"] == "latent":
                                u_next = u_index
```

So with `train_actor=true` **and** `policy_index=latent` the TD target's next latent is
`u'`, the prior index draw, **not** the actor's output. That is *internally consistent*
with the semantics — `psi(s,u,u')` is the successor measure of "take `u`, then follow
`pi_{u'}`", so the backup `psī(s', u', u')` is the write-up's — and it is why
`backup_explore_frac` is documented as inert here.

The consequence, which is the load-bearing part: **the actor is completely decoupled from
the critic.** Nothing the actor does enters `measure_loss`, and `flow_actor_loss` reads
`psi` at stored params outside the `argnums`. Verified on the two runs, same seed, same
everything but `train_actor`:

| step | `psm_loss` strict / actor | `w_enc_spread` strict / actor |
|---|---|---|
| 5k | 608.81 / 611.96 | 1.1005 / 1.0995 |
| 10k | 958.89 / 960.97 | 1.2407 / 1.2403 |
| 305k | 4720.1 / 5852.5 | — |
| 500k | 33959 / 169705 | — |

Identical to GPU non-determinism early, chaotically divergent late (both series are
blowing up regardless). So **`affine_actor` is `affine_strict`'s critic with a different
acting rule bolted on**, and cube 0.415 vs 0.146 is a clean measurement of the acting rule
alone. Flagged inconsistency: the actor maximises `Q(s,u,u')` at one `u'` while acting
would need `max_{u'}`; the fix is to align the actor's objective with the acting rule, not
to change the backup.

### (d) DSRL fidelity — reference code vs this repo

Reference (verbatim, `sb3_dsrl/stable_baselines3/sac/sac.py:279-294` and
`dsrl/dsrl.py:308-363`):

- **DSRL-SAC** actor: `actor_loss = (ent_coef * log_prob - min_qf_pi).mean()`, tanh-squashed
  diagonal Gaussian with the stock SB3 log-prob correction
  (`log_prob -= sum(log(1 - a^2 + 1e-6))`), `log_std` clamped to `[-20, 2]`, **min** over
  the 2-critic ensemble, entropy coefficient **auto-tuned** with **target entropy 0.0**
  (not `-dim(A)`), noise bounded to a box `[-b, b]` with **b = 1.5** by default (gym +
  robomimic; 1.0/2.0/2.5 in pi0). Prior draws used for exploration are clipped to the
  same box. LayerNorm after every hidden Linear in **both** actor and critic. tau 0.005,
  γ 0.99, UTD 20, lr 3e-4, batch 256, `n_critics=2`.
- **DSRL-NA**: two critics — `critic` over *decoded actions*, `critic_noise` over *noise*.
  The actor maximises **`critic_noise`** only. `critic_noise` is distilled from prior draws
  (`dsrl.py:349-363`): one fresh unclipped `N(0,I)` per replay state, decoded through the
  frozen policy, and each ensemble member regressed member-to-member onto the action
  critic's value with unweighted `0.5·MSE`, in an inner loop of `noise_critic_grad_steps=10`
  after the UTD block.
- **Support control**: there is **no KL or BC penalty toward `N(0,I)` anywhere**. The only
  devices are (i) the hard box `b=1.5` ≈ 1.5σ — a deliberate truncation of the prior and
  the paper's main "stay on support" knob, (ii) clipping exploration draws to it,
  (iii) entropy maximisation at target 0.0, and (iv) for NA, that the noise critic is only
  ever *fit* at prior draws.

Against this repo's `train_actor=true` arm:

| axis | DSRL reference | psmflow today | verdict |
|---|---|---|---|
| policy | tanh-Gaussian, reparameterised, log-prob corrected | **deterministic given an input noise draw**; the 09-03 audit measured `u_a ≈ 0.83·noise` (per-dim corr 0.951–0.968, mean 0.961), so its "stochasticity" is a shrunk copy of its own input and its `w`-sensitivity is 2.9% of its noise-sensitivity | **gap** |
| entropy | auto α, target entropy 0.0 | none | **gap** |
| ensemble reduction for the actor | min over 2 | `mean − 0.5·unc` at `P=2` **is exact min** (09-03 audit §6 verified) | match |
| LayerNorm | actor + critic, every hidden layer | `_simple_embedding` and `_AffinePsiTower` both LayerNorm the first hidden layer only | partial |
| noise box | `[-1.5, 1.5]` ≈ 1.5σ | `u_clip = 3.0` ≈ 3σ; dataset preimages exceed ±3 on only 0.46% of entries, so 3.0 is the *typical set*, not a truncation | **gap (ours is looser)** |
| off-support control | box + entropy only | box + a `bc_coeff=1.0` distillation to the latent-CFM rollout, which DSRL does not have and which dominates the value gradient 2.8–3.6:1 | different |
| NA distillation | `Q_W` regressed onto `Q(s, G(s,ε))` at prior draws | absent | **gap** |
| UTD | 20 | 1 | gap (not addressed here) |

Where the actor can go: `tanh · u_clip` keeps it inside the box, and *nothing else*
constrains it. The clipped prior at `d_a = 5` has median `‖u‖ ≈ 2.13`; the box corner is
`3√5 ≈ 6.7`. `psi`'s index slot is only ever trained at clipped-prior draws, so a latent
in that outer shell is exactly where the critic is extrapolating. The 09-05 diagnostic
already measured the shipped GPI argmax selecting `‖u‖ = 2.8` against a candidate mean of
2.13 — and the critic-free `max_norm` control at `‖u‖ = 3.81` scores 0.078, i.e. BC.

### (e) Eval path

`tools/eval_checkpoint.py:122-198` (`merge_run_config`) takes the run's own `flags.json`
as defaults with typed CLI overrides on top; `acting=actor` decodes one actor draw
(`agents/psmflow.py:791-796`), one shot, no selection. Verified in the 09-05 JSONs:

```
eval500_affine500k_actor_cube_sd0.json:    success 0.162, acting "actor",
    train_actor true, policy_index "latent",
    acting_mode "decode(amortized actor latent) — action branch disabled"
```

and the same for `sd1` (0.130) and antmaze (`0.224` / `0.206`). **The affine_actor numbers
were evaluated as actor.** Gap found: the report records `acting`, `train_actor`,
`policy_index`, `gpi_*`, `u_clip` — but **not** `psi_form`, `index_agg`, or (now)
`actor_mode`. Fixed by adding those three keys to the report dict
(additive only).

---

## 2. What changed (Part 2)

Smallest switchable set; **the default path is byte-identical** (`actor_mode=ddpg`,
`actor.index_panel=0`, `train_actor=false`), guarded by static Python `if`s exactly as the
existing `index_agg` / `action_critic` branches are, and by tests.

### 2.1 `actor.index_panel` — align the actor's Q with the acting rule

New key `agent.actor.index_panel` (default **0** = the shipped single-draw behaviour).
At `K > 0` the actor's Q becomes

```
Q(s, u_a) = agg_{j=1..K} [mean_P − pess·unc] ( psi(s, u'_j, u_a)^T w ),   u'_j ~ clipped p0
```

with `agg = max` (`index_agg=max`) — literally the object `gpi_select` maximises. Under
`index_agg=expectile` the actor already reads the amortized aggregate `q_dist` and this
key is ignored (asserted). Applies to `actor_mode` ∈ {`ddpg`, `dsrl_sac`}.

### 2.2 `actor_mode: ddpg | dsrl_sac | dsrl_na`

| mode | actor network | loss |
|---|---|---|
| `ddpg` (default, unchanged) | `NoiseConditionedActor`, deterministic given noise | `−Q/\|Q\| + bc_coeff·distill + bc_flow` |
| `dsrl_sac` | new `TanhGaussianLatentActor(s, w) → (μ, logσ)`, reparameterised, tanh-squashed, scaled to `[-u_clip, u_clip]`, `log_std` clamped `[-20, 2]` | `(α·logπ − Q)/\|Q\| + bc_coeff·distill + bc_flow`, α auto-tuned to `target_entropy` (DSRL's 0.0) |
| `dsrl_na` | same tanh-Gaussian head | `na_coeff·‖u_a − stopgrad(u*)‖² + q_coeff·(α·logπ − Q)/\|Q\| + bc_coeff·distill + bc_flow`, where `u*` is the **GPI argmax over `na_candidates` prior action latents × `index_panel` prior indices** — the actor regresses onto the acting rule's own choice, so its target is always a prior draw |

`dsrl_na` is this substrate's analogue of the reference's noise-critic distillation: the
reference distils a *critic* over noise from prior draws; we already have one
(`psi(s,·,u')^T w`), so we distil the *policy* onto its argmax over prior draws. Both keep
every training signal defined at latents `p0` actually produces. Optional
`na_advantage_weight` weights each state's regression by `(Q(u*) − mean_k Q(u_k))` normalised
over the batch, so states where the choice does not matter contribute less.

Everything is additive to the pytree (`sac_actor`, `log_alpha` `TrainState`s, always
created). `restore_agent` (`utils/flax_utils.py:198-212`) already keeps fresh params for
fields a checkpoint predates, so every existing checkpoint still restores and evaluates.

### 2.3 Known cosmetic nit

The new `fold_in` constants are 112 (`sac_actor` init **and** `_actor_q`'s index panel) and
113 (`log_alpha` init **and** the NA target draw). Init happens in `create` off the local
`rng`, which is also the `rng` the agent is constructed with, so on the FIRST update only —
before `update`'s split has moved `self.rng` — the panel draw shares a key with a parameter
initialisation. Different consumers, different shapes, one step out of 500k: nil effect, and
it is the same class of thing as the pre-existing `r_next` reuse the 09-03 audit recorded as
cosmetic. Not fixed here because the 18 runs were already launched against these constants
and renumbering mid-queue would split the ablation across two code versions. Follow-up: move
the two runtime uses to 114 / 115.

### 2.4 Not changed

The critic and measure losses, `sample_step_inputs`' backup, `gpi_select`, and the
actor-free path. `bc_coeff` stays available and is reported.

---

## 3. Hyperparameters for the ablation

Common to all 18 runs (inherited from `configs/agent/psmflow.yaml`, verified in each
`flags.json` after launch):

```
agent=psmflow  psi_form=affine  policy_index=latent  index_agg=max
z_dim 128 · affine.w_dim 128 (norm_w true) · num_parallel 2 · batch_size 1024
sf 1024x1 emb2 · phi 256x2 · discount 0.98 · tau 0.01 · ortho_coef 1000.0
pessimism_penalty 0.5 · actor_pessimism_penalty 0.5 (= exact min at P=2)
mix_ratio 0.5 · norm_z true · backup_explore_frac 0.0
lr_phi 1e-5 · lr_sf 1e-4 · lr_actor 1e-4 · lr_actor_vf 3e-4
u_clip 3.0 · use_point_preimage true · gpi_decode onestep · gpi_num_u 64
offline_steps 500000 · save_interval 50000 · seeds 0,1,2
actor: hidden 512x2 emb2 · vf 512x4 · flow_steps 10
```

Arm-specific:

| key | `dsrl_na` | `dsrl_sac` |
|---|---|---|
| `actor_mode` | `dsrl_na` | `dsrl_sac` |
| `acting` / `train_actor` | `actor` / `true` | `actor` / `true` |
| `actor.index_panel` | 16 | 16 |
| `actor.na_candidates` | 16 | — |
| `actor.na_states` | 256 | — |
| `actor.na_coeff` | 1.0 | — |
| `actor.na_advantage_weight` | false | — |
| `actor.q_coeff` | 0.0 (pure distillation) | 1.0 |
| `actor.bc_coeff` | 0.0 | 0.0 |
| `actor.entropy` | auto | auto |
| `actor.target_entropy` | 0.0 | 0.0 |
| `actor.init_alpha` / `lr_alpha` | 1.0 / 3e-4 | 1.0 / 3e-4 |

`na_candidates = 16` and `na_states = 256` rather than `gpi_num_u = 64` over the whole
1024-row batch: the target costs `na_candidates` psi forward passes per state, and the
affine head's A output is a `1024 -> z_dim * w_dim` layer. 16 x 256 = 4096 (s, u) pairs is
2x what `measure_loss` already evaluates and lands the step at 0.032 s (below); 64 x 1024
would be 16x and would not finish in a day. The entropy term sits outside `q_coeff`, so at
`q_coeff = 0` the tanh-Gaussian is still entropy-regularised and cannot shrink onto the
conditional mean of a re-randomised regression target (failure mode 2 below).

**200-step smokes (SLURM 2491930 `dsrl_na`, 2491931 `dsrl_sac`, cube, seed 0).** Both
completed; `flags.json` re-read and matching the table above.

| | `dsrl_sac` | `dsrl_na` | actor-free reference |
|---|---|---|---|
| s/step (`epoch_time`) | 0.0249 | 0.0321 | 0.0194 |
| projected 500k wall | ~3.5 h | ~4.5 h | 2 h 42 m (cube sd2) |
| `actor_alpha` @50 -> @200 | 0.971 -> 0.942 | 0.957 -> 0.943 | — |
| `actor_entropy` @200 | 3.41 | 2.74 | — |
| `actor_q` @200 | 1.50 | 1.83 | — |
| `actor_u_norm` @200 | 3.86 | 2.93 | — |
| `actor_na_error` / `na_target_norm` @200 | — | 2.95 / 2.40 | — |

`na_target_norm` 2.40 against the clipped prior's median `‖u‖ = 2.13` at `d_a = 5` confirms
the regression target is a prior draw, and `actor_u_norm` 2.93 (NA) vs 3.86 (SAC) is the
first sign that the distillation is doing the support job `bc_coeff` used to.
Walls set from the smoke rate: cube 8 h, antmaze / pointmaze 11 h; in-loop
`eval_episodes = 10` (the in-loop eval is on record as uninformative and antmaze episodes
cost ~11 s each).

**First logged step of the real runs (cube `dsrl_na`, step 5000, all three seeds).** All
three agree to three digits, which is what a healthy shared init and a working objective
look like:

| | sd0 | sd1 | sd2 |
|---|---|---|---|
| `actor_na_error` | 1.871 | 1.868 | 1.863 |
| `actor_na_target_norm` | 2.579 | 2.541 | 2.548 |
| `actor_u_norm` | 2.034 | 1.979 | 2.007 |
| `actor_entropy` | 0.748 | 0.625 | 0.723 |
| `actor_alpha` | 0.293 | 0.293 | 0.292 |
| `actor_q` | 224.8 | 203.1 | 211.8 |
| s/step | 0.0398 | 0.0402 | 0.0400 |

`actor_u_norm` ~2.0 against the clipped prior's median 2.13: the actor is INSIDE the prior
shell, which is what the whole NA design is for and is the opposite of the off-support
climb the 09-05 ablations indicted. `actor_alpha` 0.94 -> 0.29 and `actor_entropy` 2.74 ->
0.7 is the auto-tuner doing its job against `target_entropy = 0.0`.

Watch item, pre-registered as failure mode 2: `actor_na_error` fell 2.95 -> 1.871 in 5k
steps, but the per-dimension MSE between two INDEPENDENT clipped-prior draws is ~2.0, so at
step 5000 the regression is only just below the "predict the conditional mean" floor. If it
plateaus at ~1.8 rather than continuing down, the actor is tracking the mean of a
re-randomised argmax and not the argmax, and the fix is `na_advantage_weight=true` or a
larger `na_candidates`. At 0.040 s/step the cube runs project to 5 h 34 m of loop time,
inside the 8 h wall.

`bc_coeff = 0.0` on both arms is deliberate and is the one place we follow DSRL over this
repo's history: the reference has no BC anchor at all, §1(b) measures ours dominating the
value gradient 2.8–3.6:1, and the 09-03 audit's standing instruction is *not* to tune
`bc_coeff` as a knob. `dsrl_na`'s regression target is itself a prior draw, which is the
support device that replaces it; `dsrl_sac` keeps the box and the entropy term, which is
exactly what the reference relies on. If `dsrl_sac` goes off-support this is the predicted
cause and `bc_coeff` is the first thing to restore.

Deviation from the reference kept on purpose: `u_clip = 3.0`, not DSRL's 1.5. Changing the
box changes the critic's training distribution too (`u_index`, `u_data` clipping,
`gpi_select`), i.e. it is a second experiment. Recorded as an open follow-up.

---

## 4. Pre-registered outcomes (written before any of the 18 runs finished)

Reporting rule, unchanged and binding: **late mean over 300k–500k checkpoints pooled
across 3 seeds, with the 95% CI; never a peak, never a best seed.** BC control quoted
beside every row.

Comparators: cube — actor-free `affine_strict` **0.415 ± 0.083** (n=15), `affine_actor`
(ddpg) **0.146** (2 seeds @500k), BC **0.072**. antmaze — `affine_strict` **0.081 ±
0.050**, `affine_actor` **0.215**, BC **0.072**. pointmaze — actor-free is in flight, no
comparator yet.

**H-A (the main hypothesis).** The single-index draw is why the actor arm is flat. Aligning
its Q with the panel max, plus a distillation target that is itself a GPI argmax, should
move cube materially above the ddpg actor's 0.146 and toward the strict arm's 0.415, with a
*smaller* spread than the strict arm's ±0.083 (the actor arm's whole appeal is smoothness:
its cube checkpoint span is 0.04 against the strict arm's 0.62).
Concretely: **`dsrl_na` cube late mean ≥ 0.25** and its across-checkpoint std **< 0.12**.

**H-B.** `dsrl_sac` — a pure critic-gradient climb with entropy and no BC anchor — will
*not* reproduce it, because the 09-05 ablations already showed a critic-gradient actor
climbing off-support does not recover the GPI value, and §1(d) shows nothing but the box
constrains it. Expected **cube 0.05–0.20**, i.e. at or near BC, possibly *below* the ddpg
actor. If it lands above 0.25 the entropy/stochastic head, not the index aggregation, is
the operative ingredient and H-A is mis-attributed.

**H-C (antmaze).** The actor arm is the only thing that has ever cleared BC on antmaze
(0.215 vs BC 0.072, strict 0.081). Expect both new arms in **0.15–0.30**, i.e. to hold that
and not much more. Antmaze's failure was diagnosed (09-06 H1) as psi having *no
index-independent notion of a good action*; a better actor objective does not fix that.
An antmaze number above 0.35 would be the first real antmaze result in this repo and would
have to be checked against the γ-sweep runs (`affine_strict_antmaze_g99{,5}`) before being
believed.

**H-C' (antmaze at `discount=0.99`), added 2026-09-07 after the g99 result landed.**
Another agent measured affine strict antmaze at `agent.discount=0.99` scoring **0.534 @100k**
(seed 2, 500 ep) against **0.081** at the default 0.98, and 0.995 reaching 0.678 @100k before
collapsing to 0 from 300k. **Antmaze was a horizon problem, not an acting-rule problem**, and
my antmaze arms at 0.98 are therefore measured against a broken actor-free baseline. Six more
runs (`affine_dsrl_{na,sac}_antmaze_g99`, 3 seeds each) repeat the ablation at 0.99; the 0.98
runs continue as the controlled comparison at the repo default.

**Arms.** Plain `dsrl_na` is NOT repeated at g99: §4a already measured it at ~7 % of the
distance explained on three cube seeds, and a sixth confirmation of a known negative is not
worth six GPU-days. The g99 slots test the REMEDY instead:

| group | arm |
|---|---|
| `affine_dsrl_sac_antmaze_g99` | `dsrl_sac`, 3 seeds, as originally planned |
| `affine_dsrl_naadv_antmaze_g99` | `dsrl_na` **with `actor.na_advantage_weight=true`**, 3 seeds |

`na_candidates` stays 16. The decision, recorded because the literal trigger fired and I
did not act on it: the 2000-step naadv smoke (SLURM 2491983) read **-5.4 % to +1.5 %**
distance explained, i.e. at or below the independence floor, which was the pre-agreed
condition for raising `na_candidates` to 64. It is not a floor, it is the initialisation
transient, and the same environment's own plain-NA run proves it:

| | naadv smoke @2000 | plain NA (2491935) @5000 | @10000 | @20000+ |
|---|---|---|---|---|
| `u_norm` | 3.73 (still falling from 4.20) | 2.839 | 2.255 | ~2.20 |
| `actor_entropy` | 4.27 (target 0.0, `alpha` 0.559 still falling) | — | — | ~0 |
| distance explained | **-5.4 %** | **2.8 %** | 6.8 % | 7.0-8.6 % |

At 2000 steps the actor's latent norm is ~65 % above its settled operating point and the
entropy controller has not converged, so `na_mse` and the floor (which is computed FROM
`u_norm`) are both still moving and their ratio is meaningless. Plain NA on this exact
environment read 2.8 % at 5000 steps and still went on to plateau at 7-8.6 %; a 2000-step
number cannot separate "at the floor" from "not yet started". Raising `na_candidates` on
that basis would have changed a second variable for no evidence. **The gate is read from
`actor_na_mse` at 50k, as registered.**

**Watch item visible already.** `actor_na_error / actor_na_mse` sits at **0.97-0.99**
through the whole smoke: the advantage weights are very nearly uniform, so the remedy is
currently close to a no-op. `actor_na_advantage` is still growing (1.42 -> 3.26 over 2000
steps), so it may differentiate as the critic sharpens — but if that ratio is still ~1.0 at
50k, advantage weighting cannot be what rescues the regression, and the ~25 % gate will fail
for a reason that is about the CRITIC's inability to separate action latents, not about the
weighting scheme. That ratio is the cheapest early read on the whole line.

**Measurement fix this required.** `actor_na_error` is the term the actor optimises, and under
advantage weighting that is a REWEIGHTED mean — not comparable to the unweighted independence
floor `(‖u_a‖² + ‖u*‖²)/d_a` that §4a's ~7 % was computed against. Comparing the two would
confound "the remedy worked" with "the weighting changed the loss scale". `actor_na_mse`, the
plain unweighted mean, is now logged alongside it and is the number every distance-explained
figure below refers to.

Pre-registration for the g99 arms, against the actor-free g99 ladder another agent is
producing (only 0.534 @100k on one seed is known so far, so the comparator is provisional):
- The interesting question is now **whether an amortized actor can hold a value the GPI rule
  demonstrably reaches on this env**, which is the first time that question has been askable
  on antmaze at all.
- **`naadv` gate, decided in advance: distance explained (from `actor_na_mse`) must exceed
  ~25 % by 50k.** Below that, advantage weighting has not rescued the regression, and taken
  with §4a the conclusion is that the GPI argmax over ACTION latents carries no amortizable
  state-dependent content — which closes the DSRL-actor line rather than motivating a third
  variant. Above 25 %, the 7 % plateau was a target-noise problem and `na_candidates` becomes
  the next knob.
- `naadv` success: **0.25-0.50**, recovering a substantial fraction of the actor-free g99
  number without exceeding it. Above 0.50 would be a genuine surprise and the strongest
  result in this ablation.
- `dsrl_sac` at g99: **0.10-0.35**. H-B's reasoning (a critic-gradient climb with no BC
  anchor and only a box constraining it) is unchanged by the discount.
- Both **above** their own 0.98 counterparts (0.15-0.30 registered under H-C). If the g99
  arms do NOT beat the 0.98 arms, the discount fix does not transfer through an actor, which
  would itself be informative — it would mean the longer horizon helps the per-step index max
  specifically, not the value function generally.
- Note the asymmetry this introduces: `naadv` at g99 has no g98 twin, so its comparison to
  the 0.98 `dsrl_na` runs confounds the discount with the weighting. Read it against the
  actor-free g99 ladder, not against my own 0.98 antmaze arms.
- Watch for the 0.995 pathology at 0.99: the actor-free g995 arm collapses to 0 from 300k, so
  a late collapse in the g99 actor arms is plausible and is why the late-mean window
  (300k-500k) may need reporting alongside a 100k-200k window for this env. State both.

**H-D (pointmaze).** No prior. Pointmaze is the easiest of the three and BC is comparatively
strong there; expect both arms within ±0.10 of the actor-free arm once it lands. Pointmaze's
value is as the third env for the ablation's row structure, not as a discriminator.

**Expected failure modes, stated in advance.**
1. `dsrl_sac` with `bc_coeff=0` and auto-α diverges: `actor_q` runs away (the 09-06 raw-action
   PSM baseline recorded exactly this — Q climbing to ~2000 at a `psi` norm the contrastive
   loss never sees) while success sits at 0.000. Detect on `actor_q` and `actor_entropy` in
   `train.csv` by 100k; the arm is then a measured negative, not a bug.
2. `dsrl_na` collapses onto the *mean* of the GPI argmaxes rather than tracking them —
   a regression target that is re-randomised every batch has a conditional mean, and the
   argmax over 64 draws is high-variance. Detect on `actor_na_error` (the regression
   residual) plateauing near the *prior variance* rather than falling, and on the actor's
   deployed latent norm collapsing toward 0. Mitigation if seen: `na_advantage_weight=true`,
   or a larger `na_candidates`.
3. Both arms simply reproduce ~0.15 on cube, i.e. the acting-rule alignment is not the
   binding constraint and the flatness is the critic's, consistent with 09-05's "stop
   looking for the cube fix at eval time". This is the outcome most consistent with the
   evidence already on record, and it would close the DSRL-actor line.

**What would make this a positive result worth writing up.** `dsrl_na` cube late mean
within the strict arm's interval (≥ 0.33) **with** a materially smaller spread, on three
seeds — i.e. the same ceiling, reproducibly. Anything that only matches 0.146 is a
confirmation of the negative.

---

## 4a. Mid-flight result: `dsrl_na`'s regression has PLATEAUED (failure mode 2)

Checked at 135-140k on all three cube seeds, as pre-registered. `actor_na_error` fell
2.95 -> ~1.70 by 10k and has been **flat ever since**:

| step | 5k | 10k | 25k | 50k | 75k | 100k | 125k |
|---|---|---|---|---|---|---|---|
| sd0 | 1.871 | 1.703 | 1.697 | 1.764 | 1.766 | 1.714 | 1.670 |
| sd1 | 1.868 | 1.515 | 1.768 | 1.708 | 1.722 | 1.655 | 1.716 |
| sd2 | 1.863 | 1.652 | 1.810 | 1.780 | 1.764 | 1.878 | 1.695 |

Quantified against the right null. If `u_a` were statistically INDEPENDENT of `u*` (both
near zero-mean), `E‖u_a - u*‖²/d_a = (‖u_a‖² + ‖u*‖²)/d_a`. From the logged norms over the
>=25k window:

| seed | `actor_na_error` | `actor_u_norm` | `actor_na_target_norm` | independence floor | **distance explained** |
|---|---|---|---|---|---|
| sd0 | 1.771 | 1.751 | 2.533 | 1.897 | **6.6 %** |
| sd1 | 1.753 | 1.755 | 2.522 | 1.888 | **7.1 %** |
| sd2 | 1.754 | 1.745 | 2.531 | 1.890 | **7.2 %** |

**The actor is capturing ~7 % of its own regression target and nothing more, on all three
seeds, for 110k steps.** The second signature is `actor_u_norm` 1.75 against
`actor_na_target_norm` 2.53: the actor has shrunk well INSIDE the shell its targets live on,
which is what an L2 regression onto a re-randomised argmax does when the argmax carries
little state-dependent signal — it converges to `E[u*|s] ≈ 0` and is held off the origin only
by the entropy term (`actor_entropy` reached its 0.0 target by 10k, `actor_alpha` settled at
0.219). That is failure mode 2 as written, not a new phenomenon.

Reading it against the audit: §1(a) established that psi separates policy INDICES ~30x more
strongly than action latents, so `max_{u'}` over the panel is informative while the inner
`argmax_u` it selects is nearly arbitrary. If the argmax over 16 action latents is that
noisy, its conditional mean given `s` is close to zero and there is little for an amortized
head to learn — which is what these numbers say.

**Remedy, not yet applied** (the runs continue; a relaunch would destroy the controlled
comparison and the 0.98 arm is still the registered control):
1. `actor.na_advantage_weight=true` — down-weight states where the panel does not separate
   the candidates, so the regression is dominated by the states where the argmax means
   something. Cheapest, no cost change.
2. `actor.na_candidates` up from 16 — a max over more draws is a less noisy target, at
   linear cost in psi forward passes.
3. If neither moves `actor_na_error` off ~1.75, the conclusion is not about the actor: it is
   that the GPI argmax over action latents has no amortizable state-dependent content, which
   would be a direct corroboration of the 09-05 finding and would close the DSRL-actor line.

**Reproduced on two more environments before those runs were stopped.** The antmaze and
pointmaze `dsrl_na` runs at the default discount were cancelled (below) at 30-40k, which is
already past the 25k point where the cube seeds had flattened. Their metric was snapshotted
first, and it is the same number:

| env | d_a | seeds | `actor_na_error` | `u_norm` | `target_norm` | floor | **explained** |
|---|---|---|---|---|---|---|---|
| cube (>=25k, running) | 5 | 3 | 1.75-1.77 | 1.75 | 2.53 | 1.89 | **6.6-7.2 %** |
| antmaze (>=20k) | 8 | 3 | 1.62-1.67 | 2.22 | 3.07 | 1.80 | **7.5-10.4 %** |
| pointmaze (>=20k) | 2 | 3 | 1.99-2.28 | 1.24 | 1.88 | 2.53 | **13.9-18.3 %** |

Nine seeds, three environments, three action dimensionalities: plain NA explains **under
20 % of its own regression target everywhere**, and under 11 % on the two harder ones. The
pre-registered gate (25 % by 50k) is missed on every one.

**Six runs cancelled 2026-09-07, on the coordinator's instruction** (SLURM 2491935-2491937
`affine_dsrl_na_antmaze`, 2491938-2491940 `affine_dsrl_na_pointmaze`, all at
`discount=0.98`). Reasons, recorded so the gap in the results table is not mistaken for a
lost run: plain NA is measured failing on cube; antmaze at 0.98 is now known to be a broken
baseline (the actor-free arm there is a horizon artefact, §H-C'); pointmaze is a settled
null. No checkpoint had been written (`save_interval=50000`, all were below 40k), so nothing
evaluable was destroyed — the table above is the entirety of what those 6 GPU-hours bought,
and it is kept. The **cube `dsrl_na` runs continue** as the controlled measurement of plain
NA, and **all `dsrl_sac` runs continue on all three envs**. The freed slots go to the g99
arms.

The 500-episode evals still run as planned — a flat regression does not entail a flat policy,
and `dsrl_na` at 7 % explained may still beat the ddpg actor's 0.146. But H-A (cube >= 0.25
with std < 0.12) should now be read as unlikely.

## 4c. Interim, cube: H-B and failure mode 1 CONFIRMED; H-A half-confirmed, half-refuted

`dna_cube_sd1` finished all 500k (SLURM 2491933, 4 h 32 m, 30.7 it/s, 10 checkpoints); the
other five cube runs are past 400k. Nothing below is a result -- the numbers are 10-episode
in-loop evals, which swing far wider than the 50-episode ones CLAUDE.md already calls
useless, and they are quoted ONLY as direction. The 500-episode evals decide.

**The regression plateau holds to 500k.** Plain NA, cube sd1, distance explained by
checkpoint: 11.7 / 12.9 / -3.2 / 2.7 / 5.9 / 5.4 / 4.8 / 11.2 / 9.3 / 5.2 % at 50k..500k;
**300k-500k mean 5.8 %**. It never approaches the 25 % gate at any point of a full run.

**Failure mode 1 is firing on `dsrl_sac`, with the mechanism visible.** The pre-registered
prediction was "`actor_q` runs away while success sits at 0.000, detectable on `actor_q` and
`actor_entropy` by 100k". All three seeds:

| | `dsrl_na` | `dsrl_sac` | clipped prior median |
|---|---|---|---|
| `actor_u_norm` (steady) | **1.72** | **4.21-4.41** | **2.13** |
| `actor_q` @400k | ~790 | **1481 / 853 / 662, still climbing** | — |
| `actor_alpha` @100k+ | 0.219 | **0.005** | — |
| in-loop 10-ep, >=100k | ~0.3 | ~0.1, sd2 pinned at 0.00 | — |

The SAC actor sits at **twice the prior radius**, out in a tail where psi was never trained,
and its Q climbs monotonically with no plateau -- the same signature the 09-06 raw-action PSM
baseline recorded (Q -> ~2000 at a psi norm the contrastive loss never sees). The entropy
controller drove `alpha` to 0.005 to reach `target_entropy=0.0`, so entropy stopped
constraining anything and the box (corner 6.7) is the only remaining limit; §1(d) predicted
exactly this, that the box alone is a weaker support device at `u_clip=3.0` than DSRL's
`b=1.5`. **This is the audit's central claim demonstrated: removing the BC anchor without
replacing it lets the actor leave the data.**

**The complement is the interesting half.** `dsrl_na` holds `u_norm` 1.72 -- INSIDE the
prior shell -- so the prior-draw regression does do the support job `bc_coeff` used to,
which was the design's whole premise. And on in-loop evals it tracks ~0.3 against the ddpg
actor's 0.146 and `dsrl_sac`'s ~0.1.

So H-A splits:
- its **level** half looks plausible (>=0.25), and would mean an amortized actor gets most of
  the actor-free arm's cube value;
- its **stability** half is already refuted. sd1 goes 0.70 -> 0.50 -> 0.10 -> 0.20 over
  consecutive 50k checkpoints. The actor arm inherits the actor-free arm's training-time
  non-stationarity rather than curing it, so the "same ceiling, reproducibly" case for
  DSRL-NA is not going to be made.

And it sharpens the real puzzle, which the 500-episode evals should be aimed at: **an actor
whose regression explains ~6 % of the GPI argmax performs near an arm whose entire value was
shown to come from the per-step max over 64 indices.** Either the 6 % is worth more than it
looks, or -- more likely, given §1(a)'s 30x index/action asymmetry -- what the NA actor
inherits is not the argmax at all but the *support*: a policy held just inside the prior
shell, which is a different and cheaper thing than GPI. If that is right, a plain "sample p0
and shrink slightly" control should match `dsrl_na`, and that control costs one eval.

## 4g. RESULT, cube: DSRL-NA gets 74 % of the actor-free ceiling, and the support
explanation is refuted

All 15 `dsrl_na` cells plus the 4 control cells, 500 episodes each, `EVAL_WORKERS=1`,
`acting_mode` verified as `decode(amortized actor latent [dsrl_na])` on every row.

| epoch | sd0 | sd1 | sd2 |
|---|---|---|---|
| 300k | 0.440 | 0.498 | 0.318 |
| 350k | 0.282 | 0.200 | **0.536** |
| 400k | 0.100 | 0.222 | **0.494** |
| 450k | 0.088 | 0.080 | **0.464** |
| 500k | 0.116 | 0.186 | **0.584** |
| per-seed mean | 0.205 | 0.237 | **0.479** |

**Pooled 300k-500k, n = 15: 0.307, sd 0.180, 95 % CI +/- 0.091, range 0.080-0.584.**

| arm | cube, 500 ep |
|---|---|
| actor-free (GPI, per-step max over 64 indices) | 0.415 +/- 0.083 |
| **`dsrl_na` (this work)** | **0.307 +/- 0.091** |
| ddpg latent actor (the pre-existing arm) | 0.146 |
| **`prior_shrunk` control** | **0.059 [0.048, 0.072]** |
| BC | 0.072 |

**The control validated itself.** Its checkpoint-invariance cell returned 28/500 at 300k
against 28/500 at 500k -- byte-identical, confirming it never reads psi or task_z, which is
what makes it a critic-free control rather than another arm. Pooled over three EVAL seeds:
89/1500 = 0.059, Wilson [0.048, 0.072], i.e. AT BC and if anything a shade below it. A 20 %
shrink of the latent buys nothing, which independently re-confirms the 09-05 finding that
`small_ball` did not help.

**So the support explanation is refuted, and the pre-registered prediction (§4d: "0.05-0.12,
at BC ... I expect this control to REFUTE the support explanation") holds.** `dsrl_na`'s
0.307 is genuine state-dependent behaviour, not "sample p0 and shrink". The line does not
close.

**Against the pre-registration (§4, H-A).** Level half **met**: 0.307 >= 0.25, more than
double the ddpg actor's 0.146 and 74 % of the actor-free arm. Stability half **refuted**:
sd 0.180 against the registered "< 0.12", with per-seed means spanning 0.205-0.479. The
entire argument for preferring an amortized actor over GPI was that it would trade a little
ceiling for reproducibility; it does not. It trades ceiling for nothing.

**Two cautions on the record, both from errors made while these numbers were landing.**
(i) After the first two cells (0.440, 0.282) I called the result "striking"; the next three
were 0.09-0.12. (ii) After two seeds I wrote that "every seed peaks at 300k and collapses";
seed 2 does the opposite, rising monotonically to its best value at 500k. Both are exactly
what the pooled-late-checkpoint rule exists to prevent, and both were generalisations from
n = 2. The only quotable form for this arm is the pooled row above.

**The open puzzle, sharpened.** §4a measured the NA regression as explaining ~6 % of the GPI
argmax on cube (7-18 % across nine seeds and three envs), and §4g measures the resulting
policy at 74 % of the arm whose value is entirely the per-step index max. The actor is
therefore NOT succeeding by distilling the argmax, and it is not succeeding by sitting in
the prior shell either -- `prior_shrunk` rules that out. What it is tracking is unidentified,
and it is the most interesting object this ablation has produced. A cheap next probe: the
per-state Spearman between the NA actor's chosen latent and the GPI score over a candidate
roster (`tools/diag_latent_ranking_oracle.py` already has the machinery), which separates
"weakly tracks the argmax" from "tracks something else entirely".

## 4h. Pre-registered: does the NA actor rank inside the GPI roster?

Probe `tools/diag_actor_vs_gpi.py`, run on the cube `dsrl_na` checkpoints (3 seeds x
{300k, 500k}) and on `affine_actor_cube` (the ddpg actor) as the reference. 256 dataset
states, ONE shared roster of 64 candidate action latents scored against 64 policy indices by
the same object `gpi_select` maximises, with the EVAL task vector from `infer_eval_z`.
Reports (a) the actor's score as a percentile of the roster, (b) Spearman across states
between the actor's score and the roster max, (c) latent-space distance from the actor's u
to the roster argmax against its distance to a typical roster member.

**Registered readings.**
- **Top ~25 % on most states** -> the arm IS a partial distillation and the MSE metric
  under-reads it. MSE is dominated by wrong-MODE errors (the argmax is one of several good
  latents; missing which one costs MSE dearly and costs ranking nothing), so 6 % explained
  and a good rank are consistent. That would make §4a a measurement artefact rather than a
  finding, and `na_advantage_weight` / `na_candidates` become worth tuning.
- **At chance (~0.5)** -> the value comes from somewhere the critic does not score, and the
  DSRL-NA framing is wrong about its own mechanism.

**My expectation, stated separately because it is a third possibility neither reading
covers: at or slightly above chance, ~0.5-0.65, AND that this will not settle the question.**
The 09-05 selection diagnostic already measured this critic's per-u ranking as barely
reproducible -- Spearman 0.21-0.36 between adjacent checkpoints, and the GPI argmax scoring
WORSE than random on the MC ground truth (regret 16.8-32.4 against 19.1 for random). A low
percentile therefore does not establish that the actor is failing to track something useful;
it can equally mean the GPI score is a bad ruler, which is already on record. The
informative cell is the **contrast with the ddpg actor**: both are scored by the same ruler,
so if `dsrl_na` ranks materially higher than the arm that scores 0.146, the ranking is
tracking something real; if they rank the same, the 0.307-vs-0.146 gap lives outside
anything this critic scores, and the next probe has to be an oracle (the MC ground truth of
`tools/diag_latent_ranking_oracle.py`), not the critic.

## 4e. `dsrl_sac` is a complete negative on all three envs, and the cause is a mis-scaled
entropy target

All twelve 0.98 runs finished. `dsrl_sac`'s terminal state, with the in-loop 10-episode eval
reading **0.0 at every checkpoint from 250k to 500k on all three seeds of both maze envs**:

| env | d_a | clipped prior median \|u\| | `actor_u_norm` (3 seeds) | ratio | `actor_q` | in-loop >=250k |
|---|---|---|---|---|---|---|
| cube | 5 | 2.13 | 4.21 / 4.33 / 4.36 | **1.97-2.05x** | 662-1481, climbing | ~0.1 |
| antmaze | 8 | 2.70 | 5.51 / 5.47 / 5.52 | **2.03-2.04x** | 25-46 | **0.0 everywhere** |
| pointmaze | 2 | 1.35 | 2.55 / 2.94 / 2.65 | **1.89-2.18x** | 3.5e3 / **2.9e7** / 4.1e4 | **0.0 everywhere** |

Pointmaze sd1's `actor_q` of **2.86e7** is the value runaway in its purest form -- seven
orders of magnitude above where the contrastive loss ever evaluates psi.

**The mechanism is now specific, and it is a ported constant, not a property of SAC.** The
ratio is ~2.0x the prior radius on all three environments, at d_a = 2, 5 and 8. That
invariance is the tell. DSRL's `target_ent: 0.0` is stated for its own noise box of
**b = 1.5** (~1.5 sigma); §1(d) recorded that ours is `u_clip = 3.0` (~3 sigma) and flagged it
as "looser", but did not follow the consequence through: **holding entropy at a fixed target
inside a box twice as wide puts the policy at twice the radius.** The auto-tuner then drives
`alpha` to ~0.005 to hit that target, so entropy stops constraining anything and the box --
the only remaining device, at 2x the width DSRL uses -- does not bind until 3 sigma. The
actor sits at 2x the prior radius, where psi is extrapolating, and Q diverges.

**Mechanism, refined 2026-09-07 after the corrected arm's smoke (an honest correction to
the sentence above).** "A fixed entropy target in a wider box puts the policy at twice the
radius" is right about the scaling but loose about the path. Entropy constrains the
PRE-SQUASH scale `log_std`, not the norm directly; the norm follows because
`u = u_clip * tanh(mu + sigma * eps)`, so a large sigma SATURATES the tanh and drives each
coordinate toward +-u_clip, i.e. toward the box corner at `u_clip * sqrt(d_a)` = 6.7 on cube.
The radius is therefore set jointly by saturation (sigma, which the entropy target controls
and which scales with u_clip) and by `mu` (which the Q gradient controls). The ~2.0x
invariance across d_a = 2, 5, 8 is empirical and stands; which of the two terms dominates is
exactly what the corrected arm measures. If lowering the target brings `actor_u_norm` down
to ~2.1, saturation was the cause and the diagnosis holds; if the norm stays at ~4.2 with a
much smaller sigma, then `mu` is being pushed out by the Q gradient and the entropy port is
NOT the cause -- in which case §4e's claim must be weakened to "the off-support climb is a
critic-gradient problem the entropy term never constrained in the first place".

The 200-step smoke (2492086) cannot separate these: at step 200 entropy is still 3.38 (from
init, target -3.4657) and `actor_u_norm` 3.86, indistinguishable from the broken arm's 3.89
at the same step. The smoke's job was the flags check (`target_entropy=-3.4657`,
`u_clip=3.0`, everything else identical) and the step rate (0.025 s/step -> ~3.5 h, inside
the 8 h wall). Both pass.

So H-B is confirmed, but the useful statement is sharper than "a critic-gradient actor goes
off-support": **this arm was run with a hyperparameter ported across a change of scale it
does not survive.** The honest reading is that `dsrl_sac` as configured tests the port, not
DSRL. Two follow-ups, neither run: `u_clip=1.5` for the SAC arm only (matching the
reference's box, at the cost of also changing the critic's training distribution -- a second
experiment, per §3), or a `target_entropy` scaled to the box, e.g. `-d_a * log(u_clip/1.5)`.

**Eval-budget consequence, and a change to the plan.** Confirming 0.000 at 5 checkpoints x
3 seeds x 2 envs is 30 evaluations of ~95 minutes to measure a floor. The informative spend
is **one checkpoint (500k) x 3 seeds x 2 envs = 6 evals**: pooled 0/1500 gives a Wilson 95%
upper bound near 0.002, which settles the arm. If any cell comes back non-zero, expand to
the ladder. That is 6 evals instead of 30, and the cube ladder (already queued) still
provides the full checkpoint-resolved picture for the one env where the arm is not at zero.

## 4f. Pre-registered: the CORRECTED SAC arm (`affine_dsrl_sac_te_cube`)

§4e diagnosed the SAC negative as a ported constant failing across a change of scale, which
makes the negative as it stands a statement about my port rather than about DSRL. One arm
fixes that: cube, 3 seeds, 500k, everything identical to `affine_dsrl_sac_cube` except

    agent.actor.target_entropy = -d_a * log(u_clip / 1.5) = -5 * log(2) = -3.4657

(for cube's d_a = 5). That is exactly the log-Jacobian of a uniform 2x contraction in each
of d_a dimensions: it asks the policy to be as concentrated IN ABSOLUTE TERMS as DSRL's
would be inside its own b = 1.5 box, which is the comparison the port was supposed to make.

**Why `target_entropy` and not `u_clip = 1.5`.** `u_clip` is read at fourteen sites in
`agents/psmflow.py`, and five of them shape what the CRITIC is trained on, not what the
actor emits:

  - `sample_step_inputs` clips the dataset preimages `u_data` (psmflow.py:390) -- the online
    side of the measure loss;
  - it clips the policy-index draws `u_index` (:407) and the backup latent `u_next` (:411,
    :418) -- the TD target;
  - `q_dist_loss`'s index panel (:721) and `gpi_select`'s candidate roster (:550-551).

Lowering `u_clip` would therefore retrain psi on a different latent distribution, and the
corrected arm would no longer share a critic with `affine_dsrl_sac_cube`, `affine_dsrl_na_cube`
or the actor-free arm -- the comparison would confound the actor fix with a different critic,
which is the same "two things at once" error §3 already refuses for the `u_clip` follow-up.
`target_entropy` is read in exactly one place, `alpha_loss`, and touches nothing else.
Keeping `u_clip = 3.0` is what makes this arm a controlled correction.

**Pre-registration (written before the smoke).**
- `actor_u_norm` settles near the clipped prior median, **2.0-2.6** (it is 4.21-4.36 at
  `target_entropy = 0.0`, and the prior median at d_a = 5 is 2.13). This is the mechanism
  test and it is the one I am most confident of; if the norm does NOT come down, the
  diagnosis in §4e is wrong and the entropy target is not what sets the radius.
- `actor_alpha` stabilises **>= 0.05** rather than collapsing to ~0.005 -- i.e. the entropy
  term keeps binding instead of being tuned away.
- `actor_q` stays **bounded**, in the few-hundred band the NA arm occupies, and does not
  climb monotonically to 500k. No 1e4+ excursions of the pointmaze kind.
- Success, 300k-500k pooled over 3 seeds: **0.10-0.25**. Better than broken SAC (in-loop
  ~0.1) because the actor is now on-support, but NOT reaching `dsrl_na`. The reason is that
  §1(a) measured the action-latent axis as carrying only 3-8 % relative Q spread, so a
  critic gradient has little to climb even from inside the prior shell, and the 09-05
  ablations already found that a critic-gradient actor does not reproduce the actor-free
  value.
- **Falsifier, stated plainly: if the corrected arm reaches >= 0.25, the entropy mis-port
  was the whole story, and the §4e negative must be withdrawn as a claim about DSRL-SAC and
  rewritten as a claim about this repo's port.** That is the outcome that would matter most
  and is the reason the arm is worth three GPUs.

## 4i. The `naadv` gate FAILS, and the remedy was genuinely applied

Read at 50k as registered (§H-C'), on all three `affine_dsrl_naadv_antmaze_g99` seeds,
from `actor_na_mse` against the independence floor `(‖u_a‖² + ‖u*‖²)/d_a` at d_a = 8:

| step | sd0 | sd1 | sd2 |
|---|---|---|---|
| **50k (the gate)** | **9.8 %** | **6.6 %** | **5.9 %** |
| 100k | 8.8 % | 9.0 % | 6.2 % |
| 200k | 6.6 % | 8.8 % | 2.2 % |
| 300k | 1.9 % | 7.8 % | -3.0 % |
| ~400k | -2.1 % | 1.6 % | -2.7 % |

**Gate missed by a factor of three, and the metric DEGRADES with training** to at or below
the independence floor. This is the same 6-10 % band as plain NA (§4a), so advantage
weighting bought nothing.

**And the remedy was really applied**, which is what makes this a clean negative rather than
an inconclusive one. The watch item registered at launch was that `actor_na_error /
actor_na_mse` stayed at 0.97-0.99 in the smoke, meaning near-uniform weights and an inert
remedy. In the real runs that ratio is **0.60-0.94** (mean ~0.83): the advantage does
separate states, the weighting is doing real work, and the regression still fails. Had the
ratio remained ~1.0 the result would have been uninterpretable; it did not.

**Registered consequence, applied.** §H-C' stated: below 25 %, "taken with §4a the
conclusion is that the GPI argmax over ACTION latents carries no amortizable
state-dependent content -- which closes the DSRL-actor line rather than motivating a third
variant." That is the finding. **No third distillation variant should be built.**

**The distinction that must survive this.** "The distillation fails" is NOT "the arm fails".
§4g measured `dsrl_na` at 0.307 on cube -- 74 % of the actor-free ceiling -- with the same
6-7 % regression. The naadv arms therefore still need their 500-episode evals: they answer
whether the ARM works on antmaze at gamma=0.99, which is independent of whether its stated
mechanism works. What is closed is the mechanism, not the measurement.

## 4j. `dsrl_sac` at gamma=0.99: the discount does not rescue the port

`affine_dsrl_sac_antmaze_g99` finished (3 seeds x 10 ckpts) and reproduces §4e exactly:
`actor_u_norm` **5.43 / 5.54 / 5.55** against antmaze's clipped-prior median of 2.70, i.e.
**2.01-2.06x** -- the same ratio as at gamma=0.98, and the same ~2.0x seen on cube and
pointmaze. `alpha` collapsed to 0.0006-0.0012, in-loop success 0.0 at every late checkpoint.
The horizon fix that lifts the ACTOR-FREE arm from 0.067 to 0.252 does nothing here, because
the failure is the entropy port and not the discount. Evals scoped to 500k x 3 seeds on the
same reasoning as §4e.

## 4d. Pre-registered: the `prior_shrunk` support control

New eval-time mode `gpi_select=prior_shrunk` with `gpi_prior_shrink` (default 0.8; off
unless asked for, `argmax` remains the shipped rule). One prior draw scaled by that factor,
decoded through the frozen flow. **It reads neither `psi` nor `task_z`** -- pinned by test --
so it is a property of the frozen flow alone. 0.8 * 2.13 (the clipped prior median at
d_a = 5) = 1.70, matching `dsrl_na`'s measured `actor_u_norm` of 1.72. At scale 1.0 the mode
IS the BC control, so it isolates the shrinkage and nothing else.

**Consequence for the eval plan, worth stating before slots are spent:** because the policy
never touches psi or the task vector, its score is IDENTICAL at every checkpoint and every
training seed. Running it at 5 checkpoints x 3 seeds would be 15 evaluations of the same
policy. The informative spend is **3 evaluations at 3 different EVAL seeds** (which is where
its only variance lives), plus **one** at a 300k cube checkpoint purely to confirm the
checkpoint-independence the test asserts. Same for antmaze g99: one eval, not a ladder.

**Prediction, registered before running: the control will NOT match `dsrl_na`.** Expected
**0.05-0.12 on cube, i.e. at BC (0.072)**, against `dsrl_na`'s in-loop ~0.3. The reasoning is
that the 09-05 ablations already swept exactly this axis and found it flat:

| critic-free rule (09-05, cube @350k) | selected \|u\| | success |
|---|---|---|
| `small_ball` (below-median norm) | 1.58 | 0.670* |
| one prior draw = BC | 2.13 | 0.072 |
| `top_quartile_random` | 3.01 | 0.078 |
| `max_norm` | 3.81 | 0.078 |

*`small_ball` still ranks by the critic; the two genuinely critic-free rules are the last
two, and they sit at BC despite spanning \|u\| 3.01 to 3.81. Nothing critic-free has ever
scored above 0.09 on cube at any radius, so a pure 20 % shrink lifting BC from 0.072 to ~0.3
would need a fourfold gain from a state-INDEPENDENT scale change.

So I expect this control to **refute** the support explanation rather than confirm it, and
the informative outcome is then the opposite of the one that closes the line: if
`dsrl_na` >> `prior_shrunk` ~ BC, then the NA actor's state-dependence is doing real work
despite its regression explaining only ~6 % of the argmax, and the interesting question
becomes what it is tracking instead. If instead the control DOES match `dsrl_na`, the arm
inherits the support and the DSRL-actor line closes cleanly -- which is the cheaper result
and worth the three evals either way.

## 4b. Incident: twelve evals killed by a py/yaml skew (2026-09-07 00:11-00:16)

Self-inflicted, found by the GPU-packing agent, fixed. Recorded because the fix is a
standing constraint on this agent, not a one-off.

**What happened.** The `actor.entropy` guard landed in `agents/psmflow.py` a few minutes
BEFORE the matching key landed in `configs/agent/psmflow.yaml`. During that window every
hydra-launched psmflow build read a key the config group did not define. Twelve queued
500-episode evaluations of *older* checkpoints started inside it and died at
`create()` with `KeyError: "'entropy'"`, through
`tools/eval_checkpoint.py -> _evaluate_shard -> create` (not through `main.py`):

  2491907, 2491909, 2491910, 2491911 (pointmaze ladder 100k/150k)
  2491912, 2491913 (antmaze discount g995/g99 @100k)
  2491914, 2491915, 2491916, 2491919, 2491920, 2491921 (pointmaze ladder 200k/250k)

All 12 exited 1 with no result written. They need resubmitting; nothing else was lost.
2491922, a resubmission of 2491912 at 00:16:25, COMPLETED — which independently dates the
end of the window to the yaml edit.

**Not this bug** (checked, different owners): 2491881-2491889 exited 126 *after* writing
their JSON reports, on `scripts/eval500.sh: line 100: <preimage npz>: Permission denied`
(the script tries to execute the npz path); 2491924 `sceneBCprobe` exited 2 *after* writing
its report, on `scripts/eval500.sh: line 134: unexpected EOF while looking for matching '"'`.
Both are `scripts/eval500.sh` shell bugs and both left usable results. 2491890-2491894
succeeded. The `fql` / BC control path has no `actor` block at all and was never affected —
verified by constructing it.

**Root cause, generalised.** A restored checkpoint's `flags.json` is by construction OLDER
than the code restoring it, so **every `actor` key this agent ever adds must be optional at
read time**, and the yaml must never lag the module. Nothing in the suite could see the gap:
every existing test builds its config from this checkout's own `get_config()`, which always
has every key.

**Fix.** `fill_actor_defaults(config)` is now the first statement of `create()`, so no read
below it — guard, network construction, or the frozen runtime config — can hit a missing
key, whatever container the caller passed. `_actor_opt` and the post-`_plain_config`
backfill remain as the fallback for a locked ConfigDict.

**Regression test**, `tests/test_psmflow_config_compat.py` (9 cases):
- three *verbatim* archived `flags.json` files in `tests/fixtures/` — pre-affine
  (`policy_index=task_vector`, no `psi_form`), affine strict, affine + latent actor — each
  driven through the eval tool's own `merge_run_config` and then `create`, asserting both
  that it builds and that the new keys land on the values reproducing that run's behaviour;
- a bare old `actor` dict built without the eval tool (the `main.py` path);
- **yaml <-> `get_config()` leaf-key parity**, the one assertion that would have caught the
  outage. Negative control run: deleting `entropy` from the yaml fails it, restoring passes.

## 5. Provenance

- New: `tools/diag_actor_grad_terms.py`, `utils/psm_networks.py::TanhGaussianLatentActor`,
  `agents/psmflow.py::flow_actor_loss` branches, `tests/test_psmflow_dsrl_actor.py`.
- Reports: `$PSM_DATA/logs/diag_actor_grad_affine_cube_sd{0,1}_500k.json`.
- Reference clones read for this audit: `dsrl_main`, `sb3_dsrl`, `dsrl_pi0` (scratch checkouts, not retained).
- g99 runs: `affine_dsrl_naadv_antmaze_g99` SLURM **2492040-2492042** (seeds 0/1/2),
  `affine_dsrl_sac_antmaze_g99` SLURM **2492043-2492045**, 500k, 11 h wall, smokes
  2491983 (naadv, 2000 steps) and 2491977 (sac); both smokes' `flags.json` verified to carry
  `discount=0.99` with the arm's actor keys. Antmaze step rates 0.0324 (naadv) / 0.0258
  (sac) s/step -> ~4.5 h / ~3.6 h of loop time.
- corrected SAC: `affine_dsrl_sac_te_cube` SLURM **2492091-2492093** (seeds 0/1/2),
  `target_entropy=-3.4657`, `u_clip=3.0` unchanged, 8 h wall, smoke 2492086.
- ranking probe: `tools/diag_actor_vs_gpi.py`, SLURM **2492090** (8 cells).
- Runs: groups `affine_{dsrl_na,dsrl_sac}_{cube,antmaze,pointmaze}`, 3 seeds each, 500k,
  `save_interval=50000`, `scripts/slurm/train_psmflow.sbatch`, launched 2026-09-07 00:23.
  SLURM **2491932-2491949** (na: cube 2491932-34, antmaze 2491935-37, pointmaze 2491938-40;
  sac: cube 2491941-43, antmaze 2491944-46, pointmaze 2491947-49), seeds 0/1/2 in that
  order. Smokes 2491930 / 2491931.
- Tests: `tests/test_psmflow_dsrl_actor.py` (18 cases), run module-per-process.

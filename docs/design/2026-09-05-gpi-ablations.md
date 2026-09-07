# Is the cube number the critic, or the max over K draws? — `gpi_select` ablations

Date: 2026-09-05 · Branch `feat/inversion-integration` · Design + pre-registration.
Code: `agents/psmflow.py` (`gpi_select` / `_gpi_select_ablation`), `configs/agent/psmflow.yaml`
(`agent.gpi_select`, `agent.gpi_topm`, `agent.gpi_index_seed`),
`scripts/slurm/eval_gpi_ablation.sbatch`, `tests/test_psmflow_gpi_select.py`.
Subject: `affine_strict_cube/sd000_s_2491597.0.20260904_181112`, the repo-default
paper-strict agent (`psi_form=affine policy_index=latent index_agg=max train_actor=false
acting=gpi`).

## 1. The question

`docs/design/2026-09-05-gpi-selection-diag.md` measured three things about the shipped
acting rule (per step: draw `K=64` action latents `u` and `K` policy indices `u'` from the
clipped prior, score all `K^2` pairs by `Q = psi(s,u,u')^T w` with `mean - 0.5*std`, decode
`argmax_u`):

1. the selected `u` has norm ~2.8 against the candidate roster's 2.13, with ~4x the
   roster's clip-coordinate fraction — **at every checkpoint, good and bad alike**;
2. the per-`u` ranking `max_{u'} Q` re-randomises between checkpoints (Spearman 0.21-0.36
   for any pair, including two that both work) and scores +0.10..+0.15 against an MC
   ground truth, where the checkpoint-independent baseline `-||u||` scores **+0.28**;
3. `psi`'s spread over the `u'` slot is ~30x its spread over `u`, so the inner argmax —
   the one that picks the action — runs on the weakest part of the signal.

Together those say the deployed rule may be, to first order, **a max over K prior draws
with a norm bias**, wearing a critic. If so, a critic-free rule with the same norm bias
should reproduce the deployed success. That is the question these arms answer, at eval
time, with no retraining:

> Is cube 0.704 @350k the critic, or is it selection bias toward large-norm latents that
> any max over K draws would produce?

## 2. The switch

`agent.gpi_select` (default `argmax`) chooses which of the `gpi_num_u` prior draws is
handed to the frozen flow. The default branch of `gpi_select` is untouched code — the
ablations live in a separate method `_gpi_select_ablation` reached only when the config
asks — and `tests/test_psmflow_gpi_select.py` pins the default's selection bit-for-bit
against the shipped expression, so no published number moves because the switch exists.
`gpi_select` did not exist when these runs were trained, so it is absent from their
`flags.json`; `merge_run_config` only inherits keys present in BOTH dicts, so the fallback
is this checkout's default `argmax` — the old behaviour — with no `LEGACY_AGENT_DEFAULTS`
entry needed (also pinned by a test).

Selected-`|u|` under each rule (`d_a=5`, clipped prior, 200k draws; the deployed rule
selects 2.8 and the roster averages 2.13):

| rule | mean selected `|u|` |
|---|---|
| one prior draw (BC control) | 2.13 |
| `small_ball` (`<=` prior median) | 1.58 |
| **shipped `argmax`** (measured) | **2.8** |
| `top_quartile_random` | 3.01 |
| `max_norm` (max of 64) | 3.81 |

`top_quartile_random` is therefore the norm-matched critic-free control: it reproduces the
deployed rule's selection statistic (slightly overshoots it) while never reading `psi`.

## 3. Hyperparameters (unchanged from the run's own `flags.json`, inherited by the eval)

| key | value | key | value |
|---|---|---|---|
| `agent_name` | psmflow | `z_dim` | 128 |
| `psi_form` | affine | `affine.w_dim` | 128 (`norm_w=true`) |
| `policy_index` | latent | `sf` | 1024 x 1, emb 2 |
| `index_agg` | max | `phi` | 256 x 2 |
| `acting` | gpi | `num_parallel` | 2 |
| `train_actor` | false | `actor_pessimism_penalty` | 0.5 (= exact min-Q) |
| `gpi_num_u` | 64 | `pessimism_penalty` | 0.5 |
| `gpi_decode` | onestep | `discount` / `tau` | 0.98 / 0.01 |
| `u_clip` | 3.0 | `ortho_coef` | 1000.0 |
| `use_point_preimage` | true | `backup_explore_frac` | 0.0 |
| `action_critic.enabled` | false | `mix_ratio` | 0.5 |

Eval protocol: `tools/eval_checkpoint.py`, 500 episodes, `seed=0`, env
`cube-single-play-singletask-v0`, flow `$PSM_DATA/flow/cube-single-play` @500000,
preimages `$PSM_DATA/preimages/cube-single-play.npz`, `eval_relabel_size=10000`,
`eval_reward_shift=1.0` — i.e. exactly the protocol that produced .704/.086. One SLURM job
per arm (`kisski-inference`, `general`, `--gres=gpu:1`, 1 h), reports at
`$PSM_DATA/logs/eval500_gpiabl_<arm>_<epoch>.json`.

## 4. The arms

| arm | `gpi_select` | epoch | notes |
|---|---|---|---|
| G | `argmax` | 350k | reproduction control: must return 0.704 |
| A1 | `max_norm` | 350k | **critic-free**: largest of the 64 draws |
| A2 | `top_quartile_random` | 350k | **critic-free**, norm-matched to the deployed pick |
| B | `argmax`, `gpi_num_u=1` | 350k | one prior draw per step: must return BC 0.072 |
| C1 | `small_ball` | 350k | candidates rejected above the prior median `|u|` |
| C2 | `small_ball` | 500k | " |
| D1 | `soft_topm`, `m=8` | 350k | uniform among the top 8 of 64 by GPI score |
| D2 | `soft_topm`, `m=8` | 500k | " |
| E | `mean` | 350k | ensemble mean instead of `mean - 0.5*std` |
| F1 | `fixed_index`, seed 0 | 350k | `u'` pinned for the whole eval; only `u` varies |
| F2 | `fixed_index`, seed 1 | 350k | second draw, to see if the pinned index is a lottery |

A1/A2 do not consult `psi` at all (a test pins that zeroing `task_z` leaves their
selection unchanged), so they are checkpoint-independent up to the frozen flow; 350k is
used only for convenience of loading an agent.

Reference points: BC control **0.072**, shipped GPI **0.704 @350k**, **0.086 @500k**
(also .532/.286/.282/.272 at 250k/300k/400k/450k). At n=500 the Wilson half-width is
about ±0.040 at p=0.7 and ±0.026 at p=0.1, so a gap above ~0.06 is resolvable.

## 5. Pre-registration (written before any arm returned)

- **G** returns 0.704 exactly (same seed, same protocol, byte-identical selection). If it
  does not, the switch broke something and nothing else here is interpretable.
- **B** returns 0.05-0.10, i.e. the BC control. The only difference from the published BC
  number is that psmflow's prior is clipped at 3.0 (0.27% of coordinates) — negligible.
- **A1 `max_norm`**: *expected 0.02-0.20, well below 0.704.* Mean `|u|` 3.81 is far outside
  the typical set the frozen flow was fit on; its decode saturates the action clip. If the
  norm bias alone were the mechanism, more of it should be better, and this is the arm
  that says whether "more" helps or hurts.
- **A2 `top_quartile_random`**: the decisive arm. *Expected 0.08-0.25.* Prediction: it does
  **not** reach 0.704 — i.e. norm-matched critic-free selection does not reproduce the
  deployed number, and the 350k critic is doing something. **The pre-registered surprise
  is A2 >= ~0.4**, which would mean the headline cube number is a selection artefact and
  the affine critic is decoration; the diag's finding (2) makes that a live possibility
  and it is the reason this arm exists.
- **C `small_ball`**: the MC ground truth endorses smaller norms (`-||u||` at +0.28 beats
  every checkpoint's critic at +0.10..+0.15). *Expected: 500k lifts, 0.10-0.35, from
  0.086; 350k roughly holds or drops moderately, 0.40-0.75.* A lift at 500k with a hold at
  350k is the one outcome that makes this a shippable stabiliser.
- **D `soft_topm`**: same medicine as checkpoint averaging (MC Spearman 0.121 -> 0.166,
  regret 23.75 -> 19.2) applied within one checkpoint. *Expected: 350k drops to 0.30-0.65
  (it gives up a sharp pick that at this checkpoint happens to be good), 500k lifts
  slightly to 0.08-0.25.* If the ranking were pure noise, D would sit near B/BC at both
  checkpoints.
- **E `mean`**: *expected a no-op, 0.60-0.75, inside G's CI.* The diag measured rank
  correlation 0.93-0.95 between the pessimistic and mean rankings. This arm is one run to
  confirm a negative that is already recorded.
- **F `fixed_index`**: `psi` discriminates `u'` ~30x more strongly than `u`, and the
  deployed rule takes a fresh max over 64 indices every step. *Expected: a large move in
  either direction, 0.10-0.60, and a visible spread between F1 and F2* — if pinning the
  index costs most of the 0.704, the per-step index max (an optimism device, not a policy
  choice) is carrying the number; if F ~= G, the `u'` axis is a nuisance dimension and the
  `K^2` scan can be replaced by a `K` scan.
- **Possible negative for the whole batch:** every arm lands in 0.05-0.20 except G and E,
  i.e. only the exact shipped rule at the exact 350k checkpoint is high. That would say
  the 0.704 is neither the norm bias nor a stabilisable critic but a coincidence between
  one lottery ticket and one rule — and the reporting protocol (a mean over checkpoints
  and seeds, never a peak) is the only honest way to quote this agent.

## 6. Results

Eleven concurrent 1 h jobs, all COMPLETED; reports in
`$PSM_DATA/logs/eval500_gpiabl_<arm>_<epoch>.json`. Section 5 was not edited afterwards.

| arm | rule | epoch | k/500 | success | Wilson 95% | vs shipped | vs BC 0.072 |
|---|---|---|---|---|---|---|---|
| G | `argmax` (shipped) | 350k | 352 | **0.704** | [0.663, 0.742] | — (exact reproduction) | +0.632 |
| A1 | `max_norm` (critic-free) | 350k | 39 | 0.078 | [0.058, 0.105] | -0.626, z=-20.3 | +0.006, p=0.72 |
| A2 | `top_quartile_random` (critic-free) | 350k | 39 | 0.078 | [0.058, 0.105] | -0.626, z=-20.3 | +0.006, p=0.72 |
| B | `argmax`, K=1 (one prior draw) | 350k | 45 | 0.090 | [0.068, 0.118] | -0.614, z=-19.8 | +0.018, p=0.30 |
| C1 | `small_ball` | 350k | 335 | 0.670 | [0.628, 0.710] | -0.034, p=0.25 | +0.598 |
| C2 | `small_ball` | 500k | 23 | 0.046 | [0.031, 0.068] | **-0.040, p=0.011** | -0.026, p=0.08 |
| D1 | `soft_topm` m=8 | 350k | 306 | 0.612 | [0.569, 0.654] | **-0.092, p=0.002** | +0.540 |
| D2 | `soft_topm` m=8 | 500k | 31 | 0.062 | [0.044, 0.087] | -0.024, p=0.15 | -0.010, p=0.53 |
| E | `mean` (no pessimism) | 350k | 367 | 0.734 | [0.694, 0.771] | +0.030, p=0.29 | +0.662 |
| F1 | `fixed_index`, seed 0 | 350k | 0 | **0.000** | [0.000, 0.008] | -0.704, z=-23.3 | **-0.072, p=1e-9** |
| F2 | `fixed_index`, seed 1 | 350k | 90 | 0.180 | [0.149, 0.216] | -0.524, z=-16.7 | +0.108, p=3e-7 |

Reference: BC control 0.072, shipped GPI 0.704 @350k and 0.086 @500k. "vs" columns are
two-proportion z-tests at n=500 each.

### 6.1 The answer: the critic, not the norm bias

**G reproduced 0.704 exactly (352/500)**, so the switch changed nothing and every row is
comparable to the published numbers.

The two critic-free controls **both land at 0.078**, indistinguishable from the BC control
(p=0.72) and 0.63 below the shipped rule. That holds for `top_quartile_random`, whose
selected `|u|` (3.01) *overshoots* the deployed rule's measured 2.8 — it is norm-matched
and then some. `max_norm` (`|u|`=3.81) is the same 0.078: pushing further into the tail
neither helps nor hurts relative to a single draw. And B, one prior draw per step, returns
0.090, consistent with BC (p=0.30), which confirms the wiring end to end.

So **the large-norm selection bias contributes nothing measurable to cube success.** The
diag's finding (1) is real as a description of what the argmax does and a red herring as
an explanation of why it scores: at 350k the critic's ranking carries essentially the
entire 0.704 - 0.072 gap. Reproducing the deployed rule's selection *statistics* without
its selection *content* buys 0.006 over behaviour cloning.

The mirror image is the 500k row: the same machinery, the same norm bias, and 0.086 — at
the BC control. The critic is worth +0.63 when it is good and worth nothing when it is
not, and **no eval-time intervention recovered it**: `small_ball` made 500k significantly
*worse* (0.046, p=0.011), `soft_topm` left it unchanged (0.062, p=0.15). The 500k failure
is in the learned `psi`, not in the rule that reads it.

### 6.2 The stabilisers: one holds, none lifts

- **`small_ball` (C)** — pre-registered as the most promising, on the strength of `-||u||`
  beating every checkpoint's critic on the MC ground truth. It **holds the good checkpoint**
  (0.670 vs 0.704, p=0.25) and **hurts the bad one** (0.046 vs 0.086, p=0.011). The MC
  proxy's endorsement of small norms did not transfer: it was measured on open-loop
  fixed-`u` rollouts, and the deployed rule reselects every step. Do not ship it.
- **`soft_topm` (D)** — costs 0.092 at 350k (p=0.002) and buys nothing at 500k (p=0.15).
  Within-checkpoint softening is not the same medicine as cross-checkpoint averaging: the
  top-8 of one checkpoint's ranking share that checkpoint's noise. Do not ship it.
- **`mean` (E)** — 0.734 vs 0.704, p=0.29. The pre-registered no-op, confirmed. (If
  anything it is nominally *higher*, which is one more datum against the pessimism term
  doing useful work in the acting path.)

### 6.3 The unregistered finding: the per-step index max is the mechanism

`fixed_index` was pre-registered as "a large move in either direction" and delivered the
largest effect in the batch. Pinning `u'` to ONE prior draw for the whole eval — leaving
the argmax over `u` intact, over the same K=64 draws, with the same critic — gives
**0.000 (seed 0)** and **0.180 (seed 1)**. Seed 0 is not merely bad, it is *significantly
below behaviour cloning* (0/500, p=1e-9): with the index pinned, ranking `u` by
`Q = psi(s,u,u')^T w` selects actively harmful actions.

This is the acting-side counterpart of the diag's `index_matters` asymmetry (psi's spread
over `u'` is ~30x its spread over `u`). The deployed rule's value is not a stable
preference over action latents — pin the index and that preference is worth between -0.07
and +0.11 against BC, depending on which `u'` you happened to draw. What produces 0.704 is
the **per-step max over 64 fresh policy indices**: an optimism device that averages over
the index lottery every single step. The `K^2` pair scan is therefore load-bearing and
cannot be collapsed by pinning `u'`.

## 7. What this says to do

1. **Keep the default.** `gpi_select=argmax` stays; `mean` is a tie and buys nothing;
   `small_ball` and `soft_topm` are measured losses. The switch stays in the tree as a
   documented ablation, not as a new default.
2. **Stop looking for the cube fix at eval time.** Five interventions on the selection
   rule leave the 500k checkpoint at or below BC. The 0.704 -> 0.086 collapse is in `psi`,
   so the next moves are training-side (or checkpoint/seed aggregation), not acting-side.
3. **The norm-bias story is dead as a performance explanation** and should not be written
   up as one. It remains a true description of the argmax's behaviour.
4. **The index axis is the live lever.** The per-step max over `u'` is what the 0.704 is
   made of, and it is an *optimistic* max over samples of a learned function — exactly the
   structure E4a indicted. `index_agg=expectile` (S2) replaces that argmax with a
   distilled upper expectile and is now the best-motivated next experiment; a larger
   `index_panel` and a `gpi_num_u` sweep on the index axis alone are the cheap companions.
5. **Reporting is unchanged**: mean and 95% CI over seeds and checkpoints, never a peak.
   0.704 and 0.086 are the same agent, the same rule and the same norm bias.


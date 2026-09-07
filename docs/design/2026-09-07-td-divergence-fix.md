# Two stabilisers for the diverging TD term — pre-registration

Date: 2026-09-07 · Branch `feat/inversion-integration` · Design + **pre-registration,
written and committed before any training job was submitted**.
Machine: KISSKI (SLURM, H100, `kisski-inference`, `general`).
Code: `agents/psmflow.py` (`bound_psi`, `psi_b`, `measure_loss`, `STABILITY_DEFAULTS`),
`configs/agent/psmflow.yaml` (`ortho_mode`, `ortho_rel_coef`, `psi_bound`,
`psi_bound_scale`). Tests: `tests/test_psmflow_stabilisers.py`,
`tests/test_psmflow_config_compat.py`.

Acts on `docs/design/2026-09-07-antmaze-g995-collapse.md` §7.3 candidates **1** and **2**.
Candidates 3 (use `gamma=0.99`) and 4 (lower `lr_sf`) defer rather than fix and are not
run here; 5 (Polyak) and 6 (reward scaling) were ruled out there.

## 1. The object being fixed

`affine_strict` (`psi_form=affine policy_index=latent index_agg=max train_actor=false
acting=gpi`) on `antmaze-medium-navigate-singletask-v0` at `agent.discount=0.995`, three
seeds. The chain the collapse note dated, on all three seeds:

| step | what | when |
|---|---|---|
| 1 | `psm_loss` leaves the `gamma=0.99` band, grows **one decade per 31.0k steps** (2.5e3 @50k -> 8e17 @500k; the rate agrees to 3 s.f. across seeds) | ~50–85k |
| 2 | it exceeds `ortho_coef * \|orth_loss\|` = 1000 x 64 = **6.4e4**; the geometry regulariser stops being a constraint | 115–125k |
| 3 | `orth_offdiag` leaves its 64 baseline; phi collapses toward rank one (max 8192) | 140–145k |
| 4 | the last off-rank-1 energy goes; `\|Q\|` 1e6 -> 3e7 -> ~1e9; success 0.125 -> 0.015 | 250k -> 300k |

500-episode ladder (BC control **0.072**): 0.492 / **0.519** / 0.201 / 0.142 / 0.125 /
0.015 / … / 0.004 at 50k / 100k / 150k / 200k / 250k / 300k / 500k. Pooled windows:
**50k–250k 0.296**, **300k–500k 0.010**. The `gamma=0.99` arm's late window is **0.252**.

phi is projected to the sphere of radius `sqrt(z_dim)` (`orth_diag` pinned at -128) and
`w(u')` is unit-norm, so `A(s,u)` and `beta(s,u)` are **the only free magnitudes in the
model**, and `psi_q_spread` is the scalar that runs 7 -> 2e8.

## 2. What was implemented

Both switches are **off by default**; `ortho_mode=fixed`, `psi_bound=none` reproduce the
published loss expression for expression. `tests/test_psmflow_stabilisers.py` pins the
default path against a from-scratch re-implementation of `sm + ortho_coef * ortho` at an
unbounded psi and an unclipped target, and pins that naming the off values explicitly is
byte-identical to naming nothing.

### 2.1 `ortho_mode: fixed | relative`, `ortho_rel_coef`

```
w_ortho = ortho_coef + ortho_rel_coef * stopgrad(|psm_loss|)
loss    = psm_loss + w_ortho * ortho_loss
```

The ratio of the two loss terms becomes `|ortho| * (ortho_coef/|psm| + r)`, bounded below
by `r * |ortho|` however far the TD term runs; under the fixed weight the same ratio is
`ortho_coef*|ortho| / |psm|` and falls through 1 at `|psm| = 6.4e4`, which **is** step 2 of
the table above. Because the weight is stop-gradded and `ortho` does not depend on psi,
this changes **phi's gradient direction and nothing else** — a test asserts psi's step is
bit-identical to the fixed arm.

`ortho_rel_coef = 1.0` holds the ortho term at `64 x psm_loss` at the healthy baseline
(`|orth_loss| = 64`). The measured healthy band of that ratio is ~2–100 (cube 1.8–43,
antmaze `g99` ~100, antmaze `g995` ~25 before it diverged), so 1.0 sits mid-band; it is not
a swept value and no sweep is run here.

### 2.2 `psi_bound: none | tanh | clip_target`, `psi_bound_scale`

**The form chosen for the arms below is `tanh`: a bounded reparameterisation**

```
psi(s,u,u') = S * tanh( (A(s,u)^T w(u') + beta(s,u)) / S )
```

**Why this form and not a penalty or a projection.** The head still emits the write-up's
`A^T w + beta`; the squash is elementwise and strictly monotone, so psi keeps its sign
structure and — the property GPI actually consumes — its **ordering in the policy
coordinate**, and at `|psi| << S` it is the identity to first order. A norm penalty would
add a second objective whose weight has to be traded against the TD term (the very trade
that failed at step 2), and a hard projection would put a kink in psi's gradient. A
bounded reparameterisation adds no term to the loss and no hyper-parameter beyond the
ceiling itself.

**`S` is derived, not swept.** The ortho term enforces `E[phi phi^T] = I`, so `phi_i` has
unit RMS and the successor feature is `psi_i = E[sum_t gamma^t phi_i(x_t)]`, giving
`|psi_i| <= 1/(1-gamma)` — **200 at `gamma=0.995`** (100 at 0.99). Anything the head wants
above that ceiling is not a successor measure of this basis. `psi_bound_scale=200.0` is
that number.

`clip_target` is implemented as the same admissible set imposed on the TD bootstrap
instead — `target_M` clipped to `+-(S * z_dim)`, the Cauchy–Schwarz image
`|psi^T phi| <= (S sqrt(z))(sqrt(z))` of the psi ball — so the two modes cap the same
quantity and `psi_bound_scale` keeps one meaning. It is **not** run in the arms below; it
exists as the "leave the forward head alone" fallback if `tanh` saturates.

Every psi READ in the agent goes through `psi_b` — the measure loss, the bootstrap, the
actor panel, `q_dist`, `index_spread`, `gpi_select` and the selection ablations — so a
bounded run is bounded at acting time too. The affine fast path in `_psi_q_over_indices`
(which contracts in `w`-space) is skipped under `tanh`, since an elementwise squash on the
`z_dim` output cannot be pushed through that contraction; a test asserts the fallback
agrees with a naive per-index computation. The strict arm never reaches that path
(`index_panel=0`), so the arms below pay no extra cost.

### 2.3 Telemetry (logged in **every** arm, off values are exact constants)

`ortho_weight`, `ortho_term_abs` (= `|w_ortho * ortho|`, so the step-2 crossing is readable
straight off `train.csv` without recomputing), `psi_absmean`, `psi_absmax`,
`td_target_absmean`, `psi_bound_frac` (saturation fraction under `tanh`, clip fraction
under `clip_target`, exactly 0.0 under `none`).

### 2.4 Old checkpoints

`STABILITY_DEFAULTS` + `fill_stability_defaults` mirror the `ACTOR_DEFAULTS` /
`fill_actor_defaults` contract that the 2026-09-07 twelve-eval outage produced: a
`flags.json` written before these keys existed restores onto the OFF values, i.e. onto the
loss the run was actually trained with. `tests/test_psmflow_config_compat.py` asserts this
against the three archived fixtures and asserts yaml <-> `get_config()` parity for the new
keys.

## 3. Arms

`ENVKEY=antmaze`, `DISCOUNT=0.995` — **the fastest-failing setting**, chosen so a fix that
does not work says so inside 500k steps rather than inside 2M. 500k steps, save every 50k,
3 seeds each, one GPU per seed, 6 h wall (the unfixed g995 runs took 4.0 h).

| arm | group | `EXTRA` |
|---|---|---|
| **F1** ortho relative | `affine_strict_antmaze_g995_orel` | `agent.ortho_mode=relative agent.ortho_rel_coef=1.0` |
| **F2** psi bound | `affine_strict_antmaze_g995_pbound` | `agent.psi_bound=tanh agent.psi_bound_scale=200.0` |
| **F3** both | `affine_strict_antmaze_g995_both` | both of the above |

Comparators, already measured, **not re-run**: unfixed `g995` (300k–500k **0.010**,
50k–250k 0.296), `g99` (late **0.252**), BC control **0.072**.

## 4. PRE-REGISTRATION

**Pre-registered success criterion (all arms judged on it): the 300k–500k pooled 500-episode
mean is `>= 0.25`, with `psm_loss` bounded over the same window.** 0.25 is the `gamma=0.99`
arm's late window (0.252) — i.e. the fix has to buy back the whole gap that `gamma=0.995`
currently gives away, not merely beat the 0.010 floor or the 0.072 BC control.

### 4.1 Per-arm predictions

| quantity | **F1** ortho relative | **F2** psi bound | **F3** both |
|---|---|---|---|
| `psm_loss` @500k | **still diverges** — 1e12–1e18, possibly slower than 8e17 | **bounded**, plateau by ~200k; point estimate **3e5** (analytic ceiling `0.5*((1+g)Sz)^2 ~ 1.3e9`) | **bounded**, same plateau as F2 |
| `psm_loss` crosses 6.4e4 | yes, ~85–125k, but **harmless** — the ortho weight moves with it | **probably yes** (plateau above 6.4e4) | yes, and **harmless** |
| `ortho_term_abs` vs `psm_loss` | ratio pinned at `>= 64`, never crosses | falls as the plateau is approached; may end `< 1` | ratio pinned at `>= 64` |
| `orth_offdiag` | **stays at baseline 64** for all 500k (RMS angle 84.9 deg, `resid` 0.992) | **uncertain — the discriminating cell.** Predict it leaves baseline but stops short of 8192; point estimate 300–3000 at 500k | **stays at baseline 64** |
| `\|Q\|` (`psi_q_spread/_rel`) | **unbounded**; predict 1e4–1e6 at 500k (below the unfixed 1e9 because phi stays full-rank) | **bounded by `S*z_dim = 2.56e4`**; operating level 1e3–1e4 | **bounded**, 1e3–1e4 |
| `psi_bound_frac` | 0.0 (off) | 0.05–0.5 by 300k | 0.05–0.5 by 300k |
| `psi_q_spread_rel` (the inner argmax's signal) | unchanged, 0.004–0.010 | **the risk**: predict 0.002–0.010; below 0.002 is the flattening failure | 0.002–0.010 |
| **300k–500k pooled 500-ep** | **0.30** | **0.20** | **0.35** |
| verdict on the criterion | **PASS** (marginal) | **FAIL** (predicted) | **PASS** |

**F3 is the pre-registered favourite**, and the reasoning that makes it so is exactly the
reasoning that makes F2 fail on its own: bounding psi caps `psm_loss` at a level set by
`S`, but that level (~3e5, analytic worst case 1.3e9) is **above** the 6.4e4 at which the
fixed ortho weight stops binding. Boundedness alone therefore does not buy a healthy basis;
it only buys a slower, shallower version of the same collapse. F1 removes the crossing but
leaves `|Q|` free to grow. Only F3 does both. Writing this down is the point of a
pre-registration: **if F2 passes the criterion on its own, the account in §7.3 of the
collapse note is wrong about which of the two mechanisms is load-bearing.**

### 4.2 Named failure modes (what each arm looks like when the fix is the problem)

* **F1 starves the TD fit.** With the ortho term pinned at 64x the TD term, phi is held so
  hard against orthonormality that it cannot shape itself to the task. Signature: success
  **falls early** — the 100k point comes in **below the unfixed arm's 0.519** (predict
  `< 0.40` is the alarm), `psi_q_index_spread_rel` stays near its 0.35 init instead of
  rising to 0.5–0.8, and the 50k–250k window comes in below the unfixed 0.296. This is a
  fix that trades the late collapse for an early ceiling, and the 100k cell is where it is
  visible.
* **F2 (and F3) flatten `Q`.** If `S=200` sits below the operating range the head needs,
  `psi_bound_frac` runs to ~1.0, psi saturates, and the GPI **lottery gets worse**: the
  per-`u` argmax margin relative to `|Q|` — the one row of the 09-07 probe that saw the
  collapse (7.3e-3 -> 6.3e-4) — falls, `psi_q_spread_rel` drops below 0.002, and success
  heads for the `fixed_index` floor (~0.05, below BC). Signature: `psi_bound_frac > 0.8`
  **and** `psi_q_spread_rel < 0.002` together. The remedy would be `clip_target` (forward
  head unbounded) or a larger `S`, not a different `ortho_mode`.
* **All three fail with a healthy basis.** `orth_offdiag` at 64, `psm_loss` bounded, `|Q|`
  bounded, and the 300k–500k window still `<= 0.10`. Then basis collapse was a **symptom**,
  not the load-bearing cause, `gamma=0.995` is outside the stable region for a reason that
  is not in the loss geometry, and candidate 3 (train at 0.99) is the answer after all.
  This is the outcome that would retire the whole §7.3 ranking.
* **Anything reads NaN.** Predicted **false** in every arm (the 09-07 probe found zero
  NaN/Inf at `|Q| ~ 1e7`, fp32 overflows at 3.4e38, and both stabilisers only ever
  *reduce* magnitudes). A NaN would mean the bound's `tanh(x/S)` is being fed an already-
  infinite psi, i.e. a different bug.

### 4.3 Evaluation plan (fixed now, so the reporting cannot be shopped)

500 episodes (`scripts/eval500.sh`, `EVAL_WORKERS=1`) at **100k, 300k, 400k, 500k** for
every run — 9 runs x 4 = **36 evals**, 3 h wall each. 100k is the pre-collapse peak of the
unfixed arm and is where F1's named failure mode shows; 300k/400k/500k are the pooled
window the criterion is stated on. Reported as mean and 95% CI across the three seeds,
never a peak and never a best seed, with the BC control (0.072) quoted beside it.

Outputs: `docs/tables/affine_antmaze_g995_fix.md`,
`docs/figures/2026-09-07-td-divergence-fix.{png,json}` (`psm_loss`, `orth_offdiag`, `|Q|`
and success per arm against the unfixed g995 and the g99 control), and a
`docs/HANDOFF.md` entry.

## 5. Results

To be filled in after the evals. Nothing above this line may change once the first job is
submitted.

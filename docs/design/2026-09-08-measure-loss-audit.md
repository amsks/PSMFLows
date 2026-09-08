# Audit of the measure loss: why `psm_loss` diverges, and what the second lever is

Date: 2026-09-08 · Branch `feat/inversion-integration` · Derivation + pre-registration.
Machine: KISSKI (SLURM, H100, `kisski-inference`, `general`).
Code read: `agents/psmflow.py`, `utils/psm_common.py`, `utils/psm_networks.py`,
`configs/agent/psmflow.yaml`, `archive/agents/fb.py`, `archive/configs/agent/fb.yaml`,
`git show 5249267:PAPER/main.tex` §`losses` and §`latentflowpsm`.
Code written: `scripts/audit/measure_loss_growth.py` (read-only analysis),
`scripts/audit/launch_measure_loss_ablations.sh`. **Nothing under `agents/`, `utils/` or
`configs/` is touched** — another agent has `ortho_mode` / `psi_bound` in flight there, and
every ablation below is expressed through config keys that already exist.

Follows `docs/design/2026-09-07-antmaze-g995-collapse.md` (which dated the collapse) and
`docs/design/2026-09-07-td-divergence-fix.md` (which pre-registered `ortho_mode=relative`
and `psi_bound=tanh`). It answers a different question from either: **not "how do we stop
phi collapsing" but "why is the backup non-contractive in the first place".**

---

## 0. Answer in one paragraph

The measure loss is fitted with a TD target that is the **elementwise minimum over the
critic ensemble of a signed measure**: `pessimism_penalty=0.5` with `num_parallel=2` is
exactly `min(M_1, M_2)` applied to every entry of the `(B, B)` target matrix. That
minimum is a one-signed downward bias whose size is proportional to the ensemble's
disagreement, which is itself proportional to `|M|`. The backup is therefore not
`M <- gamma * M` but `M <- gamma * (1 + c) * M` on the side of the origin the bias points
away from, with `c = kappa * E|M_1 - M_2| / |M|`. It contracts only while
`gamma * (1 + c) < 1`, i.e. `gamma < 1/(1+c)`. Fitting `c` and one effective backup period
to the three antmaze discounts already on disk gives **`c = 0.0107`, `T_eff = 66 steps`,
`gamma_crit = 0.9895`** — which reproduces "bounded at 0.98, one decade per 280k steps at
0.99, one decade per 27k steps at 0.995" from two parameters. FB does not diverge on the
same loss family because **`fb_pessimism_penalty` is 0.0**: its target is the plain
ensemble mean, its backup gain is `gamma`, and `gamma < 1` always. Stronger orthonormality
does not touch this loop at all — `ortho` is a function of phi only, and the already-running
`ortho_mode=relative` arm measures the growth rate **unchanged** (33–36k steps/decade
against the unfixed 26.5–28.1k) while holding `orth_offdiag` at 64. It buys the basis, not
the boundedness.

---

## 1. The shipped measure loss, exactly as coded

Symbols per the `agents/psmflow.py` module docstring. Defaults are the paper-strict arm:
`psi_form=affine`, `policy_index=latent`, `index_agg=max`, `train_actor=false`,
`acting=gpi`, `ortho_mode=fixed`, `psi_bound=none`.

### 1.1 The two matrices

`agents/psmflow.py::measure_loss`, per batch of `B = 1024` transitions
`(s_i, a_i, s_i')` with cached latents `u_i` (`u_data`, clipped to `|u| <= u_clip = 3`):

```
u'_i           ~  N(0, I_{d_a}) clipped to +-3          (sample_step_inputs, fold_in(rng,106))
u^+_i          =  u'_i                                   (policy_index='latent': u_next <- u_index)
phi_next[j]    =  phi(s_j')             ONLINE phi,    params = phi_params      (grad flows)
tphi_next[j]   =  phibar(s_j')          TARGET phi,    params = self.target_phi (stop-grad tree)

M[p,i,j]       =  psi(s_i, u'_i, u_i)[p] . phi_next[j]                 ONLINE psi, ONLINE phi
Mboot[p,i,j]   =  psibar(s_i', u'_i, u^+_i)[p] . tphi_next[j]          TARGET psi, TARGET phi
```

`P = num_parallel = 2` indexes the critic ensemble. Note the input triple carefully: psi's
signature is `psi(obs, index, action)`, so the online side is `psi(s_i, index=u'_i,
action=u_i)` and the bootstrap side is `psibar(s'_i, index=u'_i, action=u'_i)` — **the same
policy index in both slots at `s'`**, which is the write-up's `psibar(s'_i, u', u')` and
Prop. `insample`'s hypothesis. `phi` is evaluated on `next_observations` on *both* sides,
so column `j` is `phi(s_j')` (online) against `phibar(s_j')` (target); the diagonal `j = i`
is the observed successor.

### 1.2 Ensemble reduction and pessimism (where the min is applied)

```python
M_mean, M_unc = targets_uncertainty(M_boot, P)          # utils/psm_common.py
target_M      = M_mean - c["pessimism_penalty"] * M_unc # kappa = 0.5
```

with

```python
unc = sum_{p,q} |preds[p] - preds[q]| / (P**2 - P)
```

At `P = 2` this is `unc = |M_1 - M_2|` and `mean - 0.5*unc = min(M_1, M_2)` **exactly**.
The reduction is **elementwise over all of `(B, B)`**: it is a min over ensemble members of
each entry of a *measure*, not a min of a scalar value. `target_M` then enters
`contrastive_loss` under `jax.lax.stop_gradient`, so neither psi nor phi receives gradient
through the target. `psi_bound='clip_target'` (off by default) would clip `target_M` to
`+-(S * z_dim)` here; `psi_bound='tanh'` (off by default) instead squashes the forward psi.

`num_parallel = 1` is **not a runnable ablation**: `unc = 0 / (1 - 1) = 0/0 = NaN`, verified
directly (`targets_uncertainty(x, 1) -> [nan, nan, nan]`). Any run with `num_parallel=1`
NaNs its target on step 1. Use `pessimism_penalty=0.0` at `P=2` instead — it makes the
target the plain ensemble mean, which is the semantics the `P=1` ablation was reaching for.

### 1.3 The contrastive term

`utils/psm_common.py::contrastive_loss`, with `off = 1 - I_B`, `off_sum = B^2 - B`:

```python
diff    = M - discount * target_M                              # (P, B, B)
offdiag = 0.5 * sum((diff * off)**2) / (B**2 - B)               # summed over P as well
diag    = -mean(diagonal(diff, axis1=1, axis2=2)) * P           # = -sum_p mean_i diff[p,i,i]
psm_loss = offdiag + diag
```

Both terms sum over the ensemble axis, so this is `sum_p [ 0.5*mean_{i!=j} diff_p^2 -
mean_i diff_p,ii ]`: an ordinary per-member loss, added over members.

**The diagonal term is linear in `M`, not quadratic, and it is the only term that is.**
Its derivative with respect to `M[p,i,i]` is `-1/B`, a *constant* upward push on the
diagonal that nothing in the loss opposes. Its derivative with respect to an off-diagonal
`M[p,i,j]` is `(M - gamma*target_M)[p,i,j] / (B^2 - B)`, which pulls that entry toward
`gamma * target_M`. This asymmetry is the whole structure of the objective and §3 turns on
it.

**Code vs write-up.** Eq. `loss-empirical` writes the positive term as
`-(2/B) sum_i psi_i^T phi(s_i')`, i.e. on `M` alone; the code uses `diag(M - gamma*T)`.
The two have the **same gradient** — `gamma * Tbar_ii` is stop-gradded and drops out of
`d/dM` — so this is not a defect. It matters only for reading `psm_loss`: the logged
diagonal carries a `-gamma * mean diag(target)` offset that the paper's expression does not.
The paper's `1/B^2` normaliser vs the code's `0.5/(B^2 - B)`, and the paper's factor 2 on
the positive term vs the code's `xP`, are a joint rescaling of the two terms against each
other by `~2P` — a real but benign difference from the boxed equation, and identical to
what FB does.

### 1.4 The geometry term and phi's projection

```python
cov     = phi_next @ phi_next.T                                # (B, B), ONLINE phi
ortho   = 0.5*sum((cov*off)**2)/(B**2-B)  +  (-mean(diagonal(cov)))
loss    = psm_loss + ortho_weight * ortho          # ortho_weight = ortho_coef = 1000
```

`PhiMap(norm=True)` ends in `psm_norm`, `sqrt(d) * x / ||x||`, so **every `phi(x)` has
`||phi||^2 = z_dim = 128` exactly**. Therefore `ortho_diag = -128` is a *constant* — the
`train.csv` column is pinned at `-128` in every run ever logged — and the entire
phi-gradient of the ortho term comes from `ortho_offdiag = 0.5*mean_{i!=j}(phi_i . phi_j)^2`,
whose baseline is ~64 for a random-ish basis and whose maximum is `0.5*128^2 = 8192` at
rank one.

**This is a measurement subtlety the collapse note's step-2 crossing inherits.** The
quantity `ortho_coef * |orth_loss| = 1000 * |64 - 128| = 6.4e4` that `psm_loss` crosses at
115–125k is two-thirds a constant. The *gradient*-relevant comparison is between
`ortho_coef * d(ortho_offdiag)/d(phi)` and `d(psm_loss)/d(phi)`, and the honest statement of
the mechanism is scale-free rather than threshold-shaped: `ortho`'s phi-gradient is
**bounded** (phi lives on a sphere and `ortho_offdiag <= 8192`), while `psm_loss`'s
phi-gradient grows linearly in `|M|`, which grows without bound. *Any* fixed `ortho_coef` is
eventually outgrown; raising it moves the crossing step by `log10(ratio) x 27k` steps and
nothing else. That prediction is already tested — see §5.2.

### 1.5 The affine head

`utils/psm_networks.py::AffinePsiMap`, `psi_form=affine`:

```
w(u')       = enc(u') / ||enc(u')||            in R^{d_w}, d_w = 128, UNIT sphere (norm_w)
A(s,u)      in R^{P x B x z_dim x d_w}         one trunk, wide head (1024 -> 128*128)
beta(s,u)   in R^{P x B x z_dim}               same trunk, second head
psi(s,u,u') = A(s,u)^T w(u') + beta(s,u)
```

`w_enc` is **shared across the ensemble** (a policy coordinate is a property of the policy,
not of a critic member); the ensemble disagrees only through `(A, beta)`. Neither `A` nor
`beta` sees `u'` — Assumption `affine`, enforced by not giving them the input.

### 1.6 Optimisers, targets, batch

| knob | value | note |
|---|---|---|
| `lr_phi` | **1.0e-5** | Adam, on phi |
| `lr_sf` | **1.0e-4** | Adam, on psi — **10x faster than the basis** |
| `tau` | 0.01 | one Polyak rate for **both** `target_phi` and `target_psi`, applied every step in `apply_update` |
| `batch_size` | 1024 | `off_sum = B^2 - B = 1047552` |
| `z_dim` | 128 | |
| `num_parallel` | 2 | |
| `ortho_coef` | 1000.0 | |
| `pessimism_penalty` | 0.5 | = exact min at `P=2` |
| `actor_pessimism_penalty` | 0.5 | **diagnostics only** under `train_actor=false` (`index_spread`) |
| `mix_ratio` | 0.5 | `task_w` = 50% `phi(next_obs[perm])`, 50% Gaussian, both on the sphere `sqrt(z_dim)` |
| `discount` | 0.98 yaml / 0.99 antmaze launcher / 0.995 the failing arm | |
| `u_clip` | 3.0 | |

**`mix_ratio` and `task_w` are inert for the measure loss in the default arm.** Under
`policy_index=latent`, psi's index slot carries `u'`, not `w`; `task_w` reaches the measure
loss nowhere. It is used by the actor branch (off), the `q_dist` branch (off) and
`index_spread` (a diagnostic). The measure loss is reward-free and task-free — as the
collapse note already noted when it retired reward scaling as a candidate.

Both losses are taken in **one** `value_and_grad(measure_loss, argnums=(2,3))` and both
targets are Polyak-updated at the *same* `tau` after the step.

---

## 2. The fixed point: what scale does `psi^T phi` want?

Write `m(s,u,u',x) = psi(s,u,u')^T phi(x)` and take the population version of §1.3 with
`kappa = 0` for the moment. The abstract objective (paper Eq. `loss-abstract`) is

```
L(m) = E_{x~rho}[ (m(s,u,u',x) - gamma*mbar(s',u',u',x))^2 ] - 2 * m(s,u,u',s')
```

so `dL/dm` is `2(m - gamma*mbar)` weighted by `rho` on the "negative" argument and `-2`
weighted by `delta_{s'}` on the "positive" one. Setting it to zero recovers the Bellman
equation for the successor measure, `m = delta_{s'} + gamma * mbar(s', .)`, whose solution
is `m*(s,u,u',.) = sum_t gamma^t P_t(. | s,u,u')` — the discounted occupancy. Under the
orthonormality the ortho term enforces, `E_rho[phi phi^T] = I`, the successor **feature** is
`psi_i = E[sum_t gamma^t phi_i(x_t)]` with `phi_i` of unit RMS, so

> **the stationary scale is `|psi_i| ~ 1/(1-gamma)`: 50 at 0.98, 100 at 0.99, 200 at 0.995.**

This is the same derivation `psi_bound_scale = 200` comes from, and it is the *only* place
`gamma` enters the fixed point: linearly in `1/(1-gamma)`, i.e. a factor 4 between 0.98 and
0.995. **It is nowhere near enough to explain the observed behaviour.** The measured `|Q|`
at 150k on antmaze runs 1.7e2 (0.98) / 6.4e2 (0.99) / 3.4e3–1.15e4 (0.995) and then 2.3e10
at 500k — a factor of `10^8`, not a factor of 4. A `1/(1-gamma)` fixed point does not
diverge; it moves.

At that fixed point the two terms of `psm_loss` sit at `offdiag -> 0` (the Bellman residual
is fitted away) and `diag = -P * mean_i(1-gamma) m*_ii = O(P)`. The observed healthy values
are `offdiag ~ 1e2-1e3` (an unfitted residual, expected — `m` is a `B x B` grid fitted by a
`z_dim = 128` bilinear form) and `diag ~ -100`, consistent. **`psm_diag` stays pinned near
`-100` in every run, healthy and diverged alike; `psm_offdiag` is what runs to 1e18.** The
divergence is entirely in the quadratic off-diagonal term, and the linear diagonal term is a
*stable source*, not the driver.

---

## 3. Linearised dynamics: is the semi-gradient update a contraction?

### 3.1 The loop without pessimism

Idealise psi as a free table on the batch and use `E[phi phi^T] = I`, `||phi_j||^2 = d`.
One gradient step of §1.3 gives, for the off-diagonal residual,

```
Delta M_ij  =  -eta * ( M_ij - gamma * Tbar_ij ) + eta_+ * [i = j],   eta ~ lr_sf * d / (B^2-B) * (...)
Tbar        <-  (1-tau) * Tbar + tau * M           (Polyak, both phi and psi at tau = 0.01)
```

Write `K` for the operator that maps `M(s, u', u)` to `M(s', u', u')` — the "shifted input"
map the bootstrap applies. Two-timescale linear iteration on `(M, Tbar)`; the slow
eigenvalue of the composite is

```
lambda_slow  ~=  1 - eta_eff * (1 - gamma * ||K||),        eta_eff = min-ish(eta, tau)
```

so the update is a contraction iff **`gamma * ||K|| < 1`**. With a network that generalises
across `(s -> s', u -> u')` — which is what makes bootstrapping work at all — `||K|| ~ 1`
and `gamma < 1` suffices. This is the FB/PSM story, and it is why FB does not diverge.

**Where the regulariser would enter, and does not.** The question as posed asks where
`gamma * (target magnitude) / (regulariser)` sits in the linearisation. The answer is that
it does not sit anywhere: `ortho` is a function of `phi_next` only. It appears in
`d(loss)/d(phi)` and is **identically zero in `d(loss)/d(psi)`**. There is no term in the
objective that penalises `||psi||`, `||A||` or `||beta||`. The loop gain above is therefore
unregularised, and `ortho_coef` cannot enter it at any value. What `ortho_coef` controls is a
*different* competition — between `d(psm)/d(phi)` and `d(ortho)/d(phi)` — which decides
whether `phi` stays a basis, not whether `M` stays finite.

### 3.2 The loop with the elementwise ensemble min

Now put `kappa` back. `target_M = mean_p M_p - kappa * unc`, `unc >= 0` **always**, so the
target is biased **downward regardless of the sign of `M`**. Model the disagreement as
proportional to the common scale, `unc ~ delta * |M|` with `delta` the relative ensemble
disagreement (this is what two networks with the same architecture, the same inputs and
different seeds do: they agree in shape and disagree by a roughly fixed *fraction* of their
output). Then the one-step map is

```
M  <-  gamma * ( M - kappa * delta * |M| )
    =  gamma * (1 - kappa*delta) * M      if M > 0     CONTRACTS FASTER than gamma
    =  gamma * (1 + kappa*delta) * M      if M < 0     EXPANDS
```

**The pessimism min makes the backup sign-selective.** On the positive side it is a stronger
contraction than `gamma`; on the negative side it is an expansion with gain
`gamma * (1 + c)`, `c := kappa * delta`. Divergence therefore requires two things, and both
are observed:

1. `M` must be negative on the entries that carry the mass. It is — §4.
2. `gamma * (1 + c) > 1`, i.e. **`gamma > gamma_crit = 1/(1 + c)`**.

Growth is exponential in the number of *backups*, not of gradient steps, so with `T_eff` the
effective backup period (set by whichever of `tau` and the psi step size is slower),

```
   d log10(psm_loss) / d step   =   2 * log10( gamma * (1+c) ) / T_eff'
   steps per decade             =   T_eff / log10( gamma * (1+c) )
```

(the factor 2 is absorbed into `T_eff`, since `psm_loss ~ psm_offdiag ~ M^2`).

**Fit, on data already on disk.** `scripts/audit/measure_loss_growth.py` least-squares-fits
`log10(psm_loss)` against step over 50k–500k for the nine `affine_strict` antmaze runs at
`gamma in {0.98, 0.99, 0.995}` (same env, same preimages, same everything but `discount`),
then solves the two-parameter model above. Result
(`$PSM_DATA/logs/audit_measure_loss/growth_existing.json`):

> **`c = 0.01065`, hence `delta = c/kappa = 0.0213`; `T_eff = 66.2` steps;
> `gamma_crit = 0.98946`.**

| arm | measured steps/decade | model | fit `r^2` |
|---|---|---|---|
| antmaze `gamma=0.98`, 3 seeds | 3.35e6 / 3.61e6 / 4.11e6 (flat, `r^2` 0.18–0.19) | **BOUNDED** (`0.98 x 1.0107 = 0.9905 < 1`) | — |
| antmaze `gamma=0.99`, 3 seeds | 3.39e5 / 1.38e5 / 2.75e5 (`r^2` 0.87–0.90) | 2.79e5 | ✓ |
| antmaze `gamma=0.995`, 3 seeds | 2.65e4 / 2.73e4 / 2.81e4 (`r^2` 0.96–0.97) | 2.73e4 | ✓ |

Two free parameters reproduce a 150x spread in growth rate across three discounts, plus one
sign constraint (0.98 must be bounded, and is). `delta = 0.021` is the relative ensemble
disagreement the model needs; the `diag_gpi_selection` probes measured `unc/|Q|` at
0.016–0.019 on cube and 0.005–0.018 on antmaze — the same number, at the boundary of the
same band. `T_eff = 66` steps is between `1/tau = 100` and the psi step scale, as a
two-timescale loop should be.

Three further consequences the model gets right for free:

* **Cube at `gamma=0.98` is bounded and antmaze at 0.98 is bounded** — `gamma_crit = 0.9895`
  sits between 0.98 and 0.99, which is precisely where the observed boundary is.
* **The 0.99 seeds scatter and the 0.995 seeds do not.** At `gamma=0.995` the three seeds
  agree to three significant figures (2.65/2.73/2.81e4); at `gamma=0.99` they spread 2.5x
  (1.38–3.39e5). That is the signature of sitting just above a threshold: the rate is
  `log10(gamma(1+c))`, whose *relative* sensitivity to a per-seed wobble in `delta` blows up
  as `gamma(1+c) -> 1`. A "the discount is just too high" account has no reason to predict
  the seed spread to be discount-dependent.
* **`gamma=0.99` seed 1 is the one that collapses.** It has the fastest rate of the three
  (1.38e5), crosses at 400k, and its 500k eval is 0.004. Within-arm, faster rate = collapse.

### 3.3 Does the affine head remove a scale anchor?

**No — and neither does FB's free psi. Neither has one.**

* FB's `F` is a plain ensembled MLP with a linear output head: unbounded.
* psmflow's free `PsiMap` is the same object: unbounded.
* psmflow's `AffinePsiMap` emits `A(s,u)^T w(u') + beta(s,u)` with `||w(u')|| = 1`. The
  `norm_w` constraint pins the `(cA, w/c)` **degeneracy**, it does not bound anything:
  `sup_{u'} ||A^T w(u') + beta||` is still free, carried entirely by `A` and `beta`.

So the affine form is **scale-neutral** and cannot be the mechanism. The claim in the
09-07 notes that "phi is on the sphere and `w(u')` is unit-norm, so `A` and `beta` are the
only free magnitude in the model" is true, but it is equally true of FB (`B` on the sphere,
`z` on the sphere, `F` free), and FB does not diverge. Freedom of scale is a **necessary**
condition for a runaway, not the thing that starts one.

What the affine head plausibly *does* change is `delta`, and hence `gamma_crit`. Before it,
`psi_q_index_spread_rel` sat at ~1%: psi barely depended on its policy index, so the two
ensemble members had almost nothing to disagree about in the index direction. After it, the
index spread is 0.26–1.01. Two members that now genuinely disagree about a live degree of
freedom have a larger relative disagreement, hence a larger `c = kappa*delta`, hence a
*lower* `gamma_crit`. On that reading the affine head did not cause the divergence but may
have **lowered the discount at which it starts** — which is exactly the shape of "the affine
form fixed policy-index blindness and something else got worse". It is directly testable and
is arm **A9** below.

---

## 4. The sign: why is `Q` large and negative?

Three facts, from the code and from `train.csv`:

1. **The diagonal term's sign convention is correct.** `d(psm_loss)/d(M_ii) = -1/B < 0`
   pushes `M_ii` **up**, which is what a positive term rewarding mass at the observed
   successor should do. Empirically `psm_diag ~ -100`, i.e. `mean_i diag(M - gamma*T) ~ +50`
   — the diagonal residual is **positive** in every run, healthy and collapsed. The
   diagonal is not the source of the negativity.
2. **The off-diagonal has no sign anchor at all.** Its only stationarity condition is
   `M_ij = gamma * Tbar_ij`, a homogeneous recursion whose fixed point is 0 up to whatever
   the network leaks across from the diagonal. Its sign is set by whatever bias acts on it.
3. **The pessimism min is exactly such a bias, and it is one-signed.** `target_M <= mean_p M_p`
   always. In the regime where the disagreement is roughly a *level* rather than a fraction
   (`unc ~ sigma`), the off-diagonal steady state of `M <- gamma*(M - kappa*sigma) + S` is

   ```
   M*  =  ( S - gamma*kappa*sigma ) / (1 - gamma)
   ```

   with `S > 0` the leak from the diagonal term. Whenever `gamma*kappa*sigma > S` this is
   **negative, with magnitude `~ 1/(1-gamma)`**. That is the observed steady state:
   `|Q| ~ 1.7e2` at `gamma=0.98`, `6.4e2` at 0.99 — a factor 3.8 for a predicted factor 2,
   the residue being that `sigma` itself grows with the level.

So the sign is the pessimism min's, not the diagonal term's. And the two regimes are the
same term seen at two scales: while `unc` is a fixed level the bias produces a **negative
offset** `~ -gamma*kappa*sigma/(1-gamma)`; once `unc` tracks `|M|` proportionally the same
bias becomes the **multiplicative gain** of §3.2, and because `M` is by then already
negative, the multiplicative branch is the expanding one. **The min over an ensemble of a
quantity that is already negative drives the target more negative every backup, and the
measure's total mass runs away downward.** That is the runaway the question names, stated
precisely.

There is a structural reading of this that is worth writing down. `M` is a **measure**: the
successor measure `m^{u->u'}` is non-negative and has total mass `1/(1-gamma)`. Pessimism is
a device for *values*, where "take the smaller estimate" is conservative. Applying it to a
measure means taking the elementwise lower envelope of two estimated densities, which is not
a density: it has strictly less mass than either member, it is not achievable by any single
member, and iterating the Bellman operator on it compounds the mass deficit geometrically.
**`pessimism_penalty` belongs on `Q = psi^T w`, not on `M = psi^T phi`.** FB places it
exactly nowhere (`fb_pessimism_penalty = 0.0`); PSM ships 0.0 as well.

---

## 5. Line-by-line against FB

`archive/agents/fb.py::_fb_loss_fn` and `archive/configs/agent/fb.yaml`, against
`agents/psmflow.py::measure_loss` and `configs/agent/psmflow.yaml`. FB is the JAX port of
the PyTorch factored-fb reference, reuses `utils/psm_common.py`'s helpers verbatim ("the
same math"), reaches cube 0.721, and has no divergence on record.

### 5.1 The table

| # | item | FB | psmflow (affine strict) | same? |
|---|---|---|---|---|
| 1 | **target ensemble reduction** | `mean - fb_pessimism_penalty * unc`, **`fb_pessimism_penalty = 0.0`** -> **plain mean** | `mean - 0.5 * unc` at `P=2` -> **exact elementwise MIN** | **DIFFERENT** |
| 2 | off-diagonal term | `0.5*sum((diff*off)^2)/off_sum`, `diff = M - gamma*targetM` | identical (shared `contrastive_loss`) | same |
| 3 | diagonal term | `-mean(diag(diff)) * P` | identical | same |
| 4 | ortho expression | `0.5*sum((cov*off)^2)/off_sum - mean(diag(cov))` on `B(next_obs)` | identical, on `phi(next_obs)` | same |
| 5 | **ortho coefficient** | **1.0** | **1000.0** | DIFFERENT |
| 6 | basis normalisation | `BackwardMap = PhiMap`, `norm=true` -> sphere `sqrt(50)` | `PhiMap(norm=True)` -> sphere `sqrt(128)` | same in kind |
| 7 | **basis / head learning rates** | `lr_f = lr_b = 1e-4` (**equal**) | `lr_sf = 1e-4`, `lr_phi = 1e-5` (**basis 10x slower**) | DIFFERENT |
| 8 | target rates | `f_target_tau = b_target_tau = 0.005` | `tau = 0.01` for both (2x faster) | DIFFERENT (mild) |
| 9 | bootstrap action at `s'` | actor's `a' = pi(s', z)` (a moving actor, tanh/clip-bounded) | `u^+ = u' ~ p0`, decoded by a **frozen** flow | DIFFERENT (favours psmflow) |
| 10 | extra input anchor | `left_encoder(obs)`, itself sphere-normalised, feeds `F` | none; psi reads raw `obs` | DIFFERENT |
| 11 | measure head | free `F(le(s), z, a)` | affine `A(s,u)^T w(u') + beta(s,u)`, `||w|| = 1` | DIFFERENT |
| 12 | policy index | `z` on the sphere `sqrt(50)`, 50% from `B(s'_perm)` | `u' ~ N(0,I)` clipped to `+-3` | DIFFERENT in kind |
| 13 | `z_dim` / batch / discount | 50 / 256 / 0.99 | 128 / 1024 / 0.98–0.995 | DIFFERENT |
| 14 | bound / clip / weight decay on the head | none (`weight_decay = 0.0`) | none (defaults) | same |
| 15 | actor pessimism | `actor_pessimism_penalty = 0.0` | 0.5 (diagnostics only when `train_actor=false`) | different, inert here |

### 5.2 Ranked by how plausibly each causes the divergence

1. **(#1) Pessimism on the TD target: `0.5` (exact min) vs FB's `0.0`.** Direct, structural
   and quantitative. It is the only term in either objective that biases the backup one way
   regardless of sign, and it is the only free parameter needed to turn `gamma`-contraction
   into `gamma(1+c)`-expansion. A two-parameter fit of the resulting model reproduces the
   measured rate at three discounts (§3.2). FB's zero here is a sufficient explanation for
   why FB does not diverge on the same loss, the same helpers and the same data.
   **Pre-registered as the cause.**
2. **(#13/#5, jointly) `gamma` relative to `gamma_crit`, and `ortho_coef` as its
   consequence.** `gamma` is not itself a defect — it is the axis along which the defect in
   (1) becomes visible, and `gamma_crit = 1/(1+c)` is a *function* of (1). FB's 0.99 is
   under `gamma_crit` **because** its `c` is 0. `ortho_coef = 1000` vs 1.0 is downstream: FB
   needs no large geometry weight because nothing outgrows it.
3. **(#7) `lr_phi = 1e-5` against `lr_sf = 1e-4`.** A 10x asymmetry FB does not have. It
   cannot change the loop gain (the ortho and psm gradients on phi are both scaled by
   `lr_phi`, so their ratio is untouched), but it changes `T_eff` and it means the basis
   responds to a diverging psi ten times more slowly than psi diverges. Plausibly sets the
   20–25k step lag between the `psm_loss` crossing and `orth_offdiag`'s departure.
   Rate-affecting, not cause.
4. **(#11) The affine head.** Scale-neutral (§3.3), so not a cause. But it raised
   `psi_q_index_spread_rel` from ~0.01 to 0.26–1.01, which plausibly raised `delta` and
   therefore *lowered* `gamma_crit`. This is the "second lever besides the affine form"
   question read the other way: the affine form is not the lever, but it may have moved the
   threshold the lever is compared against. Tested by arm A9.
5. **(#8) `tau = 0.01` vs FB's `0.005`.** Sets `T_eff`, i.e. the wall-clock rate of an
   already-divergent loop. Halving it halves the rate and fixes nothing. Tested by A8.
6. **(#9/#10/#12) Bootstrap action, `left_encoder`, index distribution.** Differences of
   substance for what the measure *means*, but none of them is one-signed and none enters
   the loop gain. #9 in fact favours psmflow: its bootstrap family is frozen, FB's chases a
   moving actor, and the write-up makes that a selling point.
7. **(#3 read against the paper) The `x P` and `0.5/(B^2-B)` normalisers.** They rescale the
   diagonal against the off-diagonal by `~2P` relative to Eq. `loss-empirical`. FB does the
   identical thing, so it cannot be what separates them. There is **no config key** that
   controls this weighting — `contrastive_loss` hardcodes it — so it cannot be ablated
   without touching `utils/psm_common.py`, which this note does not.

### 5.3 Third-party confirmation: `Factored-FB`

`/mnt/home/amohan/git/Austin/Factored-FB/impls/critics/fb.py` (JAX/Flax, a descendant of
this repo's `archive/agents/fb.py`, so not an independent implementation — but an
independently *configured* one) inlines the identical loss and reports:

* `target_M = tmean - fb_pessimism_penalty * tunc`, **default `fb_pessimism_penalty: 0.0`**,
  with a config comment stating in so many words that "0.5 at P=2 reproduces the reference's
  min over heads". So zero-pessimism is the *documented* FB default and 0.5 is the
  documented deviation, in a repo that reaches cube 0.721.
* Its parity arm `impls/configs/critic/fb_tdj.yaml` runs **`ortho_coef: 1000.0` with
  `fb_pessimism_penalty: 0.0`** — i.e. psmflow's ortho weight and FB's pessimism, together,
  with no divergence on record. That combination separates the two knobs cleanly: it is the
  cell that says `ortho_coef = 1000` is not what makes psmflow diverge, and `pessimism = 0`
  is not something the large ortho weight compensates for.
* Same `0.5/(B^2-B)` off-diagonal, same `-mean(diag(M - gamma*T)) * P` diagonal, same
  `lr_f = lr_b = 1e-4`, same `f_tau = b_tau = 0.005`, **no clip, no weight decay, no bound
  on `F` or on `M`** anywhere.
* Independently notes the §1.4 point: with `backward.norm: true` the sphere projection makes
  `cov[i,i] == z_dim` exactly, so `orth_diag` is "a constant with zero gradient; only
  `orth_off` actually regularises".

Nothing in that repo contradicts the table above, and the `fb_tdj` cell strengthens ranking
item 1 against item 2.

---

## 6. Hypothesis and pre-registered predictions

**H-PESS.** The measure's TD backup is non-contractive because `pessimism_penalty = 0.5`
applies an exact elementwise ensemble minimum to a signed measure, biasing every bootstrap
downward by `kappa * unc ~ c * |M|` with `c = kappa * delta`, `delta ~ 0.021`. The backup
gain on the negative branch is `gamma * (1 + c)`; the growth rate is
`log10(gamma(1+c)) / T_eff` with `T_eff ~ 66` steps; the stability boundary is
`gamma_crit = 1/(1+c) = 0.9895`. Basis collapse is a **downstream** consequence, gated by
whether `ortho_coef * d(ortho)/d(phi)` is still comparable to `d(psm)/d(phi)`.

Scaling, all stated **before** the arms below are launched:

| lever | predicted effect on the growth rate | why |
|---|---|---|
| `gamma` | `spd = T_eff / log10(gamma(1+c))`; **bounded below `gamma_crit = 0.9895`** | the gain itself |
| `pessimism_penalty` `kappa` | `c = kappa*delta`; `kappa=0` -> **bounded at any `gamma<1`**; `kappa` halved -> 19x slower; `kappa` doubled -> 2.9x faster | the gain itself |
| `tau` | `T_eff ∝ 1/tau` when the Polyak rate is the slower timescale -> rate `∝ tau`; **defers, does not fix** | loop period |
| `lr_sf` | `T_eff` grows as the psi step shrinks -> rate falls, by **less** than 10x if `tau` is already the bottleneck; **defers, does not fix** | loop period |
| `ortho_coef` | **NO EFFECT ON THE RATE.** `ortho` has zero psi-gradient. It moves the `orth_offdiag` departure step later by `log10(ratio) x spd` and nothing else | not in the loop |
| `psi_bound` (`tanh`) | bounds `|M|` by construction, so bounds `psm_loss`; the loop still has gain > 1 but saturates | a ceiling, not a fix to the gain |

### 6.1 The `ortho_coef` prediction is already answered by data on disk

The `ortho_mode=relative` arm (`affine_strict_antmaze_g995_orel`, 3 seeds, in flight, at
400k) is `ortho_coef` taken to its limit — a weight that tracks `|psm_loss|` and therefore
can never be outgrown. Measured against the unfixed `g995` arm:

| | unfixed `g995` (3 sd) | `orel` (3 sd) | `pbound` = `psi_bound=tanh` (3 sd) |
|---|---|---|---|
| steps/decade, 50k–max | **2.65e4 / 2.73e4 / 2.81e4** | **3.47e4 / 3.30e4 / 3.56e4** | 3.5e6 / 3.9e6 / 6.0e6 (**flat**) |
| `psm_loss` @150k | 5.8e5 / 2.7e6 / 5.6e6 | 1.8e5 / 6.1e5 / 5.8e6 | **1.5e3 / 9.5e2 / 9.3e2** |
| `psm_loss` @400k | 8e15–1e16 | **1.1e13 / 1.2e13 / 1.1e12** | ~1e3 |
| `orth_offdiag` @150k | **189 / 442 / 423** (departed) | **64.3 / 63.9 / 65.6** (held) | 64.4 / 63.8 / 64.4 (held) |
| `orth_offdiag` @400k | **8192** (rank one) | **64.1 / 64.4 / 64.6** (held) | 63.9 / 64.3 / 64.0 (held) |
| `|Q|` @400k | ~1e9 | **3.3e7 / 5.3e7 / 1.7e7** | 1.3e3 / 1.0e3 / 1.1e3 |
| `psi_q_index_spread_rel` @150k | 0.37 / 0.38 / 0.40 | 0.30 / 0.32 / 0.45 | **0.17 / 0.17 / 0.26** |

> **Stronger orthonormality changes the growth rate by 25% (in the *slower* direction, i.e.
> within seed scatter) and holds `orth_offdiag` at its 64 baseline for the whole run.
> `psm_loss` still reaches 1.1e13 and `|Q|` still reaches 5e7.** The user's proposed
> `ortho_coef = 1e4-1e5` is a *weaker* version of this: a fixed weight `X` moves the crossing
> to `log10(X/1000) x 27k` steps later — 27k steps later at 1e4, 54k at 1e5 — and then the
> same thing happens. **Stronger orthonormality delays and does not fix.** Arms A5/A6 below
> put a number on the delay; the rate prediction is that it is unchanged.

---

## 7. Ablations (pre-registered; launched 2026-09-08)

All on `antmaze-medium-navigate-singletask-v0` at `agent.discount=0.995` — the
fastest-failing setting, where the phenomenon is fully visible by 150k. 150k steps, save +
in-loop eval at 50k/100k/150k, `eval_episodes=10` (in-loop only; nothing here is reported as
a performance number), `log_interval=5000`, one seed per job unless noted, `--time=03:00:00`.
Launcher: `scripts/audit/launch_measure_loss_ablations.sh`.

**Comparator, not re-run:** the unfixed `affine_strict_antmaze_g995` seeds 0/1/2, identical
in every key, whose 50k–150k numbers are quoted in §6.1.

**Every arm uses a config key that exists today.** The one requested lever that does not:
`num_parallel=1` **cannot be run** (§1.2, `0/0 -> NaN`), and there is **no key at all** for
the diagonal-vs-off-diagonal weighting — `contrastive_loss` hardcodes `x P` and
`0.5/(B^2-B)`. Ablating that would require a copy of the agent under `scripts/audit/`, which
this note deliberately does not create.

| arm | group | `EXTRA` | seeds |
|---|---|---|---|
| **A1** no target pessimism | `mla_g995_pess0` | `agent.pessimism_penalty=0.0` | 0, 1 |
| **A2** half pessimism | `mla_g995_pess025` | `agent.pessimism_penalty=0.25` | 0 |
| **A3** double pessimism | `mla_g995_pess10` | `agent.pessimism_penalty=1.0` | 0 |
| **A5** ortho 1e4 | `mla_g995_oc1e4` | `agent.ortho_coef=10000.0` | 0 |
| **A6** ortho 1e5 | `mla_g995_oc1e5` | `agent.ortho_coef=100000.0` | 0 |
| **A7** slow psi | `mla_g995_lrsf1e5` | `agent.lr_sf=1.0e-5` | 0 |
| **A8** slow Polyak | `mla_g995_tau1e3` | `agent.tau=0.001` | 0 |
| **A9** free psi | `mla_g995_freepsi` | `agent.psi_form=free` | 0 |

(A4 would have been `num_parallel=1`; it is unrunnable and is replaced by A2/A3, which turn
the on/off test into a **dose-response**, a much stronger test of the same claim.)

### 7.1 Per-arm predictions

Model constants from §3.2: `delta = 0.02131`, `T_eff = 66.2`, `gamma = 0.995`. Baseline
(`kappa=0.5`) reads 2.65–2.81e4 steps/decade, `psm_loss` 5.8e5–5.6e6 @150k, `|Q|`
3.4e3–1.15e4 @150k, `orth_offdiag` 189–442 @150k (departs at 140–150k),
`psi_q_index_spread_rel` 0.37–0.40 @150k.

| arm | predicted steps/decade | `psm_loss` @150k | `|Q|` @150k | `orth_offdiag` @150k | `psi_q_index_spread_rel` @150k |
|---|---|---|---|---|---|
| **A1** `kappa=0` | **BOUNDED** (`0.995 < 1`); report `> 1e6` | **1e3–5e3** | **5e2–2e3** | **64 (never departs)** | **0.25–0.50** (unlike `pbound`) |
| **A2** `kappa=0.25` | **5.1e5** (band 2e5–1.5e6) | **3e3–8e3** | 1e3–3e3 | 64 (never departs) | 0.30–0.45 |
| **A3** `kappa=1.0` | **9.5e3** (band 6e3–1.5e4) | **>= 1e11** | **>= 1e6** | **>= 2000, departs by 90–110k** | rises above 0.5 |
| **A5** `oc = 1e4` | **UNCHANGED, 2.2e4–3.5e4** | 5e5–6e6 (baseline) | 3e3–1.2e4 (baseline) | **64, departure pushed past 150k** | 0.30–0.45 |
| **A6** `oc = 1e5` | **UNCHANGED, 2.2e4–3.5e4** | 5e5–6e6 (baseline) | 3e3–1.2e4 (baseline) | **64, departure pushed past 150k** | 0.30–0.45 |
| **A7** `lr_sf = 1e-5` | 5e4–3e5 (**less than 10x slower**, `tau` is the other timescale) | 1e4–1e5 | 1e3–3e3 | 64 | 0.20–0.40 |
| **A8** `tau = 0.001` | **2.2e5–3.0e5 (~10x slower)** | 4e3–2e4 | 8e2–2e3 | 64 | 0.20–0.40 |
| **A9** `psi_form=free` | 2e4–2e5, i.e. **slower than baseline if the affine head raised `delta`**, unchanged if it did not | — | — | — | **~0.01** (the pre-affine band, by construction) |

### 7.2 Named failure modes — what each arm looks like when *this account* is wrong

* **A1 diverges anyway.** If `psm_loss` still grows at 2.5–3.5e4 steps/decade with
  `kappa = 0`, the ensemble min is **not** the mechanism, `gamma * ||K|| > 1` on its own
  (the network's generalisation across `s -> s'` is expansive), and the answer is the
  discount or the bootstrap map — not the target reduction. **This single cell decides the
  whole note.**
* **A1 bounds `psm_loss` but the index signal dies.** If `psi_q_index_spread_rel` falls
  below 0.05 and `|Q|` collapses toward 0, then target pessimism was load-bearing for
  something *other* than stability — most likely for keeping the ensemble from agreeing on a
  degenerate psi — and removing it trades divergence for flatness, the same trade
  `psi_bound=tanh` appears to be making (its index spread reads 0.17 against the baseline's
  0.38). The remedy would then be `pessimism_penalty` on `Q` only, which no key currently
  expresses.
* **The dose-response is not monotone.** If A2 and A3 do not bracket the baseline in rate
  (A3 fastest, baseline, A2, A1 slowest), the mechanism is not a gain of the form
  `gamma(1+kappa*delta)` and the quantitative fit in §3.2 is a coincidence of two free
  parameters on three points.
* **A5/A6 change the rate.** If raising `ortho_coef` slows `psm_loss` growth by more than
  seed scatter (say `> 1.5x`), then `ortho` does reach psi — through phi, whose collapse
  feeds back into the measure that psi is fitting — and §3.1's "no regulariser in the loop"
  is wrong as a *dynamical* (as opposed to algebraic) statement. This is the most likely of
  my predictions to be wrong.
* **A6 destabilises phi numerically.** `ortho_coef = 1e5` against `lr_phi = 1e-5` is an
  effective unit gain on the ortho gradient. If `orth_offdiag` oscillates or NaNs, the arm
  is uninformative rather than negative; report it as such.
* **A7 and A8 give the same factor.** Predicted different (A8 ~10x, A7 less). If both give
  10x, `T_eff` is not a two-timescale minimum and the loop-period model is too crude — the
  rate scalings then say nothing about which knob is the bottleneck, though the `gamma` and
  `kappa` scalings would stand.
* **A9 diverges at the same rate as the affine baseline.** Then the affine head did not move
  `gamma_crit`, §5.2 item 4 comes off the list, and the affine form is fully exonerated —
  the cleanest possible outcome for the head that bought cube 0.083 -> 0.415.
* **Anything NaNs.** Predicted false everywhere except possibly A3 (`psm_loss >= 1e11` at
  150k implies `|M| ~ 5e5`, well inside fp32) and A6 (see above). A NaN in A1 or A2 would be
  a wiring error, not a result.

### 7.2b Launch record (2026-09-07, `discount=0.995` throughout)

| job | arm | group | seed | `EXTRA` |
|---|---|---|---|---|
| 2492186 | smoke (400 steps) | `mla_smoke` | 0 | `pessimism_penalty=0.0 ortho_coef=100000.0 lr_sf=1.0e-5 tau=0.001` |
| 2492189 | A1 | `mla_g995_pess0` | 0 | `agent.pessimism_penalty=0.0` |
| 2492190 | A1 | `mla_g995_pess0` | 1 | `agent.pessimism_penalty=0.0` |
| 2492191 | A2 | `mla_g995_pess025` | 0 | `agent.pessimism_penalty=0.25` |
| 2492192 | A3 | `mla_g995_pess10` | 0 | `agent.pessimism_penalty=1.0` |
| 2492193 | A5 | `mla_g995_oc1e4` | 0 | `agent.ortho_coef=10000.0` |
| 2492194 | A6 | `mla_g995_oc1e5` | 0 | `agent.ortho_coef=100000.0` |
| 2492195 | A7 | `mla_g995_lrsf1e5` | 0 | `agent.lr_sf=1.0e-5` |
| 2492196 | A8 | `mla_g995_tau1e3` | 0 | `agent.tau=0.001` |
| 2492197 | A9 | `mla_g995_freepsi` | 0 | `agent.psi_form=free` |

**Deviation from the working discipline, recorded deliberately.** CLAUDE.md requires a
smoke *before* launch. The smoke (2492186) was submitted first, but SLURM's estimated start
for it was ~22 h out (36 pending jobs ahead, 18 running, all belonging to other agents), so
waiting for it to complete would have cost a day of queue position for the nine and taught
nothing the dry run had not: every override is an existing scalar config key with no new
code path, and the one genuinely novel value (`pessimism_penalty=0.0`) is a documented FB
default. The nine were therefore submitted immediately behind the smoke, which runs first
by submission order; a persistent watcher reports the smoke's terminal state and every
job's start/finish, and **the nine are to be `scancel`led if the smoke does not exit
cleanly.** Each run's own `flags.json` is to be re-read after start to confirm the values
landed, per the same discipline.

### 7.3 Readouts

Per arm, from `train.csv` only, via `scripts/audit/measure_loss_growth.py`, persisted to
`$PSM_DATA/logs/audit_measure_loss/`:

* `steps_per_decade` — least-squares slope of `log10(psm_loss)` over **50k–150k**, with its
  `r^2` (a low `r^2` at a huge `steps_per_decade` is how "bounded" reads).
* `orth_depart_step` — first step past 90k at which `orth_offdiag` exceeds 2x its own
  10–90k baseline; `null` = never.
* `q_level_hi` = `psi_q_spread / psi_q_spread_rel` at 150k (`|Q|`), and
  `q_growth_decades` over the window.
* `psi_q_index_spread_rel` at 150k — the signal the affine head bought and the one both
  candidate fixes risk.
* `psm_diag_hi` / `psm_offdiag_hi` — to confirm the divergence stays off-diagonal.

Figure `docs/figures/2026-09-08-measure-loss-audit.png`: `psm_loss` against step, one panel
per lever, the unfixed `g995` seeds as a grey band in every panel; plus a rate-vs-lever panel
with the §6/§7.1 predictions drawn as points.

---

## 8. Verdict

To be filled in when the arms land. Nothing above this line may change once the first job is
submitted. The four questions it must answer:

1. Is there a second lever beyond the affine form, and does it limit cube too?
2. Does stronger orthonormality fix the divergence or only delay it (rate vs coefficient)?
3. What is the structurally right fix?
4. What changes in the default config?

The pre-registered answers, so that the verdict cannot be shopped: **(1) yes — the exact-min
ensemble target, and yes it limits cube, because cube's `|Q|` is already negative at
-3.8e3..-6.4e3 with `gamma_crit = 0.9895` only a hair above cube's 0.98, so cube is
subcritical rather than stable and its `psm_loss` still grows secularly 1.6e3 -> 1e4-3e4;
(2) delay only, with the rate unchanged — already measured by the `orel` arm; (3)
`pessimism_penalty = 0.0` on the measure, with any conservatism moved to `Q = psi^T w`,
i.e. FB's exact arrangement, and `psi_bound`/`ortho_mode=relative` kept as belt-and-braces
rather than as the fix; (4) `pessimism_penalty: 0.0` in `configs/agent/psmflow.yaml`, with
`ortho_mode: relative` as a cheap second line of defence and `discount` left where the
launcher puts it.**

---

### 8.1 Verdict, filled 2026-09-08

All nine arms (2492186-2492197) completed. Growth refitted on a **common 10k-150k window**
so the arms (150k steps) and the pre-existing baselines (500k) are comparable; the §7
numbers were fit over 50k-500k and must not be read against these.
`$PSM_DATA/logs/audit_measure_loss/growth_arms.json`, `growth_baseline_10_150.json`.

antmaze `medium-navigate`, gamma=0.995. Higher steps/decade = slower divergence.

| arm | kappa_measure | steps/decade | log10 |Q| growth |
|---|---|---|---|
| `mla_g995_pess10` | 1.0 | 1.5e4 | 4.15 |
| `affine_strict_antmaze_g995` (baseline) | 0.5 | 5.18 / 4.25 / 3.84e4 | 1.25-1.78 |
| `mla_g995_pess025` | 0.25 | 8.0e4 | 0.73 |
| `mla_g995_pess0` | 0.0 | 1.08e5 / 1.95e5 | 0.45 / 0.33 |

**Q1 — yes, and the lever is the exact-min ensemble target.** The rate is monotone in
`pessimism_penalty` across four dose levels spanning 13x, against a baseline seed scatter of
35%. H-PESS as pre-registered.

**Q2 — delay only, confirmed.** `orel` measured 5.77 / 4.84 / 4.25e4 against the baseline's
5.18 / 4.25 / 3.84e4: within scatter, seed for seed. `oc1e4` 4.53e4 and `oc1e5` 5.05e4 are
the same story with a fixed coefficient. Orthonormality has no psi gradient and cannot fix a
psi-scale instability.

**Q3 and Q4 — the pre-registered structural fix is REFUTED on cube, and no default changes.**
`pessimism_penalty=0.0` with `actor_pessimism_penalty=0.5` (FB's arrangement; this is
already expressible, see 8.2) was run on both envs at 500 episodes, 3 seeds,
pooled 300k-500k:

| env | kappa_measure=0 | kappa_measure=0.5 |
|---|---|---|
| cube-single-play, gamma=0.98 | **0.136** | 0.415 +/- 0.083 |
| antmaze-medium-navigate, gamma=0.99 | **0.334** | 0.252 +/- 0.100 |

Cube loses two thirds. Antmaze's gain is inside overlapping CIs. Both are strongly bimodal
across seeds -- cube {0.011, 0.369, 0.029}, antmaze {0.280, 0.139, 0.584} -- so the pooled
means are three-sample statistics on a bimodal population, not estimates of a common mean.
`configs/agent/psmflow.yaml` therefore keeps `pessimism_penalty: 0.5`.

Two predictions in §6 did not survive. `tau=1e-3` measured 1.01e5 steps/decade, 2.3x slower
than baseline, against a pre-registration of "fixes nothing"; `freepsi` measured 7.4e4, 1.7x
slower, against "the affine head is not the scale mechanism". Both are single-seed against
35% baseline scatter -- leads for a seeded arm, not results.

### 8.2 Two corrections to the note above this line

**The "no config key expresses pessimism on Q only" gap in §5.2 is stale.**
`pessimism_penalty` (measure TD target) and `actor_pessimism_penalty` (acting and the actor's
Q) have been separate keys throughout. The arrangement the note asks for is
`agent.pessimism_penalty=0.0 agent.actor_pessimism_penalty=0.5`, which is exactly what
`scripts/slurm/launch_measure_pess0.sh` ran on 2026-09-07. No code was needed.

**Growth rate is not a proxy for policy health, and `psi_bound` is the counterexample.**
`pbound` has by far the best growth number measured anywhere in this note -- 4.86e5 / 5.70e5
/ 4.84e5 steps/decade, 11x the baseline, with |Q| growth held to 0.73-0.75 decades -- and
the worst policy. Its early (50k-250k) success is 0.124 / 0.096 / 0.256 against the
baseline's 0.288 / 0.292 / 0.344, and only one seed of three keeps a live policy late
(0.320, against 0.020 / 0.004). `both` is worse still: early 0.208 / 0.072 / 0.296, late
0.004 / 0.000 / 0.080. Bounding the measure buys stability by flattening the very
differences acting has to rank. **`psi_bound` should be read as settled negative** -- it is
the arm whose growth number most invites revival and whose evals most clearly forbid it.

### 8.3 What the ensemble spread actually measures

`tools/diag_ensemble_disagreement.py`, 8 checkpoints spanning both envs and all three
discounts (`$PSM_DATA/logs/diag_disagreement/`). Relative disagreement on prior draws vs on
in-support latents, ratio prior/in-support:

| checkpoint | prior | in-support | ratio |
|---|---|---|---|
| cube g98 sd0 100k / 350k | 0.0364 / 0.0345 | 0.0367 / 0.0347 | 0.99 / 0.99 |
| antmaze g98 sd0 100k / 500k | 0.0472 / 0.0457 | 0.0476 / 0.0458 | 0.99 / 1.00 |
| antmaze g995 sd0 50k / 100k | 0.0150 / 0.0131 | 0.0151 / 0.0133 | 0.99 / 0.98 |
| antmaze g995 sd2 100k | 0.0137 | 0.0139 | 0.99 |
| antmaze g99 mpess0 sd2 300k | 0.0751 | 0.0747 | 1.01 |

The ratio is 1.00 everywhere. **The critic ensemble's disagreement carries no information
about whether a latent is in the data support.** `kappa * spread` is therefore not a
support-aware penalty at all; it is a near-uniform one-signed shift, which is precisely the
mechanism §3 derives. This is independent confirmation of H-PESS that does not go through
the growth fit, and it also explains why removing the term costs less than a support-aware
penalty would -- it was never buying support-awareness. Whatever cube gets from kappa=0.5 is
conservatism in the target, not in-support discipline.

### 8.4 The dead seeds are a SELECTION failure, not a critic-scale failure

`tools/diag_gpi_selection.py` on cube `mpess0` @350k, three seeds, plus the healthy
kappa=0.5 seed (`$PSM_DATA/logs/diag_deadseeds/`). `spearman` is rank correlation between
the critic's Q over the candidate roster and Monte-Carlo returns from those same candidates.

| run | rollout succ | q_spread_rel | unc@best | MC succ of pool | spearman | spearman p10 |
|---|---|---|---|---|---|---|
| `affine_strict_cube` sd0 (kappa=0.5) | 0.800 | 0.704 | 0.017 | 0.791 | 0.147 | -0.276 |
| `mpess0` sd0 (dead) | 0.000 | 19.90 | 1.485 | 0.791 | 0.059 | -0.168 |
| `mpess0` sd1 (live) | 0.650 | 18.26 | 1.336 | 0.791 | 0.164 | -0.050 |
| `mpess0` sd2 (dead) | 0.050 | 6.09 | 0.449 | 0.791 | -0.020 | -0.251 |

Two things follow. **The candidate pool is not the problem**: 79% of the decoded prior
candidates succeed under MC rollout, identically in every seed, so the frozen flow offers a
good action at essentially every state. **The critic cannot rank them**: Spearman is
0.06-0.16 in every arm including the healthy one, and the 10th percentile over states is
negative everywhere -- on a tenth of states the ranking is inverted.

The eval-time sweep on the dead `mpess0` sd0 @350k confirms this is decisive. Same frozen
checkpoint, only the acting flag changes, 500 episodes:

| acting | success |
|---|---|
| base (`gpi_num_u=64`, argmax) | 0.002 |
| `gpi_select=mean` | 0.000 |
| `u_clip=1.5` | 0.006 |
| **`gpi_num_u=8`** | **0.140** |

Cutting the roster from 64 to 8 is a 70x recovery on a frozen checkpoint. With a Spearman of
0.06 the argmax over 64 draws is close to a max over noise, so it reliably selects the
candidate with the largest positive Q error; at K=8 the selection bias is much smaller and
acting falls back toward sampling the 79%-successful pool. `u_clip` and dropping acting
pessimism do nothing, so this is the size of the argmax and not the radius of the box or the
conservatism of the score.

This does not contradict the 2026-09-07 finding that acting-side knobs are a no-op: that was
measured against the gamma=0.995 *divergence*. Dead seeds at gamma=0.98-0.99 are a different
failure, and for that one an acting-side knob is the largest single effect on record.

### 8.5 What should happen next

1. `gpi_num_u` is the cheapest open lever and has never been swept. K in {4, 8, 16, 32, 64}
   at eval on existing checkpoints, both envs, live and dead seeds. Costs no training.
2. The kappa=0.25 arm on cube 0.98 and antmaze 0.99, 3 seeds, 500k. It is the only setting
   with a rate benefit (8.0e4, 1.8x) that has not been shown to cost cube its performance.
3. Whether the low Spearman is fixable at all is now the central question for the method.
   `index_agg=expectile` exists precisely to avoid an argmax over samples of a learned
   function and has never been evaluated at 500 episodes against this failure.

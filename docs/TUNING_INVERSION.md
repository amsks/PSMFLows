# Tuning the preimage inversion

The flow `G(s, u)` maps a latent `u ~ N(0, I)` to an action. Inversion asks the reverse:
which `u` would have produced the recorded action `a` at state `s`? Two answers are stored
per transition — the exact backward-ODE **point** preimage `u*`, and a Gaussian **mixture**
around it. Downstream (`use_point_preimage=false`) trains on the mixture.

## The one trade-off

The mixture exists to densify: instead of one latent per transition it attaches a whole
ε-region of latents to the recorded outcome, so the successor measure is fitted over latent
space rather than at isolated points. Every latent it admits asserts "policy `u` would have
taken `a` here", which is exactly true only at `u*`. So ε trades:

- **coverage** — a latent the actor could emit at deployment must have been trained
  somewhere. Draw `u ~ p0` at a state; is it inside any mixture component of that state or
  a nearby one?
- **label noise** — `‖G(s, u) - a‖` for a mixture draw. This is the lie in the label.

Coverage is trivially 1.0 for wide enough covariances, so the two numbers are only readable
together. Pick the widest setting inside a decode-error budget.

The point preimage decodes at ~5e-6 on every setting measured, so the inversion itself is
essentially exact: all of the mixture's decode error is width that was chosen, not flow
error. Decide the budget in action units — cube actions are `[-1, 1]^5`, so an L2 of 0.22 is
~0.10 per dimension, against ~1.29 for a uniformly random action.

## Knobs

In `configs/inversion/default.yaml`. All of them invalidate existing preimage npz files.

| knob | effect |
|---|---|
| `alpha` | inverse temperature, `1/ε`. **The direct width knob.** Up = tighter = less coverage, less label noise. |
| `num_samples` | proposal draws the EM fits from. **Dominates everything below.** Too few and every component is shrunk toward `u*` regardless of the other settings. |
| `prior_scale` | weight on `N(u; 0, I)` in the target. 1.0 is the true posterior. 0.0 lets the mixture leave the prior's typical set entirely, which is what produced diverging decodes. |
| `n_steps` | EM iterations. With `prior_scale > 0` the target is proper, so **more iterations concentrate** — this points the opposite way from the pre-2026-08-14 behaviour. |
| `num_clusters` | mixture components. Not a useful lever; see below. |
| `n_initial_steps` | must equal `agent.flow_steps` (asserted). Not a tuning knob. |

## What to look at

`tools/tune_preimage_inversion.py` prints, per setting:

- `covered_at_k` — the coverage curve. `k=0` is the row's own mixture, `k=64` allows any of
  its 64 nearest states to supply the latent. Read the curve, not one number: how far the
  measure generalises across states is not pinned down.
- `covered_by_prior_quantile` — **where** the misses are. A miss past `u_clip` (3.0) costs
  nothing, since psmflow clamps every draw to that box. A miss in `q0-25` is fatal.
- `decode_mix` — the label noise. `decode_pt` is the floor.
- `E‖u‖²` against `d_a` — typicality (assumption A2). Should sit near `d_a`; far above means
  the mixture has left the prior and the latents seen at train time are not the ones drawn
  at test time.
- `nan_decode_frac` — draws so far off-support the forward integration blows up. Should be 0.
- `ess` — bounded by `num_samples`, and measured against whatever target the setting
  defines. Compare as ESS/N and only within a column; it says nothing about typicality.

## What is already measured (cube, `d_a=5`)

- `num_samples` is the dominant knob. At `alpha=5`, 32 → 128 moves k=64 coverage
  0.53 → 0.96 and decode 0.193 → 0.227.
- `num_clusters` 1/2/3/5 at fixed `num_samples=32` moves coverage 0.53/0.39/0.22/0.11 —
  monotonically down, because K components fitted from N/K draws each cover less than one
  fitted from N. At `num_samples=128` the dependence nearly vanishes (0.962 at K=1, 0.978 at
  K=3). Leave it at 1.
- Coverage falls off geometrically with `d_a`: the same `alpha` covers 0.74 of the prior on
  pointmaze (`d_a=2`) and 0.19 on cube. Antmaze (`d_a=8`) will be worse again.
- The `alpha` frontier in `docs/PREIMAGES.md` was measured at `num_samples=32` and
  understates coverage throughout. Treat it as the *shape* of the trade-off.

## Scripts

Nothing here needs the full precompute; inverting the whole buffer takes hours and is the
wrong loop to tune in.

**Tune** — inverts one sampled batch per setting, minutes each:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python tools/tune_preimage_inversion.py \
  agent=fql env_name=cube-single-play-singletask-v0 \
  restore_path=$PSM_DATA/flow/cube-single-play restore_epoch=500000 \
  agent.flow_steps=100 +n_rows=1280 +sample_k=128 +cov_k=64 \
  '+sweep={alpha: [5.0, 20.0], prior_scale: [0.5, 1.0]}' \
  report_out=/tmp/tune.json
```

Every combination runs against the same batch and the same draw seed, so a difference
between rows is the setting. The batch is sampled as whole neighbourhoods (`+sample_mode`,
`+sample_k`) because coverage is a question about what nearby transitions share, and a
uniform sample of a 1M-row buffer has no neighbourhoods left to ask about.

**Persist a batch** — `tools/precompute_preimages.py +preimage_sample=<n>` writes an npz for
the sampled rows only, recording `source_index` so the dynamics test can still find the
simulator state. Drop `+preimage_sample` for the full buffer.

**Score an npz** — `tools/validate_decode_recovery.py` (decode error per latent source;
whether generated actions exist anywhere in the buffer) and
`tools/validate_dynamics_recovery.py` (restore MuJoCo to the recorded state, step the
decoded action, compare against the recorded action replayed from the same state).
`tools/analyze_preimages.py` summarises a stored npz without a flow.

**Fetch artifacts** — `scripts/fetch_preimages_hf.sh` pulls the published checkpoints and
npz files into `$PSM_DATA` and repairs their provenance sidecars for this machine;
`scripts/run_recovery_tests.sh` runs both recovery tools against them.

## Gotchas

- `agent.flow_steps` must equal `inversion.n_initial_steps`, and must equal the npz
  sidecar's `flow_steps`. Both are asserted. A decode under a different discretization than
  the inverse measures the mismatch, not the recovery.
- An npz records the absolute path of the checkpoint it was inverted from and will refuse to
  pair with one that resolves elsewhere — hence the sidecar repair in the fetch script.
- `MUJOCO_GL=egl` is invalid on macOS; leave it unset.
- Keys not declared in `configs/config.yaml` need a leading `+`.

# DSRL-SAC: a scalar critic over LATENTS in `agents/latentrl.py`

**Date:** 2026-09-03 · **Branch:** `feat/inversion-integration` · **Machine:** KISSKI (SLURM, H100)

## Motivation

DSRL (Wagenmaker et al. 2025) steers a frozen diffusion/flow policy through its *input
noise*: the RL problem is posed in noise space and the generative policy is never touched.
Its offline variant needs **noise aliasing** — offline data carries actions but no noise —
and that is exactly what our Stage-B preimages supply: for every transition we already have
the `u` that decodes to the recorded action. So the steering-only, offline DSRL variant is
runnable here, and has never been run.

What has been run, and why it is not this:

* **psmflow's shipped actor** already steers in latent space, but against the successor
  feature readout `psi^T w`, which D3 measured flat: relative Q spread **1.1%** of |Q| over
  512 prior draws, actor at the 44th percentile of that distribution, actor gradient
  BC-dominated 5:1 (COMPENDIUM §4.7).
* **`agents/latentrl`** (the per-task ceiling probe, P2) is a *per-task scalar* critic, but
  its critic scores **executed/decoded actions** `Q(s, a)` with the gradient flowing back
  through the frozen decoder. At `residual_eps=0` it caps at **0.142 ± 0.025** (§4.1); the
  ε=0.05 residual arm reaches **0.905 ± 0.020** — i.e. everything that worked, worked by
  letting the critic see and move a raw *action*.

Nobody has run the middle cell: **a per-task scalar critic over latents, `Q(s, u)`, with the
frozen flow never called during training.** That isolates "is a latent TD loop learnable at
all when the critic input is the latent", separately from (a) the zero-shot representation
and (b) the decoder-in-the-loop gradient.

## Spec (as implemented)

New switch `agent.critic_input: action | latent` in `configs/agent/latentrl.yaml`, default
`action` so every existing run reproduces byte-for-byte (the `action` path traces to exactly
the pre-existing computation: same rng splits, same shapes, same info values).

Under `critic_input=latent`:

* Critic `Q(s, u)`, same `Value` ensemble, initialised on `(ex_observations, ex_u)`.
* TD: prediction on `u_data = clip(batch['noise_preimage'], ±u_clip)` (the in-sample
  anchor); target `r + gamma * mask * [mean − pessimism_penalty·unc](Q_target(s', u_next))`
  with `u_next = actor_latent(s', noise)` at stop-grad. Reward/mask/discount handling
  identical to the `action` path.
* Actor: `u_a = actor_latent(s, noise)`; loss `−Q(s, u_a)/stopgrad(|Q|.mean())` +
  `bc_alpha_latent · mean((u_a − u_data)^2)`. **No decode, no residual.** `bc_alpha_latent`
  defaults to the existing `alpha` (10.0); `0.0` is legal — that is the pure-DSRL setting.
* The residual head is inert: `residual_eps` must be `0.0`, asserted at `create()`.
* Acting is unchanged: actor latent → frozen one-step decode, so `tools/eval_checkpoint.py`
  works with `agent=latentrl` without changes.
* New logging: `critic_q_data`, `actor_q`, `bc_loss`, and `q_spread` / `q_spread_rel` —
  relative Q spread over 16 clipped prior draws at the same states, the analogue of
  psmflow's `ac_q_spread_rel`. Its key is folded out of `rng`, so the training stream is
  untouched.

`u_clip` stays a config knob and is the second axis of the sweep.

## Arms

Frozen Stage-A flow: `$PSM_DATA/flow/cube-single-play` @ 500000 (run group
`bcflow_cube_single_20260726_135032`, the same checkpoint every cube number in the
compendium decodes through). Preimages: the canonical `cube-single-play` npz
(alpha=20, num_samples=200), the same file the existing latentrl runs used.

| arm | `critic_input` | `u_clip` | `bc_alpha_latent` | `residual_eps` | seeds |
|---|---|---|---|---|---|
| `latsac_cube_uclip3` | latent | 3.0 | 10.0 | 0.0 | 0, 1 |
| `latsac_cube_uclip1` | latent | 1.0 | 10.0 | 0.0 | 0, 1 |

500k offline steps, `eval_interval=50000`, `eval_episodes=50`. Reported numbers come from
`tools/eval_checkpoint.py` at 500 episodes, quoted beside the BC control (cube 0.068).

## Expected outcomes (pre-registered)

* **near FQL (0.949) / the residual result (0.905):** offline latent TD works with a scalar
  critic; the zero-shot loss is in the *representation* (D2: best linear readout of φ
  explains ~13% of reward variance).
* **near 0.14:** offline latent TD does not survive without an action-space critic; next
  step is DSRL-NA (`Q_A` over dataset actions, latent Q distilled over prior draws).

**Prediction for the live Q-spread diagnostic:** psmflow's measure head sits at ~1% relative
spread. If this scalar critic is also ~1%, the actor gradient will again be BC-dominated and
the outcome will be the 0.14 branch. A spread materially above that (≳5%) with a low result
would instead point at the actor/optimisation, not at critic flatness.

`u_clip=1.0` is in because the tighter typical-set box shrinks the decode manifold the
critic has to separate; if flatness is a "everything decodes to nearly the same action"
artifact, the tighter box should raise the relative spread.

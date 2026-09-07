# GPI selection anatomy — why do neighbouring checkpoints score 0.70 and 0.28?

Date: 2026-09-05 · Branch `feat/inversion-integration` · Design + pre-registration.
Code: `tools/diag_gpi_selection.py`, `scripts/slurm/diag_gpi_selection.sbatch`,
`tests/test_diag_gpi_selection.py`. Subject: `affine_strict_cube/sd000`
(`psi_form=affine policy_index=latent train_actor=false acting=gpi`, `u_clip=3.0`,
`gpi_num_u=64`, `gpi_decode=onestep`).

## 1. The observation

500-episode success of one seed, checkpoint by checkpoint:

| epoch | 250k | 300k | 350k | 400k | 450k | 500k | BC control |
|---|---|---|---|---|---|---|---|
| success | .532 | .286 | **.704** | .282 | .272 | .086 | .072 |

Nothing in `train.csv` separates 350k from 400k: `psi_q_index_spread_rel` rises
monotonically through the whole run, `w_enc_spread` falls monotonically, `psm_loss` grows
monotonically. A monotone training trace and a non-monotone eval trace means the failure
is in the **acting rule**, not in a scalar the loss can see.

## 2. What `gpi_select` can get wrong

Acting is the paper's Rung 1: per step, draw `K=64` action latents `u ~ clip(N(0,I), 3)`
and `K` policy indices `u'` from the same prior, score all `K^2` pairs by
`Q = psi(s,u,u')^T w` with ensemble mean minus `0.5 * std`, and decode `argmax_u`. Three
failure modes are distinguishable and this probe separates them:

- **(a) edge-seeking.** The argmax moves to the corner of the `u_clip` box. The prior's own
  clip fraction is `P(|x|>3) = 0.27%` per coordinate, so the SELECTED latent's clip
  fraction against the CANDIDATE SET's clip fraction is a direct read of whether the
  critic prefers latents at the edge of the typical set — which the frozen flow was never
  asked to decode, and whose decode saturates the action clip.
- **(b) flatness.** Top-1 and top-2 collapse into each other, so the argmax is a coin flip
  among 64 prior draws — which is exactly the BC control (0.072). Measured as the
  top-1-minus-top-2 gap in units of the spread over the roster.
- **(c) wholesale ranking drift.** The critic still discriminates, but its ORDER over a
  fixed candidate roster is uncorrelated with the neighbouring checkpoint's. Then no
  checkpoint is "the" critic and the eval trace is a lottery over training noise.

## 3. The three sections

1. **rollout** — `N=20` eval episodes, seeded exactly as `utils.evaluation.evaluate`
   (`env.reset(seed)` once, action keys split from `PRNGKey(seed)`), with `gpi_select`'s
   latent branch re-implemented verbatim and instrumented. Per step: the `K^2` Q summary,
   the top-1/top-2 gap over pairs and over the per-`u` reduction `max_{u'} Q`, the selected
   `u` and `u'` (norm, clip fraction) against the candidate set's own norm and clip
   fraction, the decoded action (norm, action-clip fraction), the ensemble disagreement at
   the pick, and the observation. Persistence is reported as the cosine between
   consecutively selected latents — the roster is redrawn each step, so an index match is
   a `1/K` coincidence and is reported only as a sanity baseline.
2. **fixed-state** — 256 dataset states (`default_rng(7)` rows) x ONE candidate roster
   (`PRNGKey(12345)`), identical in every checkpoint's job. The `(256, 64, 64)` Q tensor
   goes to an npz; `compare` turns four of them into pairwise Spearman (over all 4096
   pairs and over the 64 per-`u` GPI scores), top-5 overlap and argmax agreement.
3. **MC ground truth** — 16 states, each `mc_lead=H/2=25` steps before a rewarding dataset
   transition in the same episode (a uniformly drawn state is unwinnable in 50 steps for
   every candidate, so the returns tie and the Spearman is meaningless; states 4 steps
   before a reward are won by every candidate, which ties the other way — the smoke run
   showed both). Restore the simulator from `qpos/qvel`, then for each of the 64 candidate
   `u` roll the frozen flow forward `H=50` steps HOLDING `u` FIXED (`a_t = G(s_t, u)`, the
   policy `pi_u` the index is supposed to name), summing reward. All-tied states are
   reported and dropped. The rollouts use only the frozen flow and the simulator, so the
   ground truth is IDENTICAL across checkpoints and `compare` can score a
   checkpoint-averaged selector against it.

The agent is built through `tools/eval_checkpoint.merge_run_config` (imported, not copied),
so the probe scores exactly the policy the 500-episode numbers came from.

## 4. Pre-registration (written before the jobs returned)

- **Expected:** the 500k checkpoint (0.086, at the BC control) shows **(b)**, a top-1/top-2
  gap that has shrunk relative to 350k, and/or **(a)**, selected-`u` clip fraction well
  above the candidate roster's 0.27%.
- **Expected:** cross-checkpoint Spearman on the fixed roster is HIGH between 250k and 350k
  (both good) and LOW between 350k and 400k, if the swing is a ranking flip rather than a
  score-scale change.
- **Possible negative:** all four checkpoints have similar gaps, similar clip fractions and
  high mutual rank correlation — in which case the 0.70/0.28 swing is not in the selection
  statistics at all and the next suspect is the decode/`task_z` inference, not GPI.
- **Expected:** MC Spearman is small at every checkpoint (`diag_latent_ranking_oracle`'s
  D1-analog already reads ~0.1 for the free head), and the interesting quantity is whether
  it is DIFFERENT between the good and bad checkpoints rather than whether it is large.

## 5. Outcome

Four 2 h jobs, `sd000` at 250k/350k/400k/500k; JSON + npz in
`$PSM_DATA/logs/diag_gpi_selection_cube_sd0_<epoch>.{json,npz}`, cross-checkpoint table in
`..._compare.json`. The probe's own 20-episode rollout reproduces the eval500 ordering
(0.65 / 0.80 / 0.40 / 0.00 against .532 / .704 / .282 / .086), so it is scoring the same
policy.

| quantity | 250k | 350k | 400k | 500k |
|---|---|---|---|---|
| eval500 success (reference) | 0.532 | 0.704 | 0.282 | 0.086 |
| probe rollout success (20 ep) | 0.65 | 0.80 | 0.40 | 0.00 |
| Q mean level | -3802 | -4113 | -4288 | -6394 |
| Q spread / |mean| | 0.564 | 0.704 | 0.927 | 1.048 |
| top1-top2 gap / std (pairs) | 0.0174 | 0.0114 | 0.0111 | 0.0123 |
| top1-top2 gap / std (per-u) | 0.354 | 0.309 | 0.334 | 0.343 |
| ensemble unc at pick / |Q| | 0.0157 | 0.0166 | 0.0158 | 0.0191 |
| selected |u| | 2.85 | 2.77 | 2.78 | 2.77 |
| candidate |u| (prior) | 2.13 | 2.13 | 2.13 | 2.13 |
| selected |u| / candidate |u| | 1.34 | 1.30 | 1.31 | 1.30 |
| selected |u| p90 | 3.61 | 3.56 | 3.56 | 3.56 |
| selected |u'| | 2.72 | 2.58 | 2.57 | 2.48 |
| selected-u clip frac | 0.0128 | 0.0125 | 0.0121 | 0.0108 |
| candidate-u clip frac | 0.0028 | 0.0027 | 0.0028 | 0.0028 |
| selected-u' clip frac | 0.0122 | 0.0069 | 0.0092 | 0.0062 |
| action |a| | 0.805 | 0.752 | 0.752 | 0.772 |
| action clip frac | 0.0181 | 0.0113 | 0.0134 | 0.0173 |
| persistence cos(u_t,u_t-1) | 0.094 | 0.139 | 0.107 | 0.108 |
| persistence cos(u'_t,u'_t-1) | 0.237 | 0.335 | 0.231 | 0.256 |
| fixed: Q spread / |mean| | 0.339 | 0.478 | 0.481 | 0.586 |
| fixed: gap/std (per-u) | 0.431 | 0.367 | 0.312 | 0.395 |
| fixed: selected |u| | 2.95 | 2.86 | 2.83 | 2.88 |
| fixed: argmax-u concentration | 0.125 | 0.203 | 0.098 | 0.121 |
| fixed: index range max-min Q | 5466 | 10776 | 8332 | 12773 |
| rank(pess) vs rank(mean-Q) | 0.926 | 0.945 | 0.942 | 0.945 |
| rank(online) vs rank(target) | 0.960 | 0.915 | 0.960 | 0.946 |
| top5 online vs target | 0.84 | 0.78 | 0.83 | 0.82 |
| MC states non-degenerate | 5 | 5 | 5 | 5 |
| MC Spearman (Q vs return) | +0.104 | +0.147 | +0.134 | +0.100 |
| MC Spearman (target critic) | +0.094 | +0.178 | +0.199 | +0.070 |
| MC Spearman (-|u| vs return) | +0.281 | +0.281 | +0.281 | +0.281 |
| MC regret of Q argmax | 16.80 | 22.80 | 23.00 | 32.40 |
| MC regret of a random pick | 19.11 | 19.11 | 19.11 | 19.11 |
| MC success rate (any cand) | 0.791 | 0.791 | 0.791 | 0.791 |

**The pre-registered "possible negative" is what happened.** On the STATE-MATCHED probe —
same 256 dataset states, same 64x64 candidate roster in every job — the four checkpoints
are nearly indistinguishable: selected-`u` norm 2.83-2.95 (prior 2.13) everywhere, top-1
minus top-2 gap 0.31-0.43 sd everywhere, selected-`u` clip fraction ~1.2% against the
roster's 0.28% everywhere, action clip fraction 1.1-1.8% everywhere. Neither pre-registered
failure mode (a) or (b) distinguishes 350k from 400k; both are permanent properties of the
rule, present at the good checkpoints too.

**Failure mode (c) is real and large.** The per-`u` GPI score — `max_{u'} Q`, the ranking
the argmax consumes — correlates only **rho 0.21-0.36** between ANY pair of checkpoints,
including 250k vs 350k (0.31), the two that both work. Top-5 overlap 0.20-0.28, argmax
agreement 11-19%. The `K x K` pair matrix looks far more stable (rho 0.64-0.83) purely
because it is dominated by the `u'` axis (`index_matters` ~5-13k against a per-`u` spread of
a few hundred): psi has learned a strong opinion about the policy index and almost none
about the action latent. The target critic drifts identically (0.23-0.40), and within a
checkpoint online and target rank the same roster at rho 0.92-0.96 — `tau=0.01` has long
since converged, so **target-critic selection is not a stabiliser here**. Dropping the
pessimism term changes the ranking even less (rho 0.93-0.95).

**MC ground truth** (16 states 25 steps before a rewarding dataset transition, 64 fixed-`u`
policies rolled H=50; 5 of 16 states non-degenerate): Q-rank vs return-rank Spearman is
+0.10 / +0.15 / +0.13 / +0.10 — flat, tiny, and unrelated to the eval swing. The
checkpoint-independent baseline `-||u||` scores **+0.28**, i.e. simply preferring latents
nearer the prior mean ranks the candidates better than any of the four critics, while GPI
selects `|u| = 2.8` against the roster's 2.13. The MC probe measures an open-loop fixed-`u`
policy and the deployed one reselects every step, so it is a weak proxy — but the sign is
consistent at every checkpoint. MC **regret** of the Q-argmax is 16.8 / 22.8 / 23.0 / 32.4
against 19.1 for a uniformly random pick: three of the four checkpoints select WORSE than
chance on this proxy, the 0.086 one by 70%.

**Checkpoint averaging, measured.** Because the MC rollouts touch only the frozen flow and
the simulator, the ground truth is shared, and `compare` can score the average of the four
per-state z-scored rankings against it: Spearman **0.121 -> 0.166**, regret **23.75 ->
19.2**. Averaging four checkpoints removes the harm (the ensemble is level with random
selection instead of worse than it) but does not make the selector good — consistent with
the drift being mostly independent noise on a weak shared signal.

## 6. What this says to do

- The eval swing is **not** a change in what selection does; it is the ranking being
  re-randomised between checkpoints with only a weak shared component. A single checkpoint
  is a lottery ticket, which is why 500-episode evals of neighbouring checkpoints disagree
  by 0.4 with a monotone training trace.
- Cheap, evidence-backed moves, in order: (i) **shrink the candidate ball** — `-||u||`
  out-ranks every checkpoint's critic on MC (+0.28 vs +0.10..+0.15) while GPI walks into
  the tail (`|u| = 2.8` against the roster's 2.13), so drawing candidates from a smaller
  ball (or rejecting draws above the prior median norm) moves in the direction the ground
  truth endorses and costs nothing; (ii) **average rather than sharpen** — checkpoint
  averaging measurably lifts MC Spearman 0.121 -> 0.166 and regret 23.75 -> 19.2, and a
  soft top-`m` over the 64 candidates is the same medicine within one checkpoint; (iii)
  **not** target-critic selection and **not** dropping pessimism, both measured here as
  ~no-ops (rho 0.92-0.96 with the deployed rule).
- The `index_matters` vs per-`u` spread asymmetry is the deeper finding: psi discriminates
  policy indices ~30x more strongly than action latents, so the inner argmax over `u` — the
  one that actually chooses the action — runs on the weakest part of the signal.

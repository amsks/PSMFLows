# Baseline: PSM in RAW ACTION SPACE, no BC anchoring — cube-single-play

**Generated** 2026-09-06 · 3 seeds × 500k offline steps × 500-episode evals every 50k.
Ladder rows regenerate with `PSM_DATA=... .venv/bin/python scripts/baselines/psm_raw_nobc_table.py`
(that script and its launcher are on the `archive` branch, not this one);
launcher `scripts/baselines/psm_raw_nobc.sh`; report JSONs
`$PSM_DATA/logs/eval500_<group>_<epoch>k_sd<seed>.json`.

## What this baseline is, and what it is not

The primary agent (`agent=psmflow`) is the **paper-strict affine LatentFlowPSM**: an affine
successor measure whose action slot holds a *flow latent* `u`, so every action it evaluates,
bootstraps or executes is a decode `G(s,u)` of the frozen behaviour flow and the bootstrap
distribution equals the data distribution by construction. This baseline **deletes the flow
and the anchor and keeps everything else**:

* **raw action space** — the measure's middle slot is the dataset action, not a preimage
  latent; no Stage-A checkpoint, no Stage-B npz, no decode at act time;
* **no BC anchoring** — the actor's objective is `-Q.mean()` and nothing else: no
  MSE-to-dataset-action, no flow distillation, no AWR weight, no in-support regulariser.

It answers one question: *does the latent action interface buy anything, or would a
raw-action PSM with an unconstrained greedy actor do as well?* Removing the anchor also
removes the only thing keeping an offline deterministic-policy-gradient actor on the data
manifold, so **critic exploitation is the pre-registered expected failure mode**, not a bug.
The anchor is deliberately not restored — that is the point of the control.

## Which archived agent, and why

`archive/agents/affine_psm.py` is the primary arm (**Arm A**). It is the *raw-action twin of
the shipped agent*: same affine/LP measure family `M(s,a,x) = Phi(s,a,x)·w + b(s,a,x)` with
the factored basis `Phi = A(s,a) phi_x(x)`, same proto/codebook contrastive TD objective,
same amortized `w`-conditioned actor, same goal-conditioned `full` (LP) inference — with the
dataset action in the slot the shipped agent fills with `u`. Since the project's algorithm
became affine on 2026-09-04, this is the like-for-like control.

`archive/agents/psm.py` — the bilinear PSM of arXiv 2411.19418, `M(s,a,x) = psi(s,z,a)ᵀ
phi(x)` — is run as **Arm B**, the literal "psi(s,a,·)ᵀ phi(s')" reading of the request and
the peer baseline of record. It is cheap (~8 ms/step) so both arms run.

Neither agent is un-archived. `scripts/baselines/run_archived.py` and
`scripts/baselines/eval_archived.py` import them as `archive.agents.*` and splice
`archive/configs` into hydra's search path; `main.py`, `agents/__init__.py`,
`agents/psmflow.py`, `agents/fql.py`, `utils/psm_networks.py` and `tools/eval_checkpoint.py`
are untouched. Deleting `scripts/baselines/` reverts the repo exactly.

## The BC-off switch, exactly

One flag per arm: **`agent.actor.bc_coeff=0.0`**, plus `agent.actor.type=ddpgbc`.

| where | at `bc_coeff=3.0` (the shipped default) | at `bc_coeff=0.0` (this baseline) |
|---|---|---|
| `affine_psm.actor_loss` | `-Q/|Q| + 3.0·mean((a − a_data)²)` | `-Q.mean()` |
| `affine_psm.flow_actor_loss` | `-Q/|Q| + 3.0·mean((a − ODE_rollout)²) + bc_flow_loss` | (not used — see below) |
| `affine_psm.distill_actor` (eval-time) | `-Q/|Q| + 3.0·mean((a − a_data)²)` | `-Q.mean()` |
| `psm.actor_loss` | `-Q/|Q| + 3.0·mean((a − a_data)²)` | `-Q.mean()` |

The `|Q|` normalisation exists only to scale the BC term against Q; at `bc_coeff=0` that
branch is skipped and the objective is the bare `-Q`.

`actor.type=ddpgbc` rather than the config default `flow`: with `type=flow` and
`bc_coeff=0` the one-step actor's objective is already pure `-Q` (the distillation target is
gated on `bc_coeff>0`), but the flow-matching velocity field is *still* trained on dataset
actions every step and nothing consumes it — dead compute, and a behaviour-cloning loss
still sitting in the printed objective. `ddpgbc` makes the claim "no BC term anywhere"
literally true. Its usual objection — a tanh-mean actor fit by MSE regresses to the mean of
multimodal cube actions — is an objection to the *BC term*, which is exactly what is removed.

## Pessimism: kept as the agents already have it, reported here

| | Arm A `affine_psm` | Arm B `psm` |
|---|---|---|
| pessimism on the TD target | **none** (no such term in the agent) | `pessimism_penalty=0.0` (present, off) |
| pessimism on the actor's Q | **none** | `actor_pessimism_penalty=0.5` × ensemble disagreement over `num_parallel=2` heads |
| bounded measure | `Phi`, `w` √d-normalised; offset `b` tanh-bounded at `b_scale=10` | `norm_z=true`; `phi` orthonormality-regularised |
| decorrelation | `ortho_coef=1000` on the x-side basis | `ortho_coef=1000` on `phi` |
| target nets | Polyak `tau=0.01` | Polyak `tau=0.01` |
| exploration noise in the actor loss | none | truncated `actor_std=0.2`, `stddev_clip=0.3` |

Arm A therefore has **no pessimism at all** — its only stabilisers are normalisation and
bounding. Nothing was added or removed on either arm.

## Hyperparameters as launched

Common: `env_name=cube-single-play-singletask-v0`, `offline_steps=500000`, `online_steps=0`,
`batch_size=1024`, `discount=0.98`, `tau=0.01`, `ortho_coef=1000.0`, `max_log_seed=16`,
seeds 0/1/2, `save_interval=50000`, `eval_interval=50000`, in-loop `eval_episodes=50`
(reported numbers are the 500-episode re-evals only), `dataset_fraction=1.0`.

**Arm A `affine_psm`** (`d_dim=z_dim=128`, `lr=lr_w=lr_actor=1e-4`):
`measure.factored=true k_dim=32 hidden_dim=1024 hidden_layers=3 b_scale=10.0`;
`actor.type=ddpgbc hidden_dim=1024 hidden_layers=1 embedding_layers=2 bc_coeff=0.0`;
`inference.mode=full use_dgd=true num_inference_steps=5120 lagrange_hidden_dim=256
lagrange_hidden_layers=2 norm_w=true num_actor_inference_steps=512`.

**Arm B `psm`** (`z_dim=128`, `lr_phi=1e-5 lr_sf=1e-4 lr_actor=1e-4`): `num_parallel=2`,
`mix_ratio=0.5`, `pessimism_penalty=0.0`, `actor_pessimism_penalty=0.5`, `actor_std=0.2`,
`stddev_clip=0.3`, `norm_z=true`, `phi_input=s`, `phi.hidden_dim=256 hidden_layers=2`,
`sf.hidden_dim=1024 hidden_layers=1 embedding_layers=2`;
`actor.type=ddpgbc hidden_dim=1024 hidden_layers=1 embedding_layers=2 bc_coeff=0.0`.

Eval protocol: `tools/eval_checkpoint.py`'s machinery via `scripts/baselines/eval_archived.py`
— the agent config is inherited from the run's own `flags.json` (verified: `actor.bc_coeff:
config 3.0 -> run 0.0` on every job), 500 episodes, Wilson 95% interval, env init and action
RNG both pinned to `seed=0` so every arm sees the same 500 initial states. Arm A infers a
goal-conditioned `w_inf` per eval (`infer_eval` -> LP); Arm B infers a reward task vector
(`infer_eval_z`, relabel 10000, shift 1.0), both exactly as the pre-archive `main.py` did.

## The ladder — 500 episodes per point

### Arm A - `affine_psm`, raw actions, `bc_coeff=0`

Affine measure `M(s,a,x) = Phi(s,a,x)*w + b(s,a,x)` (factored `Phi = A(s,a) phi_x(x)`), amortized `ddpgbc` actor, actor loss `-Q.mean()`. No pessimism term of any kind.

| epoch | sd0 | sd1 | sd2 | mean ± sd |
|---|---|---|---|---|
| 50k | 0.000 | 0.000 | 0.000 | 0.000 ± 0.000 |
| 100k | 0.000 | 0.000 | 0.000 | 0.000 ± 0.000 |
| 150k | 0.000 | 0.000 | 0.000 | 0.000 ± 0.000 |
| 200k | 0.000 | 0.000 | 0.000 | 0.000 ± 0.000 |
| 250k | 0.000 | 0.000 | 0.000 | 0.000 ± 0.000 |
| 300k | 0.000 | 0.000 | 0.000 | 0.000 ± 0.000 |
| 350k | 0.000 | 0.000 | 0.000 | 0.000 ± 0.000 |
| 400k | 0.000 | 0.000 | 0.000 | 0.000 ± 0.000 |
| 450k | 0.000 | 0.002 | 0.000 | 0.001 ± 0.001 |
| 500k | 0.000 | 0.000 | 0.000 | 0.000 ± 0.000 |

**Late mean (300k-500k, n=15 checkpoint x seed measurements, 500 episodes each): 0.000, sd 0.001, min 0.000, max 0.002.**

### Arm B - `psm` (bilinear PSM), raw actions, `bc_coeff=0`

Bilinear measure `M(s,a,x) = psi(s,z,a)^T phi(x)` (arXiv 2411.19418 port), amortized `ddpgbc` actor, actor loss `-Q.mean()` with the agent's own `actor_pessimism_penalty=0.5` ensemble-disagreement penalty kept on.

| epoch | sd0 | sd1 | sd2 | mean ± sd |
|---|---|---|---|---|
| 50k | 0.000 | 0.000 | 0.000 | 0.000 ± 0.000 |
| 100k | 0.000 | 0.000 | 0.002 | 0.001 ± 0.001 |
| 150k | 0.000 | 0.000 | 0.000 | 0.000 ± 0.000 |
| 200k | 0.000 | 0.000 | 0.000 | 0.000 ± 0.000 |
| 250k | 0.000 | 0.000 | 0.000 | 0.000 ± 0.000 |
| 300k | 0.000 | 0.000 | 0.000 | 0.000 ± 0.000 |
| 350k | 0.000 | 0.000 | 0.000 | 0.000 ± 0.000 |
| 400k | 0.000 | 0.000 | 0.000 | 0.000 ± 0.000 |
| 450k | 0.000 | 0.000 | 0.000 | 0.000 ± 0.000 |
| 500k | 0.000 | 0.000 | 0.000 | 0.000 ± 0.000 |

**Late mean (300k-500k, n=15 checkpoint x seed measurements, 500 episodes each): 0.000, sd 0.000, min 0.000, max 0.000.**

## Reading

**Both arms are flat zero for the whole 500k, on every seed.** Pooled over the entire
sweep, Arm A scored **1 success in 15 000 episodes** (30 checkpoint x seed evals x 500) and
Arm B **1 in 15 000** — Wilson 95% upper bound 0.0004 on each. The only non-zero cells in
the ladder (Arm A 450k sd1, Arm B 100k sd2) are single episodes, i.e. 0.002 with a Wilson
interval of [0.000, 0.011]; neither is a signal. The in-loop 50-episode eval read 0.00 at
all 11 points of all 6 runs, so nothing was missed between checkpoints either.

| | cube-single-play, 500 episodes |
|---|---|
| **BC control** (frozen Stage-A flow acting alone, `agent=fql agent.bc_only=true`) | **0.072** |
| **affine LatentFlowPSM** (`agent=psmflow`, late-checkpoint mean >=250k, n=10) | **0.424 ± 0.141** |
| **this baseline, Arm A** (`affine_psm`, raw actions, no BC) late mean 300k-500k, n=15 | **0.000** [0, 0.0004] |
| **this baseline, Arm B** (`psm` bilinear, raw actions, no BC) late mean 300k-500k, n=15 | **0.000** [0, 0.0004] |

The baseline does not merely underperform the method — it does not clear the behaviour-
cloning control, and it does not clear **zero**. That is the pre-registered expected failure
mode, and the training logs say it is the predicted mechanism rather than an optimisation
failure. The critic is *learning*: `psm_loss` and `orth_loss` converge normally on both
arms. What diverges is the greedy actor's own Q:

| arm | seed | Q at 5k | 50k | 100k | 250k | 500k |
|---|---|---|---|---|---|---|
| A `affine_psm` | 0 | 60 | 249 | 110 | 16 | 10 |
| A | 1 | 142 | **1810** | 388 | 20 | 67 |
| A | 2 | 60 | 15 | 23 | 22 | 28 |
| B `psm` | 0 | 406 | 1440 | 1820 | 1900 | **1970** |
| B | 1 | 407 | 1290 | 1970 | 1950 | **2340** |
| B | 2 | 372 | 1440 | 1790 | 1920 | **2080** |

For Arm B the actor's Q is `psi(s,z,a)^T z` with `norm_z=true`, so `|z| = sqrt(128) = 11.3`
and a Q of 2000 means a `psi` norm of ~175 at the actor's chosen action — a value the
contrastive measure loss never sees on data, reached and held from 100k onward. The
`actor_pessimism_penalty=0.5` disagreement penalty over two heads does not stop it: two
heads agree with each other about an action neither was trained on. For Arm A the affine
measure is **not** bounded in the factored path — only the x-side basis `phi_x` is
sqrt(k)-normalised, so the product `Phi = A(s,a) phi_x(x)` is unnormalised and `Q` is free
— and the same spike appears early (seed 1 at 50k) before collapsing back. Either way the
actor is optimising a value the measure only assigns off the data manifold.

## Caveats

* **Not a tuned baseline.** These are the archived agents' own cube hyperparameters with one
  flag changed. No sweep over `lr_actor`, `ortho_coef` or the two-timescale ratio was run;
  a tuned no-BC raw-action PSM might do better than zero. The claim supported here is
  narrow: *at the settings that produce the method's numbers, deleting the anchor and the
  latent interface together produces a policy that never succeeds.*
* **The baseline confounds two removals.** Raw action space and no BC anchoring are both
  changed relative to `agent=psmflow`. It therefore bounds the pair, not either alone. The
  separating arm — raw actions **with** `bc_coeff=3.0` — is the archived `affine_psm` cube
  push, PARKED 2026-07-26 without a 500-episode ladder, so that cell is empty.
* **`actor.type=ddpgbc` is a second change from the archived cube default** (`flow`), taken
  so "no BC term" is literally true; see the argument above. A `flow`-actor no-BC arm was
  not run.
* **Arm A's eval runs the agent's own `full` (LP) inference** — 5120 constrained-optimisation
  steps plus a 512-step actor re-fit against the solved `w_inf`, per checkpoint. That re-fit
  also maximises the bare `-Q` at `bc_coeff=0`, so it cannot rescue the policy; it is kept
  because it is the agent's inference procedure, not an anchor.
* Zero success means the ladder carries no information about checkpoint-to-checkpoint
  non-stationarity, the pathology that dominates the affine LatentFlowPSM arm. Nothing here
  speaks to that.

## Provenance

* Training: SLURM 2491706-8 (Arm A, 1h49m/seed) and 2491709-11 (Arm B, 1h13m/seed), all
  COMPLETED, `kisski-inference`, one H100 per seed.
* Runs: `$PSM_DATA/exp/PSMFLows/psm_raw_nobc_cube/sd00{0,1,2}_*` and
  `.../psm_bilinear_raw_nobc_cube/sd00{0,1,2}_*`.
* 60 eval jobs, ~4 min each; reports
  `$PSM_DATA/logs/eval500_psm_raw_nobc_cube_{epoch}k_sd{seed}.json` and
  `$PSM_DATA/logs/eval500_psm_bilinear_raw_nobc_cube_{epoch}k_sd{seed}.json`.
* Smoke before launch: SLURM 2491695/2491696 (200 steps, both arms) and 2491712/2491713
  (the eval path on those checkpoints); `flags.json` re-read after launch on all 6 runs and
  confirmed `actor.type=ddpgbc`, `actor.bc_coeff=0.0`, `save_interval=50000`,
  `offline_steps=500000`.


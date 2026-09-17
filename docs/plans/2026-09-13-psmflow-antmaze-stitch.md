# Affine PSMFlow AntMaze medium-stitch launch plan

**Goal:** Start the live affine PSMFlow on the exact medium-stitch data already
downloaded for the raw-action full-affine campaign, with three Stage-C seeds.

**Architecture:** Reuse existing Stage-A `main.py agent=fql`, Stage-B
`tools/precompute_preimages.py`, Stage-C `main.py agent=psmflow`, and
`tools/eval_checkpoint.py`. Freeze the current source with the existing snapshot
utility; submit scheduler dependencies with numerical/provenance checks. No agent,
loss, dataset-loader or inversion implementation changes are needed.

**Tech stack:** Existing JAX/Flax, Hydra, OGBench/MuJoCo, Slurm H100.

**Spec:** User's September13 request to also start affine flow PSM on AntMaze stitch.
Here this means the current affine-psi latent-index/GPI method, not the archived
RLU full-goal solver already running separately.

## Binding recipe

Campaign: `/mnt/home/amohan/git/Austin/PSMFLows/outputs/psmflow_antmaze_stitch_20260913`.
Source: campaign `code/`, copied by `scripts/prepare_affine_action_snapshot.py
--skip-evaluation-manifests`; the source manifest includes current working-tree content.
Data: exact Drive files in sibling `outputs/affine_antmaze_stitch_20260913/data/`.
Train SHA256 `b3fa34eef8d24dc812d734abaf74db279a5f39cd63debf23961d62c314d71e99`;
validation SHA256 `3015f74bb9d0f0fef0037d310c2be157b78af2c613f8cb0579f1de018e7c0af0`.
Explicit non-overwriting symlinks in `/mnt/home/amohan/.ogbench/data/` provide these
files to the installed OGBench loader, which ignores `OGBENCH_DATASET_DIR`.

Training/inversion env: `antmaze-medium-stitch-singletask-v0`. OGBench resolves this
to `antmaze-medium-singletask-v0` and the **stitch**, not navigate, dataset. Native
pairing yields1M train/100k validation transitions, obs29/action8. Retain existing
dataset action clipping at epsilon1e-5. Dataset fraction1.0, fraction seed0, no
augmentation, frame stacking or online training.

| Setting | Value |
|---|---|
| Shared Stage-A seed / updates | 0 /500000 |
| Stage-A batch / LR / distillation coefficient | 256 /3e-4 /10 |
| Stage-A actor and unused critic widths | [512,512,512,512] |
| Stage-A actor / critic layer norm | false /true (critic unused) |
| Stage-A flow steps / BC-only | 100 /true |
| Stage-A discount / tau / Q aggregation | .99 /.005 /mean (reward-dependent branch unused) |
| Stage-A normalize Q loss / encoder / skill conditioning | false /none /false |
| Stage-A log / save / in-loop eval | 1000 /100000 /disabled |
| Stage-B seed / components / batch | 0 /1 /256 |
| Stage-B inverse and forward steps | 100 /100 |
| Stage-B alpha / prior scale / samples / EM steps | 50 /1 /200 /10 |
| Stage-B data | Full1M training transitions; no validation rows |
| Stage-C seeds / updates / batch | [0,1,2] /500000 /1024 |
| Stage-C psi / policy index / current-action input | affine /latent /latent |
| Stage-C phi and index dimension / ensembles | 128 /2 |
| Stage-C phi width / depth | 256 /2 |
| Stage-C SF width / hidden / embedding layers | 1024 /1 /2 |
| Stage-C index encoder width / depth / norm | 256 /2 /unit norm |
| Stage-C phi / SF LR | 1e-5 /1e-4 |
| Stage-C discount / tau / orthogonality coefficient | .99 /.01 /1000 |
| Stage-C orthogonality mode / psi bound / train phi | fixed /none /true |
| Stage-C target / acting pessimism | .5 /.5 |
| Stage-C point preimage / samples per transition | true /1 |
| Stage-C invalid-row loss mask | false; existing dataset sampler independently excludes invalid rows; Stage-B aborts above1% invalid |
| Stage-C latent / index clipping | 3 /3 (index_clip=null inherits u_clip) |
| Stage-C action selection | GPI argmax,64 current latents x64 policy indices, max aggregation |
| Stage-C deployed decode | Frozen one-step decoder (ODE inverse uses100 steps) |
| Stage-C actor / DSRL-NA / action critic | All disabled |
| Stage-C reward inference | Closed form, normalized z,10000 relabel rows, reward shift+1 |
| Stage-C log / save / periodic evaluation | 1000 /100000 /10 episodes on training task every100k |
| Final evaluation | Saved500k endpoint, five tasks,500 episodes/task, seed0,4 workers,2 CPU threads/worker |
| Control | Same frozen Stage-A flow,500 episodes/task, five tasks |
| Aggregate | Task mean within each training seed, then three-seed Student-t95% CI |

Unused default knobs remain in each run's own complete flags.json; no hidden actor
branch is enabled. Stage-A BC-only and default Stage-C measure ignore training
rewards/masks. Rewards enter Stage-C task-vector inference/evaluation only. Stage B
retains native reward columns for that readout. Do not zero them or swap task rewards
into representation updates.

This is a new untuned dataset transfer, not a matched-architecture causal ablation of
the raw-action full-affine run. One shared flow/preimage artifact is used by the three
representation seeds. The one-step/ODE interface and known stability limitations
remain unchanged; no performance claim follows from a smoke pass.

## Operational steps and gates

- [x] Verify no compatible stitch flow/preimages in the checked artifact roots.
- [x] Link the exact downloaded files into OGBench's expected cache; verify both hashes.
- [x] Verify native loader dimensions/counts/environment and freeze current source.
- [x] Display the full active hyperparameter table before GPU launch.
- [x] Run the existing BC-only tests and a200-update full-size Stage-A GPU smoke;
  verify finite train/validation losses, flags, checkpoint and source/data hashes.
- [x] Submit Stage-A500k production only after that smoke passes; verify saved flags.
- [x] Queue trained-checkpoint D1/D3 diagnostics and a Stage-B256-row smoke, followed
  by the full1M inversion only if D3's existing gate (typicality>=.95, mean roundtrip<.1,
  mean ESS>20), the validity guard and artifact checks pass. The sample is an inversion
  diagnostic, never Stage-C training input. Record point roundtrip and ESS separately;
  the EM posterior is not sampled by the declared Stage-C arm. If only ESS fails,
  report the gate failure for a later explicit method decision, not an automatic waiver.
- [ ] After full inversion, require exact flow/checkpoint/env/row pairing and retained
  validity flags; run200 Stage-C updates on that full artifact and five2-episode
  restored-checkpoint evaluations. Require finite metrics/actions and unchanged flow.
- [x] Queue three Stage-C500k jobs dependent on the successful Stage-C smoke, with
  final five-task evaluation. Pending dependencies are not running representation jobs.
- [x] Queue the same-flow BC evaluation after Stage A and the existing strict
  `tools/report_seed_comparison.py` aggregation after all three final evaluations.
- [x] Persist exact submissions/dependencies, startup checks and newest HANDOFF entry.

Use `main.py` and the tool entrypoints directly, not the environment-key wrappers:
their `antmaze` case hardcodes **navigate**, and the standard Stage-B wrapper also
publishes to Hugging Face, which is not requested here. All outputs stay local.

Every scheduled phase checks the immutable source and original data hashes before
work. Each training phase uses an exclusive run group, validates its own flags and
finite logged metrics, and records a receipt. Downstream stages require those receipts.
Inversion remains expensive (historically4–25h); use48h for its production allocation,
not a promise of completion time. Stage-C numerical failure must prevent final-result
aggregation; no fallback to a different arm, dataset, seed or checkpoint.

## Launch observation —2026-09-13 00:56:45 UTC

The launch/setup work is complete; the scheduled experiments are not. The remaining
unchecked item above is the real-data Stage-C gate, which cannot run until inversion
finishes. Exact commands and dependencies are in campaign `launch_manifest.json`;
fresh scheduler/config evidence is in `reports/postlaunch_validation.json`.

| Stage | Job | Observed state / dependency |
|---|---|---|
| Stage-A200 smoke | 2506433 | COMPLETED0:0; finite losses and171 checkpoint leaves |
| Stage-A500k | 2506471 | RUNNING;257k logged updates, all metrics finite |
| D1/D3 + sampled256 Stage-B smoke | 2506527 | PENDING after successful Stage A |
| Full1M Stage B + decode diagnostic | 2506529 | PENDING after successful B gate |
| Full-artifact200-update Stage-C smoke | 2506567 | PENDING after successful full B |
| Stage-C seed0 /1 /2,500k + final eval | 2506571 /2506573 /2506574 | PENDING after successful C smoke |
| Same-flow BC, five-task eval500 | 2506575 | PENDING after successful Stage A |
| Three-seed aggregate | 2506577 | PENDING after all three C jobs and BC control |

Source/data hashes, all nine active-job dependencies, H100/CPU/memory requests,
Stage-A own flags, and finite training logs were freshly verified.64 focused tests
passed (10 BC/aggregation +54 affine/index/preimage/GPI); JUnit files are retained.
Independent read-only review found no clean-run launch blocker. Its operational
feedback added non-overwriting B-output gates and two threads per evaluation worker.
No model/loss/loader/inversion code changed. Existing Walker and raw-action affine
jobs were not altered. All future success/performance claims require their saved
gate/evaluation artifacts; pending jobs are not results.

## Authorized point-preimage relaunch — 2026-09-13 20:59 UTC

This supersedes the original ESS launch condition **only for the declared
point-preimage arm**. After the original D3 ESS-only failure, the user explicitly
approved proceeding with ESS diagnostic-only. Original jobs2506529/2506567/
2506571/2506573/2506574/2506577 were cancelled before starting; do not use the old
launch observation as current status.

Reuse the completed original Stage-A500k checkpoint and BC control. Keep all
binding model/inversion/evaluation settings above unchanged. New outputs and exact
commands are under campaign `relaunch_point_v2/`. Its local `point_gate.py`
requires the trained, hash-paired flow, the exact point-only affine latent/GPI
configuration, finite diagnostics, point roundtrip <0.1 and typicality >=0.95.
Both sampled and full preimage validators enforce these point metrics on all
valid rows, invalid fraction <=1%, and exact native row/checkpoint pairing.
Existing Gaussian/point computation and joint validity rules are not changed.
C readers require the point-specific full-data receipt; C200 still requires
finite metrics/actions and bitwise frozen-flow equality before production.

- [x] Rehash frozen source, exact data, trained flow and existing BC reports.
- [x] Show unchanged active hyperparameter table before relaunch.
- [x] Implement the campaign-local gate using red/green tests; 27 pass in
  `tests_final_formatted.xml`; Ruff passes.
- [x] Independently review all seven payloads; validate 39 inline Python blocks,
  28 Hydra calls, all four C receipt gates and unchanged expected model flags.
- [x] Fresh D3 and B256 smoke: job2513739 COMPLETED 0:0. D3 ESS18.012 remains
  diagnostic-only; point roundtrip0.000205, typicality0.9844. Sample smoke has
  zero invalid rows, roundtrip0.0002077203, typicality0.984375, and exact pairing.
- [x] Submit full B2513740 after the successful B gate; RUNNING at20:59:12 UTC,
  source/data and authorized receipt verified in its own startup log.
- [ ] Complete full 1M inversion and all full-data point/artifact checks.
- [ ] Pass full-artifact C200 job2513745 (after successful full B), including
  restored five-task two-episode checks and frozen-flow equality.
- [x] Queue C500k seeds0/1/2 as2513746/2513748/2513749 after successful C smoke.
- [x] Queue aggregate2513750 after all three C jobs; reuse original BC receipt.
- [x] Verify scheduler resources and dependencies, save the relaunch manifest,
  postlaunch evidence and a newest HANDOFF entry.

No representation seed is running yet and no new performance result exists.
Full B retains its 48h allocation; elapsed time is not a completion estimate.
The original trained-checkpoint D1 report and BC evaluation are reused, not
rerun or reinterpreted. The one-step/ODE decoder interface remains unchanged.

## Additional distribution arm — 2026-09-13 21:21 UTC

User explicitly requested distribution-preimage training alongside point. Keep the
point chain unchanged. In `distribution_v1/`, change only
`agent.use_point_preimage=false`; retain `measure_u_samples=1`,
`measure_u_mixture_shrink=null`, and all other binding settings. This samples the
full stored K=1 Gaussian, not point-plus-shrunk-extra Arm C. The full1M artifact from
running B2513740 already includes that Gaussian; no second Stage A/B is needed.

Pre-registration: test the five-task500k endpoint across the same three seed IDs
against point and the same-flow BC control. No robust gain is presumed; the Sep10
follow-up withdrew Arm C as a general fix. Equal configurations/seeds do not imply
identical NumPy minibatch streams, because the distribution path consumes extra draws.

- [x] Show the full active table before launch; compose both arms and verify that the
  only resolved agent difference is `use_point_preimage`.
- [x] Run53 focused tests and actual posterior sampler routing/reproducibility checks.
- [x] Complete B256 one-step diagnostic2513809; preserve its ancillary ODE10 report
  with an explicit limitation note rather than mislabel it ODE100.
- [x] Complete exact ODE100 diagnostic2513837: distribution meanL2.201563,
  point.002531, prior.943807, no nonfinite decodes. Low ESS and same-action-fidelity
  failure remain reported scientific limitations, not certified away.
- [x] Queue full-artifact distribution checks and C200 smoke2513889 after fullB2513740.
  Require exact full recipe/data/flow/receipt pairing, valid covariance and posterior
  sampling, finite decode diagnostics, false/1/null run flags, finite training/actions,
  frozen flow equality and restored five-task2-episode smoke.
- [x] Queue distribution C500k seeds0/1/2:2513895/2513897/2513898, after successful smoke.
- [x] Queue strict500x5x3 aggregate2513899; reuse the original matching BC receipt.
- [x] Verify current Slurm resources/dependencies and save exact launch/evidence records.
- [ ] Complete full1M inversion, distribution full-data checks and C200 smoke.
- [ ] Complete all three500k runs and final500-episode five-task evaluations.

The distribution receipts mark numerical integrity separately from posterior accuracy,
with explicit exploratory authorization and `posterior_accuracy_certified=false`.
No posterior hyperparameters/shrinkage are tuned in response to the diagnostic.
The original point-only ESS gate helper is not modified or reused as distribution
accuracy approval. See newest HANDOFF entry and
`distribution_v1/reports/postlaunch_validation.json` for the diagnostic caveats.

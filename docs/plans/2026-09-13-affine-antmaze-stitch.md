# Full-affine PSM AntMaze medium-stitch launch plan

> **For agentic workers:** Use the subagent-driven-development review workflow for
> the bounded launch adapter. No commits, live-agent restoration, or Walker changes.

**Goal:** Queue three runs of our archived RLU-derived full-affine PSM on the exact
AntMaze medium-stitch training/validation files supplied through Google Drive.

**Architecture:** Preserve the archived agent and matching model utilities in a
checksummed execution snapshot. A dedicated adapter loads the explicit OGBench files,
corrects only the proto-action range, runs reward-free representation/actor training,
and evaluates all five goals through the existing full-inference implementation.

**Tech stack:** Existing Python/JAX/Flax/Optax, OGBench/Gymnasium/MuJoCo, NumPy, Slurm H100.

**Spec:** User's September13 request, archived `AffinePSMAgent` at
`ee0e1746b36f48300eee06ad02e66b16f46851ff`, and the lineage correction in
[the interface audit](../design/2026-09-13-psm-interface-audit.md).

## Binding scope and recipe

| Setting | Value |
|---|---|
| Agent | Archived RLU-derived `AffinePSMAgent`, full goal inference; not PSMFlow |
| Source | Archive `ee0e1746b36f48300eee06ad02e66b16f46851ff` |
| Data source | Exact user Drive folder `1OqKCtqNwQoS7gUmM1QaFc8NN5aYHqBbt` |
| Training file | `outputs/affine_antmaze_stitch_20260913/data/antmaze-medium-stitch-v0.npz` |
| Training SHA256 | `b3fa34eef8d24dc812d734abaf74db279a5f39cd63debf23961d62c314d71e99` |
| Validation file | Same directory, `antmaze-medium-stitch-v0-val.npz` |
| Validation SHA256 | `3015f74bb9d0f0fef0037d310c2be157b78af2c613f8cb0579f1de018e7c0af0` |
| Valid transitions | 1,000,000 train; 100,000 validation; no mixing |
| Training seeds / updates | 0,1,2 / 500,000 |
| Batch / d_dim / z_dim | 1024 /128 /128 |
| Factored measure / rank | true /32 |
| Measure width / hidden depth / bias scale | 1024 /3 /10 |
| Proto code bits / action law | 16 /bounded `2*uniform-1`; retain optional released mode |
| Discount / tau | .99 /.01 |
| Measure / coordinate / actor LR | 1e-4 /1e-4 /1e-4 |
| Orthogonality coefficient | 1000 |
| Actor type / BC coefficient | ddpgbc /.3 |
| Actor width / hidden / embedding layers | 1024 /1 /2 |
| Full inference | DGD=true, normalized coordinate, 5120 coordinate updates |
| Dual network | width256, hidden layers2; hinge coefficient5 is inactive |
| Actor adaptation | 512 steps after full coordinate inference, exactly once |
| Training reward / flow | None /none |
| Environment tasks | `antmaze-medium-singletask-task{1..5}-v0` |
| Evaluation | 10 episodes/task every100k; 500/task at saved500k endpoint |
| Episode horizon / workers | 1000 /4 |
| Goal and inference sampling | Seed0 schedule, full29-D goal saved per task; no 2-D oracle substitution |
| Checkpoint / log intervals | 100k /1000 |
| Reporting | Success per task; average tasks within seed, then Student-t95% CI across3 seeds |

Ruling: this is an **untuned AntMaze transfer**, not reference parity. Archive settings
are retained except the explicitly documented AntMaze actor/BC choice, discount.99
(current repository AntMaze convention), and corrected bounded proto actions. The
archive sampler is `2*(uniform-1)` and its `proto_table_path` knob is unused; setting a
path is not a correction. Apply the bounds through the existing codebook seam or an
explicit post-construction table replacement, with a regression test. Do not modify
the affine measure, normalized immediate source, regularizer, or inference objectives.

Full inference includes task-time optimization; report its5120+512 updates separately
from the500k representation-training budget. Each task starts from the same saved
pre-inference agent, with representation parameters fixed. Do not call distillation
again after `infer_w_goal`, which already performs it.

## Task 1: data verification

Owner: data agent. Outputs: campaign `data/` and `data/provenance/`.

- [x] Download only the two identified Drive assets; retain listing/headers/checksums.
- [x] Check ZIP integrity, safe NPZ loads, finite values, action bounds and schema.
- [x] Verify raw terminal sentinels every201 rows; use OGBench's native transition
  construction, never pair across sentinels or shift explicit successors twice.
- [x] Confirm29-D observation/goal,8-D action, five tasks and1000-step runtime horizon.
- [x] Persist structured provenance/schema JSON alongside the raw source evidence.

## Task 2: minimal launch adapter

Owner: implementation agent. Create `tools/affine_stitch/`,
`tests/test_affine_stitch.py`, `scripts/slurm/affine_stitch.sbatch` only.

- [x] Write focused failing tests for terminal-row pairing/reward exclusion, local
  inference RNG isolation, bounded proto actions, and checkpoint/source/data binding.
- [x] Materialize archived `affine_psm.py` as `affine_agent.py` and its required utils;
  never import the live registry or all archived baselines.
- [x] Load the exact checksummed paths through native OGBench parsing. Training batches
  contain only observations/actions/next_observations/stable index. Validation remains
  separate and is never sampled for updates or goal inference.
- [x] Seed environment and action_space before constructing each full goal; save the
  goal vector and check fixed goal XY across evaluation episodes. Use an independent
  dataset RNG for each task's inference, without advancing training's sampling stream.
- [x] Save exclusive checkpoint/run/source/data artifacts. Bind source/data hashes into
  flags, and flags into checkpoints. Final evaluation must restore the saved checkpoint.
- [x] Support smoke and train through the same runner, plus strict final aggregation.
- [x] Run focused tests, lint and shell syntax; persist XML and preflight JSON.

Expected synthetic fixture: raw observations `[0,1,2,10,11,12]`, terminal mask
`[0,0,1,0,0,1]` gives current rows `[0,1,10,11]` and successors `[1,2,11,12]`.
Changing a terminal sentinel, input checksum, manifest, or proto range must be detected.

## Task 3: independent review and launch

Owner: parent and separate reviewer. No change to Walker jobs2505904/2505905/2505906.

- [x] Review data/goal/inference semantics and saved-source reproducibility.
- [x] Print the full resolved hyperparameter table before GPU launch.
- [x] Run200 training updates with the full model/buffer and full5120+512 goal inference
  budgets; evaluate2 episodes for each of the five tasks. Require finite losses/actions,
  unchanged representation during inference, valid checkpoint restoration, and correct goals.
- [x] Estimate train+inference+evaluation time; submit seeds0/1/2 only after smoke passes.
  Use one H100 per run, and schedule final500-episode/task evaluation plus aggregation.
- [x] Inspect each run's own flags and initial finite training metrics (or record PENDING
  accurately if scheduling prevents startup); persist job IDs and a dated HANDOFF entry.

Expected outcomes before results: short smoke policy success may be zero; this does not
fail the wiring gate. Full-affine performance on stitch is unvalidated. A poor500k result
is evidence about this transferred recipe, not proof that every affine PSM is incapable
of stitching. No best-seed/early-checkpoint substitution for the registered endpoint.

## Execution record

Campaign: `outputs/affine_antmaze_stitch_20260913/`. Parent verification:12 focused
tests passed in27.65s; Ruff and shell syntax passed. XML, full-size CPU preflight and
checkpoint roundtrip are persisted in the campaign. Source snapshot `code/` manifest
SHA256 is `b8138ff7e12fb88a47a2ba61bff18ae2634854fca4b5acc37b48e7bca1c8c0de`.
Independent read-only review found no critical or important adapter issue.

The fixture tests exposed an archived numerical edge case: with width8,d4,rank2,
an exactly zero x-side feature produces NaN normalization gradients in the first
training update despite a finite forward loss. This precedes goal inference. The
width8 reproducer is retained as an expected rejection test; width32 passes full
inference repeatability and exactly-once actor adaptation. Evidence is saved in
`tiny_feature_diagnostic.json`. No archived normalization or objective was changed.
The actual full-size configuration still requires the GPU gate; tiny-fixture success
alone is not solver validation for this campaign.

Diagnostic smoke job2506198 completed0:0 in3m32s. Its restored200-update checkpoint
passed all five full5120+512 inference/rollout gates, with unchanged representation
and identical full goals to CPU preflight. Flags and source/data binding match.
At70.633 updates/s,500k training alone projects1.966h; including periodic and final
inference/evaluation by short-run extrapolation projects2.757h, not a runtime guarantee.

Production submitted before2026-09-12 23:56:49 UTC: seed0 job2506231, seed1 job2506233,
seed2 job2506232, one H100 and24h each. Aggregate job2506237 depends on all three jobs
successfully finishing training plus final evaluation. Source is the same `code/`
snapshot as the passing smoke. Checkpoint state is saved but interrupted-training
resume is not supported by this launch adapter. Live startup checks belong in the
campaign `postlaunch_validation.json` and newest HANDOFF entry.

Startup validated at2026-09-12 23:59:25 UTC: all three jobs RUNNING, seed0/1/2 logged
6k/5k/5k updates with finite metrics. Their declared and resolved configs, exact input
paths, bounded proto tables,500k budgets, final500-episode settings, and source/data
hashes all match the passing smoke/protocol. Aggregation is PENDING on dependencies.

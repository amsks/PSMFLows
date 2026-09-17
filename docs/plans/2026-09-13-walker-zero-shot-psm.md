# Walker zero-shot PSM implementation and launch plan

> **For agentic workers:** Use the subagent-driven-development review workflow for
> the independently owned data and runner work. Preserve uncommitted user work;
> do not commit or restore archived agents into the live registry.

**Goal:** Download ExORL Walker RND, run our archived raw-action zero-shot PSM port,
and evaluate stand/walk/run/flip without any flow or task-reward training.

**Architecture:** A checksummed execution snapshot contains the archived PSM agent
and its matching utilities. A small Walker-specific loader and runner supply aligned
transitions, deterministic reward relabeling, checkpointed training, and evaluation.
The RLU-derived full-affine agent is a separate method and is not substituted here.

**Tech stack:** Existing Python/JAX/Flax environment, dm_control/MuJoCo, NumPy,
Slurm H100. Torch is used only to construct the reference proto-action lookup table.

**Spec:** User's September 13 request and
[interface audit](../design/2026-09-13-psm-interface-audit.md), with the explicit
source-lineage and sampler corrections below.

## Global constraints and declared recipe

| Setting | Primary baseline |
|---|---|
| Agent source | archive commit `ee0e1746b36f48300eee06ad02e66b16f46851ff`, raw-action `PSMAgent` |
| Official comparator source | `agarwalsiddhant10/PSM`, revision `b1a2e7f388f789a0d6abaabd317692c1f50f42b2` |
| Data | Walker RND, first 5,000 numeric episode IDs, 5,000,000 transitions |
| Training seeds / endpoint | 0, 1, 2 / 2,000,000 updates, no automatic extension |
| Batch / feature dimension | 1,024 / 128 |
| Discount / target Polyak rate | .98 / .01 |
| Adam learning rates | phi, proto successor, task successor, actor: 1e-4 |
| Orthogonality coefficient | 1 |
| Successor ensemble / target pessimism | 2 / 0 |
| Actor pessimism / BC coefficient | .5 / 0; pure-Q TD3-style actor, no Q normalization |
| Actor noise std / clip | .2 / .3 |
| Phi architecture | width 256, 2 hidden layers, state input, no encoder |
| Successor architecture | width 1,024, 2 embedding layers, 1 hidden layer |
| Actor architecture | width 1,024, 2 embedding layers, 1 hidden layer |
| Task-vector sampling | .5 feature mixture, .5 Gaussian; sqrt(128)-norm |
| Proto policies | 16-bit codes; exact Torch-seeded table, 85,536 rows x 6 actions |
| Primary proto action law | `2*rand-1`, uniform in [-1,1); explicit correction to released sampler |
| Training rewards | None, including collection rewards; all environment discounts verified one |
| Evaluation tasks | stand, walk, run, flip; raw observations 24, actions 6 |
| Task inference | Same 10,000 sampled next-state/physics rows for all tasks and seeds, eval seed 0 |
| Periodic checkpoint / eval / log | Every 100,000 / 100,000 / 1,000 updates |
| Periodic / final evaluation | 10 / 500 episodes per task; 1,000 environment steps, no action repeat |
| Episode workers | 4; same reset-seed schedule across training seeds |
| Reporting | Average tasks within seed, then mean and Student-t 95% CI across training seeds |

Ruling: use a **sampler-corrected standardized baseline**, not an exact published
replication. The release samples `2*(rand-1)` in [-2,0), passes it unclipped to the
proto TD target, retains only 10% of supplied replay, and defaults to 3M updates.
We retain the exact released lookup artifact and selectable mode, but do not use it
for the primary valid-action experiment. Our numerical learning rates/architecture
follow released code; data and update budget follow the paper. If this recipe fails,
it cannot alone refute the paper or identify a port defect.

Ruling: start with **plain zero-shot PSM**, as the user requested a Walker positive
control. The archived RLU-derived full-affine implementation has goal-conditioned
constrained inference only. Dense reward integration is a separate required change;
an identity decoder or a single goal is not an equivalent implementation.

## Task 1: acquire and verify data

Owner: data agent. File: `tools/walker_psm/download_prepare.py`.

- [x] Resolve the official download script and fetch the Walker RND ZIP into the
  new campaign directory; retain the original archive.
- [x] Reject unsafe ZIP paths, verify CRCs, and validate every episode's shape,
  numeric finiteness, environment discounts, and checksum.
- [x] Preserve official pairing `obs[t-1], action[t], obs[t], physics[t]` for
  t=1..1000; do not train on collection rewards.
- [x] Generate and verify both bounded and released proto tables with separate
  filenames and provenance; never replace an existing table.

## Task 2: isolated training and evaluation runner

Owner: runner agent. Files: `tools/walker_psm/{data,materialize,run,report}.py`,
`tests/test_walker_psm.py`, `scripts/slurm/walker_psm.sbatch`.

- [x] Add alignment, discount rejection, numeric episode order, and missing-data tests.
- [x] Build a full-size archived agent from an isolated temporary snapshot and check
  all four environments, episode lengths, actor action bounds, and checkpoint restore.
- [x] Validate both proto modes; focused runner tests returned 8 passed.
- [x] Bind source/data manifest hashes into flags, and hence into checkpoints;
  independent review identified that comparing replaceable sidecars alone was weaker.
- [x] Review the implementation independently, fixing launch-blocking findings.
- [x] Run the focused tests and lint; persist their results before the final snapshot.

Commands:

```bash
JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m pytest tests/test_walker_psm.py -q -x \
  --junitxml=outputs/walker_psm_20260913/pytest_runner.xml
.venv/bin/ruff check tools/walker_psm tests/test_walker_psm.py
bash -n scripts/slurm/walker_psm.sbatch
.venv/bin/python -m tools.walker_psm.materialize \
  --destination outputs/walker_psm_20260913/code
```

## Task 3: smoke, launch, and verify

Owner: parent. Outputs: `outputs/walker_psm_20260913/`; lab record: `docs/HANDOFF.md`.

- [x] Print the complete resolved configuration and run the exact full-size path
  for 200 updates on one H100, with all four tasks and 2 episodes/task.
- [x] Require finite losses, a restorable checkpoint, successful evaluation, and
  matching saved flags. Measure steady-state update time to size the job walltime.
- [x] Submit seeds 0/1/2 only after that gate; inspect each run's own flags and
  advancing finite training metrics. Schedule strict endpoint aggregation.
- [x] Record source/data manifests, commands, job IDs, current states, and limitations
  in HANDOFF. Do not call the smoke's two-episode scores benchmark performance.

Expected before results: wiring tests should pass and smoke losses remain finite;
an untrained/200-update policy may perform poorly and this is not a failure criterion.
The fixed-endpoint benchmark should ideally approach the published Walker task profile
(stand/walk stronger than run). Either success or failure remains informative about
this baseline, but neither directly validates the RLU full-affine method or isolates
the frozen-flow interface. No 500k peak or favorable seed replaces the fixed endpoint.

## Execution notes

Initial smoke job 2505858 uses snapshot `outputs/walker_psm_20260913/code`.
An independent reviewer verified real observation/physics compatibility on five
dataset rows in all four tasks (maximum absolute error 3.57e-7) and found no
algorithm/task-alignment defect. It requested stronger checkpoint-to-manifest
binding before production. That metadata-only fix receives a new snapshot and a
final exact-code-path smoke; the initial artifacts remain intact.

Final snapshot `code_v2`: smoke 2505882 completed0:0 in2m19s; its own flags,
200-update saved-checkpoint evaluation, and source/data hash binding passed checks.
Production jobs are seed0=2505905, seed1=2505904, seed2=2505906, one H100 each,
24h walltime; all observed RUNNING. Aggregate job2505911 depends on all three
training/evaluation jobs succeeding. Post-launch saved-flags verification passed at
2026-09-12 23:03:36 UTC: all declared settings and common data/source hashes matched;
seeds0/1/2 had reached5k/9k/5k with finite logged metrics. Evidence is in
`outputs/walker_psm_20260913/postlaunch_validation.json`. The CLI does not resume
interrupted training; the measured
roughly4.2h update-only estimate leaves substantial room inside the24h allocation.

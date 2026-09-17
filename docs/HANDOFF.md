---
marp: true
title: PSMFlows — Lab Record
theme: default
paginate: true
---

<!-- _class: lead -->

# PSMFlows — Lab Record

Current work: **PSMFlow v1** (`docs/plans/2026-07-20-psmflow-v1.md`) — **Tasks 1–8 code
complete**; the pipeline is now operator-driven (GPU runs + gates), with one open method
call on the D3 ESS gate. The **affine PSM** cube push (07-22 → 07-26, below) is **PARKED**,
not closed: the 07-26 entry's Findings 3 and 4 (rectangular measure mesh, split
x-/(s,a)-branch) are still the open leads if we return to it. The older bilinear-PSM parity
hunt (2026-07-07 → 07-15) is **CLOSED** — see the 07-13 entry and `PAPER/RESEARCH_NOTE.md`
§4: no code bug, the gap was seed variance + a training-budget ceiling.

Branch: `feat/inversion-integration` · Machine: `kisski` (GWDG, SLURM/H100) · prior: `midi-01` (UT CS)
Date: **2026-09-17** (latest) · prior: 2026-09-16, 2026-09-15, 2026-09-14, 2026-09-11, 2026-09-10, 2026-09-09, 2026-09-08, 2026-09-07, 2026-09-06, 2026-09-05, 2026-09-04, 2026-09-03, 2026-09-01, 2026-08-31, 2026-08-30, 2026-08-29, 2026-08-14, 2026-08-13, 2026-08-12, 2026-08-10, 2026-08-05, 08-04, 07-29, 07-28, 07-26, 07-15, 07-13, 07-07

---

## 2026-09-17 (eval) — psmgoal five-task result: negative

All 6 runs (jobs 2518992-97) finished at 1M steps, scale-anchored, TD stable throughout
(measure capped at 2D, `w_norm` 11.31). Five-task eval500 (500 episodes/task, 3 seeds):

| arm | 250k | 500k | 750k | 1M |
|---|---|---|---|---|
| nocon (constraint_coef 0) | 0.075 ±0.078 | 0.057 ±0.105 | 0.002 ±0.003 | 0.002 ±0.003 |
| con (constraint_coef 1) | 0.118 ±0.131 | 0.007 ±0.010 | 0.004 ±0.004 | 0.029 ±0.031 |

Best cell con 250k 0.118, at the BC level (0.111) and inside its CI; no cell reaches 0.284.
Pre-registered reading met: both arms <= 0.284 with stable TD. The goal-indexed affine
successor measure over flow latents does not carry a goal-reaching value on cube. The
constraint does not change the verdict (con ~ nocon within noise). Both arms decay from 500k
on; the 250k checkpoint acts better than 1M. Seed variance is large. Comparators: BC 0.111,
EMaQ 0.282, 2P-GPI 0.328, FB 0.496, HILP 0.742. Spec + telemetry: `docs/design/2026-09-17-psmgoal.md`.

Eval-pipeline note: the psmgoal eval watcher (`/tmp/psmflow_psmgoal/psmgoal_evals.sh`) submitted
nothing in 13h — it ran a stale copy, used `MODE=psmflow` (the checkpoints are `agent=psmgoal`,
which trips the eval's agent-name assertion), and relied on `--export=ALL` for `PSM_REPO` which
was not in its shell. Resubmitted directly with `MODE=psmgoal` and explicit `PSM_REPO`
(scratchpad `submit_psmgoal_evals.sh`), smoke-checked one JSON, then all 120. JSON names
`psmgoal_cube_{nocon,con}_sd00{k}_{ep}_task{t}.json` in `$PSM_DATA/logs`.

## 2026-09-17 — handoff: every arm's state, the psmgoal build in flight, and how to pick this up

Branch `fix/psmflow-paper-strict` (worktree `.claude/worktrees/psmflow-fix`; all SLURM jobs
import from it). Data root `$PSM_DATA=/mnt/home/amohan/psm-data`; eval JSONs in
`$PSM_DATA/logs`; SLURM logs in `$PSM_DATA/logs/slurm`. Cluster: KISSKI, partition
`kisski-inference`, account `general`, one GPU per job. Five-task means over 500 episodes per
task are the only numbers quoted.

**Correction to the 09-16 entry above.** The 24 phase-2 cancellations it records as "an
external interruption" were deliberate, user-approved cuts made in this session: the DSRL
variant (0.04–0.14 five-task on the 500k bases, 12 cells; probe 0.131) at 13:28; the 2M and
1.5M bases (joint phi already collapsed by 1.5M: sd0 task-3 0.026) at 13:40–13:48; the
50k/100k/250k/3M eval steps and phase-1 evals; and a STOP2M rule that cancelled each GPI run
once its `params_2000000.pkl` landed. Reasons and numbers: `docs/design/2026-09-15-psmflows-2p.md`
("2026-09-16" sections) and the watcher log `/tmp/psmflow_2p_main/p2_watcher.log`.

### Where the number stands

| arm | five-task cube | source |
|---|---|---|
| 2P-GPI, 500k basis, psi 500k (3 seeds) | 0.328 ± 0.264 (0.439 / 0.318 / 0.227) | `p2_gpi_phi500k_sd00k_500000_task*.json` |
| same, pooled with Task A (6 cells, step fixed in advance) | 0.324 ± 0.092 | design doc |
| 2P-GPI ladder, 500k basis, psi 500k / 1M / 2M | 0.328 / 0.250 / 0.274 | grid closed 09-17 00:34 |
| 2P-GPI ladder, 1M basis | 0.279 / 0.155 / 0.231 | basis age does not help |
| EMaQ (`bootstrap=gpi_argmax`, task-vector index), psi 250k / 500k / 750k / 1M | 0.233 / 0.173 / 0.243 / **0.282 ± 0.147** (runs complete; task 1 0.42-0.86, task 3 0.49-0.67, task 2 0.02-0.27) | `emaq_phi500k_sd00k_*_task*.json` |
| joint affine GPI (previous best) | 0.284 | results.md |
| BC / FB (TD-JEPA) / HILP (TD-JEPA) | 0.111 / 0.496 / 0.742 | |

Settled negative this week, with the reason on record: Eq. 10 coefficient inference on the
frozen affine head (linear objective, inert row multipliers, deployed as the fixed-index
mode; 0.000–0.012), RLDP basis (0.175 vs 0.320 frozen-own), 2P-DSRL, freeze-phi, basis
age, psi length. Memory files carry the one-line versions.

### What is running / in flight right now

- `emaq_phi500k` sd0/1/2, jobs 2518887/88/89, 1M psi steps, ~9 h in at 30 it/s; the 1M
  checkpoint lands imminently. `/tmp/psmflow_emaq/emaq_evals.sh` (pid 2746579, nohup) submits
  the five-task eval500 at 250k/500k/750k/1M as checkpoints land and exits on ALLDONE; its
  log is `/tmp/psmflow_emaq/emaq_evals.log`. If it is gone, resubmit by hand with
  `scripts/slurm/eval500.sbatch` (MODE=psmflow, ENVKEY=cube, RUN_DIR, OUT, RESTORE_EPOCH,
  ENV_NAME=cube-single-play-singletask-task{t}-v0). Design doc with the ladder:
  `docs/design/2026-09-16-emaq-backup.md`.
- **psmgoal build, in flight by a subagent** (uncommitted at the time of writing: new
  `agents/psmgoal.py`, `configs/agent/psmgoal.yaml`, edits to `agents/__init__.py`, `main.py`,
  `scripts/eval500.sh`, `tools/eval_checkpoint.py`, `utils/datasets.py`, `utils/psm_common.py`,
  `utils/psm_networks.py`). Spec: `docs/design/2026-09-17-psmgoal.md` — read it first; the
  notation collation behind it is `docs/design/2026-09-16-goal-indexed-measure.md`. The
  subagent was told: TDD with `tests/test_psmgoal.py` and `tests/test_hindsight_goals.py`, no
  A/beta split (general `phi_theta(s,u,s+)`, `b_theta(s,u,s+)` on the triple), `w*(g) =
  h_theta(g)/||h_theta(g)||`, per-triple softplus multiplier `l_theta` behind
  `constraint_coef`, EMaQ backup over 8 prior draws, goal-set inference at eval
  (`infer_eval_goals`), commit with no AI attribution, no SLURM launch. To check its state:
  `git log --oneline -5`, `git status --short`, then per file
  `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/test_psmgoal.py -q -p no:cacheprovider`
  (never the whole suite in one process). If the work is half done and untested, finish it
  against the spec's test list before anything else.

### psmgoal launched 2026-09-17 00:58 (commit `8e5a292`, smoke passed, table in the spec doc)

First launch (2518981-86) diverged — the measure had no scale anchor (positive column at
3e5 by 10k steps) — and was cancelled at ~15k steps; fix in commit `cfbb285` (phi and w*(g) on
the sqrt(D) sphere, b bounded by D tanh(b/D)), recorded in the spec doc. **Relaunched 01:11:**
`psmgoal_cube_nocon` sd0/1/2 = 2518992/93/94, `psmgoal_cube_con` sd0/1/2 = 2518995/96/97,
1M steps, save 250k, walltime 1-12:00:00. The cancelled runs' directories (jids 2518981-86)
still exist under the same groups; the newest directory per seed is the live one. Eval submitter `/tmp/psmflow_psmgoal/psmgoal_evals.sh`
(nohup; log `psmgoal_evals.log`; marks under `/tmp/psmflow_psmgoal/marks`) submits five-task
eval500 at 250k/500k/750k/1M per run; reports land as
`$PSM_DATA/logs/psmgoal_cube_{nocon,con}_sd00k_{step}_task{t}.json`. Results go into
`docs/design/2026-09-17-psmgoal.md` against its pre-registered readings.

### The launch gates, for reference (already passed for the runs above)

1. CPU smoke, 200 steps, exact flags (the subagent's report gives the command; pattern is the
   `main.py agent=psmgoal ...` line with `agent.flow_ckpt_path=$PSM_DATA/flow/cube-single-play
   agent.flow_ckpt_epoch=500000 agent.preimage_path=$PSM_DATA/preimages/cube-single-play.npz
   env_name=cube-single-play-singletask-v0 offline_steps=200 eval_episodes=0
   save_dir=$PSM_DATA/exp_smoke`). Confirm `flags.json` and that `train.csv` logs `psm_loss`,
   `obj`, `pen`, `viol_frac`, `mult_mean`, `backup_adv`, `w_norm` (must read 1.0),
   `td_target_absmean`.
2. Print the hyperparameter table (spec §Flags) in the design doc before submitting.
3. Submit 3 seeds × 2 arms with `scripts/slurm/train_psmflow.sbatch` and `AGENT=psmgoal`:
   `--export=ALL,PSM_REPO=<worktree>,PSM_DATA=/mnt/home/amohan/psm-data,OGBENCH_DATASET_DIR=/mnt/home/amohan/.ogbench/data,ENVKEY=cube,PREIMAGES=$PSM_DATA/preimages/cube-single-play.npz,AGENT=psmgoal,SEED=k,GROUP=psmgoal_cube_{nocon|con},STEPS=1000000,SAVE_INT=250000,EVAL_INT=50000,EVAL_EPS=50,LOG_INT=5000,EXTRA="agent.constraint_coef={0|1}"`,
   `--time=1-12:00:00`. Then re-read each run's `flags.json`.
4. Copy `/tmp/psmflow_emaq/emaq_evals.sh`, change `G=` to each group, run under nohup.
5. Pre-registered readings are in the spec; write results into the spec's doc.

### Paused

Rollout videos (`tools/eval_checkpoint.py` `video_episodes`/`video_dir`): tests written and RED
in `tests/test_eval_videos.py` (writer `write_render_videos` + `evaluate(...,
return_render_infos=True)` in `utils/evaluation.py`), implementation not started. Approved
render set: cube 2P-GPI `p2_gpi_phi500k/sd000` @500k tasks 1–5 ×3, cube BC tasks 2,4 ×3,
antmaze `affine_strict_antmaze_g99/sd002` @250k task 1 ×4, antmaze BC ×3, EMaQ sd0 @250k
tasks 2,4 ×2; contact-sheet PNGs (5 frames) per episode so the agent can read them.

### Working agreements from this session (the user was explicit)

- Only the zero-shot PSM policy's five-task number counts; BC and reward-labelled arms do not.
- Do not launch experiments unprompted; present the hyperparameter table and expected
  outcomes, get a yes, smoke, launch, re-read flags.json.
- Notation in every explanation: `u` action, `u'` policy index, `w(u')` coefficient,
  `phi(s,u,s+)`, `b(s,u,s+)`, `l(s,u,s+)` multipliers, `theta` for anything trained. Do not
  introduce other symbols, do not use the `A`/`beta` split or a separate state basis when
  writing the objective, do not add modelling the user did not ask for.
- Flat declarative reporting; numbers in tables; what it shows / what it does not show.

---

## 2026-09-16 — PSMFlows-2P: refit psi on a frozen basis; basis age and psi length both flat or negative

Branch `fix/psmflow-paper-strict`. Design doc: `docs/design/2026-09-15-psmflows-2p.md` (all
tables, per-seed cells, ladders and the cancellation record). Every number here was
recomputed from the eval JSONs in `$PSM_DATA/logs`; 500 episodes per task, five tasks,
`num_episodes` asserted at 500 in every file read.

**The recipe.** PSMFlows-2P is two phases on the frozen behaviour flow.

- Phase 1 trains the default affine agent end to end, phi and psi together.
- Phase 2 loads phi from a phase-1 checkpoint, freezes it, re-initialises psi and trains psi
  from scratch. Flags: `agent.phi_restore_path`, `agent.phi_restore_epoch`,
  `agent.train_phi=false`. Only the online phi parameters are read; no optimiser state, no psi.
- 2P-GPI is the paper-strict agent on the frozen basis (`psi_form=affine`,
  `policy_index=latent`, `acting=gpi`, `u_clip=3.0`).
- 2P-DSRL adds `policy_index=task_vector train_actor=true acting=actor actor_mode=dsrl_sac
  actor.bc_coeff=0.0 actor.target_entropy=-3.4657`.

Phase-2 seed k always uses phase-1 seed k's basis.

**Runs.** The plan was 2 variants x 4 bases x 3 seeds = 24 phase-2 runs at 3M psi steps. An
external interruption cancelled all 24 SLURM jobs on 2026-09-16 between 13:28:55 and
19:15:47 (`CANCELLED by 10025`). This record does not attribute the cancellations.

| arm | jobs | started | last psi step | 500-episode evals on disk |
|---|---|---|---|---|
| phase 1 joint, 2M | 2518506-08 | yes | 2.000M, COMPLETED | 500k, 1M (3 seeds); 1.5M (sd0, sd2) |
| 2P-GPI 500k basis | 2518522/29/36 | yes | 2.000M / 2.015M / 2.025M | 50k, 100k, 250k, 500k, 1M, 2M |
| 2P-GPI 1M basis | 2518640/47, 2518633 | yes | 2.005M / 2.015M / 2.000M | same, 2M only sd1 and sd2 |
| 2P-GPI 1.5M basis | 2518783/95, 2518776 | sd0, sd2 | 690k / -- / 725k | none |
| 2P-GPI 2M basis | 2518861/65, 2518858 | no | -- | none |
| 2P-DSRL 500k basis | 2518523/30/37 | yes | 1.380M / 1.415M / 1.385M | 50k, 100k, 250k, 500k |
| 2P-DSRL 1M basis | 2518641/48, 2518634 | yes | 965k / 940k / 980k | 50k, 100k, 250k |
| 2P-DSRL 1.5M basis | 2518784/96, 2518777 | sd0, sd2 | 540k / -- / 550k | none |
| 2P-DSRL 2M basis | 2518862/66, 2518859 | no | -- | none |

Basis checkpoints at 500k, 1M, 1.5M and 2M remain on disk for all three phase-1 seeds, as do
all phase-2 checkpoints written before the cancellation.

**Consolidated five-task cube.** Mean over the five tasks per seed, then a t interval over
the seed means (df 2; df 1 on rows marked n=2).

| arm | 50k | 100k | 250k | 500k | 1M | 2M |
|---|---|---|---|---|---|---|
| phase 1 joint | -- | -- | -- | 0.261 ± 0.206 | 0.192 ± 0.381 | -- |
| 2P-GPI, 500k basis | 0.173 ± 0.246 | 0.093 ± 0.111 | 0.287 ± 0.317 | **0.328 ± 0.264** | 0.250 ± 0.418 | 0.274 ± 0.192 |
| 2P-GPI, 1M basis | 0.215 ± 0.242 | 0.059 ± 0.108 | 0.242 ± 0.310 | 0.279 ± 0.137 | 0.155 ± 0.130 | 0.250 ± 0.719 (n=2) |
| 2P-GPI, task A shared basis | 0.247 ± 0.084 | 0.100 ± 0.084 | -- | 0.320 ± 0.218 | -- | -- |
| 2P-DSRL, 500k basis | 0.125 ± 0.039 | 0.100 ± 0.121 | 0.072 ± 0.035 | 0.090 ± 0.111 | -- | -- |
| 2P-DSRL, 1M basis | 0.129 ± 0.029 | 0.104 ± 0.014 | 0.088 ± 0.038 | -- | -- | -- |

The phase-1 joint run at 1.5M reads 0.182 ± 1.428 on two seeds. References: joint affine GPI
pooled 0.284, BC control 0.111, FB 0.496.

**Reading 1: refit does not beat joint training.** 2P-GPI on a 500k basis reads 0.328 ± 0.264
at 500k psi. The joint runs that produced those bases read 0.261 ± 0.206 at their own 500k
step, and the older pooled joint reference is 0.284. The intervals overlap.

**Reading 2: longer psi training does not help.** Both bases read their highest five-task
mean at 500k psi steps and fall after it (500k basis 0.328 -> 0.250 at 1M; 1M basis 0.279 ->
0.155 at 1M). All three 1M-basis seeds fall from 500k to 1M; two of three 500k-basis seeds
fall. `td_target_absmean` and `psm_loss` rise with psi steps in all six runs, in the seeds
that rose and the seeds that fell, so neither gives a reward-free stopping rule. Three seeds
do not establish one either way.

**Reading 3: basis age changes nothing beyond seed spread.** 500k vs 1M basis reads 0.328 vs
0.279 at 500k psi, 0.250 vs 0.155 at 1M psi, 0.274 vs 0.250 at 2M psi. The older basis is
lower at all three steps and every interval covers the other cell. The 1.5M and 2M rungs are
not answerable: no 500-episode evals exist for any 1.5M- or 2M-basis phase-2 run.

**2P-DSRL, partial.** At or below the BC control 0.111 at every step measured, except the two
50k cells (0.125 and 0.129). It is below 2P-GPI in 5 of the 7 matched cells; the two
exceptions are the 100k psi cells, where both arms sit at their own minimum. Successes are
concentrated on task 1 (0.23-0.41) and task 3 (0.09-0.33); tasks 2, 4 and 5 read 0.00-0.03 in
every cell. This matches the pre-registered expectation and every earlier measure-as-critic
actor arm.

**Two things found while checking the data.**

1. Phase-1 seed 0 diverged after about 1.1M steps. Its phi Gram condition number goes 5.67 at
   600k, 11.9 at 1M, 587 at 1.2M, 4.58e5 at 1.4M, 1.28e14 at 1.8M; `psm_loss` reaches 6.73e10
   at 2M. Seeds 1 and 2 stay at Gram condition 4-6 through 2M. The seed-0 bases at 1.5M and
   2M are degenerate, so the basis-age axis could not have been read past 1M on that seed
   even without the cancellation.
2. Phase-1 seed 1 reproduces `affine_strict_cube` seed 1 exactly (`psm_loss` 1588.7823 at 50k
   in both), so `p2_gpi_phi500k` sd1 and `cube_frozen_own_phi_gpi` sd1 are one run under two
   names, with 15 bit-identical eval cells. The two groups cannot be pooled as independent.

Commits: this entry and the design-doc results section. Prior 2P commits: 583fcbd (the
`bootstrap=gpi_argmax` seam and the emaq launch), dc678d5 and 0a6e277 (the 500k- and
1M-basis ladders as they stood mid-schedule), d53d118 (emaq 250k).

---

## 2026-09-15 (evening) — handoff: where the PSMFlow failure sits, what is settled, what is in flight

Branch `fix/psmflow-paper-strict`, worktree
`/mnt/home/amohan/git/Austin/PSMFLows/.claude/worktrees/psmflow-fix`, HEAD 19afb30. This
entry is the resume point. It links to the entries below for their tables and repeats no
table already recorded. Sources: the three entries below (09-15 afternoon, 09-15 00:08,
09-14 night), `docs/design/2026-09-15-policy-family-diversity.md` (3e579dc),
`docs/design/2026-09-14-flow-psm-dsrl-paper-versions.md`,
`docs/design/2026-09-14-paper-vs-code-diagnosis.md`, `docs/design/2026-09-15-rldp-phi-basis.md`
(untracked at write time), `git log daa0268..HEAD`, `squeue -u amohan` at about 20:10.

**1. State of the branch.** 30 commits since the branch point daa0268 (`feat/inversion-integration`
HEAD), 30 files changed, +6384 / −66 lines, of which `docs/HANDOFF.md` is +718. Nothing is
pushed. Grouped by kind:

| kind | commits |
|---|---|
| diagnosis and design docs | 98cbce5 (paper-vs-code diff), 5d06aa3 (paper versions), 82429fa (five-task rule), b9f8e1a, 9b6ac38 (measure vs scalar Q), b470f47, 1d75240 (fixes launched), 43295b3 (policy-side), 3e579dc (diversity table) |
| tools | 2c7189b, 5d06aa3 (`relabel_reward_rhat.py`), 76eb9cf (`utils/psm_proto.py`), cd371fc, 395475d, 92aba31 (`diag_measure_vs_scalar_q.py`), 28fac72 (`diag_policy_family_diversity.py`), 19afb30 (`pretrain_rldp_phi.py`) |
| seams in `agents/psmflow.py` / `main.py` | 2c7189b (`dataset.reward_override_path`), ebb2a5c (affine head under `policy_index=task_vector`), e373ab7 + abbd1fd (`proto.enabled`), 98539ec (`phi_restore_path`, `dsrl_na.ignore_masks`, `dsrl_na.reward_scale`), 2cbd723 (`psm_scalar_coef`, `psi_dueling`), a3fc920 (`measure_action_input=action` with the task-vector index) |
| arm launches | 98cbce5 (posterior latent), ebb2a5c (Section 10 free/affine), d51de4c (antmaze repeat), e373ab7 (reference-critic port), a3fc920 (Fix 3), d6c7bb1 (bc0 arms) |
| handoff entries | a6b39dd, 0df9dcd, d547171, 7135157 |

Working tree at write time: modified `agents/psmflow.py`, `configs/agent/psmflow.yaml`,
`tools/eval_checkpoint.py` (the `acting=fixed_coeff` seam, uncommitted); untracked
`tools/infer_policy_lagrangian.py`, `tests/test_infer_policy_lagrangian.py`,
`tests/test_psmflow_fixed_coeff.py`, `docs/design/2026-09-15-rldp-phi-basis.md`. Two other
agents own those files and were editing them when this entry was written. Commit only with
an explicit pathspec.

Merge note. The main checkout on `feat/inversion-integration` carries an uncommitted
697-line modification of `docs/HANDOFF.md` (the 09-13 and 09-14 entries) plus untracked
design and plan docs (`2026-09-08-critic-signal-and-dsrl-na.md`,
`2026-09-11-affine-action-conditioning.md`, `2026-09-11-freeze-phi-diagnostic.md`,
`2026-09-13-psm-interface-audit.md`, three plans). Both branches insert at the top of
`docs/HANDOFF.md`, so the merge will conflict there; the resolution is to keep both sets of
entries in date order.

**2. The finding chain.**

1. The measure loss in `agents/psmflow.py` is the paper's loss; the code-vs-paper deviations were listed and each was measured or shown rank-preserving (`docs/design/2026-09-14-paper-vs-code-diagnosis.md`; posterior latent measured 09-14).
2. The flow, the inferred reward `r_hat = phi(s')^T w` and the preimages are cleared: a scalar Bellman critic on the scaled `r_hat` reaches the real-reward level on cube (0.867 vs 0.882, task 2) and antmaze (0.853 vs 0.979, task 1) (09-14 evening, 09-14 night).
3. The measure `psi^T w` used as the critic is where the number drops: every measure-critic arm on cube scores 0.04–0.28 five-task against BC 0.111 and FB 0.496, on both loss forms (Section 10, reference PSM proto stage) and every fix tried (09-15 00:08, 09-15 afternoon).
4. The measure fits its own Bellman equation (residual 4% of its spread) but its readout is close to flat along the action latent and moves 4–193 times more with the policy index; across states it agrees with the scalar critic at Spearman 0.10, within a state at 0.12 (design doc, "Diagnostic: measure value vs scalar Q" and "Policy-side diagnostic").
5. The policies the index ranges over (one fixed noise vector repeated at every step) are distinct from each other and coherent, but all of them score below BC and their state clouds drift off the data's states, so the family GPI selects from holds no good member (`docs/design/2026-09-15-policy-family-diversity.md`).

**3. Consolidated results: every five-task cube number produced on this branch.** 500
episodes per task, 500k checkpoint, t interval over 3 seeds (df 2) unless stated. BC and
FB/HILP are the references the method has to beat.

| group | what | five-task mean ± CI | entry |
|---|---|---|---|
| BC control | frozen flow alone | 0.111 | 09-11 |
| `affine_strict_cube` | affine GPI, the default agent (300k–500k window) | 0.284 | 09-06 / `docs/tables/results.md` |
| FB | TD-JEPA Table 1 | 0.496 | — |
| HILP | TD-JEPA Table 1 | 0.742 | — |
| `cube_affine_posterior_u` | affine GPI, posterior-sampled dataset latent | task 2 only, 18 cells 250k–500k: 0.252 ± 0.057 vs point control 0.424 ± 0.076; not a five-task number, not quotable | 09-14 |
| `cube_sec10_free_actor` | Section 10 actor, free psi | 0.171 ± 0.027 | 09-14 evening |
| `cube_sec10_affine_actor` | Section 10 actor, affine psi (the template for the fixes) | 0.246 ± 0.041 | 09-14 evening |
| `cube_psmref_actor` | reference PSM critic (proto stage) on latent inputs | 0.156 ± 0.023 | 09-15 00:08 |
| `cube_fix1_scalar_tc` | task-conditioned scalar critic on frozen phi, DSRL-NA actor | 0.044 ± 0.014 | 09-15 afternoon |
| `cube_fix2_scalar_only` | Section 10 affine + projected Bellman grounding (`psm_scalar_coef=1`) | 0.233 ± 0.031 | 09-15 afternoon |
| `cube_fix2_scalar_dueling` | + dueling head (`psi_dueling=true`) | 0.219 ± 0.073 | 09-15 afternoon |
| `cube_fix3_action_measure_actor` | Section 10 affine, measure at the decoded action | 0.217 ± 0.008 | 09-15 afternoon |
| `cube_sec10_affine_bc0` | Section 10 affine, `actor.bc_coeff=0` | 0.038 ± 0.072 | 09-15 afternoon |
| `cube_fix2_dueling_bc0` | Fix 2 dueling, `actor.bc_coeff=0` | 0.003 ± 0.014 | 09-15 afternoon |

Single-task cube rows that belong with the table (task 2, 500 episodes, 500k, 3 seeds):
`cube_dsrlna_rhat_scaled` (scalar critic on scaled `r_hat`) 0.867 ± 0.153 against the
real-reward control 0.882 ± 0.121; `cube_dsrlna_rhat_frozen` (raw-scale `r_hat`) 0.010 ± 0.013
(09-14 evening).

Antmaze-medium-navigate rows (500 episodes per cell, 500k, 3 seeds):

| group | what | number | entry |
|---|---|---|---|
| BC control | frozen flow alone | task 1: 0.072 | `docs/tables/results.md` |
| `affine_strict_antmaze_g99` | affine GPI, discount 0.99 | task 1, 30-cell ladder: 0.294 ± 0.070 | 09-07 |
| `antmaze_sec10_affine_actor` | Section 10 actor, affine psi | five-task: 0.105 ± 0.025 | 09-14 night |
| `antmaze_dsrlna_rhat_scaled` | scalar critic on scaled `r_hat` | task 1: 0.853 ± 0.334 (control `dsrlna_antmaze` 0.979 ± 0.008) | 09-14 night |
| FB / HILP | TD-JEPA Table 1 | five-task: 0.730 / 0.836 | — |

**4. Mechanism tests.** Each row names the JSON that holds the full table.

| test | what was measured | numbers | source |
|---|---|---|---|
| scalar critic on `r_hat` | DSRL-NA scalar Bellman critic fed the scale-matched inferred reward, same phi, w and flow as the measure arms | cube task 2 0.867 vs real 0.882; antmaze task 1 0.853 vs real 0.979 | 09-14 evening, 09-14 night |
| measure vs scalar Q (`affine_strict_cube/sd001` vs `cube_dsrlna_rhat_scaled/sd001`, 10k rows x 64 latents) | Bellman residual of each critic on its own target; agreement of the two values at the same (s, u) | measure residual 4.2% of its spread, scalar 4.1%; across states Pearson 0.05 / Spearman 0.10; per-state Spearman over 64 u mean 0.12, 39% of states negative; measure argmax in the scalar top-8 on 33% of states (chance 12.5%); success rows at chance (rho 0.01) | `$PSM_DATA/logs/diag_measure_vs_scalar_q_cube_sd001.json`; design doc "Diagnostic: measure value vs scalar Q" |
| policy-side (64 x 64 grid of action latent x index, 500 states, three checkpoints) | share of the readout's within-state variance on the index slot vs the action slot; whether different indices rank the actions alike | index share 0.993 (GPI) / 0.917 (Section 10 affine) / 1.000 (bc0); action share 0.002 / 0.079 / 0.000; std ratio index/u 36.2 / 4.21 / 193; rank correlation across u between indices 0.29 / 0.92 / 0.35 | `$PSM_DATA/logs/diag_policy_side_cube_{gpi,sec10,sec10bc0}_sd001.json`; design doc "Policy-side diagnostic", block G |
| bc0 actor and the latent box | where the free actor's latent sits and how both critics score it | `cube_sec10_affine_bc0/sd001` actor latent at `|u|_inf = 3.0` (p90 and max), `|u|_2` 5.88 of the 6.7 the box allows, 14% of components at the clip; measure scores it +3.45 panel-std, scalar critic +2.54; on success rows the scalar critic scores it −0.74 (qa) and −2.23 (qw) panel-std | same JSONs, block H; 09-15 afternoon (b) |
| policy-family diversity (`affine_strict_cube/sd001`, 16 members x 20 episodes, task 2) | how different the state clouds of fixed-noise policies are, against a resampling floor (bc) and an off-support reference (random actions) | noise_index pairwise MMD² raw 0.2441 vs bc floor 0.0069 (35x); linear classifier 0.859 (chance 0.0625); latent explains 0.510 of within-state action deviation; task-2 success 0.006 vs bc 0.084; MMD² vs pooled bc 0.137 vs goal-directed DSRL-NA 0.024 and random actions 0.053 | `$PSM_DATA/logs/diag_policy_family_diversity_cube.json`; `docs/design/2026-09-15-policy-family-diversity.md` |

Readings of the diversity table that correct the afternoon entry's (d). The fixed-noise
policies are distinct and coherent. All of them are bad on the task. Their state clouds sit
farther from the data's states (0.137) than a goal-directed policy's (0.024) or uniform
random actions' (0.053). The in-support guarantee holds per action and fails per trajectory.
"Random actions" is not a diversity ceiling on cube: sixteen seeds of one action
distribution have pairwise MMD² at the bc floor. The index-slot dominance of the measure
is real and is what the policy-side diagnostic measured.

**5. Two papers read today, not yet in any doc.**

PSM (arXiv 2411.19418), Sec. 5.3, Eq. 10. Its test-time inference does not search the
trained family. It optimises the policy coefficient w over the whole affine set of valid
measures, with the constraint that `Phi w + b` is non-negative on dataset (s, a), solved by
gradient descent-ascent on a batch of 10^4 transitions; then `Q* = M* r` and an argmax
(discrete) or DDPG (continuous). Our GPI only takes the argmax over the trained family's
coefficients `w(u')` at 64 prior draws. Eq. 10 has never been run here.

RLDP (arXiv 2603.15857). A policy-free basis phi is trained by latent-dynamics prediction
plus an orthonormality term and then frozen; an FB-style psi and actor are trained on top.
No scalar critic, no per-task refit. The paper's claim is that Bellman-trained bases lose
span under low coverage. It reports no OGBench numbers.

**6. In flight at write time.**

| arm | code | jobs / state | expected | where results land |
|---|---|---|---|---|
| Eq. 10 policy inference | `tools/infer_policy_lagrangian.py` (untracked), seam `acting=fixed_coeff` + `agent.fixed_index_coeff_path` (uncommitted diff in `agents/psmflow.py`, `configs/agent/psmflow.yaml`, `tools/eval_checkpoint.py`), tests `tests/test_infer_policy_lagrangian.py`, `tests/test_psmflow_fixed_coeff.py` | no job submitted; `$PSM_DATA/logs/eq10/` does not exist yet | the tool fits one coefficient c per (checkpoint, task) on 10k rows of `affine_strict_cube` in minutes; the five-task eval500 with `acting=fixed_coeff` follows, per task the usual eval cost | `$PSM_DATA/logs/eq10/infer_policy_lagrangian_cube_<tag>.json` (+ `.npz`), `$PSM_DATA/logs/eq10/eq10_<variant>_<tag>_task<t>.npz`; write-up in the design doc the Eq. 10 agent names (the tool docstring is the spec until then) |
| RLDP phi pretraining | `tools/pretrain_rldp_phi.py` (19afb30), `scripts/slurm/pretrain_rldp_phi.sbatch` | 2518382 (`rldp_phi_cube_H5`), 2518383 (`rldp_phi_cube_H1`), RUNNING since 19:57, 4 h limit; 1M steps, d 128, lambda 1, batch 1024, Adam 3e-4, at ~798 it/s; step 345k reached at 20:10, `params_250000.pkl` written | done at about 20:20; `params_1000000.pkl` in `$PSM_DATA/exp/PSMFLows/rldp_phi_cube/{H5,H1}_sd0/` | `rldp_curve.json` in the same dirs; `PRETRAIN_TABLE` / `CURVES` placeholders in `docs/design/2026-09-15-rldp-phi-basis.md` |
| `cube_rldp_phi_gpi` | template `affine_strict_cube` + `phi_restore_path=.../rldp_phi_cube/H5_sd0 train_phi=false` | not yet submitted; waits on the H5 checkpoint | 3 seeds x ~3 h 10 min once submitted; five-task eval500 after | `$PSM_DATA/exp/PSMFLows/cube_rldp_phi_gpi/`, `$PSM_DATA/logs/cube_rldp_phi_gpi_sd00{0,1,2}_500000_task{1..5}.json`; design doc placeholders `JOBS_TABLE`, `LADDERS`, `FIVE_TASK`, `READING` |
| `cube_frozen_own_phi_gpi` (control) | template `affine_strict_cube` + `phi_restore_path=$PSM_DATA/exp/PSMFLows/affine_strict_cube/sd001_s_2491601.0.20260904_181115 train_phi=false` (re-read from `flags.json`: `psi_form=affine policy_index=latent acting=gpi train_actor=false discount=0.98 ortho_coef=1000`, 500k steps, eval every 50k) | 2518384 (sd0), 2518385 (sd1), 2518386 (sd2), RUNNING since 20:01, 12 h limit; CPU smoke `$PSM_DATA/logs/cpusmoke/cube_frozen_own_phi_gpi.out` passed | ~3 h 10 min each (the 09-14 posterior arm took 3h08–3h16), so about 23:15; five-task eval500 after | `$PSM_DATA/exp/PSMFLows/cube_frozen_own_phi_gpi/sd00{0,1,2}_s_251838{4,5,6}.0.20260915_2001*`, eval JSONs `$PSM_DATA/logs/cube_frozen_own_phi_gpi_sd00{0,1,2}_500000_task{1..5}.json`; same design doc |

Pre-registered readings, stated before results.

| arm | outcome | reading |
|---|---|---|
| Eq. 10 | c* stays inside the trained family's coefficient patch and scores about 0.284 | the family's own coefficients were already the optimum the basis allows; GPI was not leaving anything on the table |
| Eq. 10 | c* leaves the patch and scores above 0.35 | the basis spans a good policy that GPI over 64 prior draws never reached |
| Eq. 10 | c* leaves the patch and success collapses | the basis has to be retrained over a more diverse family before Eq. 10 can help |
| RLDP + frozen-own | both about 0.28 | the basis is not the limit |
| RLDP | above 0.35 | the basis geometry was the limit |
| frozen-own | below 0.28 | freezing phi itself costs, and the RLDP number has to be read against that |

**7. Open list, ranked.**

1. Eq. 10 result (row above). Decides whether the trained basis already contains a good policy.
2. RLDP and frozen-own results. Decides whether the basis geometry is the limit.
3. A policy family with good members: the index has to range over policies that are not one fixed noise vector. Candidates are an actor-conditioned, goal-conditioned or skill-conditioned flow as the family, or state-dependent index policies `u = f_k(s)` from fixed random networks (afternoon entry (d), fix A). Not run.
4. Task-vector coverage: whether the inferred eval `w` sits inside the training mixture of task vectors (32 projected `N(0, I)` draws plus 32 projected `phi(s'_j)`). Not measured.
5. Section 10 with the DSRL box (`u_clip=1.5`) instead of 3.0, since the bc0 actors sit at the 3.0 corner. Not run.

Settled negative on cube, one line each, with the entry that holds the number: posterior
sampled dataset latent, 0.252 vs 0.424 task 2 (09-14); Section 10 free psi 0.171 and affine
psi 0.246 (09-14 evening); reference PSM critic on latent inputs 0.156 (09-15 00:08); Fix 1
task-conditioned scalar critic on frozen phi 0.044, Fix 2 scalar grounding 0.233 and dueling
0.219, Fix 3 measure at the decoded action 0.217, bc0 on the affine measure 0.038 and on the
dueling measure 0.003 (09-15 afternoon); raw-scale `r_hat` in the scalar critic 0.010 (09-14
evening). On antmaze: Section 10 affine 0.105 five-task (09-14 night). Earlier settled items
are in `docs/COMPENDIUM.md` and the 09-06 to 09-11 entries.

**8. How to resume.**

Tools (all write JSON through `report_out`; run with the flow, preimage and restore flags
the afternoon and 00:08 entries record):

| tool | what it does |
|---|---|
| `tools/diag_measure_vs_scalar_q.py` | blocks A–H: Bellman residuals, measure-vs-scalar agreement, policy-slot variance split, box exploitation, actor latent placement; takes a measure checkpoint and a `cube_dsrlna_rhat_scaled` checkpoint |
| `tools/diag_policy_family_diversity.py` | rolls out fixed-noise, bc, random, tilted-bc and goal-directed families and writes the MMD² / classifier / success table |
| `tools/relabel_reward_rhat.py` | writes the `r_hat = phi(s')^T w` reward file (raw or scale-matched) that `dataset.reward_override_path` consumes |
| `tools/infer_policy_lagrangian.py` | PSM Eq. 10 coefficient fit per (checkpoint, task); writes the npz `acting=fixed_coeff` deploys |
| `tools/pretrain_rldp_phi.py` | policy-free phi by latent-dynamics prediction; writes a checkpoint `phi_restore_path` loads |

Seams in `agents/psmflow.py` / `configs/agent/psmflow.yaml`, all OFF by default so the
default agent is unchanged: `proto.enabled` (reference PSM critic), `dataset.reward_override_path`
(`main.py`), `phi_restore_path` + `train_phi=false` (frozen external phi),
`dsrl_na.ignore_masks` and `dsrl_na.reward_scale`, `psm_scalar_coef`, `psi_dueling` +
`psi_dueling_samples`, `measure_action_input=action`, `acting=fixed_coeff` +
`fixed_index_coeff_path` (uncommitted). `tools/eval_checkpoint.py` reads the agent config
from the run's `flags.json`; pass a flag only to evaluate off a run's own config.

Eval rule: five tasks, 500 episodes per task, 500k checkpoint, mean over tasks, t interval
over 3 seeds. Single-task numbers are paired directions only and are not quoted as results.
BC control and FB beside every number.

Standing rules are in `~/.claude/projects/-mnt-home-amohan-git-Austin-PSMFLows/memory/MEMORY.md`:
five-task average only, no local GPU (sbatch, not tmux), pytest per file, no AI attribution
in commits, plain RL language, no rhetorical writing. The cluster and paths:
`PSM_DATA=/mnt/home/amohan/psm-data`, SLURM logs `$PSM_DATA/logs/slurm/`, CPU smokes
`$PSM_DATA/logs/cpusmoke/`.

Caveats. The in-flight rows are as of about 20:10 on 09-15; no in-flight arm has a number.
The Eq. 10 tool had no job at write time and its seam is uncommitted, so a resume should
`git status` first. The RLDP `H1` pretraining run has no stage-C arm planned in the design
doc; only `H5` feeds `cube_rldp_phi_gpi`. The diversity numbers are one checkpoint
(sd001), N = 20 episodes per member.

---

## 2026-09-15 (afternoon) — three fixes and two bc0 arms on cube: all settled negative; the measure's readout varies with the policy slot, not the action slot

Branch `fix/psmflow-paper-strict`, worktree `.claude/worktrees/psmflow-fix`. Six cube arms,
three seeds each, 500k steps, all COMPLETED and scored at 500 episodes per task. Design,
objects, per-arm flag diffs and the pre-registered criteria:
`docs/design/2026-09-14-flow-psm-dsrl-paper-versions.md`, sections "Fixes launched
2026-09-15", "bc_coeff = 0 arms" and "Policy-side diagnostic (2026-09-15)". Code: 98539ec
(Fix 1 seams: `phi_restore_path`, `dsrl_na.ignore_masks`, `dsrl_na.reward_scale`), 2cbd723
(Fix 2: `psm_scalar_coef`, `psi_dueling`), a3fc920 (Fix 3: the action-input measure allowed
with the task-vector index), d6c7bb1 (bc0 arms, `actor_u_norm` telemetry); docs b470f47,
1d75240. Diagnostics: cd371fc, b9f8e1a, 395475d, 9b6ac38 (measure vs scalar Q), 92aba31
(policy-slot and box-exploitation blocks).

**What was run.** Every arm: cube-single-play, flow `$PSM_DATA/flow/cube-single-play`
@500000, preimages `$PSM_DATA/preimages/cube-single-play.npz`, eval every 50k with 50
episodes, one GPU per seed. Flags re-read from each run's `flags.json` after the runs
finished; the diffs below are against `cube_sec10_affine_actor/sd000` (Section 10 affine:
`psi_form=affine policy_index=task_vector train_actor=true acting=actor actor_mode=ddpg
actor.bc_coeff=1.0 u_clip=3.0 batch_size=1024 lr_actor=1e-4 discount=0.98`) unless stated.

| group | change vs template | jobs | run dirs (`$PSM_DATA/exp/PSMFLows/<group>/`) |
|---|---|---|---|
| `cube_fix1_scalar_tc` | template `cube_dsrlna_rhat_scaled` (DSRL-NA dual scalar critic, `actor_mode=dsrl_sac`, `bc_coeff=0`, `batch_size=256`, `u_clip=1.5`, `policy_index=latent`): reward `0.01262 * phi(s')^T w` at a fresh `w` per row (`reward_source=synthetic_w task_conditioned=true`), phi restored from `affine_strict_cube/sd001 @500k` and frozen (`train_phi=false`), `ignore_masks=true` | 2518243, 2518244, 2518245 | `sd000_s_2518243.0.20260915_031441`, `sd001_s_2518244.0.20260915_031441`, `sd002_s_2518245.0.20260915_031441` |
| `cube_fix2_scalar_only` | `psm_scalar_coef=1.0` (projected Bellman equation of `psi^T z` added to the measure loss) | 2518249, 2518250, 2518251 | `sd000_s_2518249.0.20260915_034039`, `sd001_s_2518250.0.20260915_034039`, `sd002_s_2518251.0.20260915_034043` |
| `cube_fix2_scalar_dueling` | `psm_scalar_coef=1.0 psi_dueling=true psi_dueling_samples=8` (state tower plus zero-mean advantage over 8 prior latents) | 2518252, 2518253, 2518254 (the 2518246-48 dirs are the cancelled first submission, no checkpoint) | `sd000_s_2518252.0.20260915_034600`, `sd001_s_2518253.0.20260915_034559`, `sd002_s_2518254.0.20260915_034600` |
| `cube_fix3_action_measure_actor` | `measure_action_input=action` (measure at the decoded action, every latent query decoded through the frozen flow; guard relaxed in a3fc920) | 2518255, 2518256, 2518257 | `sd000_s_2518255.0.20260915_035530`, `sd001_s_2518256.0.20260915_035531`, `sd002_s_2518257.0.20260915_035531` |
| `cube_sec10_affine_bc0` | `actor.bc_coeff=0.0` | 2518258, 2518259, 2518260 | `sd000_s_2518258.0.20260915_040357`, `sd001_s_2518259.0.20260915_040357`, `sd002_s_2518260.0.20260915_040358` |
| `cube_fix2_dueling_bc0` | `psm_scalar_coef=1.0 psi_dueling=true psi_dueling_samples=8 actor.bc_coeff=0.0` | 2518261, 2518262, 2518263 | `sd000_s_2518261.0.20260915_040357`, `sd001_s_2518262.0.20260915_040358`, `sd002_s_2518263.0.20260915_040358` |

Eval JSONs: `$PSM_DATA/logs/<group>_sd00{0,1,2}_500000_task{1..5}.json`, 500 episodes
each, `restore_epoch=500000`. The `proto.*`, `phi_restore_*`, `psm_scalar_coef`,
`psi_dueling*` and `dsrl_na.{ignore_masks, reward_scale}` keys are absent from the
templates' `flags.json` and sit at their OFF values in every run where the table does not
name them.

**Five-task table: 500 episodes per cell, 500k checkpoint, t interval over 3 seeds (df 2).**

| group | seed | task 1 | task 2 | task 3 | task 4 | task 5 | seed mean | mean ± CI |
|---|---|---|---|---|---|---|---|---|
| `cube_fix1_scalar_tc` | 0 | 0.128 | 0.018 | 0.058 | 0.032 | 0.012 | 0.050 | |
| | 1 | 0.068 | 0.020 | 0.104 | 0.022 | 0.010 | 0.045 | |
| | 2 | 0.000 | 0.026 | 0.110 | 0.040 | 0.014 | 0.038 | **0.044 ± 0.014** |
| `cube_fix2_scalar_only` | 0 | 0.238 | 0.288 | 0.256 | 0.204 | 0.126 | 0.222 | |
| | 1 | 0.162 | 0.294 | 0.478 | 0.198 | 0.102 | 0.247 | |
| | 2 | 0.216 | 0.262 | 0.344 | 0.220 | 0.112 | 0.231 | **0.233 ± 0.031** |
| `cube_fix2_scalar_dueling` | 0 | 0.170 | 0.330 | 0.230 | 0.228 | 0.048 | 0.201 | |
| | 1 | 0.274 | 0.300 | 0.378 | 0.188 | 0.124 | 0.253 | |
| | 2 | 0.198 | 0.278 | 0.324 | 0.170 | 0.044 | 0.203 | **0.219 ± 0.073** |
| `cube_fix3_action_measure_actor` | 0 | 0.182 | 0.334 | 0.280 | 0.232 | 0.040 | 0.214 | |
| | 1 | 0.174 | 0.326 | 0.268 | 0.250 | 0.068 | 0.217 | |
| | 2 | 0.168 | 0.352 | 0.246 | 0.284 | 0.050 | 0.220 | **0.217 ± 0.008** |
| `cube_sec10_affine_bc0` | 0 | 0.010 | 0.002 | 0.040 | 0.000 | 0.000 | 0.010 | |
| | 1 | 0.104 | 0.000 | 0.068 | 0.002 | 0.000 | 0.035 | |
| | 2 | 0.286 | 0.004 | 0.048 | 0.000 | 0.004 | 0.068 | **0.038 ± 0.072** |
| `cube_fix2_dueling_bc0` | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | |
| | 1 | 0.000 | 0.000 | 0.050 | 0.000 | 0.000 | 0.010 | |
| | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | **0.003 ± 0.014** |

Task means: Fix 1 0.065 / 0.021 / 0.091 / 0.031 / 0.012; Fix 2 scalar-only 0.205 / 0.281 /
0.359 / 0.207 / 0.113; Fix 2 dueling 0.214 / 0.303 / 0.311 / 0.195 / 0.072; Fix 3 0.175 /
0.337 / 0.265 / 0.255 / 0.053; affine bc0 0.133 / 0.002 / 0.052 / 0.001 / 0.001; dueling
bc0 0.000 / 0.000 / 0.017 / 0.000 / 0.000.

References (same protocol): Section 10 affine `cube_sec10_affine_actor` 0.246 ± 0.041
(seeds 0.252 / 0.259 / 0.227, 09-14 evening); Section 10 free 0.171 ± 0.027; affine GPI
`affine_strict_cube` 0.284; reference-critic port `cube_psmref_actor` 0.156 ± 0.023
(09-15 00:08); BC control 0.111; FB (TD-JEPA Table 1) 0.496.

**In-loop ladder (50 episodes, task 2, `eval.csv` column `evaluation/success`).**

| group | seed | 50k | 100k | 150k | 200k | 250k | 300k | 350k | 400k | 450k | 500k |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `cube_fix1_scalar_tc` | 0 | 0.16 | 0.04 | 0.04 | 0.38 | 0.00 | 0.00 | 0.00 | 0.00 | 0.02 | 0.04 |
| | 1 | 0.00 | 0.06 | 0.04 | 0.26 | 0.04 | 0.10 | 0.00 | 0.04 | 0.00 | 0.00 |
| | 2 | 0.02 | 0.04 | 0.20 | 0.02 | 0.02 | 0.08 | 0.08 | 0.02 | 0.00 | 0.04 |
| `cube_fix2_scalar_only` | 0 | 0.30 | 0.28 | 0.30 | 0.42 | 0.28 | 0.28 | 0.22 | 0.30 | 0.30 | 0.26 |
| | 1 | 0.14 | 0.20 | 0.20 | 0.36 | 0.28 | 0.30 | 0.14 | 0.24 | 0.18 | 0.32 |
| | 2 | 0.16 | 0.30 | 0.42 | 0.32 | 0.28 | 0.34 | 0.32 | 0.42 | 0.28 | 0.32 |
| `cube_fix2_scalar_dueling` | 0 | 0.24 | 0.28 | 0.28 | 0.36 | 0.30 | 0.42 | 0.32 | 0.22 | 0.34 | 0.42 |
| | 1 | 0.18 | 0.24 | 0.26 | 0.40 | 0.26 | 0.14 | 0.32 | 0.26 | 0.16 | 0.28 |
| | 2 | 0.22 | 0.22 | 0.42 | 0.30 | 0.42 | 0.18 | 0.34 | 0.28 | 0.34 | 0.26 |
| `cube_fix3_action_measure_actor` | 0 | 0.20 | 0.14 | 0.30 | 0.36 | 0.24 | 0.22 | 0.18 | 0.30 | 0.30 | 0.34 |
| | 1 | 0.18 | 0.36 | 0.26 | 0.24 | 0.28 | 0.24 | 0.20 | 0.28 | 0.26 | 0.36 |
| | 2 | 0.14 | 0.34 | 0.22 | 0.24 | 0.20 | 0.34 | 0.38 | 0.28 | 0.38 | 0.42 |
| `cube_sec10_affine_bc0` | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.02 | 0.02 | 0.00 | 0.00 | 0.00 |
| | 1 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.04 | 0.02 | 0.02 | 0.04 | 0.00 |
| | 2 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.04 | 0.00 | 0.06 | 0.00 | 0.02 |
| `cube_fix2_dueling_bc0` | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| | 1 | 0.00 | 0.00 | 0.00 | 0.02 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| | 2 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| template `cube_sec10_affine_actor` | 0 | 0.28 | 0.38 | 0.18 | 0.34 | 0.30 | 0.24 | 0.24 | 0.30 | 0.30 | 0.24 |
| | 1 | 0.24 | 0.36 | 0.42 | 0.28 | 0.36 | 0.22 | 0.26 | 0.42 | 0.34 | 0.32 |
| | 2 | 0.26 | 0.30 | 0.38 | 0.30 | 0.24 | 0.26 | 0.12 | 0.36 | 0.28 | 0.32 |

The three Section 10 variants (Fix 2 both forms, Fix 3) move within 0.14–0.42 across the
ladder with no trend, inside the template's 0.12–0.42. Fix 1 is at 0.00–0.10 on every
cell but three (0.16, 0.20, 0.26, 0.38). The two bc0 arms are at 0.00–0.06 on every cell
from 50k on.

Actor telemetry on the bc0 arms (`train.csv`, `training/actor_u_norm`, mean `||u_actor||`
at 100k / 300k / 500k): affine bc0 5.56 / 5.79 / 5.93, 5.53 / 5.81 / 5.85, 5.49 / 5.83 /
5.89; dueling bc0 5.57 / 5.48 / 5.79, 5.62 / 5.49 / 5.83, 5.52 / 5.48 / 5.83.
`training/actor_bc_error` (distance to the flow rollout) is 6.9–7.9 on the bc0 arms
against 0.12–0.14 at 500k on the bc = 1 template and Fix 2 dueling. The prior's typical
norm at d_a = 5 is about 2.1; the box corner is 6.7. The design doc's expectation was a
norm near 2; the actors sit at 5.5–5.9 from 100k on.

**Verdicts.**

(a) Fix 1 (`cube_fix1_scalar_tc`) scores 0.044 ± 0.014 against the pre-registered 0.5.
Settled negative. The task-conditioned scalar critic on the frozen phi's synthetic reward
scores below the BC control (0.111); the single-task form of the same critic on the scaled
`r_hat` scored 0.867 on task 2. Fix 3 (`cube_fix3_action_measure_actor`) scores 0.217 ±
0.008 against the pre-registered 0.35, and below the template's 0.246. Settled negative.
Fix 2 scalar-only scores 0.233 ± 0.031 and Fix 2 dueling 0.219 ± 0.073, both against the
pre-registered "> 0.246". Both settled negative; the intervals of all three Section 10
variants overlap the template's, and the differences from it are −0.013, −0.027 and −0.029.

(b) The bc0 arms score 0.038 ± 0.072 (plain affine measure) and 0.003 ± 0.014 (dueling
measure). The pre-registered reading was: above 0.246 means the pinned actor was the limit
of the Section 10 affine arm; below it, the measure's content is. Both are below. Freeing
the actor from the BC term collapses success to about 0 on both measures. The actor climbs
the measure's value (`actor_u_norm` 5.5–5.9, at the box edge; the design doc's block H puts
the sd001 bc0 actor latent at `|u|_inf = 3.0` on the p90 and the max, with the measure
scoring it 3.45 panel-std above the panel mean), and the latents it reaches decode to
actions that do not solve the tasks. The measure's value points to bad latents when it is
maximised without the BC anchor.

(c) Policy-side diagnostic (`tools/diag_measure_vs_scalar_q.py` blocks G and H, 92aba31;
job 2518330; reports `$PSM_DATA/logs/diag_policy_side_cube_{gpi,sec10,sec10bc0}_sd001.json`,
all three present; the write-up in the design doc is by the other agent and was in
progress at the time of this entry). On a 64 x 64 grid of (action latent u, index) per
state over 500 states, the within-state variance of the readout `psi^T w` splits as: GPI
checkpoint (`affine_strict_cube/sd001`, index = prior latent u') share from the index slot
0.993, from u 0.002; Section 10 affine checkpoint (`cube_sec10_affine_actor/sd001`, index
= task vector z) share from the index 0.917, from u 0.079; bc0 checkpoint 1.000 / 0.000.
The pooled ratio of the std across the index at the data latent to the std across u at one
index is 36.2 (GPI), 4.21 (Section 10), 193 (bc0). On the Section 10 checkpoint the
ordering of the 64 action latents is rank-correlated 0.916 (p10 0.808) across the 64 task
vectors, i.e. every task vector ranks the actions at a state alike. The action-axis spread
is 0.6–1.4 times the ensemble disagreement on every checkpoint; the index-axis spread is
5–133 times it. Acting requires the value to separate actions at a state; the measure's
readout moves 4–193 times more with the policy it is asked about than with the action.

(d) Closed on cube by this and the preceding entries: Fix 1 (task-conditioned scalar
critic on frozen phi), Fix 2 (scalar grounding, dueling head), Fix 3 (measure at the
decoded action), bc0 on both measures, the reference PSM critic port (0.156), Section 10
free (0.171) and affine (0.246), and posterior-sampled dataset latents (09-14). The
remaining candidates are two. First, the policy family the flow-latent interface hands the
measure: a fixed noise vector u' as the index is behaviour cloning with one random mode
chosen per state, so the index panel spans policies that differ in which mode they pick,
and the measure's variance lands on that axis (block G). Proposed fix A: state-dependent
index policies `u = f_k(s)` from fixed random networks, so that indexed policies differ in
a state-consistent way and the successor measure has to separate actions to fit them.
Second, the coverage of the test `w` by the training task-vector mixture (32 projected
`N(0, I)` draws plus 32 projected `phi(s'_j)`): the sec10 readout through its own z shows
share 0.994 on the index axis with rank correlation 0.042 across z, and whether the
inferred eval `w` sits inside that mixture's support has not been measured. Neither
candidate was run today.

**Day summary: every arm scored on 2026-09-15 (500 episodes per cell, 500k checkpoint, t
interval over 3 seeds, df 2, five-task).**

| group | what | number | criterion | verdict |
|---|---|---|---|---|
| `cube_fix1_scalar_tc` | task-conditioned scalar critic on frozen phi, DSRL-NA actor | 0.044 ± 0.014 | ≥ 0.5 | settled negative |
| `cube_fix2_scalar_only` | Section 10 affine + scalar grounding | 0.233 ± 0.031 | > 0.246 | settled negative |
| `cube_fix2_scalar_dueling` | + dueling head | 0.219 ± 0.073 | > 0.246 | settled negative |
| `cube_fix3_action_measure_actor` | Section 10 affine, measure at the decoded action | 0.217 ± 0.008 | ≥ 0.35 | settled negative |
| `cube_sec10_affine_bc0` | Section 10 affine, `bc_coeff=0` | 0.038 ± 0.072 | > 0.246 means the actor was the limit | below; the measure's content is the limit |
| `cube_fix2_dueling_bc0` | Fix 2 dueling, `bc_coeff=0` | 0.003 ± 0.014 | same | same |
| `cube_psmref_actor` | reference PSM critic port (09-15 00:08) | 0.156 ± 0.023 | > 0.171 | settled negative |
| `cube_sec10_affine_actor` | Section 10 affine (09-14 evening), the template | 0.246 ± 0.041 | — | reference |
| BC control | frozen flow alone | 0.111 | — | reference |
| FB | TD-JEPA Table 1 | 0.496 | — | reference |

Caveats. One 500k checkpoint per arm, no 250k five-task eval. The Fix 2 dueling arm's
seed 1 (0.253) is above the template's mean and its seed interval is the widest of the
day (± 0.073). Fix 1 changes the actor family, the batch size and the latent box along
with the critic (its template is the DSRL-NA arm, not Section 10), so its 0.044 does not
isolate the task conditioning from the frozen phi or the mask change. The policy-side
numbers in (c) are from one seed (sd001) of each checkpoint. The bc0 actors were not
evaluated at any checkpoint before 500k at 500 episodes; the in-loop cells are at 0.00–0.06
throughout.

---

## 2026-09-15 (00:08) — reference PSM critic on latent inputs: 0.156 five-task; the measure-critic limit is not the loss form

Branch `fix/psmflow-paper-strict`, worktree `.claude/worktrees/psmflow-fix`. One cube arm,
three seeds, 500k steps, all COMPLETED. It was launched in the 09-14 (night) entry below;
design, objects, losses and the pre-registered expectation:
`docs/design/2026-09-14-flow-psm-dsrl-paper-versions.md`, section "Reference PSM critic
(proto stage) on latent inputs". Commits 76eb9cf (`utils/psm_proto.py`, tests), e373ab7
(`agent.proto.enabled` in `agents/psmflow.py`), abbd1fd (`main.py` dataset row for the
proto stage).

**What was run.**

The `cube_sec10_free_actor` template (`psi_form=free policy_index=task_vector
train_actor=true acting=actor actor_mode=ddpg ortho_coef=1000 lr_phi=1e-5
pessimism_penalty=0.5 discount=0.98 u_clip=3.0 batch_size=1024 lr_actor=1e-4
actor.bc_coeff=1.0 num_parallel=2 mix_ratio=0.5`, flow `$PSM_DATA/flow/cube-single-play`
@500000, preimages `$PSM_DATA/preimages/cube-single-play.npz`) plus
`agent.proto.enabled=true` (`proto.max_log_seed=16 proto.proto_seed=0 proto.lr=1e-4
proto.ortho_coef=null`). The critic is the reference PSM critic (arXiv 2411.19418): a proto
stage on latent inputs trains phi and `proto_psi(s, z_bin, u)` against the pseudo-random
latent policy family, then a separate SF head `psi(s, w, u)` is fit against the post-proto
phi with phi stop-gradded and no ortho term. The actor, the flow, the preimages, `infer_z`
and the update budget are the template's. Re-read from `flags.json` after launch: the diff
against the template is the four `proto.*` keys and `run_group`.

| item | value |
|---|---|
| group | `cube_psmref_actor` |
| jobs | 2518213 (sd0), 2518214 (sd1), 2518215 (sd2), 500k steps, COMPLETED |
| run dirs | `$PSM_DATA/exp/PSMFLows/cube_psmref_actor/sd000_s_2518213.0.20260914_211207`, `.../sd001_s_2518214.0.20260914_211206`, `.../sd002_s_2518215.0.20260914_211206` |
| eval JSONs | `$PSM_DATA/logs/cube_psmref_actor_sd00{0,1,2}_500000_task{1..5}.json`, 500 episodes each, `restore_epoch=500000` |

**Five-task table: 500 episodes per cell, 500k checkpoint, t interval over 3 seeds (df 2).**

| seed | task 1 | task 2 | task 3 | task 4 | task 5 | seed mean |
|---|---|---|---|---|---|---|
| 0 | 0.134 | 0.232 | 0.276 | 0.152 | 0.038 | 0.166 |
| 1 | 0.100 | 0.190 | 0.256 | 0.128 | 0.078 | 0.150 |
| 2 | 0.144 | 0.180 | 0.256 | 0.138 | 0.036 | 0.151 |
| task mean | 0.126 | 0.201 | 0.263 | 0.139 | 0.051 | **0.156 ± 0.023** |

The seed means are 0.166 / 0.150 / 0.151; the t half-width over them is 0.0227. The lowest
task is task 5 at 0.036–0.078 per seed, as for the free Section 10 template (0.052–0.070).

**In-loop ladder (50 episodes, task 2, `eval.csv` column `evaluation/success`), with the
template's cells from `cube_sec10_free_actor` (HANDOFF 09-14 evening).**

| arm | seed | 50k | 100k | 150k | 200k | 250k | 300k | 350k | 400k | 450k | 500k |
|---|---|---|---|---|---|---|---|---|---|---|---|
| port `cube_psmref_actor` | 0 | 0.22 | 0.22 | 0.20 | 0.26 | 0.34 | 0.26 | 0.18 | 0.34 | 0.24 | 0.24 |
| | 1 | 0.20 | 0.20 | 0.12 | 0.12 | 0.16 | 0.24 | 0.22 | 0.18 | 0.26 | 0.10 |
| | 2 | 0.38 | 0.36 | 0.38 | 0.20 | 0.32 | 0.22 | 0.22 | 0.08 | 0.30 | 0.18 |
| template `cube_sec10_free_actor` | 0 | 0.28 | 0.24 | 0.46 | 0.22 | 0.24 | 0.18 | 0.30 | 0.32 | 0.26 | 0.24 |
| | 1 | 0.20 | 0.48 | 0.30 | 0.16 | 0.28 | 0.22 | 0.18 | 0.26 | 0.20 | 0.18 |
| | 2 | 0.24 | 0.34 | 0.20 | 0.28 | 0.14 | 0.24 | 0.10 | 0.14 | 0.26 | 0.28 |

The port moves within 0.08–0.38 across the ladder with no trend. Its 500-episode task-2
numbers at 500k (0.232 / 0.190 / 0.180) sit inside the in-loop range.

**Verdicts.**

(a) The port scores 0.156 ± 0.023 five-task. The free Section 10 template scored
0.171 ± 0.027 and the affine Section 10 arm 0.246 ± 0.041; BC is 0.111 and FB 0.496. The
pre-registered expectation was a number above 0.171, with 0.35 as the mark for "the proto
stage was the missing piece" and 0.17–0.25 for "the reference critic inherits the same
limit on latent inputs". The number landed below the whole band. The proto stage does not
rescue the measure critic on latent inputs. Swapping the Section 10 critic for the reference
PSM critic, with the actor, the flow, the preimages and the optimiser held, changes the
five-task mean by −0.015; the two seed intervals overlap.

(b) The loss form is now matched to the reference that scores 694 on Walker (arXiv
2411.19418, ExORL). The candidates that remain for the Walker/OGBench difference are the
data (ExORL RND exploratory trajectories against OGBench play data), the action space (raw
actions there, flow latents here), and psmflow's optimiser and regulariser values, which
this arm kept from the template: `ortho_coef=1000`, `lr_phi=1e-5`,
`pessimism_penalty=0.5`, 500k steps. None of the three was varied today.

(c) The scalar-grounded route (DSRL-NA on a scalar Bellman critic fed by `phi^T w`) remains
the only composition above 0.5 on either env: cube task 2 0.867 ± 0.153, antmaze task 1
0.853 ± 0.334, against the measure-critic rows at 0.10–0.28. Its zero-shot form is the
proposed next arm: a task-conditioned scalar Q on `phi(s')^T w` for sampled `w`, phi
frozen, no termination masks (`dsrl_na.task_conditioned` is the seam; the earlier D1b/D2
arms of that shape had the sign and mask defects recorded on 09-13 and have to be redone
with the scaled reward).

(d) The proposed next measurement is eval-only: on the same `(s, u)` pairs, compare
`psi^T w` from a measure-critic run against the scalar Q from the 0.87 run
(`cube_dsrlna_rhat_scaled`). Report the correlation and the rank correlation over 64
sampled `u` per state, split by the distance of `u` to the dataset preimage at that
state. This measures whether the measure critic orders latents the way the scalar critic
does, and where in latent space the ordering breaks.

**Codebase check (facts only, no code changed).** TD-JEPA trains no scalar critic: its
actor maximises `predictor(phi(s), z, a)^T z` (arXiv 2510.00739, Alg. 1). Meta Motivo's
plain FB has an optional scalar `q_loss` on `F^T z` whose default coefficient is 0.0.
FB-CPR's scalar critic is grounded on the discriminator reward. psmflow's defaults match
TD-JEPA's shape: no scalar critic. The `dsrl_na` seam is psmflow's scalar-critic path.

**Day summary: every arm run on 2026-09-14 (500 episodes per cell, 500k checkpoint unless
stated, t interval over 3 seeds, df 2).**

| env | group | what | metric | number | entry |
|---|---|---|---|---|---|
| cube | `cube_affine_posterior_u` | affine GPI, posterior-sampled dataset latent | task 2, pooled 18 cells 250k–500k | 0.252 ± 0.057 (point control 0.424 ± 0.076) | 09-14 |
| cube | `cube_dsrlna_rhat_frozen` | DSRL-NA on raw-scale `r_hat` | task 2 | 0.010 ± 0.013 | 09-14 evening |
| cube | `cube_dsrlna_rhat_scaled` | DSRL-NA on scaled `r_hat` | task 2 | 0.867 ± 0.153 (real-reward control 0.882 ± 0.121) | 09-14 evening |
| cube | `cube_sec10_free_actor` | Section 10 actor, free psi | five-task | 0.171 ± 0.027 | 09-14 evening |
| cube | `cube_sec10_affine_actor` | Section 10 actor, affine psi | five-task | 0.246 ± 0.041 | 09-14 evening |
| cube | `cube_psmref_actor` | reference PSM critic (proto stage), free psi | five-task | 0.156 ± 0.023 | this entry |
| cube | BC control | frozen flow alone | five-task | 0.111 | HANDOFF 09-11 |
| cube | FB | TD-JEPA Table 1 | five-task | 0.496 | — |
| antmaze | `antmaze_dsrlna_rhat_scaled` | DSRL-NA on scaled `r_hat` | task 1 | 0.853 ± 0.334 (real-reward control 0.979 ± 0.008) | 09-14 night |
| antmaze | `antmaze_sec10_affine_actor` | Section 10 actor, affine psi | five-task | 0.105 ± 0.025 | 09-14 night |
| antmaze | BC control | frozen flow alone | task 1 | 0.072 | `docs/tables/results.md` |
| antmaze | FB | TD-JEPA Table 1 | five-task | 0.730 | — |
| antmaze-stitch | `relaunch_point_v2` | affine GPI, point preimage, stitch flow (campaign launched 09-13, cells finished 09-14) | five-task | 0.049 ± 0.113 (seeds 0.056 / 0.091 / 0.000) | this entry |
| antmaze-stitch | `distribution_v1` | affine GPI, distribution preimage, same stitch flow | five-task | 0.087 ± 0.179 (seeds 0.149 / 0.008 / 0.104) | this entry |
| antmaze-stitch | BC control | stitch flow alone | five-task | 0.013 (0.002 / 0.002 / 0.052 / 0.000 / 0.008) | `outputs/psmflow_antmaze_stitch_20260913/reports/bc_flow_task{1..5}.json` |

The stitch rows come from
`outputs/psmflow_antmaze_stitch_20260913/{relaunch_point_v2,distribution_v1}/reports/psmflow_seed{0,1,2}_task{1..5}.json`;
all thirty cells are 500 episodes at `restore_epoch=500000`. Tasks 1, 2 and 4 are at
0.000–0.032 on every seed for both arms; task 5 carries the whole mean (point 0.276 / 0.422
/ 0.000, distribution 0.740 / 0.036 / 0.520). Their flow is the campaign's own Stage A
(`stitch_stage_a/sd000_s_2506471`), not the published `antmaze-medium-navigate` flow.

Caveats. One arm, three seeds, 500k only; no 250k five-task eval. The port keeps the
template's `discount=0.98` and the free psi; the affine head under `proto.enabled` was
not run (`create` does not forbid it; its guards are `policy_index=task_vector`,
`train_actor=true`, `train_phi=true`, `max_log_seed` in [1, 30]). The template's cells in the ladder are from the 09-14
evening runs, not re-run. The stitch aggregates (`tools/report_seed_comparison.py`, jobs
2513750 / 2513899) were not checked; the stitch numbers above are recomputed from the
per-cell JSONs.

---

## 2026-09-14 (night) — antmaze repeat: scaled r_hat 0.853 vs control 0.979 on task 1; Section 10 affine 0.105 five-task; reference PSM critic ported and launched

Branch `fix/psmflow-paper-strict`, worktree `.claude/worktrees/psmflow-fix`. Two antmaze arms,
three seeds each, 500k steps, all COMPLETED; one cube arm (the reference-critic port)
launched and RUNNING at write time. Design and pre-registered expectations:
`docs/design/2026-09-14-flow-psm-dsrl-paper-versions.md`, sections "Antmaze repeat launched"
and "Reference PSM critic (proto stage) on latent inputs". Commits d51de4c (antmaze
launches), 76eb9cf (`utils/psm_proto.py` + `tests/test_psmflow_psm_ref.py`), e373ab7
(`agent.proto.enabled` in `agents/psmflow.py`, `configs/agent/psmflow.yaml`), abbd1fd
(`main.py` emits the dataset row for the proto stage; docs).

**What was run.**

Arm 1 is the DSRL-NA pipeline of the antmaze real-reward control `dsrlna_antmaze` (HANDOFF
09-10 §1, jobs 2492660-62): `psi_form=affine policy_index=latent train_actor=true
acting=actor actor_mode=dsrl_sac dsrl_na.enabled=true dsrl_na.discount=0.99
dsrl_na.hidden_dim=2048 dsrl_na.num_ensembles=2 dsrl_na.inner_steps=10 u_clip=1.5
batch_size=256 discount=0.99`, env `antmaze-medium-navigate-singletask-v0` (OGBench task 1,
the maze default), flow `$PSM_DATA/flow/antmaze-medium-navigate` @500000, preimages
`$PSM_DATA/preimages/antmaze-medium-navigate.npz`. The only change is
`dataset.reward_override_path`: the dataset `rewards` array is replaced by
`r_hat = phi(s')^T w` from `affine_strict_antmaze_g99/sd002 @500k` (`w` closed form on 10k
relabel rows, `reward_shift=1.0`), mapped by least squares onto the 0/1 reward and shifted
by −1 (`0.002591 r_hat − 0.012158 − 1`; output mean −0.991, std 0.027, range
[−1.074, −0.791]). The masks stay the dataset's (`1 - success`). The `flags.json` diff
against the control is the override path plus three keys that did not exist on 09-09, at
their defaults (`dsrl_na.reward_refit_every=10000`, `measure_action_input=latent`,
`train_phi=true`).

Arm 2 is the paper's Section 10 agent on the affine head: the `affine_strict_antmaze_g99`
template (discount 0.99) plus `policy_index=task_vector train_actor=true acting=actor
actor_mode=ddpg u_clip=3.0 pessimism_penalty=0.5 num_parallel=2 mix_ratio=0.5
actor.bc_coeff=1.0 lr_actor=1e-4 batch_size=1024`, same flow and preimage file. The dataset
reward is not used.

| arm | group | jobs | run dirs | wall clock |
|---|---|---|---|---|
| 1 scaled r_hat | `antmaze_dsrlna_rhat_scaled` | 2518182, 2518183, 2518184 | `$PSM_DATA/exp/PSMFLows/antmaze_dsrlna_rhat_scaled/sd00{0,1,2}_s_25181*` | 4h29–4h32 |
| 2 Section 10, affine psi | `antmaze_sec10_affine_actor` | 2518179, 2518180, 2518181 | `.../antmaze_sec10_affine_actor/sd00{0,1,2}_s_25181*` | 3h43–3h45 |

Reward file:
`$PSM_DATA/rewards/antmaze-medium-navigate_rhat_affine_strict_antmaze_g99_sd002_500k_scaled.npz`
(its `.meta.json` carries the fit and `w`). Eval JSONs, 500 episodes each,
`restore_epoch=500000`: `$PSM_DATA/logs/antmaze_dsrlna_rhat_scaled_sd00{0,1,2}_500000_task1.json`,
`$PSM_DATA/logs/dsrlna_antmaze_sd00{0,1,2}__500000.json` (control),
`$PSM_DATA/logs/antmaze_sec10_affine_actor_sd00{0,1,2}_500000_task{1..5}.json`.

Relabel fit of `r_hat` against the shifted reward on all 1M rows: Pearson 0.286, R² under
the best affine map 0.082, top-1% precision 0.291 at a base rate 0.0091. Cube (09-14
evening): 0.330 / 0.109 / 0.369 at 0.021.

**Mechanism table: task 1, 500 episodes, 500k checkpoint, t interval over 3 seeds (df 2).**

| arm | reward channel | sd0 | sd1 | sd2 | mean ± 95% |
|---|---|---|---|---|---|
| 1 scaled r_hat | `phi^T w` mapped to −1/0 | 0.888 | 0.966 | 0.704 | **0.853 ± 0.334** |
| control `dsrlna_antmaze` | dataset reward | 0.982 | 0.978 | 0.976 | **0.979 ± 0.008** |

**Zero-shot table: five tasks, 500 episodes per task, 500k checkpoint, t interval over 3 seeds (df 2).**

| arm | sd0 | sd1 | sd2 | mean ± 95% | source |
|---|---|---|---|---|---|
| BC (frozen flow alone) | | | | 0.072 | `docs/tables/results.md` (task 1, 500 ep) |
| 2 Section 10, affine psi | 0.109 | 0.113 | 0.094 | **0.105 ± 0.025** | this entry |
| FB | | | | 0.730 | TD-JEPA Table 1 |
| HILP | | | | 0.836 | TD-JEPA Table 1 |

No five-task GPI row exists for antmaze. The affine GPI antmaze number on record is task 1
only: `affine_strict_antmaze_g99`, 30-cell ladder 0.294 ± 0.070 (HANDOFF 09-07). The BC row
is task 1 as well. Arm 2 per task (mean over seeds): 0.177 / 0.075 / 0.089 / 0.053 / 0.132
on tasks 1–5; the lowest task is task 4 at 0.048–0.060 per seed.

**In-loop ladders (50 episodes, task 1, `eval.csv` column `evaluation/success`).**

| arm | seed | 50k | 100k | 150k | 200k | 250k | 300k | 350k | 400k | 450k | 500k |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 scaled r_hat | 0 | 0.84 | 0.68 | 0.92 | 0.94 | 0.92 | 0.96 | 0.86 | 0.94 | 0.92 | 0.82 |
| | 1 | 0.88 | 0.60 | 0.86 | 0.96 | 0.92 | 1.00 | 0.92 | 0.82 | 0.94 | 0.94 |
| | 2 | 0.76 | 0.92 | 0.90 | 0.92 | 1.00 | 0.86 | 0.92 | 0.84 | 0.84 | 0.68 |
| control `dsrlna_antmaze` | 0 | 0.88 | 0.88 | 0.78 | 0.94 | 0.98 | 0.96 | 0.90 | 0.96 | 0.98 | 1.00 |
| | 1 | 0.80 | 0.96 | 0.90 | 1.00 | 1.00 | 0.98 | 0.98 | 1.00 | 1.00 | 1.00 |
| | 2 | 0.94 | 0.90 | 0.98 | 0.94 | 0.98 | 0.98 | 0.96 | 1.00 | 0.74 | 0.98 |
| 2 Sec10 affine | 0 | 0.16 | 0.22 | 0.14 | 0.14 | 0.18 | 0.16 | 0.16 | 0.24 | 0.12 | 0.24 |
| | 1 | 0.12 | 0.16 | 0.24 | 0.24 | 0.16 | 0.22 | 0.20 | 0.14 | 0.22 | 0.22 |
| | 2 | 0.22 | 0.22 | 0.28 | 0.16 | 0.22 | 0.16 | 0.18 | 0.18 | 0.18 | 0.16 |

Arm 1 is at 0.60–1.00 from 50k on. Seed 2's last three in-loop cells are 0.84 / 0.84 /
0.68 and its 500-episode number at 500k is 0.704, so the low seed-2 cell agrees with its
ladder. Arm 2 moves within 0.12–0.28 across the ladder with no trend; its task-1 in-loop
numbers sit above the five-task 500-episode mean because task 1 is its best task.

**Reference PSM critic port, launched.**

`agent.proto.enabled=true` (commits 76eb9cf, e373ab7, abbd1fd) replaces the Section 10
critic with the reference PSM critic (arXiv 2411.19418): a proto stage fits
`proto_psi(s, z_bin, u)` against a fixed pseudo-random latent policy family keyed on the
dataset row and trains phi (with the ortho term); a separate reward-conditioned SF head
`psi(s, w, u)` is then fit against the online phi the proto stage just stepped, with phi
stop-gradded and no ortho term. Everything else stays: frozen flow, point preimages,
closed-form `infer_z`, the ddpg noise-space actor, `task_w` mixing, `P=2`. With
`enabled=false` the params after 3 updates equal the pre-change baseline on both the strict
and the actor arm (`tests/test_psmflow_psm_ref.py`). Objects, losses and update order are
tabulated in the design doc.

| item | value |
|---|---|
| group | `cube_psmref_actor` |
| template | `cube_sec10_free_actor` flags: `psi_form=free policy_index=task_vector train_actor=true acting=actor actor_mode=ddpg ortho_coef=1000 lr_phi=1e-5 pessimism_penalty=0.5 discount=0.98 u_clip=3.0 batch_size=1024 lr_actor=1e-4 actor.bc_coeff=1.0`, flow `$PSM_DATA/flow/cube-single-play` @500000, preimages `$PSM_DATA/preimages/cube-single-play.npz` |
| change | `agent.proto.enabled=true` (`proto.max_log_seed=16 proto.proto_seed=0 proto.lr=1e-4 proto.ortho_coef=null`); the `flags.json` diff against the template is these four keys and `run_group` only |
| jobs | 2518213 (sd0), 2518214 (sd1), 2518215 (sd2), 500k steps, RUNNING |
| run dirs | `$PSM_DATA/exp/PSMFLows/cube_psmref_actor/sd00{0,1,2}_s_25182*` |

In-loop so far (50 episodes, task 2, checkpoints 50k–200k), with the template's cells at the
same checkpoints from `cube_sec10_free_actor`:

| arm | seed | 50k | 100k | 150k | 200k |
|---|---|---|---|---|---|
| port `cube_psmref_actor` | 0 | 0.22 | 0.22 | 0.20 | 0.26 |
| | 1 | 0.20 | 0.20 | 0.12 | 0.12 |
| | 2 | 0.38 | 0.36 | 0.38 | 0.20 |
| template `cube_sec10_free_actor` | 0 | 0.28 | 0.24 | 0.46 | 0.22 |
| | 1 | 0.20 | 0.48 | 0.30 | 0.16 |
| | 2 | 0.24 | 0.34 | 0.20 | 0.28 |

Pre-registered expectation for the port: five-task mean above the free Section 10 arm
(0.171 ± 0.027). Above 0.35 means the proto stage was the missing piece. 0.17–0.25 means the
reference critic inherits the same limit on latent inputs. Eval at 500k, five tasks, 500
episodes each, once the jobs finish.

**Verdicts.**

(a) On antmaze the scaled inferred reward reaches 0.853 ± 0.334 on task 1 against the
real-reward control's 0.979 ± 0.008. The gap is 0.13. It is driven by seed 2 (0.704; seeds
0 and 1 are 0.888 and 0.966). Cube's gap on the same test was 0.015 (0.867 vs 0.882). The
larger antmaze gap goes with the weaker relabel statistics (Pearson 0.286 vs 0.330, R²
0.082 vs 0.109, top-1% precision 0.291 vs 0.369). The scaled reward still scores 2.9x the
measure-critic route on the same task (affine GPI `affine_strict_antmaze_g99` 0.294 ± 0.070).

(b) Section 10 with the affine head scores 0.105 ± 0.025 five-task on antmaze: above BC
0.072, far below FB 0.730 and HILP 0.836. The pre-registered band was 0.1–0.3; it landed at
the bottom edge. On cube the same arm scored 0.246 ± 0.041 five-task.

(c) The isolation to the measure critic holds on both envs. The same phi, w and flow give
the real-reward level when a scalar Bellman critic consumes the scaled r_hat (cube 0.867 on
task 2, antmaze 0.853 on task 1) and 0.1–0.3 when the measure `psi^T w` is the critic (cube
0.246 five-task Section 10, 0.284 five-task GPI; antmaze 0.105 five-task Section 10, 0.294
task-1 GPI). The gap to FB is in the measure-as-critic path on both environments.

(d) The port arm `cube_psmref_actor` is the direct test of the critic swap: the reference
PSM critic (proto stage plus SF head) in place of the Section 10 critic, every other piece
held. Its 500k five-task number decides between the two readings pre-registered above.

Caveats. Arm 1 is task 1 only; `w` was inferred from task-1 relabel rows and the scale map
was fit on the task-1 reward (two scalars), as on cube. Three seeds per cell; the arm-1
interval (± 0.334) is wide because of seed 2. No 250k eval was run for either antmaze arm.
The port arm has no 500-episode number yet.

---

## 2026-09-14 (evening) — flow + PSM + DSRL: reward channel settled, Section 10 arms measured five-task

Branch `fix/psmflow-paper-strict`, worktree `.claude/worktrees/psmflow-fix`. Four arms,
three seeds each, cube-single-play, 500k steps, all COMPLETED. Design and pre-registered
expectations: `docs/design/2026-09-14-flow-psm-dsrl-paper-versions.md`; the paper-vs-code
diff behind them: `docs/design/2026-09-14-paper-vs-code-diagnosis.md`. Commits 98cbce5
(diagnosis, posterior-latent arm), 2c7189b (`tools/relabel_reward_rhat.py`,
`dataset.reward_override_path`), 5d06aa3 (scale-matched relabel), ebb2a5c (affine head
under `policy_index=task_vector`, Section 10 launches), 82429fa (five-task reporting rule).

**What was run.**

Arms 1 and 2 are the DSRL-NA pipeline of the real-reward run `dsrlna_cube` (HANDOFF 09-09,
`scripts/slurm/launch_dsrl_na.sh`, `REWARD_SOURCE=real`): `psi_form=affine
policy_index=latent train_actor=true acting=actor actor_mode=dsrl_sac dsrl_na.enabled=true
dsrl_na.discount=0.99 dsrl_na.hidden_dim=2048 dsrl_na.num_ensembles=2 dsrl_na.inner_steps=10
u_clip=1.5 batch_size=256 discount=0.98`, flow `$PSM_DATA/flow/cube-single-play` @500000,
preimages `$PSM_DATA/preimages/cube-single-play.npz`. The only change is the dataset
`rewards` array, replaced through `dataset.reward_override_path`. `r_hat = phi(s')^T w` is
computed once from `affine_strict_cube/sd001 @500k` with `w` inferred as at eval (10k
relabel rows, `reward_shift=1.0`, closed form). The masks stay the dataset's (`1 - success`).

Arms 3 and 4 are the paper's Section 10 agent (`psi(s,u,w)` indexed by the task vector,
DDPG-style noise-space actor, the actor's latent in the bootstrap): the `affine_strict_cube`
template plus `policy_index=task_vector train_actor=true acting=actor actor_mode=ddpg
u_clip=3.0 pessimism_penalty=0.5 num_parallel=2 mix_ratio=0.5 actor.bc_coeff=1.0
lr_actor=1e-4 batch_size=1024 discount=0.98`, same flow and preimage file. Arm 3 adds
`psi_form=free`; arm 4 keeps `psi_form=affine`, which needed the guard relaxed in ebb2a5c.

| arm | group | jobs | reward file | run dirs |
|---|---|---|---|---|
| 1 raw-scale r_hat | `cube_dsrlna_rhat_frozen` | 2518095, 2518096, 2518097 | `$PSM_DATA/rewards/cube-single-play_rhat_affine_strict_cube_sd001_500k.npz` (mean +3.95, std 10.58, range [−32.0, 62.6]) | `$PSM_DATA/exp/PSMFLows/cube_dsrlna_rhat_frozen/sd00{0,1,2}_s_25180*` |
| 2 scaled r_hat | `cube_dsrlna_rhat_scaled` | 2518113, 2518114, 2518115 | `..._500k_scaled.npz` = `0.00445 r_hat + 0.00325 − 1` (least-squares map onto the 0/1 reward, then −1; mean −0.979, std 0.047, range [−1.139, −0.718]) | `.../cube_dsrlna_rhat_scaled/sd00{0,1,2}_s_25181*` |
| 3 Section 10, free psi | `cube_sec10_free_actor` | 2518116, 2518117, 2518118 | dataset reward not used | `.../cube_sec10_free_actor/sd00{0,1,2}_s_25181*` |
| 4 Section 10, affine psi | `cube_sec10_affine_actor` | 2518119, 2518120, 2518121 | dataset reward not used | `.../cube_sec10_affine_actor/sd00{0,1,2}_s_25181*` |

Eval JSONs, 500 episodes each, `restore_epoch=500000`:
`$PSM_DATA/logs/cube_dsrlna_rhat_{frozen,scaled}_sd00{0,1,2}_500000_task2.json`,
`$PSM_DATA/logs/dsrlna_cube_sd00{0,1,2}_500000_task2.json` (control),
`$PSM_DATA/logs/cube_sec10_{free,affine}_actor_sd00{0,1,2}_500000_task{1..5}.json`.

Reporting rule from today (82429fa): zero-shot arms are quoted as five-task means only.
Arms 1 and 2 are single-task mechanism tests by construction: `w` and the real-reward
control are task-2 objects, so they are reported on task 2 and labelled as such.

**Mechanism table: task 2, 500 episodes, 500k checkpoint, t interval over 3 seeds (df 2).**

| arm | reward channel | sd0 | sd1 | sd2 | mean ± 95% |
|---|---|---|---|---|---|
| 1 raw-scale r_hat | `phi^T w`, raw | 0.012 | 0.004 | 0.014 | **0.010 ± 0.013** |
| 2 scaled r_hat | `phi^T w` mapped to −1/0 | 0.856 | 0.812 | 0.934 | **0.867 ± 0.153** |
| control `dsrlna_cube` | dataset reward | 0.842 | 0.868 | 0.936 | **0.882 ± 0.121** |

Relabel fit of `r_hat` against the shifted reward on all 1M rows: Pearson 0.330, R² under
the best affine map 0.109, top-1% precision 0.369 at a base rate 0.021. The fit is
scale-free and identical for arms 1 and 2.

**Zero-shot table: five tasks, 500 episodes per task, 500k checkpoint, t interval over 3 seeds (df 2).**

| arm | sd0 | sd1 | sd2 | mean ± 95% | source |
|---|---|---|---|---|---|
| BC (frozen flow alone) | | | | 0.111 | HANDOFF 09-11 |
| 3 Section 10, free psi | 0.179 | 0.158 | 0.175 | **0.171 ± 0.027** | this entry |
| 4 Section 10, affine psi | 0.252 | 0.259 | 0.227 | **0.246 ± 0.041** | this entry |
| affine GPI (`affine_strict_cube`, 300k–500k) | | | | 0.284 | HANDOFF 09-06 / `docs/tables/results.md` |
| FB | | | | 0.496 | TD-JEPA Table 1 |
| HILP | | | | 0.742 | TD-JEPA Table 1 |

Per task (mean over seeds): free 0.149 / 0.205 / 0.289 / 0.149 / 0.061 on tasks 1–5;
affine 0.221 / 0.316 / 0.333 / 0.211 / 0.149. Free psi's lowest task is task 5 at 0.052–0.070
per seed; affine's lowest is task 5 at 0.096–0.218.

**In-loop ladders (50 episodes, task 2 for all four arms, `eval.csv` success column).**

| arm | seed | 50k | 100k | 150k | 200k | 250k | 300k | 350k | 400k | 450k | 500k |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 raw r_hat | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.02 | 0.02 | 0.00 | 0.00 | 0.00 |
| | 1 | 0.00 | 0.00 | 0.02 | 0.00 | 0.02 | 0.04 | 0.02 | 0.00 | 0.00 | 0.00 |
| | 2 | 0.04 | 0.00 | 0.06 | 0.04 | 0.00 | 0.02 | 0.06 | 0.00 | 0.10 | 0.04 |
| 2 scaled r_hat | 0 | 0.88 | 0.88 | 0.96 | 0.86 | 0.96 | 0.96 | 0.96 | 0.98 | 0.84 | 0.86 |
| | 1 | 0.88 | 0.90 | 0.90 | 0.94 | 0.92 | 0.90 | 0.94 | 0.92 | 0.82 | 0.86 |
| | 2 | 0.90 | 0.82 | 0.90 | 0.88 | 0.96 | 0.88 | 0.80 | 0.86 | 0.84 | 0.92 |
| 3 Sec10 free | 0 | 0.28 | 0.24 | 0.46 | 0.22 | 0.24 | 0.18 | 0.30 | 0.32 | 0.26 | 0.24 |
| | 1 | 0.20 | 0.48 | 0.30 | 0.16 | 0.28 | 0.22 | 0.18 | 0.26 | 0.20 | 0.18 |
| | 2 | 0.24 | 0.34 | 0.20 | 0.28 | 0.14 | 0.24 | 0.10 | 0.14 | 0.26 | 0.28 |
| 4 Sec10 affine | 0 | 0.28 | 0.38 | 0.18 | 0.34 | 0.30 | 0.24 | 0.24 | 0.30 | 0.30 | 0.24 |
| | 1 | 0.24 | 0.36 | 0.42 | 0.28 | 0.36 | 0.22 | 0.26 | 0.42 | 0.34 | 0.32 |
| | 2 | 0.26 | 0.30 | 0.38 | 0.30 | 0.24 | 0.26 | 0.12 | 0.36 | 0.28 | 0.32 |

Arm 2 is at 0.82–0.98 from the first checkpoint on; the scaled reward needs no warm-up
beyond 50k. Arm 1 never leaves 0.00–0.10. Arms 3 and 4 move within 0.10–0.48 across the
ladder with no trend; the Section 10 single-task in-loop numbers sit above their five-task
500-episode means because task 2 is their second-best task.

**Verdicts.**

(a) The inferred reward `phi^T w`, once mapped to the dataset's −1/0 convention, supports
latent Q-learning plus the noise actor at the real-reward level: 0.867 ± 0.153 against
0.882 ± 0.121 on task 2. The two intervals overlap almost entirely. The reward vector `w`
and the features `phi` carry enough of the task to drive a Bellman critic, even though the
per-row fit to the true reward is weak (R² 0.109).

(b) The raw-scale `r_hat` reproduces the earlier actor-on-inferred-reward failures:
0.010 ± 0.013. The per-step reward is positive on average (+3.95) and the termination mask
ends the episode at success, so the critic values staying alive above finishing. This
is the likely cause of the 09-07 to 09-10 DSRL-NA-on-r_hat results (0.307 and below) and
of D1b/D2, which used the same raw readout with the masks on; the 09-13 audit already
recorded the mask defect for D1b/D2. The pre-registered expectation for arm 1 was low,
and it landed there.

(c) Section 10 with the affine head scores 0.246 ± 0.041 five-task, 0.075 above the free
head's 0.171 ± 0.027; the seed intervals do not overlap. The affine head has no task
below 0.096, the free head has task 5 at 0.052–0.070. The affine Section 10 arm is at the
affine GPI level (0.284) and stays below FB (0.496). The pre-registered band for arm 4 was
0.2–0.5; it landed at the bottom of it.

(d) The reward channel and the flow are therefore not where the gap to FB is. The scaled
`r_hat` reaches the real-reward number when a scalar Bellman critic uses it; the same
`phi`, `w` and flow give 0.25–0.28 when the measure `psi^T w` is the critic. The gap is in
the measure-as-critic path (rung 1 GPI, Section 10 actor on `psi^T w`). The working
composition is per-task rungs 2–3: relabel with the scaled `r_hat`, then DSRL-NA. It uses
reward labels only (no dataset reward), but it is one model per task, so it is not
single-model zero-shot. The next decision is between (i) running it per task, 5 tasks × 3
seeds, for a five-task number comparable to the zero-shot table, and (ii) building a
task-conditioned scalar-Q version (`dsrl_na.task_conditioned` exists as a seam; the earlier
D1b/D2 arms of that shape had the sign and mask defects and have to be redone with the
scaled reward and the masks handled).

Caveats. Arms 1 and 2 are task 2 only and `w` was inferred from task-2 relabel rows; no
five-task number exists for the scaled-reward composition. The scaled map was fit on the
task-2 reward, which is the quantity a zero-shot method may not see; the mechanism test
uses it to fix the scale and sign, and the map's slope and offset are two scalars. Section
10 arms are at 500k only; no 250k five-task eval was run.

---

## 2026-09-14 — posterior-sampled dataset latent on cube (paper Alg. pretrain), paper-vs-code diagnosis

Branch `fix/psmflow-paper-strict`, worktree `.claude/worktrees/psmflow-fix`. The diagnosis
of the paper's LatentFlowPSM against `agents/psmflow.py` is in
`docs/design/2026-09-14-paper-vs-code-diagnosis.md`. Its conclusion: the loss in the code
is the paper's loss. The remaining deviations are the point dataset latent instead of a
posterior sample, one-step decode at acting, the ensemble-min target, the reward shift and
sphere projection of `w`, and the box latent support. Only the first had never been
measured on cube with the affine head. This entry measures it.

**What was run.** One arm, three seeds, 500k steps, cube-single-play task 2. Default
`agent=psmflow` (`psi_form=affine policy_index=latent train_actor=false acting=gpi`) with
the single change `agent.use_point_preimage=false`: the dataset latent `u_i` is one fresh
sample per visit from the stored Gaussian posterior `q_alpha(u | s_i, a_i)`, which is the
paper's Alg. `pretrain`. Every other key equals the `affine_strict_cube` baseline's
`flags.json` (`discount=0.99`, `gpi` K=64, flow `$PSM_DATA/flow/cube-single-play` @500000).
The preimage file is `$PSM_DATA/preimages/cube-single-play-a20p6-ps0p69-ns12-N200.npz`
(alpha 20.57, prior weight 0.69, 12 ODE steps, 200 samples); the baseline's legacy
`cube-single-play.npz` was inverted with `prior_scale=None` and `main.py` refuses
`use_point_preimage=false` on it. The point preimages of the two files differ by at most
0.035 (max abs over 1M rows), so the baseline point arm was not rerun on the new file.

| item | value |
|---|---|
| group | `cube_affine_posterior_u` |
| jobs | 2516368 (sd0), 2516369 (sd1), 2516371 (sd2), all COMPLETED, 3h08m-3h16m |
| run dirs | `$PSM_DATA/exp/PSMFLows/cube_affine_posterior_u/sd00{0,1,2}_s_25163*` |
| eval JSONs | `$PSM_DATA/logs/cube_posterior_u_sd00{0,1,2}_{250000..500000 step 50000}.json` |
| baseline JSONs | `$PSM_DATA/logs/eval500_affine{250..500}k_strict_cube_sd{0,1,2}.json` |

**Results, 500 episodes per cell, checkpoints 250k-500k.**

| ckpt | posterior sd0 | posterior sd1 | posterior sd2 | point sd0 | point sd1 | point sd2 |
|---|---|---|---|---|---|---|
| 250k | 0.288 | 0.402 | 0.104 | 0.532 | 0.620 | 0.246 |
| 300k | 0.202 | 0.332 | 0.076 | 0.286 | 0.532 | 0.360 |
| 350k | 0.238 | 0.410 | 0.148 | 0.704 | 0.494 | 0.350 |
| 400k | 0.152 | 0.428 | 0.114 | 0.282 | 0.346 | 0.546 |
| 450k | 0.114 | 0.402 | 0.152 | 0.272 | 0.568 | 0.564 |
| 500k | 0.320 | 0.394 | 0.254 | 0.086 | 0.548 | 0.292 |
| seed mean | 0.219 | 0.395 | 0.141 | 0.360 | 0.518 | 0.393 |

| arm | n (ckpt x seed) | pooled mean | sd | 95% CI | min | max |
|---|---|---|---|---|---|---|
| posterior latent (`use_point_preimage=false`) | 18 | **0.252** | 0.123 | ± 0.057 | 0.076 | 0.428 |
| point latent (`affine_strict_cube`, recomputed) | 18 | **0.424** | 0.164 | ± 0.076 | 0.086 | 0.704 |
| BC control | — | 0.072 | — | — | — | — |

CI convention: both rows use the `docs/tables/results.md` late-window rule
(`tools/make_tables.py::_actor_ablation_agg`): mean over the 18 cells, 1.96 x sd / sqrt(18).
The baseline recomputed from its own 18 JSONs reproduces the recorded 0.424 ± 0.076 exactly.
Across the three seed means with a t interval (df 2) the rows are 0.252 ± 0.322 and
0.424 ± 0.207.

| paired difference (posterior minus point, same seed and checkpoint) | value |
|---|---|
| n | 18 |
| mean | **−0.172** |
| sd | 0.171 |
| 95% CI, 1.96 x sd / sqrt(n) | ± 0.079 |
| 95% CI, t (df 17) | ± 0.085 |
| cells with posterior below point | 16 of 18 |
| cells with posterior above point | 2 of 18 (sd0 @500k +0.234, sd1 @400k +0.082) |

**Verdict against the pre-registered expectation** (diagnosis doc §5): the expectation was a
move of less than 0.1 in either direction, and less than 20% chance of exceeding 0.55. The
measured move is −0.172 with interval [−0.251, −0.093]. The interval excludes the
pre-registered band. No cell exceeded 0.428. Per the rule stated in §5, item (a) of the
diagnosis (point latent instead of posterior sample) joins the settled list: training on
the posterior-sampled latent scores below the point latent on cube with the affine head,
the same sign as the free-psi era comparison (0.239 point vs 0.194 posterior). The
posterior arm's spread is smaller (sd 0.123 vs 0.164), and its seed 1 is the only seed
whose mean is within the baseline's range.

Caveats. The comparison uses two preimage files. Their point preimages agree to 0.035 but
their posteriors differ by up to 15.7 in the mean; the posterior is the quantity under
test, so this is the intended difference, but the prior weight is 0.69 where the paper
uses 1.0, and no cube file at 1.0 exists. The number of posterior samples per row is one
(`measure_u_samples=1`).

**Stitch state at 2026-09-14 ~12:30 UTC** (antmaze-medium-stitch, point vs distribution
preimage, campaign `outputs/psmflow_antmaze_stitch_20260913/`). All six training jobs are
still RUNNING at 6h08m of a 24h limit; their training receipts record 500000 steps, and the
jobs are now in the per-task 500-episode eval phase (about 70 min per task).

| arm | jobs | task 1 (sd0/1/2) | task 2 (sd0/1/2) | tasks 3-5 |
|---|---|---|---|---|
| point (`relaunch_point_v2`) | 2513746/48/49 RUNNING | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | evaluating |
| distribution (`distribution_v1`) | 2513895/97/98 RUNNING | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | evaluating |

The twelve finished cells are `.../{relaunch_point_v2,distribution_v1}/reports/psmflow_seed{0,1,2}_task{1,2}.json`,
each 500 episodes at `restore_epoch=500000`. Aggregates 2513750 (point) and 2513899
(distribution) are PENDING on `afterok` of their three C jobs; they write the five-task
three-seed Student-t summary into the same `reports/` directories via
`tools/report_seed_comparison.py`, logs at `.../logs/aggregate_2513750.log` and the
`distribution_v1/logs/` counterpart. Nothing was resubmitted. No five-task CI exists yet.

---

## 2026-09-14 03:43 UTC — Conditional Walker affine retry queued; both Stitch arms past halfway

The user explicitly approved a 12-hour retry of affine Walker seed 1 **only if** its
current evaluation fails or times out. Job **2516137** is PENDING with
`afternotok:2514952`, TimeLimit12:00:00, one H100, 8 CPUs and 96GB.
`KillOnInvalidDependent=Yes` prevents the fallback from running if the original succeeds.
The original seed-1 job remains RUNNING, with zero restarts and its unchanged 8-hour
limit / 10:20:18 Berlin deadline. No original job was cancelled, restarted or modified.

This is a complete reevaluation from the unchanged seed-1 2M checkpoint, not a resumable
continuation. The evaluator, source/data/config/checkpoint hashes, runtime versions,
fresh per-step GPI64x64, eval seed0, workers4, inference10k, and 500 episodes for each
of stand/walk/run/flip are unchanged. Only the output paths and scheduler fallback
settings differ. No training or test-time adaptation was added.

Recovery aggregate **2516143** is PENDING on successful original seeds0/2 plus retry
2516137, with 1 CPU, 8GB and 10 minutes. It uses the existing strict three-seed
aggregator and writes a separate report. The original aggregate2514955 is untouched;
the recovery branch auto-cancels if its dependencies become impossible.

New files and exact commands are under
`outputs/walker_identity_20260913/eval500_affine_20260914/seed1_retry12h/`:
`preflight.json`, both `.sbatch` payloads, both submission records, and
`postlaunch_verification.json`. Both payloads pass `bash -n`; Slurm's saved payloads
match the local files exactly. The existing corrected smoke report and validation
receipt were revalidated, as were all three checkpoint/config provenance records.
The retry has not started: its startup/final checks will run only if it becomes eligible.

Fresh progress at03:42 UTC: corrected affine Walker seeds0/1/2 are approximately
47%/36%/47% through logged rollouts, with no final reports yet. All six Stitch jobs
remain RUNNING. Point seeds0/1/2 are at311k/313k/313k of500k; full stored-posterior seeds
are at300k/305k/300k. Latest training metrics are finite. Saved sampler flags were
rechecked: the distribution arm is the full stored Gaussian sampler, not the shrunk
point-plus-extra variant. These are progress/numerical checks, not evidence of improved
policy performance or posterior accuracy.

Existing final reports were revalidated (hashes, 500-episode counts and aggregate
arithmetic), not rerun: the plain zero-shot Walker port has mean return694.131 with
three-seed95% t halfwidth62.030; raw-action RLU-derived full-affine Stitch has success
3.20% with halfwidth2.508 percentage points. The latter includes task-time constrained
inference/actor adaptation and is not training-free zero-shot extraction. The plain
Walker result validates that reference setup, not the identity-affine or flow interface.
No new final PSMFlow success result is available.

---


## 2026-09-14 02:46 UTC — Stitch full inversion and both C smokes passed; six seeds training

On the user's request to continue, fresh accounting confirmed fullB2513740 completed0:0
in4h37m03s. Point C200 smoke2513745 completed0:0 in11m34s; full-posterior C200 smoke2513889
completed0:0 in12m52s. These are numerical/wiring checks, not policy-performance results.

Full1M preimage, metadata and exact Stage-A500k checkpoint hashes were reverified.
The artifact has239 invalid rows (0.0239%). Both smoke checkpoint/config hashes and all
five two-episode task-report hashes were reverified; their training receipts record
finite checkpoint leaves and bitwise unchanged frozen flow parameters.

All six production jobs are RUNNING. Their own flags were read, confirming affine psi,
latent policy index, actorless GPI64x64, same flow/preimage artifacts,B1024,z128,gamma.99,
500k target updates. Point seeds0/1/2 have `use_point_preimage=true`; full stored-posterior
seeds0/1/2 have `false`, `measure_u_samples=1`, `measure_u_mixture_shrink=null`.
The latest logged steps were156k/157k/157k for point and147k/152k/150k for distribution.
All latest logged training metrics were finite. Final500-episode/task performance is pending.

Full-data distribution receipts still explicitly mark posterior accuracy uncertified:
passing numerical/wiring gates must not be reported as passing ESS/same-action fidelity.
No arm settings or experimental gates were changed.

Walker corrected affine evaluations2514951/52/54 are also RUNNING, roughly33%/25%/33%
through logged rollouts. No final reports are present. Seed1 still has8h and the unchanged
10:20:18 Berlin deadline; its average-rate projection is roughly9.23h, not a guaranteed
completion time. A conditional12h retry, only if the current evaluation fails/times out,
was proposed for user approval. **No retry was submitted and no existing job was changed.**

Verification record:
`outputs/psmflow_antmaze_stitch_20260913/reports/continuation_20260914_0246.json`.

---


## 2026-09-14 01:28 UTC — Requested Walker affine seed1 time extension denied

The user approved extending only affine Walker evaluation seed1, job2514952,
from8h to12h after its observed average rollout rate projected roughly9.4h total.
The exact job name, owner and RUNNING state were checked before the attempt.

`scontrol update JobId=2514952 TimeLimit=12:00:00` returned exit1:
`Access/permission denied for job 2514952`.
A subsequent scheduler read confirmed RUNNING, TimeLimit08:00:00 and the unchanged
scheduler-local deadline2026-09-14T10:20:18 (Berlin). **The extension did not happen.**
This was a Slurm permission denial, not a denied local-tool approval.

No job was restarted, cancelled or resubmitted; other jobs and all experimental
settings remain unchanged. An in-place increase requires cluster permission/admin
assistance. Exact before/after scheduler records and the denied command are saved in
`outputs/walker_identity_20260913/eval500_affine_20260914/seed1_time_extension_denied_20260914.json`.
The earlier progress projection included startup and is not a guaranteed completion time.

---


## 2026-09-14 00:21 UTC — Walker affine final evaluation running with corrected fresh GPI draws

The user requested the completed Walker affine PSM evaluation and explicitly approved
correcting its evaluation RNG. This is the **affine identity-decoder/no-flow control**,
not the RLU full-affine AntMaze adapter. Only affine seeds0/1/2 are evaluated here;
the free-head final evaluation was not additionally submitted.

**Confirmed evaluator bug.** The identity harness called `sample_actions(observations)`
without a key. The immutable agent defaulted to its unchanged `self.rng`, so each worker
reused its action/index panels across steps and episode batches, contrary to the planned
fresh draws. The original sampler regression failed with identical actions at identical
states; an explicit fresh-key adapter passes. This finding concerns the inspected Walker
identity harness, not the production PSMFlow evaluator. Existing small free/affine Walker
reports remain intact but do not measure the corrected fresh-draw protocol.

**Bounded correction.** `tools/walker_psm/eval_identity.py` supplies a new explicit key
per batched step, advancing across episode batches. Each task restarts the same eval-seed
stream for common random numbers, independently of the checkpoint training RNG. It wraps
the original evaluator in memory and retains original source/data/flags/checkpoint checks,
environment/reward code, task inference, candidate distributions and action scoring.
No training code, trained weights, original source manifests, flags or checkpoints were
changed. Full dynamic agent state, including targets/optimizers/RNG, is fingerprinted
before and after evaluation. New results are separate from the original eval JSONs.

**Fixed protocol:** unchanged2M checkpoints, seeds0/1/2, same5M ExORL Walker transitions,
stand/walk/run/flip,500 episodes/task, horizon1000, four workers, eval seed0,10k inference
rows,64x64 GPI, no actor or test-time learning. All saved model/training HPs are unchanged
(B1024,z128,gamma.98,tau.01,ortho1,bothLR1e-4). The full table was shown before launch.
Expected outcome is corrected evaluation, not an assumed performance improvement.

Campaign-local evaluation directory:
`outputs/walker_identity_20260913/eval500_affine_20260914/`.
Immutable `eval_manifest.json` binds all three checkpoint/config hashes and the separate
frozen evaluation adapter `code/eval_identity.py` (SHA256
`b788425ce7e74b0c2ba567b178c1cf262d48855818840a5bee47a185742a16fd`).
`preflight.json` is the earlier historical diagnosis, not the executable manifest.
Submission commands, timing caveats and observed job states are in `launch_manifest.json`.

| Job | Purpose | Observed at00:21 UTC |
|---|---|---|
| 2514903 | Restore seed0 at2M; corrected4 episodes/task smoke | COMPLETED0:0,5m14s |
| 2514951 | Corrected500 episodes/task, affine seed0 | RUNNING |
| 2514952 | Corrected500 episodes/task, affine seed1 | RUNNING |
| 2514954 | Corrected500 episodes/task, affine seed2 | RUNNING |
| 2514955 | Strict three-seed aggregate | PENDING after all three evaluations |

The smoke verified four full1000-step episodes per task,1000 fresh-key calls per task,
reproducible paired task streams and identical before/after agent-state SHA256. Its
scores are not benchmark results. Production startup records were re-read: all three
have the correct seed,2M checkpoint,500 episodes/task,workers4,K64 and fresh RNG mode.
Each gets one H100,8 CPUs,96GB and8h. Historical periodic-eval timing suggests roughly4.7h;
the corrected smoke includes startup/loading, so it is not an isolated acting-rate
measurement or a guaranteed ETA. Final performance is still pending.

Validation:35 focused tests passed, Ruff clean, five shell payloads passed `bash -n`,
and independent read-only code plus manifest/import/provenance reviews found no launch
blocker. Production requires the completed smoke receipt and validates complete task
reports, source/config/checkpoint pairing, fresh-key traces and unchanged state.
The dependent aggregator requires three distinct training seeds and500 episodes/task,
averages tasks within each seed, then reports the mean and95% Student-t interval.
Other campaigns and existing user work were not modified; no artifacts were published.

---


## 2026-09-13 21:21 UTC — Matched distribution-preimage stitch arm queued alongside point

The user challenged the point-preimage choice and explicitly requested a distribution
run as well. The preceding relaunch had inherited `use_point_preimage=true` from the
existing strict baseline; it was not selected by a new point-vs-distribution comparison.
The latest evidence does **not** establish distribution sampling as generally better:
September10 demotes the earlier shrunk-mixture Arm C gain after its longer continuation
and AntMaze comparison. Keep that distinct from this new stitch experiment.

**Requested arm and unchanged comparison.** New directory:
`outputs/psmflow_antmaze_stitch_20260913/distribution_v1/`.
Use `agent.use_point_preimage=false`, `measure_u_samples=1`,
`measure_u_mixture_shrink=null`: one fresh draw from the full stored K=1 Gaussian
per sampled transition, followed by the existing coordinate clip at3. This is NOT
point-plus-shrunk-extra-latents Arm C and not a multimodal posterior.
Resolved Hydra agent configs differ from the point arm only in
`use_point_preimage`; only output/run-group names change elsewhere.
All source/model/loss/inversion code is unchanged. The full hyperparameter table was
shown before launch. Seeds0/1/2,500k updates,B1024,gamma.99,d128,affine latent-index GPI,
no actor/DSRL/action critic, trainable phi, same rewards-only-at-readout protocol,
five tasks x500 final episodes, and same frozen BC control remain matched.

**Reuse, do not recompute.** Stage-B2513740 already computes both point and
distribution arrays. Both arms wait for that exact full1M artifact, native-row and
Stage-A500k pairing checks. No point job was modified or cancelled.
Distribution draws consume additional NumPy RNG values, so matching seeds/configs
does not mean identical minibatch streams between arms.

**Fresh posterior diagnostics, not policy results.** B256 contains no invalid rows;
weights are exactly1 and minimum symmetrized covariance eigenvalue is1.02e-6.
The real Dataset sampler at fixed row indices matched `sample_preimage_noise`
bitwise; consecutive visits redraw, resetting the seed reproduces, and draws differ
from stored points. Sample coordinate clip fraction is0.001465.
ODE-100 with the training clip (256 valid rows, one draw each) measured:

| Latent source | Mean action L2 error | p90 | Nonfinite decodes |
|---|---:|---:|---:|
| Distribution sample | 0.201563 | 0.295266 | 0 |
| Point inverse | 0.002531 | 0.000292 | 0 |
| Independent prior | 0.943807 | 1.516194 | 0 |

Stored un-clipped point roundtrip on these rows is0.000208; clipping explains its
larger evaluated mean/tail. Posterior samples are informative relative to the prior,
but fail the standing point-p90 same-action-fidelity diagnostic. D3 meanESS18.012
and B256 final-ESS16.75856 also fail the historical >20 criterion.
**Proceeding is the requested exploratory comparison, not posterior certification.**
Receipts explicitly carry `posterior_accuracy_certified=false`, numerical-check
scope and `exploratory_authorized=true`; full-data same-action fidelity is measured,
recorded and typed, not forced to either verdict. The point-only waiver is not
evidence that the distribution is accurate.

Independent review caught `diag_mixture_decode.py --ode` inheriting10 decode steps.
That completed ODE10 report is retained but quarantined by
`reports/diagnostic_limitations.json`; job2513837 reran the correct
`validate_decode_recovery.py agent.flow_steps=100` path.
One-step diagnostics use first4096 rows on the full artifact; ODE100 selects random
valid4096, so they are not row/RNG-paired. The one-step tool clips sample/prior but
not point/mean; that baseline is not a fully training-matched clipping comparison.

| Stage | Job | Observed at21:21:51 UTC |
|---|---|---|
| Shared full1M inversion | 2513740 | RUNNING; retained unchanged |
| B256 one-step + ancillary ODE10 diagnostic | 2513809 | COMPLETED 0:0 |
| Explicit matched ODE100 B256 diagnostic | 2513837 | COMPLETED 0:0 |
| Full distribution integrity/decode checks + C200 smoke | 2513889 | PENDING after successful2513740 |
| Distribution seed0 /1 /2,500k + five-task final eval | 2513895 /2513897 /2513898 | PENDING after successful2513889 |
| Distribution three-seed aggregate | 2513899 | PENDING after all three distribution seeds |

The full-data distribution gate pins the complete inversion recipe, subset, flow
and Stage-B receipt hashes, checks covariance/weights/validity, and exercises the
actual posterior sampler. The C200 path checks its own false/1/null flags, finite
losses/checkpoint/actions, frozen-flow bitwise equality and restored five-task
two-episode evaluation. Production readers require both distribution input and decode
receipts with hashes. Numerical failures block production; no silent switch to
point, shrinkage, different inversion settings or a reward-trained arm is allowed.

Validation:53 tests passed (13 preimage pipeline/validity +40 affine/index/eval-config),
seven shell payloads,32 inline Python blocks and26 Hydra commands parsed. Independent
read-only review's provenance/decoder cautions were incorporated. Source manifest
remains `439f3717ee47514ffd52afd063303ce86047c558388fd44189a52a2b395e68c3`.
An initial C-smoke submission was rejected because completed diagnostic2513809 had
left Slurm's live records; no job was created. Its COMPLETED0:0 accounting status and
saved diagnostic hashes were verified; the successful chain depends on active fullB
and checks those completed artifacts by hash instead. See
`reports/dependency_submission_note.json`.

Exact commands/IDs: `distribution_v1/launch_manifest.json`; verification:
`distribution_v1/reports/postlaunch_validation.json`. All C work is still pending,
not trained or performance-validated. Existing Walker/raw-action-affine jobs and all
user edits remain untouched.

---


## 2026-09-13 20:59 UTC — Stitch PSMFlow point-preimage relaunch is running

The user explicitly approved the point-specific relaunch after the original
ESS-only gate failure. This applies to the existing affine-psi, latent-index,
GPI, point-preimage PSMFlow arm, **not** the separate raw-action RLU full-affine
campaign. No model, loss, loader, inversion computation, decoder or hyperparameter
changes were made. The original trained Stage-A500k flow and completed same-flow
BC evaluation are reused; the 171-file frozen source manifest is unchanged.

The old D3 job2506527 failed only its mean-ESS>20 condition. Its dependent full-B
and all three Stage-C jobs were cancelled before starting; they have no results.
The relaunch lives in
`outputs/psmflow_antmaze_stitch_20260913/relaunch_point_v2/`, with separate exact
submissions and `launch_manifest.json`. The original launch records are historical.

**Bounded gate change.** Campaign-local `point_gate.py` treats finite, nonnegative
EM ESS as diagnostic-only for the explicitly checked point-preimage arm. It still
requires point roundtrip <0.1, central-99%-chi-square typicality >=0.95, the exact
trained flow/configuration and artifact hashes. Sampled and full-data receipts
also enforce these point metrics on valid rows, invalid fraction <=1%, and exact
native transition pairing. The existing EM+point validity/repair checks remain;
Gaussian computation is not skipped. Full inversion therefore remains expensive.
C readers require the explicit point-gate receipt fields before training.

Fresh D3 reproduced mean ESS18.012, mean point roundtrip0.000205 and typicality
0.9844. The original combined D3 report correctly remains `gate_pass=false`;
the separate authorized point receipt passes. The 256-row inversion smoke passed:
0 invalid rows, point roundtrip0.0002077203, typicality0.984375, diagnostic final
ESS16.75856, and all native transition columns bitwise matched. This is an
inversion diagnostic, never the Stage-C training dataset.

| Stage | New job | Observed at 20:59:12 UTC |
|---|---|---|
| Fresh D3 + point gate + B256 smoke | 2513739 | COMPLETED 0:0, 2m16s |
| Full 1M inversion + full point/artifact checks | 2513740 | RUNNING; source/data and authorized receipt verified |
| Full-artifact C200 + restored-action/eval checks | 2513745 | PENDING after successful full B |
| C500k seed0 + five-task final evaluation | 2513746 | PENDING after successful C smoke |
| C500k seed1 + five-task final evaluation | 2513748 | PENDING after successful C smoke |
| C500k seed2 + five-task final evaluation | 2513749 | PENDING after successful C smoke |
| Strict three-seed aggregate | 2513750 | PENDING after all three C jobs |

The C200 gate requires finite losses/checkpoint/actions and both frozen flow trees
bitwise unchanged, with five restored two-episode task evaluations. Production
remains 500k updates per seed and 500 final episodes per task over five tasks;
no actor or DSRL-NA branch, no reward-trained representation arm, no online data.
The reused BC control is the previously measured 32/2500 successes (1.28% mean);
it was hash-verified, not rerun. No new PSMFlow performance result exists yet.

Validation: 27 focused gate tests passed (`tests_final_formatted.xml`); Ruff
passed; seven shell payloads, 39 inline Python blocks and 28 Hydra configurations
parsed; all four C readers contain the point-specific receipt checks; expected
model flags equal the original submissions. Independent read-only review found
no remaining clean-run launch blocker. Scheduler resources/dependencies, fresh
D3/smoke receipt hashes, and Stage-A/source pairing were checked. See relaunch
`reports/submission_verification.json` and `reports/postlaunch_validation.json`.
Point-gate SHA256:
`1613a34fa199af3528d95d0045affd1a7d636f24724455abb4fbfa1e5f86329c`.
The full-B and C gates remain uncompleted; submission/smoke success is not an
implementation-validity or performance claim. Walker and raw-action affine jobs
were not altered; no publication or destructive cleanup was performed.

---


## 2026-09-13 — AntMaze oracle-aim is 0.966: the reachable set is fine there too; batch 512 is not a lever

### E1 on antmaze (`tools/diag_oracle_aim.py`, 500 ep, K=512, ODE-100)

The cube-only E1 row now has its antmaze counterpart. Same harness, same eval-seed
protocol, the frozen Stage-A flow `antmaze-medium-navigate@500000` for decode and
`fqlexpert_ant_a10/sd000@500000` as the oracle.

| arm | success | Wilson 95% |
|---|---|---|
| **oracle_aim** | **0.966** | [0.946, 0.979] |
| oracle (ceiling control) | 0.972 | [0.954, 0.983] |
| random_latent_onestep (floor) | 0.078 | [0.058, 0.105] |
| random_latent_ode (floor) | 0.048 | [0.033, 0.070] |

Both controls land where the record says they should: the ceiling matches the expert's own
0.95-1.00 in-loop evals, the one-step floor matches the antmaze BC control 0.072. That is
what makes the aim number readable.

**Verdict.** AntMaze is a selection failure, not a reachability ceiling. Drawing the same
512 prior latents the deployed GPI path draws and executing the one closest to the expert's
action reaches 0.966, against 0.078 for executing a random one and 0.294 pooled for the
trained agent. `COMPENDIUM` §4.3's conclusion — "the entire Stage-C loss is latent
*selection*" — now holds on both environments rather than on cube alone.

**One earlier reading is overturned.** The 09-02 smoothness probe recorded antmaze's
best-of-512 distance as 15.7% of the mean action norm (0.313 / 1.99) against cube's 6.9%
(0.060 / 0.87), and that was written up as the "third independent sighting of the
cube/antmaze split". It costs nothing in rollout: antmaze's aim arm (0.966) sits above
cube's (0.934). The distance gap is real and is not the antmaze problem.

Report: `$PSM_DATA/logs/oracle_aim_antmaze.json`. `tools/diag_oracle_aim.py` was restored
from the archive to run this and is kept.

### Batch size 512, both environments, 3 seeds each

Pre-registered before launch: no gain, and more swing between checkpoints, because the
measure loss sums `B^2 - B` off-diagonal pairs and 1024 → 512 cuts that term ~4x.

Runs: `bs512_cube` and `bs512_antmaze`, identical to `affine_strict_cube` and
`affine_strict_antmaze_g99` except `agent.batch_size=512`. In-loop 50-episode evals over
the 250k-500k ladder, 18 cells per arm:

| arm | mean | sd | swing per 50k |
|---|---|---|---|
| cube 1024 | 0.427 | 0.147 | 0.121 |
| cube 512 | 0.381 | 0.197 | 0.219 |
| antmaze 1024 | 0.267 | 0.184 | 0.147 |
| antmaze 512 | 0.188 | 0.201 | 0.188 |

Per-seed ladder means: cube 1024 0.350/0.500/0.430 against 512 0.393/0.297/0.453; antmaze
1024 0.197/0.183/0.420 against 512 0.090/0.097/0.377. Cell-level differences +0.046
(cube, permutation p 0.223) and +0.079 (antmaze, p 0.115).

**Reading.** Batch size is not a lever. The mean is lower at 512 in both environments and
neither gap separates from the seed spread. The swing between neighbouring checkpoints is
larger at 512 in both, as pre-registered, but that difference is p 0.058 on cube and
p 0.372 on antmaze, so it is not established either. These are 50-episode evals and do not
count as reportable numbers; the 500-episode ladder was not run.

### Training-log diagnostics from those six runs

`psm_offdiag` is the squared residual `M - gamma*target_M` on mismatched pairs
(`utils/psm_common.py:24-26`). It grows through training in every arm: antmaze seed 0 goes
754 → 11968 (16x) monotonically, cube seed 0 spikes to 36841 at 350k and 34129 at 500k. The
diagonal term stays flat near -170 throughout, so the growth is confined to mismatched
pairs. `psi_q_spread_rel`, the critic's score range across candidates relative to its own
magnitude, stays at 0.009-0.020 on antmaze and 0.026-0.058 on cube for the whole run — the
separating part does not grow with the scale.

Across 120 checkpoints, no training signal tracks the eval score: Spearman +0.124 for
`psm_loss`, -0.308 for `w_enc_spread` (which decays monotonically with time in every seed),
+0.270 for `psi_q_spread_rel`.

**Correction to an earlier note in this record's margins.** `index_agg=max` does not put a
max operator inside the measure backup under the default config. At `agents/psmflow.py:652`,
`policy_index=latent` sets `u_next = u_index`, a prior draw, so the TD target is
`psi_bar(s', u', u')^T phi_bar` with `u'` sampled and never maximised. Classic maximization
bias does not apply to the measure fit; the ensemble reduction is exact min, so the target
is biased low if anything.

### Repo: the archived agents moved to the `archive` branch

`archive/` (58 Python files) and `scripts/baselines/` (11 launchers) are off
`feat/inversion-integration` as of `daa0268` and live on the `archive` branch, cut at that
commit's parent. Five network classes (`PSMActor`, `AffineMeasureNet`,
`FactoredAffineMeasureNet`, `LagrangeNet`, `WNet`, -208 lines) and four helpers
(`proto_sample`, `_HashableDict`, `_step`, `_soft`, -31 lines) were dead once those left and
are deleted; each was checked for references outside `archive/` first. `pyproject.toml`
drops the ruff exclude and the pytest `norecursedirs` entry. Retrieval is
`git show archive:archive/<path>` to read and `git checkout archive -- archive/<path>` to
restore. Five test files (62 passed, 2 skipped) and the ruff count on `utils/` and `agents/`
(42 findings) are unchanged across the deletion.

---

## 2026-09-13 — flow-backed affine PSM on AntMaze medium-stitch queued

User additionally requested affine **flow** PSM on stitch. This campaign uses the
current `agent=psmflow`: affine successor features, latent current-action coordinate
and policy index, GPI argmax, no trained actor/DSRL-NA/action critic. It is distinct
from the archived raw-action RLU full-affine solver below. No live algorithm code,
loss, loader or inversion routine was changed. [Full binding recipe and plan](plans/2026-09-13-psmflow-antmaze-stitch.md).

Campaign: `outputs/psmflow_antmaze_stitch_20260913/`. Exact submissions/dependencies:
`launch_manifest.json`; saved wrap payloads: `stage_a_submission.json`,
`stage_b_submissions.json`, `stage_c_submissions.json`, `aggregate_submission.json`.
Frozen171-file working-tree snapshot: `code/`, source-manifest SHA256
`439f3717ee47514ffd52afd063303ce86047c558388fd44189a52a2b395e68c3`.
The existing shared `.venv` is reused, with package versions recorded rather than
claimed as an independently frozen installation. No existing Walker/raw-action
affine jobs were altered, and no artifact was published to Hugging Face.

**Data:** Reuses the exact Drive files already downloaded for the raw-action campaign;
both hashes and the native1M/100k train/validation pairing were freshly verified.
`envs.env_utils` does not pass a dataset directory and installed OGBench ignores
`OGBENCH_DATASET_DIR`. Approved non-overwriting links in `/mnt/home/amohan/.ogbench/data/`
point to the original stitch NPZs. No navigate dataset is substituted. Obs29/action8,
five tasks,1000-step horizon; provenance in `reports/runtime_preflight.json`.

**Recipe:** One shared seed0 behaviour flow,500k BC-only updates, B256,LR3e-4,
4x512,distillation10,100 flow steps. B retains the existing point-plus-EM path:
alpha50,prior1,N200,10EM steps,100 inverse steps,K1,B256,full1M transitions.
Representation seeds0/1/2 each500k updates,B1024,d128,2 ensembles,gamma.99,tau.01,
ortho1000,LRphi1e-5/LRsf1e-4,affine latent index,point preimages,64x64 GPI,
frozen one-step decode. Phi trains; flow does not. Training rewards/masks are not
read by the default measure loss; rewards enter task-vector inference/evaluation only.
`mask_invalid_preimages=false` is the unchanged loss flag, but the existing dataset
sampler independently excludes invalid preimage rows. This is an untuned transfer,
not a matched-architecture causal comparison with the raw-action full-affine arm.

**Verification before release:**64 focused tests passed (10 BC/aggregation +54
affine/index/preimage/GPI), with JUnit records. Stage-A full-size200-update H100 smoke
2506433 COMPLETED0:0 in1m01s; all logged train/validation losses and171 serialized
checkpoint leaves were finite. Its flags/data/source/checkpoint binding is recorded
in `reports/stage_a_smoke_receipt.json`. Independent read-only launch review found
no clean-run blocker; all34 scheduled Hydra invocations composed, shell and inline
Python parsed. Review feedback added exclusive B-output guards and2 CPU threads per
evaluation worker. None of this establishes stitch benchmark performance.

**Observed at2026-09-13 00:56:45 UTC:** Stage-A production2506471 RUNNING on one H100,
257k/500k logged updates, all metrics finite, own flags verified unchanged. Inversion
gate2506527 is PENDING after Stage A; full inversion2506529 after that gate; full-data
Stage-C200-update smoke2506567 after full inversion. Stage-C seeds0/1/2 jobs
2506571/2506573/2506574 are PENDING after successful C smoke, **not running yet**.
Same-flow BC evaluation2506575 waits on Stage A. Aggregate2506577 waits on all three
C train/final-eval jobs and the BC control. All dependency IDs and requested resources
were checked against the scheduler in `reports/postlaunch_validation.json`.

**Pending gates/results:** D1 is descriptive; D3 must meet its existing gate:
chi-square99%-band typicality>=.95, mean roundtrip<.1, final ESS>20. No automatic ESS
waiver is applied for point mode. A sampled256 inversion checks plumbing/pairing but
is never C input; full B checks all transition columns against the native loader and
binds NPZ/sidecar/flow hashes. C smoke checks200 updates, finite checkpoint/actions,
bitwise-unchanged velocity and one-step flow trees, and restored2-episode evaluations
on all five tasks. Only successful gates release production. Final saved500k endpoints
and the same-flow BC control get500 episodes/task, five tasks,4 workers; the strict
aggregate averages tasks within seed and then computes the three-seed Student-t95% CI.
Failed prerequisites prevent downstream execution; no method/seed/checkpoint fallback.
Inversion has a48h allocation (not an ETA). No completed stitch result is claimed.

---

## 2026-09-13 — full-affine PSM on AntMaze medium-stitch queued

User requested our full-affine PSM on the exact assets from Drive folder
`1OqKCtqNwQoS7gUmM1QaFc8NN5aYHqBbt`. This campaign uses the archived RLU-derived
`AffinePSMAgent`, **raw actions and full goal inference**, not PSMFlow and not the
plain Walker zero-shot PSM baseline. [Declared recipe and completed launch plan](plans/2026-09-13-affine-antmaze-stitch.md).
Campaign: `outputs/affine_antmaze_stitch_20260913/`; exact job record:
`launch_manifest.json` there. Walker jobs/files were not modified.

**Data:** Downloaded the exact Drive `antmaze-medium-stitch-v0.npz` and `-val.npz`.
SHA256, ZIP integrity, finite schema/action bounds and native OGBench sentinel pairing
were verified. The files contain1M training transitions in5000 short200-step episodes
and100k validation transitions in500 episodes. Observation/goal width29, action8,
five fixed tasks,1000-step evaluation horizon. Training and inference use only the
training split; batches exclude rewards, masks and terminal indicators. Exact hashes,
Drive IDs and raw source evidence are in `data/provenance/manifest.json`. Official
OGBench names/content lengths match; no official publisher cryptographic checksum
was available, so this is not a claim of independently verified cross-host byte equality.

**Recipe is an untuned AntMaze transfer, not reference parity:** seeds0/1/2,500k updates,
B1024,d=z128, factored measure rank32,width1024,depth3,bias scale10, gamma.99,tau.01,
all LR1e-4,ortho1000, DDPG+BC actor with BC.3,width1024,hidden1,embedding2. Proto16-bit
codebook uses archived draws shifted+1 to correct the released[-2,0) range to[-1,1).
No archived loss/inference code changed. Full primal-dual goal inference uses5120
updates with a normalized coordinate and dual256x2, followed by exactly512 actor
adaptation updates per task. It is task-time optimization, not training-free extraction.
Evaluate10 episodes/task every100k and500/task at the saved500k endpoint; average
tasks within each seed, then compute a Student-t95% interval across the three seeds.

**Implementation verification:** isolated `tools/affine_stitch/` adapter,12 passing
focused CPU tests, clean Ruff/shell syntax and an independent read-only review with
no critical or important adapter issue. Full-size CPU construction/data loading,
checkpoint restoration and all five full29-D goals passed. OGBench uses global NumPy
for initial-position noise before its seeded reset; the adapter seeds/restores this
stream along with environment/action-space RNGs. Inference has an independent dataset
RNG and every task starts from the same pre-inference agent. It verifies unchanged
representation parameters/optimizer states/targets and exactly-once actor adaptation.
Source/data manifests bind through flags to checkpoints; final evaluation restores the
saved endpoint. Artifacts are exclusive and the archived source is checksummed in
`code/`, manifest SHA256 `b8138ff7e12fb88a47a2ba61bff18ae2634854fca4b5acc37b48e7bca1c8c0de`.

**Numerical edge case found, not silently fixed:** a synthetic width8,d4,rank2 fixture
reaches an exactly zero x-side basis. Its forward loss is finite but `psm_norm` has NaN
gradients at zero; measure parameters become nonfinite before task inference. A
width32 fixture passes deterministic inference. The small failure remains an expected
rejection regression; evidence is `tiny_feature_diagnostic.json`. This does not show
that the same event occurs in the full-size campaign or explain historical PSMFlow
performance. The archived normalization remains unchanged for this run.

**GPU gate passed:** smoke2506198 COMPLETED/0:0 in3m32s,200 full-size updates plus
full5120+512 inference and2 rollout episodes on each of five tasks from the restored
checkpoint. Logged losses/actions/coordinates were finite, representation hashes
unchanged and full goals identical to CPU preflight. Flags and source/data/checkpoint
binding verified in `smoke_validation.json`. Its zero successes are wiring diagnostics,
not a benchmark result. Post-compile70.633 updates/s projects1.966h for500k training
alone; including evaluation via short-run extrapolation projects2.757h, not a guarantee.

**Production submitted and observed RUNNING:** seed0 job2506231, seed1 job2506233,
seed2 job2506232, one H100/24h each, using the passing smoke's exact `code/` snapshot.
At2026-09-12 23:59:25 UTC, each run's own declared/resolved config, bounded proto
range, data paths,500k budget, final500-episode protocol and source/data hashes matched.
Seeds0/1/2 had reached6k/5k/5k logged updates with finite metrics. Full evidence:
`postlaunch_validation.json`. Checkpoints save optimizer/RNG state but this adapter
does not implement interrupted-training resume. Aggregate job2506237 is PENDING on
successful completion of all three training/final-evaluation jobs and will write
`aggregate.json`. No completed stitch benchmark result is claimed.

---

## 2026-09-13 — Walker no-flow PSM setup and RLU source-lineage correction

The user authorized downloading ExORL and running a raw-action PSM positive control.
[Declared recipe and launch plan](plans/2026-09-13-walker-zero-shot-psm.md).
Campaign: `outputs/walker_psm_20260913/`; job record: `launch_manifest.json` there.

**Data acquired and validated:** official `denisyarats/exorl` Walker RND ZIP, 10,000
episodes /10M transitions. All ZIP CRCs, episode schemas, numeric finiteness and
checksums passed. First 5,000 numeric episode IDs provide the declared5M training
transitions. Observations24, actions6, physics18; every environment discount is1.
Loader pairs `obs[t-1], action[t], obs[t], physics[t]`, excluding dummy initial rows
and all collection rewards. Provenance: `data/provenance.json`, archive SHA256
`384bc064ea62ccea19e5be22646d2af7eb5d1c2b19bf3dbb23b7bb5641b63b1c`.

**Implementation:** isolated archived raw-action `PSMAgent` and matching utilities,
archive commit `ee0e1746b36f48300eee06ad02e66b16f46851ff`; no live agent restored.
New harness: `tools/walker_psm/`, `scripts/slurm/walker_psm.sbatch`, and focused tests.
The final harness has **10 passing CPU tests**, including changed-sidecar rejection.
Independent review verified real physics-to-observation agreement on five rows in all
four tasks (max absolute error3.57e-7) and approved the source/data-to-checkpoint binding
fix. Full-size construction, episode horizons, and checkpoint restoration were checked.
Test reports and preflight JSON are retained in the campaign.

**Protocol is explicitly not exact published replication.** The official zero-shot
PSM release revision `b1a2e7f388f789a0d6abaabd317692c1f50f42b2` samples proto actions
as `2*(rand-1)` in[-2,0), feeding them unclipped to its TD target despite Walker's
[-1,1] bounds. The primary baseline corrects this to `2*rand-1`, preserving per-row
Torch seeds; the exact released table/mode remain available but are not the primary
run. The bounded table matches released+1 exactly, has zero out-of-bounds rows, and
hashes to `730718b792a05097adbe1d22af8ffa3375380f5b69cd99481232d0fd2494130e`.

Recipe: seeds0/1/2,5M transitions,2M updates,B1024,d128,gamma.98,tau.01,all LR1e-4,
ortho1,raw TD3-style actor with BC0,target pessimism0,actor pessimism.5. Numerical
architecture/optimizer settings follow released code; data/update budget follow the
paper. JAX initialization/sampling and the pre-proto task-mixture feature timing also
differ from upstream execution. Stand/walk/run/flip use fixed10k reward inference rows,
10 periodic episodes/task and500/task at the saved2M checkpoint; aggregate across
training seeds after task averaging. Published Table1's Walker average689.07 (five
seeds) is context, not a newly reproduced score or an exact-protocol comparator.

**GPU gate:** initial smoke2505858 completed0:0 in3m10s, with200 full-size updates,
finite logged losses, restored-checkpoint four-task evaluation (2 episodes/task),
and the displayed settings in its flags. Post-compile rate132.718 updates/s projects
to4.186h for2M updates alone; startup/evaluation are additional. Its two-episode
returns are wiring diagnostics, not benchmark evidence. Metadata-only provenance
hardening created snapshot `code_v2`; final smoke2505882 completed0:0 in2m19s in
`smoke_v2/seed0`. Its flags, restored200-update checkpoint evaluation, and checkpoint-
bound source/data hashes passed the launch gate.

**Production submitted and observed RUNNING:** seed0 job2505905, seed1 job2505904,
seed2 job2505906, one H100 each with24h allocation, all using `code_v2`. Aggregate
job2505911 depends on all three training/evaluation jobs succeeding and will write
`outputs/walker_psm_20260913/aggregate.json`. Startup validation passed at
2026-09-12 23:03:36 UTC: every declared agent/run setting matched each run's own flags,
source/data hashes agreed across seeds and with the bound sidecars, and seeds0/1/2
had advanced to5k/9k/5k with finite logged metrics. Full check:
`outputs/walker_psm_20260913/postlaunch_validation.json`.
The runner saves optimizer/RNG state but does not currently
support interrupted-training resume; the observed update rate fits comfortably inside
the allocation with evaluation headroom. No completed Walker benchmark result is claimed.

**RLU correction:** the supplied full-affine source is
`CalCharles/RLU/controllable_agent`, not the separate zero-shot PSM release. The URL
was not freshly accessible; local archive/design/code provide the inspected lineage.
Archived `AffinePSMAgent` has goal-conditioned constrained coordinate inference and
actor distillation, unlike current PSMFlow's GPI-only extraction. Its normalized
source and factored bias branches also differ. The archived design explicitly deferred
reward-based full inference; dense Walker rewards need that interface before this can
be called a full-affine Walker run. A plain zero-shot PSM result does not validate
that separate method or, by itself, isolate the frozen-flow interface.

---

## 2026-09-13 — interface audit, diagnostic corrections, freeze-phi final result

[Full audit and Walker protocol](design/2026-09-13-psm-interface-audit.md).
No production training code changed, no new long job launched, and existing uncommitted
work was preserved. Historical policy evaluations below were revalidated from their
reports; they were not rerun today.

**Fresh implementation validation:** 97 tests passed with 2 checkpoint-gated skips; the
two real-flow cases were then enabled using the local cube Stage-A checkpoint and passed,
together with the default affine chain-task test. Total: **100 distinct PSMFlow tests
passed**. The sibling Factored-FB PSM fixture suite passed **35 tests**; that fixture
isolates network/loss transcription using injected actions and post-Torch-update phi, so
it does not validate full original-PSM optimizer/sampler equivalence or Walker performance.
Persisted outputs: `outputs/psm_interface_audit_20260913/`.

**The freeze-phi job is complete.** Current Slurm accounting confirms job2494139
COMPLETED/0:0. Its ten endpoint reports use 500 episodes/task, seed0, four workers, and
absolute step500k (50k source +450k continuation). Configurations match except `train_phi`.

| Arm | task1 | task2 | task3 | task4 | task5 | Five-task mean |
|---|---:|---:|---:|---:|---:|---:|
| Continue phi | .076 | .004 | .000 | .000 | .002 | **.0164** |
| Freeze phi | .128 | .000 | .000 | .000 | .000 | **.0256** |
| Same-flow BC | .056 | .174 | .026 | .006 | .100 | **.0724** |

Source task1 was .844 at50k. Both arms lose most of that performance. Completion checks
confirm frozen phi/optimizer preservation, synchronized target phi, unchanged flow,
finite psi states, and matched RNG streams. Endpoint Gram deviations are .347 frozen
and .350 control. Continuing phi movement is not necessary for this continuation's
performance loss; freezing this basis was insufficient. This is one training seed,
not an across-seed inference or a validated alternative basis experiment.

**Three earlier causal interpretations need correction:**

- D1b (`phi_readout_fixed`) holds w and scale, but reads the changing online phi each
  step. It does not hold the reward function fixed between refits unless phi is frozen.
  A small-agent probe changed the fitted reward by .0385 while keeping w/scale exact.
- D2 (`synthetic_w`) still uses task-specific success masks in its TD target. The actual
  preimage files contain 20,810/1M cube and 9,113/1M AntMaze zero masks; masks equal the
  negative task rewards. Its loss is invariant to changing real rewards but changes
  when masks change. This is synthetic-reward learning with task-specific termination,
  not strictly task-agnostic reward-free pretraining. Its measured failures still stand.
- The fitted-return tool evaluates each policy under its own phi/w/scale. Its historical
  7.805x ratio compares different surrogate rewards and cannot demonstrate improvement
  under one fixed surrogate. Freeze one reward evaluator and cross-score policies before
  using that causal reading. The tool also refits w instead of reading D1b's held w.

**Structural counterexample:** marginal behavior matching does not establish joint
action/policy-index coverage. An exact Gaussian identity flow, fixed phi, and realizable
affine model give projected-TD multiplier **1.086396 at gamma .9**, verified analytically
and numerically. This establishes a possible interface failure, not the live network's
root cause. The exact two-head minimum Bellman operator remains gamma-contractive in
sup norm; the September8 disagreement-growth fit was not a proof to the contrary.
Projection can break stability. Density minimum still loses mass and is not pessimistic
for every signed reward. The existing P=1 uncertainty-helper NaN was reproduced; default
P=2 is unaffected.

**Walker:** environment construction/reset/step passed; no Walker RND buffer or Walker
Stage-A/B artifacts were found in inspected project/home storage. The official PSM paper,
released defaults, archived cube port, and current affine agent are different protocols.
The audit specifies reference-equivalence and original-PSM positive-control gates, followed
by a matched identity-decoder free/affine pair. Plain PSM versus full PSMFlow alone changes
the policy family and extraction algorithm as well as the flow. No Walker run is claimed.

---

## 2026-09-11 — exact-action affine PSM: final 500k results, AntMaze collapses

All six runs and aggregate job2493461 completed0:0. All30 task reports were revalidated:
exact500k checkpoints,500episodes/task, five tasks per training seed, matching saved
configurations, seeds0/1/2. The fixed endpoint results are:

| Environment | seed0 five-task mean | seed1 | seed2 | Mean ±95% t halfwidth |
|---|---:|---:|---:|---:|
| cube |41.84%|60.16%|29.76%|**43.92% ±38.02 percentage points**|
| antmaze |0.72%|1.76%|0.76%|**1.08% ±1.46 percentage points**|

These are across-training-seed Student-t intervals,df2, after averaging tasks within
each seed; they are not episode-level intervals. The symmetric AntMaze interval extends
below zero and is left untruncated in the artifact. The recipe misses the requested70%
target and does not outperform all TD-JEPA comparator means. Cube's point estimate is
above the historical strict affine25.33%, but uncertainty is too wide for a reliable
improvement claim. Its same-flow five-task BC control is11.08% (one shared frozen flow,
500episodes/task). The matched AntMaze BC control job2493549 also completed0:0:
task scores are **5.6%,17.4%,2.6%,0.6%,10.0%**, a five-task mean of **7.24%**
(one shared frozen flow; these are episode estimates, not an across-training-seed CI).

The AntMaze collapse is confirmed at500episodes on the same task and worker protocol:
task1 moves from **.844/.564/.752** at50k (mean.720) to **.036/.004/.038** at500k
(mean.026). Early50k was chosen retrospectively for diagnosis; it cannot replace the
registered500k endpoint. Reports and aggregates:
`outputs/affine_action_20260911/reports/{cube,antmaze}_aggregate.json` and
`antmaze_50k_task1_diagnostic_summary.json`.

The paired freeze-phi diagnostic is implemented and its 200-step GPU smoke job
**2494102** completed0:0. Both arms reached absolute step50200 with finite losses,
exact frozen-phi/optimizer preservation, synchronized target-phi, advancing control phi,
unchanged flow weights, and matched RNG streams. This is only a wiring check (2 episodes).
The predeclared 450k continuation, job **2494139**, is running from the seed0 50k
checkpoint; it shares each batch between continue-phi and freeze-phi arms and counts
50k source +450k new updates as500k total. See [the protocol](design/2026-09-11-freeze-phi-diagnostic.md)
and `outputs/freeze_phi_20260911/`.

The pilot's provisional in-loop task1 readout is mixed: at absolute100k, frozen/control
were **70%/36%** on50 episodes; at absolute150k **10%/18%**; and at absolute200k
**10%/12%**. These are noisy diagnostics, not endpoint evidence, and the continuation
remains in progress.

**Next-step decision after interim AntMaze collapse:** verify the saved50k checkpoints
with 500 episodes, then conditionally run a matched continue-versus-freeze-phi pilot.
The early .76/.56/.66 and later350k .06/.02/.14 are only task1 in-loop50-episode scores.
The subsequent500-episode50k check confirmed **.844/.564/.752**, mean **.7200 ± .3545**
(three-training-seed Student-t95% halfwidth,df2), on task1 only. This is a retrospective
diagnostic and does not meet the500k/five-task benchmark protocol. Evaluation jobs
2493462/2493463/2493464 (seeds0/1/2) completed0:0; settings and the full calculation are
recorded in `outputs/affine_action_20260911/early_eval_manifest.json` and
`reports/antmaze_50k_task1_diagnostic_summary.json`. The freeze implementation and smoke
are complete; production pilot job2494139 is still running. Both branches synchronize
target/online phi before the fork and use the same data/RNG sequence for450k further
updates (50k source +450k continuation =500k total). This isolates ongoing feature
learning before introducing a HILP/RLDP objective. [Diagnostic design](2026-09-11-freeze-phi-diagnostic.md).

The user requested at least 70% success with
three-seed intervals and superiority to FB/RLDP/HILP as reported by TD-JEPA. Its state-based
OGBench table uses five-task averages, 1M updates and ten seeds (printed uncertainties are
SE): cube HILP **74.20 ± 3.53**, FB 49.60 ± 3.83, RLDP 19.80 ± 2.41; antmaze HILP
**83.60 ± 2.63**, FB 73.00 ± 2.72, RLDP 74.60 ± 4.15. Thus 70% alone does not beat all
comparators. [Primary paper](https://arxiv.org/html/2510.00739). The phrase “with steering”
was not a named variant in the official paper/repo; clarification remains pending.

The controlled change is `measure_action_input=action` (default remains `latent`):
`F(s,a,c)=A(s,a)c+beta(s,a)` trains at recorded dataset actions; latent queries use
`F(s,G(s,u),w_enc(u'))`. Bootstrap and acting use the same frozen decoder. This removes
approximate-preimage/decoder-coordinate inconsistency while keeping the affine policy
index, current losses, gamma, reward-free training and flow GPI. Affine GPI now factors
its K-by-K scoring panel in this branch; tests check explicit-pair equivalence and ties.

Search/math audit: the marginal in-sample C=1 argument does not prove joint
`(state, action, policy-index)` coverage or projected-TD stability. Arbitrary affine-coordinate
LP optimization is not justified by a feasible occupancy set in the current implementation.
Strong reward-trained DSRL steering also does not prove good constant-index continuation
values. Reference RLDP/HILP supply possible reward-free basis objectives if this arm fails;
basis training must count within the 500k budget. Details and sources:
[design](design/2026-09-11-affine-action-conditioning.md).

Validation: 9 action/parity tests, 11 config tests, 12 legacy affine tests, 18 core tests
passed (2 checkpoint-dependent skips); 5 reporting tests passed. Independent review checked
the action paths, decoder gradients/frozen weights, factorization, snapshot and reporting.
GPU jobs **2493453** (cube) and **2493454** (antmaze) each completed 200 steps with exit
0:0. Saved flags match the manifest; losses are finite. These are execution checks.

Artifacts: `outputs/affine_action_20260911/`. `code/source_manifest.json` hashes 159 source
files from the dirty tree; the job verifies hashes and imports that snapshot.
`experiment_manifest.json` includes full resolved agent configurations from both smokes.
Cube gamma .98, antmaze .99; seeds 0/1/2; frozen Stage-A epoch500k shared across seeds;
Stage-A training cost remains separate from the 500k Stage-C budget.

| Environment | seed 0 | seed 1 | seed 2 |
|---|---:|---:|---:|
| cube | 2493455 | 2493456 | 2493457 |
| antmaze | 2493458 | 2493459 | 2493460 |

Each job trains to exactly 500k and then evaluates all five tasks, 500 episodes each,
with four workers and eval seed0. Job **2493461** depends on all six succeeding and writes
`reports/{cube,antmaze}_aggregate.json` using `tools/report_seed_comparison.py`.
The tool checks restored flags, distinct training runs, matching configs, complete task
matrices and episode/epoch provenance; it first averages tasks within each training seed,
then reports mean ± Student-t 95% CI across three seeds (df=2). No best checkpoint selection.

Re-audited historical cube control at exactly 500k: five-task means per seed
**.156/.308/.296**, overall **.2533 ± .2099** (three-seed t95 halfwidth), from
`$PSM_DATA/logs/eval500_affine500k_strict_cube[_task{1,3,4,5}]_sd{0,1,2}.json` (unqualified
names are task2). No complete strict antmaze five-task matrix was found; its task1 result
must not be substituted for a benchmark mean. Old runs differ in some subsequently added
default config keys, so the new strict raw-config reporter is not silently relaxed for them.

Resume by inspecting `jobs.json`, Slurm state, saved `train.csv` and aggregate reports.
Do not relaunch completed work. Do not run raw-`psi` calls in `diag_gpi_selection`,
`diag_actor_grad_terms` or `diag_latent_ranking_oracle` on the action branch without adapting
them to the decoded-query helper. The target is still unproven while jobs run.

---

<!-- _class: lead -->

## 2026-09-10 — five pre-registrations scored, four failed, and antmaze is steerable

All 500 episodes, `stability_ladder.ladder_from_eval500` off the run's own reports.

---

### 1. Antmaze CAN be steered. **0.967** through the frozen flow.

`dsrlna_antmaze`, DSRL-NA with the task's real reward, gamma 0.99, Item 2 recipe otherwise
unchanged. Six cells:

| seed | 250k | 500k |
|---|---|---|
| sd0 | 0.938 | 0.982 |
| sd1 | 0.972 | 0.978 |
| sd2 | 0.956 | 0.976 |

Pooled **0.967**, flat, against BC **0.072** and the zero-shot antmaze control's **0.294**.
NOT zero-shot and not quotable beside a zero-shot row. What it settles: antmaze is not
substrate-capped the way pointmaze is, and its zero-shot failure is the value, exactly as on
cube (0.910). Two environments now say the same thing.

---

### 2. Four pre-registrations, scored

| arm | pre-registered line | result | verdict |
|---|---|---|---|
| **D1b** (readout as reward, fit as deployed) | near 0.9 exonerates the readout; well below 0.415 makes it the wall | **0.008 / 0.010 / 0.014** | the readout is the wall |
| **Arm C + gradient actor** | above 0.14 means the value has slope; near 0.14 means it does not | **0.000 x3** | no usable slope |
| **Arm C 1M continuation** | no seed falls back below its 500k value by more than 0.08 | sd0 0.792 → 0.654 → **0.612**; sd1 0.804 → 0.758 → 0.846; sd2 0.322 → 0.086 → **0.012** | **FAILED** |
| **Arm C dose response** | monotone in c if the shape is the mechanism | c=0.3 **0.472**, c=0.5 **0.515**, c=0.7 **0.409**, control **0.426** | non-monotone, all inside the control's band |

**Arm C is demoted.** Yesterday's entry (6e) recorded a monotone 9/9 rise and said the
continuation would decide it. It decided against: seed 0 falls 0.18 and seed 2 collapses by
0.31 over the next 500k, both far outside the ladder's own 0.083 step. The rise was a slow
transient. Add that Arm C does not replicate on antmaze (0.285 against the control's 0.266 on
matched cells, peaking at 100k-250k and collapsing to 0.02-0.08 by 500k), and that the dose
response is flat in `c`, and what is left is one favourable window on one environment.
**Arm C is not a fix; it is back to being a stability question.**

---

### 3. The pattern worth carrying: optimising the fitted reward is what fails

| arm | how hard it optimises the fitted reward | 500-ep |
|---|---|---|
| Arm C + gradient actor | gradient ascent on `psi^T w` | **0.000** |
| D2 (synthetic w) | TD + actor on `phi^T w` | **0.004** |
| D1b (readout as reward) | TD + actor on `phi^T w_hat` | **0.011** |
| BC control | not at all | 0.072 |
| 09-08 faithful DSRL arm | actor on `psi^T w` | 0.136 / 0.141 |
| affine strict, best-of-64 GPI | argmax over 64 prior draws | 0.380 |
| Arm C, best-of-64 GPI | argmax over 64 prior draws | 0.503 |
| DSRL-NA, REAL reward | TD + actor on the true reward | 0.910 (cube) / 0.967 (antmaze) |

Three of the four arms that optimise the fitted reward hard land **below** the
behaviour-cloning control; the fourth lands at about twice it and a third of the argmax.
The ordering runs the wrong way for an optimisation problem and the right way for a
**task-inference** problem: the harder an arm pushes on `w^T phi`, the worse it does, and the
same machinery pushed on the TRUE reward reaches 0.91-0.97. The proposed reading is that
best-of-64 survives because it barely optimises. That is a hypothesis about a pattern, not a
measurement — `tools/diag_fitted_vs_true_return.py` is what tests it, by scoring the true and
the fitted discounted return under one rollout.

---

### 4. Two more proxies that do not predict success

Both were reached for as cheap stand-ins for a 500-episode eval. Neither survives.

- **`na_signal_over_disagreement`** — the std of the latent critic over `u` divided by the
  action critic's ensemble disagreement. Antmaze DSRL-NA reads **0.39-0.62** and scores
  **0.967**; cube DSRL-NA reads **0.94-2.23** and scores 0.910; D1b reads **0.44-1.15** and
  scores **0.011**. Three-to-one spread in the metric, no relation to the outcome.
- **top-8 ordering agreement with an expert critic** — retired 2026-09-09; see the design
  doc's RETIRED block. Predicts nothing about which checkpoint succeeds (rho +0.371, p 0.24
  at n=12, against +0.721 at the n=10 it was adopted on).

With reward-reconstruction quality (09-09 6b) and roster-ranking quality (09-09 3b), that is
**four** proxies for "is this critic good" that fail to predict the policy. Only the
500-episode ladder has ever tracked the outcome. Stop building diagnostics that predict
success and build ones that explain a mechanism.

---

---

<!-- _class: lead -->

## 2026-09-09 — the reward readout is the wall, and the ortho loss was telling us

Design doc: `docs/design/2026-09-08-critic-signal-and-dsrl-na.md`. Everything below is
CPU-measured or in-loop unless it says 500 episodes; the 500-episode evals are queued.

---

### 1. The one-step ground truth was invalid, and the Item 1 conclusion is WITHDRAWN

The rewritten GPI-selection probe (success ONSETS, 256 states, discounted at the training
gamma, three continuations) carried a check on the ground truth itself: rank the candidates
by distance to a frozen FQL expert's action. E1 says executing that candidate at EVERY step
scores 0.934, so it must rank a one-step return.

**It ranks worse than random** — regret 8.74 against a random pick's 8.58, rho +0.022. And
not for want of a good candidate: at K=64 the closest sits **0.049** from the expert's action
against a roster mean of 0.174 (mean ||a|| = 1.231).

So "the signal exists and both critics miss it" is withdrawn. What replaces it is sharper:
**one near-expert action followed by 49 steps of behaviour cloning is indistinguishable from
one random in-support action followed by the same 49 steps.** The value of acting well on
cube is not located in any single step — which sits exactly between the two anchors, 0.934
aiming every step and 0.072 never aiming.

---

### 2. STEP A — ordering agreement, and a diagnostic that needs no simulator

If a rollout cannot score a ranker, score the ranker against a ranker. Per state, over the
same roster: does the checkpoint order candidates the way a frozen expert's critic does, and
does agreeing predict that checkpoint's 500-episode success?

| statistic | rho vs success (n=10) | exact p |
|---|---|---|
| **top-8 overlap with the expert CRITIC** | **+0.721** | **0.024** |
| top-8 overlap with expert DISTANCE | +0.382 | 0.279 |
| full-ordering rho, either reference | +0.56 / +0.58 | ~0.09 |

Two seeds move in opposite directions and agreement follows each. **Adopted as the Arm B /
Arm C readout beside ladder swing**; the full-ordering Spearman is suggestive only. Agreeing
with a VALUE ordering predicts success; agreeing with an IMITATION ordering does not.

Note the scale: the same critic reads 0.16-0.27 against the expert's ordering where it read
0.04 against rollout returns. The return was the insensitive instrument.

---

### 3. Item 2 — a real DSRL-NA arm. The frozen flow is NOT the constraint on cube.

The repo's "DSRL-NA" had copied DSRL's actor and none of its critic. Added the missing half
(`dsrl_na`, default off): `qa(s,a)` by TD on the real reward, distilled into `qw(s,u)` at
prior draws, actor climbs `qw` only.

**In-loop 50-episode, NOT reportable; eval500 queued (2492559-64):** 0.80-1.00 across eleven
checkpoints and three seeds, from 50k steps on. Against BC 0.072, actor-free GPI 0.415, the
09-08 "faithful DSRL" arm 0.136/0.141, per-task FQL 0.949. The pre-registered bar was 0.5 by
250k.

Three readings. (i) A reward-specific critic steers this exact frozen decoder to near the
per-task reference, so the flow/inversion is not the binding constraint on cube and the
zero-shot representation is the whole gap. (ii) **The arm is FLAT** — 0.80-1.00 across the
ladder against the zero-shot arm's 0.086 <-> 0.704 traverse within one seed. The oscillation
is a property of the zero-shot measure, not of the substrate. (iii) Its internal
signal-to-noise over `u` runs 1.4-2.0 where the zero-shot measure reads 57/124 = 0.46.

---

### 3b. 500-episode confirmation, and the roster verdict (CONDITIONAL)

**Item 2 at 500 episodes: pooled 0.910** (sd0 0.902/0.842, sd1 0.946/0.868, sd2 0.966/0.936
at 250k/500k, n=6) against BC 0.072, actor-free GPI 0.415, the 09-08 faithful-DSRL arm
0.136/0.141, per-task FQL 0.949. **96% of the per-task reference through the same frozen
flow**, and flat: 0.842-0.966 where the zero-shot arm traverses 0.086-0.704 inside one seed.

**A critic scoring 0.910 ranks the roster no better than one scoring 0.415.** Same 256-onset
roster as section 1:

| ranker | rho `onestep_bc` | regret | its deployed success |
|---|---|---|---|
| `psi^T w` | +0.065 | 7.88 | 0.415 |
| `Q_A(s, G(s,u))` | **+0.062** | 6.49 | **0.910** |
| `Q_W(s, u)` | **+0.062** | 6.67 | **0.910** |
| random | — | 8.58 | 0.072 |

So roster-ranking ability does not determine success. **But the verdict is conditional and
the condition is the critic, not the deployment.** The 09-08 `dsrlfaithful` arm deployed the
same way — latent actor, no roster argmax — climbing the MEASURE readout, and scored
0.136/0.141. Same deployment, a sixth of the number. What separates them is a Bellman critic
on the real reward. So: **actor deployment wins WHEN the critic behind it is a Bellman one;
roster argmax is the wrong deployment either way, but replacing it buys nothing on its own.**
Whether a *zero-shot* Bellman critic keeps the property is what D2 decides; D1 prices the
readout.

No contradiction with section 2: Step A measured ordering agreement *within* the
roster-argmax family, where it does track success (rho +0.72). This arm does not deploy that
way.

**Cube whitened eval500** (pre-registered: no change expected, Gram deviation 0.035):
healthy sd0@350k **0.704 -> 0.698**, intervals overlapping, no change. Dead sd0@500k
**0.086 [0.064,0.114] -> 0.160 [0.130,0.195]**, non-overlapping: **+0.074, real but small —
12% of the way back to the healthy checkpoint.** Reading: a piece of the cube oscillation is
the estimator and the bulk of it is not; the dead checkpoint stays dead. `orth_offdiag`'s
r = -0.375 gets a **partial** cause on cube, not its whole cause. (The maze whitened evals,
where the Gram actually collapses, are still running.)

---

### 4. The GATE — the reward is only weakly readable from phi, and from anything else

`tools/diag_reward_readout.py`, 200k rows, 12 checkpoints. Pre-registered: R2 > 0.5 viable,
< 0.2 capped.

| statistic | value |
|---|---|
| topline R2 (least squares, shifted reward) | **0.116** |
| closed form `w = E[r phi]`, rescaled | 0.116 |
| cos(closed form, least-squares optimum) | **0.997** |
| topline R2 on the RAW unshifted reward | **-0.20** |

**CAPPED against the pre-registered 0.2 line — but that line had no baseline, and with one
the reading changes.** Same 200k rows, same shifted reward: raw observations (28-d) 0.049,
untrained random tanh features at phi's width (128-d) 0.065, random Fourier 512-d 0.096,
random Fourier 2048-d 0.169 in-sample / 0.107 held out. Trained phi reads 0.116, about 2x a
matched random basis and level with a 2048-d nonlinear one. So "phi cannot express the
reward" is withdrawn; what holds is that no linear basis of this size expresses a reward
2.1% of rows pay, and the agent optimises the reward projected onto phi's span. Whether that
blurred goal suffices is what Arm D1b measures.
(`psm-data/logs/diag_reward_baseline_cube.json`.) It reproduces COMPENDIUM 4.7's D2 (0.129)
independently. Four things R2
alone does not say: the ESTIMATOR is not the problem on cube (cos 0.997, within 0.003 of the
topline); the cap is **flat across the ladder** (0.107-0.120 while success swings eight-fold
— it is a constant ceiling, NOT the oscillation); **86.5% of the predicted reward mass sits
on rows that pay nothing** (rewarding rows are 2.1% of the data and the readout predicts
0.128 on them against a target of 1.0); and `eval_reward_shift=1.0` is load-bearing, since
a no-intercept readout of the raw -1/0 reward does worse than predicting the mean.

---

### 5. The gate extension — the cap is not one thing, and pointmaze is a different failure

**The ortho coefficient moves the cap the WRONG WAY**: 0.116 at `ortho_coef=1e3` against
0.091 at 1e4. Not the lever.

| env | topline R2 | deployed R2 | cos(deployed, optimal) | Gram dev from I | cond |
|---|---|---|---|---|---|
| cube | 0.115 | 0.114 | **0.997** | 0.035 | 1.3 |
| antmaze | 0.104 | 0.083 | 0.893 | 0.423 | 15.4 |
| **pointmaze** | **0.514** | 0.149 | **0.075** | 2.39 | **2e9** |

Pointmaze's basis is the MOST expressive of the three and its deployed estimator points
almost ORTHOGONALLY to the best readout. `affine_strict_pointmaze` scores exactly **0.000**
on every seed and checkpoint.

---

### 6. The mechanism, and the two consequences worth carrying

30 pointmaze checkpoints, measured Gram beside the `orth_loss` the run logged at that step:

| relationship | Spearman |
|---|---|
| `orth_loss` vs Gram deviation from I | **+1.000** |
| Gram condition vs **deployed** R2 | -0.829 |
| Gram condition vs **topline** R2 | +0.400 |

Deployed R2 swings 17x (0.017-0.305) while the topline holds 0.25-0.60; the Gram condition
number swings **four orders of magnitude**, non-monotonically — the same signature as cube's
success ladder under the same loss.

**Both pre-registered branches are wrong.** The regulariser is not blind (rank correlation
1.000) and the collapse is not constant-from-the-start. The ortho term is simply **losing,
intermittently**, and when it loses the reward inference stops working while the basis
underneath stays fine.

> **Consequence 1.** This gives the 09-08 stability campaign's strongest correlate a
> mechanism. `orth_offdiag` was the best single predictor of cube success (r = -0.375) and
> nobody could say why. It is measuring whether `w = E[r phi]` is still the right estimator
> at all.
>
> **Consequence 2.** The write-up already stated the condition. Cor. `reward-inference` says
> `w = E_D[r phi]` is the least-squares projection EXACTLY when `E[phi phi^T] = I`, and in
> its own words that is "why the orthonormality loss is part of the specification and not a
> stability regulariser". Stated, never checked.

---

### 6b. The whitened estimator: hypothesis REFUTED by its own test

The Gram-collapse story predicted that whitening the reward inference would rescue
pointmaze. Pre-registered: any non-zero pointmaze eval500 is decisive, above 0.2 means the
estimator was the whole failure. Eval-only, 500 episodes, same checkpoints:

| env | checkpoint | closed form | whitened | delta |
|---|---|---|---|---|
| **pointmaze** | sd0/1/2 @500k | 0.000 | **0.000** | **+0.000** |
| antmaze | sd0 @500k | 0.078 | 0.010 | -0.068 |
| antmaze | sd1 @500k | 0.004 | 0.042 | +0.038 |
| antmaze | sd2 @500k | **0.428** | **0.062** | **-0.366** |
| cube | sd0 @350k healthy | 0.704 | 0.698 | -0.006 |
| cube | sd0 @500k dead | 0.086 | 0.160 | +0.074 |

**Pointmaze does not move at all. The estimator was not the pointmaze failure.** Whitening
also HURTS antmaze, worst on the seed that worked (0.428 -> 0.062). And the topline that
motivated the hypothesis was partly overfitting: held out on a 50/50 row split, pointmaze
falls 0.584 -> 0.395 with `||w||` 40-70x the other envs (unregularised least squares on a
Gram whose smallest eigenvalue is 2e-4). The "most expressive basis" claim survives weakened
-- 0.395 still beats cube's 0.104 and still clears 0.2 -- but the 0.514 headline was inflated.

**The useful part.** On antmaze the whitened `w` reconstructs the reward equally well
(held-out R2 0.102 either way) and produces a much worse policy. So **reward-reconstruction
quality does not determine policy quality** -- the second instance today of that shape, after
section 3b's finding that roster-ranking quality does not either. Two natural proxies for
"is this critic good", both failing to predict the outcome.

What survives from section 6 unchanged: the Gram does collapse, `orth_loss` tracks it at rank
correlation 1.000, and the write-up stated the precondition without anyone checking it. What
does not survive is the causal claim that this is why pointmaze fails.

---

### 6c. Pointmaze: the substrate, not the agent

**POINTMAZE IS SUBSTRATE-CAPPED (2026-09-09).** The BC control on
`pointmaze-medium-navigate` — the frozen Stage-A flow acting alone on a fresh prior latent
each step — scores **0 / 100 episodes**, Wilson 95% upper bound **0.037**
(`tools/diag_action_coherence.py`, report `diag_pointmaze_zero.json`). The flow cannot do the
task at all, so **no Stage-C number on this environment means anything** and it must not be
used to judge the method. This is why `affine_strict_pointmaze` reads exactly 0.000
everywhere and why whitening the reward inference moved it by zero episodes: there was
nothing to move. Arm C's three pointmaze seeds were cancelled mid-training on this basis.

**Stage A re-examination is PARKED as a separate item**, with one oddity flagged for whoever
picks it up: a behaviour-cloned flow scoring *exactly* zero on a medium point-maze is
surprising on its face — the task is the easiest of the three — so a checkpoint-selection or
eval-goal mismatch is at least as likely as a capacity limit, and should be ruled out before
anything is retrained.

---

### 6e. ARM C — the first arm that raises the mean AND halves the swing

`armc_cube`: the measure head fitted at **4 latents per transition instead of 1**, the extra
three drawn from the stored EM posterior with each component's covariance scaled by
`c = 0.5`, carrying the same `(s, s')` target. Arm B is the same seam with the extra latents
from a **round ball**, and Arm B is null — so the pre-registered claim that the SHAPE is what
matters is the one that survived. 500 episodes, `stability_ladder.py`, 250k–500k:

| seed | 250k | 350k | 450k | 500k |
|---|---|---|---|---|
| sd0 | 0.386 | 0.454 | 0.624 | **0.792** |
| sd1 | 0.662 | 0.706 | 0.758 | **0.804** |
| sd2 | 0.124 | 0.180 | 0.228 | 0.322 |

| arm | mean | mean adjacent step | source |
|---|---|---|---|
| **Arm C** | **0.503** | **0.083** | 12 cells |
| control, same 4 cells | 0.442 | 0.181 | 12 cells |
| control, all cells | 0.426 | 0.163 | 18 cells |
| Arm B jitter 0.3 | 0.462 | 0.250 | 6 cells |
| Arm B jitter 0.5 | 0.334 | 0.141 | 6 cells |

**Monotone 9 of 9.** Every seed rises at every step; the control over the same cells goes
down as often as up (sd0: 0.532 → 0.704 → 0.272 → 0.086). Under a coin-flip null on the
direction of each step, 9/9 is p = 0.002 one-sided.

**This is n = 3 seeds on one environment and is NOT yet a stability claim.** The ladder is
rising, not settled, so the swing number is partly just the rise. What decides it is the 1M
continuation: if a seed falls back below its 500k value by more than the swing, the rise was
a slow transient and the arm returns to being a stability question. Report:
`psm-data/logs/stability_armc_vs_control.json`.

> **ANSWERED 2026-09-10, AGAINST THIS ARM.** The continuation ran and the pre-registration
> failed. sd0 0.792 -> 0.654 -> 0.612 (down 0.18), sd2 0.322 -> 0.086 -> 0.012 (collapsed),
> sd1 held at 0.846. The rise WAS a slow transient. Arm C also does not replicate on antmaze
> (0.285 vs the control's 0.266 on matched cells) and its dose response is flat in `c`
> (0.472 / 0.515 / 0.409 at c = 0.3 / 0.5 / 0.7 against the control's 0.426). **Read the
> 2026-09-10 entry, not this one, for the standing verdict.** The table above is the
> 250k-500k window and is real; it is just not the whole ladder.

Note for the continuation: `restore_agent` DOES carry the optimiser state, the target
networks and the per-`TrainState` step counter — verified leaf-by-leaf in
`tests/test_checkpoint_restore_carries.py` — so no fresh 1M run is needed. But `main.py`'s
outer counter restarts at 1, so a continuation writing to the same `save_dir` would
**overwrite the first run's checkpoints**. Use a new `run_group`.

---

### 6d. Two launch traps, both caught by a 200-step smoke (2026-09-09)

Recorded because each would have wasted a full run and neither is visible from the code you
are editing.

- **`configs/agent/psmflow.yaml` is a second source of truth from `get_config()`.** A key
  added only to `get_config` makes every unit test pass and then dies on the cluster in 20
  seconds: `Could not override 'agent.dsrl_na.reward_refit_every' ... not in struct`. Hydra
  builds the override schema from the yaml. Add new agent keys to BOTH.
- **`CsvLogger` fixes its header from the first row it writes** (`utils/log_utils.py:41`),
  and later rows are filled with `header.get(k, '')`. A metric logged only on its own
  schedule — a periodic refit, say — never reaches `train.csv` at all, only wandb. Carry
  the latest values onto every logged row instead, and make sure the first one precedes the
  first log step.

---

### 7. Code changes (all default-off; no published number moves)

- `dsrl_na` block: `qa`/`qw` scalar critics, `dsrl_qa_loss`, `dsrl_qw_loss`, `na_spread`,
  `_actor_w`, `_actor_q` dispatch, `create` guards. `reward_source` = `real` (Item 2) |
  `phi_readout` (Arm D1) | `synthetic_w` (Arm D2); `task_conditioned` for D2.
- `reward_inference` = `closed_form` (default, every published number) | `whitened`, which
  solves the normal equations and is a **no-op wherever the Gram is I, hence on cube by
  construction**. Added to `stability_ladder.ACTING_OVERRIDES` before the first such eval,
  since it changes the policy's `w` and would otherwise overwrite a run's own ladder entry.
- `measure_u_samples` / `measure_u_source` (`mixture` | `jitter`) / `measure_u_jitter_std` /
  `measure_u_mixture_shrink`; `u_extra_dist` and `u_extra_clipfrac` logged.
- **`phi_gram_dev`, `phi_gram_cond`, `phi_gram_eig_min/max` are now standard in-loop
  metrics**, so every future ladder carries the validity condition beside success.
- `actor.layer_norm` on the tanh-Gaussian trunk. `sample_preimage_noise(..., scale=)`.
- New tools: `diag_mixture_decode.py`, `diag_ranker_agreement.py`, `diag_reward_readout.py`,
  `diag_commitment_horizon.py`; `diag_gpi_selection.py` rewritten (onsets, three
  continuations, four rankers, in-box columns, `table` subcommand).
- **Arm D1b** (`dsrl_na.reward_source=phi_readout_fixed`, 2026-09-09): the readout channel
  fit the way it is DEPLOYED, since D1's per-256-row refit made `r_hat`'s std 13.6 against
  the reward's 0.15 and trained `Q_A` on a reward redrawn every step. `w` is the closed form
  on a 10k relabel batch, refit every `dsrl_na.reward_refit_every` steps and held fixed
  between; `r_hat` is rescaled to the real reward's std; `na_rhat_corr_heldout` is measured
  on a disjoint 10k batch. New agent fields `na_rw` / `na_rw_scale`, zero-initialised so
  older checkpoints restore clean; the refit runs in `main.py`, outside the jitted update.
- Tests: `tests/test_psmflow_dsrl_na.py`, 44 cases.

---

### 8. RUNNING

Arm B (jitter, 2 doses x 3 seeds, cube), Arm C (shrunken mixture, 3 envs x 3 seeds), D1
(`phi_readout`, 3 seeds cube), the DSRL-NA eval500s, the whitened eval500s on
pointmaze/antmaze/cube, the Item 2 roster diagnostic, the commitment-horizon sweep, and
eval500 for the two ortho stability arms whose verdict was provisional.

**Arm A (ensemble mean at acting) was DROPPED before launch**: a frozen FQL expert's critic,
trained by a real max-backup on real rewards, ranks the same roster no better than the
measure, so this is not a noise problem that averaging fixes.

---

## 2026-09-08 — the measure-loss verdict, the K sweep, and the first faithful DSRL arm

---

### 1. Measure-loss audit: verdict filled

`docs/design/2026-09-08-measure-loss-audit.md` §8. All nine pre-registered arms landed.
Growth refitted on a **common 10k-150k window** (the §7 numbers were 50k-500k and must not
be read against these). antmaze `medium-navigate`, gamma=0.995:

| arm | kappa_measure | steps/decade | log10 |Q| growth |
|---|---|---|---|
| `mla_g995_pess10` | 1.0 | 1.5e4 | 4.15 |
| baseline | 0.5 | 5.18 / 4.25 / 3.84e4 | 1.25-1.78 |
| `mla_g995_pess025` | 0.25 | 8.0e4 | 0.73 |
| `mla_g995_pess0` | 0.0 | 1.08e5 / 1.95e5 | 0.45 / 0.33 |

**H-PESS confirmed** -- monotone in `pessimism_penalty` across four dose levels spanning
13x, against 35% baseline seed scatter. Orthonormality delays only (`orel` within scatter
seed-for-seed). **The pre-registered structural fix is REFUTED on cube**: kappa_measure=0
scores 0.136 vs 0.415, while antmaze's 0.334 vs 0.252 is inside overlapping CIs. Default
stays at 0.5.

**`psi_bound` is settled negative** -- best growth number measured anywhere (11x baseline)
and the worst policy (early success 0.096-0.256 vs baseline 0.288-0.344). Growth rate is
not a proxy for policy health.

**Ensemble spread carries no support information**: prior/in-support disagreement ratio is
1.00 across eight checkpoints. `kappa*spread` is a uniform one-signed shift, not a
support-aware penalty.

---

### 2. Dead seeds are a SELECTION failure

`$PSM_DATA/logs/diag_deadseeds/`. Over the candidate roster, Spearman(Q, MC return) is
**0.06-0.16 in every arm including healthy seeds**, with a negative 10th percentile --
on a tenth of states the ranking is inverted. Meanwhile **79% of decoded prior candidates
succeed under MC**, identically in every seed. The flow offers a good action nearly
everywhere; the measure cannot pick it.

---

### 3. `gpi_num_u` sweep (new, `scripts/slurm/launch_gpi_num_u_sweep.sh`)

500 episodes, frozen checkpoints, only `agent.gpi_num_u` overridden. Positive control
(K=8 on the dead seed) reproduced 0.134 against the 0.140 on record before the rest ran.

| checkpoint | K=4 | K=8 | K=16 | K=32 | K=64 | best |
|---|---|---|---|---|---|---|
| cube healthy sd0 | 0.274 | 0.488 | **0.514** | 0.452 | 0.312 | K=16 |
| cube healthy sd1 | 0.132 | 0.244 | 0.334 | 0.340 | **0.374** | K=64 |
| cube dead (`mpess0`) | **0.218** | 0.134 | 0.078 | 0.012 | 0.004 | K=4 |
| antmaze g99 sd2 | 0.350 | 0.394 | 0.458 | 0.482 | **0.510** | K=64 |

**The optimum is per-seed, not just per-domain, and there is no single right K.** The dead
seed collapses **54x** from K=4 to K=64 -- max over a critic it cannot rank, i.e. max over
noise. Healthy cube sd0 is an inverted U peaking at 16 and losing a third by 64. Healthy
cube sd1 and antmaze are monotone increasing out to 64.

Shared-K comparison (one hyperparameter, no per-seed picking): on the two healthy cube
seeds K=16 gives 0.424 against K=64's 0.343. But antmaze prefers 64 (0.510 vs 0.458).
So cube and antmaze want different K, which is what the literature reports. **Two seeds per
env; cube sd2 was not swept. Do not change the default on this alone.**

Note antmaze here is monotone INCREASING, the opposite of Diffusion-DICE's antmaze result.
Its seed is the healthiest we have, which fits the pattern that K-tolerance tracks critic
quality rather than the domain.

Literature agrees there is no consensus: published N spans 1-400, tuned per task, and
FQL's own IDQL/IFQL rejection-sampling baselines use **32 for every OGBench task including
cube-single-play**. Diffusion-DICE (arXiv:2407.20109) Table 4 shows an 82.0 -> 51.8
monotone collapse on antmaze-umaze-diverse over K=1..16. The argmax's optimization
pressure is `log N - (N-1)/N` nats, exact for continuous latents: K=64 is 3.16 nats.

---

### 4. What we called DSRL was not DSRL

External review, verified against the code. Two findings:

**`dsrl_na` is not DSRL-NA.** DSRL-NA is a dual-critic scheme: an ACTION-space critic Q_A
trained on real transitions, distilled into the latent critic, exploiting the fact that
many latents decode to the same action. Ours regresses the actor onto the argmax of 16
prior draws scored by the GPI rule -- candidate-argmax behaviour distillation, the
SfBC/IDQL family. **Renamed `gpi_distill`**, with `dsrl_na` kept as a read-time alias so
old flags.json files still restore. `psi_a(s, w, a)` in the action branch already IS the
action-space critic real NA would need.

**The default arm structurally cannot do SAC.** Under `policy_index=latent`,
`sample_step_inputs` computes the actor's bootstrap latent and then overwrites it with the
prior draw u'. psi is the successor measure of the constant-latent policy pi_{u'}, which
is what makes GPI well-posed -- and means the backup never contains the actor's action. An
actor trained against it is doing policy improvement against a value function that is not
Q^pi for any pi it converges to. The assertion chain closes the escape hatch:
`gpi_distill` -> `index_panel>0` -> `policy_index=latent`.

**`psi_form=affine` and a faithful DSRL arm are mutually exclusive.** Now stated in the
module docstring; the config space does not enforce it.

**The paper is unaffected.** `table_headline.tex`'s DSRL-SAC rows (0.183, 0.127) come from
the archived `latentrl` agent, whose backup does use the actor's latent
(`archive/agents/latentrl.py:176,182`). Only the psmflow ablations were mislabelled, and
no published number depends on them.

**One review claim did not hold.** The BC anchor was said to cancel the steering at
`bc_coeff=1.0`. Every shipped DSRL run used `bc_coeff=0.0` (14 `gpi_distill` + 18
`dsrl_sac`); the `bc_coeff=1.0` runs are the 33 older ddpg-era ones. So the plateau
happened with no BC pull, and the actor's 1.70-vs-2.13 contraction is regression-to-mean
onto a re-randomised target, not a constraint.

---

### 5. Code changes (all default-off; published numbers unchanged)

- `actor_mode=gpi_distill`, `dsrl_na` aliased on read (`ACTOR_MODE_ALIASES`).
- `MEASURE_DEFAULTS` now also backfills `index_panel`, `expectile_mu`,
  `mask_invalid_preimages`. (The earlier gap: `psi_form`/`index_agg`/`gpi_select`/
  `index_clip` were guarded with `.get(legacy)` in `create` but read unguarded at runtime;
  both `index_clip` tests were failing before this landed.)
- **Mode-based eval.** `sample_actions` ignored its `temperature` argument, so the DSRL
  arms were evaluated stochastically while `utils.evaluation` passes `eval_temperature=0`.
  Now returns `u_clip * tanh(mu)` at temperature 0, which is how SAC is normally evaluated.
  **Every DSRL number on record was a stochastic-policy eval.**
- **`mask_invalid_preimages`** (default false). `preimage_valid` was recorded by
  `repair_invalid_preimages` and never consulted, so diverged rows trained the measure on
  (u, a) pairs where u does not decode to a. `contrastive_loss` gained an optional
  `row_weight`; verified to reproduce the published loss bit for bit at weight 1.
  **Scale check: cube has 13 diverged rows in 1,000,000 (0.0013%)**, so on this dataset the
  flag is correctness hygiene and will not move a number. It matters only if an inversion
  is ever run at settings that diverge more.
- **`actor.prior_init`** (default false). Default init gives per-dim latent std **1.88** at
  u_clip=3 against the flow's prior 1.0, so the policy started outside the support the flow
  was fitted on. `prior_init` zeroes the mu head and biases log_std to log(0.3).
- `flow_steering.py`: jitted both `vmap`s (bare vmap over the proposal reproducibly returns
  all-NaN on GPU), seeded the mixture draw from the passed rng instead of global numpy
  state, and clamped it to `u_clip`.
- Corrected the `u_data` clip comment: its justification does not hold under
  `policy_index=latent`, where the bootstrap is a prior draw.

---

### 6. RUNNING: the first faithful DSRL arm

`scripts/slurm/launch_dsrl_faithful.sh`, cube, 3 seeds x 2 boxes, 500k.

    actor_mode=dsrl_sac policy_index=task_vector psi_form=free index_agg=max
    actor.index_panel=0 train_actor=true acting=actor
    actor.bc_coeff=0 actor.prior_init=true

`policy_index=task_vector` is the one index under which `u_next` survives into the backup,
so this is the first arm where the latent MDP is genuinely on-policy. It gives up
Prop. `bilinear` by necessity.

Two boxes so the structural change and the box change are separable: **u_clip=1.5 and 3.0**.
DSRL recommends `b_W` in **[0.5, 1.5] for offline** and [1, 3] for online; every run in this
repo has used 3.0, the top of the *online* range (150 runs at 3.0, 2 at 1.0). Their decoder
also trains with `randn_clip_value: 3`, so 3.0 sits at the edge of trained support with no
margin.

**Read it against the BC control (0.072), not against actor-free GPI.** The question is
whether steering works at all on this substrate, not whether it beats GPI.

#### RESULT (landed same night, 500 episodes)

| arm | sd0 250k/500k | sd1 250k/500k | sd2 250k/500k | pooled |
|---|---|---|---|---|
| u_clip=1.5 | 0.100 / 0.018 | 0.174 / 0.298 | 0.162 / 0.064 | **0.136 +/- 0.079** |
| u_clip=3.0 | 0.102 / 0.116 | 0.250 / 0.026 | 0.274 / 0.076 | **0.141 +/- 0.079** |

against BC **0.072**, `gpi_distill` **0.307 +/- 0.091**, actor-free GPI **0.415 +/- 0.083**.

**Fixing the backup did not rescue steering.** The structural criticism was right -- under
policy_index=latent the actor's action never enters the Bellman target, so `dsrl_sac` was
not SAC -- but making the latent MDP genuinely on-policy yields ~0.14, *below* the
mislabelled `gpi_distill` and a third of actor-free GPI, with a CI whose lower edge is
under the BC control. The backup was not the binding constraint.

**No box effect.** The in-loop 50-episode read suggested u_clip=1.5 beat 3.0 by 2.2x
(0.184 vs 0.085) and that was reported as empirical support for DSRL's offline
b_W in [0.5, 1.5]. It does not survive 500 episodes: 0.136 vs 0.141, indistinguishable.
**Recorded so the claim is not revived from the in-loop curves** -- and as one more instance
of why the 50-episode evals are not reportable.

**Training past 250k hurts this arm.** u_clip=3.0 pools 0.209 at 250k against 0.073 at
500k; per-seed swings are 0.250 -> 0.026 and 0.100 -> 0.018. n=3, but it matches the
instability seen everywhere else.

**Interpretation.** Every thread this session converges on the same place: the critic
cannot rank (Spearman 0.06-0.16, negative 10th percentile) while 79% of decoded prior
candidates succeed. An actor climbing that critic has nothing to climb, on-policy backup or
not. The ranking objective, not the backup and not the box, is the binding constraint.

Verified directly from `ajwagen/dsrl` configs: `target_ent: 0.0` is held constant across
`action_magnitude` 1.0 / 1.5 / 2.5, so it is NOT tied to a box size (an earlier claim in
this record that it was b=1.5-specific is withdrawn). Our log-prob is computed in the
squashed-but-unscaled space, so our entropy target is box-invariant and comparable.

---

### 7. RUNNING: the stability campaign -- oscillation is the blocker, not ranking

`docs/design/2026-09-08-oscillation-stability.md` (pre-registered); scored by `tools/stability_ladder.py`. Every cube seed already
reaches 0.6-0.7 and none of them holds it: per-seed maxima **0.704 / 0.620 / 0.596**, mean
of maxima **0.640**, against a pooled 300-500k figure of **0.415**. Seed 0 traverses
0.532 -> 0.286 -> 0.704 -> 0.282 -> 0.272 -> 0.086 at 500 episodes, where sampling noise is
+/-0.04. The capability is present; the consistency is not.

Correlating every logged training metric against 500-episode success over 28
(seed, checkpoint) pairs: **`orth_offdiag` is the strongest predictor at -0.375** (tighter
orthonormality, better policy) and **`psm_loss` is +0.079, i.e. nothing.** Standing warning:
on this method loss diagnostics are not a proxy for policy quality -- the same lesson
`psi_bound` taught.

The ortho settled-negative does NOT cover this. It was measured against loss-growth rate at
gamma=0.995; `ortho_coef=1e4/1e5`, `lr_sf=1e-5` and `tau=1e-3` have never been scored on
cube success at gamma=0.98.

Four arms, 3 seeds each, checkpoints every 50k, control = existing `affine_strict_cube`:
`tau1e3_cube`, `oc1e4_cube`, `lrsf1e5_cube`, `oc1e4_lrsf1e5_cube`.

**Scored on stability, not the mean** -- per-seed swing and mean step over the 300-500k
ladder, reported beside the pooled mean. An arm at 0.45 flat beats an arm averaging 0.70
and 0.09. Control swing to beat: 0.618 / 0.222 / 0.254.

Also corrected here: at a good checkpoint the critic is worth ~8x over random
(argmax 0.704, mean 0.734, small_ball 0.670 vs every critic-free rule at ~0.08). The
"critic picks worse than random" result in 8.4 came from the MC probe's held-fixed-latent
regime, which is not what GPI deploys.

**INTERIM VERDICT (10 of 12 runs complete).** Every arm failed; see section 8 of the design
doc. In-loop ladders, control scored on the same basis:

| arm | mean | swing | swing/mean |
|---|---|---|---|
| control | **0.432** | 0.353 | **0.82** |
| `tau1e3` | 0.119 | 0.193 | 1.62 |
| `oc1e4` | 0.296 | 0.533 | 1.80 |
| `lrsf1e5` | 0.104 | 0.173 | 1.66 |

`tau1e3` and `lrsf1e5` halve the swing and drop the mean 4x -- they sit near the floor,
where a small swing is free. Every arm is *relatively* more unstable than the control.
**The ortho line is closed**: `oc1e4` was worse on both axes, so the r=-0.375 correlation
was confounded. This is section 7's pre-registered expected failure, and it says the
instability is in the contrastive objective rather than in any rate.

---

### 8. Where this leads: a Bellman-trained selection head

Three facts compose: slowing the optimiser does not help; `psm_loss` predicts success at
r=+0.079; and the critic that ranks weakly is trained contrastively. Contrastive training
constrains the ANGLE and leaves norm growth off-support free -- a scorer that retrieves
well and ranks badly. arXiv:2607.27422 isolates this on our exact architecture:
parameter-matched, contrastive ranks at Kendall tau ~0.40 against Bellman's ~0.70.

`q_dist(s, w, u)` already exists but is REGRESSED onto `psi^T w`, so it inherits the
contrastive ordering. Give it a real TD target instead. Zero-shot survives because the
reward for any task is recoverable from the basis, `r_w(x) = phi(x)^T w`, so the head can
be trained by ordinary TD on synthetic rewards over the `w` the measure loss already draws.

**Gate before any code:** confirm `r_w = phi^T w` gives TD a learnable signal rather than
being too sparse. Offline, no GPU. Fallbacks if it fails: larger critic ensemble
(`num_parallel` is 2, and its spread carries no support information) or weight averaging.

Full argument in `docs/design/2026-09-08-oscillation-stability.md` section 9.

---

### 9. Next

Ranked. Items struck through landed during this session.

**DONE this session.** ~~Read the faithful-DSRL result against BC~~ -- it scores
0.136 / 0.141 against BC 0.072, with a CI whose lower edge is under the control, so it does
NOT clearly clear BC (section 6). ~~Finish the K sweep~~ -- complete, section 3.

1. **Gate the Bellman selection head (offline, no GPU).** Check that `r_w(x) = phi(x)^T w`
   gives TD a learnable signal rather than being too sparse to bite on. This single check
   decides whether the whole line in section 8 is viable. Do it before writing any loss.
2. **If the gate passes: give `q_dist` a real TD target** instead of regressing it onto
   `psi^T w`. New loss in `agents/psmflow.py`, behind a config seam defaulting to the
   current distillation so published numbers are untouched. This is the only proposal on
   record that attacks WHY the ordering is bad rather than the machinery around it.
3. **eval500 the stability arms** to confirm section 7's interim verdict. 4 arms x 3 seeds
   x 5 checkpoints = 60 jobs, or narrow to `tau1e3` and `oc1e4` (30) since those carry the
   two pre-registered predictions. The 4x mean collapse is far outside 50-episode noise so
   the direction is not in doubt -- this is for the record, not the decision.
4. **Larger critic ensemble.** `num_parallel` is 2, which makes the uncertainty estimate
   `|M1 - M2|` two samples, and 8.3 of the measure-loss audit showed its spread carries no
   support information at all. The best-of-N literature reports ensembles largely remove
   argmax over-optimisation. A config change and one arm; cheapest untested lever left.
5. **Weight-averaging probe, not commitment.** Averaging across the oscillation only works
   if the swing is around ONE solution. Check whether consecutive good and bad checkpoints
   are near each other in weight space before investing.
6. **Decide `gpi_num_u`.** Cube sd2 was never swept and cube/antmaze disagree (K=16 vs
   K=64). Do not change the default on two seeds per env.
7. `kappa=0.25` on cube 0.98 + antmaze 0.99 -- still the only rate benefit not shown to
   cost cube.
8. Real DSRL-NA on `psi_a` -- only if a DSRL comparator is wanted. Lower priority now that
   the faithful arm underperforms: `latentrl` already supplies a defensible DSRL-SAC row.

**Superseded.** `2026-09-08-measure-loss-audit.md` 8.5 lists `gpi_num_u`, `kappa=0.25` and
`index_agg=expectile` as the next moves. The first two survive as items 6-7 above. The
third is superseded: `q_dist` under `index_agg=expectile` is distilled from `psi^T w` and
inherits the contrastive ordering, so evaluating it as-is tests the wrong thing -- item 2
is that idea done properly.

---

<!-- _class: lead -->

## 2026-09-07 — actor ablation: DSRL-NA gets 74 % of the actor-free ceiling, and it is not the support

---

### The table (`docs/tables/actor_ablation.md`, generated)

500-episode evals, `EVAL_WORKERS=1`, pooled over 300k–500k × 3 seeds, mean ± 95 % CI.
Substrate identical across arms (`psi_form=affine policy_index=latent index_agg=max
u_clip=3.0`, same frozen flow and preimages); only the actor and acting rule differ.

| env | arm | late mean | n | BC |
|---|---|---|---|---|
| cube | actor-free (GPI) | **0.415 ± 0.083** | 15 | 0.072 |
| cube | **`dsrl_na`** | **0.307 ± 0.091** | 15 | 0.072 |
| cube | ddpg latent actor (prior arm) | 0.146 | 2 | 0.072 |
| cube | `dsrl_sac` | 0.108 ± 0.053 | 15 | 0.072 |
| cube | **`prior_shrunk` control** | **0.059 [0.048, 0.072]** | 1500 ep | 0.072 |
| antmaze γ=0.98 | actor-free | 0.067 ± 0.036 | 7 | 0.072 |
| antmaze γ=0.98 | `dsrl_sac` | **0.000** | 3 | 0.072 |
| antmaze γ=0.99 | actor-free | 0.252 ± 0.100 | 15 | 0.072 |
| pointmaze | actor-free | 0.000 | 15 | 0.002 |
| pointmaze | `dsrl_sac` | **0.000** | 3 | 0.002 |

---

### What was wrong with the old latent actor

Audit: `docs/design/2026-09-06-dsrl-actor-audit.md`. The shipped actor climbed
`psi(s,u,u')^T w` at **one** random policy index per batch element, while acting takes
`max_{u'}` over 64. Measured on `affine_actor_cube` @500k with the new
`tools/diag_actor_grad_terms.py`: **cos(∇q_single, ∇q_panel_max) = −0.49 / −0.20** — the
actor's gradient was *anti-correlated* with the objective GPI maximises. BC domination on
the affine critic is 2.8–3.6 : 1 (milder than the free-psi 5 : 1 of 09-03).

The critic is untouched by the actor under `policy_index=latent` (`u_next = u_index`), so
`affine_actor` is the actor-free critic with a different acting rule bolted on — 0.415 vs
0.146 is a clean measurement of the acting rule alone.

Fix: `actor.index_panel=K` makes the actor climb the same panel max, nearly free under the
affine head (A(s,u) does not see u'). New `actor_mode: ddpg | dsrl_sac | dsrl_na`.

---

### The two decisive controls

**`prior_shrunk`** (new `gpi_select` mode): one prior draw scaled to the NA actor's own
operating radius (`u_norm` 1.72 vs prior median 2.13), reading neither `psi` nor `task_z`.
Its checkpoint-invariance cell returned **28/500 at both 300k and 500k, byte-identical**,
validating it as critic-free. It scores **0.059, at BC** → `dsrl_na`'s 0.307 is genuine
state-dependent behaviour, **not** "shrink your latents". Pre-registered prediction held.

**Ranking probe** (`tools/diag_actor_vs_gpi.py`, 256 states × 64-candidate roster):
`dsrl_na` sits at roster percentile **0.614** (top-quartile fraction 0.369 vs 0.250 chance);
the ddpg actor at **0.533** (0.248 — chance). The ordering matches success (0.307 > 0.146),
so the distillation is real but **weak**. Both actors are *farther* from the roster argmax
than from a typical candidate (ratio 1.22 / 1.25) — neither approaches it in latent space.
The specified Spearman metric is **confounded and uninformative**: ≈0.97 vs the roster max
*and* ≈0.99 vs the roster mean, for both arms, i.e. it measures per-state Q scale.

---

### Two negatives, both pre-registered

**`dsrl_na`'s regression plateaus.** Distance explained vs the independence floor
`(‖u_a‖²+‖u*‖²)/d_a`: cube **6.6–7.2 %**, antmaze 7.5–10.4 %, pointmaze 13.9–18.3 % — nine
seeds, three envs, flat from 25k to 500k. The 25 % gate is missed everywhere. So the arm
scores 0.307 **without** distilling the argmax and **without** riding the support.
What it tracks is unidentified and is the most interesting object here.

**`dsrl_sac` is 0.000 on both mazes** (0/1500 each) and 0.108 on cube. Cause: `actor_u_norm`
is **~2.0× the prior radius on all three envs** (d_a = 2, 5, 8) and `actor_q` diverges
(pointmaze sd1: **2.9e7**). `target_entropy=0` is DSRL's value for a box of **b=1.5**; ours
is `u_clip=3.0`. **These rows measure the port, not DSRL-SAC.** A corrected arm
(`affine_dsrl_sac_te_cube`, `target_entropy=−d_a·log(u_clip/1.5)`, `u_clip` unchanged so the
critic's training distribution is identical) is running — SLURM 2492091–2492093.

---

### Pre-registration scorecard

| hypothesis | outcome |
|---|---|
| H-A level (`dsrl_na` cube ≥ 0.25) | **met** (0.307) |
| H-A stability (std < 0.12) | **refuted** (0.180; per-seed 0.205/0.237/0.479) |
| H-B (`dsrl_sac` at/near BC) | **met**, and worse than registered |
| `prior_shrunk` at BC (0.05–0.12) | **met** (0.059) |
| `dsrl_na` NA-gate ≥ 25 % explained | **failed** (6.6–7.2 %) |

**Discipline note.** Two over-readings were made *while these numbers landed* and are
recorded in the design doc: "striking" after 2 of 15 cells (the next three were 0.09–0.12),
and "every seed peaks at 300k and collapses" after 2 of 3 seeds (seed 2 rises monotonically
to its best value at 500k). Both were generalisations from n = 2. Only the pooled row is
quotable.

---

### Incident: 12 evals killed by a py/yaml skew

The `actor.entropy` guard landed in `agents/psmflow.py` a few minutes **before** the key
landed in `configs/agent/psmflow.yaml`. Twelve queued evals of *older* checkpoints died at
`create()` with `KeyError: 'entropy'` (SLURM 2491907, 2491909–2491916, 2491919–2491921);
they need resubmitting, nothing else was lost. Separately, 2491881–2491889 (exit 126) and
2491924 (exit 2) failed on `scripts/eval500.sh` shell bugs *after* writing their reports —
different owner, results usable.

Root cause generalised: **a restored checkpoint's `flags.json` is older than the code
restoring it, so every `actor` key added from now on must be optional at read time.** Fix:
`fill_actor_defaults()` is now the first statement of `create()`. Regression test
`tests/test_psmflow_config_compat.py` (9 cases) drives three *verbatim archived*
`flags.json` fixtures through the eval tool's own `merge_run_config` + `create`, and asserts
**yaml ↔ `get_config()` leaf-key parity** — the one assertion that would have caught it
(negative control run: deleting `entropy` from the yaml fails it).

---

### Artifacts

- `docs/design/2026-09-06-dsrl-actor-audit.md` — audit, pre-registrations, all findings.
- `docs/tables/actor_ablation.md` — generated; `ACTOR_ABLATION` appended to `make_tables.py`.
- New: `tools/diag_actor_grad_terms.py`, `tools/diag_actor_vs_gpi.py`,
  `utils/psm_networks.py::{TanhGaussianLatentActor, tanh_gaussian_sample, LogAlpha}`,
  `gpi_select=prior_shrunk`, `tests/test_psmflow_{dsrl_actor,config_compat}.py`.
- Runs: `affine_dsrl_{na,sac}_{cube,antmaze,pointmaze}` 2491932–2491949 (six plain-NA maze
  runs cancelled at 30–40k, data salvaged); g99 arms 2492040–2492045; corrected SAC
  2492091–2492093; probe 2492090.
- Tests: 138 passed / 2 skipped over 14 modules, module-per-process.

---

<!-- _class: lead -->

## 2026-09-06 — three-seed affine ladder + baselines

---

## Affine strict, cube: **a third seed does not settle it** — the ladder is
## 0.086 … 0.704 over 30 measurements and the swing is within-run, on every seed

Seed 2 of `agent=psmflow` (repo default = paper-strict affine: `psi_form=affine
policy_index=latent train_actor=false acting=gpi`) was trained to 500k on both envs with a
config verified byte-equal to sd000's `flags.json` (only `seed`, `run_group` and the three
eval-only `gpi_select` keys added 09-05 differ), and the cube ladder was completed for all
three seeds: **every 50k checkpoint from 50k to 500k, 500 episodes each, 30 cells, no gaps.**

Table: `docs/tables/affine_cube_ladder.md` (generated).
Figure + series: `docs/figures/2026-09-06-affine-cube-ladder.{png,json}`
(generator `tools/fig_affine_cube_ladder.py`; the 09-05 two-seed figure is left in place).

---

## The full cube ladder (500 episodes per cell, Wilson 95%)

| epoch | sd0 | sd1 | sd2 | mean ± std |
|---|---|---|---|---|
| 50k | 0.192 [0.160, 0.229] | 0.412 [0.370, 0.456] | 0.224 [0.190, 0.263] | 0.276 ± 0.119 |
| 100k | 0.078 [0.058, 0.105] | 0.330 [0.290, 0.372] | 0.178 [0.147, 0.214] | 0.195 ± 0.127 |
| 150k | 0.088 [0.066, 0.116] | 0.268 [0.231, 0.308] | **0.596** [0.552, 0.638] | 0.317 ± 0.258 |
| 200k | 0.264 [0.227, 0.304] | 0.298 [0.260, 0.340] | 0.466 [0.423, 0.510] | 0.343 ± 0.108 |
| 250k | 0.532 [0.488, 0.575] | **0.620** [0.577, 0.661] | 0.246 [0.210, 0.286] | 0.466 ± 0.196 |
| 300k | 0.286 [0.248, 0.327] | 0.532 [0.488, 0.575] | 0.360 [0.319, 0.403] | 0.393 ± 0.126 |
| 350k | **0.704** [0.663, 0.742] | 0.494 [0.450, 0.538] | 0.350 [0.309, 0.393] | 0.516 ± 0.178 |
| 400k | 0.282 [0.244, 0.323] | 0.346 [0.306, 0.389] | 0.546 [0.502, 0.589] | 0.391 ± 0.138 |
| 450k | 0.272 [0.235, 0.313] | 0.568 [0.524, 0.611] | 0.564 [0.520, 0.607] | 0.468 ± 0.170 |
| 500k | 0.086 [0.065, 0.114] | 0.548 [0.504, 0.591] | 0.292 [0.254, 0.333] | 0.309 ± 0.232 |

| pooled window | n (ckpt × seed) | mean | std | 95% CI | min | max |
|---|---|---|---|---|---|---|
| **300k–500k** | 15 | **0.415** | 0.164 | ± 0.083 | 0.086 | 0.704 |
| **250k–500k** | 18 | **0.424** | 0.164 | ± 0.076 | 0.086 | 0.704 |
| BC control | — | 0.072 | — | — | — | — |

---

## What the third seed changed, and what it did not

**Did not change the headline.** The 250k–500k pooled mean moved 0.424 → 0.424 (the interval
tightened, ± 0.141 → ± 0.076, purely from n 10 → 18). The 09-05 claim stands verbatim: this
arm is ~0.42 as a *mean over late checkpoints*, ~6× BC, and **no single checkpoint of it is
reproducible to ± 0.1**.

**Settled the "is sd0 the broken one?" question: no.** Seed 2 oscillates like seed 0, not
like seed 1. Per-seed spread over the ten checkpoints (min → max, largest jump between
*adjacent* 50k checkpoints):

| seed | min | max | range | sd | largest adjacent jump | mean ≥300k |
|---|---|---|---|---|---|---|
| 0 | 0.078 | 0.704 | 0.626 | 0.202 | **0.422** (350k→400k) | 0.326 |
| 1 | 0.268 | 0.620 | 0.352 | 0.126 | 0.322 (200k→250k) | 0.498 |
| 2 | 0.178 | 0.596 | 0.418 | 0.152 | **0.418** (100k→150k) | 0.422 |

Seed 2's 0.178 → **0.596** → 0.466 → 0.246 over four consecutive checkpoints is the same
±0.4 within-run swing sd0 shows, with Wilson intervals ±0.04 wide — disjoint by a wide
margin, so it is training-time non-stationarity on **two of three seeds**. Seed 1 is the
outlier for being *calm*, not seed 0 for being wild. Its own late mean (0.498) is the best
of the three, so "pick the stable seed" and "pick the good seed" happen to coincide here —
which is exactly the trap the pooled row exists to avoid.

**No checkpoint-selection rule is available from the logs.** The in-loop 50-episode eval
tracks the 500-episode ladder only loosely (middle panel of the figure), and
`w_enc_spread` — the policy-encoder collapse diagnostic — decays smoothly to the same place
on all three seeds (peak ~1.31 → 0.36 / 0.25 / 0.31 at 500k) while success swings ±0.4
around it. The onset differs (it first crosses 0.6 at 190k / 215k / **305k** for sd0 / sd1 /
sd2) but the ordering does not match the success ordering: sd2 collapses *latest* and is
neither the best nor the worst arm. The encoder collapse is real and monotone; it does not
predict which checkpoint, or which seed, is good.

---

## Antmaze, seed 2: still nothing

| epoch | sd0 | sd1 | sd2 |
|---|---|---|---|
| 50k | 0.112 [0.087, 0.143] | 0.000 [0.000, 0.008] | — |
| 250k | — | 0.178 [0.147, 0.214] | — |
| 300k | — | — | 0.088 [0.066, 0.116] |
| 350k | 0.082 [0.061, 0.109] | — | — |
| 400k | 0.056 [0.039, 0.080] | 0.134 [0.107, 0.167] | — |
| 500k | 0.002 [0.000, 0.011] | 0.096 [0.073, 0.125] | **0.008** [0.003, 0.020] |

Seed 2 was evaluated at 500k and at its single best in-loop checkpoint (300k, in-loop 0.40).
**500k = 0.008, i.e. the third seed also decays to the floor**; 2 of 3 seeds now end below
the BC control (0.072). The late-checkpoint mean (≥250k, 8 measurements) is **0.081 ± 0.050**
against BC 0.072 — the interval still contains BC, so antmaze remains **not a result**.

The in-loop eval is worse than useless here: 300k read **0.40 over 50 episodes** and
**0.088 over 500**. Any checkpoint picked off `eval.csv` on this env is picking noise.

---

## Provenance / caveats for the two entries above

* Every cell is `tools/eval_checkpoint.py`, 500 episodes, agent config inherited from the
  run's own `flags.json` (no arm flags on the CLI). SLURM 2491687-94 (sd0/sd1 gap fill),
  2491730-2491798 (sd2 cube ladder), 2491799-2491800 (sd2 antmaze).
* Training: cube sd2 02:41:50, antmaze sd2 04:05:51, matching sd0/sd1 to a few minutes.
* The sd0/sd1 **100k and 250k** JSONs carry `acting_mode: "decode(actor latent)"`. That is
  the pre-09-05 **label** bug only — `acting: gpi` in the same JSON, and the shipped
  `gpi_select=argmax` code path is unchanged by the 09-05 ablation commit (the ablations
  branch out before it). The numbers are comparable; the string is not.
* `tools/make_tables.py` now carries one row per cube checkpoint 50k–500k plus a
  300k–500k pooled row, and an antmaze @300k row; the `sd?` globs already span three seeds.


---

## Baseline — **plain FB with the actor's BC term off: 0.000 on cube, every seed,
## every checkpoint; the same critic with BC on clears the BC control**

Full write-up, hyperparameters and caveats: `docs/tables/fb_nobc_cube_ladder.md`.
Repro: `bash scripts/baselines/fb_nobc.sh {table|train|eval}`.

**Where the code came from.** `https://github.com/LUH-AI/Factored-FB.git`, branch
`density-fb`, commit `b62dc9d5e73f282924c29d0ab64d1f889b43532e`, checked out at
`/mnt/home/amohan/git/Austin/Factored-FB`. That branch is simply where the newest code
sits; the agent run is the **plain Forward-Backward critic** `fb`
(`impls/critics/fb.py`), not the density variant. Nothing from PSMFlows was used — this
agent trains F, B, the left encoder and the actor from scratch in one 500k run, so
`$PSM_DATA/flow/cube-single-play` never entered it.

**"BC anchoring off" = `--actor ddpgbc --override actor.alpha=0.0`.** `DDPGBCActor.loss`
is `q_loss + bc_loss` with `q_loss = -q.mean() / sg(|q|.mean() + 1e-6)` and
`bc_loss = -(alpha * dist.log_prob(dataset_action)).mean()`. `alpha=0` zeroes the anchor
exactly and keeps the `|q|` normaliser (it is gated by the separate `q_normalize`, default
true), so the actor ascends `Q = <F(le(s), a, z), z>` and nothing else. `alpha` is the only
BC-ish knob on this actor, so there is no second reading to hedge against. Picking `ddpgbc`
over the repo's canonical `flowbc` is also what makes the arm flow-free: `flowbc` carries
`bc_coeff: 3.0` and distils a flow-matching policy. This is the repo's own idiom —
`scripts/launch_exorl.sh` pins exactly these two flags for its no-BC arms. The runs' logs
show `bc_loss: 0.0` throughout and their `config.json` records `actor.alpha = 0.0`.

**The ladder** (500 episodes per cell, OGBench task_id=1, i.e. what PSMFlows calls
`cube-single-play-singletask-v0`; reports at
`$PSM_DATA/logs/eval500_fb_nobc_cube_{epoch}k_sd{S}.json`):

| epoch | 50k | 100k | 150k | 200k | 250k | 300k | 350k | 400k | 450k | 500k |
|---|---|---|---|---|---|---|---|---|---|---|
| seed 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| seed 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| seed 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

**Late mean (300k–500k, 15 measurements): 0.000 ± 0.000.** Same on the affine ladder's
250k–500k window (18 measurements). Wilson 95% on each 0/500 cell is [0.000, 0.008] — 15000
evaluation episodes without a single success, not a noisy zero.

Against the two numbers that matter: **BC control 0.072**, **affine psi strict, cube, late
mean 0.424**. The baseline is below the control, not merely below the method.

**Two controls, and they are the actual result.** Both seed 0, same critic, same data, same
500-episode protocol:

| arm | flags | 300k | 400k | 500k |
|---|---|---|---|---|
| **BC ON** — the repo's canonical cube-single pairing | `--actor flowbc` (`bc_coeff 3.0`) | **0.110** [0.086, 0.141] | **0.084** [0.063, 0.112] | **0.108** [0.084, 0.138] |
| BC OFF + strong orthonormaliser | `--actor ddpgbc actor.alpha=0.0 ortho_coef=1000` | 0.000 | 0.000 | 0.000 |

1. **The behaviour anchor is load-bearing and it is the only thing that is.** Same FB
   critic, same budget: with the anchor the arm clears the BC control at all three
   checkpoints (0.110 / 0.084 / 0.108 vs 0.072); without it, zero.
2. **`ortho_coef` is not the confound.** The archived PSMFlows FB reached cube 0.721 with
   `ortho_coef=1000` while this repo ships the reference-native 1.0, so the obvious
   objection was that the baseline was mis-tuned rather than BC-starved. Re-run at
   `ortho_coef=1000` with BC still off: still 0/500 at 300k, 400k and 500k.
3. The raw-action PSM baseline in this same entry reached 0.000 by deleting the same kind of
   anchor from a different agent in a different codebase. Two independent confirmations that
   "successor measure + pure Q ascent in raw action space" does not stand up on cube.

**Hyperparameters** (merged config, identical across seeds): `batch_size 256 · z_dim 50 ·
L_dim 50 · num_parallel 2 · discount 0.99 · f/b_target_tau 0.005 · ortho_coef 1.0 ·
train_goal_ratio 0.5 · fb_pessimism_penalty 0.0 · q_loss_coef 0.0 ·
actor_pessimism_penalty 0.0 · norm_z true · lr_f/lr_b 1e-4 · lr_actor 3e-4 ·
forward {512, 2, emb 2} · backward {512, 4, norm} · left_encoder {512, 4, norm} ·
actor {[512,512,512], alpha 0.0, const_std true, q_normalize true}`. 500k steps, ckpt every
50k, one H100 per seed, 41–45 min per run.

**Caveats.** (a) This is the *shipped* FB arm with its BC coefficient zeroed, not a tuned
pure-Q FB: the four Touati-parity knobs the repo exposes (`actor.q_normalize=false`,
`boot_noise_std/clip`, `actor_pessimism_penalty=0.5`, `q_loss_coef=1/z_dim`) all default off
and none were tried. The claim is not "no pure-Q FB can work". (b) Eval conditions FB on the
goal-Dirac `z = project_z(B(goal))`, this repo's default for `fb` on cube — *more*
information than PSMFlows hands `psmflow`, so the arm is not being under-sold. The
reward-inference route is wired and smoke-tested in `scripts/baselines/fb_eval500.py`
(`--relabel_infer`) but an arm at 0/500 under easier conditioning cannot be rescued by
harder. (c) `actor_loss` sits at exactly −1.0 all run — what `-q.mean()/sg(|q|.mean())`
degenerates to once `q` has a consistent sign; the gradient is live, the logged scalar is
not informative. (d) One seed each for the two controls. (e) The repo has no recorded FB
cube-single number to check against; `results/table.csv` carries `fb` rows for ManiSkill and
Metaworld only.

**If we port this FB back into PSMFlows** (to replace `archive/agents/fb.py`): the two
implementations are the *same maths*. `impls/critics/fb.py::stage_loss` and
`archive/agents/fb.py::_fb_loss_fn` share the off-diagonal squared TD residual, the
`-mean(diag)*P` term, the `0.5*sum((cov*off)^2)/off_sum - mean(diag(cov))` orthonormaliser,
the `train_goal_ratio=0.5` hindsight/sphere z-mix, and `infer_cond` = `(r^T B)/N` then
`project_z` — and the two yamls agree on every shared key. What the newer one adds is (i)
knobs, all defaulting to the old behaviour: `q_loss_coef` (the reference's auxiliary Q loss,
eq. 42), `left_encoder.identity` (Touati parity, F reads raw obs), `boot_noise_std/clip`
(target-policy smoothing), `q_normalize`, `motivo_archi`; (ii) a **goal-conditioned eval
route** (`map_eval_cond`: `z = project_z(B(goal))`) that PSMFlows' version has no equivalent
of — PSMFlows only ever infers `z` from rewards; and (iii) the critic×actor factoring, which
buys the Gaussian `ddpgbc` policy as an alternative to PSMFlows' in-agent `actor.type:
flow|td3` switch. A port does **not** require adopting the factoring: it is (1) re-register
`fb` in `agents/__init__.py` + restore `configs/agent/fb.yaml`, (2) copy the five knobs
across, (3) add `ddpgbc` as a third `actor.type`, (4) optionally add the goal-Dirac eval
path. Roughly a day. Because every new knob defaults to the pre-existing behaviour, the
archived equivalence fixture (`archive/tests/fixtures/fb_reference.npz`,
`test_fb_agent_equiv.py`) should still pass unchanged — which is the cheapest available
check that the port did not silently change the loss.


---

## Baseline — **PSM in raw action space, no BC anchoring: 0.000 on cube, on every seed,
## at every checkpoint**

Full write-up, hyperparameters and caveats: `docs/tables/psm_raw_nobc_cube_ladder.md`.
Repro: `scripts/baselines/psm_raw_nobc.sh`.

**What was run.** The two archived successor-measure agents, in RAW action space (dataset
action in the measure's middle slot — no behaviour flow, no preimages, no decode) with the
BC anchor deleted (`agent.actor.bc_coeff=0.0`, so the actor objective is the bare
`-Q.mean()`):

* **Arm A** `archive/agents/affine_psm.py` — the raw-action twin of the shipped agent
  (affine measure `M = Phi(s,a,x)·w + b`, factored `Phi = A(s,a) phi_x(x)`, goal-conditioned
  LP inference). **Preferred**, because the project's algorithm is affine.
* **Arm B** `archive/agents/psm.py` — the bilinear PSM of arXiv 2411.19418,
  `M = psi(s,z,a)ᵀ phi(x)`, the peer baseline of record.

`actor.type=ddpgbc` on both, so no flow-matching BC loss remains anywhere in the objective.
Pessimism kept exactly as each agent has it and reported: Arm A has **none** (only √d
normalisation, `b_scale=10`, `ortho_coef=1000`, `tau=0.01`); Arm B keeps
`actor_pessimism_penalty=0.5` over `num_parallel=2` heads (and `pessimism_penalty=0.0` on
the target, its own default). Nothing was added back.

3 seeds × 500k offline steps × 500-episode evals at every 50k, cube-single-play.

| epoch | Arm A sd0/sd1/sd2 | Arm B sd0/sd1/sd2 |
|---|---|---|
| 50k | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 |
| 100k | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.002 |
| 150k | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 |
| 200k | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 |
| 250k | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 |
| 300k | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 |
| 350k | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 |
| 400k | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 |
| 450k | 0.000 / 0.002 / 0.000 | 0.000 / 0.000 / 0.000 |
| 500k | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 |

**Late mean (300k–500k, n = 15 checkpoint×seed measurements each, 500 episodes each):
Arm A 0.000, Arm B 0.000.** Pooled over the whole sweep each arm scored **1 success in
15 000 episodes** (Wilson 95% upper bound 0.0004). Against **BC 0.072** and the affine
LatentFlowPSM's **0.424 ± 0.141** (late-checkpoint mean, ≥250k), the baseline clears
neither — it does not clear zero. The in-loop 50-episode eval read 0.00 at all 11 points of
all 6 runs, so nothing was missed between checkpoints.

---

## Why it is zero: the actor leaves the manifold, exactly as pre-registered

The representation trains fine on both arms (`psm_loss`, `orth_loss` converge normally).
What runs away is the greedy actor's own Q:

| arm | seed | Q @5k | @50k | @100k | @250k | @500k |
|---|---|---|---|---|---|---|
| A | 0 | 60 | 249 | 110 | 16 | 10 |
| A | 1 | 142 | **1810** | 388 | 20 | 67 |
| A | 2 | 60 | 15 | 23 | 22 | 28 |
| B | 0 | 406 | 1440 | 1820 | 1900 | **1970** |
| B | 1 | 407 | 1290 | 1970 | 1950 | **2340** |
| B | 2 | 372 | 1440 | 1790 | 1920 | **2080** |

Arm B's Q is `psi(s,z,a)ᵀz` with `norm_z=true`, so `|z| = √128 = 11.3` and Q ≈ 2000 means a
`psi` norm of ~175 at the action the actor picked — a value the contrastive loss never sees
on data, reached by 100k and held to the end. Two ensemble heads with
`actor_pessimism_penalty=0.5` do not stop it: they agree about an action neither was
trained on. Arm A's factored measure is *unnormalised* (only `phi_x` is √k-normalised, so
`Phi = A·phi_x` is free), and its Q spikes the same way early.

**This is the control the anchor exists for, and it is a clean negative.** It bounds the
*pair* (raw actions + no anchor), not either alone: the separating cell — raw actions
**with** `bc_coeff=3.0` — is the archived `affine_psm` cube push, PARKED 2026-07-26 with no
500-episode ladder. Not a tuned baseline either: archived cube hyperparameters, one flag
changed, no sweep.

---

## Infra: running an archived agent without un-archiving it

`archive/` stays frozen — nothing moved, nothing edited, and `main.py`,
`agents/__init__.py`, `agents/psmflow.py`, `agents/fql.py`, `utils/psm_networks.py` and
`tools/eval_checkpoint.py` are untouched. New, all under `scripts/baselines/`:

* `_archive.py` — imports `archive.agents.*` as a namespace package and registers a hydra
  `SearchPathPlugin` so `agent=affine_psm` resolves to `archive/configs/agent/`.
* `run_archived.py` — `main.py` for archived agents. Restores the two seams the live entry
  point dropped: `dataset.return_index = True` (the proto sampler keys `pi_z` on the global
  buffer row index, not batch position) and the goal-conditioned eval branch
  (`infer_w_goal` -> `infer_eval`), both verbatim from pre-archive `main.py` (`db96e48`).
* `eval_archived.py` — 500-episode eval; **reuses** `tools/eval_checkpoint.py`'s
  `merge_run_config` / `_cli_agent_keys` / `wilson` unchanged, so the agent config is
  inherited from the run's own `flags.json` (verified on every job:
  `actor.bc_coeff: config 3.0 -> run 0.0`).
* `train_archived.sbatch`, `eval_archived.sbatch`, `psm_raw_nobc.sh`,
  `psm_raw_nobc_table.py` (regenerates the ladder from the JSONs).

Deleting `scripts/baselines/` reverts the repo exactly. Discipline log: 200-step smokes of
both arms (2491695/6) and of the eval path (2491712/3) before launch; `flags.json` re-read
after launch on all 6 runs; expected failure mode stated before results; every number from
a persisted JSON.

---

## Antmaze H1 — **pinning the GPI policy index does not rescue antmaze; it destroys it**
## (11 of 12 cells significantly BELOW behaviour cloning, 6 of them exactly 0/500)

Pre-registration: `docs/design/2026-09-06-antmaze-failure-tests.md` §2, written and
committed to before any job was submitted; §5 is the results section and §2 was not edited
after they returned.

**The hypothesis.** Affine paper-strict is null on antmaze (late-ckpt mean 0.081 ± 0.050 vs
BC 0.072, episodes timing out at 1000 steps) while the arms with a *persistent* policy — the
affine latent-actor arm — score ~0.21 on the same env, flow and checkpoints. H1: the acting
rule redraws `K=64` policy indices `u'` **every step** and maximises over them, so the
executed behaviour is a different policy at every step of a 1000-step maze episode, and that
temporal incoherence is the failure.

**The test.** Eval-time only, no retraining, no code changes: `agent.gpi_select=fixed_index`
pins `u'` to `PRNGKey(agent.gpi_index_seed)` for the whole evaluation while the inner
`argmax_u` over 64 action latents still runs per step. Three checkpoints x four index seeds,
500 episodes each, config inherited from each run's own `flags.json` with only
`gpi_select`/`gpi_index_seed` typed on top (confirmed in each JSON's `agent_config_source`).
SLURM 2491818-2491829, all COMPLETED, 11 min each — **64x cheaper than the shipped rule**,
because `n_idx=1` collapses the `K x K = 4096` pair scan to `K = 64`.

---

## H1 results: 500 episodes per cell, Wilson 95%, BC control 0.072

| checkpoint | argmax ref | index seed 0 | 1 | 2 | 3 |
|---|---|---|---|---|---|
| sd1 @250k | 0.178 | 0.002 [0.000, 0.011] | **0.142** [0.114, 0.175] | 0.000 [0.000, 0.008] | 0.000 [0.000, 0.008] |
| sd2 @300k | 0.088 | 0.020 [0.011, 0.036] | 0.026 [0.015, 0.044] | 0.004 [0.001, 0.015] | 0.000 [0.000, 0.008] |
| sd0 @350k | 0.082 | 0.000 [0.000, 0.008] | 0.000 [0.000, 0.008] | 0.000 [0.000, 0.008] | 0.000 [0.000, 0.008] |

Pooled over the 12 cells: **97 / 6000 = 0.016**, against BC 36/500 = 0.072 —
z = -8.47, **p = 2.4e-17**. Every cell except sd1/is1 is individually below BC at
p <= 8.7e-05. Reference points: BC **0.072**, affine strict late-ckpt mean **0.081 ± 0.050**,
affine **latent-actor** arm (persistent policy) **~0.21** (0.224/0.206 @500k, 0.218/0.224
@50k, 0.176/0.216 @100k).

**Verdict: H1 refuted, in the opposite direction.** The pre-registered not-H1 threshold was
"every cell <= 0.10"; the cells are an order of magnitude below it. The single exception,
sd1 @250k index seed 1 at 0.142, is not a lift: it sits on the one checkpoint that already
scores **0.178** with the shipped per-step argmax (-0.036, p = 0.12, no significant change),
and that same checkpoint's other three index draws return 0.002 / 0.000 / 0.000. §2
pre-registered exactly this caveat — a move confined to the best-of-eight checkpoint is
about the checkpoint, not the rule. The checkpoints the hypothesis needed (argmax at the
floor, 0.082 / 0.088) return 0.000 on seven of their eight cells.

---

## What H1 changes

1. **It reproduces the 09-05 cube `fixed_index` finding on a second env and task family.**
   Cube: 0.704 -> 0.000 (index seed 0) / 0.180 (seed 1). Antmaze: 0.082-0.178 -> 0.000-0.142.
   The per-step max over 64 fresh policy indices is **load-bearing on both**, and it behaves
   the same on a 200-step manipulation task and a 1000-step navigation task. It is an
   optimism device averaged over the index lottery every step, not a policy choice — and
   removing it is worse than behaviour cloning, not merely equal to it.
2. **Temporal incoherence is not why antmaze fails.** The per-step reselection is the only
   thing holding this arm *at* BC rather than at zero.
3. **The latent-actor arm's ~0.21 is not "persistence".** A pinned index is maximally
   persistent and pools to 0.016. Whatever the actor arm has comes from the amortized actor
   being *trained* — a distillation over the index panel with its own learning signal. That
   makes the actor arm, not the strict arm, the interesting antmaze object.
4. **The index lottery is enormous on antmaze.** Four draws of `u'` on one checkpoint give
   0.142 / 0.002 / 0.000 / 0.000. Conditioned on a fixed index, `psi`'s preference over
   action latents is worth between -0.072 and +0.070 against BC depending on the draw: the
   learned measure has no index-independent notion of a good action.

Reports: `$PSM_DATA/logs/eval500_antmaze_fixedidx_sd{S}_{epoch}k_is{I}.json`, registered as
12 separate `ablation` rows in `tools/make_tables.py` (they never pool — a mean over the
four index seeds of one checkpoint would hide the entire finding).

---

## Antmaze H2 (pre-registration) — is `discount=0.98` (~50-step horizon) too short
## for a 1000-step maze?  [results in the next two slides]

Pre-registered in the same note, §3. Six retrains of the repo-default affine-strict antmaze
agent with **`agent.discount=0.99`** and **`0.995`** (horizons 100 and 200 vs the default's
50), 3 seeds each, 500k steps, `save_interval=50000`. SLURM 2491831-2491836, launched
2026-09-06 22:18, ~4 h expected. Groups `affine_strict_antmaze_g99` /
`affine_strict_antmaze_g995`.

Pre-registered expectation: under H2 the in-loop eval leaves the floor by ~250k and the
500-episode success at 500k is >= 0.15 for `gamma=0.995`, with `gamma=0.99` intermediate,
**monotone in the horizon**; under not-H2 all six stay in 0.00-0.10 like `gamma=0.98`. An
anticipated failure mode was stated in advance: a longer discount can *destabilise* the TD
backup, and `gamma=0.995` collapsing earlier than `gamma=0.98` is a plausible outcome.
The in-loop 50-episode eval over-reads badly on antmaze (sd2 @300k read **0.40** in-loop and
**0.088** at 500 episodes) and is used only to pick which checkpoint gets a 500-episode eval.

---

## Antmaze H2 — **CONFIRMED: the discount was the blocker.** gamma=0.99 pools to
## **0.294 [0.224, 0.364]** over a 30-cell ladder against gamma=0.98's **0.076**

All six retrains COMPLETED (SLURM 2491831-2491836, 3 h 53 m - 4 h 10 m each); every
`flags.json` re-read after launch differs from `affine_strict_antmaze/sd001` in `discount`
alone. The ladder is **51 500-episode cells** — gamma=0.99 at every 50k checkpoint 50k-500k
(30) and gamma=0.995 at 50k-300k plus a 500k endpoint (21) — all on the serial eval path
(`EVAL_WORKERS=1`), the one the 12 H1 cells and every historical antmaze number used. Full
per-cell table: `docs/tables/affine_antmaze_discount_ladder.md`. Figure:
`docs/figures/2026-09-07-affine-antmaze-discount-ladder.{png,json}`.

| arm | horizon | window | n | mean | 95% CI | min | max |
|---|---|---|---|---|---|---|---|
| gamma=0.98 (default) | 50 | all measured | 10 | 0.076 | ± 0.043 | 0.000 | 0.178 |
| **gamma=0.99** | 100 | **all measured** | 30 | **0.294** | **± 0.070** | 0.004 | 0.624 |
| gamma=0.99 | 100 | 50k-250k | 15 | 0.336 | ± 0.097 | 0.050 | 0.624 |
| gamma=0.99 | 100 | 300k-500k | 15 | 0.252 | ± 0.100 | 0.004 | 0.590 |
| gamma=0.995 | 200 | all measured | 21 | 0.214 | ± 0.092 | 0.002 | 0.678 |
| gamma=0.995 | 200 | 50k-250k | 15 | 0.296 | ± 0.102 | 0.074 | 0.678 |
| gamma=0.995 | 200 | **300k-500k** | 6 | **0.010** | ± 0.008 | 0.002 | 0.020 |

BC control **0.072**; the affine latent-actor arm — previously the only antmaze arm off the
floor — **~0.21**. gamma=0.99's interval is **disjoint from gamma=0.98's**, it is ~3.9x BC,
and it is the first antmaze arm whose *late* window stands on its own (300k-500k = 0.252 ±
0.100, n=15). **Antmaze-medium is not structurally closed to this agent.** The 50-step
effective horizon of gamma=0.98 was the obstruction, and the failure it produced — episodes
running to the 1000-step timeout with success pinned at BC — is exactly what a value
function blind past 50 steps does on a maze whose goal is hundreds of steps away.

---

## H2: the pre-registered signature is wrong on BOTH halves, and that is the useful part

§3 pre-registered "in-loop leaves the floor by ~250k **and** 500-ep success at 500k >= 0.15
for gamma=0.995, **monotone in the horizon**".

* **Not monotone.** gamma=0.995 beats gamma=0.99 only at 50k-100k (0.492/0.519 vs
  0.377/0.305). Pooled over any window it does not: 50k-250k 0.296 vs 0.336, all-measured
  0.214 vs 0.294. Horizon 100 is enough; horizon 200 is past the useful point.
* **gamma=0.995 collapses, now dated at 500 episodes.** Its 300k-500k window is **0.010 ±
  0.008 over 6 cells**, every one between 0.002 and 0.020 — *below the BC control* — on all
  three seeds at both checkpoints (300k: .008/.020/.018; 500k: .006/.004/.002). This was the
  destabilisation mode named in advance, and it is **the most reproducible behaviour this
  agent has ever shown**: three of three seeds, 200k steps apart, same near-zero value.
* **A 500k-only eval would have got this wrong in both directions** — reading gamma=0.995 as
  null (0.004) and gamma=0.99 as marginal (0.170 ± 0.227, its *worst* checkpoint). The
  originally specified "500k plus best in-loop" protocol was inadequate; the 50k ladder is
  what makes the result readable. Nothing here may be quoted at a single checkpoint.

**Seed spread is still the dominant term**, as on cube: within gamma=0.99, seed 2 runs
0.43-0.62 across the whole ladder while seed 1 wanders 0.004-0.414, and the across-seed std
is 0.14-0.28 at every epoch.

**Unregistered calibration finding: the in-loop 50-ep eval is well calibrated here.**
gamma=0.995 in-loop at 100k read .32/.56/.78 against 500-episode .332/.546/.678, and its
in-loop 0.000 from 300k is confirmed at 0.002-0.020 over 500 episodes. The 0.40 -> 0.088
over-read recorded for gamma=0.98 is a property of an agent **sitting at the floor** — a
50-episode sample of a near-zero rate is almost all noise — not a property of antmaze.

---

## H2 provenance and what remains

* Eval: **51 COMPLETED jobs** — 2491922-2491923, 2491961-2491964, 2491984-2492016,
  2492023-2492025, 2492030-2492038. All 51 JSONs verified programmatically: 500 episodes,
  `restore_epoch` matching the filename's epoch token, `num_workers=1`.
* 2491912/2491913 died at startup on another agent's in-flight `agents/psmflow.py` change
  (`KeyError: 'entropy'`), wrote no JSON, and were resubmitted as 2491922/2491923. No other
  eval failed.
* `tools/fig_affine_antmaze_discount_ladder.py` is idempotent and partial-safe: it reads
  whatever JSONs exist, renders the rest as gaps, and prints a coverage line
  (`gamma=0.99: 30, gamma=0.995: 21` = complete). Re-run it plus `tools/make_tables.py`,
  which carries 19 generated H2 rows (one per discount x checkpoint, seeds pooled, never
  pooled across checkpoints — gamma=0.995 wins early and collapses late, so an early mean
  and a late mean describe different agents).

**What remains.** (1) Make `discount=0.99` the antmaze setting and decide whether it becomes
the repo default — a one-line config change with a 30-cell three-seed ladder behind it.
(2) Cube and pointmaze at gamma=0.99, in flight with another agent
(`docs/design/2026-09-07-discount-sweep.md`); cube should be unaffected and pointmaze should
stay at zero for the COMPENDIUM 4.11 reason, and those two pre-registered negatives are what
would confirm the mechanism is *horizon* rather than "a bigger gamma helps everything".
(3) The gamma=0.995 collapse is an unusually clean object — `w_enc_spread` and the psi-spread
diagnostics run straight through it (third panel of the figure) and whatever explains it may
also explain the cube arm's ±0.4 within-run swings. (4) The gamma=0.98 antmaze ladder still
has 20 of 30 cells missing; filling it is 20 evals of already-trained checkpoints and would
make the three-arm comparison exact rather than pooled-over-what-exists. (5) Combined with
H1, the antmaze story is now **"the critic was horizon-starved"**, not "the acting rule was
temporally incoherent".

Discipline log: pre-registration written before submission; a 200-step
smoke of the exact training path (SLURM 2491830) whose `flags.json` differed from
`affine_strict_antmaze/sd001` **only** in `discount` (plus the three eval-only `gpi_select`
keys added 09-05, which training never reads); all six runs' `flags.json` re-read after
launch and confirmed to carry 0.99 / 0.995; the H1 eval path verified from job 2491818's log
(`CLI overrides kept: ... gpi_index_seed, gpi_select`) before reading any number; every
number from a persisted JSON.

---

## H3 (in flight) — **does the discount horizon matter beyond antmaze?** cube and
## pointmaze retrained at `gamma=0.99`, with opposite predictions registered for the two

Pre-registration: `docs/design/2026-09-07-discount-sweep.md`, written and committed to
before any job was submitted. Nothing in `agents/` or `utils/` is touched; the only change
is `agent.discount`.

**Why.** The 09-06 antmaze H2 test (section above) is still running, but two things are
already visible. `gamma=0.99` reads **0.534** and `gamma=0.995` reads **0.678** at 100k on
seed 2 over 500 episodes, against the `gamma=0.98` arm's late-checkpoint mean **0.081 ±
0.050** and BC **0.072** — a 6-8x move on the env this project had written off. And
`gamma=0.995` then **dies**: all three of its seeds read exactly 0.00 in-loop at 300k and
350k while all three `gamma=0.99` seeds hold (0.28/0.24/0.64 and 0.28/0.12/0.46). H2's §3
named that destabilisation as an anticipated failure mode before the runs started, and it
fired. `gamma=0.99` is therefore the value carried forward: it is the one that both moved
and survived.

**The test.** Six retrains of the repo-default paper-strict affine agent (`agent=psmflow`),
3 seeds x 2 envs, 500k steps, `save_interval=50000`, groups `affine_strict_cube_g99` and
`affine_strict_pointmaze_g99`. SLURM **2492017-2492019** (cube sd0/1/2) and
**2492020-2492022** (pointmaze sd0/1/2), each gated `--dependency=afterok` on its env's
200-step smoke (**2491981** cube, **2491982** pointmaze) so no training job can start unless
the smoke of the exact code path exited 0. Everything else is byte-identical to
`affine_strict_cube` / `affine_strict_pointmaze`: same frozen flow, same preimage npz, same
budget, same eval protocol.

---

**The two envs were chosen because they predict opposite things** — a sweep that expects the
same outcome everywhere tests nothing.

* **Cube: no change.** Cube episodes are 200 steps and the reward is reachable well inside
  `gamma=0.98`'s ~50-step horizon, so unlike antmaze (goal 200-400 steps away,
  `0.98^300 ≈ 2e-3`) cube is not horizon-starved. Registered point prediction for the
  300k-500k late mean: **0.42**, interval **0.33-0.50**, i.e. within ±0.09 of the
  `gamma=0.98` value **0.415 ± 0.083**. Registered separately and more falsifiably: the
  **oscillation does not shrink** — at least two of three seeds will still show a
  >= 0.30 jump between adjacent 50k checkpoints and a >= 0.35 range over the ten. Nothing in
  the diagnosed mechanism (the `w_enc_spread` encoder collapse, the per-step index lottery
  of 09-06 H1) is a function of `gamma`, so a *stabilised* ladder would contradict the 09-06
  diagnosis and is recorded here as the outcome I consider least likely.
* **Pointmaze: 0.000 everywhere, and the discount cannot change it.** COMPENDIUM §4.11 is a
  settled negative with a mechanism upstream of anything `gamma` touches: a 13x13 grid tiles
  the whole `d_a=2` latent box, and **0 of 233 enumerated latents reach the goal** in two
  full 1000-step rollouts each; the expert's route is latent white noise (within-episode
  preimage variance / marginal = **0.99**) because the goal is not in the observation, so
  routes exist only as latent *sequences*. `discount` changes which member of the policy
  family `psi` prefers; it does not add a member to a family that has none. Registered
  prediction: **0.000 on all 30 cells** (15000 episodes), anything <= 0.004 counted as
  consistent since BC itself is 0.002. The one channel that is not a flat no is stated in
  the note: the deployed rule redraws 64 indices per step, so it executes a *sequence*, and
  §4.11's probe used a fixed `u` — if per-step stitching were possible, a 50-step horizon in
  a ~200-step maze is exactly where it would be invisible. **Any cell >= 0.05** would reopen
  §4.11 for per-step-reselecting acting rules; cells in 0.004-0.05 are floor noise and will
  be reported as noise, not as a partial rescue.

---

**Failure modes named in advance.** (1) The `gamma=0.995`-style late collapse — in-loop
success pinned at *exactly* 0.00 on all three seeds from some epoch on; monitored per 50k in
each run's `eval.csv`, and it does **not** abort the ladder, because the in-loop number is
the thing this project has repeatedly caught over-reading (antmaze sd2 @300k: 0.40 in-loop,
0.088 over 500). (2) `training/w_enc_spread` crossing 0.6 markedly earlier than the
`gamma=0.98` runs' 190k / 215k / 305k. (3) TD divergence in `psm_loss` / `psi_q_spread` /
`psi_q_range_rel`. (4) Reading the answer off the in-loop eval at all — listed as a failure
mode because it is a process failure this project has committed before.

---

**Eval plan.** 500 episodes at **every** 50k checkpoint for **all** seeds — 30 cells per env,
60 jobs — with `EVAL_WORKERS=1`. That is deliberate: `scripts/eval500.sh`'s parallel path
seeds worker `w` as `seed*N + w` and therefore draws a *different* sample of 500 episode
inits, so it agrees with an existing number only within the Wilson interval. Every
`gamma=0.98` ladder this sweep is compared against was produced serially, and the comparison
is uniform only at N=1. Reports
`$PSM_DATA/logs/eval500_affine{N}k_strict_{cube|pointmaze}_g99_sd{S}.json`, disjoint from
every existing basename. Table `docs/tables/affine_discount_ladder.md` and figure
`docs/figures/2026-09-07-affine-discount-ladder.{png,json}` (generator
`tools/fig_affine_discount_ladder.py`, which carries both discounts for both envs so the
object plotted is always the pair); 22 append-only rows in `tools/make_tables.py`, the
`_g99` infix keeping them from pooling with the `gamma=0.98` rows, whose basenames carry no
discount token.

Discipline log: hyperparameter table printed in the note's §6 before submission; 200-step
smokes of the exact training path submitted per env (2491981 / 2491982, both COMPLETED
0:0 in ~2 min) and made a hard `afterok` gate on all six trainings; both smoke
`flags.json` diffed against their `gamma=0.98` references before the gate released, which
caught real config drift since the antmaze `g99` runs 14 h earlier (`agent.actor_mode`,
the `agent.actor.*` DSRL keys, `eval_workers`) and established it is inert for this arm --
all of it is actor-only code gated on `train_actor=false`, or an eval-time knob whose
default is the serial path (note §8.0); expected outcomes *and* expected failures registered before any number
arrived; and all six runs' own `flags.json` re-read after launch (§8.1) — **identical
across seeds within each env**, and differing from their `gamma=0.98` references in
**`agent.discount` alone**. SLURM 2492017-2492022 started 07:49-08:10.

---

## Every cube number this project has ever reported is OGBench **task 2**.
## Re-evaluated on all five: 0.284 avg vs BC 0.111 — the zero-shot claim survives, unevenly

`cube-single-play-singletask-v0` carries no task token, and `cube_env.py:359` defaults
`reward_task_id` to **2**; `utils/evaluation.py:86,93` resets with `env.reset()` and never
passes `options={'task_id': ...}`, so the bare id has always been one task out of five.
Stated plainly, for the record:

- **every cube number in this file, in `docs/tables/results.md`, in the ladder tables and in
  `PAPER/ICLR/tables/table_headline.tex` — headline, arms, ablations — is task 2 alone;**
- **every antmaze and pointmaze number is task 1** (`locomaze/maze.py:348` defaults mazes to 1).

Nothing had to be retrained to fix that. `agents/psmflow.py` reads `batch['rewards']` **only**
in `infer_z`/`infer_z_a`/`infer_eval_z` (`:609-628`) — the eval-time closed-form task vector —
so the frozen flow, phi, psi and the latent index are reward-free.
`ogbench/utils.py:199-202` loads the **same** `cube-single-play-v0.npz` for every task id and
only calls `relabel_dataset`, which reads `env.unwrapped._reward_task_id`. So changing
`env_name` to `cube-single-play-singletask-task{K}-v0` gives each task its own dataset reward
column, its own inferred `w`, and its own success criterion, against one unchanged
checkpoint. `tools/eval_checkpoint.py` loads no preimage npz and never runs `main.py`'s
pairing guard, so the env-name change trips nothing (verified: env ids
`cube-single-singletask-task{1..5}-v0` are all registered, `_reward_task_id` reads 1/2/3/5 as
expected, and the bare id reads 2).

---

## The five-task table (500 episodes per cell, 3 seeds x 5 checkpoints 300k-500k)

64 SLURM jobs: 4 tasks x 5 checkpoints x 3 seeds on the finished `affine_strict_cube` runs,
plus one BC control per task. No arm flags on any eval line — the agent config came from each
run's own `flags.json` (`policy_index=latent acting=gpi train_actor=false psi_form=affine`,
confirmed in every report JSON).

| task | n (ckpt x seed) | late mean 300k-500k ± 95% CI | min … max | BC control (500 ep) | ratio |
|---|---|---|---|---|---|
| task 1 | 15 | 0.314 ± 0.123 | 0.000 … 0.832 | 0.152 [0.123, 0.186] | 2.1x |
| **task 2** (the default — all earlier numbers) | 15 | **0.415 ± 0.083** | 0.086 … 0.704 | 0.072 [0.052, 0.098] | 5.8x |
| task 3 | 15 | 0.480 ± 0.074 | 0.190 … 0.730 | 0.230 [0.195, 0.269] | 2.1x |
| task 4 | 15 | 0.100 ± 0.034 | 0.038 … 0.294 | 0.086 [0.065, 0.114] | 1.2x |
| task 5 | 15 | 0.113 ± 0.045 | 0.002 … 0.296 | 0.014 [0.007, 0.029] | 8.1x |
| **5-task average** | 75 cells | **0.284** (± 0.152 across task means) | — | **0.111** | **2.6x** |

Table: `docs/tables/affine_cube_multitask.md` (generated) and a section in
`docs/tables/results.md`. Figure + series:
`docs/figures/2026-09-06-affine-cube-multitask.{png,json}`
(generator `tools/fig_affine_multitask.py`).

---

## Per-checkpoint mean across the three seeds

| epoch | task 1 | task 2 | task 3 | task 4 | task 5 |
|---|---|---|---|---|---|
| 300k | 0.507 | 0.393 | 0.339 | 0.087 | 0.213 |
| 350k | 0.258 | 0.516 | 0.410 | 0.165 | 0.121 |
| 400k | 0.375 | 0.391 | 0.505 | 0.091 | 0.091 |
| 450k | 0.233 | 0.468 | 0.532 | 0.097 | 0.054 |
| 500k | 0.195 | 0.309 | 0.616 | 0.061 | 0.087 |
| BC | 0.152 | 0.072 | 0.230 | 0.086 | 0.014 |

Per-seed late means (5 checkpoints each), showing the 09-05/09-06 oscillation is **not**
task-2-specific and is not a fixed per-seed ranking either:

| task | sd0 | sd1 | sd2 |
|---|---|---|---|
| 1 | 0.416 | **0.074** | 0.451 |
| 2 | 0.326 | 0.498 | 0.422 |
| 3 | 0.464 | 0.530 | 0.446 |
| 4 | 0.103 | 0.130 | 0.068 |
| 5 | 0.126 | 0.048 | 0.164 |

---

## What this settles, and what it does not

**The zero-shot claim survives, and it is no longer a one-task claim.** One representation per
seed, five reward functions inferred in closed form, all beating or matching their own BC
control: **0.284 vs 0.111 averaged over the five, 2.6x.** Every task is at or above BC; three
of five (2, 3, 5) are clearly above it, task 1 marginally so (CI lower bound 0.191 vs the BC
Wilson upper bound 0.186).

**Task 2 was a lucky draw for the headline, and the paper must stop quoting it alone.** Its
5.8x is the second-best *ratio* but rests on the weakest BC control of the five (0.072). Read
by absolute success the arm is best on task 3 (0.480) and task 2 (0.415), and lands near
0.10 on tasks 4 and 5.

**Task 4 is the honest null.** 0.100 ± 0.034 against BC 0.086 [0.065, 0.114] — indistinguishable.
The method neither helps nor hurts there. This is the first cube task on which the arm has no
effect, and it is worth understanding before the next environment: whatever makes task 4 hard
is *not* the inversion (the same latents serve all five tasks) and *not* the representation
(same phi/psi), so it is the reward-inference step or the task's own difficulty.

**Task 5 is where the method earns its claim most cleanly.** 0.113 vs a BC floor of 0.014 —
8.1x — a task the behaviour flow essentially cannot do at all, done eight times as often by
selecting inside the same flow's latent space. Small absolute number, but the cleanest
separation from the control in the whole project.

**The oscillation is a property of the arm, not of the task.** The per-seed table shows the
same non-convergence found on 09-05/09-06: seed 1 is best on task 2 (0.498) and worst on task
1 (0.074); no seed dominates. Task 3 is the exception — monotone 0.339 → 0.616 across the five
checkpoints on all three seeds, the only clean learning curve any cube task has produced.

**Consequence for the roadmap.** `docs/plans/2026-09-06-env-roadmap.md` §6.0 costed this at
~0 GPU-hours of training and predicted it would multiply the evidence base by five. It did:
75 cells instead of 15, on artifacts that already existed. Every future eval — including the
next environment — should report all five tasks, and `scripts/eval500.sh` now takes an
`ENV_NAME` override for exactly that.

---

## Tooling changed with this result

- `scripts/eval500.sh` / `scripts/slurm/eval500.sbatch`: `ENV_NAME` overrides the ENVKEY's
  default env id (this is the whole multi-task mechanism); the sbatch also lets a preset
  `EVAL_LOGS` through, so BC controls land in `$PSM_DATA/evals` beside `bc_cube.json`.
- `tools/eval_checkpoint.py`: the report now records **`train_seed`**, read from the restored
  run's own `flags.json`. `seed` was and remains the *eval* seed (0 on every eval500 line ever
  run), so until now a JSON carried no machine-readable trace of which training seed produced
  it — only `restore_path` and the filename. Covered by four new cases in
  `tests/test_eval_checkpoint_flags.py` (18 pass).
- `tools/fig_affine_multitask.py` (new) and a `multitask_section` in `tools/make_tables.py`;
  `docs/tables/results.md`'s header now states the task-2 scope of every other row.

Discipline log: full hyperparameter table printed before submission; a 5-episode task-1 smoke
(SLURM 2491837) confirmed the env id, the inherited arm config and the report's `env` field
before the 64 jobs went in; all 60 + 4 JSONs re-validated after the fact for env id, epoch,
episode count, `train_seed` and arm flags (0 mismatches); every number here from a persisted
JSON. Nine jobs report SLURM state FAILED with exit 126 *after* writing a complete report and
printing the final success line — a teardown artifact, not a lost eval; all 60 cells are
present and were validated.

---

## Pointmaze — **the third published env now has a complete `gamma=0.98` ladder:
## 0.000 in all 30 cells, 15,000 episodes, zero successes** (and it is horizon-confounded)

Table: `docs/tables/affine_pointmaze_ladder.md` (generated). Rows registered in
`tools/make_tables.py`; `docs/tables/results.md` regenerated.

The third environment with published Stage-A/B artifacts was brought up end to end and run
at the repo default (`agent=psmflow`, no flags = paper-strict affine). Three seeds to 500k
(SLURM 2491814/15/16, **4h00 / 3h55 / 4h00**), `save_interval=50000`, then the full
500-episode ladder at every 50k checkpoint on every seed — **30 of 30 cells, no gaps**.

| epoch | sd0 | sd1 | sd2 | mean ± std |
|---|---|---|---|---|
| 50k … 500k, all ten rungs | 0.000 [0.000, 0.008] | 0.000 [0.000, 0.008] | 0.000 [0.000, 0.008] | 0.000 ± 0.000 |

| pooled | n | mean | 95% CI |
|---|---|---|---|
| 300k–500k | 15 | **0.000** | ± 0.000 |
| all rungs 50k–500k | 30 | **0.000** | ± 0.000 |
| **BC control** (frozen flow, per-step prior) | 500 ep | **0.002** (1/500) | Wilson [0.0004, 0.0112] |

The in-loop 50-episode evals agree: `evaluation/success` is `0.0` at **every** logged point
on all three seeds, step 1 through 500k. No figure was produced — 30 identical zeros against
33 identical in-loop zeros plot as a flat line on the axis and would dress a structural null
as a measurement.

Note the control: pointmaze BC is **0.002**, not antmaze's 0.072. This env offers almost no
bar to clear, and "the agent is below BC" would be over-reading a one-episode difference.
Both the method and its control are at zero here.

---

## The pointmaze null is CONFOUNDED by the same discount that manufactured antmaze's

This ladder ran at `discount=0.98` — a ~50-step effective horizon — on a maze with a
**1000-step** episode limit. The H2 section above then showed that this exact setting, not
the environment, is what pinned antmaze at the BC floor (`gamma=0.995` → **0.519 ± 0.175**).
So the correct statement is **"the affine arm scores 0.000 on pointmaze at `gamma=0.98`"**,
not "pointmaze is closed to this method".

COMPENDIUM §4.11 remains the standing prior — 0 of 233 enumerated latents reach the goal;
within-episode preimage variance / marginal = 0.99, so routes exist only as latent
*sequences* — and it was pre-registered here as the expected outcome before results arrived.
But it does not settle this table, for two reasons. The reachability probe rolled **fixed-`u`**
policies while this arm acts by `gpi`, re-drawing `u` from 64 candidates every step (a
strictly larger family); and whether the critic can *steer* that per-step choice toward a
goal hundreds of steps away is exactly what the discount controls. The coherence mechanism
and the horizon mechanism are not alternatives, and a `gamma=0.98` run cannot separate them.

**This ladder is therefore the baseline for H3's `affine_strict_pointmaze_g99` arm**
(SLURM 2492020-2492022, pre-registered in `docs/design/2026-09-07-discount-sweep.md` with the
prediction that pointmaze stays at 0.000). Read the two together. A `gamma=0.995` pointmaze
arm was *not* launched: H2 showed 0.995 destabilises (all three antmaze seeds at exactly 0.00
in-loop from 300k), so 0.99 is the value worth spending GPUs on, and H3 already has it.

---

## Infra — the OGBench dataset host is dead; a HF mirror is the working route

`pointmaze-medium-navigate-v0.npz` was missing locally and **could not be downloaded from
upstream**. `https://rail.eecs.berkeley.edu/datasets/ogbench/<name>.npz` now 302s to
`https://iris.eecs.berkeley.edu/datasets/ogbench/<name>.npz`, which returns a **404**
WordPress page. This hits *every* OGBench dataset, including `cube-single-play-v0.npz` which
we already hold — it is an upstream move, not a proxy problem (the no-proxy route fails TLS at
the same host, and ogbench 1.2.1 on PyPI plus GitHub master still hardcode the dead URL, so no
version bump or compute-node retry helps).

**What worked:** the HF mirror `ryanhoangt/ogbench_data`, which carries the standard
`<env>-v0.npz` / `-val.npz` at repo root.

```bash
.venv/bin/python - <<'EOF'
import os, shutil
from huggingface_hub import hf_hub_download
dst = os.path.expanduser('~/.ogbench/data')
for f in ['<env>-v0.npz', '<env>-v0-val.npz']:
    shutil.copyfile(hf_hub_download('ryanhoangt/ogbench_data', f, repo_type='dataset'),
                    os.path.join(dst, f))
EOF
```

Verified against the preimage npz (`obs[0]`, `act[0]` match to the pipeline's action clip),
and `main.py`'s row-count + first/last-1000-row guard would refuse a wrong copy anyway. The
mirror also holds `scene-play`, `cube-double/triple/quadruple-play` and `puzzle-3x3/4x4-play`,
so the outage blocks no environment on the roadmap. Also worth knowing:
**`OGBENCH_DATASET_DIR` is a no-op** with ogbench 1.1.0 — `ogbench/utils.py:10` hardcodes
`~/.ogbench/data` and `envs/env_utils.py:139-140` never passes `dataset_dir`; it works here
only because `$HOME/.ogbench/data` is the same path.

Preimages came from `scripts/hf_preimages.py pull --name pointmaze-medium-navigate --dest
$PSM_DATA --with-flow` (52 MB, 1M rows, sidecar `restore_path` repaired automatically).

---

## Provenance / caveats (pointmaze)

* `flags.json` re-read after launch on all three seeds: byte-identical to
  `affine_strict_cube/sd000` except `env_name`, `run_group`, the flow/preimage paths and the
  three eval-only `gpi_select` keys added 09-05.
* Ten ladder cells (100k sd2, 150k/200k/250k all seeds; SLURM 2491907-2491921) died at
  startup in 20-41 s with `KeyError: 'entropy'` — an `assert a_cfg["entropy"]` against a key
  these runs' `flags.json` predates, since `eval_checkpoint.py` inherits the run's own config.
  Fixed in `agents/psmflow.py` by another agent (`_actor_opt` fallback + `fill_actor_defaults`
  as the first statement of `create`); `tests/test_psmflow_config_compat.py` passes 9/9. All
  ten were resubmitted (2491965-2491974) and completed. **They wrote no JSON, so no partial
  result entered the table** — the ladder was regenerated only once all 30 files existed.
* Every cell is `tools/eval_checkpoint.py`, 500 episodes, agent config from the run's own
  `flags.json`, no arm flags on the CLI. The generator asserts `restore_epoch` and
  `num_episodes` per file, so a mislabelled JSON cannot silently populate a cell.
* Eval throughput on this arm: ~11 s/episode serial (measured in the 200-step smoke), i.e.
  ~1.5 h per 500-episode cell; later cells used the new `EVAL_WORKERS=4` default.

---

<!-- _class: lead -->

## 2026-09-05 — the affine strict arm **oscillates**: cube swings 0.086 ↔ 0.704
## between adjacent 50k checkpoints, and no logged diagnostic predicts which

The 09-04 entry quoted this arm at 250k (0.532 / 0.620) because that was the latest
checkpoint the runs had reached. All eight runs have now finished 500k and the full
checkpoint ladder has been evaluated at 500 episodes. **The 250k number was not the arm
converging — it was one draw from a wide, non-monotone distribution over checkpoints.**
The headline claim of 09-04 does not survive as stated; the arm is still the best thing on
cube, but only as a *mean over late checkpoints*, and it is dead on antmaze.

Figures + machine-readable series: `docs/figures/2026-09-05-affine-cube-collapse.{png,json}`,
`docs/figures/2026-09-05-affine-antmaze-sweep.{png,json}`.

---

## Repo reorg: done

The 09-04 reorganisation is complete and is the state of the tree: `agent=psmflow` defaults
to the paper-strict affine arm (`psi_form=affine policy_index=latent train_actor=false
acting=gpi`), and every agent other than `psmflow` and `fql` now lives under `archive/`.
See the **2026-09-04 (evening)** entry below for what was built, the test evidence and the
design note (`docs/design/2026-09-04-affine-psi.md`). The only `agents/` change here
is the eval-time `gpi_select` ablation switch (`_gpi_select_ablation`), added for
the ablation batch below and left as a documented switch, not a new default.

---

## Every 500k number, 500 episodes, both envs, both arms

| arm | env | sd0 | sd1 | mean ± 95% CI (t, n=2) |
|---|---|---|---|---|
| affine **strict** (`train_actor=false acting=gpi`) | cube | 0.086 | 0.548 | 0.317 ± 2.935 |
| affine **actor** (`train_actor=true acting=actor`) | cube | 0.162 | 0.130 | 0.146 ± 0.203 |
| affine **strict** | antmaze | 0.002 | 0.096 | 0.049 ± 0.597 |
| affine **actor** | antmaze | 0.224 | 0.206 | 0.215 ± 0.114 |

Comparators (all 500k): BC control **cube 0.072**, **antmaze 0.072**; free-psi Arm B cube
0.083 ± 0.191 (3 seeds); the old latent-actor agent cube 0.230 ± 0.051, antmaze
0.213 ± 0.140; FB cube 0.721.

**At 500k, on the terminal checkpoint alone, the strict arm beats nothing.** Cube strict
0.317 is inside the old actor agent's interval and its two seeds (0.086, 0.548) do not
overlap each other; antmaze strict 0.049 is *below* the BC control; the affine actor arm
matches the old actor agent on antmaze (0.215 vs 0.213) and is worse on cube (0.146 vs
0.230). Every n=2 t-interval here is worthless and is printed only for form.

---

## Cube, affine strict: the full checkpoint sweep (500 episodes each)

| epoch | sd0 | sd1 |
|---|---|---|
| 100k | 0.078 [0.058, 0.105] | 0.330 [0.290, 0.372] |
| 250k | **0.532** [0.488, 0.575] | **0.620** [0.577, 0.661] |
| 300k | 0.286 [0.248, 0.327] | — |
| 350k | **0.704** [0.663, 0.742] | — |
| 400k | 0.282 [0.244, 0.323] | 0.346 [0.306, 0.389] |
| 450k | 0.272 [0.235, 0.313] | 0.568 [0.524, 0.611] |
| 500k | 0.086 [0.065, 0.114] | 0.548 [0.504, 0.591] |

Ten late (≥250k) checkpoint×seed measurements, 500 episodes each:
**mean 0.424, sd 0.196, 95% CI ± 0.141, min 0.086, max 0.704.**

sd0 alone goes 0.532 → 0.286 → **0.704** → 0.282 → 0.272 → 0.086 on consecutive 50k
checkpoints of one run. Each of those is a 500-episode measurement whose Wilson interval is
±0.04 wide, so **the swing is real training-time non-stationarity, not evaluation noise** —
adjacent intervals are disjoint by a wide margin. sd1 swings less (0.620 … 0.346 … 0.568 …
0.548) but is not monotone either.

---

## Antmaze, affine strict: the same non-stationarity, but around the BC control

Four new 500-episode evals (the two highest un-evaluated in-loop checkpoints
per seed; 50k and 500k were already done). SLURM 2491639-42, ~93 min each at ~11 s/episode.

| epoch | sd0 | sd1 |
|---|---|---|
| 50k | 0.112 [0.087, 0.143] | 0.000 [0.000, 0.008] |
| 250k | — | **0.178** [0.147, 0.214] |
| 350k | 0.082 [0.061, 0.109] | — |
| 400k | 0.056 [0.039, 0.080] | 0.134 [0.107, 0.167] |
| 500k | **0.002** [0.000, 0.011] | 0.096 [0.073, 0.125] |

Late-checkpoint mean (≥250k, 6 measurements): **0.091 ± 0.064**, against BC 0.072, the
affine *actor* arm's 0.215 and the old actor agent's 0.213.

**Not flat-low, and not the cube pattern either.** The two seeds move in *opposite*
directions: sd0 decays monotonically to zero (0.112 → 0.082 → 0.056 → 0.002) while sd1
rises off the floor and then decays (0.000 → 0.178 → 0.134 → 0.096). Every antmaze
checkpoint sits within ±0.11 of the BC control, so on this env the arm is **not a result at
all** — its late mean's interval contains BC. The in-loop 50-episode eval is useless here
(0.00 at 8 of 11 points on sd0 while 500 episodes read 0.002-0.112).

**The encoder does not collapse on antmaze.** `w_enc_spread` *rises* monotonically to ≈1.24
and stays there for 400k steps — the opposite of cube's 1.30 → 0.25 decay — and the arm is
still bad. Together with cube — where the spread collapses monotonically while
success swings up and down four times — that settles it: `w_enc_spread` moves monotonically
in whichever direction the env dictates and success does not follow it in either env.

`docs/figures/2026-09-05-affine-antmaze-sweep.{png,json}`.

---

## What the oscillation is not: no logged scalar predicts it

Every quantity this arm logs was paired with the 12 cube 500-episode measurements
(checkpoint × seed) and correlated against success. **Nothing predicts it.**

| logged series | Pearson vs 500ep success | Spearman |
|---|---|---|
| `w_enc_spread` (encoder pairwise distance) | −0.386 | −0.231 |
| `psi_q_index_spread_rel` | +0.017 | +0.266 |
| `psi_q_spread_rel` | +0.032 | +0.203 |
| `psi_q_range_rel` | −0.003 | +0.210 |
| `psi_q_spread` (absolute) | −0.072 | +0.084 |
| `psm_loss` (measure TD) | +0.106 | +0.322 |
| `orth_loss` | −0.398 | −0.385 |
| *in-loop 50-episode eval at the same step* | **+0.916** | **+0.916** |

Read that last row carefully. The in-loop 50-episode eval — the thing this project's
reporting rules forbid quoting — **tracks the 500-episode number at r = 0.92 on this arm**
(12 paired points, Pearson = Spearman = 0.916). That is not a licence to quote it; it is
evidence about *where* the variance lives. The 500-episode CI is ±0.04 and the
checkpoint-to-checkpoint swing is ±0.3, so the noise is in the **policy at that
checkpoint**, not in the measurement, and 50 episodes already see most of it.

Ruled out by the table: encoder collapse (`w_enc_spread` decays smoothly and monotonically
on cube and *rises* monotonically on antmaze, while success does neither); the
orthonormality constraint (`orth_loss` pinned at ≈−64 from 10k on, flat to three digits in
every run); a diverging measure TD (`psm_loss` grows ~10x and spikes constantly, unaligned
with the dips); and any index/readout scale artefact (all four `psi_q_*` statistics,
|Spearman| ≤ 0.27). **A monotone training trace with a non-monotone eval trace means the
failure is in the acting rule or in what psi has learned, not in a scalar the loss sees.**

---

## Two probes settled it

### Selection diagnostic — `tools/diag_gpi_selection.py`, `docs/design/2026-09-05-gpi-selection-diag.md`

State-matched probe: the same 256 dataset states and the same 64×64 candidate roster scored
by four checkpoints of `affine_strict_cube/sd000` (250k / 350k / 400k / 500k).

- **The two pre-registered failure modes are not it.** Selected-`u` norm 2.83-2.95 (roster
  2.13), top1−top2 gap 0.31-0.43 sd, selected-`u` clip fraction ~1.2% vs the roster's 0.28%
  — **identical at the good and bad checkpoints**. Edge-seeking and flatness are permanent
  properties of the rule, present at 0.704 too.
- **The ranking is re-randomised.** The per-`u` GPI score the argmax consumes correlates
  only **rho 0.21-0.36 between any pair of checkpoints**, including 250k vs 350k (0.31), the
  two that both work. Top-5 overlap 0.20-0.28; argmax agreement 11-19%.
- **Target-critic selection is not a stabiliser.** Online vs target rank the same roster at
  **rho 0.92-0.96** within a checkpoint (`tau=0.01` has long converged), and the target
  drifts across checkpoints identically (0.23-0.40). Dropping pessimism changes the ranking
  even less (0.93-0.95).
- **The index/action asymmetry.** psi's spread over `u'` is **~30x** its spread over `u`
  (`index_matters` ~5-13k against a per-`u` spread of a few hundred): the inner argmax over
  `u`, the one that picks the action, runs on the weakest part of the signal.
- **MC ground truth** (fixed-`u` rollouts near rewarding transitions): Q-rank vs return-rank
  Spearman +0.10 / +0.15 / +0.13 / +0.10, while the checkpoint-free baseline `−||u||` scores
  **+0.28**. MC regret of the Q-argmax 16.8 / 22.8 / 23.0 / 32.4 against 19.1 for a random
  pick — three of four checkpoints select *worse than chance* on this proxy. Averaging the
  four rankings lifts Spearman 0.121 → 0.166 and regret 23.75 → 19.2: checkpoint averaging
  removes the harm, it does not make the selector good.

---

### Eval-time GPI ablations — `docs/design/2026-09-05-gpi-ablations.md`

Eleven 500-episode evals, one run (`affine_strict_cube/sd000`), acting rule swapped at eval
time with nothing retrained. `eval500_gpiabl_*.json`, now registered in `make_tables.py`.

| arm | rule | @350k | @500k |
|---|---|---|---|
| G | `argmax` (shipped) | **0.704** (exact reproduction, 352/500) | 0.086 |
| E | `mean` — ensemble mean, no pessimism | 0.734 (p=0.29, tie) | — |
| C | `small_ball` | 0.670 (p=0.25) | **0.046** (p=0.011, *worse*) |
| D | `soft_topm` m=8 | 0.612 (p=0.002, *worse*) | 0.062 (p=0.15) |
| A1 | `max_norm`, critic-free | 0.078 | — |
| A2 | `top_quartile_random`, critic-free | 0.078 | — |
| B | one prior draw, K=1 | 0.090 | — |
| F1 | `fixed_index` seed 0 | **0.000** | — |
| F2 | `fixed_index` seed 1 | 0.180 | — |

Three results, in order of importance:

1. **The critic carries the whole gap, and the norm bias carries none of it.** Both
   critic-free controls that reproduce the deployed rule's *selection statistics* land at
   **0.078**, indistinguishable from BC (p=0.72) — `top_quartile_random` even overshoots the
   deployed |u|=2.8 at 3.01. One prior draw gives 0.090. So the large-norm story is a true
   description of the argmax and **dead as a performance explanation**.
2. **The per-step max over fresh policy indices is the mechanism.** Pinning `u'` to one
   prior draw for a whole eval — same critic, same 64 `u` draws, same argmax over `u` —
   gives **0.000** and **0.180** depending only on which `u'` was drawn. Seed 0 is
   *significantly below BC* (0/500, p=1e-9): with the index pinned, ranking `u` by
   `psi(s,u,u')ᵀw` selects actively harmful actions. The `K×K` pair scan is load-bearing.
3. **No eval-time intervention recovers 500k.** Five rules leave it at or below BC. The
   0.704 → 0.086 collapse is in the learned `psi`, not in the rule that reads it — so the
   fix is training-side or aggregation-side, not acting-side.

---

## Reporting rule (binding, effective now)

> **No single checkpoint of the affine strict arm may be quoted.** Not the best, not the
> last, not 250k, not 350k. Any number reported for this arm is the **mean over late
> checkpoints (≥250k) and seeds**, with the spread stated.

| env | late-ckpt mean (≥250k) | n (ckpt×seed) | range | BC control |
|---|---|---|---|---|
| cube | **0.424 ± 0.141** | 10 | 0.086 - 0.704 | 0.072 |
| antmaze | **0.091 ± 0.064** | 6 | 0.002 - 0.178 | 0.072 |

On cube the arm is still a real result — the only PSMFlow configuration above 0.4, against
Arm B's 0.083, the old actor agent's 0.230 and FB's 0.721. On antmaze its interval contains
BC and it is not a result at all. The 09-04 headline "0.532 / 0.620 @250k" is **superseded**
and must not be re-quoted on its own; likewise 0.704 and 0.086 are the same agent, the same
rule and the same norm bias.

`tools/make_tables.py` now enforces the shape: one row per (arm, env, **checkpoint**), plus
one explicitly labelled `late-ckpt mean` row per env, plus the GPI ablation block. The four
rows that previously globbed `eval500_affine_strict_*_sd?.json` under a "500k" label were in
fact reading the **100k** JSONs; that is fixed. Both BC controls now resolve (they live in
`$PSM_DATA/evals/` on this cluster, not `logs/`), and `_load` de-duplicates an eval reached
through two names so a 1-seed control cannot print as 2.

---

## Open items

**1. `index_agg=expectile` — the best-motivated next experiment.** Both probes point at the
same object: the deployed 0.704 is an **optimistic per-step max over 64 samples of a learned
function** on the axis where psi has ~30x more spread, which is exactly the structure E4a
indicted. `index_agg=expectile` (already in the tree, `configs/agent/psmflow.yaml:86-88`)
distils an upper expectile of `psi(s,u',u)ᵀw` over `u' ~ p0` into a z-free scalar head and
takes no argmax over samples at acting time. Run it **on the affine agent**
(`psi_form=affine policy_index=latent train_actor=false acting=gpi index_agg=expectile`),
The ablations note's §7.4 names the cheap companions: a **larger `index_panel`** (shipped
16; the yaml records `index_panel: 1` as the InFOM-like setting) and a **`gpi_num_u` sweep
on the index axis alone**. Neither note proposes specific sweep values — pick them, print
the table before launch. Judge the result on the **late-checkpoint mean and its spread**,
never a peak.

**2. Aggregation, not sharpening.** Checkpoint averaging is the only intervention measured
to help (MC Spearman 0.121 → 0.166, regret 23.75 → 19.2). A param EMA of psi is the way to
turn "report the mean over checkpoints" into an *agent*. Within-checkpoint softening
(`soft_topm`) is a measured loss and is not the same medicine.

**3. Closed — do not re-run.** Target-critic selection (rho 0.92-0.96 with the deployed
rule, a no-op), ensemble mean instead of min (0.734 vs 0.704, p=0.29, a tie),
`small_ball` (holds 350k, hurts 500k) and `max_norm`/`top_quartile` (BC-level). `lr_sf`
lowering is *not* yet tested and remains open, but note the drift is in the ranking, not in
the loss scale.

**4. DSRL-NA actor.** The actor arms are stable and capped (cube 0.108 / 0.113 / 0.146 at
100k / 250k / 500k — a total span of 0.04 against the strict arm's 0.62); the strict arm is
unstable and high. A noise-aliased/DSRL-NA-style latent actor is the standing proposal for
getting the strict arm's ceiling with the actor arm's smoothness. Not started.

**5. Antmaze needs a reason, not another seed.** The arm sits at BC on antmaze at every
checkpoint while `w_enc_spread` behaves in the opposite direction to cube. Nothing here
explains that; the affine *actor* arm reaches 0.215 there, so the substrate is fine
and the strict/GPI path is what fails.

---

<!-- _class: lead -->

## 2026-09-04 (evening) — the affine measure head psi = A(s,u)ᵀ w(u′) + beta(s,u):
## cube strict 0.532 / 0.620 at 250k, against Arm B's 0.083 at 500k

Design + pre-registration: `docs/design/2026-09-04-affine-psi.md`, written **before** launch,
expectations included. Rem. `tradeoff` records that the shipped agent adopts Prop.
`bilinear`'s form but **not** the affineness of psi in the policy coordinate — `w^{u′}` is
absorbed into a free network. The affine form is implemented explicitly below. The paper
asserts `w^{u′}` exists (Assumption `affine`) and gives **no formula** for it, so the encoder
`u′ -> w(u′)` is **our design choice**, stated as such in the design note.

### What was built

| what | where |
|---|---|
| `AffinePsiMap` — `psi(s,u,u′) = A(s,u)ᵀ w(u′) + beta(s,u)`, drop-in for `PsiMap` | `utils/psm_networks.py:439-500` |
| `_AffinePsiTower` — `(s,u)` trunk, two heads (`A`: `h -> z_dim·d_w`, `beta`: `h -> z_dim`), **no `u′` input at all** | `utils/psm_networks.py:403-436` |
| `encode_index` / `sa_terms` — the halves exposed so affineness and collapse are measurable | `utils/psm_networks.py:481-495` |
| `psi_form: free \| affine` + `affine:{w_dim,encoder_hidden,encoder_layers,norm_w}` | `configs/agent/psmflow.yaml:52-65`, `agents/psmflow.py:899-906` |
| head selection + the `policy_index=latent` guard | `agents/psmflow.py:699-726` |
| `index_spread` — `psi_q_spread_rel`, `psi_q_range_rel`, `psi_q_index_spread_rel`, `w_enc_spread` | `agents/psmflow.py:298-349`, called at `:512-516` |
| `RESTORE_EPOCH` (default 500000) so arms can be evaluated at a common earlier checkpoint | `scripts/eval500.sh:29,83,90` |
| eval-JSON registration, 4 rows (not run) | `tools/make_tables.py:92-105` |

**Design choices.** ONE encoder shared across the `num_parallel` ensemble — `w^{u′}` is a
property of the policy, not of a critic member, so the ensemble disagrees only through
`(A, beta)`, which is what `pessimism_penalty` measures. `w(u′)` on the **unit sphere** (not
`psm_norm`'s `sqrt(d)`): `(A,w)` is identified only up to `(cA, w/c)`; unit norm pins it,
keeps psi's scale independent of `d_w`, and makes collapse read as small *pairwise distance*
at fixed radius. `A` and `beta` never see `u′` — Assumption `affine` enforced by
construction, not by a penalty. `d_w = 128 = z_dim`.

---

### Tests — full suite clean

`tests/test_psmflow_affine.py`, **11 tests**: default path **byte-identical** to
`psi_form=free` (post-update params compared leaf by leaf, plus identical
`psm_loss`/`orth_loss`/`actor_loss`, and the shipped info dict gains no key); psi **exactly
affine** in `w(u′)` — `psi(u′₁) − psi(u′₂) == Aᵀ(w(u′₁) − w(u′₂))` and `psi == Aᵀw + beta`,
both to 1e-4; `A`, `beta` index-free; `w(u′)` on the unit sphere and not constant; finite
step in **both** arms; `psi_form=affine` + `policy_index=task_vector` refused; unknown
`psi_form` refused; `sample_actions` in both acting modes; affine checkpoint structurally
distinct from the free one.

Whole suite, module-per-process: **248 passed, 3 skipped, 0 failed** (39 modules).
`ruff` on the touched files adds one finding (`C408` on the new `affine=dict(...)`), the
same kind as the 8 sibling config entries; it removes one (`I001` in `psm_networks.py`).

**Restore path checked on GPU, not just in unit tests.** `tools/eval_checkpoint.py` was run
against an affine checkpoint with **none** of `psi_form / policy_index / train_actor /
acting` on the CLI: `agent_config_source.inherited` came back
`['acting','policy_index','psi_form','train_actor']` from the run's own `flags.json`
(`logs/affine_restorecheck.json`). The reported evals below pass those four explicitly, so
their `agent_config_source` shows `flags_json` set and the four keys under `cli_overrides`.

---

### Runs — 8 jobs, both envs, seeds 0/1, launched at 500k

`agent=psmflow`, `psi_form=affine`, `policy_index=latent`, `use_point_preimage=true`,
`u_clip=3.0`, `offline_steps=500000`, `eval_interval=50000`, `eval_episodes=50` (in-loop,
never quoted), `save_interval=50000`; `affine_strict_*` = `train_actor=false acting=gpi`,
`affine_actor_*` = `train_actor=true acting=actor`. Full HP table printed pre-launch;
200-step GPU smokes run for both arms; all 8 `flags.json` re-read after launch and confirmed.

**The runs did NOT reach 500k inside the compute window.** The affine head's `A` output
(`1024 -> 16384`) makes a step 24-36 ms under 8-way contention against the free head's ~7 ms,
and the strict arm's antmaze GPI eval costs **10.1 s/episode** (84 min per 500-episode eval).
Everything below is therefore quoted at the latest checkpoint each arm reached and could be
evaluated at — **100k for all four arms, plus 250k for cube** — never at 500k. Every number
is a fresh 500-episode `tools/eval_checkpoint.py` run; no in-loop 50-episode value is quoted
anywhere in this entry. The 8 training jobs are still running toward 500k.

---

### 500-episode results (per-seed Wilson; comparators are 500k numbers)

**Headline — the affine strict arm on cube, at 250k, is the best zero-shot PSMFlow number
on record and both seeds agree.**

| arm | epoch | sd0 | sd1 | mean ± 95% CI (t, n=2) | pooled (1000 ep) |
|---|---|---|---|---|---|
| **affine strict, cube** | **250k** | **0.532** [0.488, 0.575] | **0.620** [0.577, 0.661] | **0.576 ± 0.559** | 0.576 [0.545, 0.606] |
| affine strict, cube | 100k | 0.078 [0.058, 0.105] | 0.330 [0.290, 0.372] | 0.204 ± 1.601 | 0.204 [0.180, 0.230] |
| affine actor, cube | **250k** | 0.130 [0.103, 0.162] | 0.096 [0.073, 0.125] | 0.113 ± 0.216 | 0.113 [0.094, 0.135] |
| affine actor, cube | 100k | 0.110 [0.085, 0.140] | 0.106 [0.082, 0.136] | 0.108 ± 0.025 | 0.108 [0.090, 0.129] |
| affine actor, antmaze | 100k | 0.176 [0.145, 0.212] | 0.216 [0.182, 0.254] | 0.196 ± 0.254 | 0.196 [0.173, 0.222] |
| affine actor, antmaze | 50k | 0.218 [0.184, 0.256] | 0.224 [0.190, 0.263] | 0.221 ± 0.038 | 0.221 [0.196, 0.248] |
| affine strict, antmaze | 50k | 0.112 [0.087, 0.143] | 0.000 [0.000, 0.008] | 0.056 ± 0.712 | 0.056 [0.043, 0.072] |
| affine strict, antmaze | 100k | *eval running (>75 min each)* | | | |

Comparators, all at **500k** and all with more training than anything above:

| comparator (cube) | | comparator (antmaze) | |
|---|---|---|---|
| Arm B — free psi, strict, `acting=gpi` | 0.083 ± 0.191 (3) | PSMFlow (point, actor) | 0.213 ± 0.140 |
| gpi-matched control (point arm, gpi) | 0.054 ± 0.032 (5) | BC per-step prior | 0.072 |
| PSMFlow actor (point) | 0.230 ± 0.051 (5) | | |
| BC per-step prior | 0.072 | | |
| FB (zero-shot, raw actions) | 0.721 ± 0.020 | | |

**Read the strict-cube row against Arm B.** Arm B is the *same algorithm with a free psi*:
`policy_index=latent`, `train_actor=false`, `acting=gpi`. Its three recorded seeds are
0.006 / 0.160 / 0.084 and its matched gpi control is 0.054. The affine head at **half the
training** returns 0.532 / 0.620 — every one of the four Wilson intervals involved is
disjoint from every Arm B seed's, and the pooled 0.576 [0.545, 0.606] sits **7x** its
comparator. It also beats the actor arm (0.230) and is the only PSMFlow configuration that
has ever come within reach of FB's 0.721 on cube. The n=2 t-interval (±0.559) is worthless
and is quoted only for form; the evidence here is the **agreement of two independent seeds
at 500 episodes each**, which Arm B never had.

**And it climbs.** The same two runs read 0.078 / 0.330 at 100k and 0.532 / 0.620 at 250k.
The 09-03 entry's warning — the gpi arm "climbs late… do not judge this arm before ~400k" —
holds here in the affine arm's favour: 250k is still early, and the runs continue to 500k.

**The actor arms do not show the effect — and cube actor was measured at the SAME 250k
checkpoint, so this is not a training-budget artifact.** cube actor 0.108 at 100k and 0.113
at 250k (flat), antmaze actor 0.196 at 100k / 0.221 at 50k — all level with their free-psi
comparators, while cube strict went 0.204 → 0.576 over the same interval. On this evidence the
affine head helps the arm that *selects by ranking psi* (gpi) and does nothing for the arm
that delegates selection to an amortized actor. That is exactly where a better-conditioned
psi should show up, and it is the first time in this project's record that the strict,
paper-faithful arm has beaten the actor arm.

### Diagnostics — the mechanism the head was built to change

At the **100k** checkpoint (`w_enc_spread`, `psi_q_index_spread_rel`, `psi_q_spread_rel`):

| run | sd | `w_enc_spread` | `psi_q_index_spread_rel` | `psi_q_spread_rel` |
|---|---|---|---|---|
| affine strict cube | 0 / 1 | 1.200 / 1.307 | **0.627 / 0.559** | 0.033 / 0.043 |
| affine actor cube | 0 / 1 | 1.232 / 1.304 | **0.612 / 0.586** | 0.035 / 0.039 |
| affine strict antmaze | 0 / 1 | 1.152 / 1.169 | **0.599 / 0.287** | 0.061 / 0.020 |
| affine actor antmaze | 0 / 1 | 1.153 / 1.170 | **0.574 / 0.292** | 0.056 / 0.019 |

At **250k**, cube strict reads `w_enc_spread` 0.453 / 0.414, `psi_q_index_spread_rel`
**0.818 / 0.877**, `psi_q_spread_rel` 0.045 / 0.034.

`psi_q_index_spread_rel` is the relative spread of `Q` across prior policy indices `u′` —
the quantity GPI's outer argmax consumes. It reads **29-88%** against the **~1%** that every
free-psi measurement in this project has reported (D1 0.9%, D3 1.1%, Arm B 0.9%, E4b 0.86%,
DSRL-SAC 1.1-1.5%), and it **rises** with training (0.56-0.63 at 100k → 0.82-0.88 at 250k)
alongside the success climb. Spread across prior *action* latents (`psi_q_spread_rel`, the
inner argmax's quantity) is 2-6% — above the 1% band, but by a factor of a few, not fifty.

`w_enc_spread` starts at 0.72-1.11 (5k), peaks at 1.15-1.31 (100k) against the ~1.414 a
random unit-sphere sample gives, and on **cube** falls to 0.41-0.45 by 250k while
`psi_q_index_spread_rel` keeps rising — the encoder is *concentrating* its directions and
`A` is growing to compensate, not collapsing (a collapsed encoder reads ~0 and would take
the index spread with it). Antmaze holds ~1.19 at 160k. **Watch this number**: if it reaches
~0 while the index spread falls with it, the affine structure has degenerated back to a
`u′`-free psi, and that is the diagnostic that would say so.

---

### Verdict against the pre-registration

Pre-registered (design note §6): **success** = `psi_q_index_spread_rel` >> 1% **and**
strict > Arm B (0.083); **failure** = the same ~1% band and BC-level strict.

- **Success fires on both halves, for cube strict.** Index spread 29-88% against ~1%, and
  0.532 / 0.620 against Arm B's 0.083 ± 0.191 — at half the training, with two agreeing
  seeds. The stated failure branch ("the affine form is not the missing piece either") is
  **refuted**.
- The hypothesis was that the affine bottleneck acts as a *regularizer on the policy slot*.
  The diagnostic says the mechanism is real and specific: psi stops being flat across `u′`,
  which is precisely the faculty Arm B lacked and E1/E4a indicted (ranking).
- **Scope, honestly.** Two seeds, at 250k not 500k, on **one environment and one arm**. The
  actor arms show nothing (cube 0.108 → 0.113, antmaze 0.196, level with their comparators).
  And **antmaze strict at 50k is 0.112 / 0.000, i.e. below its BC control** — at a fifth of
  cube-strict's 250k budget, on the env whose latent geometry the 09-03 S1 probe already
  found less navigable (ratio 0.79 vs 0.91). Its 100k evals were still running when the
  compute window closed. Nothing here says the affine head fixes antmaze, and nothing says the
  cube result survives to 500k.
- Also unchanged: this is still below FB's 0.721 and far below FQL's 0.949.

**Next, in order.**
1. Let the 8 jobs reach 500k and run `scripts/eval500.sh` at `RESTORE_EPOCH=500000` for all
   8. The four `eval500_affine_{strict,actor}_<env>_sd<k>.json` names are already registered
   in `tools/make_tables.py` (not yet run).
2. Collect the pending antmaze-strict evals (50k and 100k, submitted, 84 min each).
3. A **third seed** on cube strict — 0.576 pooled over two seeds is strong, but this project
   has been burned by two-seed cube numbers before (09-03, seeds spanning 0.090-0.374).
4. The obvious ablation now: `psi_form=affine` with `acting=actor` is already run and flat,
   so the next cut is `d_w` (128 → 32/512) to test whether the *bottleneck width* is the
   active ingredient, and `norm_w=false` to test whether the sphere constraint is.

**Housekeeping.** The working tree carried a **pre-existing, uncommitted deletion** of 20
comment lines in `agents/psmflow.py` (the `q_dist` / action-branch / FB-graft field
docstrings added in `6e166d7`) before this work touched the file. It is not part of it and shows up in `git diff` as a deletion — restore it before committing. **DONE** in
the reorg below: restored verbatim, since all three switches they document still exist.

---

<!-- _class: lead -->

### 2026-09-04 (late) — the affine agent is now THE agent; everything else moved to `archive/`

Consequence of the number above, not a new experiment. **No results in this section.**

**The default flipped.** `agent=psmflow` with no flags is now the paper-strict affine
LatentFlowPSM. The four arm flags and one dataset flag that the 0.532 / 0.620 runs passed on
the command line are now the config:

| key | was | is | source |
|---|---|---|---|
| `psi_form` | `free` | **`affine`** | `affine_strict_cube_sd{0,1}` flags.json |
| `policy_index` | `task_vector` | **`latent`** | ditto |
| `train_actor` | `true` | **`false`** | ditto |
| `acting` | `actor` | **`gpi`** | ditto |
| `use_point_preimage` | `false` | **`true`** | ditto; `main.py` refuses a corrected-target npz with `false`, so the old default was unusable anyway |

Everything else is byte-identical to those runs' `flags.json` (`z_dim` 128, `num_parallel` 2,
`discount` 0.98, `tau` 0.01, `ortho_coef` 1000, both pessimism penalties 0.5, `lr_phi` 1e-5,
`lr_sf` 1e-4, `sf` 1024x1, `phi` 256x2, `affine` `{w_dim:128, encoder_hidden:256,
encoder_layers:2, norm_w:true}`, `index_agg` max, `u_clip` 3.0, `gpi_num_u` 64).
`configs/agent/psmflow.yaml` and `get_config()` both moved, and both now name the free psi
and the task-vector index as *ablations*.

**Back-compat for old checkpoints.** Flipping a default silently rewrites what
`tools/eval_checkpoint.py` builds for a run whose `flags.json` predates the key — a pre-09-04
psmflow checkpoint has no `psi_form` at all, so it would have been rebuilt as an affine head
(and, with its own `policy_index=task_vector` inherited on top, would have tripped
`create()`'s guard). `LEGACY_AGENT_DEFAULTS` in that file back-fills the **pre-2026-09-04**
value for any of the five keys the run does not carry, records it in
`agent_config_source.legacy_defaults`, and a typed CLI override still wins. Three tests pin it.

**`archive/`.** Every agent except `psmflow` and `fql` moved there with `git mv`, with its
yaml, its tests and the tools/scripts/plans built on it: `psm`, `affine_psm`,
`latent_affine_psm`, `latentrl`, `fb`, `ifql`, `iql`, `rebrac`, `sac`. Also archived: the
diagnostics for hypotheses the compendium records as settled negative (fixed-`u` family,
interface ceiling, flow Jacobian, mixture-preimage width, D1–D3 free-head critic forensics),
the dropped ICLR F1–F4 figure scripts, and the superseded plan queue. `archive/README.md` is
a one-line-each index; `pyproject.toml` now excludes `archive` from ruff and pins
`testpaths = ["tests"]`. **`fb` was archived rather than kept**: `tools/make_tables.py` is
pure JSON I/O, so its FB rows (cube 0.721) keep printing from the eval JSONs already on disk;
only *re-running* FB needs the agent back.

**Two edits the move forced.** (1) `agents/psmflow.py` imported seven pure helpers from
`agents/psm.py`, so they are now `utils/psm_common.py` and `archive/agents/psm.py`
re-exports them. (2) The latent actor's network and loss are **kept, not archived** — it is
the DSRL-style comparator and the substrate for the planned DSRL-NA distillation of the GPI
argmax; it is simply off by default, and `tests/test_psmflow_actor.py` now builds it
explicitly through a `_legacy_agent()` helper.

**Tests rewritten, not deleted.** Six modules asserted the OLD default was the default.
Each now pins the new default explicitly and keeps the old arm as a named ablation through
`_legacy_agent()` (`test_psmflow_agent`, `_actor`, `_policy_index`, `_backup_explore`,
`_affine`, `_groundtruth`). `test_psmflow_groundtruth.py` gained a **new** chain-MDP pin for
the default arm: with no actor to interrogate, its ground truth is `gpi_select` itself —
the (u_i, u′_j) search must return a goal-ward latent at every non-goal state, which it does.

### Verification (all run after the move)

| check | result |
|---|---|
| `python -c "from agents import agents"` | `['fql', 'psmflow']` |
| all nine archived agents still import from `archive.agents.*` | 9/9 OK |
| full suite, module-per-process (26 modules) | **176 passed, 3 skipped, 0 failed** |
| `ruff check .` | 145 findings, from 357 at HEAD under the same config (archive excluded). The only *added* finding in the live tree is the `affine=dict(...)` `C408` already noted above |
| 200-step GPU smoke, `main.py agent=psmflow` with **no** agent flags (sbatch 2491622, kisski-inference) | COMPLETED in 1:53. Its `flags.json` agent block diffs against `affine_strict_cube_sd0`'s: **NONE** — the defaults reproduce the 0.532/0.620 configuration exactly |
| `tools/eval_checkpoint.py` on `affine_strict_cube_sd0@250000`, 5 episodes, **no** arm flags (sbatch 2491624) | restored and ran; provenance printed *"run config already matches this checkout's defaults"*, CLI overrides only the three path flags. 1/5 solved (n=5) |

`squeue` before: 7 training jobs running, 0 pending. After: 6 running (`affine_strict_cube_sd0`
reached 500k on its own in the meantime), 0 pending — **no job was ever pending while the
tree was mid-move**, so nothing could import a half-moved tree.

---

<!-- _class: lead -->

## 2026-09-04 — S1: the latent value landscape IS navigable; S2 built, and its first run was invalid

Executes `docs/plans/2026-09-03-latent-coherence-and-infom.md` (added here; it was written
but never committed). S1 is the plan's gate and it **passed navigable**, which per the
plan's own decision rule authorises S2. S2 is implemented and re-running; its first launch
measured a bug and is withdrawn — see below.

### S1 — `tools/diag_latent_smoothness.py`, four panels, all navigable

Forward passes only: the frozen Stage-A flow for `decode` and a frozen FQL expert for the
oracle score. No Stage-C checkpoint, no `psi`, so the answer holds for every Stage-C arm at
once. 64 on-path states x 512 unclipped `N(0,I)` latents, `v = -||a - a*||`.

| env | decoder | `knn_r2_a` (control) | `knn_r2_u` | basin | disp_u | corr da/du | best-of-512 |
|---|---|---|---|---|---|---|---|
| cube | onestep | 0.850 | **0.769** | 0.848 | 0.627 | 0.850 | 0.049 |
| cube | ODE-100 | 0.796 | **0.726** | 0.880 | 0.579 | 0.880 | 0.060 |
| antmaze | onestep | 0.746 | **0.587** | 0.714 | 0.728 | 0.774 | 0.256 |
| antmaze | ODE-100 | 0.744 | **0.595** | 0.727 | 0.741 | 0.793 | 0.313 |

Pre-registered: navigable needs `knn_r2_u` > 0.5, basin > 0.4, dispersion < 0.8, corr > 0.7;
scrambled is `knn_r2_u` < 0.05. **All four statistics clear on all four panels**, at 12-15x
the scrambled threshold.

**Reading. The geometry is benign.** The ~1% flatness that D1 (0.9%), D3 (1.1%), Arm B
(0.9%), E4b (0.86%) and DSRL-SAC (1.1-1.5%) all report is NOT a property of the latent
space; it is a property of the learned critics. Live hypothesis #1 ("the landscape is
rough") dies by measurement rather than by another 3x500k run.

**Validity.** The harness reproduces E1 cold: mean best-of-512 oracle distance **0.060**
(ODE-100) against E1's recorded **0.062**, from a freshly trained expert on another machine.

**Caveat, stated because it missed its bar.** The calibration control `knn_r2_a` was
pre-registered > 0.9 and came in 0.74-0.85 on every panel. That is a finite-sample ceiling
(512 points in d_a 5 / 8), not a broken estimator -- the failure mode it exists to catch
(reading ~0 where it must read ~1) did not occur. A depressed ceiling drags every R^2 down,
so it makes the navigable verdict **conservative**. Ratio form: `knn_r2_u / knn_r2_a` =
**0.91** cube, **0.79** antmaze.

**Third independent sighting of the cube/antmaze split.** Antmaze is navigable but less so
(ratio 0.79 vs 0.91), and its best-of-512 sits at 0.313 against mean ||a|| 1.99 (15.7%)
where cube sits at 0.060 against 0.87 (6.9%) -- on a probe with no inversion, no preimage
and no alpha in it.

Reports: `$PSM_DATA/logs/d5_latent_smoothness_{cube,antmaze}_{onestep,ode}.json` (+ `_raw.npz`
with the `(u, a, v)` arrays so the statistics recompute without re-rolling).

### The oracle did not exist on this cluster, and had to be built

S1 needs a frozen FQL expert to roll the states AND to score. HF ships only the three
Stage-A BC flows; every local run dir is Stage-C. Trained here, 500k, 28 min each:

| env | recipe | 500-ep success | Wilson 95% |
|---|---|---|---|
| cube | `agent.alpha=300` (the recorded 0.949 recipe) | **0.966** (483/500) | [0.946, 0.979] |
| antmaze-medium | `agent.q_agg=min agent.alpha=10` | **0.980** (490/500) | [0.964, 0.989] |

Cube overlaps the recorded FQL 0.949 [0.936, 0.960], which is E1's readability condition.
The antmaze recipe is **adapted** from the FQL reference's antmaze-*large* line -- the
reference gives no antmaze-medium value. Incidentally this is the repo's **first per-task
topline on antmaze**: PSMFlow's 0.213-0.247 there is ~4x below its own ceiling, the same
shape as cube's 0.230 against 0.949.

### S2 — implicit GPI by expectile distillation (`index_agg=expectile`)

New in `agents/psmflow.py`: a scalar head `q_dist` fitted by upper-expectile regression onto
`psi(s, u', u)^T w` over `index_panel` prior draws `u'`, replacing the argmax at two call
sites (`gpi_select` drops the KxK pair scan; `flow_actor_loss` climbs the distilled head).
`measure_loss` untouched in v1 so any regression is attributable. Keys `index_agg`,
`expectile_mu`, `index_panel`, `q_dist` in `get_config()` and `configs/agent/psmflow.yaml`;
`create()` asserts `expectile` requires `policy_index=latent`. Static switch, keys folded
out of `rng` (108/109/110), so `index_agg=max` keeps a byte-identical stream -- pinned by
29 passed / 2 skipped across the three psmflow test modules including the golden fingerprints.

**Read the InFOM source before trusting the spec.** `agents/infom.py` at
github.com/chongyi-zheng/infom: the intention encoder conditions on
`(next_observations, next_actions)` -- S3's `p_e(z|s',a')` is **confirmed as written**. But
S2's stated premise is **wrong**: InFOM has **no z-conditioned -> z-free distillation**. Its
latents are prior draws, its target is a Monte-Carlo average
`target_q = 1/(1-gamma) * future_rewards.mean(axis=0)` over `num_flow_goals`, and the
expectile is IQL-shaped asymmetric L2 on the regression residual
(`weight = where(adv >= 0, expectile, 1 - expectile)`, default 0.9, task values 0.9-0.99).
So: **mu = 0.9 and ONE latent per update, not a panel.** `index_panel=16` is this repo's
lower-variance choice, `index_panel=1` is InFOM's analogue, and S2 should be described as
InFOM-inspired rather than a port. Also `latent_dim` defaults to 512, not the README's 128.

### The first S2 launch was invalid — `q_dist` could not see the task

**Withdrawn:** an earlier reading of `q_dist_spread_rel` = 0.7-1.2% at 115k steps as the
plan's pre-registered kill signal ("not above 5% by 50k => the argmax is not the
bottleneck, S2 is dead"). That measured a bug.

The plan specifies `q_dist(s, u)`. Its regression target `psi(s, u', u)^T w` is a function
of the task vector, and `sample_mixed_z` redraws `w` **per batch element every step**, so a
head without `w` can only fit the task-MARGINAL expectile -- and `gpi_select` then ranked
candidate latents at eval while blind to the `task_z` it was acting for. A task-blind head
also trivially shows little spread, so the kill statistic was contaminated too. InFOM can
omit the task because it adapts per-task with reward labels; a zero-shot method cannot.

Fixed to `q_dist(s, w, u)` at all five sites (init, actor loss, distillation loss, spread
diagnostic, `gpi_select` broadcasting `task_z`). Evidence it matters: on the identical
200-step smoke, `q_dist_loss` 23.3 -> 9.0 and `q_dist_pred` **+0.94 -> -1.77** against a
target of -6.83 -- the old head could not even get the sign right. Jobs 2491543/2491545
cancelled at 39 min rather than carried to 500k.

### Control arm result, and why it is not quotable yet

`index_agg=max` (= Arm B: `policy_index=latent train_actor=false acting=gpi`) never touches
`q_dist`, so these two runs are unaffected by the bug.

| seed | 500-ep success | Wilson 95% |
|---|---|---|
| 0 | **0.374** (187/500) | [0.333, 0.417] |
| 1 | **0.090** (45/500) | [0.068, 0.118] |

Seed 0 is the highest zero-shot cube number on file other than FB's 0.721 -- above the actor
arm (0.220 +/- 0.037) and cube point (0.230 +/- 0.051). **Do not quote it.** Two seeds
spanning 0.090-0.374 give a t-interval of +/-1.80, wider than the measurement, and Arm B's
own record (0.006 / 0.160 / 0.084, 0.083 +/- 0.191) is an arm whose seed spread already
dwarfs its mean. A third control seed is running.

Both seeds also **climb late** -- sd0 0.06 at 350k -> 0.26 -> 0.52 -> 0.40; sd1 0.06 -> 0.20
-> 0.20 -> 0.12. An in-loop reading taken at 300k called this arm flat at BC level and was
simply premature; do not judge this arm before ~400k.

### In flight

`s2_cube_bexp09` seeds 0/1/2 and `s2_cube_bmax` seed 2 (jobs 2491549-2491552), 500k each on
one H100. Expectile arms run ~2.6x slower than the control (the 16-draw panel per step).
Launched at 3 seeds, not the plan's 2: the control's measured spread makes n=2 unable to
separate the arms. Pre-registered: success = beating the BC control (0.072 on this cluster)
at 500 episodes with non-overlapping CIs; the `q_dist_spread_rel` > ~5% necessary condition
is now re-testable on a head that can actually see the task.

---

<!-- _class: lead -->

## 2026-09-03 — DSRL-SAC launched: a scalar critic over LATENTS, flow never called in training

New switch `agent.critic_input: action | latent` in `agents/latentrl.py`. `action` is the
default and is byte-for-byte the pre-existing computation (same rng splits, same shapes,
same logged values — pinned by golden numbers in `tests/test_latentrl_smoke.py`). `latent`
is the offline DSRL arm: `Q(s, u)`, TD anchored on `u_data = clip(noise_preimage, ±u_clip)`,
actor `−Q(s, u_a)/stopgrad(|Q|) + bc_alpha_latent·‖u_a − u_data‖²`, **no decode anywhere in
the training graph** (a test perturbs `flow_onestep` and requires the actor loss not to
move). Residual head must be inert (`residual_eps=0`, asserted at `create()`). Acting is
unchanged: actor latent → frozen one-step decode. Plan: `docs/plans/2026-09-03-latentrl-dsrl-sac.md`.

### Why this cell of the grid was empty

DSRL (Wagenmaker et al. 2025) steers a frozen generative policy through its input noise; the
offline variant needs noise aliasing, and Stage-B preimages ARE that aliasing. What we had
run instead: psmflow's actor steers latents against `ψᵀw` (flat — D3, 1.1% relative spread),
and `latentrl`'s critic scored **decoded actions** with the gradient through the decoder
(0.142 ± 0.025 at ε=0, 0.905 ± 0.020 at ε=0.05). Nobody had run *scalar latent critic +
latent actor + no decode in training*.

### Arms (SLURM, one H100 per seed — no tmux on this machine, job name is the handle)

| SLURM job (id) | group | `u_clip` | seed | run dir under `$PSM_DATA/exp/PSMFLows/` |
|---|---|---|---|---|
| `latsac_cube_uclip3_sd0` (2491460) | `latsac_cube_uclip3` | 3.0 | 0 | `latsac_cube_uclip3/sd000_s_2491460.0.20260903_021732` |
| `latsac_cube_uclip3_sd1` (2491461) | `latsac_cube_uclip3` | 3.0 | 1 | `latsac_cube_uclip3/sd001_s_2491461.0.20260903_021732` |
| `latsac_cube_uclip1_sd0` (2491462) | `latsac_cube_uclip1` | 1.0 | 0 | `latsac_cube_uclip1/sd000_s_2491462.0.20260903_021732` |
| `latsac_cube_uclip1_sd1` (2491463) | `latsac_cube_uclip1` | 1.0 | 1 | `latsac_cube_uclip1/sd001_s_2491463.0.20260903_021731` |

All four on `gpu002.kisski`, `--gres=gpu:1` each, so one seed per H100 and no sharing.
Logs `$PSM_DATA/logs/<job name>-<id>.out`. Submitted through
`scripts/slurm/train_psmflow.sbatch` with `AGENT=latentrl`.

All: `agent=latentrl agent.critic_input=latent agent.use_point_preimage=true`,
`bc_alpha_latent=10.0`, `residual_eps=0.0`, `alpha=10.0`, `lr=3e-4`, `batch_size=256`,
critic `[512]×4` + layer_norm, `num_parallel=2`, `discount=0.99`, `tau=0.005`,
`pessimism_penalty=0.5`, `actor_pessimism_penalty=0.5`, actor/residual `512×2` with 2
embedding layers, `offline_steps=500000 eval_interval=50000 eval_episodes=50`.
Frozen Stage-A flow `/mnt/home/amohan/psm-data/flow/cube-single-play` @ 500000 (run group
`bcflow_cube_single_20260726_135032` — the checkpoint every cube number decodes through);
preimages `/mnt/home/amohan/psm-data/preimages/cube-single-play.npz` (canonical, alpha=20,
`num_samples=200`, 13/1M invalid), pulled from HF with `--with-flow`.

### Expected outcomes, pre-registered

- **near FQL (0.949) / the residual result (0.905):** offline latent TD works with a scalar
  critic; the zero-shot loss is in the representation (D2: best linear readout of φ explains
  ~13% of reward variance).
- **near 0.14:** offline latent TD does not survive without an action-space critic; next
  step is DSRL-NA (`Q_A` over dataset actions, latent Q distilled over prior draws).

**Prediction on the new live diagnostic:** psmflow's measure head sits at ~1% relative Q
spread; if this scalar critic is also ~1%, the actor gradient will again be BC-dominated.
`q_spread` / `q_spread_rel` (16 clipped prior draws at ≤64 states, key folded out of `rng`
so the training stream is untouched) is now logged every `log_interval`.

**Early reading (in flight, NOT a result).** 400-step smoke `latsac_smoke2`: `q_spread_rel`
0.034 → 0.026 → 0.020 → 0.019 over steps 100–400. All four real arms at 25k steps:
`q_spread_rel` **0.0017–0.0037** (uclip3 sd0/sd1 0.0034/0.0017, uclip1 sd0/sd1
0.0037/0.0020), i.e. already **below** psmflow's 1.1%, with `critic_q_data` running away
downward (−75 to −81) exactly as the 08-13 calibration forensics describe. On the
pre-registered reading that is the 0.14 branch, and `u_clip=1.0` did **not** raise the
relative spread — so the "everything decodes to nearly the same action" explanation for
flatness is not what the tighter box tests it to be. By 165k–175k steps `q_spread_rel` has
risen to **0.006–0.012**, i.e. settling right on psmflow's ~1% band, and the 50-episode
in-loop evals are wandering 0.06–0.32 (±0.115 CI each — do not quote them). Wait for the
500-episode evals before concluding; the prediction is on record either way.

### RESULT — all four runs finished 500k, 500-episode evals in

All four SLURM jobs `COMPLETED 0:0` (22–24 min wall clock each on one H100);
`params_500000.pkl` present in every run dir. Evaluated one job per GPU through
`scripts/slurm/eval500.sbatch` → `scripts/eval500.sh latentrl cube`, with
`agent.critic_input=latent` and the training `u_clip` repeated; each was cross-checked
against the run's own `flags.json` before submitting and both flags are now recorded **in
the eval JSON itself** (`u_clip`, `critic_input` added to `tools/eval_checkpoint.py`'s
report), so the numbers below are verifiable, not asserted. Eval seed left at the config
default 0 for all four — the protocol every recorded eval500 number used
(`scripts/reeval_ode_e2.sh` never passes a seed), so the 500 episode inits are paired.

| arm | seed | success (500 ep) | Wilson 95% |
|---|---|---|---|
| `latsac_cube_uclip3` | 0 | **0.186** (93/500) | [0.154, 0.223] |
| `latsac_cube_uclip3` | 1 | **0.180** (90/500) | [0.149, 0.216] |
| `latsac_cube_uclip1` | 0 | **0.096** (48/500) | [0.073, 0.125] |
| `latsac_cube_uclip1` | 1 | **0.158** (79/500) | [0.129, 0.193] |

Per arm, mean ± 95% CI across the 2 seeds (t interval, the project convention; with n=2 the
t multiplier is 12.7, so the u_clip=1 interval is uninformative on its own):

| arm | mean ± 95% CI |
|---|---|
| `latsac_cube_uclip3` (u_clip=3) | **0.183 ± 0.038** |
| `latsac_cube_uclip1` (u_clip=1) | **0.127 ± 0.394** |

**BC control measured on this cluster** (`agent=fql agent.bc_only=true`, same frozen flow the arms decode through, 500 episodes, `$PSM_DATA/evals/bc_cube.json`): **0.072** (36/500) [0.052, 0.098] — reproducing midi-01's 0.068 [0.049, 0.093], so both arms genuinely clear the control on this machine's own measurement.

Comparators, from `docs/tables/results.md` (same env, same 500-episode protocol):
FQL per-task **0.949 ± 0.063**; latent RL per-task ε=0.05 **0.905 ± 0.020**; latent RL
per-task ε=0 pure decode **0.142 ± 0.025**; BC control (per-step prior through the same
frozen flow) **0.068 [0.049, 0.093]**.

**Verdict: the 0.14 branch, not the 0.9 branch.** Both arms land in the pure-decode band —
u_clip=3 at 0.183 ± 0.038 is a hair above the ε=0 latent-critic result (0.142 ± 0.025) and
squarely inside the PSMFlow zero-shot band (0.236 ± 0.071 / re-eval 0.220 ± 0.037); u_clip=1
at 0.127 ± 0.394 is indistinguishable from it. Nothing is within reach of 0.905 or 0.949.
Both beat the BC control (0.068) — the loop is not inert, it just does not improve past what
a decode-only policy already gets. So a *scalar critic over latents with no decode in the
training graph* does not recover the per-task ceiling: everything that ever worked here
worked by letting the critic see and move a raw action. Per the plan, the next step is
DSRL-NA (`Q_A` over dataset actions, latent Q distilled over prior draws).

**The Q-spread prediction was right.** Final in-loop diagnostics at step 500000 (`train.csv`,
not an eval number):

| arm | `q_spread_rel` | `critic_q_data` | `actor_q` | `bc_loss` |
|---|---|---|---|---|
| uclip3 sd0 | 0.0125 | −165.50 | −166.76 | 0.961 |
| uclip3 sd1 | 0.0150 | −177.05 | −177.73 | 0.892 |
| uclip1 sd0 | 0.0110 | −166.20 | −167.22 | 0.415 |
| uclip1 sd1 | 0.0153 | −191.72 | −192.92 | 0.426 |

`q_spread_rel` 1.1–1.5% is exactly psmflow's measure-head band (1.1%), i.e. the critic
separates clipped prior draws about as poorly as ψᵀw does, and `actor_q` sits within ~1 of
`critic_q_data` — the actor is not finding anything the data anchor does not already have.
`critic_q_data` at −165 to −192 (from −75 to −81 at 25k) is the same monotone downward drift
the 08-13 calibration forensics describe, so the scale the actor normalises by is set by
runaway pessimism rather than by real value differences. `u_clip=1.0` did not raise the
relative spread at 500k any more than it did at 25k, so the tighter typical-set box does not
rescue separability — the "everything decodes to nearly the same action" story is not what
is limiting this.

`tools/make_tables.py` gained the two rows (`eval500_latsac_uclip{3,1}_sd?.json`, patterned
on the existing latent-RL rows) and they resolve against `$PSM_DATA/logs` here, but the tool
was **not run**: this cluster holds none of the historical `eval500_*.json`, so
`make_tables.py --logs $PSM_DATA/logs` would rewrite `docs/tables/results.md` with `--` in
every other cell. Run it on midi-01, or after the four JSONs are copied to that logs dir.

### Also

- `scripts/eval500.sh` gained a `latentrl` mode and now **rejects unknown modes**: it
  previously fell through to `agent=psmflow` for any first argument other than `bc`, so
  `eval500.sh latentrl ...` would silently have evaluated the wrong agent.
- **`u_clip` must be repeated at eval.** Nothing outside the agent config hard-codes 3.0
  where it matters (`utils/flow_inversion.py` never clips; `tools/analyze_preimages.py`'s
  `U_CLIP_DEFAULT = 3.0` is a `--u_clip`-overridable diagnostic default; `psmflow.py:776` is
  a fallback `get_config`), but `tools/eval_checkpoint.py` rebuilds the agent from the hydra
  config, and the actor is `tanh * u_clip`. Evaluating the `u_clip=1.0` arm without
  `agent.u_clip=1.0` scales every action by 3x. Noted in `eval500.sh`'s header.
- `scripts/slurm/train_psmflow.sbatch` takes `AGENT`, `EVAL_EPS`, `LOG_INT` (defaults
  reproduce every earlier submit line).
- `scripts/eval500.sh`'s midi-01 paths are now env-overridable (`PSM_REPO`, `EVAL_LOGS`,
  `EXP_ROOT`, `FLOW_DIR`, `PRE_NPZ`, `OGBENCH_DATASET_DIR`); the defaults are unchanged, so
  every midi-01 invocation still works. It also fails fast if the flow dir or preimage npz
  is absent, and validates `MODE` before touching paths. New `scripts/slurm/eval500.sbatch`
  wraps it for the scheduler, the way `train_psmflow.sbatch` wraps `main.py`.
- `tools/eval_checkpoint.py` now records `u_clip` and `critic_input` in the report JSON, so
  a latentrl eval can be checked against the run's `flags.json` after the fact rather than
  trusted.
- Full suite module-per-process on this cluster: **37/37 modules, 223 passed, 3 skipped,
  zero failures** (`test_latentrl_smoke.py` 13 passed).
- On this machine `.venv` had neither pytest nor ruff; installed. `ruff check .` reports 354
  pre-existing findings repo-wide under ruff 0.16.5 (no `lint.select` in `pyproject.toml`,
  so the default rule set moved) — `agents/latentrl.py` is clean and the one finding in
  `tests/test_latentrl_smoke.py` (C408) is pre-existing at HEAD.
- `tests/test_decode_recovery.py` fails on this cluster with a pre-existing float32/float64
  carry mismatch in `agents/fql.py:51` (`implicit_euler_step`); unrelated to this change.

### Addendum (2026-09-03) — does the behaviour flow MEMORIZE cube? No: train ≈ val, everywhere

First run of `tools/diag_flow_fit.py` (untracked, written but never executed). SLURM
`flowfit` (2491473), one H100, 7 min: 2048 anchors x 512 unclipped N(0,I) latents, exact
ODE at `flow_steps=100` and the distilled one-step head, k_max=512, 250k-row standardized
k-NN index, eps swept in multiples of the median anchor-1NN state distance (1.029). Report:
`$PSM_DATA/logs/diag_flow_fit_cube.json`. Added a `min_mse` block to the tool (per-anchor
`min_j |G(s,u_j) - a|^2` and its min-over-the-whole-ball twin, with the data-to-data value
in the same ball as the scale); everything else is as written.

Lowest MSE per anchor, in units of `(mean|a|)^2 = 0.311^2` (ODE):

| split | own mean | median | p90 | ball@1x mean | data-self@1x mean | own / data-self |
|---|---|---|---|---|---|---|
| train (in Stage-A training set) | **0.0708** | 0.0437 | 0.1486 | 0.0782 | 0.699 | 0.101 |
| val (OGBench held-out split, 100k rows) | **0.0718** | 0.0452 | 0.1504 | 0.0615 | 0.619 | 0.116 |

val/train ratio **1.014** (ODE) and **1.003** (one-step) — the memorization line is flat.
The val split is genuinely held out: `main.py` only ever calls `train_dataset.sample()` for
gradients, `val_dataset` is logging-only, and OGBench ships it as a separate file.

- The flow reproduces a recorded action ~10x better than the nearest *other* dataset action
  in the same ball does (0.071 vs 0.70), so it is well inside the aleatoric noise floor of
  the behaviour conditional, and identically so on states it never saw.
- One-step decode is ~1.5x worse than ODE-100 on the same latents (own mean 0.105 vs 0.071)
  and splits the same way (1.003), so the deployed head loses fidelity but not generality.
- `null` (same ball population, wrong location) sits at 2.6-2.8 against `recall_ball`
  0.24-0.82: conditioning on s carries real information at every radius.
- Ceiling from the stored point preimage: **3.9e-4** — the exact inverse decodes ~180x
  tighter than the best of 512 prior draws, i.e. the residual is coverage of the prior, not
  capacity.
- `min_vs_n` at 1x: recall_ball falls 1.120 (N=1) -> 0.322 (N=512), still not flat, so any
  single min-over-N number is N-bound and must be quoted with N.

**Reading: the flow fits the conditional, it does not memorize.** Whatever is wrong
downstream (the 0.14/0.18 band above) is not a Stage-A overfitting problem.

### Addendum (2026-09-03, later) — the same diagnostic on antmaze: fits worse, memorizes a little

SLURM `flowfit-antmaze` (2491474), one H100, 7 min, through the new
`scripts/slurm/diag_flow_fit.sbatch`. Identical protocol to cube (2048 anchors x 512
unclipped `N(0,I)` latents, ODE-100 + one-step head, k_max=512, 250k-row standardized k-NN
index, the same 0.25x-3x eps sweep). Flow `$PSM_DATA/flow/antmaze-medium-navigate` @ 500000,
`d_a = 8`, `mean|a| = 0.6395`, median 1-NN state distance 1.789. Report:
`$PSM_DATA/logs/diag_flow_fit_antmaze.json`. **The diagnostic needs no preimages** — every
statistic samples `u ~ N(0, I)`; the npz is read only for the `point_preimage_ceiling` line,
so the 881/1M invalid antmaze rows are irrelevant here (the ceiling subsamples 256 rows).

Lowest MSE per anchor, in units of `(mean|a|)^2 = 0.6395^2` (ODE-100):

| split | own mean | median | p90 | ball@1x mean | data-self@1x mean | own / data-self |
|---|---|---|---|---|---|---|
| train | **0.2392** [0.229, 0.252] | 0.1809 | 0.4447 | 0.2602 | 1.125 | 0.213 |
| val (100k held-out rows) | **0.3047** [0.284, 0.330] | 0.2112 | 0.5300 | 0.2994 | 1.200 | 0.254 |

val/train ratio **1.274** (ODE), **1.131** (one-step) — non-trivially above 1, against
cube's 1.014/1.003. `recall_ball` (train, ODE) 0.461 / 0.675 / 1.091 / 1.168 at
0.5x / 1x / 2x / 3x with `null` 2.83 at 1x; one-step is within 0.005 of ODE at every radius
(0.463 / 0.676 / 1.080 / 1.156, null 2.82). Ceiling from the point preimage **3.8e-4**,
essentially cube's. `min_vs_n` at 1x falls 1.683 (N=1) -> 0.675 (N=512), still not flat.
Caveat: antmaze balls saturate the k_max=512 neighbour cap (62% of anchors at 2x, 99% at
3x; cube 55% at 3x), so the 2-3x radii are truncated balls, not full ones.

**Reading vs cube.** Antmaze fits the conditional *worse in absolute terms* — own min-MSE
0.239 vs 0.071, and 0.21 of the data's own local spread vs cube's 0.10 — and it is the one
env with a real, if small, train/val split (1.27x). Not memorization in the damaging sense:
a memorizing flow reproduces train anchors and fails held-out ones outright, and 1.27x on a
statistic whose own bootstrap CI is ±5% is a mild generalization gap on top of a fit that is
still ~4x tighter than the nearest other dataset action. The one-step head is the surprise
in the other direction: on cube it costs 1.49x on own min-MSE, on antmaze only 1.16x, and
its recall curves are indistinguishable from ODE-100 — the distillation gap is a cube
phenomenon, not a general one. pointmaze was **not** run: its Stage-A checkpoint pulled fine
from HF, but the OGBench `pointmaze-medium-navigate-v0` dataset is not on this cluster and
`rail.eecs.berkeley.edu` is unreachable through the proxy.

Figures (`tools/fig_diag_flow_fit.py`, reads every `$PSM_DATA/logs/diag_flow_fit_<env>.json`;
`.pdf` beside each `.png`), plus the generated table `docs/tables/flow_fit.md`:
`PAPER/ICLR/figures/fig_flow_fit_recall.png`,
`PAPER/ICLR/figures/fig_flow_fit_memorization.png`,
`PAPER/ICLR/figures/fig_flow_fit_min_vs_n.png`.

---

<!-- _class: lead -->

### Addendum (2026-09-03, later still) — three audit bugs fixed; point-vs-mixture measured at 500 episodes on cube AND antmaze

Acting on `docs/design/2026-09-03-latent-actor-audit.md` (bugs 1, 2, 4) and
`docs/design/2026-09-03-paper-code-audit.md` (discrepancy #3, the mixture `q_alpha`).
Nothing on the shipped code path changed — verified byte-identical, see below — so the
point arms are a straight reproduction check.

#### Fixes

| # | file:line | what | severity |
|---|---|---|---|
| 1 | `agents/psmflow.py:253` | `flow_actor_loss` read psi at the hardcoded task vector `w`; now `self._index(sampled)`. Under `policy_index=latent` that slot is `d_a` wide, so the old call both mis-typed the head and raised `ScopeParamShapeError` on the first update. | real, unreachable from any shipped config |
| 2 | `agents/psmflow.py:183` | `r_expl, r_emask` were split from `r_next` **after** `normal(r_next, ...)` had consumed it. Now `split(fold_in(rng, 107))`, the same convention as the `104`/`106` branches. | cosmetic (`backup_explore_frac=0.0` by default) |
| 3 | `tools/eval_checkpoint.py:69,106,177,263` | new `_cli_agent_keys()` + `merge_run_config()`: the **run's own `flags.json` supplies the agent config**, typed CLI overrides layered on top; provenance recorded in the report JSON as `agent_config_source`, and `policy_index`/`train_actor` added to the report. | operational — the one path that produced a wrong number **silently** |

Semantics of fix 1, decided rather than asserted: the actor stays **w-conditioned** (it is
the policy the deployed action comes from), the index slot carries **u'** exactly as
`measure_loss` and `gpi_select` read it, and the readout stays `Q = psi(s, u', u_a)^T w`.
That is coherent, so `create()` gained no assert — `policy_index=latent` × `train_actor=true`
now simply works. Fix 3 inherits **only keys the current config already has**, so a stale or
newer `flags.json` can change a value this checkout reads but cannot add or drop a field;
`u_clip`, `acting`, `train_actor`, `policy_index` and `critic_input` can no longer mismatch
in silence. Verified live: the eval of the 09-02 antmaze run printed
`agent config defaults from .../flags.json`, `CLI overrides kept: flow_ckpt_epoch,
flow_ckpt_path, preimage_path, use_point_preimage`, `(run config already matches this
checkout's defaults)`.

#### Tests

New: `tests/test_psmflow_policy_index.py:105,126` (Arm B with a trained actor takes a
finite step and the actor moves; psi is read at the policy index, and the old
hardcoded-`w` call is pinned as the shape error it was),
`tests/test_psmflow_backup_explore.py:74` (the explore keys are folded out of `rng`, not
split from the consumed `r_next`), and `tests/test_eval_checkpoint_flags.py` (11 cases:
flags.json supplies defaults, a typed override — flat or nested — still wins, the schema
stays this checkout's, a different `agent_name` is refused, and no/unreadable/agent-less
flags.json plus no `restore_path` are all no-ops, which is what keeps every recorded
eval500 number reproducible; plus an explicit pin that the `bc` control's semantics do not
move).

Both psmflow tests were confirmed to **fail** against the unpatched file before the fix.
The **default path is byte-identical**: a fixed-seed fingerprint of `sample_step_inputs`
(all seven draws, hex bits) and of a full `update`'s losses and post-step
phi/psi/actor/actor_vf parameters is unchanged before vs after — which is why the point
arms below are a reproduction and not a new measurement.

#### What the mixture arm can and cannot show

Audit row `paper-code-audit.md:75`: the write-up's loss draws `u_i ~ q_alpha(.|s_i,a_i)`,
the epsilon-relaxed posterior; **every shipped result instead used
`use_point_preimage=true`, i.e. `q_alpha = delta_{u*}`.** This is the first time the
mixture arm has been run to 500 episodes. Caveat that limits the ceiling of the result
(audit row `:74`): `configs/inversion/default.yaml` ships `num_clusters: 1`, so the
"mixture" is a **single** Gaussian — the multi-modal refinement the paper describes has
never been run at K>1. What this measures is "point estimate vs a Gaussian around it",
not "point vs a multi-modal posterior".

#### The mixture arm cannot use the published npz — finding, not a choice

`use_point_preimage=false` on `$PSM_DATA/preimages/{cube-single-play,antmaze-medium-navigate}.npz`
is **refused by `main.py:143-148`**: neither sidecar records `inversion.prior_scale`, and
absent means the pre-08-14 likelihood-only target, whose fits sit outside the prior the
latent actor samples from. The canonical npz files therefore support the point arm only.
The mixture arm runs instead on the HPO-corrected npz pulled from the HF dataset repo —
`cube-single-play-a20p6-ps0p69-ns12-N200` (alpha 20.57, prior_scale 0.691, n_steps 12) and
`antmaze-medium-navigate-a26p5-ps0p60-ns5-N200` (alpha 26.51, prior_scale 0.597, n_steps 5),
the 09-01 sweep winners. Both carry `noise_preimage_{mean,cov,weights}` at K=1, both
sidecars were repaired to this machine's flow dir by `hf_preimages.py pull --with-flow`.

That makes point-vs-mixture confounded with npz-vs-npz, so a **third arm** was added at seed
0: `pointps` = POINT preimages read from the SAME corrected npz. `mix` vs `pointps` is the
clean mixture ablation; `point` vs `pointps` isolates the inversion settings.

#### Arms — psmflow stage C, shipped defaults throughout

`batch_size=1024 z_dim=128 num_parallel=2 discount=0.98 tau=0.01 ortho_coef=1000
pessimism_penalty=actor_pessimism_penalty=0.5 mix_ratio=0.5 backup_explore_frac=0.0
acting=actor policy_index=task_vector train_actor=true u_clip=3.0 gpi_decode=onestep
action_critic.enabled=false actor.bc_coeff=1.0 lr_phi=1e-5 lr_sf=1e-4 lr_actor=1e-4
lr_actor_vf=3e-4`; `offline_steps=500000 online_steps=0 eval_interval=50000
eval_episodes=50 save_interval=250000`; flow `$PSM_DATA/flow/<env>` @ 500000.
Every run's own `flags.json` was re-read after launch and all seven fields confirmed.

#### Pre-registered expectations (written before any eval landed)

1. **cube point reproduces 0.220 ± 0.037.** No shipped-path code changed and the golden
   fingerprint is byte-identical, so anything outside that band is a machine/artifact
   difference, not the fixes.
2. **cube mix within noise of cube point.** E4b already measured the mixture-trained
   checkpoint's ranking Spearman and Q spread in the same band as the point arm — the
   mixture does not create ranking signal, so it should not move success.
3. **antmaze (both arms) lands near its BC control.** D2/D3 say `Q_W = psi^T w` carries
   almost no ranking signal, the 09-03 actor audit measured the deployed action as 0.85%
   sensitive to `w` and a 0.961-correlated pass-through of its own noise, and the 09-01
   HPO recorded antmaze coverage 0.036 at its own optimum. **Expected failure**: no
   separation from BC.
4. **`pointps` ≈ `point`.** The corrected inversion changes the BC anchor's target, not the
   critic; if it moves success materially, the standing "the actor is BC with an 8%
   perturbation" reading needs revisiting.

#### RESULT — all ten runs finished 500k; 500-episode evals

| env | arm | seed 0 (500 ep) | seed 1 (500 ep) | mean ± 95% CI (t, across seeds) | pooled |
|---|---|---|---|---|---|
| cube | point preimage, canonical npz | 117/500 = **0.234** [0.199, 0.273] | 113/500 = **0.226** [0.192, 0.265] | **0.230 ± 0.051** | 230/1000 = 0.230 [0.205, 0.257] |
| cube | mixture `q_alpha`, corrected npz | 94/500 = **0.188** [0.156, 0.225] | 100/500 = **0.200** [0.167, 0.237] | **0.194 ± 0.076** | 194/1000 = 0.194 [0.171, 0.220] |
| cube | point preimage, corrected npz *(control)* | 143/500 = **0.286** [0.248, 0.327] | 109/500 = **0.218** [0.184, 0.256] | **0.239 ± 0.102** (3 seeds) | 358/1500 = 0.239 [0.218, 0.261] |
| antmaze | point preimage, canonical npz | 112/500 = **0.224** [0.190, 0.263] | 101/500 = **0.202** [0.169, 0.239] | **0.213 ± 0.140** | 213/1000 = 0.213 [0.189, 0.239] |
| antmaze | mixture `q_alpha`, corrected npz | 101/500 = **0.202** [0.169, 0.239] | 105/500 = **0.210** [0.177, 0.248] | **0.206 ± 0.051** | 206/1000 = 0.206 [0.182, 0.232] |
| antmaze | point preimage, corrected npz *(control)* | 131/500 = **0.262** [0.225, 0.302] | 94/500 = **0.188** [0.156, 0.225] | **0.247 ± 0.131** (3 seeds) | 370/1500 = 0.247 [0.226, 0.269] |

**Seeds 1 and 2 of both `pointps` arms** (added 2026-09-03 later; SLURM `2491503`–`2491506`
for training, `2491507`–`2491510` for the evals; each run's `flags.json` is byte-identical to
its seed-0 sibling except for `seed`, verified after launch). Seed 2 has no column in the
table above, so all three seeds together:

| env | arm | seed 0 | seed 1 | seed 2 | mean ± 95% CI (t, n=3) | pooled |
|---|---|---|---|---|---|---|
| cube | point preimage, corrected npz | **0.286** (143/500) | **0.218** (109/500) | **0.212** (106/500) | **0.239 ± 0.102** | 358/1500 = 0.239 [0.218, 0.261] |
| antmaze | point preimage, corrected npz | **0.262** (131/500) | **0.188** (94/500) | **0.290** (145/500) | **0.247 ± 0.131** | 370/1500 = 0.247 [0.226, 0.269] |

Across-seed SD is 0.041 (cube) and 0.053 (antmaze) — this arm is the **noisiest** of the
three, and seed 0 was its best seed on both envs. Wall clock 50 min (cube) / 56 min
(antmaze) per 500k run, 4–9 min per 500-episode eval.

Comparators. **cube**: PSMFlow re-eval **0.220 ± 0.037**, FB **0.721 ± 0.020** (3 seeds,
`docs/tables/results.md:12`), BC control **0.072** [0.052, 0.098]. **antmaze**: PSMFlow
mixture 09-02, 3 seeds, **0.215 ± 0.041** (`$PSM_DATA/evals/psmflow_antmaze_sd00{0,1,2}.json`);
midi-01 doc-recorded 1-seed PSMFlow **0.222** [0.188, 0.261] and BC **0.090** [0.068, 0.118]
(`docs/HANDOFF.md:1137`); BC control re-measured on this cluster **0.072** [0.052, 0.098]
(`$PSM_DATA/logs/bc_control_antmaze.json`, 36/500). **There is no FB number on antmaze at
any episode count** — FB has only ever been run on cube, so the antmaze rows have no
zero-shot baseline to sit beside.

Two exact-count collisions were checked and are not artefacts: antmaze point sd1 and the
09-02 mixture sd0 are both 101/500 but their per-episode vectors differ in 162 places, and
the cube and antmaze BC controls are both 36/500 with different vectors. Wall clock:
cube 50 min, antmaze 56-59 min per 500k run on one H100 (10 concurrent).

#### Verdicts against the pre-registered expectations

1. **CONFIRMED.** cube point = 0.230 ± 0.051, pooled [0.205, 0.257] — inside the recorded
   0.220 ± 0.037 band. Stronger than that: the antmaze mixture arm reproduced the 09-02
   runs **bit-exactly** — 101/500 and 105/500 with **Hamming distance 0** on the
   per-episode vectors, from independently trained checkpoints on different nodes a day
   apart. Fixes 1 and 2 provably did not touch the shipped path.
2. **REFUTED.** The mixture is **worse**, not equal. Against the point arm on the SAME
   corrected npz, cube drops 0.286 [0.248, 0.327] → 0.194 [0.171, 0.220] — the Wilson
   intervals do not overlap, a real ~0.09 loss. Antmaze drops 0.262 [0.225, 0.302] →
   0.206 [0.182, 0.232], also non-overlapping. E4b showed the mixture does not create
   *ranking* signal; this shows it actively degrades the BC anchor. Read with the K=1
   caveat above: sampling a Gaussian around `u*` instead of `u*` injects noise into the
   CFM target and the TD anchor without adding modes.
3. **REFUTED.** Antmaze does **not** collapse to its BC control: 0.213 and 0.206 against
   0.072 [0.052, 0.098], ~3x and far outside the interval, the same separation cube shows
   (0.230 vs 0.072). Whatever the actor audit's "BC with an 8% perturbation" reading
   explains, it is not that the agent is indistinguishable from BC on success. It remains
   true that neither env comes near cube's FB 0.721.
4. **CONFIRMED at 3 seeds — and the 1-seed reading below it is WITHDRAWN.** With seeds 1
   and 2 in, `pointps` ≈ `point` on both envs, exactly as pre-registered: cube
   **0.239 ± 0.102** vs point **0.230 ± 0.051** (diff +0.009, Welch t=0.36); antmaze
   **0.247 ± 0.131** vs point **0.213 ± 0.140** (diff +0.034, Welch t=1.04). Every t-interval
   overlaps, and so do the pooled Wilson intervals (cube [0.218, 0.261] vs [0.205, 0.257];
   antmaze [0.226, 0.269] vs [0.189, 0.239]). **The provisional 1-seed claim that the
   HPO-corrected inversion buys ~0.05 does not survive replication**: seed 0 was the best
   seed of three on *both* envs (cube 0.286 vs 0.218/0.212, antmaze 0.262 vs 0.188/0.290),
   and the across-seed SD of this arm (0.041 / 0.053) is about the size of the effect that
   was read off it. Stage-B inversion quality is **not** shown to move Stage-C success, and
   the standing "the actor is BC with an 8% perturbation" reading does not need revisiting
   on this evidence. Do not quote the +0.05 anywhere — it was a one-seed artefact, and this
   is the second time in this entry that a single seed pointed the wrong way.

   *Consequence for verdict 2:* the mixture is still the worst arm, but with the correct
   3-seed `pointps` comparator the cube gap narrows from 0.286 → 0.194 to
   0.239 ± 0.102 → 0.194 ± 0.076 (Welch t=1.82) and the antmaze gap from 0.262 → 0.206 to
   0.247 ± 0.131 → 0.206 ± 0.051 (t=1.33) — **neither is significant across seeds** now,
   though the pooled Wilson intervals still separate on cube ([0.218, 0.261] vs
   [0.171, 0.220]) and now touch on antmaze ([0.226, 0.269] vs [0.182, 0.232]). Verdict 2's
   direction survives; its "the intervals do not overlap" phrasing was resting on the
   single lucky `pointps` seed and should be read as *point ≥ mixture, cube significant
   pooled, antmaze not*.

Net ordering at 3 seeds: **point/corrected ≈ point/canonical > mixture**, consistent
across both environments — the two point arms are indistinguishable and the paper's
`q_alpha` is the worst of the three (significantly so only on cube, pooled).

#### Artifacts

Run dirs under `$PSM_DATA/exp/PSMFLows/`: `psmflow_{cube,antmaze}_{point,mix}/sd00{0,1}_s_<jobid>.*`
(jobs 2491480-2491487) and `psmflow_{cube,antmaze}_pointps/sd00{0,1,2}_s_<jobid>.*`
(seed 0: 2491488-2491489; seeds 1-2: 2491503-2491506, evals 2491507-2491510).
Eval JSONs in `$PSM_DATA/logs/`: `eval500_psmflow_{cube,antmaze}_{point,mix,pointps}_sd<k>.json`,
`eval500_psmflow_antmaze_mix_20260902_sd0.json` (the fix-3 validation re-eval),
`bc_control_antmaze.json`. All registered in `tools/make_tables.py` (not run — its `LOGS`
default is still the dead midi-01 path, so it needs `--logs $PSM_DATA/logs`, and
`docs/tables/results.md` is cube-only until someone decides how to split the envs).

Tests: `.venv/bin/python -m pytest <module> -q` per module over all 38 test files, on a
compute node (the login node thrashes at load 95 and one module ran >15 min there):
**38/38 exit 0, 237 passed, 3 skipped, 0 failed** — the skips are the
`PSMFLOWS_STAGE_A_CKPT`-gated ones. `ruff check` on every touched file reports exactly the same findings as
the unmodified versions — 14 across `agents/psmflow.py` + `tools/eval_checkpoint.py`, 7 in
`tools/make_tables.py`, 2 across the two edited test modules, all pre-existing, and
`tests/test_eval_checkpoint_flags.py` clean: **zero new**.


---

<!-- _class: lead -->

## 2026-09-01 (later) — inversion HPO redone on the corrected target; cube regenerating, antmaze not

The 08-28 50-trial sweeps and every mixture npz they produced measured the OLD inversion
(un-squared likelihood, alpha^2 Laplace covariance). Both were redone on the corrected
target, same settings as 08-28 (`n_rows=256 sample_k=32 cov_k=16`, `flow_steps=100`,
`num_samples=128` pinned, alpha in [2,100] log, `prior_scale` in [0.3,1.0],
`n_steps` in [5,20]), 50 trials each. Logs: `logs/hpo_{cube,antmaze}_fixed.log`.

| env | alpha | prior_scale | n_steps | coverage@16 | decode_mix | ESS | E\|u\|^2 / d_a |
|---|---|---|---|---|---|---|---|
| cube | 25.26 | 0.677 | 17 | **0.982** | 0.2165 | 68.2 | 1.04 |
| antmaze | 26.51 | 0.597 | 5 | **0.036** | 0.2653 | 8.5 | 1.07 |

Three readings.

**Both incumbents sit exactly on the decode budget** (0.2165 vs 0.22; 0.2653 vs 0.28). The
objective maximizes coverage subject to a fidelity hinge, so the optimizer widens until the
hinge binds. That is a frontier position, not evidence of latent multiplicity: the exact
inverse of a diffeomorphism is unique, and the corrected alpha sweep reproduces the same
width-is-temperature collapse (cube cov_k16 0.945 at alpha=20 down to 0.003 at alpha=3200).

**Typicality is what the fix actually bought.** E\|u\|^2 / d_a is now 1.04 (cube) and 1.07
(antmaze) against the prior's 1.0. The shipped npz built under the old target sits at
3.92/5 = 0.78 -- measurably UNDER-dispersed. Lemma "typicality" says an exact flow's
preimages are prior-distributed; the corrected target nearly satisfies it, the old one did
not. This is the first mixture arm whose latents look like the ones the actor emits.

**Antmaze was not rescued.** Coverage 0.036 at its own optimum against cube's 0.982. The
corrected inversion does not close the cube/antmaze gap, consistent with the 08-30 Jacobian
probe having already excluded local geometry. Regenerating 1M antmaze rows at 3.6% coverage
would cost ~13 h for a file already measured useless as augmentation; **not run**.

**Cube is regenerating** at the runner-up alpha=20.57 / prior_scale=0.691 / n_steps=12
(cost -0.9805 vs the incumbent's -0.9819, coverage 0.981 vs 0.982, decode 0.2117, ESS 62) --
a statistical tie at ~23 h instead of ~33 h, since EM cost scales with n_steps.
`num_samples` 128 -> 200 for the real file (the HPO pins it during the width search so the
optimizer cannot buy coverage with compute). Output
`preimages_cube_single_fixed_a20p6_ps0p69_ns12_N200.npz`, tmux `pre-cube-fixed`, ETA
2026-09-02 ~19:00 CDT.

**Why this is worth running at all**, given the mixture arm's record: every mixture result
on file is confounded. Pre-08-14 arms had no prior factor; the 08-28 HPO and its npz had the
un-squared target and alpha^2 covariance; E4b's probe used that same npz. So the mixture has
never been tested with a proposal that has a Laplace mode at its own centre. The mechanism
under test is NOT "the preimage is a set" -- that is dead -- but whether a wider, typical,
faithful latent target regularizes psi. Against it stands the transition-label bias:
any u~ != u* pairs a perturbed action with the recorded s'.

**Gate, pre-registered.** Judge the resulting Stage-C runs on representation identifiability
before success: D1 policy-ranking Spearman (point arm 0.10, Arm B 0.079, old mixture 0.054),
D3 relative Q spread (1.1% / 0.9% / 0.86%), and the actor's percentile under Q (44th). If
those do not move, the mixture does not create ranking signal and the arm closes for good.

**Point preimages are unaffected and this was measured, not assumed.** `noise_preimage_point`
is the backward-ODE solution and takes no alpha or prior_scale; recomputing 4096 cube rows
with today's code reproduces the 08-04 npz **bit-identically** (max abs diff 0.000e+00). So
no shipped result, and neither E3 arm, is invalidated by the inversion bugs.

---

<!-- _class: lead -->

## 2026-09-01 — E3 landed: the paper's construction does not work on cube

Closes `docs/plans/2026-08-31-interface-fork-experiments.md`. Both arms ran 3 seeds ×
500k steps, evaluated at 500 episodes on the post-P0.2 pinned-stream harness. Table
rows regenerated from the eval JSONs; CIs are the repo's t95-across-seeds.

| arm | success | matched control |
|---|---|---|
| Arm A — `backup_explore_frac=1.0`, `acting=actor` | **0.171 ± 0.113** (3) | point arm, actor: 0.220 ± 0.037 (5) |
| Arm B — ψ(s, u, u′), no actor, `acting=gpi` | **0.083 ± 0.191** (3) | point arm, gpi: 0.054 ± 0.032 (5) |

Controls: BC per-step prior 0.068 [0.049, 0.093]; FB 0.721 ± 0.020; FQL 0.949 ± 0.063.
Each arm is quoted against the control that ACTS the same way — Arm B selects by latent
argmax, so the actor-arm 0.220 is not its comparison and using it would price the removal
of the actor as if it were the index change.

**Arm A is within noise of its control.** Making every bootstrap action a genuine p0
decode — the hypothesis Prop. "insample" needs for C=1, which no shipped run ever
satisfied — changes nothing. Pre-registered expectation held: a correct backup
distribution does not by itself create ranking signal.

**Arm B lands at BC level.** 0.083 ± 0.191 against a gpi control of 0.054 ± 0.032, with a
seed spread (0.006 / 0.160 / 0.084) far wider than any effect. It does not separate from
the 0.068 BC control, and is nowhere near the 0.45 bar or FB's 0.721. The pre-registered
reading fires: **the writeup's construction is refuted on its own terms on cube.**

What this does NOT isolate: Arm B changes the ψ index and drops the actor together, per
§8, so it does not price the policy-index idea alone. What it does establish is that the
algorithm as written does not work here, and that is coherent with E1 — Arm B's only means
of choosing an action is ranking latents by ψᵀw, and E1 showed ranking is the broken
faculty. The oracle reached 0.934 selecting from the same action set these agents fail in.

Taken with E1 and E2, the three experiments converge on one statement: the interface is
not the problem (oracle-aim 0.934), the decoder is not the problem (exact decode costs
−0.038 ± 0.027 paired), and neither of the paper's two deviations was load-bearing. The
open problem is a critic that can rank latents.

### Infrastructure, same day

`/var/local` (24 GB) hit 100% at 16:42 and killed all six runs mid-training with
`OSError: [Errno 28]`, losing ~3.5 h; they were relaunched on `/data-local` at 17:40 and
the numbers above are from those clean 500k runs. `scripts/launch_psmflow.sh` now takes
`STORE=`, defaulting to `/var/local` so older launch lines reproduce. 11.82 GiB of
finished July experiments were uploaded to `amsks/psmflows-checkpoint-archive`, verified
file-by-file, then deleted (`docs/reference/archived-checkpoints.md`).

Two mid-training `*_ep250000.json` evals of the killed runs exist in the logs. They are at
half the specified budget and `tools/make_tables.py` now globs `sd?` rather than `sd?*` so
they cannot be pooled into the arm rows.

---

<!-- _class: lead -->

## 2026-08-31 — oracle-aim: the reachable set is fine (0.934), the loss is ranking; paper-faithful arms launched

Plan and pre-registered readings: `docs/plans/2026-08-31-interface-fork-experiments.md`.
Everything below is cube unless stated. Table regeneration: commit `8e89d76`.

### The audit that set this up

A three-way audit of code vs the formal writeup (`git show 5249267:PAPER/main.tex`)
found the shipped agent replaced both hypotheses of Prop. "insample" (C=1): the
bootstrap latent is the actor's (`psmflow.py:139`), and the ψ index slot carries the
task vector, not a policy latent (`psmflow.py:106,111`). So every negative Stage-C
number to date tested latent-space FB, not the paper's construction. Three inversion
bugs also surfaced and are now FIXED (commits `313948e`, `3ef0afe`): the likelihood
norm was un-squared, the Laplace proposal used α²JᵀJ instead of the spec's 2αJᵀJ
(10× too narrow at α=20), and `preimage_valid` never reached sampling (u=0-repaired
rows trained on wrong pairs). Validation at matched fidelity: cube ESS 112.9 → 132.6,
antmaze ESS 8.8 → 18.8 with coverage 0.016 → 0.064 — the ESS collapse was partly this
bug, not purely d_a=8 geometry; antmaze mixture remains short of usable. Default alpha
retuned 20 → 50 with a frontier table in `configs/inversion/default.yaml`. All mixture
npz files and both 08-28 HPO sweeps predate the fix and measure the old target.

### E1 — oracle-aim rollout (`tools/diag_oracle_aim.py`, 500 ep, K=512, ODE-100)

Execute, at each step, the decoded prior latent closest to a frozen FQL expert's
action. **oracle-aim 0.934 [0.909, 0.953]** against the expert's own 0.960 [0.939,
0.974]; random-latent floor 0.086 (consistent with the BC control). Mean min-distance
to the expert action over K=512: 0.062. Pre-registered fork: ≥0.7 ⇒ **cannot aim** —
the flow's reachable action set contains near-expert behavior and the entire Stage-C
loss is latent *selection*. No flow retraining is indicated by this number; the target
is a critic that can rank.

### E2 — ODE re-eval of the 5 shipped checkpoints (post-P0.2 seeding, 500 ep × 5 seeds)

| arm | one-step | ODE-100 |
|---|---|---|
| actor | 0.220 ± 0.037 | 0.182 ± 0.019 |
| gpi | 0.054 ± 0.032 | 0.044 ± 0.043 |

Exact decode helps nowhere (slightly worse everywhere; even the random-latent floor
drops, 0.086 → 0.014). The one-step distilled decoder is exonerated as a loss source;
its 0.0886 preimage decode error stays a bookkeeping caveat only. The re-measured
one-step control (0.220 ± 0.037) supersedes the pre-seeding-fix 0.236 ± 0.071 for
comparisons; sd0's recorded 0.318 is not reproducible post-fix (0.240).

### E3 — paper-faithful arms (training today; results below when evals land)

Arm A: `backup_explore_frac=1.0` — the exact `u′~p₀` bootstrap, all else shipped
config, acting=actor, 3 seeds. Arm B: `policy_index=latent` — ψ(s, u, u′) with a
fresh prior policy-index latent, `train_actor=false`, acting=gpi, 3 seeds. Both point
arm, pessimism 0.5, flags verified from the runs' own flags.json. Pre-registered:
Arm A alone likely within noise of 0.22 (a correct backup does not create ranking
signal by itself); Arm B is the first number Prop. insample applies to — clear FB's
0.721 and the mechanism is demonstrated; land ~0.2 and the C=1 construction is
refuted on its own terms on a field where E1 proves a winning selection exists.

### Arm results at step 250k (training was torn down externally at ~16:40; 500k finals do not exist)

All six arm trainers stopped at ~16:40 with tmux sessions removed and no note; last
checkpoints are at 250k (armA sd0-2, armB sd0). armA sd2's params_250000.pkl was
mid-write at the kill and is truncated — unrecoverable, so Arm A is a 2-seed number.
armB sd1/sd2 died before their first checkpoint and were relaunched to 250k
(`psmflow_paperfaith_armB_relaunch_20260831`, slowed ~10x by GPU contention; their
evals follow separately). Everything below is
**at-step-250k, not final** — but the shipped agent's own in-loop curves were flat by
250k, so these are read as strong signal, weak proof.

| arm | 500-ep success | seeds |
|---|---|---|
| Arm A (u'~p0 bootstrap, actor acting) | 0.189 ± 0.089 (0.196 / 0.182) | 2 |
| Arm B (psi(s,u,u'), no actor, gpi K=64) | **0.006** [0.002, 0.018] | 1 |
| Arm B floor control: same path, gpi_num_u=1 (random pick) | 0.080 [0.041, 0.150] (100 ep) | 1 |

- **Arm A = the pre-registered null.** The correct backup distribution alone lands
  inside the shipped agent's noise band (control 0.220 ± 0.037). A right backup does
  not create ranking signal.
- **Arm B anti-selects.** The K=1 control proves the eval path sound (0.080 ≈ the
  0.086 random-latent floor); letting the critic pick among 64 candidates drops
  success 13x below random. Probes on the same checkpoint agree:
  - D3 (adapted for latent index, `d3_q_landscape_armB_ep250k.json`): Q relative
    spread over 512 prior draws **0.0092** of |Q| — flatter than the shipped agent's
    0.011. H2 again.
  - D1-analog (`d1a_latent_ranking_armB_ep250k.json`, new
    `tools/diag_latent_ranking_oracle.py`): Spearman of psi-ranking vs FQL-oracle
    distances over the same K=128 candidates: **0.079** mean (old D1: 0.10, chance).
    The critic's argmax sits 0.359 from the oracle action when 0.086 was available
    in the candidate set — worse than the median candidate (0.265).
- Reading, per the pre-registration: at 250k the C=1 construction shows **no ranking
  signal and active anti-selection** — the failure mode survives the paper-faithful
  bootstrap and index. Caveats before calling it refuted: half-trained, one seed,
  and the E2 finding that gpi acting under-performs even for the shipped agent.
  The honest verdict gate is the relaunched-seed evals plus (if pursued) a 500k
  retrain; but nothing in these numbers points at the bootstrap/index substitutions
  as the missing ingredient, and everything continues to point at the critic's
  inability to rank latents — now measured directly against an oracle over the very
  candidate set it deploys on.

### E4 (same evening) — the ranking failure is the deployment scheme, not our critics

**E4a, the known-good-ranker control** (`e4a_fql_critic_aim_cube_sd0.json`): the E1
harness with one change — candidates scored by the frozen FQL expert's OWN critic
(ensemble mean, its own reduction) instead of oracle distance. Success **0.032**
[0.020, 0.051] — below the one-step random floor (0.086). The nuance that explains
it: per-step Spearman vs the oracle ranking is bimodal (mean 0.285, median 0.346,
54% of steps above 0.3, p10 −0.33) — the expert's critic ranks *moderately well on
most steps*, but argmax over K=512 reliably lands on its most overestimated
candidate: picked action 0.426 from the expert vs 0.105 available. Winner's curse at
K=512. Pre-registered branch: **decode-then-score is dead as a deployment scheme;
even a proven ranker fails under per-step best-of-K GPI.** This exonerates ψᵀw as
the specific culprit — lambda-rank, FB-graft, Arm B, and the FQL critic all fail
the same way — and kills the "critic with a direct action pathway" fix on its own
pre-registration.

**E4b, mixture-checkpoint probes** (`d1a_latent_ranking_mixhpo_ep500k.json`,
`d3_q_landscape_mixhpo_ep500k.json`): the mixture-trained 500k checkpoint shows
ranking Spearman **0.054** and Q spread **0.86%** of |Q| — the same band as the
point arm (0.10 / 1.1%) and Arm B (0.079 / 0.9%). As pre-registered: mixture
training does not create ranking signal; the mixture arm stays closed.

**Where this leaves the fork:** E1 (oracle 0.934) + E4a (every learned critic
fails at K=512 argmax) means the remaining live directions are (a) actor-based
improvement in latent space — no argmax over a large candidate set, the actor
moves smoothly against the (weak but locally usable) critic gradient, which is
where the eps=0.05 residual's 0.96 peak also lives; and (b) small-K or
regularized selection (K where the winner's curse is weaker than the ranking
signal — the E4a Spearman distribution says an optimal K may exist and is small).
Both are specced next-step candidates, not started.

### Housekeeping from the 16:40 incident

The teardown cause was `/var/local` filling (killed six runs; since cleaned to
47%). Repo policy going forward: experiment STORE on `/data-local`. The armB
sd1/sd2 relaunches to /var/local died of the same disk exhaustion at ~100k with no
checkpoint and are DROPPED — Arm A/B seed replication rides on the parallel
full-500k b-runs (`psmflow_paperfaith_arm{A,B}_20260831b`, 3 seeds each on
/data-local, ~300k/500k as of this entry); their Arm B results supersede the
250k sd0 numbers above when they land.



---

<!-- _class: lead -->

## 2026-08-30 — the agent we have been running is FB, not PSM; the preimage is exact for a decoder we never use

Two things came out of reading the ICLR draft against the code, and one port followed.

### The draft describes a method we are not running

`PAPER/ICLR` was re-added (skeleton: prewriting form + intro + preliminaries
+ a working-notes method section; `experiments.tex` and `related-work.tex` are empty, the
abstract is template text). Audited claim by claim against the run record:

- **Basis policies.** The draft defines `Pi = {G(s, u_0) | u_0 ~ p_0}` — fix `u_0` and hold
  it. That family was measured non-goal-covering on 08-05 (0/233 fixed-u policies reach the
  pointmaze goal). Everything that works uses per-step latent selection.
- **Preimage sets.** The draft's second contribution is identifying the DISTRIBUTION of
  latents decoding to the same action and augmenting the dataset with it. Against that: the
  width diagnostic on all three environments says the fitted mixture is a blurred point (far
  in `u` AND faithful in `a` peaks at 3.4% pointmaze, 0.7% antmaze). For that claim the cube
  is the open case — the 08-29 sweep found coverage@64 0.990 inside budget with near-flat
  Jacobian directions, which is the opposite reading. The Jacobian probe settles it.
- **No tuned regularizer.** The pitch is that the latent parameterization removes the
  pessimism/BC knobs. In practice the per-task result depends on a tuned residual budget
  (eps=0.05 -> 0.905 +/- 0.020 over 3 seeds; eps=0 -> 0.142 +/- 0.025 over 2) and the
  collapse forensics traced failure to a pessimism spiral.
- **Not in the draft at all:** the hybrid action critic + residual, which carries the only
  limited-coverage evidence we have (10% data: deployed 0.238 vs FB 0.030).

### `psmflow` is FB with a latent action space

`agents/psm.py` — the one pinned to the torch reference by weight transplant
(`tests/test_psm_agent_equiv.py`, 5 passed) — has two measure branches: `proto_loss` learns
the basis against a hash codebook of policies, and `sf_loss` fits the task head on a FROZEN
basis with ONLINE phi in the target ("matching the reference").

`agents/psmflow.py:96 measure_loss` has one branch: `M = psi(s,w,u) @ phi(s')^T`, ortho on
phi, phi trained by that same contrastive term, target from `target_psi` and `target_phi`,
`w` Gaussian mixed with `phi(next_obs[perm])`, `w = E[r phi]`. That is `agents/fb.py:49
_fb_loss_fn` line for line with F->psi, B->phi, a->u. No codebook basis, no affine `b`, no
constrained-LP inference.

This was a deliberate call — `docs/design/2026-07-20-psmflow-v1-design.md:89`: "One branch
only — PSM's proto/sf split collapses: the flow family IS the codebook." That held under
Rung-1, where the fixed-`u` family WAS the codebook. The 08-05 redesign moved policy
identity to `w` and made the bootstrap the actor's latent, which removed the codebook;
nothing re-derived the argument afterwards. The code is reference-faithful — to FB.

### The preimage is exact for a decoder the agent never uses

New measurement, 4096 rows of the cube HPO npz, same Stage-A checkpoint:

| latent | decode distance to `a`, ODE-100 | one-step decoder |
|---|---:|---:|
| stored point preimage | **0.00012** | **0.0886** |
| stored mixture mean | 0.1153 | 0.1495 |
| prior draw N(0, I) | — | 0.2855 |

Mean `||a||` = 0.875. The inversion solves `G_100(s,u) = a` and `precompute_preimages.py`
asserts `inversion.n_initial_steps == agent.flow_steps` to keep it exact — but every psmflow
run sets `gpi_decode: onestep`, so acting and training go through the distilled one-step net.
Under that decoder the "exact" latent misses by 0.0886, ~10% of action scale and comparable
to the mixture width the HPO spent 50 trials tuning (cube winner decode_mix 0.196). Still
3.2x better than a prior draw, so the latent carries real information — but the point arm is
NOT the noise-free control the point-vs-mixture ablation treats it as. Fix: invert at the
one-step decoder, or act at `flow_decode_steps=100`.

Two smaller integration findings:

- `utils/datasets.py:141-161` computes `next_noise_preimage` (u_0') on every batch and NO
  agent reads it — a Rung-1 leftover. In mixture mode it doubles the per-step mixture
  sampling (numpy Cholesky over B rows) on the training path. The `idx+1` pairing rule, its
  "incorrect at the end of a trajectory" warning, and the guard it forced into `main.py`
  all exist for that dead field.
- `preimage_valid` is computed and diverged rows are repaired (mixture -> prior, point -> 0)
  but the mask never excludes them from the loss: a point-arm row reset to `u=0` decodes to
  `G(s,0)` and trains as a legitimate pair. cube 13/1M, antmaze 881/1M.

What is solid: the pairing guards (env, ckpt realpath, epoch, `dataset_fraction` + seed,
`prior_scale > 0` whenever the mixture is read, first/last-1k row content check), `u_clip`
keeping online and target inputs on one support, per-visit mixture resampling, and `infer_z`
identical to the reference-verified `psm.py`.

### Ported: PSM into the latent action space (single latent)

`agents/latent_affine_psm.py` (commit `db96e48`). `LatentAffinePSMAgent(AffinePSMAgent)`
inherits the whole PSM — affine `M(s,u,x) = Phi(s,u,x).w + b(s,u,x)`, `WNet`, the codebook
basis branch, the constrained-LP `infer_w_goal` and closed-form `infer_w_zeroshot`, the
amortized actor — and substitutes the action slot for the flow's latent. Four overrides:
`_slot` (the clipped `noise_preimage`), `_slot_scale` (`u_clip`), `_emit` (decode through
the frozen flow — the only place an action exists), and `_codebook_table` (codebook policies
are latents drawn from the flow's own N(0, I) prior, still keyed on the row hash so `pi_z`
varies with the state and is NOT the fixed-u family).

`agents/affine_psm.py` gained those four seams and routes all nine action-slot reads through
them; its defaults leave behaviour unchanged, and all five affine test modules still pass
(smoke 6, networks 3, inference 5, flow 9, factored 8). New `tests/test_latent_affine_psm.py`
(7 passed) includes the load-bearing one: negating `batch['actions']` leaves the measure loss
bit-identical while shifting `noise_preimage` moves it. End-to-end smoke on the real cube npz
ran 200 steps plus LP inference, actor distillation and eval.

Open on the port: no HP tuning (it inherits affine PSM's raw-action cube values);
`inference.mode` defaults to `full`, i.e. a 5120-step LP at every eval, untested in latent
space against `zero_shot`; and no BC control yet, without which no number is quotable.

### The Jacobian probe refutes the 08-29 structural claim

`tools/diag_flow_jacobian.py` (new): singular spectrum of dG(s, .)/du at each transition's
OWN preimage — the point the inversion solved for, so the numbers describe the region the
mixture is fitted in. 2048 rows per environment, `flow_steps=100`, reports at
`/data-local/amsks/PSMFLows/logs/jacobian_{cube,antmaze}.json`.

| | cube (d_a 5, mean ||a|| 0.87) | antmaze (d_a 8, mean ||a|| 1.99) |
|---|---:|---:|
| sigma pooled, median | 0.083 | 0.201 |
| sigma_min, median | 0.068 | 0.050 |
| sigma_max, median | 0.117 | 0.325 |
| condition number, median | 1.79 | 6.67 |
| free radius in u for a 0.05 action move | 0.73 | 1.00 |
| rows dropped (inverse diverged) | 0 / 2048 | 5 / 2048 |

The 08-29 entry read cube's 0.068 as evidence of near-flat directions and inferred antmaze
must be near-bijective. **Both halves are wrong.** Cube's condition number is 1.79 and its
spectrum runs 0.068-0.117: a near-ISOTROPIC contraction by ~0.1 with no null direction —
the same object the 08-14 width diagnostic described ("a near-uniform contraction sigma
~ 0.07, which is injective"). And antmaze's smallest singular value is SMALLER than cube's
with a LARGER free radius, so locally antmaze has more latent slack per unit of action
change, not less.

The cube/antmaze split therefore is NOT local injectivity, and "cube has a preimage set,
antmaze has a point" should not be written down. Coverage@64 is a global property — where
prior draws land relative to the data — and antmaze differs on two axes this probe cannot
reach: 8 action dims instead of 5, and actions of twice the norm, so a fixed decode budget
is spread thinner. Whatever explains antmaze's 0.115 coverage and ESS 7/128 is still open;
the Jacobian is now excluded, and lead 2 of the 08-29 entry is closed negative.

### In flight

- **Cube mixture arm**, 3 seeds, `psmflow_mixture_hpo_20260829`, tmux `mix_sd0/1/2`: the
  first runs to use the HPO npz with `use_point_preimage=false`. Point arms of the old and
  new npz are bit-identical, so the 5-seed 0.236 +/- 0.071 is an exact control. Prediction on
  record: no better than 0.236, because mixture draws sit 1.38 in `u` from the point inverse.
- **Antmaze precompute**, queued behind them (tmux `pre_antmaze_queued`) at the best trial of
  the 50 (alpha 11.55, prior_scale 0.489, n_steps 5): ~13.5 h. Run to be safe, but its own
  sweep says coverage@64 0.115 and decode_mix 0.263 vs decode_pt 0.0002, against a working
  point-arm number of 0.220 +/- 0.005 (3 seeds, BC control 0.090).

### Also

- `PAPER/` untracked from `main`, `feat/psm-integration` and `feat/inversion-integration`
  (tip removal only; history keeps it), then the ICLR skeleton re-added — it builds clean
  with latexmk, 6 pages, one undefined citation (`wagenmaker`).
- `utils/log_utils.py`: wandb 0.29 rejects `start_method` / `_disable_stats` through
  pydantic, so every run died in `setup_wandb` before training. Commit `d634b63`.

---

<!-- _class: lead -->

## 2026-08-29 — inversion HPO: cube solved and precomputed; antmaze structurally closed to width tuning

Branch `feat/inversion-integration` (= main + Claas's `tuning_inversion` merge + HPO work).
Three unpushed commits: `643d4d7` (hypersweeper+SMAC harness), `75fa57d` (typicality hinge
+ per-trial json fix), `7d635a4` (jittered Cholesky for near-singular EM covariances —
required or every antmaze trial crashes).

**What was built.** `tools/hpo_preimage_inversion.py` — single-trial SMAC target derived
from Claas's `tools/tune_preimage_inversion.py`: inverts one fixed neighbourhood batch
(same batch + draw seed across trials), returns
`cost = −coverage@k64 + 10·max(0, decode_mix − budget) + 10·max(0, E‖u‖²/d_a − 1.1)`.
Budget rule: 0.1 per action dim (cube 0.22, antmaze 0.28), config keys
`hpo_decode_budget` / `hpo_typicality_ratio`. Sweeper config `configs/hpo_preimage.yaml`
(SMAC BlackBox BO, 50 trials; alpha log [2,100], prior_scale [0.3,1.0], n_steps [5,20];
num_samples pinned 128, num_clusters 1). ~65 s/trial cube, ~127 s antmaze.
The typicality hinge exists because the first cube sweep's incumbents cheated: prior_scale
0.30 bought coverage 0.999 at E‖u‖² ≈ 8 vs expected 5 (mixture leaves the prior).

**Cube result (sweep dir `/data-local/amsks/PSMFLows/hpo/cube_20260828_000356`, log
`cube_launch.log`, one JSON line per trial).** Feasible region comfortably non-empty:
43/50 trials inside budget, 17 also typical. Winner after typicality filter:
**alpha 14.1, prior_scale 0.69, n_steps 14** — coverage@64 0.990, decode 0.196,
E‖u‖² 5.06, ESS 58/128. Beats the hand grid (0.96 @ 0.227, over budget).
Full 1M-row precompute at exactly that setting is **DONE**:
`/var/local/amsks/exp/PSMFLows/preimages_cube_hpo_a14p1_ps0p69_ns14_N128.npz` (+ meta),
13/1M diverged rows reset to prior (flagged in `preimage_valid`).
**Next pending step: recovery tests on that npz** (`tools/validate_decode_recovery.py`,
`tools/validate_dynamics_recovery.py`, or `scripts/run_recovery_tests.sh`), then it can
replace the shipped cube npz for mixture-mode (`use_point_preimage=false`) runs.

**Antmaze result (sweep dir `.../antmaze_20260828_020557`, log `antmaze_launch.log`) —
the important one.** Ckpt: only **antmaze-medium** exists
(`bcflow_antmaze-medium-navigate_20260805_014546/sd000` @ 500000; no -large ckpt),
env `antmaze-medium-navigate-singletask-v0`, d_a=8. Over 50 BO trials:
best coverage inside budget **0.115**; best coverage of ANY trial, budget ignored,
**0.24**; ESS 6–7/128 across every setting. Conclusion, now with search-based evidence
rather than a grid: **no width setting makes the antmaze mixture usable — the conflict is
structural.** The cube flow has near-flat Jacobian directions (median singular value
0.068), so a genuine wide preimage region exists and tuning finds its width; the antmaze
flow is near-bijective, the preimage is a point, and widening it is label noise by
definition. **[SUPERSEDED 2026-08-30: the Jacobian probe refutes this. Cube's 0.068 is its
sigma_MIN; the spectrum is 0.068-0.117, condition number 1.79, i.e. an isotropic contraction
with no flat direction, and antmaze's sigma_min is SMALLER at 0.050. The conclusion that no
width setting works on antmaze stands on the sweep; the structural explanation offered for
it does not.]** This refutes the "re-run with a re-tuned alpha" hope in
`scripts/upload_preimages_hf.py` and explains why antmaze runs use
`use_point_preimage=true` (mixture is the default elsewhere — `psmflow.yaml:95`).

**Open leads, in order.**
1. Recovery tests on the new cube npz (above) — blocks adopting it.
2. Jacobian probe: singular spectrum of dG/du at data points, antmaze vs cube. Cube's
   0.068 is on record; if antmaze's is ~10× larger, the structural claim is quantified.
3. The actual fix direction: retrain the antmaze flow so preimage SETS exist — extra
   latent dims beyond d_a, or a decoder noise floor / entropy regularizer — then re-run
   the same sweep. Tuning the inversion harder is ruled out; don't respend there.
4. Housekeeping: stray `hpo_trial.json` + `smac3_output/` in repo root are pre-fix sweep
   droppings (safe to delete / gitignore); the three commits above are unpushed.

---

<!-- _class: lead -->

## 2026-08-14 — the residual is DATA-DEPENDENT: it carries the low-data regime and costs on full data

Full plan, every launch and every number: `docs/plans/2026-08-14-priority-stack.md`.
Tables regenerate from the eval JSONs with `tools/make_tables.py`
(`docs/tables/results.md`, `PAPER/ICLR/tables/*.tex`).

### The result that reorganises the story

Every hybrid checkpoint is now evaluated in BOTH acting modes — deployed (actor draw +
ε-residual) and decode-only (`agent.action_critic.eval_rank_k=1`: same draw, no residual,
no selection). The residual's sign flips with dataset size:

| data | deployed | decode-only | plain PSMFlow | residual effect |
|---|---|---|---|---|
| 10% | **0.238** | 0.072 | 0.068 | **+0.166** |
| 50% | **0.228** | 0.070 | 0.052 | **+0.158** |
| 100% | 0.162 ± 0.168 (4 seeds) | 0.226 ± 0.068 | 0.236 ± 0.071 | **−0.065** |

At 100% the residual helps on one seed of four (+0.102 on sd1) and hurts on the rest
(−0.218, −0.126, −0.018), so the flagship 0.302 was the seed where the coin landed heads;
the deployed arm's ±0.168 interval is the real headline. At 10–50% the opposite holds and
the residual carries everything — decode-only sits exactly on the BC floor. At 10% the
hybrid is also the best zero-shot method measured (0.238) while **FB collapses to 0.030**,
below the BC control. One seed per fraction cell: replicate before claiming.

Also: λ-rank acting (K=32 decoded candidates ranked by Q_a, no residual) is WORSE than
decode-only on both seeds (0.004 / 0.162), so Q_a's ordering over reachable actions is not
merely uninformative. 1M did not pay off (fresh 1M seeds 0.134/0.162; extending sd1
500k→1M moved 0.302→0.284). HP-matching to FB buys nothing (0.276 / 0.192 vs 0.236 ± 0.071).

### Two paper-bound claims failed their re-check

1. **The radius claim was vacuous** — `clip(N(0,1), ±r)` never samples wider latents. With
   scaled draws, coverage rises 0.63 → 0.99 → 1.41 at 1/1.5/2×, i.e. it DOES reach the
   split-half reference, while the distance to FQL's action gets WORSE (0.19 → 0.23 → 0.28).
   `note.tex` corrected in both places (body + figure caption).
2. **C1's "FQL is off-support" had no yardstick.** Measured against the data-matches-itself
   baseline it was missing: a_FQL sits 0.537 from its k-NN actions while real data sits
   0.582 from its own (p95 1.165), and only 6.3% of FQL's actions exceed that p95. FQL is as
   data-like as the data; what is anomalous is our decode at 0.187, ~3× TIGHTER. The
   capacity/retrain arm was cancelled on the uncalibrated comparison. Verdict string left
   unchanged in the tool, baseline now recorded beside it — this one still needs a call.

### The preimage pipeline had a real bug (user-flagged, confirmed)

The inversion target was π(u) ∝ exp(−α‖G(s,u)−a‖) with **no N(0,I) prior factor**, so it is
flat wherever the decoder is insensitive and the EM fit runs away (covariance eigenvalue
1→6→34→305→2281→6095 over 8 steps; in the shipped npz files 82–84% of cube rows and 50% of
pointmaze rows are fitted WIDER than the prior, worst case 3.8e4, means to |μ|=206). The
exact point preimages in the same files are prior-like (|u| mean 0.85), which identifies the
target rather than the inverter. Fixed via `inversion.prior_scale` (default 1.0; 0.0
reproduces legacy files) — on the real cube flow this puts 0% of rows outside the prior and
lifts ESS 87 → 119 of 200.

**But mixtures still are not worth switching to.** Sweeping α (`tools/diag_preimage_posterior_width.py`)
shows the posterior's width IS the temperature, not decoder degeneracy: width, distance to
the point inverse and decode error all shrink together (α=20/100/500 → per-dim var
0.49/0.093/0.020, decode error 0.165/0.045/0.009), and samples only become faithful once the
fit has collapsed onto the point inverse. The decoder is a near-uniform CONTRACTION
(σ ≈ 0.07), which is injective — hence a point preimage — while still compressing the
reachable action set 3× below the data's dispersion. Historical mixture arms trained on
latents decoding 53% of an action scale away, so that ablation was never fair.

### Everything else that landed

- eval reproducibility: `evaluate` pinned the env's init RNG but drew ACTION noise from OS
  entropy regardless of `seed`; now pinned, relabel batch seeded, false docstring fixed.
- one k-NN protocol (`utils/geometry.NeighbourIndex`, standardized obs, k=32, exact
  self-exclusion) shared by the three support probes; raw-geometry numbers kept, labeled.
- ψ_a pessimism is now scalar-Q (blend toward the least-task-valued ensemble member);
  the old per-feature penalty was sign-indefinite in Q-space and RAISED Q where w < 0.
  Bit-identical at λ=0, i.e. every run to date.
- 2a Q-gap probe + calibration penalty fix (actor knob, |q0−q1| not |q0−q1|/2): prior
  collapse verdicts UNDERSTATED the over-estimation; directions unchanged.
- pairing guard now records `dataset_fraction`/seed in the npz sidecar and spot-checks the
  first/last 1k observation rows; `restore_agent` tolerates checkpoints predating a field
  (the hpmatch checkpoints could not otherwise be loaded at all).
- eval reports are self-identifying (`acting_mode`, `action_critic`, fraction, flow/npz
  paths): three acting modes per checkpoint were distinguishable only by FILENAME.
- **P2 FB-graft FAILED its gate (sd1, 1M): deployed 0.064 [0.046, 0.089] against a 0.45
  gate — at the BC control, below the hybrid.** ψ_a got its own backward map B_a trained by
  the FB measure loss and its own w_a inferred from B_a; the residual got WORSE
  (decode-only 0.174 → deployed 0.064). Its decode-only number sits inside the hybrid's own
  decode-only spread, confirming the graft left the latent actor alone as designed — so the
  implementation did what it claimed and the claim did not help. Pre-registered rule
  applied: shared-φ was NOT the explanation for the weak action critic; report, do not
  iterate. sd0 still running only to give the negative result n=2.
- **Run the test suite module-per-process** — all 29 modules pass (177 passed, 1 skipped),
  but the whole suite in ONE process dies in the XLA CPU compiler at a moving victim
  (pre-existing; reproduced with every one of today's tests excluded).

---

<!-- _class: lead -->

## 2026-08-13 — the residual dial WORKS (0.90–0.91 at 500 ep, 3 seeds); the collapse is a PESSIMISM spiral, not optimism

The W4/l1stab arc, verified end to end. All artifacts in
`/data-local/amsks/PSMFLows/logs/`.

### Peak checkpoints hold at 500 episodes

`eval500_l1stab_*_peak*.json`, summary `eval500_l1stab_peaks_summary.json`:
ε=0.05 → **0.910 / 0.910 / 0.896** (sd2@75k, sd3@50k, sd5@50k); ε=0.1 → 0.888 / 0.850.
Anchors: ε=0 final 0.140/0.144 · BC 0.068 · FB 0.716–0.730 · FQL 0.949. A 5%-of-action-
scale residual budget recovers ~95% of the per-task ceiling. The w4 finals table
(`eval500_w4_res*_final.json`): ε=0.1 sd0 survivor **0.820** [0.784, 0.851]; everything
else collapsed (0.000–0.144).

### The collapse mechanism is pessimism-driven UNDERestimation

`diag_calibration_collapse.json` (new `tools/diag_fql_calibration.py`), same-run
peak-vs-collapsed pairs (res0.05 sd2 75k/250k, sd5 50k/300k): at the peak Q is mildly
optimistic (bias **+6.6/+8.8**, Spearman 0.14–0.25); at collapse Q has crashed −49 →
−105/−112 while realized return only fell −56 → −87 — bias **−18.7/−25.0**. Exact-min
pessimism (0.5 backup + actor) compounds as ensemble disagreement grows off-data. The
in-flight `l1_pess0_e010` wave (pessimism 0.0, ε=0.1, seeds 2–4) is the mechanism test.

### The winning policies barely leave the data (regime verdict)

`dist_res*.json` (new `tools/diag_action_distance.py`): executed-action distance to
k=32 NN dataset actions, vs the data-matching-itself baseline (median 0.520, p95 1.036).
ε=0.05 peak policies: median **0.504–0.509** — AT data-noise distance — with only
10–12% of steps beyond the baseline p95. ε=0.1: median 0.710, 23% beyond. Pre-registered
reading: improvement arrives while actions stay behavior-close ⇒ **decoder undercoverage
(Hypothesis A) was the binding cost**; pushing further off-data buys nothing more. NB
this also recalibrates C1: real data actions sit ~0.52 from their own neighbours, so
FQL's 0.577 is near data-noise level under the matched statistic — the "off the data"
framing in note.tex needs this baseline context.

### Also

- **Inversion re-cert PASSES** (`audit_inversion_recert.json`): ESS 81.3/94.6/7.6,
  roundtrip 1e-4/1e-4/2e-4, invalid 13/0/881 — all matching the HF card.
- Peer bar: PSM cube 3-seed 500-ep spread **0.056 / 0.156 / 0.532** — quote the spread,
  never a mean (n=3 CI is wider than the range).
- Audit 0a/0b (commit `1df9594`): FB run-config clean, 0.72 stands; HP shortlist top
  suspect is pessimism 0.5-vs-0.0 — now directly implicated by the collapse forensics.
- T1/T2 flow tests (`diag_preimage_sampling_*.json`, `diag_generated_pair_support_*.json`)
  — see those JSONs for the mixture-arm decode fidelity and the generated-pair support
  verdicts.
- New tools: `diag_fql_calibration.py`, `diag_action_distance.py`,
  `diag_preimage_sampling_fidelity.py`, `diag_generated_pair_support.py`.

NEXT: read pess0 (does removing pessimism stop the collapse?); if yes, the stabilized
recipe (ε=0.05, pessimism tuned down, or early-stopping on the calibration bias signal)
is the port target for zero-shot via the w-conditioned action critic (Idea 1).

---

<!-- _class: lead -->

## 2026-08-12 (L0) — the ordering survives, but it is 1.7x and not 3x

`logs/diag_action_coverage_cube_robust.json`, `tools/diag_geometry_robustness.py`.

C1 compared a minimum over **512** decodes against a minimum over **32** neighbours. That
is a combinatorial advantage, not a geometric one. Matching the candidate counts:

| | k=16 | k=32 | k=64 |
|---|---:|---:|---:|
| d_flow, 512 candidates (C1's number) | 0.183 | 0.183 | 0.183 |
| d_flow, matched to k | 0.370 | **0.345** | 0.296 |
| d_data over k neighbours | 0.649 | **0.577** | 0.521 |
| ratio (matched) | 1.75 | **1.67** | 1.76 |

**The ordering survives in all six geometry x k settings (raw and per-dim standardized
observations), ratio 1.64-1.80 — but the magnitude roughly halves, from ~3x to ~1.7x.**
Quote the matched pair (0.577 vs 0.345 at k=32), never C1's 512-vs-32 pair.

The conditioning asymmetry that matching CANNOT remove: d_flow conditions on exactly s,
d_data on a neighbourhood of s. So the defensible claim is "the decoder interpolates at
least as close to the task-optimal action as nearby data does", not a statement about the
behaviour conditional at s.

Everything downstream of C1 stands: the winning policy is still off-support, the capacity
arm stays cancelled, W3 stays retired. Only the number changes.

Two smaller reads: coverage is mildly k-dependent (0.38-0.48) and its split-half reference
moves with it (0.92-1.11), so coverage must always be quoted relative to that reference.
And the coverage statistic is numerically unstable at k=16 under standardized observations
(one cell read 364 — an action dim with near-zero sd in a 16-neighbour set); use k>=32 for
coverage, the distance statistics are unaffected.

note.tex updated: the abstract's "3x" is now "1.7x", Hypothesis B carries the matched
numbers and the residual asymmetry, and the provisional-geometry caveat is resolved.

---

<!-- _class: lead -->

## 2026-08-12 — the interface ceiling is a SUPPORT fact, not a decoder fact

Chain: P0 Branch B → differentiation probe → P2 ceiling → W1 coverage → W3 gate failure →
C1. Reports in `logs/diag_action_coverage_cube{,_fs100,_c123}.json`.

### C1 — the reframe. Read this before quoting any interface number.

| cube, 64 states, normalized by action scale | |
|---|---:|
| a_FQL → the dataset's OWN 32-NN actions | **0.577** |
| a_FQL → our best decode (512 candidates) | **0.187** |

**FQL's actions sit 0.577 from the dataset's own nearest neighbours while our decoder gets
within 0.187 of them — the flow interpolates closer to the winning actions than the
empirical data itself does.** Therefore 0.187 was never a decoder failure, and no retrain
could have closed it: the actions FQL uses are not in the distribution being cloned.
Capacity/retrain arm **cancelled** (automatically, by the pre-committed rule); **W3
retired**.

This is easy to garble later, so state it in this order: (1) the winning policy is
off-support, (2) by 3x more than our decode gap, (3) hence the ceiling is a property of
the data + support constraint, not of the flow.

### W1 / W3 / C2 / C3, briefly

- **W1:** coverage 0.415–0.418 across all six decode × radius settings on the deployed
  flow. ODE gain +0.002, radius gain +0.001 — doubling the latent radius changes nothing,
  so the decoder saturates rather than being clipped too tightly. W2 arms A and B both
  eliminated without training either.
- **W3:** retraining Stage-A at `flow_steps=100` lifted coverage 0.42 → **0.63** — the
  07-29 caveat about the fs10 checkpoint was correct — but left the FQL-action distance at
  0.187, unmoved to three decimals. Gate (<0.10) failed; the flow was NOT inverted.
- **C2:** residual spreads evenly across all 5 action dims (max 23%) ⇒ blanket ε.
- **C3:** aleatoric coverage ceiling 0.936, so fs100 is at **67% of achievable**, fs10 at
  44%. Coverage was being judged against the wrong reference (1.0).
- Tool debt closed: the ODE step count now reads the flow's `flags.json` (fixing the
  `gpi_decode=ode` step mismatch for all envs), and the knob-sweep verdict is scoped to
  the flow under test — the fs100 report had reprinted the fs10 recommendation verbatim.

### Peer bar, 500 episodes — PSM does NOT survive its in-loop numbers

PSM cube in-loop finals were 0.38 / 0.02 / 0.58; at 500 episodes sd0 reads **0.156**
[0.127, 0.190] and sd1 **0.056** [0.039, 0.080] (sd2 pending). Do not quote the in-loop
numbers. FB at **0.716–0.730** remains the real peer bar; BC control 0.068.

### In flight

W4 residual sweep on the instrument: ε ∈ {0, 0.05, 0.1, 0.2, 0.4}, 2 seeds each, tmux
`w4_q0` / `w4_q1`. The W4 critic scores executed ACTIONS, not latents — a residual head
gets no gradient from a latent critic — so ε=0 is its own anchor arm and the P2 numbers
are NOT its left endpoint. Gate: any arm ≥ 0.40 at 500 episodes ⇒ port to LatentFlowPSM.
Per-arm off-support budget (executed-action distance to k-NN dataset actions) is the
x-axis of the headline tradeoff figure, not a side stat.

---

<!-- _class: lead -->

## 2026-08-10 (evening) — P0 says BRANCH B: FB's basis is WORSE than ours, and it still wins

`docs/plans/2026-08-10-psm-fix-roadmap.md` P0, the deciding probe. Report:
`logs/p0_fb_basis_probe.json`. New tool `tools/diag_fb_basis_probe.py`.

| cube, 500k | linear reward read-out (ridge R^2, held out) | Q relief (spread / |Q|) | 500-ep success |
|---|---:|---:|---:|
| LatentFlowPSM (our phi) | **0.129** | 0.011 | 0.236 |
| FB (3 seeds) | **0.069** | 0.023-0.031 | **0.716 / 0.730 / 0.716** |

**FB's backward map carries LESS reward signal than our phi -- at z_dim=50 against our 128,
so it is worse per-dimension too -- and FB still scores 3x our success.** The premise
behind P1 (copy FB's measure objective so the basis carries the task) is therefore not
supported: the property we were going to import is one FB does not have either.

FB's Q relief is 2-3x ours (2.3-3.1% vs 1.1%) but nowhere near the >=5% Branch-A bar, so
that is not the explanation on its own either.

**Consequence, per the plan's pre-registered decision rule: P1a (the loss swap) is NOT
indicated; LatentFB (P3) is promoted to the primary path, and the PSM-internal work
narrows to P1b (backup exploration + actor unshackling).** The differentiator has to be
the improvement loop -- FB's actor optimizes F^T z from step one against a critic trained
on ITS OWN visitation -- not the reward read-out of the basis.

Caveat worth carrying into the paper: R^2 of a linear read-out on a 2%-sparse reward is a
weak instrument in absolute terms for both methods. The comparison is meaningful because
it is like-for-like; the absolute levels are not evidence that either basis is "good".

### Peer baselines, 500 episodes

| cube-single | success | 95% CI |
|---|---:|---|
| BC control | 0.068 | [0.049, 0.093] |
| LatentFlowPSM (5 seeds pooled) | 0.236 | [0.219, 0.253] |
| **FB sd0/1/2** | **0.716 / 0.730 / 0.716** | [0.675,0.754] / [0.689,0.767] / [0.675,0.754] |
| FQL per-task | 0.949 | [0.936, 0.960] |

FB is a ZERO-SHOT peer, not a per-task topline: same env, dataset, budget, flow actor, no
reward at train time. This is the bar, and we are far from it. Antmaze LatentFlowPSM is now
3 seeds (0.22 / 0.26 / 0.22 in-loop @500k) vs its 0.090 control.

### In flight

- `abl_be_sd{0,1}`: backup exploration `backup_explore_frac=0.5`, cube, 2 seeds (P1b).
  Implemented config-gated with a test pinning that the default 0.0 path is BIT-IDENTICAL
  (the extra RNG keys are split inside the branch, so published runs stay reproducible).
- PSM cube sd1 finishing, sd2 queued; 500-ep PSM evals after.

---

<!-- _class: lead -->

## 2026-08-10 (later) — diagnosis: the gain is in-support latent BC + pessimism, not value improvement

Ran §B of `docs/plans/2026-08-10-critic-diagnosis-and-baselines.md`. Three diagnostics,
three verdicts, and they agree. Reports: `logs/d1_policy_ranking.json`,
`logs/d2_task_projection_{cube,antmaze}.json`, `logs/d3_q_landscape_cube.json`.

### D1 — H0 REJECTED. The critic cannot rank policies either.

Ten cube policies spanning 0.068–0.320 measured success (5 final seeds, sd2 at
50k/150k/250k/400k, and the BC prior), all scored by ONE frozen representation (sd0) at 64
seeded start states. **Spearman 0.10 (permutation p = 0.78).** Predicted values span
−1816 to −1832 — a **0.9% spread** — while realised success varies 4.7×. So panel (c) of F4
was not a confounded diagnostic: the critic is uninformative at the ranking job too.

### D2 — H1 RELOCATED. Inference is fine; the basis is weak.

Closed-form w = E[r·φ] achieves R² **0.129** on held-out cube transitions against a ridge
topline of **0.127** on the same features — the estimator extracts everything φ contains,
so task inference is NOT lossy. But the topline itself is low: the best linear read-out of
φ explains ~13% of reward variance. Antmaze is the same (0.135 / 0.134), so **no
cube-vs-antmaze asymmetry** — H5 (horizon myopia) gets no support from this. w_inf is
stable across disjoint relabel batches (min pairwise cosine 0.93). Fix direction, if this
is pursued: a φ-grounding auxiliary, not a better estimator.

### D3 — H2 SUPPORTED, decisively. H3 not supported.

At 64 states, Q over 512 prior latent draws has relative spread **0.011 of |Q|** — flat.
Around the actor's own latent it is flatter still (0.0038). Two details that settle the
story:
- **The actor sits at the 44th percentile of the prior-Q distribution** (median 38th). If
  Q meant anything the actor would be near the top; it is below the middle.
- **Actor gradient is BC-dominated 5:1**: ‖∇q-term‖ 0.168 vs ‖∇distill‖ 0.839.
  (bc_flow reads 0.0 w.r.t. actor params by construction — it only touches the vf.)
- **Dispersion does NOT collapse** over training (‖u‖² 4.66 → 4.54 across 50k–500k), so
  H3's co-collapse signature is absent. The flatness is not a shrinking-support artifact.

### What this means

The honest reading: **LatentFlowPSM's 3.5× over BC is explained by in-support latent
behaviour cloning plus ensemble pessimism, not by value-driven improvement.** Every piece
fits — flat curves after 50k, chance-level calibration, GPI *below* the BC control (argmax
over a flat noisy Q is worse than sampling), and an actor whose gradient is 5:1 imitation.

Per the plan's decision tree §E this is the "flat-Q" branch, and its own C-tier gating says
bc_coeff↓ runs only if "D3 shows bc-term dominance **AND** Q has relief". Q has no relief
(1.1%), so **the bc_coeff sweep is not indicated** — do not run it reflexively. The
remaining candidate is backup exploration (mix p0 latents into the bootstrap), which
targets the flatness itself rather than the actor's weighting.

### In flight

Baselines from §D, three tmux queues (`base_q0/1/3`, ~2.5–4 h): PSM cube ×3 seeds
(`psm_cube_flow_20260810`), FB cube ×3 (`fb_cube_ortho1000_20260810`, ortho_coef=1000 —
the reference value, repo default is 1.0), antmaze LatentFlowPSM seeds 1,2. R3 is the one
that matters for framing: if action-space PSM matches 0.236, the headline narrows to the
support/coverage story.

---

<!-- _class: lead -->

## 2026-08-10 — LatentFlowPSM measured end to end; ICLR figures F1-F4

Executed `docs/plans/2026-08-10-iclr-figures.md`. Everything below is a 500-episode
evaluation with a Wilson interval, quoted beside the per-step-prior BC control from the
same frozen flow. Reports in `/data-local/amsks/PSMFLows/logs/eval500_*.json`; every
figure has a sidecar JSON in `PAPER/ICLR/figures/data/`.

### Headline: the actor works, the value function still does not rank

| cube-single, 500 ep | success | Wilson 95% |
|---|---:|---|
| BC control (frozen flow, per-step prior) | 0.068 | [0.049, 0.093] |
| flow-GPI over fixed u (Rung 1) | 0.015 | 50-ep in-loop |
| LatentFlowPSM `acting=gpi` | 0.055 | [0.047, 0.064] |
| **LatentFlowPSM `acting=actor`** | **0.236** | [0.219, 0.253] |
| FQL per-task (alpha=300) | 0.949 | [0.936, 0.960] |

Across 5 seeds the actor is **0.236 ± 0.071** (mean ± 95% CI); per-seed 0.318 / 0.236 /
0.176 / 0.260 / 0.188. Antmaze: LatentFlowPSM **0.222** [0.188, 0.261] vs BC **0.090**
[0.068, 0.118]. Pointmaze BC control is **0.002** — the env is near-unreachable from the
prior, which is why pointmaze zeros never discriminated between methods.

**Three findings, in order of how much they should change what we do next.**

1. **The F1 prediction landed to three decimals.** Fixed-u flow-GPI scores 0.015 against a
   measured family ceiling of 2/128 = 0.0156. GPI was extracting exactly what the family
   contained — the 08-05 root cause is now quantitative, not argued.
2. **`acting=gpi` (0.055) is BELOW the BC control (0.068)**, non-overlapping intervals, on
   the *same* representations the actor uses. So psi is a usable training signal for the
   actor but not a reliable ranker of latents. The Finding-3 weakness survived the
   redesign; it moved rather than disappeared. This is the open problem now.
3. **FQL reaches 0.949 with alpha=300** — the FQL reference's per-env value; the repo
   default of 10 would have understated the topline badly. LatentFlowPSM recovers ~25% of
   per-task performance, and its curve is flat from 50k, so it is converged, not
   budget-limited.

### What was run

- **WP0**, 17 evals: cube sd0-4 x {actor, gpi}, antmaze sd0, BC controls for all three
  envs. `scripts/eval500.sh` wraps the invocation (psmflow and `bc` modes).
- **WP1**, cube FQL 3 seeds, `fqlbaseline_cube_a300_20260810`. **34 min per seed**, not the
  3-4 h the plan budgeted — the tqdm ETA during JIT warmup is what mislead the estimate.
- **D1 re-runs**, 6 (3 envs x trained/random). Pointmaze reproduced the historical numbers
  exactly (MMD 3.0e-4 vs 4.0e-3).
- **Figures** `tools/fig_{reachability,flow_fit,actor_comparison,policy_anatomy}.py` +
  `tools/figstyle.py` (Okabe-Ito order, run through a colorblind validator).
  `tools/viz_policy_rollouts.py` records per-step (state, actor latent, action, reward)
  and the initial-state predicted score.

### Corrections to the plan, from the data

- Antmaze family ceiling is **5/64 = 8%**, not the ~4% the plan cites. Figures compute it
  from the JSON rather than quoting prose.
- The Rung-1 GPI bar **cannot** get a 500-episode rerun: that checkpoint predates the
  latent-PSM redesign, so its parameter tree will not load in the current agent. F3 uses
  its in-loop 50-episode late window and labels the bar as such.
- Calibration needs **AUC against binary success**, not Spearman on returns: every failed
  episode returns the identical value, so most of the sample is one tie group and
  arbitrary tie-breaking manufactures a correlation out of array order.

### Finding 5: the measure does not rank, measured a second way

`tools/viz_policy_rollouts.py`, 60 episodes x 3 cube seeds + 40 antmaze:
- **Support claim holds, with room to spare.** Actor latents mean ||u||^2 = 3.87 (cube,
  d_a=5) and 5.64 (antmaze, d_a=8) — narrower than the prior AND narrower than the dataset
  latents (5.59 / 8.49), 0% on the clip boundary. Every executed action is a decode of a
  typical latent.
- **The value does not predict the outcome.** Predicted psi(s,w,u)^T w at s0 separates
  success from failure with AUC **0.588 [0.480, 0.694]** on cube (180 ep, 33 successes) and
  **0.354 [0.168, 0.567]** on antmaze. Both cover chance. Bootstrap CI, seeded.

So the GPI-below-BC result is not an artifact of the argmax: the critic genuinely carries
little outcome information. The unexplained part is why DPG on it still produces a policy
3.5x better than its behavior prior. That is the next thing to isolate.

### Open

- Why does GPI underperform the prior? If psi cannot rank latents at a state, the
  representation is not doing the job the method claims for it, and the actor is
  succeeding for a reason we have not isolated.
- Antmaze is 1 seed. Cube seed spread (0.176-0.318) is wide enough that 5 seeds pins
  "beats BC" but not the level.
- Antmaze mixture-arm preimages remain unusable (ESS 7.6, 6% > 20); alpha was tuned at
  d_a=5. Point arm is what every run uses.

---

<!-- _class: lead -->

## 2026-08-05 — ROOT CAUSE: the fixed-u family is structurally non-goal-covering; Rung-1 is dead on navigate data

All Stage-C variants (pointpre x2, mixture, pointpre1M, mix0 audit — 11 runs) read **0.0
success** on pointmaze task1; cube Stage-C floors at 0.02–0.06. D1–D3 pass. Root cause
below. **It is not a bug anywhere — the policy family itself has no goal-reaching
member, so even a perfect psi gives GPI a flat landscape.** Three measurements:

### 1. Exhaustive reachability: 0 of 233 latents ever reach the goal

`tools/latent_reachability.py` (new). d_a=2 ⇒ a 13x13 grid covers the ENTIRE box
[-3,3]^2 — no sampling escape — plus 32 dataset preimages + 32 preimages of actual
goal-reaching transitions, each rolled 2 full 1000-step episodes (kills the 200-step
confound in `calibration_check`). **num_success_any = 0.** 75% of latents never get
closer than ~23.5 (maze is ~30 across); best is 1.32 at the saturated corner u≈[2,2.5].
Goal-transition preimages do NO better than random ones — a preimage decodes to its
action only in its own state. Report:
`/data-local/amsks/PSMFLows/logs/latent_reachability_pointmaze.json` + `latent_reachability.png`.

### 2. The expert's route is latent WHITE NOISE — no fixed u encodes a route

From the npz alone: within-episode preimage variance / marginal variance = **0.99**
(1.0 = zero episode-level identity); lag-1 autocorr 0.27, ~0 by lag 50; goal-reaching
segments identical (0.987). The BC flow factorizes behavior as (state → conditional,
u → quantile); the expert's direction choice is driven by a goal that is NOT in the
observation (obs = xy only), so that variance is forced into u independently each step.
Routes exist only as latent *sequences*; Rung-1 assumed u is a persistent policy index,
but Stage A/B construct it as per-step noise. **D2's "pass" measured coherence+diversity,
not usefulness — it could not see this.**

### 3. What the family actually contains: orbiters and constant headings

`tools/viz_fixed_u_field.py` (new): quiver of a = G(xy, u) over the maze per fixed u +
rollout overlay (`fixed_u_fields.png`). Two degenerate regimes and nothing else:
- **Typical u** (prior bulk, e.g. [0,0]): state-dependent field that follows corridors
  but with per-cell arbitrary direction choices → circulation → the rollout ORBITS near
  start (path length 164, net displacement 1.4). heading circ_var 0.83.
- **Saturated u** (box corners): tanh saturation ⇒ near-constant heading everywhere
  (circ_var 0.01–0.03) → wall-sliding diagonal marches. The up-right one gets to 1.32
  from the goal and slides past — it cannot turn, by construction.
Per-cell coverage is FINE: angle circ_var across 512 draws at fixed cells = 0.45–0.78 —
every direction stays available per state. The deficiency is purely temporal.

### Why Stage C then behaves exactly as observed

`agents/psmflow.py` TD target (line ~79) bootstraps `psi(next_obs, u_idx, u_idx)` — the
continuation latent IS the index, faithful to the fixed-u semantics. Since no pi_u
reaches the goal, true V^{pi_u}(task) is the same "never" value for every u → psi^T w is
flat (Q spread ~10% of mean, `viz_latent_value.png`), gpi argmax is noise and drifts to
the saturated corners (the only occupancy-distinct members), calibration reads
predicted 85–212 / realized all-0. Every symptom follows from the one cause.

### Where this leaves the method (decision needed)

The paper's support constraint (every considered action is a flow decode, C=1) survives;
the "fixed noise = policy index" premise does not. Candidate directions, in rough order:
1. **Latent-space PSM (Rung-3 completed properly):** keep the frozen flow; make the
   index a task vector w conditioning a per-step latent actor u(s,w) (already built,
   08-04), and change the psi backup to bootstrap the ACTOR's latent at s'
   (psi(s', ·, u(s',w))) instead of u_idx — policy improvement in latent space, still
   in-support, per-cell coverage measured sufficient. Aligns with in-sample TD Prop 7.2.
2. **Trajectory-level latent in Stage A** (skill-VAE/OPAL-style G(s,u;w_traj), invert w
   per trajectory) — makes fixed-w routes exist by construction; biggest retool.
3. Report as a negative-result finding for flow-noise-indexed families + pivot envs
   where behavior modes ARE single-step-consistent (cube floor says probably not).

### In flight / artifacts

- tmux `audit_psm_pm` (PSM baseline, pointmaze task1): 0.0 through 200k, ETA ~5h — the
  control for whether an RL-optimized family navigates pointmaze at all.
- tmux `audit_psmflow_mix0`: finished flat 0.0 (as the root cause predicts).
- New tools: `tools/latent_reachability.py`, `tools/viz_fixed_u_field.py` (uncommitted).

---

<!-- _class: lead -->

## 2026-08-04 — full audit + roadmap; the D3 ESS statistic was wrong (again), and it changes the story back

Deliverable: **`docs/plans/2026-08-04-status-roadmap-audit.md`** — status, OGBench roadmap
(gates, baselines, success criteria, viz/analysis scripts), full code audit, rewrite plan.
Read that first; this slide records only what changed on disk.

### The D3 gate was averaging ESS over the whole EM trace, not the final iterate

`compute_full_proposal_distribution_em` returns `ess` shaped `(n_steps,)` per row (one per
EM iteration); after vmap the D3 tool took `mean(ess)` over BOTH axes. Stage B stores — and
`precompute_preimages.py` persists, and the intuition figure plots — only `ess[:, -1]`, and
ESS improves across EM iterations, so **every number the tool printed understated the
stored posterior**. Fixed: the gate now uses final-step ESS and also prints
`mean_ess_trace` for comparability with pre-08-04 printouts.

**Re-measured (cube Stage-A ckpt, 256 rows, seed 0, alpha=20, N=100): final-step mean ESS
21.7, trace-mean 17.7 — the gate PASSES at N=100.** So the 08-03 working-tree conclusion
"no alpha reaches 20; num_samples is what clears the gate" was an artifact of the trace
statistic; alpha=20 was the right call and N buys *margin* (linear), not the pass.
`configs/inversion/default.yaml`'s comment block now carries the correction. The in-flight
N=200 recomputes (`watch_cube` / `watch_pointmaze` tmux, ETA ~12:30 08-04) are NOT wasted —
they give comfortable headroom over a marginal 21.7 — let them finish, then run the fixed
D3 on both npz.

### Also fixed (audit P0s)

- `tools/latent_q_sanity.py` (D4): the 10k relabel batch was **unseeded** (global
  np.random) — the gate number changed run to run. Now seeded off `cfg.seed`, same pattern
  as D3. D4 is a spec gate; its number must be a function of the checkpoint.
- `scripts/launch_fb_cube.sh` / `launch_psm_cube.sh` ran bare `python` (= 2.7 on midi-01).
  Now `$REPO/.venv/bin/python`.
- `agents/affine_psm.py` `infer_w_goal`: the constraint-set permutation used
  `np.random.default_rng()` (OS entropy) — identical seed + weights gave different `w_inf`
  and eval success. Now seeded off the `seed` argument.

### Later the same day — D2 recovered (PASSES both envs), preimage analyzer landed

- **D-tools now persist their reports** (`utils/log_utils.py:write_report`; `report_out`
  config key, default `<hydra run dir>/<tool>.json`) — the class of loss that ate the
  08-03 D2 numbers is closed. D2's `across` metric also fixed: it averaged over self- and
  within-u pairs, deflating `across` and flattering the ratio.
- **D2 re-run on both Stage-A ckpts** (reports in `/data-local/amsks/PSMFLows/logs/`):
  consistency ratio **0.51 cube / 0.53 pointmaze** (within-u final-state distance ≈ half
  of across-u). Fixed `u` is reproducible AND distinct — the policy family is real, which
  answers note §7 risk 1 and gives A3 its first favorable evidence.
- **`tools/analyze_preimages.py`** (new): full-npz report (ESS/roundtrip/typicality
  distributions, posterior geometry vs prior and u_clip, correlations) + figure, JSON next
  to the npz. On the OLD alpha=1 cube npz: mean ESS 9.8, **39% of posterior means outside
  the u_clip box**, 85% of posteriors wider than the prior — quantifies what the alpha=20
  recompute restores. Run it on the new npz when they land.
- **New finding: 8 rows/1M have finite-but-astronomical point preimages** (||u*||² up to
  1e59; trimmed-mean typicality is healthy at 5.57 vs expected 5). The finiteness-based
  `preimage_valid` does NOT catch these. P1: extend `compute_preimage_validity` with a
  typicality bound (e.g. ||u*||² > 100 ⇒ invalid) — do it before point-mode training is
  ever used; EM-mode Stage C is unaffected (their mixtures are finite).

### Later still — amortized flowBC LATENT actor (Rung 3), off by default

`agents/psmflow.py` now carries the fb_flowbc actor recipe transposed to latent space
(`actor.enabled`, default **false** — v1 inference stays flow-GPI, and the pending Stage-C
launches are unaffected):

- `actor_vf` = FlowVectorField, CFM toward the dataset **preimage latents** (the
  latent-space behavior distribution — N(0,I) only under a perfect flow, so worth
  learning); `actor` = NoiseConditionedActor(s, w, noise) → tanh·u_clip.
- Actor loss mirrors `psm.flow_actor_loss`: −Q/|Q| + bc_coeff·distill + bc_flow_loss,
  with Q = the **diagonal** score ψ(s,u_a,u_a)ᵀw (value of committing to index u_a),
  same ensemble pessimism as gpi_select. ψ/φ take no actor gradients; flow frozen.
- w per batch = PSM's sample_mixed_z (task_mix_ratio phi(next_obs[perm]) vs random unit z).
- `acting: gpi | actor` switches deployment; both decode through the frozen flow, so
  actions stay data-like either way. Knobs in `configs/agent/psmflow.yaml` under `actor:`;
  reference cube used bc_coeff **3.0** (our default 1.0 — sweep when the ablation runs).
- Tests: `tests/test_psmflow_actor.py` — disabled default is a no-op; enabled trains only
  (actor, actor_vf) with psi deltas byte-identical to the disabled run; acting=actor
  emits a boxed latent and a valid action.

### Top open items before Stage C (full list + priorities in the roadmap doc)

1. **npz↔checkpoint pairing guard** in `main.py` — row count is the only check today; a
   wrong-vintage npz trains silently on mismatched latents. The `.meta.json` sidecar
   exists for exactly this. Do it before the first real Stage-C launch.
2. **Persist diagnostic reports** — the 08-03 D2 results were lost because the tools print
   JSON to stdout only. Add a `report_out` (or always `| tee`). D2 must be re-run.
3. Commit the 08-03 + 08-04 working tree as one story (inversion config + fixed D3 tool +
   benchmarks doc + this slide); fix `tools/plot_preimage_intuition.py`'s now-stale panel
   titles ("alpha=1 as run", "raising alpha is what clears the gate") when regenerating.

---

<!-- _class: lead -->

## 2026-07-29 (later) — the D3 ESS gate is an `alpha` problem, not a sample-count problem

Root-caused the failing gate by auditing the stored cube preimages. **`inversion.alpha=1.0`
is too low by ~20x**, and that single knob accounts for the failure. Chain of measurement:

**1. The flow map is shallow in `u`.** Singular values of J = d(action)/d(noise) at the
preimage, 4096 cube rows: sigma_max median **0.117**, sigma_min median **0.068**.

**2. So the Laplace proposal carries no information.** cov = (1/alpha^2)(J^T J)^{-1} with
sigma ~ 0.1 gives eigenvalues ~100, and they are clipped to `[0.01, 1.0]` — for **99.8%** of
rows *every* eigenvalue is clipped, leaving the proposal exactly the N(0, I) prior. The
"local Laplace covariance" is adaptive in name only.

**3. And the target is ~10x broader than that proposal.** With
`||G(s,u) - a|| ~ sigma*||u - u*||`, the target `exp(-alpha*||G(s,u)-a||)` has width
`1/(alpha*sigma) ~ 10` at alpha=1. A prior-width proposal against a 10x-wider target is
exactly the regime that collapses ESS — and it explains the EM's measured 4x-per-iteration
covariance growth as it chases a genuinely broad target.

**4. Raising alpha fixes every symptom monotonically** (512 cube rows, all else as configured):

| alpha | mean ESS | frac ESS>20 | post. width / prior | ‖mu‖ | ‖G(s,mu)-a‖ |
|---|---|---|---|---|---|
| **1.0** (as run) | 9.99 | 0.115 | 4.124 | 3.945 | 0.323 |
| 5.0 | 15.35 | 0.301 | 2.653 | 2.770 | 0.166 |
| 10.0 | 20.31 | 0.404 | 1.977 | 2.575 | 0.104 |
| **20.0** | **21.81** | 0.438 | **0.939** | 2.602 | 0.055 |
| 50.0 | 21.79 | 0.436 | 0.292 | 2.534 | 0.022 |

At alpha=20 the gate passes (mean ESS > 20), the posterior becomes *narrower* than its prior
for the first time — i.e. starts behaving like a posterior — ‖mu‖ lands near the chi_5
typical radius of 2.13, and the mean's round-trip improves 6x. ESS saturates by 20; past it
the posterior over-tightens for nothing. **This is the opposite of the 07-28 note's
suggestion to lower alpha**; flattening the target raises ESS only by discarding information,
and is measurably not the mechanism here. Table is recorded in
`configs/inversion/default.yaml`; the value is left at 1.0 pending the recompute call
(~10 h / 1M rows, and it invalidates the existing npz).

Two hypotheses checked and **rejected** — worth recording so they are not re-run:
- *More samples.* ESS tracks posterior geometry, not sampling noise: corr(ESS, cov trace)
  **+0.33**, corr(ESS, ‖mu‖) **−0.31**, but corr(ESS, ‖a‖) **−0.003** and mean ESS is flat
  at 9.6–9.8 across 0/1/2/3/4 clipped action dims. `num_samples` buys sqrt(N) against a
  mismatch it cannot remove.
- *The flow collapsed to a deterministic policy.* It has not. Varying `u` over the whole
  typical set moves the action with per-dim sd **0.105**, vs **0.270** for the behaviour data
  across the 32 nearest states — **39%**, narrower than the data but nowhere near degenerate.
  The policy family pi_u is real, so `psi(s,u',u)` has something to discriminate. (The small
  Jacobian invited the opposite conclusion; the nonlinear map over ‖u‖<=3 is what saves it.)

### The 13 NaN rows: root cause found, and it is NOT the EM

`_get_preimage_and_jacobian`'s **implicit-Euler inverse diverges** — a fixed 5 sweeps with no
convergence check. Traced step by step on row 32009: the preimage, Jacobian, J^T J and the
whole proposal are already NaN *before EM step 0*. Deterministic: 13/13 poisoned rows
reproduce across 5 independent rng seeds, 0/13 finite controls ever do. So a NaN mixture in
an npz means the inverse diverged, not that the EM misbehaved — the earlier attribution to
`fql.py`'s masking was wrong.

### Fixed

- **`preimage_ess` reported 100/100 on total failure** (`fql.py`, both EM and non-EM
  variants). When no sample is usable the uniform fallback makes every weight
  `1/num_samples`, so `1/sum(w^2)` evaluates to `num_samples` — the metric's *best* value.
  All 13 NaN rows scored a perfect 100, so **D3 gated on a metric that was blind to exactly
  its worst cases**. Now 0.0, with a test that pins it.
- **NaN latents could reach Stage C** (`utils/flow_inversion.py`, `main.py`,
  `tools/precompute_preimages.py`). New `preimage_valid` mask + `repair_invalid_preimages`:
  invalid rows are reset to the N(0, I) prior — the honest posterior under no information —
  counted, warned about, and **aborted above a 1% ceiling** so this can never quietly paper
  over a broken inversion. Rows are *not dropped*: `Dataset.sample` pairs each transition
  with row `idx+1`, so deleting rows would silently re-pair across the gap. Applied at load
  as well as at write, so the existing cube npz (which predates the key) is covered.

Audit notes: all 999,987 finite stored covariances are Cholesky-able in float32 (0 negative
eigenvalues), so `sample_preimage_noise` is safe; its unseeded global-`np.random` default is
fine because `main.py:76` seeds it from `cfg.seed`. `utils/flow_steering.py` discards ESS and
never checks finiteness — same exposure, but it is WP2 groundwork reachable only from its own
test, so latent. `tools/validate_flow_fidelity.py`'s `mmd_rbf` ignores `preimage_limit`
(hardcoded `act[:1024]`), so that one metric is always a 1024-sample estimate.

### D1 — Stage A signs off, but pointmaze is weakly certified

| config | MMD (RBF) | mode-hist TV | off-support @0.2 |
|---|---|---|---|
| pointmaze trained @100 | 3.0e-4 | 0.038 | 0.314 |
| pointmaze RANDOM control | 3.97e-3 | 0.111 | 0.522 |
| cube trained @100 | 1.5e-4 | 0.016 | 0.473 |
| cube trained @10 | 2.7e-4 | 0.019 | 0.350 |
| cube RANDOM control | 1.46e-1 | 0.507 | 0.9995 |

Both trained flows beat their controls (the spec sets no numeric threshold). **Cube's margin
is ~1000x on MMD; pointmaze's is 13x, and its random control is nearly competent** — with
`d_a=2` a near-identity flow already reproduces the action marginal, the same reason ESS is
not comparable across envs. A pointmaze D1 pass certifies much less than a cube one, which
matters because pointmaze carries the D4 gate. off-support is a real ~31% for pointmaze, not
a subsample artifact (0.350 @1024 → 0.329 @2048 → 0.314 @4096 → 0.316 @8192). Cube's *local*
fidelity is worse at flow_steps=100 (0.473) than at 10 (0.350) while global MMD/TV improve —
and 100 is the setting the preimages ran at.

### Stage A on pointmaze — DONE

500k steps, 1h27m, `flow_steps=100` (inversion-safe from the start, unlike the cube ckpt):
`/var/local/amsks/exp/PSMFLows/bcflow_pointmaze-medium-navigate_20260729_142219/sd000_20260729_142225`
wandb `qcmsr46t`. Note `utils/log_utils.py:73` `mkdtemp()`s the wandb dir and so **ignores
`WANDB_DIR`** repo-wide; offline runs land in `/tmp` and must be copied out before syncing.

### NEXT

1. **Decide alpha** (recommend 20) → recompute cube preimages → D3 gate should pass.
2. Stage B on pointmaze at the same alpha → D3 there.
3. **D2** on either ckpt — independent of all the above, and it is what confirms `u` indexes
   *distinct* policies rather than merely non-degenerate ones.
4. Then Stage C, D4, zero-shot vs PSM/FB.

---

<!-- _class: lead -->

## 2026-07-29 — PSMFlow v1 Tasks 3–8 done; cube preimages landed; D3 ESS gate FAILS

Reference docs: note `PAPER/RESEARCH_NOTE.md` · spec `docs/design/2026-07-20-psmflow-v1-design.md`
· plan `docs/plans/2026-07-20-psmflow-v1.md`.

| Task | What landed | Commit |
|---|---|---|
| 3 | `PSMFlowAgent` core — flow-indexed measure loss, frozen-flow load, jitted update | `0c4f2ba` |
| 4 | Flow-GPI inference (`infer_z`, `gpi_select`, frozen-flow decode) | `c444a16` |
| 5 | Chain-MDP ground-truth test (GPI ranks goalward latents) | `b463b87` |
| 6 | Agent registered, `configs/agent/psmflow.yaml`, `main.py` wiring + smoke | `76310c8` |
| 7 | Diagnostics D1–D4 | `1f0fd42` |
| 8 | Stage launchers + this slide | *this commit* |

Suite: **102 passed, 1 skipped.**

---

## The three-stage pipeline — one command each

```bash
# Stage A — reward-free behaviour flow (FQL bc_only). One seed per GPU.
SEEDS="0" bash scripts/pretrain_behavior_flow.sh 1 500000 cube-single-play-singletask-v0

# Stage B — invert the flow over the dataset (MUST be flow_steps>=100)
.venv/bin/python tools/precompute_preimages.py agent=fql \
    env_name=cube-single-play-singletask-v0 \
    restore_path='/var/local/amsks/exp/PSMFLows/bcflow_*/sd000_*' restore_epoch=500000 \
    agent.flow_steps=100 preimage_out=/var/local/amsks/exp/PSMFLows/preimages_cube_single.npz

# Stage C — psmflow representation training
FLOW_CKPT='/var/local/amsks/exp/PSMFLows/bcflow_cube_single_20260726_135032/sd000_*' \
FLOW_EPOCH=500000 PREIMAGES=/var/local/amsks/exp/PSMFLows/preimages_cube_single.npz \
SEEDS="0" bash scripts/launch_psmflow.sh cube-single-play-singletask-v0 1 500000
```

**Stage A is `pretrain_behavior_flow.sh`, not the plan's `launch_flowbc.sh`** — Task 1 had
already created it under that name and two tool docstrings cite the path, so Task 8
extended it (multi-seed, `FLOW_STEPS`/`SAVE_INT`) instead of landing a second launcher for
the same stage. Both scripts guard their required inputs and exit 1 before JIT.

---

## Diagnostics D1–D4 (Task 7) — all `stdout` JSON, no pytest

| | Tool | Asks |
|---|---|---|
| D1 | `tools/validate_flow_fidelity.py` | does the flow reproduce the dataset's action distribution (RBF-MMD, mode-hist TV, off-support frac) |
| D2 | `tools/eval_fixed_u_rollouts.py` | is a fixed `u` reproducible AND distinct (within-u vs across-u final-state distance) |
| D3 | `tools/validate_flow_inversion.py` | typicality / round-trip / ESS — carries the spec gate |
| D4 | `tools/latent_q_sanity.py` | oracle-reward flow-GPI, isolating representation+GPI from reward inference. Gate: ≥50% of FQL on pointmaze-medium |

**D3 is now seeded — and that settles the 07-28 open question.** `Dataset.sample` drew from
global `np.random`, so every invocation scored a *different batch*; the earlier
"KS 0.095 → 0.061" reading was that, not a changed flow. Pinned at `seed=0`, cube gives
p = **3.4e-9** @100 and **6.3e-5** @200 — **A2 IS rejected at both**. This supersedes the
07-28 slide's "A2 is not rejected at 5% @200" caution. D3 also built its agent from
`fql.get_config()` rather than the Hydra agent group (same defect fixed in
`tools/precompute_preimages.py` in `03bfc71`); it now uses `cfg.agent`.

---

## OPEN — the D3 ESS gate FAILS on the real cube flow

| `flow_steps` | roundtrip | χ² band | mean ESS | gate |
|---|---|---|---|---|
| 100 | 1.2e-4 | 0.9873 | **7.16** | **FAIL** (needs > 20) |
| 200 | 7.5e-5 | 0.9912 | **7.14** | **FAIL** (needs > 20) |

Round-trip and typicality pass comfortably; only ESS fails. **Not a regression** — ESS was
~7 at every `flow_steps`, including settings that never NaN'd. It is the EM's actual
operating point on cube: `num_clusters=1` fits a 5×5 covariance (15 free parameters) from
~7 effective samples, and `min_ess=1.0` means some transitions put all posterior mass on a
single draw. Knobs: `inversion.num_samples` (linear cost) and `inversion.alpha` (flatter
target ⇒ higher ESS, less sharp posterior).

**ESS is not comparable across environments** — untrained pointmaze reads 16.9, *higher*
than the trained cube flow, because `d_a` is 2 not 5 and a near-identity flow gives a flat
posterior. Do not gate on the cross-env comparison.

**The cube preimages were computed at this ESS.**
`/var/local/amsks/exp/PSMFLows/preimages_cube_single.npz` — 319 MB, n=1,000,000, finished
07-29 00:47, from the 500k Stage-A ckpt at `flow_steps=100` (`.meta.json` sidecar records
the full inversion config). **The decision to make before Stage C: is 20 the right
threshold, or does this get recomputed at larger `num_samples`?** That is a method call,
not a code one. A recompute is ~10 h wall-clock at last night's rate.

---

## NEXT — Phase A exit (spec §5), operator-driven

1. Stage A flow on **pointmaze-medium** (cube ckpt exists) → **D1 gate** both envs.
2. **Resolve the ESS question**, then Stage B preimages for pointmaze (+ cube recompute if
   that is the call) → **D3 gate**.
3. **D2** rollout report — informational, feeds note §7 risk 1 (does `u` index distinct
   behaviours at all).
4. Train psmflow both envs, 500k, 3 seeds → **D4 oracle-GPI gate on pointmaze**.
5. Zero-shot eval vs in-repo PSM and FB, same envs/steps/seeds; extend
   `scripts/compare_multiseed.py` to psmflow groups.

Coverage-ladder stress tests and VC-FB comparisons are the *next* spec, once these gates pass.

Caveat carried forward: the landed cube Stage-A ckpt was trained at the FQL default
`flow_steps=10` and is inverted by overriding to 100 at precompute time. Stage A now
*trains* at 100 by default (safe only because `utils/xla_guard.py` disables the autotuner —
see the 07-28 slide), so a re-trained flow will not be step-identical to the current one.

---

<!-- _class: lead -->

## 2026-07-28 — PSMFlow v1 Tasks 1–2; two numerics bugs cleared

Workstream switched to the **PSMFlow v1 plan**. Execution order is 1→2→…→8; GPU work
(Stage A → D1/D3 gates → preimages → psmflow 3 seeds → D4 → zero-shot vs PSM/FB) is
operator-driven after Task 8.

### Task 1 — DONE (`290b951`)

FQL `bc_only`: reward-free behaviour-flow pretraining. Strips the reward terms from the
actor loss, leaving `bc_flow_loss + alpha*distill_loss`; the critic branch is skipped and
`rewards`/`masks` are never read (the pretraining dataset has neither). Param tree is
unchanged so checkpoints restore into a default-shaped FQL agent in Tasks 2/3.
**Stage-A cube checkpoint landed at 500k:**
`/var/local/amsks/exp/PSMFLows/bcflow_cube_single_20260726_135032/sd000_20260726_135037`.

### The XLA autotuner miscompiles the flow integration (`d2b3ec1`)

D3's round-trip of 2.26 and NaN ESS were **both artifacts**. Past a threshold unroll
length, XLA:GPU's autotuner picks a wrong kernel for `compute_flow_actions`: 84.5% of
outputs pinned at the ±1 clip jitted vs 4.5% un-jitted, reproducibly, across processes;
CPU correct in every form. Onset is between `flow_steps` 30 and 100, so **training at
`flow_steps=10` was never affected and the Stage-A checkpoint is sound** — but the plan's
Stage-A launcher specifies `flow_steps=100` and WOULD have trained on a corrupted
distillation target. Fix: `utils/xla_guard.py` sets `--xla_gpu_autotune_level=0` **before
jax initialises** (it is read at XLA init; setting it later is a silent no-op), imported
ahead of jax by `main.py`, the GPU tools, and `tests/conftest.py`.
**The trigger needs TRAINED weights** — random weights agree to 3.5e-5, which is why a
self-contained numerics test would pass vacuously (it is opt-in behind `PSMFLOWS_FLOW_CKPT`;
the committed test pins the import ORDER instead).

### The residual NaN ESS — fixed

Not "the EM proposal is broken at `flow_steps>=100`". One unguarded NaN with a wide blast
radius, from two compounding causes:

1. **The BC flow genuinely diverges from tail proposal samples.**
   `_get_predistribution_proposal` clips the Laplace cov eigenvalues to `[0.01, 1.0]` and
   on cube it **saturates at the upper bound**, so the EM effectively samples `~N(x_0, I)`
   — as wide as the prior. ~0.02% land at `||u||~6.8` (mean 3.1) and integrate
   `6.5 -> 2.3e10 -> inf -> NaN` by step ~95. **`flow_steps` 10 and 30 under-resolve the
   blow-up and stay finite** — that, not any instability of fine discretization, is why it
   only appeared at >=100.
2. **One NaN killed the whole row.** The 07-28 guards floored `log_q` but never checked
   `log_energy`, and softmax over a vector containing one NaN is NaN in EVERY position.
   Cascade: 17% of rows NaN at EM step 0 -> 69% at step 1 -> **100% by step 5**.

Fix: mask on both grounds (`ok = q_ok & isfinite(log_energy)`) -> logit `-inf` (weight
exactly 0), uniform fallback if nothing survives. Masked samples keep responsibility
`1/K` so `gamma = resp * weight` stays finite — **`0 * NaN` would not**. Same guard applied
to the non-EM `compute_full_proposal_distribution`, which had identical exposure.
`test_em_ess_survives_a_diverging_flow_sample` injects one NaN action (random weights do
not diverge on their own) and asserts the sample is EXCLUDED, not merely tolerated
(`ESS <= 23` of 24). **77 passed, 1 skipped.**

### D3 gate — Stage-A cube checkpoint, 1024 transitions, guard active

| `flow_steps` | roundtrip | KS (p) | mean ESS | min ESS |
|---|---|---|---|---|
| 10 | **NaN** | 0.51 (0) | 6.88 | 1.0 |
| 30 | 3.4e-4 | 0.217 (6e-43) | 6.96 | 1.0 |
| 100 | 1.2e-4 | 0.061 (9e-4) | 7.13 | 1.0 |
| 200 | 7.6e-5 | 0.039 (**0.087**) | 7.13 | 1.0 |

- **Preimage precompute must run at `flow_steps>=100`.** At 10 the implicit-Euler fixed
  point (5 sweeps) diverges outright; 10 is the *training* default and is NOT safe for
  inversion.
- **mean ESS ~7/100 at EVERY step count**, including the regimes that never NaN'd ⇒ that is
  the EM's normal operating value, not damage from the divergence and not something the fix
  changed. Low, and `min_ess=1.0` means some rows put all mass on one sample — a
  proposal-quality question worth a threshold on the Task-2 `preimage_ess` health scalar,
  but not a blocker.
- **CAUTION — D3 is unseeded.** `Dataset.sample` draws from global `np.random`, so
  typicality moves batch-to-batch: these numbers differ from `d2b3ec1`'s (KS 0.095->0.061
  @100, 0.068->0.039 @200) because it is a **different batch**, NOT because anything
  improved. A2 is not rejected at 5% @200 *on this batch only* — do not read that as A2
  now holding until D3 takes a seed. Worth fixing before D3 is used as a gate.

### NEXT — Task 2 (preimage pipeline hardening)

The ESS blocker is cleared, so Task 2 is unblocked. Per the plan: checkpoint restore +
Hydra agent cfg + meta sidecar in `tools/precompute_preimages.py`; `noise_preimage_point`,
`preimage_roundtrip`, `preimage_ess` npz keys; `preimage_point_mode` on `Dataset`;
`allow_untrained: false` in `configs/inversion/default.yaml`; `tests/test_preimage_pipeline.py`.
**No preimage `.npz` exists yet.** Run the precompute at `flow_steps>=100`.

---

<!-- _class: lead -->

## 2026-07-26 — flowBC actor for affine PSM; reference-HP audit

**Question:** affine PSM sits at floor on `cube-single-play-singletask-v0`. Are we
using the flowBC actor the reference uses for cube?

### Finding 1 — we were not (now fixed)

`../Factored-FB` README is explicit: `fb_flowbc` (flow-matching VF + distilled noise
actor) for **cube/scene/puzzle**; the deterministic actor only for antmaze/locomotion.
The proven cube recipe whose `ortho_coef=1000` our config copied
(`psm_state_orthohi__ortho_coef1000`) launches with `agent=psm_flowbc`. We had inherited
the hyperparameter and dropped the actor it was tuned around. Mechanism: cube actions are
multimodal, and a tanh mean fit by MSE-to-data regresses to the **mean of the modes**,
which is not a valid action.

- **Ported** the flow path into `agents/affine_psm.py` behind `actor.type: ddpgbc | flow`
  (mirrors `agents/psm.py`): `flow_actor_loss` = CFM velocity field `v(s,x_t,t)` + one-step
  `NoiseConditionedActor` distilled from its 10-step Euler rollout; new `actor_vf`
  TrainState; dispatch in `apply_update`/`total_loss`/`sample_actions`. Q is unchanged —
  still the frozen-measure goal Q `Φ(s,a,g)·w_g + b(s,a,g)`, factored into `_goal_task_coord`
  / `_goal_q` so both actor branches share it.
- Config defaults from the reference `psm_flowbc.yaml` (512×2 actor, 512×4 VF,
  `flow_steps=10`, `lr_actor_vf=3e-4`, `bc_coeff` 1.0 → **3.0** = the cube value).
- `tests/test_affine_psm_flow.py` (9 tests) + one pre-existing test rerouted through
  `sample_actions` (it called `agent.actor(obs, w)` with the ddpgbc arity). **27/27 green.**

**Result: the actor was NOT the binding constraint.** 3 seeds × 500k, 50 eval episodes
(`affine_flow_500k_20260726_022644`): peaks **0.08 / 0.10 / 0.04**, no trend. At a matched
500k the ddpgbc baselines were 0.00–0.10, so flow is **not worse** — but it does not fix it.

### Finding 2 — affine PSM is running RLU's DMC-scale HPs, not the cube recipe

`batch_size=32`, `d_dim=z_dim=50`, `lr=1e-4` all come verbatim from
`RLU/controllable_agent/url_benchmark/agent/psm.py` (a DMC/gridworld codebase), not from
anything tuned on OGBench cube. Deviation table vs `../Factored-FB/configs/agent/psm.yaml`:

| HP | reference (cube) | affine_psm | matchable? |
|---|---|---|---|
| `batch_size` | 1024 | 32 | **NO — see Finding 3** |
| basis dim | 128 | 50 | free |
| basis LR | **1e-5** (sweep winner; `psm.yaml` says "NOT 1e-4") | 1e-4 | **NO — see Finding 4** |
| `max_log_seed` | 16 | 12 | free |
| `target_tau` / `discount` / `ortho_coef` / `lr_actor` | 0.01 / 0.98 / 1000 / 1e-4 | same | already matched |

### Finding 3 — batch_size=1024 is architecturally blocked (measured)

The bilinear PSM gets B² free: `M = ψ(s,z,a) @ φ(g)ᵀ` is an outer product of two `B×d`
matrices, so B=1024 costs **1024 network evals**. The affine net takes `x` *inside* the
network, so B=1024 costs **1024² = 1,048,576 evals** of the 1024×3 measure MLP.
**Probed on GPU: it OOMs** — a single backward fusion alone requested 4.02 GiB and failed
at `XLA_PYTHON_CLIENT_MEM_FRACTION=0.45`. This is the architecture, not a missing knob.

*Options if we want the reference's effective batch:* a **rectangular mesh** — decouple
source rows from measure-argument columns (`B_src=1024 × B_x=64` ≈ 65k pairs, feasible)
so the source-side gradient noise and the contrastive negative count are restored
independently; or a square B=256 (same 65k pairs). Both need a change to `measure_loss`,
which currently builds `i_idx=repeat(arange(B),B)` / `j_idx=tile(arange(B),B)`.

### Finding 4 — the reference's two-timescale split has no landing spot in affine PSM

The reference does **not** run actor-vs-critic two-timescale: `lr_sf = lr_actor = 1e-4`.
Its split is **basis vs everything else** — `lr_phi=1e-5`, 10× slower, marked the sweep
winner. We already match on the actor axis (`lr_actor == lr == 1e-4`; the `lr_actor` knob
from `0a82050` defaults to equal, and the one probe — `affine_tt_1e4` vs `affine_tt_3e5` —
was inconclusive at 10 eval episodes).

But `AffineMeasureNet` (`utils/psm_networks.py:236`) is a **single trunk on
`concat[obs,action,x]`** with φ/b heads: the basis shares every weight with the successor
side, so **there is no parameter group to slow down**. Restoring the reference's structure
means splitting it into separate x- and (s,a)-branches so the x-branch carries its own
optimizer.

### IN FLIGHT (launched 03:41, ~45 min, 2 seeds/GPU on 0,1,3)

| group | vs the 500k flow runs |
|---|---|
| `affine_refdims_20260726_034122` | `d_dim`/`z_dim` 50→**128**, `max_log_seed` 12→**16** |
| `affine_refdims_slowbasis_20260726_034122` | same + `lr` 1e-4→**1e-5** |

Both flow actor, `zero_shot` inference, 3 seeds, 500k, 50 eval episodes. refdims vs the
flow runs isolates the dims; slowbasis vs refdims isolates the representation timescale.
**Caveat on slowbasis:** `lr=1e-5` slows the *whole* measure net (basis + offset +
successor side), whereas the reference keeps ψ at 1e-4. If it helps ⇒ the Finding-4
refactor is worth doing properly. If it hurts, suspect plain underfitting at 500k before
concluding the timescale idea is wrong.

### Corrections to earlier reads (don't re-derive these)

- The ddpgbc baselines do **not** top out at 0.2 once. Reading only the last CSV rows is
  misleading — `affine_psm_amortized` hits 0.20 @200k, 0.20 @600k, **0.30 @800k**; the good
  numbers all sit **past 500k**. Those were 10-episode evals (each 0.1 = one episode).
- Reported `psm_loss` for affine PSM is dominated by a constant: with sqrt(d)-normalized Φ
  the ortho diagonal term is inert at ≈ −d, so `ortho_coef=1000` contributes a ≈ −50000
  offset. **Read `orth_offdiag` (decorrelation) and `psm_offdiag`, not `psm_loss`.**

### Known issue, NOT fixed (flagged, out of scope here)

`main.py:80` sets `dataset.return_index = True` only for `agent_name == 'psm'`, so
`affine_psm` falls back to `jnp.arange(B)` for the codebook hash — a transition's proto
action changes with **where it lands in the batch**. `agents/psm.py:315-319` documents this
as incorrect. Worth fixing before trusting any conclusion about the measure.

### Runnable

`bash scripts/launch_affine_psm_cube.sh [GPUS] [STEPS] [WANDB_MODE]` (new) — comma-separated
GPUs, **one seed per GPU**, `SEEDS=`/`ACTOR=`/`EXTRA=`/`EVAL_INT=` env overrides, uses
`.venv/bin/python` explicitly. Call it twice with different `GROUP` to stack 2 configs/GPU.

---

<!-- _class: lead -->

## 2026-07-15 — FB agent ported to JAX (bit-exact)

Ported the PyTorch **Forward–Backward (FB)** agent (`../Factored-FB` `agents/fb/*`)
to JAX/Flax, mirroring the PSM port protocol. **Spec** `docs/design/
2026-07-14-fb-jax-port-design.md`, **plan** `docs/plans/2026-07-14-fb-jax-port.md`.

- **Scope:** cube-default `fb_flowbc` path — Forward map `F(left_enc(obs),z,a)`,
  Backward map `B(next_obs)` (measure basis + z-source), a **left_encoder** trunk
  (with target), td3/flow actor. Measure `M=F·Bᵀ`, off-diag/diag + ortho loss.
  2 backward passes (FB, actor); targets on forward/backward/left_encoder, none on
  actor; taus 0.005. z mixed 50/50 with `B(next_obs[perm])`. **Not** ported:
  iql critic, traj goal-mode, fixed_b, goal_cond, onestep, reweight.
- **Files:** `agents/fb.py`, `utils/fb_networks.py` (ForwardMap/BackwardMap/FBTd3Actor;
  reuses NoiseConditionedActor/FlowVectorField), `configs/agent/fb.yaml`
  (z_dim=50, batch=256, disc=0.99), registered `fb` in `agents/__init__.py`.
  `main.py` needs NO FB branch (no `index`; `infer_eval_z` picked up generically;
  seeded eval applies).
- **Bit-exact parity (like PSM):** `tools/export_fb_fixture.py` → `tests/fixtures/
  fb_reference.npz`; `tests/test_fb_{networks,agent,smoke}_equiv.py`. Per-module
  atol 1e-10, 10-step `apply_update` atol 1e-8. **13/13 FB tests pass.** Also the
  first bit-exact check of our shared NoiseConditionedActor/FlowVectorField.
- **Fixture gotchas (cost a long debug):** (1) torch `.numpy()` **aliases** memory —
  per-step param snapshots must `.copy()` or they all show the final trained weights;
  (2) the reference torch build's in-place first Adam step lands ~10× too large under
  some construction orders, so the K-step trace uses a **manual optax-matching Adam**.
- **Runnable:** `bash scripts/launch_fb_cube.sh [GPU] [STEPS]` (one seed/GPU,
  save_interval=100k, seeded eval, eval_episodes=10). Verified 3-step end-to-end
  through `main.py` (exit 0). **NEXT:** cube-single parity run vs the reference FB
  benchmark (wandb `amsks/factored-fb`; see [[reference-fb-flow-benchmarks]]).

---

<!-- _class: lead -->

## 2026-07-13 — fresh from-scratch parity runs

**Goal:** measure how much our JAX PSM+flow recovers vs the reference on
from-scratch runs (not transplant), 3 seeds, 500k.

- Launched seeds **0/1/2**, flow, `ortho_coef=1000`, `z_dim=128`, `lr_phi=1e-5`,
  eval@50k / 50 ep. Group `psm_recover500k_flow_ortho1000_20260713_160709`.
- **NEW, CONTRASTS with prior finding:** **seed 2 recovers the reference
  in-window.** seed2 @500k = **0.42** vs ref **0.60** (−0.18); but
  **mean(0–500k) ours 0.238 vs ref 0.240 — a wash.** We LEAD early (50k/100k the
  ref is still 0.00), ref leads the 250–300k bump. Peak ours 0.44@350k vs ref's
  own in-window 0.50. This is NOT the old "plateau at ~0.05–0.11" story.
- **Seeds 0 & 1 weaker so far** (peaks 0.18 / 0.22 at 450k) — but the reference
  for THOSE seeds is also weak early (s0 ref 0.30 by 300k). ⇒ **high seed
  variance dominates**; single-seed reads are noisy.
- **THE remaining gap is the CEILING, not the trajectory.** The reference keeps
  climbing past 500k: peaks **0.80 @750k**, holds **0.5–0.8 to 1.5M**. At our
  500k cap the −0.18 is real but small; the reference's *ceiling* only appears
  with 2–3× more training we hadn't run.
- **⇒ Launched seed 2 to 1M** on GPU 1 (solo) to test the ceiling.
  Group `psm_recover1M_flow_ortho1000_s2_20260713_174415`. **At 250k already
  0.60** (ref 0.50 @250k). ETA ~2.5–3h from 17:44.
- **Infra lesson:** 2 seeds sharing one GPU run ~2× slower (seeds 0/1 on GPU3
  took ~2.5h; seed 2 solo on GPU1 did 500k in ~1h23m). **One seed per GPU** for
  even wall-clock. `XLA_PYTHON_CLIENT_MEM_FRACTION=0.30` per proc.

### Runs (2026-07-13)

| group | seeds | steps | GPU | status |
|---|---|---|---|---|
| `psm_recover500k_flow_ortho1000_20260713_160709` | 0,1 | 500k | 3 (shared) | running (~450k) |
| ″ | 2 | 500k | 1 | **done** @500k=0.42 |
| `psm_recover1M_flow_ortho1000_s2_20260713_174415` | 2 | **1M** | 1 (solo) | running (~250k, 0.60) |

Compare: `.venv/bin/python scripts/compare_multiseed.py [GROUP]` (defaults to
`/var/local/amsks/exp/multiseed_group.txt`; the 1M group is in
`recover1M_s2_group.txt`). Reference cache has seeds 0,1,2,3,4,5,7.

### Next (from 2026-07-13)

1. **Read the 1M seed-2 result** — does it reach the ref's 0.7–0.8 ceiling? If
   yes ⇒ we DO have parity, just needed budget; the "systematic gap" was a
   500k-cap artifact + seed variance. If it plateaus ~0.4 ⇒ real ceiling gap.
2. Let seeds 0/1 finish 500k; re-run `compare_multiseed.py` for the 3-seed table.
3. If ceiling confirmed, consider 1M runs for seeds 0/1 too (one-per-GPU) to
   get a real 3-seed peak distribution vs the reference's spread.
4. GPUs 1 & 3 usable; 0 & 2 were busy/full (other users) at the time.

---

## TL;DR — the headline finding

- Our JAX PSM+flow **task2 success plateaus at peak ~0.20–0.32** and stays near the floor.
- The **code-matched reference climbs to 0.30–0.60 by 500k and peaks 0.80–0.90** (at 650k–1350k).
- This is a **real, systematic difference** (all 3 seeds fail uniformly), **NOT noise**.
- It is **NOT** in the eval/acting path, the loss math, masks, the proto table, or obs-norm
  (all audited/matched). **It is in TRAINING/init** — our update yields weaker weights.
- **CONFIRMED by transplant-eval:** reference weights (seed5 @100k) score **0.58** in OUR eval
  (task2); our own weights top ~0.25 in the same eval. Our eval is faithful; **training is where
  our weights end up worse.**
- **UPDATE (round 9): init scheme ALSO faithful.** After 9 rounds, EVERY code path is faithful
  (per-step math, training-gen, acting, eval, init scheme, config). **No code bug found.**
- Remaining diffs are **pure cross-framework RNG** (init draws + minibatch order). The reference
  is **hugely noisy** (same seed5: 0.20 wandb vs 0.52 fresh @100k). Effect size is uncertain
  with 3 seeds×1 run — may be a low draw, not a bug.

---

## How we got here

1. Started from an "eval@100k red flag". First pass concluded *faithful* — but that was an
   **improper comparison** (wrong reference algo for s5/10/7 + mid-run vs 1.5M peak).
2. Corrected: pulled the **code-matched PSM-orthohi** reference curves (wandb
   `amsks/factored-fb` `psm_state_orthohi__ortho_coef1000__lr_phi1e-5__s{0,5,7}`), aligned
   step-for-step. At matched steps ≤140k we tracked it — but the reference **blooms late**.
3. User (correctly) pushed: a faithful port must reproduce the reference's tight 0.8–0.9
   **peak distribution**. It doesn't.
4. Transplanted the reference **proto behavior table** to kill the last RNG diff → **gap did
   NOT close**. Confirmed the difference is real.

---

## What is RULED OUT (audited faithful / matched)

- **Per-step update math** — `tests/test_psm_agent_equiv.py` transplants reference weights,
  runs 10 optimizer steps on injected data, matches to **atol 1e-8**. Network+objective+HPs
  are bit-identical *given identical inputs*.
- **Eval z-inference** — formula-identical (`infer_z` == ref `reward_inference`); reward
  source matches (our dataset 2.08% nonzero == ref relabel ~2%).
- **Eval task identity** — `cube-single-play-singletask-v0` has `cur_task_id=2`; we compared
  task2-to-task2 all along.
- **Acting path** — `sample_actions` + actor nets faithful (one-step flow sample, tanh/clip,
  z-proj all match; multi-step Euler rollout is only a distill target, never acted).
- **masks** (always-γ), **obs normalization** (Identity for state, both sides), **proto table
  distribution** (`(rand-1)*2 ∈ [-2,0)`), now transplanted verbatim.

---

## KEY INSIGHT — why "losses match" proved nothing

- `actor_loss ≈ -Q/|Q|` and `train/q` is the **critic's own** estimate — both read ~-0.77 /
  ~600 **whether or not the policy reaches goals**. `bc_flow_loss` only measures fit to the
  behavior dist. **None of the scalar losses measure task success.**
- The equiv test **injects** the batch + all latent/action samples, so it verifies the update
  *given inputs* but **never how those inputs are GENERATED**.
- ⇒ The bug must live in **training-signal generation** (batch sampling, `z_cont` mixing,
  next-action injection, target updates) — invisible to both the losses and the equiv test.

---

## Two deep audits — BOTH came back "faithful"

- **Acting/eval-path audit: FAITHFUL.** `sample_actions` + actor nets match; eval task is
  task2 both sides; z-inference reward matches.
- **Training-generation audit: FAITHFUL.** batch sampling, `z_cont` mix, SF/proto next-action,
  target updates, per-stage optimizers all match. Only nits: `z_cont` mixed half uses
  pre-proto-step phi (ref uses post-step) — one `lr_phi=1e-5` step, negligible; proto table
  (now transplantable).
- **Config verified matching:** our runs used **z_dim=128, ortho_coef=1000, actor=flow**
  (checked `flags.json`). The audit's `z_dim=50` worry was stale-handoff text, not real.

⇒ **No code/config bug found on either side.** Remaining suspects: **init SCHEME** (orthogonal
gain/draw — audits didn't deep-check; all 3 seeds fail uniformly = systematic, argues against
pure basin luck) OR genuine cross-framework init/data-order basin variance (weakened by 3/3
uniform failure).

## STILL IN FLIGHT — the discriminator

**Torch reference checkpoint dump** — `seed5`, **ortho_coef=1000** (gotcha: psm_flowbc default
is 1.0!), GPU 1, `/var/local/amsks/exp/ref_ckpt_s5_300k`, saving `step_{100,200,300}k.pt`.
**Transplant-eval:** load ref weights into our JAX eval →
- ref-weights score **high** ⇒ bug is in **our training/init** (not eval).
- score **low** ⇒ bug is in **our eval** (both audits say unlikely).
To separate init-vs-training: load ref *early* weights into our JAX and TRAIN — climbs ⇒ init.

---

## Transplant-eval — READY to build (checkpoint landed)

`step_100000.pt` (+200k,300k) saved in `/var/local/amsks/exp/ref_ckpt_s5_300k/`. state_dict
naming **matches the fixture** convention → prefix keys with `w__` and reuse existing loaders.

- **phi / sf_psi / psm_psi:** reuse `load_phi_params` / `load_psi_params` as-is. Keys:
  `phi.net.{0,1,3,5}`, `{sf,psm}_psi.{embed_z,embed_sa}.{0,1,3}` + `Fs.{0,2}` (ensembled, P=2,
  shapes `[2,in,out]`). targets present too (`target_*`).
- **NEW converters needed** (torch_to_flax is ddpgbc-only):
  - `_actor` = **NoiseConditionedActor** (18 params): `embed_z.{0,1,3}`, `embed_s.{0,1,3}`,
    + noise/policy layers → our `utils/psm_networks.py:NoiseConditionedActor` tree.
  - `_actor_vf` = **FlowVectorField** (10 params): `net.{0,2,4,6,8}` (5 Linears, GELU between)
    → our `FlowVectorField` tree.
- Harness: build `PSMAgent` with these params (config z_dim=128, ortho1000, actor=flow), run
  `infer_eval_z` + `evaluate` on task2. **Ref weights should score ~0.20 (100k)** in a faithful
  eval — matches this run's own eval (seed5 task2 0.08@50k, climbing). Low ⇒ eval bug; ok ⇒
  training/init bug. **A buggy converter gives a false low — validate by matching phi/Q outputs
  to the torch model on a shared batch first.**

---

## Code changes

- **Committed & pushed** `31ed43d` "PSM: reference-parity fixes (eval z-inference + masks) +
  audit tooling": masks always-γ (`agents/psm.py`), eval z-shift/relabel (`main.py`,
  `config.yaml`), `eval_interval 100k→20k`, `docs/`, `scripts/launch_psm_cube.sh`.
  - **UNCOMMITTED** (working tree): `proto_table_path` hook — `agents/psm.py` `create()` loads a
  transplanted table when `agent.proto_table_path` set; `configs/agent/psm.yaml` declares it
  (`null` default). 15/15 PSM tests still pass.

---

## Runs

| group / run | what | status |
|---|---|---|
| `psm_maskfix_flow_ortho1000_20260707_164832` | s0/5/10/7, 500k, eval+masks fix | **killed** (superseded) |
| `psm_protoxplant_flow_ortho1000_20260707_184559` | s0/5/7, 500k, +proto transplant | **done** — gap NOT closed |
| `/var/local/amsks/exp/ref_ckpt_s5_300k` (torch) | ref s5 ortho1000, ckpt dump | **running** (GPU 1) |

Late-window (350–500k) task2 mean, protoxplant: ours s0/5/7 = **.06/.11/.05** vs ref **.08/.35/.33**.

---

## Reference curves (code-matched PSM-orthohi, task2)

Cached: `/var/local/amsks/exp/ref_orthohi_task2_curves.json` (seeds 0/5/7, every 50k to 1.5M).

| seed | @100k | @300k | @500k | peak |
|------|------:|------:|------:|-----:|
| 0 | 0.10 | 0.20 | 0.00 | 0.80 @1350k |
| 5 | 0.20 | 0.60 | 0.30 | 0.90 @650k |
| 7 | 0.00 | 0.40 | 0.10 | 0.80 @900k |

Reference is **very noisy** but clearly trends up; ours does not. `s10` has **no** code-matched
reference (orthohi set is seeds 0–9) — dropped it.

---

## Next steps (priority order) — DONE: transplant-eval + init audit (both clean)

All 9 rounds of code audit are clean. The question is no longer "where's the code bug" but
"is our training genuinely worse, or a low RNG draw." Two ways to settle it:

1. **DECISIVE: continue-training from ref weights in OUR loop.** Load ref 100k weights (score
   0.58 in our eval) into a trainable `PSMAgent` + fresh opt_states, train 100k more with our
   `update`, eval. **DEGRADES toward ~0.2 ⇒ our training dynamics actively hurt (real bug);
   PRESERVES/climbs ⇒ it was init-draw/basin luck.** (Extend `scripts/transplant_eval.py`:
   add opt_states init + our training loop over the dataset.)
2. **Characterize distributions:** run s0/5/7 (+more seeds) as **multiple replicates** and
   compare to the reference's own spread — the reference bounces 0.20–0.52 for the same seed.
3. Commit the `proto_table_path` hook (+ converter/harness) once settled.
4. Peak parity ultimately needs **1.5M** runs (user capped at 500k this round).

---

## Environment gotchas (unchanged + new)

- **`python` = system Python 2.7** on this box. ALWAYS use `.venv/bin/python` (ours) or
  `/var/local/amsks/ffb-venv/bin/python` (torch). The launcher needs `source .venv/bin/activate`.
- **midi-01 = HTCondor, no Slurm.** Run directly (nohup/tmux + `CUDA_VISIBLE_DEVICES`).
- Home NFS quota tiny → all bulk data in `/var/local/amsks/`. GPUs shared — check `nvidia-smi`.
- Torch stdout is **block-buffered** to a file (looks "stuck"; it's flushing in bursts).
- **Reference ortho_coef override is bare `ortho_coef=1000`** (`@package _global_`), NOT
  `agent.*`; psm_flowbc default is 1.0. The old repro `ref_psm_cube_s0_100k` was ortho=1.0.

---

<!-- _class: lead -->

## Pointers

- Agent: `agents/psm.py` · Networks: `utils/psm_networks.py` · Converter: `utils/torch_to_flax.py`
- Config: `configs/agent/psm.yaml`, `configs/config.yaml` · Launcher: `scripts/launch_psm_cube.sh`
- Reference (PyTorch): `/u/amsks/git/Factored-FB` (`agents/psm/*`, `nn_models.py`), torch venv
  `/var/local/amsks/ffb-venv`, data `/dev/shm/factored-fb/datasets`
- Investigation tools (now in repo): `scripts/transplant_eval.py` (load ref torch weights →
  our JAX eval), `scripts/compare_protoxplant.py` (step-aligned curve vs cached ref)
- Cached ref curves: `/var/local/amsks/exp/ref_orthohi_task2_curves.json`
- **Memory: `psm-vs-reference-audit` (rounds 1–6 — the full investigation trail)**

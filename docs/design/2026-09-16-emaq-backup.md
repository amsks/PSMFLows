# EMaQ-style backup in the measure: `bootstrap=gpi_argmax` (cube)

Date: 2026-09-16. Status: implemented and unit-tested; launch pending the GPUs freed by the
2P 500k-basis runs stopping at 2M.

## Why

The 2026-09-15 isolation (`docs/design/2026-09-15-policy-family-diversity.md`,
`diag_measure_vs_scalar_q`) put the remaining gap in whose value the measure holds. Under
`policy_index=latent` the backup continues the same index at s', so psi(s, u, u') is the
successor measure of "repeat one noise vector forever". Those policies score 0.006 on their
own; GPI's per-state max over 64 of them scores 0.28-0.44 five-task. No arm to date has made
psi hold the value of the policy that is actually deployed.

FB (0.496) does: its backup follows the actor at s'. The earlier task-vector arms here
(0.17-0.25) bootstrapped on a trained actor network whose decodes left the data support.

## What

Under `policy_index=task_vector`, define the policy indexed by w as GPI under w and put its
choice in the backup's action slot:

    u*(s') = argmax_{m <= N} [mean_P - pessimism_penalty * unc] psi_target(s', w, u_m)^T w,
             u_m ~ clip(N(0, I), +-u_clip)

    target_M = psibar(s', w, u*(s')) @ phibar(s'_j)^T   (ensemble-reduced as before)

psi(s, w, u) is then the successor measure of "take u, then act by GPI-under-w forever".
Every candidate is a prior draw decoded by the frozen flow, so the bootstrap action is
in-support by construction; this is EMaQ's backup (max over N behaviour samples) with the
flow prior as the behaviour sampler and N as the improvement/overestimation dial.

Acting is `gpi_select`'s existing task-vector branch (argmax over `gpi_num_u`=64 prior
draws of the same score at w = task_z), i.e. the same operator at N=64.

Code: `agents/psmflow.py` `_gpi_argmax_latent`, the `u_next` branch in
`sample_step_inputs` (key `fold_in(rng, 108)`, so every other draw is untouched),
`BOOTSTRAP_MODES`, `MEASURE_DEFAULTS` backfill, `bootstrap/adv` telemetry
(Q(u*) - mean_m Q(u_m) at s'). `bootstrap=index` reproduces every earlier arm bit-for-bit
(pinned in `tests/test_psmflow_bootstrap.py`, with `test_psmflow_policy_index.py`,
`test_psmflow_backup_explore.py`, `test_psmflow_stabilisers.py` unchanged and passing).

## Hyperparameters

| key | value | note |
|---|---|---|
| basis | `p2_phase1_phi2M/sd00k` @500k, seed k | the phi behind 0.439 / 0.318 / 0.227 (2P-GPI @500k psi) |
| `phi_restore_epoch` / `train_phi` | 500000 / false | 2P recipe |
| `policy_index` | task_vector | index slot carries w |
| `bootstrap` / `bootstrap_candidates` | gpi_argmax / 8 | N = 8 |
| `acting` / `train_actor` | gpi / false | no actor anywhere |
| `psi_form` / `u_clip` / `index_clip` | affine / 3.0 / null | defaults |
| `pessimism_penalty` / `num_parallel` | 0.5 / 2 | exact min over the ensemble in the backup |
| `mix_ratio` | 0.5 | w: half goal-phi(s'), half random unit |
| `discount` | 0.98 | cube default |
| `offline_steps` | 1,000,000 | 2P-GPI rose 250k -> 500k on every seed; 1M is the second reading |
| save / in-loop eval | every 250k / 50k x 50 episodes | eval500 at 500k and 1M, five tasks |
| walltime | 1-12:00:00 | measured 30 it/s on a KISSKI GPU (0.033 s/step): 1M steps in ~9.2 h |

Everything not listed is the affine default (`configs/agent/psmflow.yaml`).

## Pre-registered readings (written before the numbers)

- Five-task mean at 1M above 0.328 (the 2P-GPI @500k mean) with seed spread no worse than
  0.23-0.44: the backup was the missing mechanism; next is N and psi length.
- 0.17-0.25 with stable TD: the task-vector w-coverage problem of the earlier arms (test
  w = E[r phi] outside the training w mix). Diagnosis is w coverage, not the backup.
- `td_target_absmean` past ~5x the 2P runs' values (they read 800-1200 at 500k) or
  non-finite loss: the max over N inside the target overestimates faster than min-ensemble
  pessimism damps it. The dial is N (4) before anything else.
- At or below 0.284 with stable TD and finite `bootstrap/adv`: the measure-as-critic path
  has no mechanism lever left in this substrate.

## Smoke

CPU, 200 steps, the exact launch flag set on the sd001 500k basis (`run_group=smoke_emaq_phi500k`,
`$PSM_DATA/exp_smoke`): exit 0, wall 133 s, 2.4 it/s steady. `flags.json` carries
`bootstrap=gpi_argmax bootstrap_candidates=8 policy_index=task_vector acting=gpi
train_actor=false train_phi=false phi_restore_epoch=500000 psi_form=affine u_clip=3.0
pessimism_penalty=0.5`. `train.csv` logs `training/bootstrap/adv` = 1.58 (step 5) and 1.84
(step 7), finite and positive; `psm_loss` and `td_target_absmean` present and finite.
Unit tests: `tests/test_psmflow_bootstrap.py` 5/5; `test_psmflow_policy_index.py` 10/10,
`test_psmflow_backup_explore.py` 5/5, `test_psmflow_agent.py` 18/18 (+2 Stage-A-gated skips),
`test_psmflow_fixed_coeff.py` 5/5, `test_psmflow_stabilisers.py` 24/24. Ruff: no new messages.

## Runs

Submitted 2026-09-16 14:37 via `scripts/slurm/train_psmflow.sbatch` (group `emaq_phi500k`,
`STEPS=1000000 SAVE_INT=250000 EVAL_INT=50000 EVAL_EPS=50`, walltime 1-12:00:00):

| seed | job | basis (phase-1 run @500k) |
|---|---|---|
| 0 | 2518887 | `p2_phase1_phi2M/sd000_s_2518506.0.20260916_000009` |
| 1 | 2518888 | `p2_phase1_phi2M/sd001_s_2518507.0.20260916_000111` |
| 2 | 2518889 | `p2_phase1_phi2M/sd002_s_2518508.0.20260916_000133` |

EXTRA = `agent.phi_restore_path=<basis> agent.phi_restore_epoch=500000 agent.train_phi=false
agent.policy_index=task_vector agent.bootstrap=gpi_argmax agent.bootstrap_candidates=8
agent.acting=gpi agent.train_actor=false`.

Started 15:34 / 15:44 / 15:50 (sd0/1/2); each run's flags.json re-read after start carries
`bootstrap=gpi_argmax bootstrap_candidates=8 policy_index=task_vector acting=gpi
train_actor=false train_phi=false phi_restore_epoch=500000`. At 10k steps:
`bootstrap/adv` 26.3 / 27.6, `td_target_absmean` 385 / 386, `psm_loss` 787 / 801 (sd0 / sd1).

Evals: `/tmp/psmflow_emaq/emaq_evals.sh` submits `scripts/slurm/eval500.sbatch` for tasks 1-5 at
psi 250k, 500k, 750k and 1M as each checkpoint lands; reports land as
`$PSM_DATA/logs/emaq_phi500k_sd00{k}_{500000,1000000}_task{t}.json`. Only the five-task mean
is quoted. Comparators on the same bases: 2P-GPI @500k psi 0.439 / 0.318 / 0.227 (mean
0.328); joint affine GPI 0.284; BC 0.111; FB 0.496.

## Results

### psi 250k (500 episodes, five tasks)

| seed | t1 | t2 | t3 | t4 | t5 | five-task |
|---|---|---|---|---|---|---|
| sd0 | 0.570 | 0.008 | 0.262 | 0.016 | 0.016 | 0.174 |
| sd1 | 0.900 | 0.028 | 0.198 | 0.020 | 0.018 | 0.233 |
| sd2 | 0.828 | 0.066 | 0.550 | 0.008 | 0.008 | 0.292 |

Mean 0.233. Same bases, 2P-GPI @250k: 0.432 / 0.239 / 0.190 (mean 0.287), with task 1 at
0.33 / 0.16 / 0.24 and task 2 at 0.72 / 0.53 / 0.14.

Training at 275-290k: `td_target_absmean` 548-605, `psm_loss` 813-868, `bootstrap/adv`
36-39, all finite; no divergence. In-loop 50-episode success on the default task 0.00-0.04
at 250k.

What it shows: task 1 reaches 0.57-0.90, above every task-1 cell in the project (2P-GPI best
0.56). Task 2, the task GPI solves best, is at 0.01-0.07. Tasks 4 and 5 are at the floor.
The five-task mean lands in the pre-registered 0.17-0.25 band with stable TD. What it does
not show yet: whether the policy depends on the task vector at all -- one task-independent
behaviour that solves task 1 would produce exactly this profile. Checked next.

### psi 500k and 750k (500 episodes, five tasks)

| psi step | seed | t1 | t2 | t3 | t4 | t5 | five-task |
|---|---|---|---|---|---|---|---|
| 500k | sd0 | 0.296 | 0.022 | 0.258 | 0.028 | 0.020 | 0.125 |
| 500k | sd1 | 0.780 | 0.190 | 0.296 | 0.008 | 0.014 | 0.258 |
| 500k | sd2 | 0.314 | 0.016 | 0.342 | 0.008 | 0.010 | 0.138 |
| 750k | sd0 | 0.212 | 0.038 | 0.292 | 0.050 | 0.004 | 0.119 |
| 750k | sd1 | 0.734 | 0.430 | 0.648 | 0.026 | 0.008 | 0.369 |
| 750k | sd2 | 0.742 | 0.002 | 0.438 | 0.016 | 0.002 | 0.240 |

Means: 250k 0.233 ± 0.146, 500k 0.173 ± 0.182, 750k 0.243 ± 0.311. Same bases, 2P-GPI:
250k 0.287, 500k 0.328, 1M 0.250. TD stable throughout (`td_target_absmean` 649-846 at
~830k, `bootstrap/adv` 35-42, `psm_loss` 984-1525, all finite).

What it shows: the mean sits in the 0.17-0.25 pre-registered band at every step, at or below
2P-GPI on the same bases. Task 1 is consistently the strongest task (0.21-0.90), the reverse
of GPI's profile (task 2 strongest). One seed (sd1) develops task 2 over psi steps
(0.03 -> 0.19 -> 0.43) and reaches 0.369 five-task at 750k, the second-best single cell
in the project; the other two seeds stay near 0 on task 2. Tasks 4 and 5 are at the floor
in every cell. Seed spread (0.12-0.37 at 750k) exceeds the between-step movement.

### psi 1M (500 episodes, five tasks) -- runs complete

| seed | t1 | t2 | t3 | t4 | t5 | five-task |
|---|---|---|---|---|---|---|
| sd0 | 0.416 | 0.124 | 0.490 | 0.030 | 0.008 | 0.214 |
| sd1 | 0.768 | 0.268 | 0.528 | 0.024 | 0.012 | 0.320 |
| sd2 | 0.858 | 0.022 | 0.666 | 0.012 | 0.002 | 0.312 |

Ladder, five-task mean over 3 seeds: 250k 0.233, 500k 0.173, 750k 0.243, 1M 0.282 ± 0.147.
The 1M point is the highest and the runs stopped there (1M budget). Same bases, 2P-GPI:
0.287 / 0.328 / 0.250 / 0.274 at 250k / 500k / 1M / 2M; joint affine GPI 0.284; BC 0.111;
FB 0.496.

Reading against the pre-registered outcomes: the 1M mean (0.282) equals the joint GPI
reference and sits below 2P-GPI at 500k (0.328); the interval covers both. TD stable
throughout, `bootstrap/adv` finite. The task profile is the reverse of GPI's and stable
across seeds: task 1 0.42-0.86 and task 3 0.49-0.67 (GPI's best cells 0.47 / 0.74), task 2
0.02-0.27 (GPI 0.56-0.79), tasks 4-5 at the floor in every cell. The mean is limited by task
2 and by tasks 4-5, not by tasks 1 and 3. What it does not show: whether the rise from 750k
to 1M continues (no checkpoints past 1M).

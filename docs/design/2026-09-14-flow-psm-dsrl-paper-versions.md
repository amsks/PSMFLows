# Flow + PSM + DSRL: the two paper versions, what is implemented, and today's arms

Date: 2026-09-14. Source: `git show 5249267:PAPER/main.tex` (the three-rung ladder, Sec.
"Zero-shot inference", and Sec. "LatentFlowPSM"), `agents/psmflow.py`,
`configs/agent/psmflow.yaml`, `docs/reference/psmflow-symbols.md`.

Both versions share the same substrate. A behaviour flow `G(s,u)` is fit by conditional
flow matching and frozen. Every transition has a cached latent `u_i` with `G(s_i,u_i) ~ a_i`
(Stage B). Every action the agent evaluates, bootstraps or executes is a flow decode. The
task vector is `w = project(E_D[r(x) phi(x)])`, inferred once from 10k relabel rows with the
reward shifted by 1.0 (`infer_z`).

## Version 1: the three-rung ladder

`psi(s, u, u')` is the successor feature of "emit latent `u` now, then follow the constant-
index policy `pi_{u'} = G(., u')`". `Q_w(s,u) = psi(s,u,u')^T w`.

| rung | what the paper says | config keys | implemented |
|---|---|---|---|
| 1, flow-GPI | draw K indices `u'_j ~ p0` and K actions `u_i ~ p0`; act with `G(s, u_i*)` at `argmax_{i,j} psi(s,u_i,u'_j)^T w`; no test-time training | `psi_form=affine policy_index=latent train_actor=false acting=gpi gpi_num_u=64 gpi_select=argmax` (the defaults) | yes; this is `affine_strict_cube` |
| 2, in-sample latent Q-iteration | initialise `Q_w(s,u) <- max_{u'} psi(s,u,u')^T w`; iterate `Q(s,u) <- r_hat(s') + gamma max_{u' ~ p0} Q(s',u')` with `r_hat = phi(s')^T w`, `(s,u)` from the cached latents, `u'` from the prior box; Prop. `rung2` says it removes the `2 eps/(1-gamma)` GPI slack | none | no. Nothing in the repo iterates a scalar `Q_w` on `phi^T w`. The closest objects are `index_agg=expectile` (an expectile over `u'` of the measure readout, still a measure fit, not a reward-driven backup) and `dsrl_na` (a Bellman critic, but over raw actions and on the dataset reward) |
| 3, amortised latent actor | distil `s -> argmax_u Q_w(s,u)` into `pi_eta(s,w)` across sampled `w`; deploy `G(s, pi_eta(s,w,eps))` | `actor_mode=gpi_distill train_actor=true acting=actor` distils the rung-1 argmax; `actor_mode=dsrl_sac` + `dsrl_na.enabled=true` distils `Q_A` into `Q_W` and climbs it | partly: the distillation targets exist for rung 1 and for `dsrl_na`; there is no rung-2 `Q_w` to distil from |

The paper's box is the chi-square typical set `U_delta`; the code uses `|u|_inf <= u_clip`
(default 3.0). The paper's reward is `phi^T w` at every rung. Rung 1 reads it only through
`Q = psi^T w`. No rung-1 or rung-3 object ever sees the dataset reward.

## Version 2: Section 10, "LatentFlowPSM"

`psi(s, u, w)` is indexed by the task vector. The second slot is the latent emitted now.
The actor `pi_eta(s, w, eps)` is the DSRL noise-space actor: a tanh output scaled to the
box `|u|_inf <= c`, `c = 3`.

| component | what the paper says | config keys | implemented |
|---|---|---|---|
| index | `w_i` per batch element: Gaussian with prob. `1-p`, else `phi(s'_j)` at a permuted `j`, projected to the sphere | `policy_index=task_vector mix_ratio=0.5` | yes (`sample_step_inputs`) |
| bootstrap | continuation latent at `s'` is the actor's, `u+ = pi_eta(s', w, eps)` | `policy_index=task_vector train_actor=true` (under `latent` the bootstrap is the prior draw `u'`) | yes |
| pessimism | target = ensemble mean minus `kappa` times std; the actor's Q likewise | `num_parallel=2 pessimism_penalty=0.5 actor_pessimism_penalty=0.5` (0.5 with P=2 is the exact min) | yes |
| actor loss | `-mean Q / |mean Q| + lambda_bc |u_a - u_tilde|^2 + CFM toward the dataset latents` | `actor_mode=ddpg actor.bc_coeff=1.0 lr_actor=1e-4` | yes (`flow_actor_loss`, audited 2026-09-03) |
| box | `|u|_inf <= 3` on actor output, dataset latents and draws | `u_clip=3.0` | yes |
| deployment | `a = G(s, pi_eta(s, w, eps))`, `eps ~ N(0,I)` | `acting=actor` | yes |
| head | the paper writes a free `psi(s,u,w)` | `psi_form=free` | yes. `psi_form=affine` asserts `policy_index=latent` (`agents/psmflow.py:409`): the affine head encodes its index slot into `w(u')` and requires `A`, `beta` independent of the policy index, so the Section 10 index cannot be placed in it without a change to the head |

The free-psi Section 10 arm is the pre-2026-09-04 shipped agent (`LEGACY_AGENT_DEFAULTS`
in `tools/eval_checkpoint.py`).

## Measured so far (500 episodes, cube-single-play, BC control 0.072)

| arm | version | number | source |
|---|---|---|---|
| affine GPI (`affine_strict_cube`, late-window mean 250k-500k x 3 seeds) | 1, rung 1 | 0.424 +/- 0.076 (n=18) | `docs/tables/results.md` |
| free-psi Section 10 arm (point preimage, canonical npz) | 2 | 0.230 +/- 0.051 (n=2) | `docs/tables/results.md` |
| DSRL-NA, real reward (`dsrlna_cube`, 3 seeds x 250k/500k) | neither (dataset reward) | 0.910 | HANDOFF 09-09 3b |
| DSRL-NA, inferred reward (`affine_dsrl_na_cube`, 09-07) | rung 3 on the measure readout | 0.307 +/- 0.091 (n=15) | HANDOFF 09-07 |

The 09-13 audit (`docs/design/2026-09-13-psm-interface-audit.md`) found the two in-loop
inferred-reward arms (D1b, D2) defective: phi kept training under the held `w`, and the
task's termination masks stayed on.

## Arms launched today

All three use the `dsrlna_cube` recipe (`scripts/slurm/launch_dsrl_na.sh`, `REWARD_SOURCE=real`)
with the dataset `rewards` array replaced through `dataset.reward_override_path`
(`tools/relabel_reward_rhat.py`). `r_hat = phi(s')^T w` from
`affine_strict_cube/sd001 @500k`, `w` inferred exactly as at eval, computed once. Masks
are the dataset's (`1 - success` on cube), as in the 0.910 run. phi does not train the reward:
the reward is a constant array.

| arm | group | jobs | reward file | pre-registered expectation |
|---|---|---|---|---|
| raw-scale r_hat | `cube_dsrlna_rhat_frozen` | 2518095, 2518096, 2518097 | `.../cube-single-play_rhat_affine_strict_cube_sd001_500k.npz`: mean 3.95, std 10.58, range [-32.0, 62.6] | low. The per-step reward is positive (mean 3.95) while the mask ends the episode at success, so surviving is worth more than reaching the goal; 74x the real reward's std |
| scale-matched r_hat | `cube_dsrlna_rhat_scaled` | 2518113, 2518114, 2518115 | `..._500k_scaled.npz` = `0.00445 r_hat + 0.00325 - 1`: mean -0.979, std 0.047, range [-1.139, -0.718] | near 0.9: the inferred reward vector is adequate and rung 2 (a reward-driven backup) is the missing piece. Near 0.3: the reward vector is the bottleneck |
| Section 10 on the affine head | `cube_sec10_affine_actor` | not launched | -- | 0.2-0.5 expected; >= 0.53 would match FB+flowbc. The CPU smoke fails at `create`: `AssertionError: psi_form=affine requires policy_index=latent` (`agents/psmflow.py:409`). Not patched |

Relabel fit of `r_hat` against the shifted reward, all 1M rows: Pearson 0.330, R^2 under the
best affine rescale 0.109, top-1% precision 0.369 against a base rate 0.021. The correlation
is scale-free, so it is the same for both files.

Both DSRL-NA arms are `--time=05:00:00`, 500k steps, eval every 50k with 50 episodes, save
every 50k. The 500-episode evals at 250k and 500k are what count, against the 0.910 control
already on disk (`$PSM_DATA/logs/dsrlna_cube_sd00{0,1,2}__{250000,500000}.json`).

## Section 10 arms launched (2026-09-14, later the same day)

The guard at `agents/psmflow.py` `create` that refused `psi_form=affine` with
`policy_index=task_vector` is removed. `AffinePsiMap.w_enc`'s input width is inferred at
init from the index sample, which is the z_dim task vector under `task_vector`, so the
head encodes the task vector into `w(.)` and `A(s,u)`, `beta(s,u)` are unchanged
(`tests/test_psmflow_affine.py::test_affine_with_a_task_vector_index_encodes_the_task_vector`).
Both arms are the `affine_strict_cube` template (`actor_mode=ddpg`, `u_clip=3.0`,
`pessimism_penalty=0.5`, `num_parallel=2`, `mix_ratio=0.5`, `actor.bc_coeff=1.0`,
`batch_size=1024`, `discount=0.98`) plus `policy_index=task_vector train_actor=true
acting=actor`; the free arm adds `psi_form=free`. 500k steps, eval every 50k with 50
episodes, save every 50k, `--time=05:00:00`.

| arm | group | jobs | expectation |
|---|---|---|---|
| free psi, Section 10 literal | `cube_sec10_free_actor` | 2518116, 2518117, 2518118 | at or near the 09-04 arm (0.230, single task); it is that arm re-run on this checkout |
| affine psi, Section 10 index | `cube_sec10_affine_actor` | 2518119, 2518120, 2518121 | 0.2-0.5 five-task mean; >= 0.50 matches FB |

CPU smokes (200 steps) passed for both: free EXIT 0 in 181 s, affine EXIT 0 in 284 s.

## Results (2026-09-14 evening; full run bookkeeping in `docs/HANDOFF.md`, same date)

All twelve jobs COMPLETED. Eval JSONs, 500 episodes each at `restore_epoch=500000`:
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

## Antmaze repeat launched (2026-09-14)

Templates. The real-reward control is `dsrlna_antmaze` (jobs 2492660-62, HANDOFF 09-10 §1):
the `launch_dsrl_na.sh` recipe with `agent.discount=0.99`, `antmaze-medium-navigate-singletask-v0`
(OGBench task 1, the maze default), preimage npz `antmaze-medium-navigate.npz`; six 500-episode
cells 0.938/0.982, 0.972/0.978, 0.956/0.976 (sd0-2 at 250k/500k), pooled 0.967; 4 h 30 wall
clock per seed. The affine template is `affine_strict_antmaze_g99` (discount 0.99, otherwise
the cube affine defaults; late-window 250k-500k means per seed 0.162 / 0.164 / 0.464, 30-cell
ladder 0.294 ± 0.070). The antmaze npz has the same convention as cube: rewards -1/0
(9,113 success rows of 1M), masks = 1 - success.

Relabel: `affine_strict_antmaze_g99/sd002 @500k`, `--match_real_scale`, output
`$PSM_DATA/rewards/antmaze-medium-navigate_rhat_affine_strict_antmaze_g99_sd002_500k_scaled.npz`.

| statistic (1M rows) | value |
|---|---|
| Pearson corr, r_hat vs shifted reward | 0.286 |
| R^2 under the best affine rescale | 0.082 |
| top-1% precision / base rate | 0.291 / 0.0091 |
| raw r_hat mean / std | 8.21 / 10.50 |
| scaled output mean / std / min / max | -0.9909 / 0.0272 / -1.0744 / -0.7910 |
| scale / offset | 0.002591 / -0.012158 |

| arm | group | jobs | template + change | expectation |
|---|---|---|---|---|
| scale-matched r_hat | `antmaze_dsrlna_rhat_scaled` | 2518182, 2518183, 2518184 | `dsrlna_antmaze` flags; only `dataset.reward_override_path`; `--time=08:00:00` | near the real-reward control (0.967) if the inferred reward vector is adequate; near the zero-shot 0.294 if the reward vector is the bottleneck |
| Section 10 on the affine head | `antmaze_sec10_affine_actor` | 2518179, 2518180, 2518181 | `affine_strict_antmaze_g99` + `policy_index=task_vector train_actor=true acting=actor`; `--time=05:00:00` | five-task mean in the 0.1-0.3 band. References: antmaze BC 0.072 (task 1, 500 ep); affine GPI antmaze g99 0.294 ± 0.070 (task 1, 30-cell ladder). No strict antmaze five-task matrix is recorded in the docs |

CPU smokes (200 steps) passed for both: Section 10 EXIT 0 in 126 s; scaled DSRL-NA EXIT 0 in
113 s with the override line `rewards mean/std -0.9909/0.0950 -> -0.9909/0.0272`.

### Antmaze results (2026-09-14 night; run bookkeeping and ladders in `docs/HANDOFF.md`, same date)

All six jobs COMPLETED (arm 1 4h29–4h32, arm 2 3h43–3h45 wall clock). Eval JSONs, 500
episodes, `restore_epoch=500000`:
`$PSM_DATA/logs/antmaze_dsrlna_rhat_scaled_sd00{0,1,2}_500000_task1.json`,
`$PSM_DATA/logs/dsrlna_antmaze_sd00{0,1,2}__500000.json` (control),
`$PSM_DATA/logs/antmaze_sec10_affine_actor_sd00{0,1,2}_500000_task{1..5}.json`.

Mechanism table: task 1, 500 episodes, 500k checkpoint, t interval over 3 seeds (df 2).

| arm | reward channel | sd0 | sd1 | sd2 | mean ± 95% |
|---|---|---|---|---|---|
| scaled r_hat | `phi^T w` mapped to −1/0 | 0.888 | 0.966 | 0.704 | **0.853 ± 0.334** |
| control `dsrlna_antmaze` | dataset reward | 0.982 | 0.978 | 0.976 | **0.979 ± 0.008** |

Zero-shot table: five tasks, 500 episodes per task, 500k checkpoint, t interval over 3 seeds (df 2).

| arm | sd0 | sd1 | sd2 | mean ± 95% | source |
|---|---|---|---|---|---|
| BC (frozen flow alone) | | | | 0.072 | `docs/tables/results.md` (task 1, 500 ep) |
| Section 10, affine psi | 0.109 | 0.113 | 0.094 | **0.105 ± 0.025** | HANDOFF 09-14 night |
| FB | | | | 0.730 | TD-JEPA Table 1 |
| HILP | | | | 0.836 | TD-JEPA Table 1 |

No five-task GPI row exists for antmaze; the affine GPI antmaze number on record is task 1
only (`affine_strict_antmaze_g99`, 0.294 ± 0.070). Section 10 affine per task (mean over
seeds): 0.177 / 0.075 / 0.089 / 0.053 / 0.132 on tasks 1–5.

Verdicts. (a) The scaled inferred reward reaches 0.853 vs the control's 0.979 on task 1, a
0.13 gap driven by seed 2 (0.704); cube's gap was 0.015. The larger gap goes with the weaker
relabel statistics (Pearson 0.286 vs 0.330). The scaled reward still scores 2.9x the
measure-critic route (0.294). (b) Section 10 affine on antmaze is 0.105 five-task, above BC
0.072, far below FB 0.730; the pre-registered band was 0.1–0.3. (c) The isolation to the
measure critic holds on both envs: scalar Bellman critic on the scaled r_hat reaches the
real-reward level (cube 0.867, antmaze 0.853), the measure `psi^T w` as critic gives 0.1–0.3
(cube 0.246 / 0.284, antmaze 0.105 five-task / 0.294 task 1).

## Reference PSM critic (proto stage) on latent inputs

Added 2026-09-14 as `agent.proto.enabled` (default false). It replaces the Section 10
critic with the REFERENCE PSM critic (arXiv 2411.19418; `archive/agents/psm.py` and
Factored-FB `impls/critics/psm.py` are the two ports it copies) while every other piece of
the pipeline stays as it is: frozen flow `G(s,u)`, point preimages `u_data` in the action
slot, closed-form `infer_z`, the ddpg noise-space actor (`train_actor=true acting=actor
actor_mode=ddpg`), `task_w` mixing at `mix_ratio`, `P=2` with `targets_uncertainty`.

Objects.

| object | definition | code |
|---|---|---|
| `z_bin` | binary code of width `max_log_seed`, one policy of the proto family; a uniform integer in `[0, 2^max_log_seed)` unpacked LSB-first, redrawn per row per update | `utils/psm_proto.sample_z_bin`, `StepInputs.z_bin` |
| `seed` | `(z_bin . powers + row) mod (2^max_log_seed + 20000)`, `powers` the reversed power list so the bit array reads MSB-first; `row` is the dataset ROW index `batch['index']`, not the batch position, so a transition meets the same proto policy on every resample | `proto_seed_ints` |
| `u_proto` | the proto policy's LATENT at that seed: `clip(N(0, I) at fold_in(PRNGKey(proto_seed), seed), +-u_clip)`. The reference gathers row `seed` of a uniform action table in `[-2, 0]`; here the same pure function of (row, code) yields a prior-typical latent, so the bootstrap action `G(s', u_proto)` is a flow decode. `proto_seed` is a constant of the code (0), not the run seed, as the reference's table is | `proto_latents`, `StepInputs.u_proto` |
| `proto_psi(s, z_bin, u)` | the proto successor tower, a `PsiMap` of the SF head's shape with a `max_log_seed`-wide index slot, `P`-fold, with a polyak target | fields `proto_psi`, `target_proto_psi`; Adam at `proto.lr` |
| `psi(s, w, u)` | the existing head, now PSM's separate reward-conditioned SF head | field `psi` under `policy_index=task_vector` |

Losses.

- Proto stage, `proto_measure_loss`: `M = proto_psi(s, z_bin, u_data) phi(s')^T`, target
  `proto_psi_bar(s', z_bin, u_proto) phi_bar(s')^T` reduced to ensemble mean minus
  `pessimism_penalty` times spread, through `contrastive_loss` at `discount`, plus
  `ortho_coef * ortho_loss(phi(s'))` (`proto.ortho_coef`, null = the agent's `ortho_coef`;
  `ortho_mode` applies here). Gradients reach phi and proto_psi.
- SF stage, `measure_loss` with `proto.enabled`: `M = psi(s, w, u_data) phi(s')^T` with phi
  STOP-GRADDED; target `psi_bar(s', w, u_next) phi(s')^T` where phi is the ONLINE phi the
  proto stage just stepped (reference `_update_sf` line 499, not `target_phi`), `u_next =
  pi_eta(s', w, eps)` the actor's latent; same ensemble reduction. No ortho term (the
  reference multiplies its copy by a literal 0; it is logged as telemetry). Gradient reaches
  psi only.
- Actor: `flow_actor_loss` unchanged, reading psi at the pre-update params as before.

Update order (`apply_update`): proto step (phi + proto_psi) -> polyak their targets -> SF
step (psi) against the post-proto phi -> polyak psi's target -> actor step. The order and
the post-proto read are pinned by
`tests/test_psmflow_psm_ref.py::test_apply_update_steps_sf_against_the_post_proto_phi`,
which replays the sequence by hand and checks the params bitwise.

Config (`configs/agent/psmflow.yaml`, mirrored in `get_config`): `proto.enabled` (false),
`proto.max_log_seed` (16), `proto.proto_seed` (0), `proto.lr` (1e-4), `proto.ortho_coef`
(null). `create` requires `policy_index=task_vector`, `train_actor=true`, `train_phi=true`
and `1 <= max_log_seed <= 30`; `main.py` sets `dataset.return_index=True` for the arm.
With `enabled=false` the params after 3 updates equal a baseline captured at d51de4c on
both the strict and the actor arm, and no `proto_*` key is logged.

First arm: `cube_psmref_actor` = the `cube_sec10_free_actor` flags plus
`agent.proto.enabled=true`; only the critic changes.

Result (2026-09-15; run bookkeeping, per-task table and ladder in `docs/HANDOFF.md`,
same date). `cube_psmref_actor` (jobs 2518213/14/15, 500k, five tasks, 500 episodes per
cell) scores 0.156 ± 0.023 (t interval over 3 seeds, df 2; seed means 0.166 / 0.150 /
0.151; per task 0.126 / 0.201 / 0.263 / 0.139 / 0.051). The pre-registered expectation was
a number above the free Section 10 template's 0.171 ± 0.027, with 0.35 as the mark for
"the proto stage was the missing piece" and 0.17–0.25 for "the reference critic inherits
the same limit on latent inputs". The number landed below the band. The proto stage does
not rescue the measure critic on latent inputs: with the actor, the flow, the preimages
and the optimiser held, swapping the Section 10 critic for the reference PSM critic moves
the five-task mean by −0.015, and the seed intervals overlap. The affine Section 10 arm
(0.246 ± 0.041), BC (0.111) and FB (0.496) are the comparators. With the loss form matched
to the reference that scores 694 on Walker, the remaining candidates for the
Walker/OGBench difference are the data (ExORL RND against OGBench play), the action space
(raw against flow latent) and the template's optimiser and regulariser values
(`ortho_coef=1000`, `lr_phi=1e-5`, `pessimism_penalty=0.5`, 500k), which this arm kept. The
scalar-grounded route (DSRL-NA on `phi^T w`) remains the only composition above 0.5 on
either env; its task-conditioned zero-shot form is the proposed next arm, and an eval-only
comparison of `psi^T w` against the scalar Q from `cube_dsrlna_rhat_scaled` on the same
`(s, u)` is the proposed next measurement.

## Diagnostic: measure value vs scalar Q (2026-09-15)

Eval-only. `tools/diag_measure_vs_scalar_q.py`, job 2518241 (2 min 25 s on one H100),
report `$PSM_DATA/logs/diag_measure_vs_scalar_q_cube_sd001.json` (+ `.npz` with the
per-(state, latent) tensors). Two checkpoints, one reward:

| object | source |
|---|---|
| measure | `affine_strict_cube/sd001 @500k`: psi(s,u,u'), phi; w from the raw reward file's sidecar (cosine 0.999997 to the w re-inferred with the eval's own seeding); gamma 0.98, P 2, kappa 0.5, u_clip 3.0 |
| scalar | `cube_dsrlna_rhat_scaled/sd001 @500k`: qa(s,a) ensemble of 2 by TD on `r_scaled = 0.00445 r_hat + 0.00325 - 1`, gamma 0.99, u_clip 1.5; qw(s,u) distilled from qa at decodes |
| data | 10,000 uniform rows of `cube-single-play.npz` (13 invalid preimages excluded) + 791 extra success rows, so 1,000 success rows in the pool (209 in the uniform set); per row K = 64 latents u ~ N(0,I) clipped to 3 and 64 indices u' likewise; ROW_SEED 20260915, PANEL_KEY 4242 |
| values | `Vm_max(s,u) = max_{u'} [mean - 0.5 unc] psi(s,u,u')^T w` (the GPI value); `Qa(s,u) = min_e qa(s, G(s,u))`, one-step decode; `Qw(s,u)` |

r_hat recomputed on the GPU matches the raw file at correlation 0.999999 (max abs diff
0.079 on values with std 12). `all` is the uniform 10k; `success` the 1,000 pool rows with
reward 0 at s'; `nonsuccess` the 9,791 uniform rows with reward -1.

**A. Bellman residual of the measure along w** (lhs = psi(s,u_data,u')^T w; rhs = gamma
[mean - kappa unc](psi_bar(s',u',u')^T w) + phi(s')^T w; one u' per row).

| split | mean lhs | std lhs | residual mean | residual RMS | RMS / std lhs | RMS / std, plain-mean target | scaled: RMS / std |
|---|---|---|---|---|---|---|---|
| all | -3566 | 3897 | -25.2 | 162.7 | 0.042 | 0.043 | 0.042 |
| success | -2869 | 4494 | -29.3 | 154.0 | 0.034 | 0.035 | 0.034 |
| nonsuccess | -3586 | 3928 | -24.9 | 163.6 | 0.042 | 0.043 | 0.042 |

Full 128-d vector residual (per-component pessimistic target), RMS(res.d)/std(lhs.d) for
unit directions d:

| split | d = w/\|w\| | 20 random d: mean | min | max | \|res\| RMS / \|lhs\| RMS |
|---|---|---|---|---|---|
| all | 0.045 | 0.067 | 0.037 | 0.111 | 0.043 |
| success | 0.036 | 0.070 | 0.031 | 0.126 | 0.038 |
| nonsuccess | 0.045 | 0.067 | 0.037 | 0.111 | 0.043 |

The measure satisfies its own Bellman equation along w to 4% of the readout's spread; the
w direction is fitted at least as well as an average direction.

**B. Global agreement** (Pearson P / Spearman S).

| split | Vm_max vs Qa, (s,u_data) | Vm_max vs Qw, (s,u_data) | Vm_max vs Qa, panel pooled | Vm_max vs Qa, panel state-centred | Vm_mean vs Qa, (s,u_data) |
|---|---|---|---|---|---|
| all | P 0.049 / S 0.103 | P 0.049 / S 0.102 | P 0.049 / S 0.102 | P 0.125 / S 0.122 | P -0.140 / S -0.249 |
| success | P 0.039 / S 0.086 | P 0.069 / S 0.247 | P 0.051 / S 0.114 | P 0.014 / S 0.013 | P 0.098 / S 0.017 |
| nonsuccess | P 0.038 / S 0.079 | P 0.037 / S 0.078 | P 0.037 / S 0.078 | P 0.128 / S 0.125 | P -0.233 / S -0.302 |

Levels (all): Vm_max mean -880 with within-state std over the 64 u of 62.9; Qa mean -29.6
with within-state std 0.41; Qw within-state std 0.40. Across the 10k dataset rows the two
values correlate at 0.05 (Pearson) and 0.10 (Spearman).

**C. Per-state ranking over the 64 u** (Spearman between Vm_max(s,.) and Qa(s,.); regret =
(max Qa - Qa at the measure argmax) / (max Qa - min Qa); a random pick scores 8/64 = 0.125
on top-8 and ~0.48 on regret).

| split | rho mean | median | q25 | q75 | frac rho > 0.3 | frac rho < 0 | argmax in Qa top-8 | regret mean | regret median | regret random |
|---|---|---|---|---|---|---|---|---|---|---|
| all, vs Qa | 0.123 | 0.173 | -0.273 | 0.544 | 0.416 | 0.392 | 0.332 | 0.428 | 0.384 | 0.483 |
| all, vs Qw | 0.124 | 0.173 | -0.268 | 0.541 | 0.416 | 0.392 | 0.327 | 0.429 | 0.387 | 0.486 |
| success, vs Qa | 0.010 | 0.012 | -0.307 | 0.356 | 0.287 | 0.489 | 0.195 | 0.484 | 0.480 | 0.479 |
| success, vs Qw | 0.086 | 0.108 | -0.191 | 0.399 | 0.332 | 0.409 | 0.169 | 0.462 | 0.442 | 0.449 |
| nonsuccess, vs Qa | 0.125 | 0.176 | -0.273 | 0.547 | 0.418 | 0.391 | 0.335 | 0.427 | 0.382 | 0.483 |

Vm_mean (mean over u' instead of max) vs Qa, all: rho mean -0.106, top-8 0.240, regret
0.539 (random 0.483). Restricted to the 48.7% of panel latents inside the scalar run's box
(|u|_inf <= 1.5): rho mean 0.117 (all), 0.002 (success). Per state, the measure's ranking of
64 latents agrees with the scalar critic's at rho 0.12 on average, with 39% of states
negative; its argmax lands in the scalar top-8 at 0.33 against 0.125 by chance and recovers
11% of the regret range over a random pick (0.428 vs 0.483). On success rows the agreement
is at chance on every number.

**D. Success split.** Above. The measure's readout lhs is -2869 on success rows against
-3586 elsewhere (r_hat 27.3 vs 3.5); Qa is -1.06 against -30.2.

**E. Coverage** (L2 distance of the 64 u to u_data(s); panel mean distance 3.09; the
measure's argmax sits at 3.58, in the near half 28% of the time).

| split | rho, near half | rho, far half | pooled rho by distance quartile [0.15,2.38] / [2.38,3.03] / [3.03,3.74] / [3.74,8.05] | regret, argmax near | regret, argmax far |
|---|---|---|---|---|---|
| all, vs Qa | 0.117 | 0.124 | 0.119 / 0.119 / 0.121 / 0.128 | 0.423 | 0.430 |
| success, vs Qa | -0.003 | 0.016 | 0.006 / 0.011 / 0.015 / 0.015 | 0.483 | 0.485 |
| nonsuccess, vs Qa | 0.119 | 0.126 | 0.121 / 0.120 / 0.123 / 0.132 | 0.422 | 0.428 |

The agreement is the same near u_data as far from it; the measure's argmax prefers
latents farther from the dataset latent than the panel average (3.58 vs 3.09).

**F. Sanity.** Scalar critic's own Bellman residual on r_scaled (target: r_scaled +
0.99 mask min_e qa_bar(s', G(s', pi(s'))), actor's sampled latent):

| split | q mean | residual mean | RMS | std q | RMS / std q | qa vs qw pooled P / S | qa vs qw state-centred P / S | per-state rho |
|---|---|---|---|---|---|---|---|---|
| all | -29.6 | -0.10 | 0.397 | 9.63 | 0.041 | 0.999 / 0.999 | 0.849 / 0.846 | 0.823 |
| success | -1.05 | -0.18 | 0.395 | 0.36 | 1.09 | 0.663 / 0.329 | 0.086 / 0.080 | 0.073 |
| nonsuccess | -30.2 | -0.10 | 0.398 | 8.76 | 0.046 | 0.999 / 0.999 | 0.866 / 0.863 | 0.840 |

The scalar critic's residual is 4% of its spread on the uniform rows; on success rows
(mask 0, target = r_scaled alone, q within 0.36 of a constant) the relative number is 1.09
on an absolute RMS of 0.40, the same absolute residual as elsewhere. qa-via-decode and qw
agree at 0.999 pooled and 0.82 per state.

Summary of what was measured. Both critics satisfy their own Bellman equation to ~4% of
their spread on the same rows (A, F). The two values of the same (s, u) under the same
reward agree at Spearman 0.10 across states and 0.12 within a state (B, C); the measure's
argmax over 64 latents recovers 11% of the scalar critic's regret range over a random pick,
and none on success rows; the agreement does not change with distance from the dataset
latent (E). Not measured here: which of the two values is closer to the true return.

## Fixes launched 2026-09-15

Three responses to the diagnostic above, in the order the plan set: Fix 3 (config only),
Fix 1 (small code), Fix 2 (code). Branch `fix/psmflow-paper-strict`, worktree
`.claude/worktrees/psmflow-fix`; commits 98539ec (Fix 1) and 2cbd723 (Fix 2). Every run:
cube-single-play, flow `$PSM_DATA/flow/cube-single-play` @500000, preimages
`$PSM_DATA/preimages/cube-single-play.npz`, 500k steps, eval every 50k with 50 episodes,
save every 50k, seeds 0 1 2, one GPU per seed, `scripts/slurm/train_psmflow.sbatch` with
`EXTRA` holding every key that differs from the yaml. Before each launch: a 200-step CPU
smoke of the same hydra line at batch 64 (logs `$PSM_DATA/logs/cpusmoke/<group>.out`,
`EXIT=0` on all three), and after launch every run's `flags.json` was re-read and diffed
against its template (tables below; keys the older template files do not carry appear as
`<absent>` and are at their OFF values in the new runs).

### Fix 3, measure over actions on the Section 10 actor: `cube_fix3_action_measure_actor`

What changes: the affine measure fitted at the recorded action, `F(s, a, c)`, with every
latent query decoded through the frozen flow (`measure_action_input=action`, the
2026-09-11 arm, 0.44 ± 0.38 five-task under `policy_index=latent acting=gpi`), now with the
Section 10 settings `policy_index=task_vector train_actor=true acting=actor`, so the
bootstrap decodes the actor's latent at `s'`: `psibar(s', G(s', u+), w)` with
`u+ = _deploy_latent(s', task_w)`. Why: the diagnostic's within-state latent differences
(std 63) sit below the residual (163), and the action coordinate removes the
inverse-coordinate error from the measure's input. Keys: `agent.measure_action_input=action`
on the `cube_sec10_affine_actor` template, with the 09-11 arm's decode keys restated at their
defaults (`psi_form=affine measure_u_samples=1 gpi_decode=onestep use_point_preimage=true`).

The first submission attempt was refused by `create` (`agents/psmflow.py:476`,
`AssertionError: measure_action_input=action requires policy_index=latent`, the 09-11
plan's "require affine + latent policy index in action mode"). Reading the action-mode
paths showed nothing else assumes a latent index: every psi query goes through
`psi_b -> _measure_input`, which decodes, so `_actor_q` already reads
`psi(s, G(s, u_actor), w)^T w` with gradient through the frozen decoder, the bootstrap
already decodes `u_next`, and `sample_actions` under `acting=actor` decodes the actor's
latent. The guard was replaced by a comment naming both pairings (same reasoning as
ebb2a5c for affine + task_vector); one new guard refuses `psi_dueling` in action mode, whose
advantage baseline feeds prior LATENTS to a tower whose action slot then carries decoded
actions. Test `tests/test_psmflow_action_input.py::test_action_measure_with_the_task_vector_index_reads_the_actor_q_at_decoded_actions`
pins the actor's Q at decoded actions, the bootstrap at the decoded actor latent, and a
finite update; the refusal line was removed from the guard test. Suites: action_input 9,
affine 12, agent 18 (+2 skipped), scalar_dueling 12, all EXIT 0. Smoke `cube_fix3_cpusmoke`
EXIT 0 (200 steps, 7.2 it/s at batch 64). Differing keys against
`cube_sec10_affine_actor/sd000`: `agent.measure_action_input latent -> action` only, on all
three runs.

| item | value |
|---|---|
| jobs | 2518255 (sd0), 2518256 (sd1), 2518257 (sd2), `--time=08:00:00`, started 03:55 |
| run dirs | `$PSM_DATA/exp/PSMFLows/cube_fix3_action_measure_actor/sd000_s_2518255.0.20260915_035530`, `.../sd001_s_2518256.0.20260915_035531`, `.../sd002_s_2518257.0.20260915_035531` |
| rate | 39.0 / 38.4 steps/s on sd0 / sd1 over 90 s at ~3-6k steps (~3.6 h for 500k; the decode in the loss costs nothing visible against the template's 3 h 41 min), inside 08:00:00 |

### Fix 1, task-conditioned scalar critic on the measure's reward, phi frozen: `cube_fix1_scalar_tc`

What changes: the DSRL-NA dual critic (`qa(s, a, w)` scalar TD, `qw(s, u, w)` distilled at
prior latents, tanh-Gaussian latent actor `pi(u | s, w)`) is trained on the synthetic reward
`reward_scale * phi(s')^T w` at a fresh task vector per row, with phi loaded from
`affine_strict_cube/sd001 @500k` and held fixed, and the TD target in the continuing
formulation (mask 1 on every row). Why: the 09-14 mechanism test scored 0.867 on task 2
when a scalar Bellman critic used `phi^T w` at the real reward's scale, against 0.25 for the
measure as critic on the same phi and w, and the diagnostic put the measure's readout at
Spearman 0.10 to that scalar critic; this arm asks whether the scalar route generalises over
`w` without any task reward. Keys (98539ec): `agent.phi_restore_path`,
`agent.phi_restore_epoch` (phi's params from another run's `params_<epoch>.pkl` into phi and
target_phi, optimiser fresh; new, no existing seam restored phi alone), `agent.train_phi=false`,
`agent.dsrl_na.ignore_masks` (a per-w reward offset changes no policy only without
termination; the 09-14 raw-r_hat arm with masks on scored 0.010), `agent.dsrl_na.reward_scale`
(multiplier on the synthetic reward). Eval: `tools/eval_checkpoint.py:347` calls
`infer_eval_z`, `sample_actions` reads `_actor_w(self.task_z)`, which under
`task_conditioned=true` is the inferred `w` itself, so the actor is conditioned on the
label-inferred task vector (the D2 arm's path; `tests/test_psmflow_dsrl_na.py::test_d2_conditions_every_head_on_the_task_vector`).

`reward_scale` constant (`$PSM_DATA/logs/fix1_reward_scale_cube_sd001.json`; phi from the
checkpoint above, 10,000 uniform dataset rows, ROW_SEED 20260915, 64 sphere-projected
`w ~ N(0, I)`, W_SEED 4242):

| quantity | value |
|---|---|
| real reward std over the 10k rows (mean -0.979, 2.07% zeros) | 0.1424 |
| `phi(s')^T w`, per-w std over rows: mean / min / max (sphere w) | 11.28 / 11.13 / 11.49 |
| `phi(s')^T w`, per-w std, w = sphere-projected phi(s') draws (the mixture's other half) | 11.34 |
| `reward_scale` = 0.1424 / 11.28 | **0.01262** |
| in-loop `na_rw_std` at batch 64 (CPU smoke, steps 100-200) | 0.117-0.127 |

Differing keys against the template `cube_dsrlna_rhat_scaled/sd001` (`flags.json`), all
three runs identical except `seed`:

| key | template | `cube_fix1_scalar_tc` |
|---|---|---|
| `agent.dsrl_na.reward_source` | `real` | `synthetic_w` |
| `agent.dsrl_na.task_conditioned` | `False` | `True` |
| `agent.dsrl_na.ignore_masks` | `<absent>` | `True` |
| `agent.dsrl_na.reward_scale` | `<absent>` | `0.01262` |
| `agent.phi_restore_path` | `<absent>` | `$PSM_DATA/exp/PSMFLows/affine_strict_cube/sd001_s_2491601.0.20260904_181115` |
| `agent.phi_restore_epoch` | `<absent>` | `500000` |
| `agent.train_phi` | `True` | `False` |
| `dataset.reward_override_path` | `..._rhat_..._scaled.npz` | `None` |
| `agent.proto.*`, `agent.psm_scalar_coef`, `agent.psi_dueling*` | `<absent>` | OFF values (`False` / `0.0` / `False` / `8`) |

Unchanged from the template: `dsrl_na.{discount 0.99, tau 0.005, lr 3e-4, hidden_dim 2048,
hidden_layers 3, layer_norm, num_ensembles 2, inner_steps 10, n_latent 1}`, `actor_mode=dsrl_sac`,
`actor.{bc_coeff 0, q_coeff 1, entropy auto, target_entropy 0, init_alpha 1, prior_init,
layer_norm, hidden_dim 2048, hidden_layers 3}`, `lr_actor 3e-4`, `u_clip 1.5`, `batch_size 256`,
`psi_form=affine policy_index=latent`, `discount 0.98`.

| item | value |
|---|---|
| jobs | 2518243 (sd0), 2518244 (sd1), 2518245 (sd2), `--time=08:00:00`, started 03:14 |
| run dirs | `$PSM_DATA/exp/PSMFLows/cube_fix1_scalar_tc/sd00{0,1,2}_s_25182{43,44,45}.0.20260915_031441` |
| rate | 44.6k steps at 26 min on sd0 (~29 steps/s, ~4.8 h for 500k) |
| tests | `tests/test_psmflow_scalar_tc.py` 10 passed (EXIT 0): phi restore loads that checkpoint's phi into phi and target_phi and stays unchanged over 3 updates under `train_phi=false`; `ignore_masks` makes the target mask all ones; `reward_scale` multiplies the synthetic reward and leaves `real` alone. `tests/test_psmflow_dsrl_na.py` 44 passed, `tests/test_psmflow_smoke.py` 2 passed |
| smoke | `cube_fix1_cpusmoke`, 200 steps, EXIT 0, 1 min 55 s |
| code state at job start | the worktree at 03:14 carried 98539ec plus the then-uncommitted Fix 2 edits (later 2cbd723) at their OFF values; the flags carry `psm_scalar_coef=0.0 psi_dueling=false`, and `tests/test_psmflow_scalar_dueling.py::test_off_reproduces_the_saved_baseline_bit_for_bit` pins that OFF path to the pre-edit code |

### Fix 2, scalar grounding plus a dueling latent head on the Section 10 affine arm: `cube_fix2_scalar_dueling`, ablation `cube_fix2_scalar_only`

What changes: on `cube_sec10_affine_actor`'s recipe (affine head, `policy_index=task_vector
train_actor=true acting=actor`, 0.246 ± 0.041 five-task), (a) `agent.psm_scalar_coef=1.0`
adds to the measure loss the readout's projected Bellman equation,
`(psi(s,u,z)^T z - gamma [mean - kappa unc](psibar(s',u+,z)^T z) - sg(phi(s'))^T z)^2`
divided by the batch variance of `sg(phi(s')^T z)`, with `z` the row's sampled task vector
and `u+` the measure's own bootstrap latent, gradient to psi only (the ICLR draft's scalar
Bellman equation of the readout; Meta Motivo's `q_loss` on `F^T z`); (b) `agent.psi_dueling=true`
rewrites the affine head as `V(s,z) + Adv(s,u,z) - mean_k Adv(s,u_k,z)` over
`psi_dueling_samples=8` clipped prior latents, V a state-only tower `beta_V(s) + A_V(s)^T w(z)`
sharing `w(z)` with the existing `A(s,u), beta(s,u)` tower. Why: (a) the diagnostic found the
measure's readout along `w` fits its own vector Bellman equation yet is ~uncorrelated with a
scalar critic on the same reward, so the scalar equation is imposed on the readout directly;
(b) the within-state spread over `u` (63) sits below the residual (163), and the dueling
form makes the u-dependent part a zero-mean advantage the state value cannot absorb. Keys
(2cbd723): `agent.psm_scalar_coef`, `agent.psi_dueling`, `agent.psi_dueling_samples`; the
in-loop columns `psm_scalar_loss`, `psm_scalar_target_std`.

Differing keys against the template `cube_sec10_affine_actor/sd000` (`flags.json`), all
six runs identical except `seed` and the one ablation key:

| key | template | `cube_fix2_scalar_dueling` | `cube_fix2_scalar_only` |
|---|---|---|---|
| `agent.psm_scalar_coef` | `<absent>` | `1.0` | `1.0` |
| `agent.psi_dueling` | `<absent>` | `True` | `False` |
| `agent.psi_dueling_samples` | `<absent>` | `8` | `8` |
| `agent.proto.*`, `agent.phi_restore_*`, `agent.dsrl_na.{ignore_masks, reward_scale}` | `<absent>` | OFF values | OFF values |

Unchanged from the template: `psi_form=affine policy_index=task_vector train_actor=true
acting=actor actor_mode=ddpg u_clip=3.0 pessimism_penalty=0.5 num_parallel=2 mix_ratio=0.5
actor.bc_coeff=1.0 lr_actor=1e-4 batch_size=1024 discount=0.98 ortho_coef=1000 lr_phi=1e-5`.

| item | `cube_fix2_scalar_dueling` | `cube_fix2_scalar_only` |
|---|---|---|
| jobs | 2518252 (sd0), 2518253 (sd1), 2518254 (sd2), started 03:46. A first submission at `--time=08:00:00` (2518246, 2518247, 2518248, run dirs `sd00{0,1,2}_s_25182{46,47,48}.0.20260915_0340*`, 1-3k steps) was cancelled at 03:45 once its rate was measured; those three directories are stale and hold no checkpoint | 2518249 (sd0), 2518250 (sd1), 2518251 (sd2), started 03:40 |
| `--time` | 12:00:00. The template took 3 h 41-44 min; measured on sd0 over 90 s, the dueling arm runs at 14.6 steps/s (9.5 h for 500k plus ten in-loop evals), so the plan's 05:00:00 and the first 08:00:00 would both have timed out before the 500k checkpoint the protocol scores | 05:00:00 (40.5 steps/s on sd0, 3.4 h) |
| run dirs | `$PSM_DATA/exp/PSMFLows/cube_fix2_scalar_dueling/sd000_s_2518252.0.20260915_034600`, `.../sd001_s_2518253.0.20260915_034559`, `.../sd002_s_2518254.0.20260915_034600` | `$PSM_DATA/exp/PSMFLows/cube_fix2_scalar_only/sd000_s_2518249.0.20260915_034039`, `.../sd001_s_2518250.0.20260915_034039`, `.../sd002_s_2518251.0.20260915_034043` |
| smoke | `cube_fix2_cpusmoke`, 200 steps, EXIT 0, 1 min 35 s, `psm_scalar_loss` 1.52-1.65 at steps 50-200 (an earlier draft of this row quoted 31-34, a mis-read column) | `cube_fix2ab_cpusmoke`, EXIT 0, 1 min 30 s |
| tests | `tests/test_psmflow_scalar_dueling.py` 12 passed (EXIT 0): both OFF reproduce the Section 10 affine arm bit for bit over 3 updates against `tests/fixtures/psmflow_sec10_affine_3updates.npz` (saved from the pre-edit code); the scalar term equals a hand-rolled reference, scales with the coefficient and sends gradient to psi only; the dueling advantage averages to 0 over its own panel (atol 1e-4), shapes, affineness in `w(z)` kept; both arms train and act, the strict GPI arm too. Regressions, one process per file: affine 12, DSRL-NA 44, agent 18 (+2 skipped), policy index 10, action input 8, GPI select 24, stabilisers 24, actor 3, psm_ref 17 (its info key-set check made additive: new telemetry keys allowed, every old key's value still pinned), Fix 1 file 10, smoke/yaml 2; all EXIT 0 | same code |

### bc_coeff = 0 arms: `cube_sec10_affine_bc0`, `cube_fix2_dueling_bc0` (config only, plus one telemetry key)

Why: the diagnostic on the Section 10 affine checkpoint
(`$PSM_DATA/logs/diag_measure_vs_scalar_q_cube_sec10_affine_sd001.json`) puts the actor's
latent at norm 0.67, at the flow mode, scoring below a random latent under the measure's own
value. The ddpg actor loss is `-Q.mean() / sg(mean |q_ens|) + bc_coeff * distill + CFM`
(`flow_actor_loss`): the value term is normalised by the mean ABSOLUTE ensemble readout,
which is ~880 against a within-state spread of ~63, so the value gradient is ~7% of unit
scale against the distillation term at weight 1.0. There is no existing option to normalise
by the within-batch std of Q instead (none was added). The two anchors in that loss: the
distillation `bc_coeff * ||u_a - rollout(v_xi)||^2` (set to 0 here) and the CFM field loss
`bc_flow_loss`, which trains `actor_vf` only and reaches the actor head solely through the
distillation term, so at `bc_coeff=0` the actor head sees the value gradient alone and
support is enforced by the latent box `u_clip=3.0` (unchanged). DSRL-NA (0.87) runs
`bc_coeff=0`. New telemetry: `actor_u_norm` (mean `||u_actor||`) in `flow_actor_loss`'s
info, matching the key the dsrl_sac loss already logs; test-free, one line.

| arm | template + change | jobs | `--time` | run dirs |
|---|---|---|---|---|
| `cube_sec10_affine_bc0` | `cube_sec10_affine_actor` + `agent.actor.bc_coeff=0.0` | 2518258 (sd0), 2518259 (sd1), 2518260 (sd2), started 04:03 | 05:00:00 (template 3 h 41 min; the change has no cost) | `$PSM_DATA/exp/PSMFLows/cube_sec10_affine_bc0/sd000_s_2518258.0.20260915_040357`, `.../sd001_s_2518259.0.20260915_040357`, `.../sd002_s_2518260.0.20260915_040358` |
| `cube_fix2_dueling_bc0` | `cube_fix2_scalar_dueling` + `agent.actor.bc_coeff=0.0` | 2518261 (sd0), 2518262 (sd1), 2518263 (sd2), started 04:03 | 12:00:00 (the dueling arm's measured 9.5 h) | `$PSM_DATA/exp/PSMFLows/cube_fix2_dueling_bc0/sd000_s_2518261.0.20260915_040357`, `.../sd001_s_2518262.0.20260915_040358`, `.../sd002_s_2518263.0.20260915_040358` |

Flags re-read after launch: against `cube_sec10_affine_actor/sd000` the first group differs
in `agent.actor.bc_coeff 1.0 -> 0.0` only; the second in that plus `psm_scalar_coef 1.0`,
`psi_dueling True`, `psi_dueling_samples 8`. CPU smokes (200 steps, batch 64,
`$PSM_DATA/logs/cpusmoke/{cube_sec10_affine_bc0,cube_fix2_dueling_bc0}_cpusmoke.out`,
both EXIT 0):

| smoke | `actor_u_norm` at 50 / 100 / 150 / 200 | `actor_bc_error` at 200 | bc = 1 smoke of the same arm, `actor_bc_error` at 200 |
|---|---|---|---|
| `cube_sec10_affine_bc0_cpusmoke` | 3.68 / 3.55 / 3.22 / 3.43 | 3.71 | `cube_fix3_cpusmoke` (Section 10 affine + action input): 0.048 |
| `cube_fix2_dueling_bc0_cpusmoke` | 2.32 / 3.21 / 2.97 / 2.62 | 2.50 | `cube_fix2_cpusmoke`: 0.073 |

The template checkpoint's actor norm is 0.67 (diagnostic); the prior's typical norm at
d_a = 5 is ~2.1, the box corner sqrt(5) * 3 = 6.7. Within 200 CPU steps the bc = 0 actors
sit at 2.3-3.7. `tests/test_psmflow_scalar_dueling.py` 12 passed (EXIT 0) after the
telemetry line.

Expectation: `actor_u_norm` rises toward the prior's typical scale (~2) within 50k steps
and stays there; five-task above 0.246 means the pinned actor was the limit of the Section
10 affine arm; below it, the measure's content is.

### Pre-registered expectations (five-task, 500 episodes per task, 500k checkpoint, t interval over 3 seeds)

| arm | expectation |
|---|---|
| Fix 1 `cube_fix1_scalar_tc` | >= 0.5 means the scalar route generalises over `w`; the reference points are the single-task 0.867 (scaled `r_hat`, task 2) and the affine Section 10 0.246 |
| Fix 3 `cube_fix3_action_measure_actor` | >= 0.35 reproduces the 09-11 arm's 0.44 ± 0.38 with lower variance (the actor replaces that arm's per-step argmax over 64 x 64 pairs) |
| Fix 2 `cube_fix2_scalar_dueling` | > 0.246 means the grounding and dueling terms help the measure critic; the ablation `cube_fix2_scalar_only` separates the two terms |
| any arm | < 0.246 (the Section 10 affine arm) is settled negative for that fix |

### Same diagnostic on the Section 10 affine checkpoint (2026-09-15)

`cube_sec10_affine_actor/sd001 @500k` (`policy_index=task_vector train_actor=true
acting=actor actor_mode=ddpg`, psi(s, u, w) indexed by the task vector). Job 2518242
(2 min 31 s), report `$PSM_DATA/logs/diag_measure_vs_scalar_q_cube_sec10_affine_sd001.json`
(+ `.npz`). Same scalar critic, same 10,000 + 791 rows, same panels and seeds. Two things
differ from the GPI run by construction:

- the measure value is `V_m(s,u) = [mean_P - 0.5 unc] psi(s,u,w)^T w` with no index panel,
  and the Bellman bootstrap is the actor's sampled latent at s', `psi_bar(s', pi_eta(s',w,eps), w)`,
  as that run trains it (bootstrap latent norm 1.87 on average);
- this checkpoint has its own phi, so `w` is ITS eval w (`+w_source=infer`, `infer_z` on
  10k rows, shift 1.0): cosine 0.751 to the strict run's w, and its `r_hat = phi^T w`
  correlates 0.816 with the file's r_hat on the pool rows (max abs diff 44.8). The scalar
  critic's reward stays the strict run's scaled r_hat; the two values are of the same true
  reward through two different readouts.

Per split, this checkpoint beside the GPI checkpoint (`gpi` = `affine_strict_cube/sd001`).

**A. Bellman residual along w** (pessimistic target; RMS / std lhs; unit-w vs 20 random
unit directions on the full vector).

| split | mean lhs | std lhs | resid mean | RMS | RMS/std | gpi RMS/std | unit-w rel | random dirs rel mean (min–max) | gpi unit-w / random |
|---|---|---|---|---|---|---|---|---|---|
| all | -1086 | 394 | -18.6 | 44.6 | 0.113 | 0.042 | 0.148 | 0.114 (0.105–0.134) | 0.045 / 0.067 |
| success | -298 | 509 | -22.2 | 78.2 | 0.154 | 0.034 | 0.176 | 0.102 (0.076–0.134) | 0.036 / 0.070 |
| nonsuccess | -1101 | 377 | -18.6 | 43.6 | 0.116 | 0.042 | 0.153 | 0.114 (0.105–0.134) | 0.045 / 0.067 |

Plain-mean target: 0.124 (all). Full-vector RMS norm / lhs norm 0.083 (gpi 0.043). The
actor-bootstrapped measure's residual along w is 11% of its spread against the GPI
measure's 4%, and the w direction is fitted worse than a random direction (0.148 vs 0.114),
where the GPI measure had it better (0.045 vs 0.067).

**B. Global agreement, Vm vs Qa-decode** (P / S).

| split | (s,u_data) | gpi | panel state-centred | gpi | Vm within-state std | Qa within-state std |
|---|---|---|---|---|---|---|
| all | 0.359 / 0.302 | 0.049 / 0.103 | 0.157 / 0.152 | 0.125 / 0.122 | 25.0 | 0.41 |
| success | 0.097 / 0.058 | 0.039 / 0.086 | -0.002 / -0.002 | 0.014 / 0.013 | 34.4 | 0.10 |
| nonsuccess | 0.285 / 0.270 | 0.038 / 0.079 | 0.160 / 0.155 | 0.128 / 0.125 | 24.9 | 0.41 |

Vm vs Qw is the same to two decimals. Across states the two values correlate at 0.36 /
0.30 here against 0.05 / 0.10 for the GPI measure; within a state 0.16 against 0.13.

**C. Per-state ranking over the 64 u, Vm vs Qa** (random: top-8 0.125, regret ~0.48).

| split | rho mean | median | q25 | q75 | frac > 0.3 | frac < 0 | top-8 | regret | regret random | gpi rho / top-8 / regret |
|---|---|---|---|---|---|---|---|---|---|---|
| all | 0.150 | 0.190 | -0.199 | 0.541 | 0.417 | 0.364 | 0.330 | 0.414 | 0.483 | 0.123 / 0.332 / 0.428 |
| success | -0.005 | -0.009 | -0.359 | 0.352 | 0.278 | 0.510 | 0.207 | 0.497 | 0.479 | 0.010 / 0.195 / 0.484 |
| nonsuccess | 0.154 | 0.193 | -0.191 | 0.544 | 0.420 | 0.361 | 0.333 | 0.412 | 0.483 | 0.125 / 0.335 / 0.427 |

In-box (|u|_inf <= 1.5) rho mean 0.143 (gpi 0.117). Per state the ranking agreement is
0.15 against the GPI measure's 0.12; top-8 and regret are the same to two decimals; on
success rows every number is at chance in both.

**E. Coverage** (distance to u_data; panel mean 3.09; argmax at 3.61, near half 28%).

| split | rho near half | rho far half | pooled rho by distance quartile | regret argmax near / far | gpi near / far |
|---|---|---|---|---|---|
| all | 0.146 | 0.152 | 0.148 / 0.151 / 0.153 / 0.154 | 0.410 / 0.415 | 0.117 / 0.124 |
| success | -0.007 | -0.005 | -0.003 / 0.009 / -0.005 / -0.007 | 0.480 / 0.504 | -0.003 / 0.016 |
| nonsuccess | 0.149 | 0.155 | 0.153 / 0.159 / 0.162 / 0.162 | 0.409 / 0.413 | 0.119 / 0.126 |

No dependence on distance from the dataset latent, as for the GPI measure.

**Actor latent.** The Section 10 actor's mode latent `pi_eta(s, w, 0)` against the same
64-panel, regret normalised as in C (a random panel pick scores ~0.48).

| split | u_pi norm | dist to u_data | scalar regret mean / median | beats panel max | in scalar top-8 | measure's own regret mean | in measure top-8 | regret vs Qw | Vm@actor vs Qa@actor P / S |
|---|---|---|---|---|---|---|---|---|---|
| all | 0.674 | 2.10 | 0.467 / 0.459 | 0.0004 | 0.009 | 0.446 | 0.028 | 0.471 | 0.367 / 0.313 |
| success | 0.612 | 2.13 | 0.465 / 0.462 | 0.001 | 0.020 | 0.446 | 0.026 | 0.408 | 0.105 / 0.086 |
| nonsuccess | 0.676 | 2.10 | 0.467 / 0.458 | 0.0004 | 0.009 | 0.446 | 0.029 | 0.473 | 0.296 / 0.282 |

The actor's latent has norm 0.67 (a clipped prior draw at d_a = 5 has norm ~2.1; no
component at the clip) and sits 2.10 from u_data. Under the scalar critic it recovers 3%
of the regret range over a random panel pick (0.467 vs 0.483) and is in the top-8 of 64
on 0.9% of states; under the measure's OWN value it recovers 8% (0.446 vs 0.485) and is
in the top-8 on 2.8% of states. Qa at the actor's decode correlates 0.991 with Qa at the
recorded action's latent.

**F.** Identical to the GPI entry (same scalar critic, same rows): relative RMS 0.041 /
1.09 / 0.046; qa vs qw per-state rho 0.823.

Summary of what was measured. The actor-bootstrapped measure fits its own Bellman
equation along w to 11% of its spread (GPI measure 4%), with the w direction fitted worse
than a random one. Its value agrees with the scalar critic's across states at Spearman
0.30 (GPI 0.10) and within a state at 0.15 (GPI 0.12); argmax top-8 and regret are the
same as the GPI measure's (0.33, 0.41). The actor's latent is not a maximiser of either
value over the 64-panel: 3% regret recovery under the scalar critic, 8% under the
measure itself. Not measured: the return of either latent.

## Policy-side diagnostic (2026-09-15)

Blocks G and H of `tools/diag_measure_vs_scalar_q.py` (commit 92aba31), one job (2518330,
9 min 3 s) over three cube checkpoints at 500k against the same scalar critic, pool, panels
and seeds as above: `gpi` = `affine_strict_cube/sd001` (latent index, no actor, file w);
`sec10` = `cube_sec10_affine_actor/sd001` (task-vector index, ddpg actor, bc_coeff 1, own w);
`sec10bc0` = `cube_sec10_affine_bc0/sd001` (task-vector index, ddpg actor, bc_coeff 0, own w;
w cosine 0.865 to the strict w, r_hat correlation 0.911 to the file). Reports
`$PSM_DATA/logs/diag_policy_side_cube_{gpi,sec10,sec10bc0}_sd001.json` (+ `.npz`); A–F
are recomputed in the same files and reproduce the entries above (bc0: A 0.058, B data-u
Spearman 0.449, C rho 0.327 / top-8 0.504 / regret 0.326).

Index panel: 64 prior draws u' (gpi) or 64 task vectors z per state, 32 project(N(0,I)) +
32 project(phi(s'_j)) at random pool rows (sec10, bc0), read through the fixed eval w and,
for the task-indexed runs, through z itself (`psi^T z`, the run's own read).

**G. Policy-slot information.** Reading key: index/action ratio << 1 means the policy
slot carries almost nothing; >> 1 means the readout moves far more with the policy slot
than with the action latent.

| checkpoint | std across index at u_data | std across u at one index | ratio index/u (pooled) | per-state median | unc along index | unc along u | std/unc index | std/unc u |
|---|---|---|---|---|---|---|---|---|
| gpi | 2635 | 72.8 | 36.2 | 51.5 | 52.4 | 51.9 | 50.3 | 1.40 |
| sec10 (psi^T w) | 89.4 | 21.2 | 4.21 | 7.42 | 17.0 | 17.0 | 5.26 | 1.25 |
| sec10 (psi^T z) | 667 | 23.8 | 28.1 | | | | | |
| sec10bc0 (psi^T w) | 2613 | 13.5 | 193 | 310 | 19.7 | 21.3 | 133 | 0.64 |
| sec10bc0 (psi^T z) | 1694 | 11.2 | 151 | | | | | |

Success rows: ratios 35.6 / 3.91 / 174 (gpi / sec10 / bc0).

64 x 64 grid, 500 uniform states, two-way split of the within-state variance of the
readout, and the mean pairwise Spearman across the 64 u between index columns (near 1:
every policy ranks the action latents alike).

| checkpoint | readout | share u | share index | share interaction | std u | std index | std total | rank corr across u between indices (mean / p10) | rank corr across indices between u |
|---|---|---|---|---|---|---|---|---|---|
| gpi | psi^T w | 0.002 | 0.993 | 0.005 | 47.7 | 2754 | 2760 | 0.287 / 0.098 | 0.991 |
| sec10 | psi^T w | 0.079 | 0.917 | 0.004 | 21.9 | 90.5 | 96.2 | 0.916 / 0.808 | 0.993 |
| sec10 | psi^T z | 0.000 | 0.994 | 0.005 | 7.5 | 653 | 654 | 0.042 / -0.002 | |
| sec10bc0 | psi^T w | 0.000 | 1.000 | 0.000 | 6.6 | 2602 | 2602 | 0.347 / 0.124 | 1.000 |
| sec10bc0 | psi^T z | 0.000 | 1.000 | 0.000 | 5.7 | 1692 | 1692 | 0.161 / 0.043 | |

Ensemble disagreement on the grid (mean): 53.1 / 17.5 / 20.1; its own split is 0.64 /
0.27 / 0.83 on the index axis. Readings: on all three checkpoints the index slot, not the
action latent, carries the variance of the readout (index share 0.92–1.00, action share
0.00–0.08); the action-axis spread is 0.6–1.4 times the ensemble disagreement on every
checkpoint, the index-axis spread 5–133 times it. Whether the ranking of action latents
depends on the policy asked about differs: on the GPI measure the 64 u' columns agree at
Spearman 0.29 (p10 0.10) — different u' rank the actions differently; on the sec10 measure
read through w at 0.92 — all task vectors rank the actions alike, i.e. the action ordering
is one fixed function of s; on bc0 at 0.35. Read through z itself the agreement is 0.04
(sec10) and 0.16 (bc0). The interaction share is at most 0.005 everywhere: psi^T w is
additively separable in (u, index) on the grid.

**H. Box exploitation.** Panel latents in global quartiles of |u|_2 (edges 1.64 / 2.09 /
2.57; max 5.42); Vm_z, Qa_z are state-z-scored. The panel, the scalar Q and the decodes are
identical across checkpoints; only Vm_z changes.

| |u|_2 bin | Vm_z gpi | Vm_z sec10 | Vm_z bc0 | Qa_z | Qw_z | \|G(s,u)\|_inf | clip any (>0.999) | clip per component |
|---|---|---|---|---|---|---|---|---|
| [0.11, 1.64) | +0.076 | +0.059 | +0.065 | +0.029 | +0.028 | 0.658 | 0.113 | 0.025 |
| [1.64, 2.09) | +0.028 | +0.024 | +0.029 | +0.013 | +0.012 | 0.661 | 0.115 | 0.025 |
| [2.09, 2.57) | -0.015 | -0.008 | -0.008 | -0.001 | -0.001 | 0.659 | 0.114 | 0.025 |
| [2.57, 5.42] | -0.089 | -0.075 | -0.086 | -0.041 | -0.039 | 0.662 | 0.115 | 0.025 |

|u|_inf quartiles (edges 1.17 / 1.52 / 1.91) give the same picture (gpi top bin Vm_z
-0.078, Qa_z -0.037). On success rows the gradient is steeper for the measures (gpi bin
0 +0.153, bin 3 -0.192) and for Qw (+0.116 / -0.182) but not Qa (+0.024 / -0.028).

| argmax over the 64-panel | in top |u|_2 quartile | in top |u|_inf quartile | mean |u|_2 of the argmax (panel 2.13) |
|---|---|---|---|---|
| measure gpi | 0.679 | 0.654 | 2.86 |
| measure sec10 | 0.709 | 0.676 | 2.90 |
| measure bc0 | 0.736 | 0.697 | 2.94 |
| scalar Qa (same for all) | 0.766 | 0.755 | 2.99 |
| random | 0.25 | 0.25 | 2.13 |

Readings: the mean readout falls with |u| in every quartile for all five values, by
0.16 panel-std across the range for the measures and 0.07 for the scalar critic; the
decoded action's size and clip fraction do not change with |u| (0.658–0.662; 0.113–0.115).
Yet the ARGMAX of every value sits in the top norm quartile on 65–77% of states — the
scalar critic's argmax more often than any measure's — with an argmax |u|_2 of 2.9–3.0
against the panel's 2.13. The mean-vs-argmax difference is the max of 64 draws: the top
quartile holds the extreme values of a value whose per-state spread is much larger than
its norm trend.

Actor latents (mode pi_eta(s, w, 0)) against the same panel and bins:

| | sec10 (bc 1) | sec10bc0 (bc 0) |
|---|---|---|
| |u_pi|_2 mean / p10 / p50 / p90 / max | 0.67 / 0.25 / 0.53 / 1.30 / 3.71 | 5.88 / 4.98 / 5.99 / 6.65 / 6.71 |
| |u_pi|_inf mean / p90 / max | 0.52 / 1.02 / 2.89 | 3.00 / 3.00 / 3.00 |
| components at u_clip = 3 | 0.000 | 0.143 |
| in top |u|_2 quartile | 0.004 | 1.000 |
| |G(s,u_pi)|_inf (panel 0.660) | 0.655 | 0.666 |
| decode clip any (panel 0.114) / per component | 0.111 / 0.024 | 0.085 / 0.019 |
| Vm at actor, panel std units | +0.17 | +3.45 |
| Qa at actor, panel std units | +0.07 | +2.54 |
| Qw at actor, panel std units | +0.07 | +2.39 |
| scalar regret vs panel (random 0.48) / beats panel max / in top-8 | 0.467 / 0.000 / 0.009 | -0.037 / 0.575 / 0.690 |
| measure's own regret / beats panel max / in top-8 | 0.446 / 0.001 / 0.028 | -0.257 / 0.820 / 0.918 |
| success rows: Qa_z / Qw_z at actor, scalar regret | +0.07 / +0.20, 0.465 | -0.74 / -2.23, 0.622 |

Readings: the bc0 actor's latent is at the box edge on every state (|u|_inf = 3.0 at the
p90 and the max; |u|_2 5.9 of the 6.7 the box allows; 14% of components exactly at the
clip), 6.2 from u_data; the bc-1 actor's is at 0.67 with no component at the clip. The
decoded action of the bc0 latent is not clipped more than a panel decode (0.085 vs
0.114). The measure scores the bc0 latent 3.45 panel-std above the panel mean and above
the panel max on 82% of states; the scalar critic scores the same latent 2.54 panel-std
above the panel mean and above the panel max on 57% of non-success states. On success
rows the scalar critic scores it 0.74 (qa) and 2.23 (qw) panel-std BELOW the panel mean,
regret 0.62 against 0.48 random. The measure of the bc0 checkpoint rewards the box edge,
and the scalar critic, at the same latents, agrees outside success states and disagrees
inside them.

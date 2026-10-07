# Campaign scripts, 2026-10-01..07

Copies of `$PSM_DATA/gc_scripts/` as used on the KISSKI SLURM cluster. Logic unchanged. Every
script hardcodes `PSM_DATA=/mnt/home/amohan/psm-data` and `REPO=/mnt/home/amohan/git/Austin/PSMFLows`
(the `PSM_REPO` checkout) at its top; edit those two lines for another machine. All of them call
`sbatch` with `--partition=kisski-inference --account=general`. Results and expectations are
summarised in `docs/HANDOFF.md` (2026-10-05 and 2026-10-07 entries) and `docs/results/2026-10-07-psmgoal-softmax.md`; the pre-launch expectation files and job
lists are in `docs/campaign_2026-10/`.

| script | what it does | groups |
|---|---|---|
| `gc_eval_watcher.sh` | polls for checkpoints, submits five-task eval500 jobs (readouts hgoal, lp, actor_rel, actor_env) at 250k and 500k | `psmgoal_gc_{sq,sm}_cube` |
| `gc_eval_watcher_antmaze.sh` | same for antmaze (ENVKEY=antmaze, longer time limit) | `psmgoal_gc_{sq,sm}_antmaze` |
| `sm_code_eval_watcher.sh` | eval500 readouts trained and lp | `psmgoal_sm_code_cube` |
| `db_eval_watcher.sh` | eval500 readouts hgoal, hgoal_each (250k, 500k) and lp (250k) | `psmgoal_db_{sq,sm}_cube` |
| `db_eval_watcher_antmaze.sh` | same for antmaze | `psmgoal_db_{sq,sm}_antmaze` |
| `dbu_eval_watcher_cube.sh` | same readouts for the uniform-goal arm | `psmgoal_dbu_{sq,sm}_cube` |
| `dbo_eval_watcher_cube.sh` | same readouts for the orthonormality arm | `psmgoal_dbo_sq_cube` |
| `ja_arms.sh` | the joint-actor arms as `GROUP -> hydra overrides` (sourced by the launcher) | `psmgoal_ja_{fbc3_sh,fbc0_sh,dsrl_sh,fbc3_pt}_cube` |
| `ja_arms_extra.txt` | the fifth arm added 2026-10-05 (`fb_bc_coeff=0.3`) | `psmgoal_ja_fbc03_sh_cube` |
| `ja_launch.sh` | submits the joint-actor arms, 3 seeds, packed (3 seeds per GPU) or one job per seed | as above |
| `ja_smoke.sh` | 200-step smoke of the joint-actor launch path and its eval readout | as above |
| `ja_eval_watcher_cube.sh` | eval500 readouts hgoal_each and actor_rel, one packed job (five tasks) per (run, step, readout) | as above plus `psmgoal_ja_fbc03_sh_cube` |
| `agg_results.py` | prints the five-task mean per (group, step, readout) from `$PSM_DATA/logs/*.json` | all groups above |

Training jobs go through `scripts/slurm/train_psmflow.sbatch` (one seed) or
`scripts/slurm/train_psmflow_packed.sbatch` (three seeds, one GPU); eval jobs through
`scripts/slurm/eval500.sbatch` with `MODE=psmgoal` and an `EXTRA` override string that names the
readout. `scripts/slurm/launch_psmgoal_eval500.sh` is the general form of the watchers (one call
submits every missing eval of one group).

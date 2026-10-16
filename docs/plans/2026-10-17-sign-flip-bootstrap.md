# 2026-10-17 — sign-flip bootstrap: does the direction of the actor's step carry the gain?

## Question

Critic A (`psmgoal_ja_fbc3_pt_cube`, the flowbc actor's latent in the TD target) scores 0.805
five-task; critic B (same, BC weight 1000, step 0.0003) scores 0.657. The step is 0.09 long on a
draw of length 2.1, 82% a fixed vector per (s, g), cosine 0.30 with the critic's gradient in u.
Two readings fit every number so far (the step-scale sweep m = 0.5 / 2 / 4 at 100k on aws, the BC
weight sweep, the switch arms, the best-of-N and mean-of-8 bootstraps):

- gradient step: the gain comes from the DIRECTION of the step (a small policy-improvement step
  on the critic's own value, TD3+BC / FQL style);
- perturbation ("exploration in noise space"): the gain comes from the bootstrap latent being
  moved off the exact prior draw at all; direction is irrelevant.

The two differ on one thing: whether reversing the step keeps the gain.

## Arm

`psmgoal_ja_fbc3_pt_bdsneg_cube` = critic A's exact configuration plus `agent.bootstrap_delta_scale=-1.0`.
The TD target bootstraps from clip(eps - delta(s', g, eps)) instead of clip(eps + delta). The
actor itself trains unchanged (its loss and acting use +delta); only the latent the target is
evaluated at is mirrored about the raw draw. `tests/test_psmgoal_bootstrap_delta_scale.py::
test_scale_minus_one_mirrors_the_step` checks the mirror on unclipped rows.

Three seeds, one per GPU, 500k steps, checkpoints 250k / 500k, like critics A and B (A and B
were tied until 150k, so the 100k horizon of the aws scale sweep is too short for this question).

## Expected outcome, stated before results

| reading | best-of-64 five-task at 250k / 500k |
|---|---|
| gradient step (direction carries the gain) | at or below critic B, 0.63 / 0.66; a step that lowers the critic's value at every backup may also sit below B |
| perturbation (direction irrelevant) | at critic A's level, 0.79 / 0.80, within the seed interval |
| in between (0.70-0.75) | both contribute; the random-direction control (same length, random direction per draw) is then the next arm |

The actor-alone readout is expected at BC level (0.10-0.11) in every case; it has been there for
every actor arm.

## Hyperparameters: `psmgoal_ja_fbc3_pt_bdsneg_cube`

The fbc3_pt overrides (from `scripts/campaign_2026-10/ja_arms.sh`, confirmed in critic A's
`flags.json`) plus `agent.bootstrap_delta_scale=-1.0`. "yaml" marks values that equal the
`configs/agent/psmgoal.yaml` default and are passed for explicitness.

| key | value | note |
|---|---|---|
| `agent.policy_index` | goal | w = h(g) |
| `agent.bootstrap_source` | actor | yaml |
| `agent.train_actor` | true | |
| `agent.actor_input` | goal | |
| `agent.goal_random_frac` | 0.0 | yaml default 0.3 |
| `agent.goal_cur_frac` | 0.0 | yaml |
| `agent.goal_sampling` | geometric | yaml |
| `agent.measure_loss` | softmax | yaml default squared |
| `agent.acting` | distill | in-loop 50-episode eval only; yaml default gpi |
| `agent.eval_redistill` | false | yaml default true |
| `agent.eval_goal_source` | relabel | yaml |
| `agent.actor_kind` | flowbc | yaml default tanh |
| `agent.fb_bc_coeff` | 3.0 | yaml |
| `agent.actor_value_kind` | point | yaml default shaped |
| `agent.bootstrap_delta_scale` | **-1.0** | THE ARM; yaml default 1.0 |
| `agent.flow_ckpt_path` | `$PSM_DATA/flow/cube-single-play` | |
| `agent.flow_ckpt_epoch` | 500000 | |
| `agent.preimage_path` | `$PSM_DATA/preimages/cube-single-play.npz` | |
| `env_name` | cube-single-play-singletask-v0 | ENVKEY=cube |
| `offline_steps` | 500000 | |
| `log_interval` | 5000 | |
| `eval_interval` | 50000 | |
| `eval_episodes` | 50 | |
| `save_interval` | 250000 | |
| `seed` | 0, 1, 2 | |
| `run_group` | psmgoal_ja_fbc3_pt_bdsneg_cube | |

Values left at the yaml default and not passed: `discount=0.98`, `goal_discount=0.98`,
`measure_temp=1.0`, `z_dim=128`, `batch_size=256`, `use_point_preimage=true`, `k_goals=32`,
`gpi_num_u=64`, `q_coeff=1.0`, `norm_future_k=0`, `measure.hidden_dim=1024`, `tau=0.01`,
`lr_measure=1e-4`.

## Launch (kisski, one seed per GPU)

```bash
REPO=/mnt/home/amohan/git/Austin/PSMFLows; PSM_DATA=/mnt/home/amohan/psm-data
GROUP=psmgoal_ja_fbc3_pt_bdsneg_cube
OVERRIDES="agent.policy_index=goal agent.bootstrap_source=actor agent.train_actor=true agent.actor_input=goal agent.goal_random_frac=0.0 agent.goal_cur_frac=0.0 agent.goal_sampling=geometric agent.measure_loss=softmax agent.acting=distill agent.eval_redistill=false agent.eval_goal_source=relabel agent.actor_kind=flowbc agent.fb_bc_coeff=3.0 agent.actor_value_kind=point agent.bootstrap_delta_scale=-1.0"
COMMON="PSM_REPO=$REPO,PSM_DATA=$PSM_DATA,OGBENCH_DATASET_DIR=/mnt/home/amohan/.ogbench/data,ENVKEY=cube,PREIMAGES=$PSM_DATA/preimages/cube-single-play.npz,STEPS=500000,EVAL_INT=50000,EVAL_EPS=50,LOG_INT=5000,SAVE_INT=250000"
for s in 0 1 2; do
  sbatch --parsable --partition=kisski-inference --account=general --time=24:00:00 --cpus-per-task=4 \
    --job-name=${GROUP}_sd$s --output=$PSM_DATA/logs/slurm/%x-%j.out --error=$PSM_DATA/logs/slurm/%x-%j.err \
    --export="ALL,$COMMON,SEED=$s,GROUP=$GROUP,EXTRA=$OVERRIDES" $REPO/scripts/slurm/train_psmflow.sbatch
done
```

Smoke: the same launcher with `STEPS=200 EVAL_INT=100 EVAL_EPS=2 LOG_INT=50 SAVE_INT=200`,
group `smoke_bdsneg_cube` (job 2525504). After launch:
`grep -o '"bootstrap_delta_scale": [-0-9.]*' $PSM_DATA/exp/PSMFLows/$GROUP/sd00*/flags.json`
must print `-1.0` for every seed.

Eval: `scripts/eval500.sh psmgoal cube <run_dir> <out>` with `hgoal_each` + gpi at 250k and
500k, five tasks, 500 episodes; aggregate with `scripts/campaign_2026-10/agg_results.py`.

## Companion diagnostic (same day)

`tools/diag_actor_shift_direction.py` (job 2525503, critic A sd000 500k, 512 states): where the
step points — against the critic's top-set centroid, the data latent, the gradient and the draw
itself; its goal dependence; its effect on the decoded action. Report
`$PSM_DATA/logs/actor_shift/shift_fbc3pt_sd000_500k.json`.

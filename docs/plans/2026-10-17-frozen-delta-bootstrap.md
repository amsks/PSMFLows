# 2026-10-17 — frozen-step bootstrap: +delta, -delta, random direction, zero, with the step field fixed

## Why this and not the sign-flip arm alone

The sign-flip arm `psmgoal_ja_fbc3_pt_bdsneg_cube` (`bootstrap_delta_scale=-1.0`, running) keeps
training its actor against a critic that is itself trained on reversed targets. By 75k that actor's
step had shrunk to 0.019 against critic A's 0.088, so direction and length change together and the
arm cannot separate them. Here the step field is FROZEN: delta is loaded from critic A's 500k
checkpoint and never updated. The target latent is then clip(eps + step) with step one of

| mode | step | what it is |
|---|---|---|
| plus | +delta_A(s', g, eps) | critic A's target with a fixed field; also tests whether co-adaptation matters |
| minus | -delta_A(s', g, eps) | the same length, the opposite direction |
| random | a random direction per row, length ||delta_A(s', g, eps)|| | the same length, no direction information |
| zero | 0 | one BC draw, critic B |

Everything else is critic A's configuration. The live actor still trains (policy_index=goal requires
it) and is not read by the target. Lengths are matched before the clip at u_clip=3, which touches
about 1 percent of rows. Code: `bootstrap_source=frozen_delta`, `frozen_delta_path`,
`frozen_delta_epoch`, `frozen_delta_mode` (`agents/psmgoal.py`, `configs/agent/psmgoal.yaml`);
tests `tests/test_psmgoal_frozen_delta_bootstrap.py` (plus equals the live-actor path when the
params coincide; zero equals scale 0; minus mirrors; random has the step's length and another
direction; frozen params do not move under an update; bad mode refused).

## Expected outcome, stated before results (best-of-64 five-task, 250k / 500k)

| mode | gradient-step reading (direction carries the gain) | perturbation reading (only the step's presence matters) |
|---|---|---|
| plus | at critic A, 0.79 / 0.80 (or below it if co-adaptation matters) | at A |
| minus | at or below critic B, 0.63 / 0.66 | at A |
| random | at or below B | at A |
| zero | at B, 0.63 / 0.66 | at B |

The informative contrasts are plus against random (direction) and random against zero (presence).
If plus sits well below A, the co-trained actor matters beyond its final field, and the frozen
design has to be rerun with a field snapshot taken earlier in training. The actor-alone readout
is expected at BC level (0.10 to 0.11) in every arm.

## Hyperparameters: `psmgoal_ja_fbc3_pt_fd{plus,minus,rand,zero}_cube`

The fbc3_pt overrides (`scripts/campaign_2026-10/ja_arms.sh`, confirmed in critic A's
`flags.json`) with the bootstrap source replaced. "yaml" marks values equal to the
`configs/agent/psmgoal.yaml` default, passed for explicitness.

| key | value | note |
|---|---|---|
| `agent.policy_index` | goal | w = h(g) |
| `agent.bootstrap_source` | **frozen_delta** | THE ARM; yaml default actor |
| `agent.frozen_delta_path` | `$PSM_DATA/exp/PSMFLows/psmgoal_ja_fbc3_pt_cube/sd000_s_2522773.0.20261006_111542` | critic A seed 0 |
| `agent.frozen_delta_epoch` | 500000 | |
| `agent.frozen_delta_mode` | plus / minus / random / zero | one per group |
| `agent.train_actor` | true | the live actor trains, unused by the target |
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
| `agent.flow_ckpt_path` / `flow_ckpt_epoch` | `$PSM_DATA/flow/cube-single-play` / 500000 | |
| `agent.preimage_path` | `$PSM_DATA/preimages/cube-single-play.npz` | |
| `env_name` | cube-single-play-singletask-v0 | ENVKEY=cube |
| `offline_steps` / `log_interval` / `eval_interval` / `eval_episodes` / `save_interval` | 500000 / 5000 / 50000 / 50 / 250000 | |
| `seed` | 0, 1, 2 | the same frozen field (A seed 0) in every seed |
| `run_group` | psmgoal_ja_fbc3_pt_fd{plus,minus,rand,zero}_cube | |

Defaults left untouched: discount 0.98, goal_discount 0.98, measure_temp 1.0, z_dim 128,
batch_size 256, k_goals 32, gpi_num_u 64, q_coeff 1.0, norm_future_k 0, bootstrap_delta_scale 1.0,
measure.hidden_dim 1024, tau 0.01, lr_measure 1e-4, u_clip 3.0.

## Launch (kisski, one seed per GPU)

```bash
REPO=/mnt/home/amohan/git/Austin/PSMFLows; PSM_DATA=/mnt/home/amohan/psm-data
A=$PSM_DATA/exp/PSMFLows/psmgoal_ja_fbc3_pt_cube/sd000_s_2522773.0.20261006_111542
BASE="agent.policy_index=goal agent.train_actor=true agent.actor_input=goal agent.goal_random_frac=0.0 agent.goal_cur_frac=0.0 agent.goal_sampling=geometric agent.measure_loss=softmax agent.acting=distill agent.eval_redistill=false agent.eval_goal_source=relabel agent.actor_kind=flowbc agent.fb_bc_coeff=3.0 agent.actor_value_kind=point agent.bootstrap_source=frozen_delta agent.frozen_delta_path=$A agent.frozen_delta_epoch=500000"
COMMON="PSM_REPO=$REPO,PSM_DATA=$PSM_DATA,OGBENCH_DATASET_DIR=/mnt/home/amohan/.ogbench/data,ENVKEY=cube,PREIMAGES=$PSM_DATA/preimages/cube-single-play.npz,STEPS=500000,EVAL_INT=50000,EVAL_EPS=50,LOG_INT=5000,SAVE_INT=250000"
for m in plus minus random zero; do tag=$m; [ $m = random ] && tag=rand
  for s in 0 1 2; do
    sbatch --parsable --partition=kisski-inference --account=general --time=24:00:00 --cpus-per-task=4 \
      --job-name=psmgoal_ja_fbc3_pt_fd${tag}_cube_sd$s --output=$PSM_DATA/logs/slurm/%x-%j.out --error=$PSM_DATA/logs/slurm/%x-%j.err \
      --export="ALL,$COMMON,SEED=$s,GROUP=psmgoal_ja_fbc3_pt_fd${tag}_cube,EXTRA=$BASE agent.frozen_delta_mode=$m" $REPO/scripts/slurm/train_psmflow.sbatch
  done
done
```

Smokes: the same launcher with STEPS=200 EVAL_INT=100 EVAL_EPS=2 LOG_INT=50 SAVE_INT=200 for modes
random and minus (groups `smoke_fdrandom_cube`, `smoke_fdminus_cube`). After launch:
`grep -o '"frozen_delta_mode": "[a-z]*"' $PSM_DATA/exp/PSMFLows/psmgoal_ja_fbc3_pt_fd*_cube/sd00*/flags.json`
must print the four modes, three seeds each, and `"bootstrap_source": "frozen_delta"` everywhere.

Eval: `ARMS="psmgoal_ja_fbc3_pt_fdplus_cube psmgoal_ja_fbc3_pt_fdminus_cube psmgoal_ja_fbc3_pt_fdrand_cube psmgoal_ja_fbc3_pt_fdzero_cube" bash scripts/campaign_2026-10/arm_eval_watcher_cube.sh`.

## Cost and placement

12 runs at about 3 h each on an H100 (critic A's pace, 0.02 s per step; the frozen field adds one
delta forward per update). The kisski partition holds 36 H100s with about 90 jobs pending and our
fair share at 0.11, so the twelve go in as a batch and start as GPUs free. The companion work
handed to the other machines the same day: luis relaunches nf2 seeds 1 and 2 to 500k (H200, 24 h
limit) and builds the deployment-actor distillation; aws reruns the fullness diagnostic with
`+rollout_policy=actor` and `+rollout_policy=bc` on critic A and nf2 seed 0 at 500k.

# 2026-10-15: `norm_future_k` -- own-trajectory future states in the softmax normalisation set

## The change

The softmax measure loss (`measure_loss=softmax`, `agents/psmgoal.py`) reads the measure at
(s_i, u_i; h(g_i)) as a distribution over a set of candidate next states. Until now that set was
the batch's N next states {s'_j}. The 2026-10-15 fullness diagnostic showed the trained measure
is flat over every column except the row's own s'_i, and the TD target has no column where the
row's trajectory actually goes later, so it can only put mass on s'_i.

`norm_future_k = K` adds each row's own later states to that set. The dataset sampler draws K
rows of the row's own trajectory at offsets d ~ Geometric(1 - goal_discount), d >= 1, clipped at
the trajectory end (the hindsight-goal draw repeated K times; `utils/datasets.py
future_state_idxs`), and serves `batch['future_states']` (N, K, ob). The loss columns become
S = [the N next states, then the N*K future states flattened], shared by every row; column i is
still the row's own s'_i. p_i = softmax over S of M(s_i, u_i, col; h(g_i)) / temp. The target is
(1 - gamma) one-hot(column i) + gamma softmax over S of M_bar(s'_i, u'_i, col; h_bar(g_i)) / temp.
The loss is the cross-entropy, as before. K = 0 is bit-identical to the previous code (verified
on a fixed batch: params and every logged value equal after three updates, both the fbc3_pt arm
and the squared default).

Left on the batch's N next states only (training): the actor's point value pool
(`actor_value`, `actor_p_own`), the `fb_goals` kernel pool, and the `bootstrap_source=argmax`
scoring pool. The eval readouts are untouched. `measure_loss=squared` refuses K > 0.

New logging keys in the softmax loss (every update, `train.csv`): `measure_p_own_next` (mean
softmax weight on the row's own s'_i; equals the existing `m_softmax_diag`), `measure_p_future`
(mean total weight on the row's own K future columns; 0 at K = 0), and `phi_eff_rank` (the
existing participation ratio of phi's Gram over the whole mesh, `(tr G)^2 / ||G||_F^2`).

Code: `agents/psmgoal.py` (`softmax_td_loss`, `measure_loss`, `_softmax_measure_loss`, `create`
assert, `get_config`), `utils/datasets.py` (`trajectory_ends`, `future_state_idxs`,
`Dataset.norm_future_k`, `get_subset`), `main.py` (sets `dataset.norm_future_k` for psmgoal),
`configs/agent/psmgoal.yaml`, `tests/test_psmgoal_norm_future.py`.

## Expected outcome

Own-next weight (`measure_p_own_next`) falls below 0.27. `measure_p_future` rises well above
K/(N + NK) = 2/768 = 0.0026. `phi_eff_rank` rises above 1 without ortho. Five-task success may
fall from 0.805.

Memory: the training mesh is 256 x 768 rows through the 1024-wide measure MLP (196,608 rows,
about 0.8 GB of activations per layer) plus the same-size target mesh without gradients; an
H100 holds it without chunking (the existing code does not chunk the training mesh either).

## Hyperparameters: `psmgoal_ja_fbc3_pt_nf2_cube`

The fbc3_pt overrides (from `scripts/campaign_2026-10/ja_arms.sh`, confirmed in
`$PSM_DATA/exp/PSMFLows/psmgoal_ja_fbc3_pt_cube/sd000_s_2522773.0.20261006_111542/flags.json`)
plus `agent.norm_future_k=2`. Every override the launch passes, with its value; "yaml" marks
the ones that equal the `configs/agent/psmgoal.yaml` default and are passed for explicitness.

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
| `agent.norm_future_k` | **2** | NEW; yaml default 0 |
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
| `run_group` | psmgoal_ja_fbc3_pt_nf2_cube | |

Values left at the yaml default and not passed: `discount=0.98`, `goal_discount=0.98`,
`measure_temp=1.0`, `z_dim=128`, `batch_size=256`, `use_point_preimage=true`, `k_goals=32`,
`gpi_num_u=64`, `q_coeff=1.0`, `measure.hidden_dim=1024`, `tau=0.01`, `lr_measure=1e-4`.

## Launch (kisski, not run)

`psmgoal_ja_fbc3_pt_cube` was launched by `scripts/campaign_2026-10/ja_launch.sh` with its
default `PACK=3`: one packed job per arm, three seeds on one GPU
(`scripts/slurm/train_psmflow_packed.sbatch`; the run dir `sd000_s_2522773.0.*` is a packed
job step). The same launch for the new group:

```bash
REPO=/mnt/home/amohan/git/Austin/PSMFLows; PSM_DATA=/mnt/home/amohan/psm-data
GROUP=psmgoal_ja_fbc3_pt_nf2_cube
OVERRIDES="agent.policy_index=goal agent.bootstrap_source=actor agent.train_actor=true agent.actor_input=goal agent.goal_random_frac=0.0 agent.goal_cur_frac=0.0 agent.goal_sampling=geometric agent.measure_loss=softmax agent.acting=distill agent.eval_redistill=false agent.eval_goal_source=relabel agent.actor_kind=flowbc agent.fb_bc_coeff=3.0 agent.actor_value_kind=point agent.norm_future_k=2"
COMMON="PSM_REPO=$REPO,PSM_DATA=$PSM_DATA,OGBENCH_DATASET_DIR=/mnt/home/amohan/.ogbench/data,ENVKEY=cube,PREIMAGES=$PSM_DATA/preimages/cube-single-play.npz,STEPS=500000,EVAL_INT=50000,EVAL_EPS=50,LOG_INT=5000,SAVE_INT=250000"
sbatch --parsable --partition=kisski-inference --account=general --time=24:00:00 \
  --job-name=${GROUP}_p3 --output=$PSM_DATA/logs/slurm/%x-%j.out --error=$PSM_DATA/logs/slurm/%x-%j.err \
  --export="ALL,$COMMON,SEEDS=0 1 2,GROUP=$GROUP,EXTRA=$OVERRIDES" $REPO/scripts/slurm/train_psmflow_packed.sbatch
```

One seed per GPU instead (the `train_psmflow.sbatch` rule; `ja_launch.sh` with `PACK=1`):

```bash
for s in 0 1 2; do
  sbatch --parsable --partition=kisski-inference --account=general --time=24:00:00 --cpus-per-task=4 \
    --job-name=${GROUP}_sd$s --output=$PSM_DATA/logs/slurm/%x-%j.out --error=$PSM_DATA/logs/slurm/%x-%j.err \
    --export="ALL,$COMMON,SEED=$s,GROUP=$GROUP,EXTRA=$OVERRIDES" $REPO/scripts/slurm/train_psmflow.sbatch
done
```

After launch: `grep -o '"norm_future_k": [0-9]*' $PSM_DATA/exp/PSMFLows/$GROUP/sd00*/flags.json`
must print `2` for every seed, and `train.csv` must carry `measure_p_own_next`,
`measure_p_future`, `phi_eff_rank`.

## Launch (aws, tmux, no SLURM)

`docs/plans/2026-10-15-aws-setup.md` section 6 conventions: `PSM_DATA` exported, flow at
`$PSM_DATA/flow/cube-single-play`, preimages at `$PSM_DATA/preimages/cube-single-play.npz`, one
tmux session and one GPU per seed.

```bash
cd $PSM_REPO; GROUP=psmgoal_ja_fbc3_pt_nf2_cube
OVERRIDES="agent.policy_index=goal agent.bootstrap_source=actor agent.train_actor=true agent.actor_input=goal agent.goal_random_frac=0.0 agent.goal_cur_frac=0.0 agent.goal_sampling=geometric agent.measure_loss=softmax agent.acting=distill agent.eval_redistill=false agent.eval_goal_source=relabel agent.actor_kind=flowbc agent.fb_bc_coeff=3.0 agent.actor_value_kind=point agent.norm_future_k=2"
for s in 0 1 2; do
  tmux new -d -s train_nf2_sd$s "cd $PSM_REPO && CUDA_VISIBLE_DEVICES=$s MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=offline \
    .venv/bin/python main.py agent=psmgoal env_name=cube-single-play-singletask-v0 \
    agent.flow_ckpt_path=$PSM_DATA/flow/cube-single-play agent.flow_ckpt_epoch=500000 \
    agent.preimage_path=$PSM_DATA/preimages/cube-single-play.npz $OVERRIDES \
    offline_steps=500000 online_steps=0 log_interval=5000 eval_interval=50000 eval_episodes=50 \
    save_interval=250000 save_dir=$PSM_DATA/exp run_group=$GROUP seed=$s \
    > $PSM_DATA/logs/${GROUP}_sd$s.log 2>&1"
done
```

With one GPU, drop `CUDA_VISIBLE_DEVICES=$s` and run the seeds one after another.

## Smoke (2026-10-15)

CPU (`JAX_PLATFORMS=cpu`, 128 cores), the launch overrides above with `norm_future_k=2`, 200
steps through `main.py`, `eval_interval=100 eval_episodes=1`, once at `batch_size=64` and once
at the real `batch_size=256` (mesh 256 x 768). Both exit 0, `flags.json` records
`norm_future_k: 2`, every logged value is finite, the in-loop eval runs.

| batch | step | psm_loss | measure_p_own_next | measure_p_future | phi_eff_rank | s/step |
|---|---|---|---|---|---|---|
| 64 | 100 | 3.319 | 0.012 | 0.014 | 1.22 | 0.11 |
| 64 | 200 | 3.152 | 0.022 | 0.017 | 1.17 | 0.14 |
| 256 | 100 | 4.188 | 0.0044 | 0.0043 | 1.27 | 0.77 |
| 256 | 200 | 4.449 | 0.0040 | 0.0064 | 1.20 | 1.10 |

At 200 steps the measure is near uniform (1/768 = 0.0013 per column), so these numbers only
show the path runs; `measure_p_future` already sits above K/(N + NK) = 0.0026 at batch 256.

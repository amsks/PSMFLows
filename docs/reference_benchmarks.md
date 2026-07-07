# Reference benchmarks — FB + Flow (`fb_flowbc`) on cube-single (state)

Pulled from wandb **`amsks/factored-fb`**, runs named `cube_single__s{N}__ortho1000__lrb1e-4`
(the vanilla state PSM/FB + flow-actor baseline). These are the comparison targets for
our JAX PSM + flow actor.

**Reference config:** `agent=fb_flowbc`, `domain=cube-single-play-v0` (state, no encoder),
`ortho_coef=1000`, `lr_f=lr_b=lr_actor=1e-4`, `lr_actor_vf=3e-4`, `z_dim=50`,
`bc_coeff=3`, `flow_steps=10`, `num_train_steps=1,000,000`. Eval = success rate,
50 episodes, logged for tasks 1–5.

> Note: our `cube-single-play-singletask-v0` env is **task2** (`cur_task_id=2`), so the
> **task2** column is the apples-to-apples target for our `evaluation/success`.
> The **5-task mean** column is the aggregate the reference paper-style table uses.

## Success rate — mean over 5 tasks

| seed | @100k | peak | final(1M) |
|------|------:|-----:|----------:|
| 3    | 0.18  | 0.60 | 0.60 |
| 4    | 0.10  | 0.56 | 0.44 |
| 5    | 0.04  | 0.74 | 0.60 |
| 6    | 0.06  | 0.62 | 0.58 |
| 7    | 0.18  | 0.64 | 0.46 |
| 8    | 0.18  | 0.60 | 0.60 |
| 10   | 0.08  | 0.66 | 0.46 |
| **mean** | **0.117** | **0.63** | **0.53** |

## Success rate — task2 only (matches our singletask-v0 eval)

| seed | @100k | peak |
|------|------:|-----:|
| 3    | 0.30  | 0.90 |
| 4    | 0.00  | 0.90 |
| 5    | 0.10  | 1.00 |
| 6    | 0.00  | 0.90 |
| 7    | 0.20  | 0.90 |
| 8    | 0.40  | 0.80 |
| 10   | 0.10  | 1.00 |

**Top-3 by peak (the seeds we matched for our comparison): 5, 10, 7.**
Their task2 success@100k = **0.10 / 0.10 / 0.20** (these seeds are late bloomers:
peak 0.9–1.0 but low at 100k).

Seeds 1,2 failed and seed 0 was never run in this vanilla `ortho1000` set; seeds 0,1,2
exist only in experimental variants (`fb_state_iwr`, `fb_iqlcritic`, `fb_iql_ontraj_flow`).

Metric keys in wandb: `eval/reward/cube-single-play-singletask-task{1..5}-v0/success`.

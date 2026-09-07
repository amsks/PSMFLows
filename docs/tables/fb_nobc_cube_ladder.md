# Plain FB with BC anchoring off — cube-single ladder

**Every checkpoint of every seed is 0/500.** The ladder below is the whole result; the
interesting content is the two controls under it, which say *why*.

## Provenance

| | |
|---|---|
| repo | `https://github.com/LUH-AI/Factored-FB.git` |
| branch | `density-fb` (just where the newest code sits; the agent run here is the plain FB critic) |
| commit | `b62dc9d5e73f282924c29d0ab64d1f889b43532e` |
| checkout | `/mnt/home/amohan/git/Austin/Factored-FB` (venv `.venv`, py3.11, jax 0.7.1 cuda12) |
| agent | critic `fb` (`impls/critics/fb.py`) x actor `ddpgbc` (`impls/actors/ddpgbc.py`) |
| domain | `cube_single` -> `cube-single-play-v0`, OGBench data at `~/.ogbench/data` |
| runs | `$PSM_DATA/exp/FactoredFB/fb_nobc_cube/cube-single-play-v0_fb_ddpgbc_seed_{0,1,2}` |
| reports | `$PSM_DATA/logs/eval500_fb_nobc_cube_{epoch}k_sd{S}.json` |
| repro | `bash scripts/baselines/fb_nobc.sh {table\|train\|eval}` |

No PSMFlows Stage-A flow checkpoint was used. This agent trains everything from scratch in
one run: `FlowBCActor`/`DDPGBCActor` own their nets, and the only checkpoint-loading paths in
`impls/main.py` are `--restore_path` (resume) and `--warm_start_path` (the td_fb phase-2
recipe). `$PSM_DATA/flow/cube-single-play` is not a format this repo can read anyway.

## What "BC anchoring off" means here, exactly

`DDPGBCActor.loss` (`impls/actors/ddpgbc.py`):

```python
q, _ = critic.q_fn(params, obs, cond, q_actions, stop_encoder_grad=True)
q_loss = -q.mean()
if self.q_normalize:                       # default true, independent of alpha
    q_loss = q_loss / jax.lax.stop_gradient(jnp.abs(q).mean() + 1e-6)
log_prob = dist.log_prob(action)
bc_loss = -(self.alpha * log_prob).mean()  # <- the BC anchor
loss = q_loss + bc_loss
```

**The flag is `actor.alpha`, set to `0.0`** (default 0.3). That zeroes `bc_loss` exactly and
leaves the `|q|` normaliser in place, so the actor ascends `Q = <F(le(s), a, z), z>` and
nothing else. This is the repo's own no-BC idiom, not an invention:
`scripts/launch_exorl.sh` pins `ACTOR=ddpgbc`, `ACTOR_OV=actor.alpha=0.0` and explains that
`flowbc` "carries bc_coeff=3.0 ... i.e. it clones an explorer. Neither reference
implementation does that."

Choosing `ddpgbc` is also what makes this a **no-flow** arm. The repo's canonical
cube-single pairing is `--actor flowbc`, whose `bc_coeff: 3.0` distils a one-step policy out
of a flow-matching vector field; `ddpgbc` is the Gaussian TD3/DDPG policy the FB paper uses.
There is no second plausible reading on this actor: `alpha` is the only BC-ish knob
(`const_std`, `q_noise_std`, `q_noise_clip`, `boot_noise_std` are policy-noise knobs, and
`actor_pessimism_penalty` lives on the critic).

Confirmed in the runs' own logs: `bc_loss: 0.0` on every logged step, and each run's
`config.json` records `actor.alpha = 0.0`.

## Hyperparameters (merged config, all three seeds)

`batch_size 256 · z_dim 50 · L_dim 50 · num_parallel 2 · discount 0.99 ·
f_target_tau 0.005 · b_target_tau 0.005 · ortho_coef 1.0 · train_goal_ratio 0.5 ·
fb_pessimism_penalty 0.0 · q_loss_coef 0.0 · actor_pessimism_penalty 0.0 · norm_z true ·
lr_f 1e-4 · lr_b 1e-4 · lr_actor 3e-4 (ddpgbc's own) ·
forward {512, 2 layers, 2 embedding} · backward {512, 4 layers, norm} ·
left_encoder {512, 4 layers, norm, identity false} ·
actor {hidden_dims [512,512,512], alpha 0.0, const_std true, layer_norm false,
q_noise_std 0.0, q_noise_clip 0.3, q_normalize true (dataclass default)}`

Training: 500k steps, `--save_interval 50000`, `--eval_interval 100000`,
`--eval_episodes 10`, seeds 0/1/2, one H100 each, 41-45 min per run.

## Eval protocol

`scripts/baselines/fb_eval500.py`: OGBench **task_id=1 only**, 500 episodes,
deterministic actor (`eval_temperature=0`), Wilson 95% interval on the episode counts —
the same reducer and report schema as PSMFlows `tools/eval_checkpoint.py`. Task 1 of
`cube-single-play-v0` is what `cube-single-play-singletask-v0` is, so these numbers sit
next to PSMFlows' without a conversion. The sibling repo's own
`scripts/reeval_checkpoint.py` loops all five `task_infos`, which would have cost 2500
episodes per checkpoint and produced a five-task mean instead.

FB's task vector at eval is the **goal-Dirac** `z = project_z(B(goal))`
(`FBCritic.map_eval_cond`), which is this repo's default route for `fb` on cube. That is
*more* information than PSMFlows hands `psmflow` (a reward-inferred `z`), so it does not
under-sell the baseline. The reward-inference route is implemented in the same script
(`--relabel_infer N --override eval_relabel_size=N`, using
`OGBenchAdapter.relabel_infer_batch`) and was smoke-tested, but was not needed: an arm at
0/500 under the *easier* conditioning cannot be rescued by a harder one.

| epoch | seed 0 | seed 1 | seed 2 | mean +/- sd |
|---|---|---|---|---|
| 50k | 0.000 | 0.000 | 0.000 | 0.000 +/- 0.000 |
| 100k | 0.000 | 0.000 | 0.000 | 0.000 +/- 0.000 |
| 150k | 0.000 | 0.000 | 0.000 | 0.000 +/- 0.000 |
| 200k | 0.000 | 0.000 | 0.000 | 0.000 +/- 0.000 |
| 250k | 0.000 | 0.000 | 0.000 | 0.000 +/- 0.000 |
| 300k | 0.000 | 0.000 | 0.000 | 0.000 +/- 0.000 |
| 350k | 0.000 | 0.000 | 0.000 | 0.000 +/- 0.000 |
| 400k | 0.000 | 0.000 | 0.000 | 0.000 +/- 0.000 |
| 450k | 0.000 | 0.000 | 0.000 | 0.000 +/- 0.000 |
| 500k | 0.000 | 0.000 | 0.000 | 0.000 +/- 0.000 |

**Late mean (300k-500k, 15 measurements): 0.000 +/- 0.000** (sd 0.000, min 0.000, max 0.000)
- seed 0 late mean: 0.000
- seed 1 late mean: 0.000
- seed 2 late mean: 0.000
- same statistic on the affine ladder's 250k-500k window (18 measurements): **0.000 +/- 0.000**

Wilson 95% interval on each 0/500 cell: **[0.000, 0.0076]**. This is not a noisy zero.

## The two controls — and they are the point

| arm | flags | 300k | 400k | 500k |
|---|---|---|---|---|
| **BC ON** (repo's canonical cube-single pairing) | `--actor flowbc` (`bc_coeff 3.0`, `q_coeff 1.0`) | **0.110** [0.086, 0.141] | **0.084** [0.063, 0.112] | **0.108** [0.084, 0.138] |
| BC OFF + strong orthonormaliser | `--actor ddpgbc actor.alpha=0.0 ortho_coef=1000` | 0.000 [0.000, 0.008] | 0.000 [0.000, 0.008] | 0.000 [0.000, 0.008] |

Both are single seed 0, same critic, same data, same 500-episode protocol.

* **The BC term is load-bearing, and it is the only thing that is.** Same FB critic, same
  500k steps: with a behaviour anchor the arm clears the BC control at every one of its
  three checkpoints (0.110 / 0.084 / 0.108 vs 0.072); with the anchor removed it is exactly
  zero on all 1500 episodes.
* **`ortho_coef` is not the explanation.** PSMFlows' archived FB reached cube 0.721 with
  `ortho_coef=1000` while this repo ships the reference-native 1.0, so the obvious worry was
  that the baseline was mis-tuned rather than BC-starved. It is not: at `ortho_coef=1000`
  with BC off the arm is still 0/500 at 300k, 400k and 500k alike.
* The independently-run raw-action PSM baseline in the same 2026-09-06 entry
  (`docs/tables/psm_raw_nobc_cube_ladder.md`) reached the same 0.000 by deleting the same
  kind of anchor from a different agent. Two agents, two codebases, one conclusion.

## Caveats

1. **This is not a tuned pure-Q FB arm, and it is not meant to be.** No attempt was made to
   rescue it (no `actor.q_normalize=false` for a bare `-Q.mean()`, no target-policy
   smoothing `boot_noise_std=0.2/clip=0.3`, no `actor_pessimism_penalty=0.5` for
   `min(Q1,Q2)`, no `q_loss_coef=1/z_dim`). Those are the four Touati-parity knobs the repo
   exposes and all four default off. The claim is "the shipped FB arm with its BC
   coefficient set to zero collapses", not "no pure-Q FB can work".
2. `actor_loss` is pinned at exactly `-1.0` for the whole run, which is what
   `-q.mean()/sg(|q|.mean())` degenerates to once `q` has a consistent sign. The gradient is
   still live (the normaliser is stop-gradded) but the logged scalar carries no information.
3. The repo has **no recorded FB cube-single result** to check against — `results/table.csv`
   has `fb` rows only for ManiSkill and Metaworld. The `fb x ddpgbc(alpha=0)` pairing has
   been run there on ExoRL/HumEnv, never on cube.
4. `requirements.txt` on this branch is stale: it installs `-e ./third_party/ogbench[train]`
   and `third_party/` does not exist (OGBench is vendored at the repo root instead). The
   venv was built by hand with the package set `impls/` actually imports.
5. One seed each for the two controls, so their point estimates carry the usual single-seed
   width; the 0.000 conclusion does not depend on them being precise.

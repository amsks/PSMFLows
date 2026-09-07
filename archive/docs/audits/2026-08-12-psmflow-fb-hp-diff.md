# PSMFlow vs FB — hyperparameter audit (2026-08-12)

Sources are the **run flags.json**, not the yaml defaults:
FB `fb_cube_ortho1000_20260810/sd000_*` (0.716 at 500 ep) vs
psmflow `psmflow_latentpsm_cube_021456/sd000_*` (0.318 at 500 ep). Both cube-single,
500k steps, same dataset.

## Full diff

| HP | FB (as run) | psmflow (as run) | matched? | classification |
|---|---|---|---|---|
| **pessimism (measure/critic)** | `fb_pessimism_penalty` **0.0** | `pessimism_penalty` **0.5** | NO | **needs-ablation** |
| **pessimism (actor)** | `actor_pessimism_penalty` **0.0** | **0.5** | NO | **needs-ablation** |
| **basis learning rate** | `lr_b` **1e-4** | `lr_phi` **1e-5** | NO | **needs-ablation** |
| **basis network** | backward 512 wide x 4 deep, norm | phi 256 x 2 | NO | **needs-ablation** |
| forward/measure net | forward 512 x 2 | sf 1024 x 1 | NO | needs-ablation |
| batch_size | 256 | 1024 | NO | needs-ablation |
| discount | 0.99 | 0.98 | NO | needs-ablation |
| target tau | f/b 0.005 | 0.005*2 = 0.01 | NO | needs-ablation |
| z_dim | 50 | 128 | NO | intentional (ours follows the PSM reference) |
| actor bc_coeff | 3.0 | 1.0 | NO | needs-ablation |
| ortho_coef | 1000 | 1000 | yes | matched |
| norm_z | true | true | yes | matched |
| num_parallel (ensemble) | 2 | 2 | yes | matched |
| lr_actor | 1e-4 | 1e-4 | yes | matched |
| lr_actor_vf | 3e-4 | 3e-4 | yes | matched |
| actor trunk | 512 x 2, emb 2 | 512 x 2, emb 2 | yes | matched |
| flow_steps (actor) | 10 | 10 | yes | matched |
| z / w mixing ratio | `train_goal_ratio` 0.5 | `mix_ratio` 0.5 | yes | matched |
| weight_decay | 0.0 | n/a | n/a | intentional |
| stddev_clip / actor_std | 0.3 / 0.2 | n/a (flow actor) | n/a | intentional |

## Ranked shortlist — what plausibly matters

1. **Pessimism 0.5 vs 0.0 (both slots).** We apply exact-min ensemble pessimism in the
   backup AND the actor; the FB run that scores 0.72 applies **none**. D3 already found
   our actor sitting at the 44th percentile of its own Q distribution — below median —
   which is what a suppressed critic looks like. Largest single suspect.
2. **Basis trained 10x slower AND 4x smaller.** `lr_phi=1e-5` with a 256x2 phi, against
   FB's `lr_b=1e-4` with a 512x4 normed backward map. D2 measured our basis explaining
   R^2 0.13 of the reward; this is a mechanism for exactly that. The 1e-5 was inherited
   from the bilinear-PSM reference sweep, where it won — it was never re-validated for
   the latent-space agent.
3. **discount 0.98 vs 0.99** (50-step vs 100-step effective horizon).
4. **tau 0.01 vs 0.005** — our targets move twice as fast, which interacts with (1).
5. **batch_size 1024 vs 256** and **bc_coeff 1.0 vs 3.0** — lower priority; bc_coeff
   moves the actor's imitation weight that D3 found already dominant at 5:1.

Items 1 and 2 are the ones that would change the story if they carry: both are
"we suppressed the critic and starved the basis", which is consistent with every
diagnostic to date (flat Q, unrankable critic, weak basis) WITHOUT requiring the
latent-interface explanation to be wrong. They are complementary, not competing: P2
showed the interface caps a per-task loop at 0.22, but P2 inherited pessimism 0.5 and the
same anchor, so it does not by itself rule out (1).

## Bundled HP-matched arm (NOT launched — awaiting review, per the plan)

`agent.pessimism_penalty=0.0 agent.actor_pessimism_penalty=0.0 agent.lr_phi=1e-4
agent.discount=0.99 agent.tau=0.005 agent.batch_size=256 agent.actor.bc_coeff=3.0`
plus phi -> 512x4 and sf -> 512x2 (network shape changes need a config edit, not an
override). 2 seeds x 500k, tmux `psmflow_hpmatch_sd{0,1}`.

## 0b — FB run-config confirmation

The FB runs used `agent=fb agent.actor.type=flow agent.ortho_coef=1000` (flags.json
confirms `ortho_coef: 1000`, `actor.type: flow`, `actor.bc_coeff: 3.0`), i.e. the
reference cube recipe. The historical gotcha (reference override is global `ortho_coef=`,
not `agent.ortho_coef=`) does not apply here: in THIS repo `ortho_coef` lives inside the
agent config group, and the run recorded the intended 1000 rather than the yaml default
of 1.0. **FB runs are correctly configured; the 0.716-0.730 numbers stand.**

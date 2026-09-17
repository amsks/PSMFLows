# RLDP phi basis and frozen-phi GPI arms on cube (2026-09-15)

Question: does a policy-free basis phi change the affine PSMFlow's five-task number on
cube, with everything else fixed? Two stage-C arms answer it. One takes phi from a
latent-dynamics pretraining with no policy in the loop (RLDP, arXiv 2603.15857) and holds
it fixed. The other takes phi from a finished affine run and holds it fixed. The second
arm separates "frozen" from "RLDP".

References (five-task cube, 500 episodes per task, mean over tasks): affine GPI 0.284,
BC 0.111, FB 0.496.

## Pre-registered reading

| outcome | reading |
|---|---|
| RLDP arm above 0.35 | the basis geometry was the limit |
| both frozen arms near 0.28 | the basis is not the limit |
| frozen-own-phi below 0.28 | freezing itself costs |

## Part 1: RLDP phi pretraining

Code: `tools/pretrain_rldp_phi.py`, test `tests/test_pretrain_rldp_phi.py`, scheduler
wrapper `scripts/slurm/pretrain_rldp_phi.sbatch`. Commit `19afb30`.

Losses as implemented (B segments of length H+1 per batch, d = z_dim):

```
h_0     = phi(s_0)
h_{t+1} = g(h_t, a_t)          MLP on (h, a), 2 x 256 ReLU, output on the sqrt(d) sphere
L_dyn   = sum_{t=1..H} mean_B || h_t - stopgrad(phi_bar(s_t)) ||^2
L_orth  = || (1/B) sum_B phi(s_0) phi(s_0)^T - I_d ||_F^2
L       = L_dyn + lambda L_orth
```

phi_bar is a Polyak target of phi with rate 0.01. phi is `utils.psm_networks.PhiMap`
with the affine run's kwargs (z_dim 128, hidden 256, 2 hidden layers, LayerNorm+tanh
first layer, output projected to the sphere of radius sqrt(d)), read from the affine
checkpoint's `flags.json` and `agents/psmflow.py::create`. Segments are drawn from rows
whose next H transitions stay inside one episode (`terminals > 0.5` ends an episode;
the final row always does, the rule `utils.datasets.add_skill_targets` uses), so no
segment crosses an episode end. Data: `$PSM_DATA/preimages/cube-single-play.npz`, the
rows the affine run trained on (1,000,000 transitions, 996,000 valid segment starts at
H = 5).

Deviations from the paper:

| item | paper | here | why |
|---|---|---|---|
| steps | 2M | 1M | budget; the curve is flat well before 1M (below) |
| d | paper's | 128 | psi's shapes are unchanged, so the stage-C run differs only in phi |
| H | 5 (best) | 5, and H = 1 as a second checkpoint | |
| lambda | 1 (best) | 1 | |
| batch | paper's | 1024 | the stage-C batch |
| optimiser | paper's | Adam 3e-4 | |
| orthonormality term | paper's | `utils.psm_common.ortho_loss` is the B x B sample-Gram form of the measure loss, so the d x d Frobenius form is implemented in the tool; it is taken on phi(s_0) of the batch | |
| predictor | g(h, a)^T w | an MLP on (h, a) with a sqrt(d)-sphere output | |

Tests (`tests/test_pretrain_rldp_phi.py`, 5 tests, EXIT 0): the episode-end rule; the
segment sampler never crosses an episode boundary and never starts in an episode shorter
than H; the loss is finite and decreases over 20 steps on a synthetic linear system; a
checkpoint written by the tool loads through `agent.phi_restore_path` into a psmflow
agent and phi's output on a probe batch equals the pretrained network's (also for
target_phi); the pickle layout matches `save_agent`'s.

CPU smoke (200 steps, batch 64, real dataset, EXIT 0): loss 8611 -> 959, gram deviation
84.7 -> 11.0.

### Pretraining runs

PRETRAIN_TABLE

### Curves

CURVES

## Part 2: stage-C arms

Template `affine_strict_cube` (affine psi, latent index, GPI, discount 0.98, batch 1024,
ortho 1000). With `train_phi=false` phi receives no gradient, so the ortho term
(`ortho_coef=1000`) has no effect on the run; it is still computed and logged. 3 seeds,
500k steps, eval every 50k x 50 episodes, save every 50k, `--time=05:00:00`,
`scripts/slurm/train_psmflow.sbatch` with `EXTRA` holding the three keys that differ.

Before each launch: a 200-step CPU smoke at batch 64 of the same hydra line
(`$PSM_DATA/logs/cpusmoke/<group>.out`), and a check that phi's params in the smoke's
`params_200.pkl` equal the source checkpoint's phi (max abs diff 0.0) while psi's step
counter is at 201 and phi's at 1.

SMOKE_TABLE

### Differing keys against `affine_strict_cube/sd001`

DIFF_TABLES

### Jobs and run dirs

JOBS_TABLE

### In-loop ladders (50 episodes, task 2)

LADDERS

## Five-task evals (500 episodes per task, `restore_epoch=500000`)

FIVE_TASK

## Reading

READING

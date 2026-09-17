# AntMaze: test whether feature drift contributes to collapse

Status: implementation and GPU smoke complete; paired production pilot job2494139 is
running from the seed0 50k checkpoint. No final five-task result exists yet.
The action-input campaign's registered 500k evaluation is complete and missed the target.
Early 500-episode checkpoint evaluations completed as Slurm jobs2493462/2493463/2493464
for seeds0/1/2, all exit0:0. Manifest and results live under `outputs/affine_action_20260911/` as
`early_eval_manifest.json` and `reports/antmaze_sd{seed}_task1_50000.json`.

## Evidence and question

AntMaze action-input PSMFlow scored .76/.56/.66 at 50k across seeds0/1/2, but
.06/.02/.14 at 350k. These are only 50-episode evaluations on task1. They do not establish
an early benchmark solve. The first action is to re-evaluate the three saved 50k
checkpoints with 500 episodes, the original configurations and frozen flow, before
selecting this checkpoint for a causal experiment. This is an explicitly retrospective
diagnostic, separate from the fixed-500k, five-task headline protocol.

That re-evaluation is now complete: **.844/.564/.752**, mean **.7200 ± .3545**
(Student-t95% halfwidth across three training seeds,df2). Each cell has500episodes;
all reports restore50k from the correct run with no agent CLI overrides. This confirms
useful early task1 performance, with substantial seed variation. It does not establish
the500k five-task target. `reports/antmaze_50k_task1_diagnostic_summary.json` records the
calculation and source paths. The premise for the predeclared freeze-phi pilot is supported.

The paired driver passed its focused CPU suite (22 tests) and a 200-step H100 smoke
(job2494102, exit0:0). The smoke reached absolute step50200 in both arms, held the frozen
phi TrainState bitwise fixed, advanced the control phi, kept psi finite, preserved the
frozen target/online equality, and kept the flow weights unchanged. The production pair
(job2494139) uses the same source checkpoint, data batches, and RNG stream; its interim
50-episode task1 values were frozen/control .70/.36 at100k and .10/.18 at150k, so the
early lead is not yet stable evidence.

Question: does continuing to change phi, and hence the reward readout and successor
feature coordinates, contribute to the loss of useful action rankings?

## Smallest controlled training experiment

Use a predeclared seed0 pilot. Branch its same 50k checkpoint into two 450k-update
continuations, both reaching a total of 500k updates including the source training:

| Arm | phi | target phi | affine psi/index encoder |
|---|---|---|---|
| Matched control | continue training | usual Polyak updates | continue training |
| Frozen basis | hold parameters exactly | same fixed basis | continue training |

Initialize target-phi from online phi **in both branches** before the fork. Freezing an
already lagging target creates a persistent online/target basis mismatch; synchronizing
only the treatment would add an uncontrolled change. This matched control is distinct
from the already-running uninterrupted trajectory.

Restore identical psi, target-psi and optimizer states. Use the same data-index sequence,
latent RNG schedule and evaluation relabel sample for both branches. A reproducible
paired continuation needs these explicitly restored or identically reseeded; a checkpoint
of the agent alone does not imply that the dataset's NumPy RNG was saved.

Freeze by bypassing phi optimizer and target updates, not by relying on an informal
learning-rate change. Check exact parameter equality across updates, target/online
agreement, finite psi updates, restored settings and the 50k+450k budget. Preserve action
conditioning, gamma=.99, loss/pessimism, flow, index family and GPI settings.

Track task1 at fixed diagnostic checkpoints; evaluate the final endpoint on all five
tasks. Improved feature metrics alone are not a win. Any general performance claim still
requires three independent training seeds, 500 episodes per task, and a mean with a
95% interval across seed means. Fresh confirmation seeds should be set before inspecting
the pilot outcome. Source-checkpoint provenance and additional update counts must be
recorded; the existing strict reporter rejects continuations and must not be bypassed or
given misleading epoch labels.

## Decisions

- If the 500-episode re-evaluation does not confirm useful early performance, revisit
  the early-basis premise before launching the continuation pair.
- If the matched control loses performance while the frozen branch retains or improves
  it, that is evidence that basis drift contributes; confirm across seeds before claiming
  a stable recipe.
- If both lose performance, freezing this basis was insufficient. It does not show that
  representation quality is adequate or rule out representation problems. Independently
  trained reward-free temporal features (HILP/RLDP), with their updates counted within
  500k, would be the next representation intervention.
- If a demonstrably useful fixed basis still fails in affine PSM, examine successor
  estimation and the constant-latent continuation family. Real-reward DSRL success alone
  does not validate that family's zero-shot values.

Mass conservation/centering is a separate idea, not bundled into this test: density-space
pessimism itself changes mass. See the action-conditioning design for that derivation.

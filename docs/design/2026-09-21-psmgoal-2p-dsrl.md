# psmgoal-2P-DSRL: 2P-DSRL on a frozen psmgoal basis (cube)

Date: 2026-09-21. Status: spec -> implement -> run.

## What this is

2P-DSRL is the co-trained measure+actor downstream on a frozen basis: freeze phi, re-init
psi, co-train psi with a task-conditioned, Q-normalized, bc-0 `dsrl_sac` actor. It already
exists in the psmflow agent and was run on psmflow's own frozen basis:

| arm (frozen psmflow basis) | five-task cube |
|---|---|
| 2P-DSRL, 500k basis (50k/100k/250k/500k psi) | 0.125 / 0.100 / 0.072 / 0.090 |
| 2P-DSRL, 1M basis | 0.129 / 0.104 / 0.088 |
| BC control | 0.111 |
| 2P-GPI (frozen basis, no actor) | 0.328 |

That arm sits at or below BC. This run repeats the exact recipe with the frozen basis replaced
by a state feature projected from psmgoal's learned measure. It closes the psmgoal-basis cell of
the measure+actor family. It is not expected to beat BC.

## Why it is likely BC floor (pre-registered)

psmflow's features are good (relabel gives 0.867), and the co-trained measure+actor downstream on
those good features still lands at BC. So the failing part is the measure-as-critic path, not the
basis. psmgoal's basis is weaker (zero-shot 0.06 vs 0.28), so swapping it in is not expected to
cross BC. This run tests that expectation directly.

## The frozen basis: a reward-agnostic state feature from psmgoal

psmgoal's basis is `RLUMeasure(s, u, g) -> (phi[128], b)` (field `agent/basis`). Define

    f(s) = project_sphere( mean over u in U, mean over g in G  of  phi_part(RLUMeasure(s, u, g)) )

- `U`: a fixed set of `K_u = 8` prior action-latent draws, sampled once, clipped to `u_clip`.
- `G`: a fixed set of `K_g = 32` states drawn once from the dataset marginal. Reward-agnostic on
  purpose -- NOT the rewarding goals. A task-specific goal set would make `f` task-dependent and
  break the zero-shot basis.
- `project_sphere`: L2-normalize to radius sqrt(128), matching psmflow's `project_z`.

`f(s)` is a pure function of `s`, dim 128, a drop-in for psmflow's `PhiMap` phi slot. psmflow only
calls phi on dataset states (TD target basis and reward readout); acting never calls phi. So `f`
never has to generalize beyond the dataset.

## Wiring (minimal, guarded -- zero change to psmflow when unused)

- `utils/psm_networks.py`: new frozen module `PsmgoalProjectedPhi(obs) -> f(s)`. Wraps
  `RLUMeasure`, holds `U` and `G` as constants. Its only params are the frozen `RLUMeasure` params.
- New agent `f_psmgoal2p` (thin: reuses `PSMFlowAgent`; overrides only phi construction and the
  basis-load path). If a subclass is not clean, use a guarded `external_basis` config branch in
  `agents/f_psmflow.py` that defaults off. Existing psmflow behavior must be bitwise unchanged when
  the flag is unset -- assert this with a test.
- Basis load: read `pickle["agent"]["basis"]["params"]` from a psmgoal checkpoint, load into the
  wrapped `RLUMeasure`, assert per-leaf shapes, freeze (`train_phi=false`). Write a sidecar
  recording env, `basis_restore_path`, `basis_restore_epoch`, matching the existing latent guards.

## Flags (mirror 2P-DSRL exactly)

    psi_form=affine policy_index=task_vector train_actor=true acting=actor
    actor_mode=dsrl_sac actor.bc_coeff=0.0 actor.target_entropy=-3.4657
    u_clip=3.0 train_phi=false discount=0.98

Head choice: keep `dsrl_sac` to stay comparable to the existing 2P-DSRL numbers. The RLDP-faithful
`ddpg` head is a follow-up only if the first cells are alive.

## Runs

- Frozen basis: `psmgoal_lift_cube_gc/sd00{0,1,2}` @ 750k (the current-code embedding; the only step
  with a five-task eval on record). Seed k of this run uses basis seed k.
- Flow: `cube-single-play` @ 500000. Preimages: `cube-single-play.npz`.
- psi 500k steps, eval every 50k, save 250k/500k. Extend to 1M only if a cell clears BC.
- Five-task eval500 (500 ep/task), 3 seeds. Report mean +/- 95% CI; quote BC 0.111 beside it.

## Expected outcomes (state before results)

| outcome | five-task | reading |
|---|---|---|
| BC floor | ~0.09-0.13 | measure+actor path is the limit, not the basis; closes the cell |
| alive | > 0.20 | psmgoal's goal-structured features help the actor downstream; extend to 1M + try the ddpg head |
| degenerate | ~0.00 | projected f(s) collapsed; check E[f f^T] conditioning, fix normalization, rerun |

Prediction: BC floor.

## Smoke + launch gates (project discipline)

CPU ~200 steps on the exact code path: assert `f(s)` finite and `E[f f^T]` non-degenerate, psi and
actor both step, the run's `flags.json` carries the values above. Then SLURM, 3 seeds, one GPU each;
re-read each run's `flags.json` after launch to confirm the frozen-basis path and flags landed.

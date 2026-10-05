# psmgoal reward-relabel diagnostic (cube task 2)

Date: 2026-09-21. Status: spec -> implement -> run. Runs in parallel with f_psmgoal2p.

## Question

Do psmgoal's features carry reward, the way psmflow's did? The psmflow test relabeled reward from
its features (`r_hat = phi(s').w`), scaled it to the dataset convention, and trained a DSRL-NA
scalar critic that reached the real-reward level: 0.867 vs 0.882 (cube task 2). If psmgoal passes
this, its collapse is the zero-shot measure readout, not the features. If it fails, the features
themselves are the problem. This gates every downstream we would build on psmgoal.

This is a labels-only, per-task diagnostic. It is NOT a zero-shot number and is not a headline.

## Method (mirror the psmflow 0.87 test)

1. New relabel path for the psmgoal agent (its measure is the triple `M(s,u,g)`, so the existing
   `tools/relabel_reward_rhat.py`, hardwired to psmflow's `phi(s').w`, cannot be used). Load
   `psmgoal_lift_cube_gc/sd000 @750k`. Produce a per-transition scalar reward, two readout modes:
   - **goal_set** (primary): `r_hat(s',u') = mean over g in G of M(s',u',g)`, via psmgoal's
     `infer_eval_goals` (goal set G + `eval_w_star`) and `mesh_M`, with `u` = the row's
     `noise_preimage` clipped to `u_clip`. This is the goal-set readout.
   - **regression** (control): `r_hat(s') = f(s').w`, `w = lstsq(f, real_reward)`, `f` = the
     psmgoal state feature (goal/action-marginalized). This isolates feature quality from the
     learned coefficient `w*(g)`; it is the direct analogue of the psmflow test.
2. Scale `r_hat` onto the dataset -1/0 reward convention (`--match_real_scale`), write a
   `reward_override` npz (reuse the existing scale + npz writer, which are agent-agnostic).
3. Train the DSRL-NA scalar critic on it: `agent=psmflow` + `dsrl_na` (the `launch_dsrl_na.sh`
   arm), `dataset.reward_override_path=<npz>`, 3 seeds, cube task 2.
4. Eval task 2, 500 episodes. Compare to the real-reward control (0.882) and the task-2 BC floor.

## Expected outcomes (state before results)

| outcome | task-2 success | reading |
|---|---|---|
| ~0.87 | near the 0.882 real-reward control | psmgoal features carry reward; the collapse is the zero-shot readout, not the features; motivates a scalar-critic zero-shot lever |
| ~0.5-0.8 | partial | features carry reward weakly |
| collapse (~0.01) | at/near zero | either the -1/0 mapping is wrong (raw-scale bug, as in the psmflow raw case) or the features do not carry reward |

If goal_set collapses but regression passes, the fault is the coefficient `w*(g)`, not the features.

## Notes

- `r_hat` source checkpoint: `psmgoal_lift_cube_gc/sd000_s_2519395.0.20260918_161523 @750000`.
- Reward file named after its source (checkpoint, step, readout mode) so two npz do not collide.
- Only the relabel tool is new. The scalar-critic training arm already accepts any
  `dataset.reward_override_path`.

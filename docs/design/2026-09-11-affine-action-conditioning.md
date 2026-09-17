# Affine PSMFlow with action-conditioned transition features

Status: all six500k runs and all30 task evaluations completed on2026-09-11.
Cube five-task mean43.92% ±38.02points; AntMaze1.08% ±1.46points (three-seed95% t
halfwidths). This recipe did not meet the performance target. Exact-action conditioning
alone did not prevent AntMaze collapse. See the dated HANDOFF and persisted aggregates.
Requested endpoint: at least 70% success on cube and antmaze after 500k Stage-C steps,
mean and 95% confidence interval across seeds 0, 1, 2. For comparison with TD-JEPA,
evaluate all five tasks per environment; retain the historical single-task results too.

## Evidence and decision

The September 10 handoff shows that real-reward DSRL-NA can steer the frozen flow,
whereas the affine zero-shot method is unstable and stronger optimization of the fitted
reward can fail. This does not identify a single cause: DSRL changes reward supervision,
critic parameterization, and policy improvement simultaneously.

The current measure fits transitions at clipped point inverses of an ODE decoder but acts
through the distilled one-step decoder. Approximate inversion plus clipping can therefore
associate a transition with a different executed action. Its action-latent input also
does not identify latents that decode to identical actions. The exact action-conditioned
affine variant below has no recorded controlled result. Archived raw-action actors without
BC and the separate action-critic auxiliary do not test it.

Implement one switch, `measure_action_input: latent | action`, default `latent`:

```
F(s,a,c) = A(s,a) c + beta(s,a)
psi(s,u,u') = F(s,G(s,u),w_enc(u'))
online transition fit: F(s,a_dataset,w_enc(u'))
bootstrap: F_target(s_next,G(s_next,u'),w_enc_target(u'))
acting: max over u,u' of F(s,G(s,u),w_enc(u'))^T task_w
```

The policy-index encoder and latent-only action search are unchanged. Every action queried
for bootstrapping or execution is decoded through the frozen flow. Actual dataset actions
train the transition model, so their successor-state targets have the correct provenance.
Affineness is in the policy coordinate, not in action or task. This is a coordinate/modeling
change, not an implementation of a new non-affine critic.

Initial experiment uses `measure_u_samples=1`. Reject action-input plus multi-preimage
augmentation rather than pretend extra approximate inverses define extra exact transitions.
Keep target pessimism, basis loss, reward inference and actor settings unchanged to isolate
the effect. A positive result would not exonerate the basis or establish a general theorem.

## Two mathematical limits

The formal in-sample proposition verifies the marginal `(s_next,a_next)`. It does not
establish coverage of the full conditional approximator input `(s_next,a_next,u')`:
training action and policy index are independent, whereas the deterministic continuation
ties them. For a one-state two-action/two-index example, training puts 1/4 on each pair,
while bootstrap puts 1/2 on each diagonal pair. The action marginal agrees but the joint
maximum density ratio is 2, not 1. With continuous deterministic identity actions, the
diagonal has zero product measure. Bellman closure remains correct; projected-function
approximation stability does not follow from the marginal argument. Action conditioning
does not eliminate this issue, though it removes inverse-coordinate errors.

Current GPI optimizes over `w_enc(u')`, not all feasible occupancy measures. Arbitrary LP
optimization over its affine coordinates is unjustified without feasibility/nonnegativity
constraints or a spanning policy family. It is not the first experiment.

Accurate values for constant-latent continuation policies also do not establish a high
success floor: the GPI theorem compares against that policy family. GPI can strictly beat
each constituent policy by changing its selection with state, but real-reward DSRL's strong
state-dependent steering result does not prove that this family supplies useful zero-shot
rankings. Low success in a finite fixed-index roster would be diagnostic evidence, not an
upper bound on the full family or on GPI.

## Alternative to test if this fails

There is a separate affine constraint to diagnose before modifying the objective. With
reference-state feature mean `mu=E[phi]`, a continuing, unnormalized successor measure
has `psi^T mu=C=1/(1-gamma)`. For policy coordinates of full affine span this requires
`A^T mu=0` and `beta^T mu=C`. With a restricted encoder image only constraints on its
affine span follow. A constant feature plus centered residual features can enforce these
identities and preserve affine policy dependence. Positive task-vector normalization does
not change rankings under an exactly policy-independent constant-value shift.

However, the current two-head target `mean - .5*spread` is a pointwise minimum of densities;
even two heads with mass C can have a minimum with mass below C. Hard mass conservation
therefore changes the current conservative learning problem. Before such an intervention,
measure candidate-dependent `psi^T mu` and reward-shift-induced ranking changes on common
candidate panels, separating each head from the pessimistic target. A missing constant
feature can make absolute mass inaccurate without damaging rankings, so mass error alone
is not a performance diagnosis. Finite horizons/termination also change the relevant mass.
No mass-constrained implementation or additional training run is part of this campaign.

Reference RLDP and HILP provide reward-free self-predictive or temporal-distance supervision
for the state basis. A staged basis-training/freeze experiment is distinct from failed
learning-rate/orthogonality sweeps and has no controlled live PSMFlow result. If implemented,
count basis pretraining inside the 500k budget and synchronize online/target bases when
freezing; zero gradients alone do not freeze Adam momentum. Do not train on task rewards.

## Validation and experiment protocol

1. Test exact action-conditioned values against the raw affine head; verify two latents with
   the same decode receive identical scores, the dataset-action fit is independent of its
   approximate cached latent, actor gradients pass through the decoder while decoder weights
   remain unchanged, and legacy/default configuration behavior is preserved.
2. Test the factored index-panel path against the explicit pair path and a finite update.
3. Smoke the exact production model on each environment for 200 steps, persist logs and
   inspect the saved flags. Report errors without changing unrelated settings.
4. Fixed recipe, three seeds per environment, exactly 500k Stage-C optimizer steps;
   cube gamma .98, antmaze gamma .99 to match their existing affine controls. Checkpoints
   every 50k. No best-checkpoint selection. Freeze code/config for submitted jobs.
5. At 500k evaluate 500 episodes per task per seed. Compute each seed's five-task mean,
   then the mean and Student-t 95% interval across the three seed means (`df=2`). Also
   report individual tasks and 400k/450k/500k stability where feasible. Do not treat task
   cells or episodes as independent training seeds.

Expected positive signature: an improvement at the fixed endpoint across seeds without
late collapse. Expected failure: the existing reward-basis or continuation-family problem
dominates and action conditioning yields no useful improvement. Finite loss, low decode
error and healthy feature covariance are wiring checks, not success criteria.

## Benchmark reference

[TD-JEPA Table 1 and Appendix E](https://arxiv.org/html/2510.00739) report state-based
OGBench five-task averages, 1M updates, ten seeds, mean plus standard error:

| Environment | HILP | FB | RLDP | TD-JEPA |
|---|---:|---:|---:|---:|
| cube-single | 74.20 ± 3.53 | 49.60 ± 3.83 | 19.80 ± 2.41 | 34.20 ± 2.88 |
| antmaze-medium-navigate | 83.60 ± 2.63 | 73.00 ± 2.72 | 74.60 ± 4.15 | 70.40 ± 3.72 |

Thus 70% alone does not beat every comparator. Be explicit about 500k versus 1M and the
separate Stage-A training cost. A superiority claim needs matched protocol and appropriate
uncertainty, not just a point estimate above a printed mean. The official sources do not
name a separate “with steering” variant; user clarification is pending.

Reference code inspected by the search audit:
`../Factored-FB/impls/critics/{psm,rldp,hilp}.py` relative to the repository parent,
plus [official TD-JEPA](https://github.com/facebookresearch/td_jepa),
[PSM](https://arxiv.org/abs/2411.19418), and
[HILP](https://github.com/seohongpark/HILP).

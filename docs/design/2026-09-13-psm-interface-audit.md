# PSM implementation and frozen-flow interface audit

Date: 2026-09-13. Source HEAD: `daa0268`; existing untracked work preserved.
Scope: implementation validation, existing experiment revalidation, mathematical
counterexamples, and a proposed ExORL Walker isolation protocol. No production training
code, checkpoints, or running jobs were changed. No Walker training was performed.

This scope statement describes the initial audit. The subsequently authorized Walker
setup/launch is tracked separately in the
[September 13 launch plan](../plans/2026-09-13-walker-zero-shot-psm.md) and newest HANDOFF entry.

## Verdict

The default affine PSMFlow passes the implementation checks performed here. Its affine
factorization, current-action/policy-index slots, diagonal continuation, target detachment,
reward readout, GPI pairing, and frozen decoder are implemented consistently with the
repository's LatentFlowPSM construction. This is narrower than validating original PSM,
the assumptions behind the construction, or the trained successor values.

There is a substantive interface problem to investigate: training covers independent
current actions and policy indices, while bootstrapping evaluates their diagonal coupling.
Perfect behavior cloning does not remove that shift. An exact, realizable affine example
below has divergent projected TD despite a perfect flow and fixed phi. This establishes a
failure mode, not the root cause of the production networks' performance.

Two diagnostic controls also fail their intended contracts: D1b's held readout is not a
fixed reward when phi learns, and D2 still receives task-specific termination labels.
Earlier evidence therefore does not isolate the reward readout as the sole cause.

## Fresh verification and its limits

The current-repository core suites returned **97 passed, 2 skipped** in 257.87 seconds.
The skipped real-flow cases were subsequently run with the available cube Stage-A
checkpoint; those two plus the default affine chain task returned **3 passed** in 107.01
seconds. Thus **100 distinct PSMFlow tests passed**. These are CPU checks, not environment
performance evaluations.

Covered suites: agent, affine head, policy index, action conditioning, freeze phi, and
DSRL-NA; plus real checkpoint loading, rejection of a foreign environment checkpoint, and
learning goalward GPI selection on the five-state chain. The chain test checks policy
direction; it does not establish calibrated successor values on continuous environments.

Independent reference-port review ran the existing sibling suite
`../Factored-FB/tests/impls/test_psm_seam.py`: **35 passed in 67.70 seconds**. Its fixture
comes from Factored-FB's Torch adaptation. It injects action samples and post-Torch-update
phi weights, so it validates network/loss transcription while deliberately excluding
sampler and optimizer equivalence. It is not a fresh reproduction of the official PSM
benchmark. The sibling tree was unchanged.

Persisted current-repository evidence:

- [Core test results](../../outputs/psm_interface_audit_20260913/pytest_core.xml)
- [Real flow and chain results](../../outputs/psm_interface_audit_20260913/pytest_real_flow_and_chain.xml)
- [Algebra, control probes, source hashes, and existing-report validation](../../outputs/psm_interface_audit_20260913/algebra_and_artifacts.json)
- [Reproduction tool](../../tools/diag_psm_interface_audit.py)

The independently expanded contrastive objective matched the live helper within
`2.49e-7`; its gradient matched within `6.39e-9`, and the stopped target gradient was zero.
At two heads, `mean - .5 * spread` matched elementwise minimum within `1.20e-7`.
The new diagnostic passed Ruff. CPU execution succeeded despite a CUDA-plugin discovery
warning on this machine; no GPU numerical result is claimed.

Reproduction:

```bash
JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python tools/diag_psm_interface_audit.py \
  --report_out outputs/psm_interface_audit_20260913/algebra_and_artifacts.json
```

## What is implemented correctly, and what that does not imply

| Component | Verified behavior | Remaining limitation |
|---|---|---|
| Affine head | `A(s,u) @ w_enc(c) + beta(s,u)`; A/beta do not see c | Architectural affineness does not impose feasible Bellman-flow occupancies or establish sufficient rank |
| Continuation | Online `psi(s,c,u_data)`; target `psi_target(s_next,c,c)` | Correct fixed-index Bellman equation; conditional input coverage still differs |
| Loss | Contrastive measure fit plus orthogonality; target stopped | Projection, finite features, and nonlinear optimization need not be contractive |
| Flow | Frozen weights and correct loading/shape checks | Reachable expert actions do not imply useful fixed-noise continuation policies |
| Action-conditioned branch | Fits actual data action; decodes latent queries consistently | Already failed to prevent AntMaze collapse; does not remove index coupling |
| Task inference | `project(E[r phi])`, optional Gram solve | Reward projection can differ from the intended task; orthogonality is approximate |
| GPI | Correct action-major/index-minor pair selection | Default is 64 action samples ×64 policy indices, freshly redrawn, not one pool of512 |
| DSRL-NA | Action TD critic, latent distillation, actor | Real-reward arm changes reward, continuation policy, and termination relative to PSMFlow |

Code locations: `utils/psm_networks.py:321–381`; `agents/psmflow.py:602–685`,
`:776–887`, `:1165–1274`, `:1490–1600`, `:1610–1678`, `:1847–1875`.

Original PSM must be distinguished from all of these. Its theory constructs an affine
Bellman-flow solution space with feasibility restrictions. Its released practical agent
learns a proto branch and phi, then a separate reward-conditioned successor-feature branch
and actor. Current PSMFlow combines a different policy family with a single affine
successor head and GPI. It is not original PSM with only its actions reparameterized.
[PSM paper](https://arxiv.org/html/2411.19418),
[author implementation](https://github.com/agarwalsiddhant10/PSM).

### Source-lineage correction after the user's clarification

The preceding author implementation is the **zero-shot PSM comparator**, not the
source of our earlier **full-affine PSM**. The user supplied the latter as
`https://github.com/CalCharles/RLU/tree/main/controllable_agent`. That URL could not
be freshly retrieved (public repository/API/raw endpoints returned 404 during the
source review); the following is verified from our archive, not an accessible
upstream RLU revision.

`archive:archive/agents/affine_psm.py` at commit
`ee0e1746b36f48300eee06ad02e66b16f46851ff` implements learned affine coordinates,
sampled nonnegativity constraints with dual/hinge task-time inference, and actor
distillation. Its July 26 source correction identifies RLU `discrete_psm.py`, not
the unused/broken continuous `psm.py`, as the executed reference. It uses a
normalized `(1-gamma)` immediate source; its factored network has separate
feature/bias branches and a tanh-bounded bias. These are substantive differences
from current `AffinePsiMap`, beyond removing a flow.

Thus the earlier statement about absent feasibility restrictions applies to
**current PSMFlow**, not to the archived RLU-derived agent. The archived agent
does have approximate constrained inference; it does not establish exact feasible
occupancies or implement an exact unconstrained-coordinate LP (coordinates are
projected onto a sphere during inference).

The archived design's section 8 explicitly defers **reward-based full inference**.
Its existing inference and actor objectives are goal-conditioned. Walker's dense
stand/walk/run/flip tasks need reward integration over the learned measure and
corresponding inference/actor objectives; substituting a single goal would change
the task. The first Walker run therefore uses plain raw-action zero-shot PSM as
a positive control. See the [declared launch plan](../plans/2026-09-13-walker-zero-shot-psm.md).

## The joint-coverage problem: an exact counterexample

The formal source, `git show 5249267:PAPER/main.tex`, Proposition `insample`, assumes
exact cloning and an independent prior index. It establishes the bootstrap marginal
`(s_next, action_next)`. The learned function also receives the policy index c.

With two uniform actions and indices, training assigns probability1/4 to each `(a,c)`
pair. A deterministic indexed continuation assigns1/2 to each diagonal pair `(c,c)`.
Both have the same action marginal, but the joint maximum density ratio is2. With an
invertible continuous flow, the diagonal has zero measure under independent continuous
action/index draws. Using raw actions in the critic preserves this distinction.

The gap can matter even with an exact solution inside the model class. Consider one
continuing state, identity flow `G(u)=u`, behavior/prior `N(0,1)`, and fixed `phi=1`.
Let

```
f(u) = tanh(u)
b = sqrt(2) - 1
lambda = (1-b)/2, kappa = (1+b)/2
h(u,c) = lambda*f(u) + kappa*f(c)
m_theta(u,c) = theta_0 + theta_1*h(u,c)
```

This is affine in the policy coordinate, and the correct constant successor density
`1/(1-gamma)` is realizable. The target uses `h(c,c)=f(c)`. Exact population least-squares
TD fitting under independent Gaussian u,c gives

```
theta_0_next = 1 + gamma*theta_0
theta_1_next = gamma*kappa/(lambda^2+kappa^2)*theta_1
             = gamma*(1+sqrt(2))/2*theta_1
```

At gamma=.9 the second multiplier is **1.086396**, greater than1. The diagnostic verifies
both the equivalent finite matrix calculation and Gaussian quadrature. There is no
pessimism, learned phi, inverse error, or flow approximation. A unit-norm policy embedding
can represent the same example using `(f(c), sqrt(1-f(c)^2))` and ignoring its second
coordinate. This disproves a general implication from marginal C=1 to projected stability;
it does not demonstrate this unstable mode in the trained network.

Also, a behavior model does not identify temporally coherent policy indices. State-dependent
prior-preserving transformations of its noise leave the cloned action distribution
unchanged while changing every constant-noise policy. DSRL's successful state-dependent
steering and the unsuccessful fixed-index family are therefore compatible.

## Concrete diagnostic defects and corrections

**D1b does not hold the reward fixed.** `phi_readout_fixed` uses current `self.phi` at
`agents/psmflow.py:1219` with held `na_rw` and scale. Phi still updates at `:1493–1498`.
The probe changes phi while preserving the held coefficients and observes a changed
reward. The comment that QA sees one reward between refits is false unless phi is also
frozen. This compromises the interpretation as stationary fitted-reward optimization.

**D2 is not task-agnostic reward-free training.** `synthetic_w` replaces the reward but
still multiplies the target by `batch['masks']` at `agents/psmflow.py:1242–1243`.
OGBench's relabeler sets these masks to one minus the selected task's success predicate.
The actual published preimage files contain **20,810/1M cube** and **9,113/1M AntMaze**
zero masks; in both files the masks exactly equal the negative task rewards. The diagnostic
confirms D2 loss is invariant to changing real rewards but changes when masks change.
It learns synthetic rewards in a task-specific terminated MDP. Its low success remains an
experimental result, but it is not a clean reward-free control. The main measure's choice
to ignore these task-specific masks is consistent with its continuing, reward-free setup.

**The fitted-return ratio compares different rewards.**
`tools/diag_fitted_vs_true_return.py:69–87` restores each policy's own phi, refits its w,
and computes its own scale; `:98` uses that reward and `:116–120` divides cross-run values.
Consequently, the historical7.805× ratio is `J_rhat_A(pi_A)/J_rhat_B(pi_B)`, not a
comparison of two policies under one rhat. Equal standard deviations do not make reward
functions equal. The probe also refits its eval readout instead of using D1b's held training
readout. It does not establish that a poor policy optimizes a single shared surrogate
better than a successful policy. A corrected mechanism probe must freeze one reward
evaluator and cross-score all policies with common discount and stopping semantics.

**The single-head configuration is numerically broken.** `targets_uncertainty` divides
by `P*(P-1)` and returns NaNs for P=1. Reproduced here. Default production P=2 is unaffected;
this is not an explanation of those production runs.

**The previous minimum/contraction explanation overclaims.** For two-head minimum,
`|min(x)-min(y)| <= max_i |x_i-y_i|`; the exact ensemble Bellman map is still a
gamma-contraction in joint sup norm. The fitted disagreement-growth model in the September8
measure-loss audit is not a proof otherwise. Projection can destroy stability, as above.
Density minimum nevertheless changes mass and is not pessimistic for every signed reward:
heads `(2,0)` and `(0,2)` have mass2, their minimum has mass0, and reward `(-1,-1)` gives
value0 under the minimum versus-2 under either head. Zero-pessimism sweeps already failed
as a general performance remedy; this correction is not a reason to repeat them.

**Several default claims need qualifications.** Prior clipping changes the prior law;
the current point latents also clip on **2.2004% of valid cube rows** and **3.1213% of
valid AntMaze rows** in the inspected files. These are freshly counted, not fresh decode
errors. The action-conditioned experiment already addressed current-action provenance.
The dataset sampler excludes invalid rows, despite stale agent comments implying all
default runs consume repaired rows. A flow decode is not by itself proof of exact data
support or C=1. GPI's theorem additionally requires accounting for finite action search
and changing policy panels; the current independent action panel need not include each
base policy's own action.

## Completed freeze-phi pilot: corrected status and result

Slurm job2494139 is **COMPLETED, exit0:0**, confirmed with current accounting. Its completion
record and ten existing500-episode reports were revalidated against flags, checkpoints,
seeds, episode outcomes, task IDs, worker protocol, and source budget. The two agent
configs differ only in `train_phi`. No evaluations were rerun.

| Seed0 endpoint at500k | Task1 | Task2 | Task3 | Task4 | Task5 | Five-task mean |
|---|---:|---:|---:|---:|---:|---:|
| Continue phi | .076 | .004 | .000 | .000 | .002 | **.0164** |
| Freeze phi from50k | .128 | .000 | .000 | .000 | .000 | **.0256** |
| Same-flow BC | .056 | .174 | .026 | .006 | .100 | **.0724** |

The source checkpoint's retrospectively selected task1 score was .844 at50k. Both
continuations lose most of it. Frozen phi and its optimizer remain bitwise preserved,
target phi equals it, flow weights remain unchanged, and the paired RNG streams agree,
according to the completed driver's checks. Frozen/control terminal Gram deviations are
approximately .347/.350, so neither endpoint presents a rank-one basis collapse.

This pilot shows ongoing phi changes are **not necessary** for the loss of early
performance along this continuation. Freezing this phi is insufficient; it does not
establish that phi is adequate, rule out successor-estimation failure, or justify an
across-seed conclusion. No seed-level CI exists from a single training seed. Source:
[validated report](../../outputs/psm_interface_audit_20260913/algebra_and_artifacts.json).

## Walker isolation protocol

Walker is a good positive-control environment, but plain PSM versus full PSMFlow alone
changes too many components to isolate the interface.

Use the following gates in order:

1. **Implementation equivalence:** original PSM versus its port, with identical weights,
   rows, proto actions, random draws and optimizer states for1/10/200 updates. Compare
   gradients and full states, not only scalar losses. Separately test uninjected sampling.
2. **Original PSM positive control:** replicate Walker with a declared reference revision
   and resolved training/evaluation config. Record its actual runtime settings.
3. **Matched free/affine control without a learned flow:** identity decoder with one shared
   bounded action prior, identical constant-index family, GPI, losses, inference, and
   training budget; only free versus affine head changes. Comparing either directly with
   original PSM does not isolate factorization because original's policy family/extractor
   also differ. Identity control is not a behavior-cloning or C=1 experiment.
4. **Flow intervention:** substitute the Walker behavior flow under the same free/affine
   protocol. Include BC from that exact flow. This needs Walker StageA/B artifacts.

The author paper and release conflict, so “matched HPs” needs an explicit label:

| Setting | Paper Walker table | Released defaults / effective code |
|---|---:|---:|
| Replay |5M | First10% of supplied replay retained |
| Batch / representation dimension |1024 /128 |1024 /128 |
| Gamma / target tau |.98 /.01 |.98 /.01 |
| Adam learning rate |3e-4; separate Double-GD1e-4 entry |phi/SF/actor1e-4 |
| Updates |2M |3M |
| Reward inference samples |10k |5k in evaluation config |
| Orthogonality weight |1 |Literal1 in proto loss |
| Architecture |Table lists1024×3 |phi256×2; separate successor/actor embeddings |

Other release hazards include proto actions sampled in `[-2,0)` despite Walker's `[-1,1]`
action bounds, and a stale Hydra target. Reproducing a release and repairing its sampling
are different experiments. Current archived cube defaults (ortho1000, slow phi, BC actor)
are not a Walker match. Sources:
[paper Appendix B](https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/siddhant_agarwal_psm.pdf),
[released agent](https://raw.githubusercontent.com/agarwalsiddhant10/PSM/main/agent/psm.py),
[released training config](https://raw.githubusercontent.com/agarwalsiddhant10/PSM/main/train.py),
[reward evaluation config](https://raw.githubusercontent.com/agarwalsiddhant10/PSM/main/evaluation/reward.yaml).

For the standardized method comparison, predeclare Walker RND5000 complete episodes,
seeds0/1/2, B1024, z128, gamma.98, tau.01, ortho1, and explicitly resolved learning rates.
Use2M as the fixed paper-budget endpoint with50k/100k/250k/500k checkpoints for diagnosis;
500k failure alone does not refute the longer reference. Use stand/walk/run/flip, fixed10k
relabel rows and500episodes/task. Average tasks within each training seed, then compute
mean and95% Student-t intervals across seeds. Any recipe that combines release learning
rates with the paper budget must be labeled a standardized comparison, not exact replication.

The sibling ExORL loader already handles dummy first actions via
`(obs[t], action[t+1], obs[t+1])`. Preserve physics alignment for reward relabeling.
Current PSMFlows' environment factory has no Walker branch. A fresh DMC Walker
construction/reset/step check passed (obs24, action6), but no Walker RND buffer, reference
checkpoint, or Walker flow/preimages were found in the inspected project/home storage.
The current venv also lacks Torch and h5py. Thus no matched training result exists from
this audit. A dataset path was requested; no long job was submitted. Exact GPU-hours should
be estimated from the required200-step smoke after the data and recipe are resolved.

## Interface directions worth testing after the gates

The first investigation should measure conditional TD amplification with phi fixed and
test policy values against known returns under the **same continuation and reward**. It
should also cross-score successful and failing policies under one frozen reward evaluator.
These address mechanisms that the existing success proxies and fitted-return ratio miss.

Three design choices remain plausible, with distinct research claims:

- A frozen codebook of state-dependent, reward-free stochastic skills makes the index a
  policy identity. Controlling each skill's latent density ratio can make conditional
  coverage explicit; temporal smoothness alone cannot supply that guarantee.
- A practical PSM proto/SF split or a true feasible affine occupancy construction can
  provide an action-value teacher to DSRL-style latent distillation. The present affine head
  alone does not supply either construction, and feasible coordinate optimization needs
  constraints before arbitrary affine coordinates can be trusted.
- An independently trained, fixed reward-free temporal basis with action-space value
  learning and latent distillation removes moving features as a confound. This is a new
  representation experiment; attaching DSRL to the current synthetic-w arm would repeat
  an already unsuccessful control with the additional mask issue described above.

None is a validated fix. The debugging workflow kept implementation checks, mathematical
guarantees, and causal experimental claims separate; the parallel review exposed where
earlier audits had conflated them.

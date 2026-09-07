# Interface-fork experiments: oracle-aim, ODE re-eval, paper-faithful PSMFlow

Date: 2026-08-31. Status: SPEC — not started.
Context: docs/HANDOFF.md (2026-08-29 entry and earlier), and the 08-31 audit which
established that (i) the shipped agent bootstraps at the actor's latent and indexes ψ by
the task vector, so Prop. "insample" (C=1) never applied to anything we trained, and
(ii) the whole experimental record forks on one unanswered question: can nothing *aim*
inside the flow's reachable action set, or is the reachable set itself the ceiling?

These three experiments answer that fork and, for the first time, run the method the
paper actually describes. **None of them uses the mixture preimage arm.** E1 uses no
preimages at all; E2 and E3 use the existing point-arm npz files only.

Paper source of record: the formal writeup is recoverable at
`git show 5249267:PAPER/main.tex` (untracked from the working tree). Key locations:
Prop. insample lines 807–835 (C=1 requires bootstrap `u' ~ p0` independent of `s'`),
§8 "PSMFlows" lines 903–916 (ψ(s, u, u′) with a *policy-index* latent, not the task
vector), GPI bound lines 981–1001. The shipped deviations are at
`agents/psmflow.py:139` (bootstrap = actor latent) and `psmflow.py:106,111`
(ψ(s, w, u) — task vector in the index slot).

Ground rules for every experiment below (non-negotiable, from repo policy):

- Before any launch: print the full hyperparameter table of the run; smoke-test the
  exact code path ~200 steps; after launch, re-read the run's own saved config
  (flags.json / .hydra) and confirm the intended values landed. State expected
  outcomes, including expected failures, before results arrive.
- Every long-running job in a named tmux session.
- Report success as mean ± 95% CI across seeds, 500 episodes, and always quote the
  controls next to it: BC per-step-prior control 0.068 [0.049, 0.093], PSMFlow point
  0.236 ± 0.071 (5 seeds), FB 0.721 ± 0.020 (3 seeds), FQL (task 2) 0.949 ± 0.063.
  Sources: docs/tables/results.md, docs/HANDOFF.md 08-14 entry. Note results.md has
  two known errata (FB-graft 0.064 should be 0.095 two-seed; sd0 is the least
  reproducible headline seed) — do not "fix" numbers to match the table.
- Commits: short subject + bullets, no AI attribution trailers, nothing pushed to
  remote without the operator.
- Environment for all runs: cube-single-play-singletask-v0 (task 2 = the headline
  task) unless stated. Frozen Stage-A flow checkpoint:
  `/var/local/amsks/exp/PSMFLows/bcflow_cube_single_20260726_135032/sd000_20260726_135037`
  @ epoch 500000, flow_steps=100 for exact decode.

---

## E1 — Oracle-aim rollout (run FIRST; decides everything downstream)

**Question.** At deployment, if an oracle picked the best latent, would the flow's
reachable action set suffice? Separates "cannot aim" (selection failure) from "cannot
reach" (interface ceiling).

**Design.** New tool `tools/diag_oracle_aim.py` (model it on the existing
`tools/diag_*.py` + `tools/viz_policy_rollouts.py` eval plumbing). Pure evaluation, no
training. Per environment step at state s:

1. Draw K=512 latents `u_k ~ N(0, I)`, clipped per-coordinate to ±3 — exactly matching
   the deployed GPI draw at `agents/psmflow.py:454`, so the candidate set is the one
   the real agent sees.
2. Decode all K through the frozen Stage-A flow with the **exact** decoder (100-step
   ODE, `compute_flow_actions`), not the one-step distilled net. This removes the
   known one-step decode error (0.0886 mean, HANDOFF 08-30) from the measurement.
3. Query the oracle action `a* = FQL(s)` from a frozen task-2 FQL expert checkpoint
   (the agent that scores 0.949; locate it under the experiment roots — check
   `scripts/eval500.sh` and run_logs for the restore_path the FQL eval used; if
   ambiguous, verify by re-running its 20-episode eval and matching ~0.95 before
   trusting it).
4. Execute `argmin_k ||a_k − a*||_2`. Log per step: the min distance, the median
   distance, and ||a*||.
5. 500 episodes, eval seed protocol identical to `scripts/eval500.sh` (P0.2 pinned
   randomness). Also run a 3-seed × 500-episode variant only if the first number lands
   in the ambiguous band (see below).

**Cost.** ~1 GPU-hour. tmux: `oracle-aim-cube`.

**Also log** (cheap, same rollouts): success if we execute the oracle's own a* directly
(sanity ceiling, should reproduce ~0.95), and success of a random-latent policy
(should reproduce BC ~0.068). Both must land before the oracle-aim number is read —
they validate the harness.

**Pre-registered reading.**
- oracle-aim ≥ 0.7: the reachable set is fine; the entire loss is selection. The
  project's work is a latent critic that can rank (E3 is the first candidate).
- oracle-aim ≤ 0.3: the exact-decode constraint is the ceiling; no critic can help.
  Work shifts to retraining Stage A (displacement-weighted CFM, latent redundancy) or
  relaxing the interface (the ε-residual arm).
- 0.3–0.7: partial ceiling; report the per-step min-distance distribution alongside so
  the gap can be attributed (distance-limited vs dynamics-limited), and run the 3-seed
  variant.

**Deliverable.** `run_logs/` JSON + a HANDOFF entry with the number, both controls,
and which fork branch it selects.

---

## E2 — ODE-decode re-evaluation of existing checkpoints (zero training)

**Question.** How much of the deployed performance is lost to the one-step distilled
decoder? The point preimage is exact under ODE-100 (decode error 0.00012) but off by
0.0886 under the one-step net the agent actually uses — ~10% of action scale.

**Design.** Evaluation-only re-run of the existing PSMFlow point-arm checkpoints
(the 5 seeds behind 0.236 ± 0.071; find their checkpoint dirs via the eval JSONs in
`/data-local/amsks/PSMFLows/logs/` and their flags.json). Two arms:

- `acting=actor` with decode through the 100-step ODE instead of the one-step net.
- `acting=gpi` likewise (current gpi baseline: 0.055 [0.047, 0.064], *below* BC).

First verify the config surface: the audit referenced `gpi_decode=ode` /
`flow_decode_steps` style keys — confirm what actually exists in
`agents/psmflow.py:433-461` and `configs/agent/psmflow.yaml:100`; if no such switch
exists, add one (eval-path only, no training-path changes) behind a config flag
defaulting to current behavior, and commit that separately before running.

500 episodes × 5 seeds per arm. tmux: `ode-reeval-cube`.

**Pre-registered reading.** If actor-arm success moves ≥ 0.05 above 0.236 ± 0.071, the
one-step decoder mismatch is real deployed loss (and E3 must use ODE decode). If it
moves nothing (the W1 result — ODE decode moved coverage +0.002 — suggests this), the
mismatch is priced as negligible at deployment and the point-vs-mixture "noise-free
control" caveat stays a bookkeeping note, not a lever.

**Deliverable.** Table row per arm: mean ± 95% CI vs the 0.236 baseline; HANDOFF entry.

---

## E3 — Paper-faithful PSMFlow (run the algorithm the paper proves things about)

**Question.** The C=1 construction has never been trained. Before abandoning the
paper's idea, run it once as written.

**Design.** Two arms, cheapest first, both on cube with the existing **point-arm**
preimage npz (dataset-side latents unchanged; the fix is the bootstrap and the index).

**Arm A — prior bootstrap only (config-level, no new code).**
`agents/psmflow.py:141-147` already implements backup exploration: it replaces a
fraction of bootstrap latents with prior draws. Set `backup_explore_frac=1.0`
(config `configs/agent/psmflow.yaml:83`, currently 0.0) — at 1.0 the bootstrap is
exactly `u' ~ p0`, which is Prop. insample's hypothesis. Note the 0.5 setting was run
once (abl_backup_explore05_20260810, in-loop ~0.20, never 500-ep evaluated) and was
null; 1.0 is the theoretically-motivated value, not an interpolation. Verify in the
code that frac=1.0 replaces ALL bootstrap latents and that the draws are clipped the
same way as deployment. 3 seeds × 500k steps, 500-ep eval. tmux: `paperfaith-armA`.

**Arm B — restore the policy-index latent (code change).**
Reintroduce ψ(s, u, u′) per §8 (`git show 5249267:PAPER/main.tex`, lines 903–916):
the ψ index slot carries a policy latent u′ sampled fresh from p0 per batch element
(shared across the row's backup), not the task vector w. w remains only in the task
readout Q = M·(task projection), i.e. w = E[r·φ(s′)] at eval exactly as now
(`psmflow.py:409-412`). Bootstrap continues the SAME u′ at s′ (that is what makes u′ a
policy index). Acting: GPI over Λ_K = {u′_k ~ p0, K=64} scoring the measure-derived Q,
per the GPI bound (lines 981–1001) — the existing gpi machinery at `psmflow.py:443-461`
is the template; the change is what ψ is indexed by. Keep the actor/BC-distill arm out
of Arm B entirely (the paper's §8 has no actor); if that makes early training unstable,
document it rather than silently re-adding BC. Implementation goes behind a config
flag (e.g. `policy_index=latent|task_vector`) with the current behavior as default;
separate commit; unit smoke: ψ input shapes, one gradient step, 200-step run.
3 seeds × 500k. tmux: `paperfaith-armB`.

**Controls to quote:** BC 0.068, shipped PSMFlow 0.236 ± 0.071, FB 0.721 ± 0.020.
Keep `pessimism_penalty` at its current value in both arms for comparability, and note
that hpmatch measured pessimism=0.0 as a non-lever.

**Pre-registered expectations.**
- Arm A: if E1 lands "cannot aim", Arm A alone probably does NOT rescue performance
  (a correct backup distribution does not create ranking signal by itself) — expect
  within noise of 0.236. It is still required as the component of Arm B that is
  testable independently.
- Arm B (with A's bootstrap): this is the paper's algorithm. If it also lands ~0.2,
  the paper's construction is refuted on its own terms on cube and the writeup must
  say so; if it clears FB's 0.721 or even the 0.45 bar, the 08-05 redesign was the
  mistake and the method is alive.
- Either way this produces the first number the paper's claims actually apply to.

**Ordering constraint.** E1 first (1 GPU-hour, changes the interpretation of E3);
E2 anytime (independent); E3 Arm A can launch in parallel with E1; Arm B only after
its smoke + a review of the ψ-signature diff.

---

## Explicitly out of scope

- The mixture preimage arm (its target/covariance bugs — un-squared norm at
  `agents/fql.py:201`, α² vs 2α proposal at `fql.py:88` — are logged in the 08-31
  audit; fix only if the mixture is ever to carry results again).
- Antmaze anything (blocked on the same fork; cube decides first).
- Re-auditing closed items: fixed-u coverage, task inference, HP bundles, backup
  exploration at 0.5, preimage width sweeps, budget.

---

# E4 addendum (2026-08-31 evening): known-good-ranker control + mixture-checkpoint probes

Added after the E3 readout. Context: E1 proved the flow's candidate actions contain
near-expert behavior (oracle-aim 0.934 vs expert 0.960, random floor 0.086). E3 Arm B
(paper-faithful C=1 construction) trained to 250k has a flat critic (Q spread 0.9% of
|Q|, ranking Spearman ~0.07) and GPI over it ANTI-selects (0.006 vs the in-path K=1
random control 0.08). Prior attempts to rank candidates with learned action-space
critics built inside our framework also failed (lambda-rank 0.004/0.162 < decode-only;
FB graft 0.095 vs 0.45 gate). All of those critics share the phi-basis/w-projection
value pathway (measured reward resolution R^2 ~0.13). What has never been tested is a
known-good ranker on our candidates.

## E4a — FQL-critic-as-scorer control (eval-only, ~1 GPU-hour, run FIRST)

Template: tools/diag_oracle_aim.py (E1 harness). Change exactly one thing: the scorer.

- Same K=512 clipped prior draws per step, same ODE-100 decode, same 500-episode
  protocol and pinned eval seeding as E1.
- Scorer: the frozen FQL expert's OWN critic Q(s, a_k) (the checkpoint E1 validated at
  0.960 when executing its actor directly — reuse the same restore_path from
  e1_oracle_aim_cube_sd0.json). Execute argmax_k Q(s, a_k). If the FQL agent has an
  ensemble, score with its standard eval reduction (read its config; do not invent one).
- Per-step logging: Spearman between the FQL-critic ranking and the oracle-distance
  ranking over the K candidates (gives a ranking-quality number independent of rollout),
  plus the distance-to-expert of the picked action.
- Controls to quote next to the result: oracle-aim 0.934 [0.909, 0.953], expert-direct
  0.960, random floor at MATCHING decode (ODE floor 0.014; onestep floor 0.086 — label
  which applies), BC 0.068.

Pre-registered reading:
- >= 0.7: candidates and decode-then-score are fine; every failure to date is OUR
  critics, and the shared phi/w value bottleneck becomes the prime suspect. Next work:
  a critic with a direct value pathway over decoded actions (e.g. psi(s, G(s,u), u'))
  — spec that only after this lands.
- <= 0.2: even a proven ranker cannot rank these candidates under per-step GPI; decode-
  then-score is dead as a deployment scheme, and the lambda-rank failure generalizes.
  Work shifts to actor-based improvement in latent space or interface relaxation.
- Middle: report the per-step Spearman distribution before interpreting.

## E4b — D1/D3 probes on the mixture-trained checkpoint (~20 min GPU)

Closes the "would the mixture have fixed ranking?" question with a measurement instead
of an inference. The mixture Stage-C control run exists (eval500_mixhpo_00{0,1,2},
success 0.242 vs point 0.236) but only success was recorded.

- Locate the mixhpo run's checkpoint via the restore path inside its eval JSONs.
- Run the same D1-analog ranking probe and D3 Q-landscape probe used on Arm B
  (committed 1ce15cd / a235372) against it.
- Quote baselines: point-arm shipped Spearman 0.10 (p=0.78), Arm B 0.072-0.079;
  Q relative spread shipped 1.1%, Arm B 0.9%.
- Pre-registered expectation: mixture jitter is prior-scale-irrelevant (width ~0.1-0.5
  per dim vs candidates spread over all of N(0,I)), so Spearman and spread land in the
  same band. If instead the mixture checkpoint shows materially better ranking, that is
  a real surprise and reopens the mixture arm as a training-signal lever.

Ordering: E4a and E4b are independent; run both immediately, E4a on the first free GPU.
Both are evaluation-only: no training, no npz regeneration, no edits to
agents/psmflow.py. Results go into the same tables/HANDOFF flow as E1-E3.

# Lifting the frozen-flow action-interface ceiling (2026-08-11)

## Context — what the overnight runs established

Cube, 500k, 500-ep or in-loop evals:
FQL 0.949 · FB 0.716–0.730 · **per-task latent RL through the frozen flow (P2) 0.20–0.24**
· LatentFlowPSM 0.236 · backup-exploration ablation ~0.22 (no change) · BC 0.068.

P2 is decisive: with the REAL task reward and a dedicated critic, the latent action
space still caps exactly where the zero-shot agent sits. The differentiation probe
(`logs/diff_{fb,psmflow}_cube.json`) killed the improvement-loop theory too (FB's
z-policies differentiate only 1.4x the noise floor and FB still wins). **Every method
acting through the frozen flow hits ~0.22; every method in raw action space clears 0.7.
The bottleneck is the latent→action interface, not the PSM measure, not the loop.**

Mechanism (already measured, 07-29 audit, `docs/HANDOFF.md`): sweeping u over the whole
typical set moves the action ~39% of the data conditional's spread (per-dim sd 0.105 vs
0.270; decoder Jacobian singular values ~0.1). The one-step distilled decoder is
mode-seeking and u is clamped to the typical set, so the decodable action set per state
is a narrow slice around the behavior mode. No critic can select actions that are not
in the reachable set.

**Strategy: P2 (per-task latent RL, `latentrl_*`) is the INSTRUMENT for every interface
fix** — it has direct reward signal, trains in ~1.5 h, and cleanly attributes any lift
to the interface. Zero-shot LatentFlowPSM reruns happen only after the instrument shows
lift. The PSM measure work stays parked (P1 arms of the 08-10 roadmap are moot until
this ceiling moves).

Ground rules: §0 of `2026-08-10-iclr-figures.md` (venv python, named tmux + kill-server,
one training seed/GPU, 500-ep Wilson / seed mean ± CI, BC control quoted, JSON reports
to `/data-local/amsks/PSMFLows/logs/`).

## W1 — coverage probe (no training, ~2 h GPU; DO FIRST — it selects the fix)

New tool `tools/diag_action_coverage.py`, cube first. At ~64 dataset states (seeded):

1. **Reachable-set spread:** decode 512 latents at each state; per-dim action sd vs the
   data conditional's sd (k-NN over the dataset, k≈32 nearest states, same protocol as
   the 07-29 measurement so numbers are comparable). Report coverage ratio.
2. **The killer stat:** min over decoded candidates of ‖G(s,u) − a_FQL(s)‖, where
   a_FQL is the action chosen by a trained FQL policy
   (`/data-local/amsks/PSMFLows/exp/PSMFLows/fqlbaseline_cube_a300_20260810/sd000_*`,
   epoch 500000) at the same state — i.e. can ANY latent decode to what the working
   policy does? Report the distribution, normalized by action scale.
3. Both stats swept over the knob grid:
   - decoder: `onestep` (deployed) vs `ode` — ODE steps MUST equal the flow's training
     value (cube ckpt was trained at flow_steps=10, so ode@10 here; add an assert that
     reads the flow's flags.json rather than trusting the config default — this fixes
     the known `gpi_decode=ode` step-mismatch defect for all envs at once);
   - latent radius: u drawn at ‖u‖ scales {typical, 1.5x, 2x} (i.e. u_clip 3 / 4.5 / 6).

Output `logs/diag_action_coverage_cube.json` with a `verdict` naming which knob(s)
recover coverage. Decision matrix:

- ODE decode recovers most of it ⇒ W2 arm A only.
- Radius recovers it ⇒ W2 arm B only.
- Both partially ⇒ both arms.
- **Nothing recovers it** (coverage stays ~0.4, FQL-action distance stays large at every
  setting) ⇒ the flow itself collapsed the conditional ⇒ skip to W3 (retrain), and W4
  becomes likely.

## W2 — interface fixes on the instrument (each arm = latentrl, 2 seeds × 500k, ~2 h/seed)

| arm | change | tmux |
|---|---|---|
| A: ODE decode | acting decodes via the flow ODE at the flow's trained step count (not the distilled one-step); training target side unchanged | `latrl_ode_sd{0,1}` |
| B: wider latent box | `u_clip` per the probe's winning radius, TRAINED at that clip (an eval-only clip change rescales the trained actor's tanh semantics — do not mix) | `latrl_uclip<r>_sd{0,1}` |
| A+B | both, only if the probe says both contribute | `latrl_ode_uclip_sd{0,1}` |

Gate for W2: **latentrl 500-ep ≥ 0.40** (≈ halfway from the 0.22 ceiling to FB's 0.72).
Pass ⇒ port the winning interface to LatentFlowPSM (same acting-path change; the ψ
backup and preimage inputs are untouched by A; B additionally needs `u_clip` consistent
in the backup clamp) and rerun cube 3 seeds + 500-ep evals + the diagnostic trio.
Marginal (0.28–0.40) ⇒ stack W3. Fail (<0.28) ⇒ W3 directly.

### W1 RESULT (2026-08-11 02:50, `diag_action_coverage_cube.json`) — W2 skipped per the matrix

Coverage is pinned at **0.42** and normalized FQL-action distance at **0.19** across the
ENTIRE knob grid (ODE +0.002, radius +0.001). Neither the decoder nor the latent radius
is the constraint — the flow's conditional itself is collapsed. W2 is dead; W3 launched
(`bcflow_cube_fs100`, in flight). Two implications recorded for W3:

- **The ODE row is a warning, not just a null result.** ODE decode bypasses the
  distilled one-step net and integrates the raw vector field — and coverage did NOT
  move. So the CFM vector field itself is narrow on this checkpoint, which lowers
  confidence that step-count alone (fs10 → fs100) fixes it. Treat the fs100 retrain as
  one arm, not the answer.
- **Metric sanity check to add before trusting any retrain readout:** the k-NN
  conditional mixes genuinely distinct states into the denominator, so 1.0 is not the
  achievable ceiling. Compute the same coverage stat for (a) a held-out half of the
  k-NN action set against the other half (the aleatoric ceiling), and (b) k=8 vs k=32.
  Report the retrained flow's coverage AGAINST that ceiling, not against 1.0.

## W3 — retrain the cube Stage-A flow (PROMOTED by W1; 1 GPU-night + ~10 h invert)

The deployed cube flow is the known-caveated checkpoint trained at `flow_steps=10`
(HANDOFF 07-29 caveat; its local fidelity at 100 was measurably worse: off-support 0.47
vs 0.35). Retrain deliberately for conditional coverage:

1. Stage A cube at `flow_steps=100` (safe under `utils/xla_guard.py`), same 500k recipe
   (`scripts/pretrain_behavior_flow.sh`); optionally a second arm with wider nets only
   if the probe blames capacity.
2. **New D1 panel first-class:** add a conditional-coverage metric to
   `tools/validate_flow_fidelity.py` (the W1 coverage ratio, computed per env) — D1's
   marginal MMD passed while the conditional was collapsed, so marginal fit can no
   longer be the sole Stage-A gate. Gate the retrained flow on it before inverting.
3. **Pre-inversion gate (hard, the inversion costs 10 h):** re-run the W1 probe on the
   retrained flow. Proceed to invert ONLY if coverage clears **≥ 0.7 of the measured
   aleatoric ceiling** (see W1-result note) AND normalized FQL-action distance drops
   below **0.10** (from 0.19). If fs100 does NOT clear it: do NOT iterate Stage-A
   blindly — launch exactly two things in parallel: (a) ONE wider/longer flow arm
   (capacity hypothesis), and (b) **W4 promoted to primary** (the residual head does
   not need a better flow, and its ε-sweep quantifies exactly the coverage the flow is
   missing). This branch is pre-committed here.
4. Re-invert: `tools/precompute_preimages.py` alpha=20 N=200 (~10 h) → new npz + sidecar.
5. Re-run the instrument (latentrl, 2 seeds) on the new flow (decode per the new flow's
   trained step count).
6. Only on success: re-run LatentFlowPSM, and re-upload flow+preimages to
   `amsks/psmflows-preimages` via `scripts/upload_preimages_hf.py` (same repo, new
   card numbers; note the old cube artifacts are superseded in the card).

### W3 RESULT (08-11 14:48, `diag_action_coverage_cube_fs100.json`) — gate FAILED; W4 promoted

fs100 retrain: coverage 0.42 → **0.63**, but normalized FQL-action distance **unmoved at
0.187** (gate needs <0.10) across every decoder/radius setting. Do NOT invert. Tool debts:
the fs100 JSON carries a stale copy of the fs10 verdict string, and the aleatoric-ceiling
control from the W1-result note was never computed — fix both.

**Three checks BEFORE any further training (all cheap, extend `diag_action_coverage.py`):**

- **C1 — is FQL off-support? (the discriminator).** Distance of a_FQL(s) to the dataset's
  own k-NN actions at the same states. ~0.19 too ⇒ the task needs off-support actions; no
  behavior flow can ever decode them; the flow-retrain path is DEAD and W4 is the only
  fix. Small ⇒ the flow still underfits the tail; the capacity arm stays live.
- **C2 — per-dimension breakdown of the residual.** Cube's gripper dim is quasi-discrete
  and BC flows smooth it. Residual concentrated there ⇒ surgical fix (discrete gripper
  head / per-dim ε) instead of a blanket residual.
- **C3 — the missing aleatoric ceiling** (held-out half of k-NN set vs other half; k=8 vs
  32). If ceiling ≈0.7, fs100's 0.63 says coverage is nearly done improving and only the
  off-support gap remains.

**Launch W4 NOW on the instrument** (pre-committed by the gate failure): ε ∈ {0.05, 0.1,
0.2} — brackets the 0.19 gap (half / ≈full). Shape ε per C2's verdict. The capacity arm
is DEPRIORITIZED until C1 reads out; if C1 says off-support, cancel it. If C1 confirms,
the paper headline becomes the support-vs-performance curve: "task-optimal actions sit
measurably off the behavior support; support-constrained methods cap at a predictable
ceiling; a bounded residual budget buys it back" — every number already instrumented.

### C1–C3 RESULT (08-12, `diag_action_coverage_cube_c123.json`) — capacity arm CANCELLED

**C1, the discriminator, reframes the whole interface story.** Distance of a trained FQL
policy's action to the dataset's OWN 32-NN actions at the same 64 states, normalized by
action scale:

| | normalized distance |
|---|---:|
| a_FQL → dataset's own 32-NN actions | **0.577** |
| a_FQL → our best decode (512 candidates) | 0.187 |

FQL sits **three times further off the behaviour support than our decoder misses it by**.
The flow interpolates CLOSER to the winning actions than the empirical data itself does.
So:

- **0.187 was never a decoder failure.** It is the residual of a decoder that already
  outperforms the data's own neighbourhood at approximating the task-optimal action.
- **No retrain could have closed it.** The actions are not in the distribution being
  cloned. Capacity/retrain arm **CANCELLED**; **W3 retired** (its coverage gain was real
  — 0.42 → 0.63 — and irrelevant to the ceiling).
- The pre-inversion gate failing at 0.187 vs the <0.10 bar was the right call for the
  wrong-seeming reason: the bar was unreachable by construction, not by underfitting.

**C2:** the min-dist residual spreads evenly over all five action dims (max share 23%,
dim 3). No gripper concentration ⇒ **blanket ε**, not a per-dim or gripper-targeted one.

**C3:** the aleatoric coverage ceiling — half of each 32-NN action set scored against the
other half — is **0.936**, not 1.0. Report coverage against that: fs100's 0.632 is **67%
of achievable** and the deployed fs10 flow's 0.416 was 44%. The against-1.0 framing
overstated how much coverage was ever on the table.

**Consequence for §6's open question.** It is answered, and should say so: task-optimal
actions on this play data are measurably off-support — three times farther from the data
than the decoder's own miss. Support-constrained methods therefore cap at a predictable
ceiling that no representation, loop, or decoder fix can lift; only a support budget can.

## W4 — bounded residual head (PROMOTED 08-11: fallback AND the paper's tradeoff dial)

If W2/W3 leave a large gap to FB: action = decode(u) + ε·tanh(δ(s,u)), ε ∈ {0.05, 0.1,
0.2} of the action scale, δ a small trained head. Implement in the instrument first
(latentrl, 2 seeds per ε). ε grid extended 08-12 to **{0, 0.05, 0.1, 0.2, 0.4}**: C1's
0.187 gap localizes the predicted onset between 0.1 and 0.2, and 0.4 is needed to see
saturation toward FQL, which is half the curve's argument. ε=0 is a genuine arm, not a
reuse of the P2 numbers — the W4 critic scores executed ACTIONS (the residual head gets no
gradient from a latent critic), so the left endpoint must be re-measured under the new
critic. This relaxes the binary support constraint into a measured budget — report, per ε: success AND an off-support metric (distance of executed actions
to dataset actions at k-NN states, the D1 off-support fraction). Deliverable either way:
the **support-vs-performance tradeoff curve** (x = support budget from 0=pure decode to
∞=FQL, y = success) — this reframes "the constraint costs 3x" into the paper's central
quantitative figure, with FB/FQL as the unconstrained endpoints.

## Paper hooks (do alongside, not after)

- New results figure (F5): the ceiling result — bar/point plot of the table at the top,
  action-space methods vs through-flow methods, with the per-task latent RL bar
  carrying the argument that the interface, not the zero-shot machinery, binds. All
  numbers exist today; build it now, extend with W2–W4 arms as they land.
- If W4 runs: the tradeoff curve becomes F6 and likely the headline.
- HANDOFF entry for the 08-10→11 verdict chain (P0 Branch B → differentiation probe →
  P2 ceiling → this plan); the decisions doc gets one entry: "the support constraint as
  implemented costs 0.72→0.22 on cube; fixing the interface, not the measure".

## Leftover mechanical (fill idle GPUs)

PSM cube sd1/sd2 500-ep evals when training completes (sd0 done: check
`eval500_psm_cube_sd0.json`); refresh `fig_actor_comparison` with FB 500-ep numbers
(0.716/0.730/0.716) + PSM bars + 3-seed antmaze row (0.22/0.26/0.22 vs control 0.090).

## Schedule

```
Day 1: W1 probe (2 h) → pick arms · launch W2 arms (2–4 GPUs) · F5 figure · leftovers
Day 2: read W2 vs 0.40 gate → port to LatentFlowPSM OR launch W3 retrain
Day 3: W3 invert + instrument rerun · W4 decision · paper hooks
```

## Acceptance checklist

- [ ] `diag_action_coverage_cube.json` with per-knob coverage + FQL-distance stats and
      a verdict; ODE-steps assert reads the flow's flags.json (defect closed)
- [ ] W2 arms trained + 500-ep evals; gate (≥0.40) applied before any zero-shot rerun
- [ ] If W3: retrained flow passes the NEW conditional-coverage D1 panel before
      inversion; npz sidecar pairing intact; HF re-upload only after instrument success
- [ ] If W4: tradeoff curve with support-budget metric per ε
- [ ] F5 built from existing numbers; HANDOFF + decisions entries; tmux server killed

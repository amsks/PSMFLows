# Boosting PSMFlows: the four levers (2026-08-12)

Frame (from the collaborator review of `PAPER/ICLR/note.tex`): the action bottleneck has
two nested layers — decoder ⊂ local empirical behavior ⊂ task-useful actions. C1 measured
FQL's actions at 0.577 (normalized) from the dataset's 32-NN actions vs 0.187 from our
best decode. Each inclusion is a separate performance lever. This plan supersedes the
W4-only path in `2026-08-11-action-interface-fix.md`; the Q(s,a)-critic design and ε=0
anchor arm approved there carry over unchanged.

Ground rules: §0 of `2026-08-10-iclr-figures.md` (venv python, named tmux + kill-server,
one training seed/GPU, 500-ep Wilson / seed mean ± 95% CI, BC control 0.068 quoted, JSON
reports to `/data-local/amsks/PSMFLows/logs/`). Instrument = per-task latentrl unless
stated. Every figure-bound number needs a matched-statistic definition in its JSON.

## L0 — geometry robustness (BLOCKING, ~hours, no training; do before anything else)

The support argument rests on k-NN geometry in RAW observation space. Extend
`tools/diag_action_coverage.py`:

1. **Matched statistics** (reviewer pt 1): report d_flow(s) = min_j ||a_FQL(s) − G(s,u_j)||
   over 512 decodes AND d_data(s) = min_i ||a_FQL(s) − a_i|| over the k-NN actions, same
   norm, same normalization, side by side. Also report both at matched candidate counts
   (subsample decodes to k). State the conditioning asymmetry in the JSON notes field.
2. **Neighborhood robustness** (pt 6): recompute coverage, d_data, and the split-half
   self-consistency reference with (a) per-dim standardized observations, (b) k ∈ {16,
   32, 64}. Output `logs/diag_action_coverage_cube_robust.json` with per-setting values.
3. **Verdict field**: does the 0.577-vs-0.187 ordering survive normalization and matched
   statistics? YES ⇒ the two-layer framing stands; propagate the numbers to note.tex
   (Finding 3 + abstract clause) and HANDOFF. NO ⇒ update note.tex to the surviving
   numbers IMMEDIATELY (it currently flags the result as provisional) and tell the user
   before launching L2/L3 — the lever ranking depends on this.

## L1 — residual sweep as a regime discriminator (approved design, launch after L0)

Implementation as approved: critic moves to Q(s,a) trained by TD on dataset
(s, a_data, s'); actor emits u; executed a = clip(G(s,u) + ε·tanh(δ(s,u))); BC anchor
retained. ε=0 anchor arm REQUIRED (not bit-identical to old P2 — never mix the curves).

- Arms: ε ∈ {0, 0.05, 0.1, 0.2, 0.4} × mean |a| (blanket ε per C2's even spread),
  2 seeds × 500k each, tmux `latrl_res<eps>_sd{0,1}`, 500-ep evals.
- **Per-arm regime logging (reviewer pt 4):** during eval, record per-step distance of
  the EXECUTED action to the local behavior data (k-NN actions of the current state,
  L0's normalization). Per arm report: success, mean/median executed-action distance,
  fraction of steps beyond the local action cloud (beyond the split-half reference
  distance).
- **The headline plot is success vs measured behavior-distance**, ε only labels points.
  Interpretations (pre-registered): rises while actions stay behavior-close ⇒ decoder
  undercoverage was the cost (L2 matters most); rises only once actions leave the cloud
  ⇒ behavior restriction itself binds (relaxation is the only fix); never rises ⇒ the
  explanation is elsewhere — STOP and report; declines at large ε ⇒ support
  regularization has measurable value and the curve's peak is the paper number.
- Gate to port into LatentFlowPSM (3 seeds): best arm ≥ 0.40 at 500 ep.

## L2 — reweighted behavior flow (parallel with L1; 1 GPU-night + probe gate)

Play data is dominated by rests, so the BC flow spends its capacity on passive actions —
a direct cause of layer-1 undercoverage. Retrain Stage A cube with per-transition weights
increasing in OBJECT DISPLACEMENT ||x_obj(s') − x_obj(s)|| (task-agnostic, reward-free):

1. Add a `sample_weight` path to the FQL bc_only losses (weighted CFM + distill); weight
   w_i = 1 + β·(disp_i / mean_disp), β=4 first arm (≈5x emphasis on moving transitions);
   log the effective sample size of the weighting.
2. fs100 recipe otherwise unchanged; tmux `bcflow_cube_wdisp_sd0`.
3. **Gate before any inversion or RL:** the L0-normalized coverage probe on the new flow.
   Success = coverage materially above fs100's 67%-of-reference AND d_flow to FQL actions
   reduced. Fail = do not iterate β blindly; report and stop this lever.
4. On success: run the instrument (2 seeds) on the new flow; only then consider the 10 h
   inversion for the zero-shot stack.

## L3 — budget-constrained FB (parallel; the hedge and possibly the headline)

Apply the dial to the method that already scores 0.72: FB's flowBC actor gets a
behavior-closeness constraint instead of our stack getting a relaxation. Minimal change
to `agents/fb.py`: penalty λ·max(0, d(a_actor, nearest-of-K decodes G(s,u_k)) − ε)
added to the actor loss (K=32 decodes from the frozen cube flow; reuse the coverage-probe
decode machinery), or hard projection if the penalty is unstable.

- Arms: ε ∈ {0.05, 0.15, ∞(=plain FB control)} × mean |a|, 2 seeds × 500k, tmux
  `fb_budget<eps>_sd{0,1}`. Same regime logging as L1.
- Reading: small-ε FB retains ≥0.6 ⇒ "FB-level zero-shot with a measured
  behavior-closeness budget" — likely the strongest available headline; small-ε FB
  collapses toward 0.22 ⇒ independent confirmation that behavior-closeness itself is
  what binds, which strengthens L1's interpretation.

## L4 — acting-time refinement (config-level, stack onto L1's best arm only)

Decode K=16 candidates, pick the Q(s,a)-argmax, apply δ as a local correction from that
candidate. One extra arm, 2 seeds, only if L1 shows a rising curve.

## Explicitly NOT funded

Stage-A capacity/step sweeps beyond L2 (C1 killed them), the PSM measure-objective swap
(parked), pointmaze anything, LatentFlowPSM reruns before an L1/L2/L3 gate passes.

## Bookkeeping

- HANDOFF entry (2026-08-12): C1/C2/C3 verdicts, the two-layer reframing, this plan.
- note.tex: L0 verdict propagates immediately (it currently marks the geometry as
  provisional); L1's curve slots into the forward-experiment section when it lands.
- Peer bar: finish `eval500_psm_cube_sd2` (sd0 0.156, sd1 0.056 — the multi-seed PSM
  number goes into the note's table with all three seeds, however it lands).
- Figures: F5 scoreboard (action-space vs decoder-restricted methods) once PSM sd2 is in;
  the L1 curve is F6.

## Schedule

```
Day 1: L0 (blocking) → verdict to user · L1 implementation + ε=0 anchor launch ·
       PSM sd2 eval
Day 2: L1 arms (4–5 GPUs) · L2 retrain + probe gate · L3 arms (2–3 GPUs) as GPUs free
Day 3: L1 regime plot + gate call · L2/L3 readouts · port winner · HANDOFF + figures
```

## Acceptance checklist

- [ ] L0 robustness JSON with matched statistics + verdict; note.tex updated either way
- [ ] L1: 5 arms × 2 seeds, ε=0 anchor included, regime logging per arm, success-vs-
      distance plot, pre-registered interpretation stated in the report
- [ ] L2: weighted flow trained, probe-gated BEFORE inversion; no blind β iteration
- [ ] L3: 3 arms × 2 seeds with plain-FB control, same regime logging
- [ ] Nothing ported to LatentFlowPSM without a passed gate; tmux server killed

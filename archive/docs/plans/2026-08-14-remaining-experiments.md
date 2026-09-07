# Remaining experiments (2026-08-14, evening)

Context updates since the priority-stack plan: the scaled-radius redo landed and REFUTES
the radial-escape hypothesis (coverage 0.63→0.99→1.41 with radius, but min-dist to
FQL's action WORSENS 0.187→0.234→0.279 and the ODE decode NaNs at r=6 — the decoder
can be widened but not aimed); the u_clip eval sweep is REJECTED (confounded by actor
tanh rescaling + premise refuted). Hybrid status is mixed: full-data seeds
{0.048, 0.284–0.302, 0.134-with-residual vs 0.26-decode-only, 0.18} — the residual is
NOT reliably positive at full data — while at 10% data the hybrid reads 0.238 vs 0.072
decode-only vs old-PSMFlow 0.067: the residual clearly works in the low-data regime.
fbgraft sd0/sd1 are training (gate 0.45). FB-frac10 is queued in tmux `fb_frac10`
(auto-starts when GPU 1 frees; train + eval500 chained).

Ground rules: unchanged from `2026-08-14-priority-stack.md` (venv python, tmux +
tracked watchers, 500-ep Wilson only, document launches in an addendum here, commit
nothing).

## E1 — acting-mode × seed table (the gate for ANY hybrid claim; mostly eval)

Implement the missing **λ-rank acting variant** (the original B-A, never built): sample
K=32 latents from the actor's noise, decode all, score each candidate a_k by
Q_a = ψ_a(s,w,a_k)ᵀw, execute argmax — NO residual. ~30 lines in
`agents/psmflow.py::sample_actions` behind `acting: rank`; test: with K=1 it equals
decode-only.

Then assemble ONE table, 500-ep cells: rows = hybrid seeds (sd0, sd1@500k, sd1ext@1M,
sd2@1M, sd3@1M, sd4 if trained) × columns = {residual, decode-only, rank}. Many cells
exist (`eval500_hybrid_*`); fill only the missing ones. Output
`logs/table_hybrid_acting_modes.json` + a verdict: does ANY mode beat old PSMFlow
0.236 [0.219, 0.253] across seeds, and is residual > decode-only on average?
**No hybrid sentence enters the paper until this table exists.**

## E2 — residual asymmetry probe (why does the same mechanism help at 10% and not 100%?)

Cheap diagnostics on existing checkpoints, no training:
1. Per-step residual magnitude ‖a_exec − G(s,u)‖ and Q_a spread over candidates during
   eval rollouts: hybrid_frac10_sd0 (residual helps) vs hybrid_1M_sd2 (residual hurts)
   vs hybrid sd1ext. Is δ bigger/noisier at full data?
2. Q_a calibration (predicted vs realized, the fixed tool) on the same three — is the
   full-data Q_a *misdirecting* the residual (negative Spearman) while the frac10 Q_a
   points right?
Output `logs/diag_residual_asymmetry.json` with a verdict naming the mechanism.
Pre-registered candidates: (a) full-data Q_a flatter/misaligned (w disease dominates
when BC is already good); (b) δ overfit at 1M; (c) frac10's weaker decoder leaves
more recoverable headroom. This probe decides whether the residual needs w fixed
(fbgraft), scheduling (anneal ε), or gating (use residual only when Q_a spread is
informative).

## E3 — fbgraft readout (riding; the audit's main hypothesis test)

When sd0/sd1 finish 1M: 500-ep evals in all three acting modes (fold into E1's table
as extra rows). Gate unchanged: ≥0.45 any mode/seed ⇒ scale to 5 seeds, flagship.
~0.3 ⇒ shared-φ theory wrong — report with E2's verdict, do not iterate.

## E4 — complete the fraction table's FB and hybrid rows

1. FB-frac10: queued (tmux `fb_frac10`), auto-completes. 
2. **FB-frac50**, 1 seed, same recipe (`dataset_fraction=0.5 seed=0`), launch when a
   GPU frees: tmux `fb_frac50`. 
3. hybrid_frac50 (running) → eval500 both modes.
4. Refresh `table_dataset_fraction.json`: rows {FQL, FB, hybrid(residual),
   hybrid(decode-only), PSMFlow, BC, instrument} × {10%, 50%, 100%}; null cells marked.
   This table is now arguably the paper's second figure: if FB also degrades hard at
   10% while the hybrid holds 0.238, "support-anchored zero-shot is the data-efficient
   regime" becomes a headline claim; if FB holds up at 10%, the hybrid's low-data edge
   evaporates — either way we need the cells before writing anything.

## E5 — tool + paper debts (carried from P0, verify each is DONE or do it)

1. **Stale-verdict bug, third recurrence**: verdict strings must be computed from the
   current run's rows, never carried from a previous JSON. Fix in
   `diag_action_coverage.py` (and grep the other diag tools for the same pattern).
2. note.tex: replace the radius-invariance sentence with the corrected finding (the
   full scaled-radius table incl. the NaN row — "widened but not aimed" is a STRONGER
   support for the thesis than the vacuous version); verify P0.2 (seeded evals) and
   P0.3 (geometry harmonization) actually landed, complete if not.
3. P0.4 Q-gap probe / P0.5 calibration fix / P0.6 sidecar guard / P0.7 pessimism
   landmine / P0.8 latentrl tests: verify each against the priority-stack checklist;
   complete stragglers.
4. HANDOFF entry (2026-08-14 evening): scaled-radius refutation, hybrid mode-table
   status, the hybrid@10% result, fbgraft launch record, FB fraction arms.

## Explicitly rejected / not funded

The u_clip eval sweep (confounded + refuted); any flow retrain/rescale arm (the
scaled-radius data kills it); LIBERO (separate decision); new hybrid full-data seeds
beyond the existing set until E1+E2 explain the mode asymmetry.

## Order of operations

```
Now:     E1 λ-rank implementation + missing cells (evals share GPUs) · E5.1 verdict fix
Next:    E2 probes (CPU/eval) · E4.2 FB-frac50 when GPU frees · E5.2-4 paper/docs
On land: E3 fbgraft readout → gate call → E1 table final → E4 table final
Report:  one message — E1 table, E2 verdict, E3 gate, E4 table, E5 checklist
```

## Acceptance

- [ ] λ-rank implemented + K=1 test; `table_hybrid_acting_modes.json` complete
- [ ] `diag_residual_asymmetry.json` with a named mechanism
- [ ] fbgraft gate call recorded with all-mode evals
- [ ] fraction table with FB + hybrid rows; nulls explicit
- [ ] stale-verdict class fixed repo-wide; note.tex radius correction in;
      P0 checklist verified item-by-item
- [ ] HANDOFF current; launches documented; tmux server killed when drained

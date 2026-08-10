# Why is LatentFlowPSM flat? — diagnosis plan + peer baselines (2026-08-10)

Context: LatentFlowPSM beats its BC control on cube (0.236 ± 0.071 over 5 seeds vs
0.068) and antmaze (0.222 vs 0.090, 1 seed), but (a) training curves are flat after
50k, (b) the critic does not rank outcomes better than chance (calibration AUC 0.59
cube / 0.35 antmaze, `logs/fig_policy_anatomy.json` panel c), (c) the actor is
under-dispersed in latent space (mean ‖u‖² 3.9 vs dataset 5.6 on cube). Working
suspicion: the gain comes from the BC anchor + pessimism keeping actions in-support,
not from value-driven improvement. This plan (A) enumerates hypotheses with their
predicted signatures, (B) specs cheap no-training diagnostics that discriminate them,
(C) specs conditional ablation runs, (D) specs the peer zero-shot baselines and the
missing antmaze seeds regardless of diagnosis outcome.

Ground rules identical to `2026-08-10-iclr-figures.md` §0 (venv python, tmux named
sessions + kill-server, one training seed/GPU, 500-ep Wilson or seed mean ± 95% CI,
BC control always quoted, JSON reports for everything).

## A. Hypotheses (ranked by prior × cheapness to test)

**H0 — the calibration diagnostic is confounded (test FIRST, costs nothing new).**
Panel (c) correlates predicted value with realized success *across episodes of ONE
policy under ONE w*. The only variation is the start state; if success is mostly
decided by factors the features can't resolve at t=0, even a good critic reads
AUC≈0.5. The critic's actual job is ranking *policies* (that's all the actor's −Q
gradient and GPI use). Signature if H0: critic ranks distinct policies fine (D1 below)
while per-episode AUC stays ~0.5. Consequence: upgrade the paper claim, keep the
architecture, and the "struggling" is partly measurement.

**H1 — task projection failure: w doesn't represent the reward.**
Eval w = E[r·φ(s')] (`agents/psmflow.py:infer_z`). φ is trained reward-free; nothing
guarantees the task reward lies near span(φ), and with ~2% nonzero rewards the
estimate is noisy on top. If ψ is a fine measure but w points at the wrong function,
ψᵀw is uninformative and the actor's −Q term is noise — exactly the flat-curve +
chance-AUC picture. Signatures: poor linear fit of φ(s)ᵀw_inf to the true reward
(R² low); oracle-w eval ≫ inferred-w eval; w_inf far outside the training w
distribution (norm/cosine stats vs sample_mixed_z draws). This is the FB-family's
classic failure mode and my top structural suspect.

**H2 — Q is flat over u, so the BC terms own the actor.**
Rung-1 measured Q spread ~10% of mean over the whole latent box; if that carries
over, the actor loss −Q/|Q| + 1.0·distill + bc_flow is dominated by its BC terms and
the actor converges to latent BC — flat curves, gain ≈ "denoised imitation".
Signatures: Q spread over u at fixed (s,w) small relative to |Q|; per-term actor
gradient norms show bc ≫ q; the actor-rank probe (D3) does not beat the plain actor.

**H3 — actor–critic co-collapse via the bootstrap.**
The measure backup bootstraps ψ(s', w, u_actor(s',w)) — it only ever evaluates the
actor's own latent. A mode-seeking actor (tanh + distillation, measured ‖u‖² 3.9<5.6)
means ψ is trained on a shrinking slice of latent space, which flattens Q off-mode,
which removes the actor's incentive to move: self-confirming equilibrium. Signature:
Q spread small *specifically off the actor's mode* while on-mode TD error is low;
under-dispersion worsening over checkpoints (sd2–4 have the 50k grid to check).
Fix direction if confirmed: mix exploration latents into the backup (e.g. bootstrap
with u ~ p0 some fraction of the time), or noise the actor's bootstrap latent.

**H4 — pessimism + bc_coeff strangle improvement.**
pessimism 0.5 with num_parallel=2 is EXACT min-Q in both backup and actor; bc_coeff
1.0 (reference cube value is 3.0 — note we run *lower*). Together they may pin the
actor at the behavior mode even where Q has signal. Signature: none from diagnostics
alone — this is what the C-tier sweep is for; only run it if D2 shows Q *has* spread
that the actor isn't exploiting.

**H5 — horizon myopia on navigation (antmaze).**
γ=0.98 ⇒ ~50-step effective horizon; antmaze successes take hundreds of steps. FB
reference uses 0.99. Would explain antmaze's below-chance calibration specifically.
Signature: cube vs antmaze asymmetry in D1/D2; cheap single ablation γ=0.99 antmaze.

**H6 — decode-path error.** Deployed action = one-step distilled decode; cube flow
trained at flow_steps=10, theorems assume the ODE map. Low prior (BC control uses the
same decode and works), only revisit if everything else clears.

## B. Diagnostics — no training, one GPU-day total. New tools, each writing
`report_out` JSON to `/data-local/amsks/PSMFLows/logs/`.

**D1 `tools/diag_policy_ranking.py` — can ψᵀw rank POLICIES? (kills or confirms H0)**
Assemble ~8–10 policies of genuinely different quality on cube, all evaluable today:
BC control (0.068), latent-GPI mode (0.055), actor sd0–sd4 finals (0.176–0.318), and
actor sd2 at 50k/250k/500k (checkpoints exist). For each: predicted value = mean
ψ(s₀, w_inf, u_actor)ᵀw_inf over a fixed seeded set of ~64 start states, evaluated by
the SAME frozen sd0 representation (one critic judges all policies — that is the GPI
use-case); realized value = the already-measured 500-ep success where it exists (reuse
`eval500_*.json`; run ~100-ep evals for the intermediate ckpts). Report Spearman of
predicted vs realized across policies, with a permutation p-value (n≈10 is small —
say so in the JSON). **Decision: Spearman ≥ ~0.6 ⇒ H0 (diagnostic confound), critic
is usable ⇒ paper claim upgrades and C-tier focuses on exploiting it. Spearman ~0 ⇒
critic genuinely uninformative ⇒ H1/H2/H3.**

**D2 `tools/diag_task_projection.py` — is w the problem? (H1)**
On cube + antmaze, from the npz + trained ckpt: (i) fit r ≈ φ(s')ᵀw by the agent's own
closed form on N relabeled samples, report R² on held-out transitions, vs a ridge
regression topline on the same φ (how much reward IS in span(φ)); (ii) w_inf vs
training-w distribution: norm, cosine to nearest sample_mixed_z draws (both mixture
components); (iii) sensitivity: w_inf across 5 disjoint relabel batches — cosine
spread (noise floor of the inference). Then ONE eval: 200-ep cube eval with an
oracle-style w (φ at goal-reaching next_obs, mirroring D4's oracle construction)
vs the standard inferred w, same checkpoint. **Decision: oracle-w ≫ inferred-w or
R² ≪ ridge-topline ⇒ H1 — the fix lives in task inference (bigger relabel batch,
reward-weighted refinement, or a φ-grounding auxiliary), not in the actor.**

**D3 `tools/diag_q_landscape.py` + one config knob — does Q have usable relief? (H2/H3)**
(i) At ~64 dataset states, w_inf fixed: Q(s,w,u) over 512 prior draws + the actor's
own u ± perturbations; report spread/|Q|, actor's percentile in the Q distribution,
and the same restricted to a ball around the actor's mode (H3's off-mode flatness).
(ii) Per-term actor gradient norms (q_loss vs distill vs bc_flow) on 100 batches of
the frozen ckpt — one number: is the q-term even competitive at bc_coeff=1.0?
(iii) **actor-rank probe**: add `acting: actor_rank` to `agents/psmflow.py` — sample
K=32 actor noises, decode ψᵀw for each, execute the argmax (IDQL-style, ~20 lines,
reuses gpi_select machinery). 500-ep cube eval, sd0. **Decision: actor_rank > actor
⇒ the critic has signal the mean-actor wastes (supports H0/H4, immediate paper-able
win). actor_rank ≈ actor while (i) shows flat Q ⇒ H2/H3 confirmed.**

**D4 dispersion-over-training (H3, free):** extend `diag_q_landscape` to run its
‖u_actor‖² stat on sd2's 10 checkpoints — does dispersion shrink with training?

## C. Ablation runs — CONDITIONAL, launch only what B implicates. Cube, 2 seeds ×
500k each, `/data-local/.../exp`, tmux `abl_<name>_sd{0,1}`, in-loop 50-ep evals +
500-ep final eval. Priority order within tier:

| ablation | override | runs iff |
|---|---|---|
| bc_coeff ↓ | `agent.actor.bc_coeff=0.3` | D3 shows bc-term dominance AND Q has relief |
| bc_coeff ↑ (reference value) | `agent.actor.bc_coeff=3.0` | control arm for the above |
| pessimism ↓ | `agent.actor_pessimism_penalty=0.25` (keep backup at 0.5) | D3 actor percentile low |
| backup exploration | code change: fraction ε of bootstrap latents drawn from p0 instead of the actor | D3 confirms off-mode flatness (H3) |
| γ=0.99 antmaze | `agent.discount=0.99` | D1/D2 show cube-vs-antmaze asymmetry (H5) |

Cap: max 3 ablations × 2 seeds before regrouping — do not sweep blindly.

## D. Unconditional runs (the paper needs these whatever B says)

**R1 — antmaze seeds 1,2.** Same pinned recipe as sd0
(`psmflow_latentpsm_antmaze`, check its flags.json), 2 GPUs, ~overnight; then
`eval500_latentpsm_antmaze_sd{1,2}.json`. Turns 0.222 into a 3-seed mean ± CI next
to the 0.090 control. tmux `latentpsm_antmaze_sd{1,2}`.

**R2 — FB zero-shot on cube, 3 seeds × 500k.** `agent=fb` (bit-exact port,
`configs/agent/fb.yaml`), `scripts/launch_fb_cube.sh` conventions, same env/eval
protocol, reward-inferred z at eval (generic `infer_eval_z` path), 500-ep final
evals. Note in the report: reference FB keeps climbing to 1M — quote 500k as the
budget-matched comparison and say so.

**R3 — PSM (bilinear, action-space) zero-shot on cube, 3 seeds × 500k.**
`agent=psm`, flow actor, ortho_coef=1000, z_dim=128, lr_phi=1e-5 (the audited
recipe, `scripts/launch_psm_cube.sh`), 500-ep final evals. Known from the parity
work: expect ~0.2–0.4 at 500k with high seed variance — which is exactly why this
bar matters: it is the honest peer, and LatentFlowPSM 0.236 may NOT beat it. If it
doesn't, the paper's claim narrows to the support/coverage story (C=1, every action
in-support) rather than raw success — decide the framing AFTER the number exists,
not before.

**R4 — F3/F4 refresh.** Regenerate `fig_actor_comparison` with FB + PSM bars and the
3-seed antmaze row; replace the F4 calibration panel with D1's policy-ranking version
if D1 lands (the per-episode AUC moves to the appendix as a caveat, not deleted).

GPU budget: R1–R3 = 8 training runs ≈ 2 nights at 3–4 GPUs; B fits in one day on one
GPU alongside.

## E. Decision tree (read after B lands)

- D1 Spearman high → critic fine, diagnostic was confounded → ship actor_rank if it
  wins, refresh paper claims, C-tier optional.
- D2 implicates w → fix task inference before touching the actor; re-run D1 with the
  improved w; the "critic uninformative" line in the paper becomes "task projection
  is the bottleneck".
- D3 flat-Q / off-mode-flat → H2/H3: backup-exploration ablation first, bc_coeff
  second; if neither moves the curve, the honest paper story is "in-support latent BC
  + pessimism explains the gain; value improvement is open" — with D1–D3 as the
  evidence section. That is still a publishable, well-diagnosed result.
- R3 PSM ≥ ours → reframe headline around support constraint + diagnosis rigor, not
  SOTA-ness. Do not bury the bar.

## F. Acceptance checklist

- [ ] D1–D3 JSONs in logs/, each with a one-line `verdict` field naming the
      hypothesis it supports
- [ ] `acting=actor_rank` implemented + tested (no-op unless selected; existing
      psmflow tests stay green) + 500-ep eval JSON
- [ ] R1: 3-seed antmaze mean ± CI; R2/R3: FB + PSM cube bars with Wilson CIs
- [ ] ≤3 C-tier ablations launched, each traceable to a B verdict
- [ ] F3/F4 regenerated; HANDOFF entry summarizing verdicts; tmux server killed

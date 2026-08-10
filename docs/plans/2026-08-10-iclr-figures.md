# ICLR figure & evidence plan — 2026-08-10

Goal: produce the minimal figure set + missing measurements that let `PAPER/ICLR/main.tex`
answer four questions, each with one figure:

- **F1** Does the preimage contain reachable skills? (the headline negative result)
- **F2** How well does the flow fit the data? (fidelity + inversion quality)
- **F3** Flow-GPI vs flowBC latent actor vs FQL — which actor earns its keep?
- **F4** What policies is LatentFlowPSM actually learning?

Everything here is either (a) an eval of an existing checkpoint, (b) a figure assembled
from artifacts already on disk, or (c) one new baseline training run (cube FQL) and one
new viz tool. No new method work; do NOT re-audit Rung-1 Stage C — its 0.0 is root-caused
(fixed-u family is structurally non-goal-covering, see `docs/HANDOFF.md` 08-05 entry and
`PAPER/decisions.tex`).

## 0. Ground rules

- **Interpreter:** ALWAYS `.venv/bin/python` (bare `python` is system 2.7 on midi-01).
- **Env vars for anything that touches an env:** `MUJOCO_GL=egl`,
  `OGBENCH_DATASET_DIR=/var/local/amsks/ogbench`.
- **GPUs:** midi-01, shared, no Slurm. `nvidia-smi` first; one training seed per GPU
  (two sharing run ~2x slower each). Evals can share with
  `XLA_PYTHON_CLIENT_MEM_FRACTION=0.30`.
- **Every long-running job goes in a named tmux session** (names given per task below);
  `tmux kill-server` when the batch is done.
- **Reporting protocol (hard rule):** any number destined for the paper is either a
  500-episode Wilson interval (`tools/eval_checkpoint.py`) or a mean ± 95% CI across
  seeds (`scripts/compare_zeroshot.py` conventions). Never the best seed, never the peak
  of a training curve, except where "peak" is itself the declared statistic (D4 gate).
  Every Stage-C number is quoted NEXT TO the per-step-prior BC control from the same
  frozen flow (`agent=fql agent.bc_only=true`); cube control = 0.068 [0.049, 0.094].
- **Commits:** short subject + bullet body, no narrative paragraphs, no AI attribution
  trailers. Figure PDFs land in `PAPER/ICLR/figures/`; every figure script also writes a
  JSON of every number shown (via `utils/log_utils.py:write_report`) to
  `/data-local/amsks/PSMFLows/logs/` so no result is stdout-only.
- **Figure scripts:** one `tools/fig_*.py` per paper figure, deterministic (seeded),
  re-runnable, reading only from checkpoints/npz/JSON reports — never from hand-copied
  numbers. Matplotlib, colorblind-safe palette, single consistent style, output PDF
  (vector) + PNG preview. Target width: `\textwidth` for F1/F3, `0.9\textwidth` others.

## 1. Artifact inventory (verified paths)

| artifact | path |
|---|---|
| Stage-A flow, cube | `/var/local/amsks/exp/PSMFLows/bcflow_cube_single_20260726_135032/sd000_20260726_135037` |
| Stage-A flow, pointmaze | `/var/local/amsks/exp/PSMFLows/bcflow_pointmaze-medium-navigate_20260729_142219/sd000_20260729_142225` |
| Stage-A flow, antmaze | `/var/local/amsks/exp/PSMFLows/bcflow_antmaze-medium-navigate_20260805_014546/sd000_20260805_014548` |
| Preimages (alpha=20, N=200) | `/data-local/amsks/PSMFLows/preimages_{cube_single,pointmaze_medium,antmaze_medium}_a20_n200.npz` (+ `.meta.json`, `.analysis.json/.png` for cube/pointmaze) |
| LatentFlowPSM cube, 5 seeds | `/data-local/amsks/PSMFLows/exp/PSMFLows/psmflow_latentpsm_cube_021456/sd00{0..4}_*` (sd0/sd1: only `params_500000.pkl`; sd2–4: 10 ckpts on the 50k grid) |
| LatentFlowPSM antmaze, 1 seed | `/data-local/amsks/PSMFLows/exp/PSMFLows/psmflow_latentpsm_antmaze/sd000_20260805_225748` |
| Rung-1 runs (dead, for F3 bar only) | `.../psmflow_pointmaze-*`, `.../audit_psmflow_cube_204354` |
| FQL baseline, pointmaze (3 seeds) | `.../fqlbaseline_pointmaze_medium{,_s1,_s2}_*` — peak 0.687 ± 0.201 |
| PSM control, pointmaze (0.0) | `.../audit_psm_pointmaze_204448` |
| 500-ep evals done | `logs/eval500_latentpsm_cube_sd{0,1}.json` (0.318 / 0.236), `logs/eval500_bcflow_cube.json` (0.068) |
| Reachability reports | `logs/latent_reachability_{pointmaze,cube,antmaze}.json` (0/233, 2/128, 5/64) |
| Fixed-u field report | `logs/viz_fixed_u_field_pointmaze.json` + `fixed_u_fields.png` (repo root, gitignored) |
| Rung-1 calibration | `logs/calib_pointpre_sd0.json` (Spearman −0.132, realized all 0) |
| D1/D2/D3 reports | `logs/d2_*.json`, `logs/d3_*_a20_n200.json`; D1 numbers only in `docs/HANDOFF.md` (pre–report-persistence) — re-run, see WP2 |
| Pinned Stage-C tree | `/data-local/amsks/PSMFLows/pinned/repo_20260805_stagec/` (`PIN_MANIFEST.txt`) |

Before evaluating any LatentFlowPSM checkpoint, **diff your overrides against the run's
own `flags.json`** in its run dir — that is ground truth for `use_point_preimage`,
`pessimism_penalty`, actor knobs, etc. The npz↔checkpoint pairing guard will refuse
mismatched preimage/flow combos; do not work around it, fix the path.

## 2. WP0 — missing evaluations (do first; everything downstream reads these)

All are `tools/eval_checkpoint.py` runs, ~1–2 h each on a shared GPU. Template (cube):

```bash
MUJOCO_GL=egl OGBENCH_DATASET_DIR=/var/local/amsks/ogbench \
CUDA_VISIBLE_DEVICES=<g> XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 \
.venv/bin/python tools/eval_checkpoint.py agent=psmflow \
  env_name=cube-single-play-singletask-v0 \
  agent.flow_ckpt_path='/var/local/amsks/exp/PSMFLows/bcflow_cube_single_20260726_135032/sd000_*' \
  agent.flow_ckpt_epoch=500000 \
  agent.preimage_path=/data-local/amsks/PSMFLows/preimages_cube_single_a20_n200.npz \
  agent.use_point_preimage=true \
  restore_path='<run_dir>' restore_epoch=500000 eval_episodes=500 \
  report_out=/data-local/amsks/PSMFLows/logs/<name>.json
```

(Check how `eval500_latentpsm_cube_sd0.json` was produced — `logs/eval500_cube.log` — and
match it exactly; if `report_out` is not a recognized key there, use its actual mechanism.)

| # | eval | tmux name | output JSON |
|---|---|---|---|
| 0.1 | cube sd2/sd3/sd4 @500k, `acting=actor` (default) | `eval500_cube_sd{2,3,4}` | `eval500_latentpsm_cube_sd{2,3,4}.json` |
| 0.2 | cube sd0–sd4 @500k with `agent.acting=gpi` (same ckpts — the actor-free ablation) | `eval500_cube_gpi_sd{0..4}` | `eval500_latentpsm_cube_sd{N}_gpi.json` |
| 0.3 | antmaze sd0 @500k, `acting=actor` | `eval500_antmaze_sd0` | `eval500_latentpsm_antmaze_sd0.json` |
| 0.4 | antmaze BC control: `agent=fql agent.bc_only=true`, restore = antmaze flow dir | `eval500_bcflow_antmaze` | `eval500_bcflow_antmaze.json` |
| 0.5 | pointmaze BC control (for the F3 caption's cross-env row): same, pointmaze flow | `eval500_bcflow_pm` | `eval500_bcflow_pointmaze.json` |

Antmaze/pointmaze evals use their own env_name, flow dir, and
`preimages_{antmaze,pointmaze}_medium_a20_n200.npz`.

**Acceptance:** 0.1+0.2 give a 5-seed mean ± 95% CI (t-critical, n=5) for actor and gpi
modes plus pooled Wilson intervals; the actor-vs-gpi delta is the actor's isolated
contribution on identical representations.

## 3. WP1 — cube FQL per-task baseline (the missing topline)

3 seeds × 500k, one per GPU, mirroring the pointmaze baseline exactly. Copy the config
from `/data-local/amsks/PSMFLows/exp/PSMFLows/fqlbaseline_pointmaze_medium_20260803_185827/*/flags.json`
and change only env/seed. **Check `agent.alpha` against the FQL paper's per-env
recommendation for cube-single** (the OGBench FQL reference uses per-env alpha; the repo
default may be pointmaze-tuned) — record what you used in the run group name.

```bash
# tmux: fqlbase_cube_sd{0,1,2}, one GPU each
MUJOCO_GL=egl OGBENCH_DATASET_DIR=/var/local/amsks/ogbench CUDA_VISIBLE_DEVICES=<g> \
.venv/bin/python main.py agent=fql env_name=cube-single-play-singletask-v0 \
  offline_steps=500000 eval_interval=50000 eval_episodes=50 seed=<s> \
  save_dir=/data-local/amsks/PSMFLows/exp
```

Then 500-ep eval of each final checkpoint (`eval500_fql_cube_sd{0,1,2}.json`).
Statistic for F3: per-seed **peak** over the 50k grid (matching the pointmaze D4
convention) AND final@500k, both as mean ± 95% CI; 500-ep Wilson on the final ckpts.
~3–4 h training + evals; can run overnight alongside WP0.

## 4. WP2 — Figure F2: "the flow fits the data and inverts cleanly"

Script: `tools/fig_flow_fit.py` → `PAPER/ICLR/figures/fig_flow_fit.pdf` +
`logs/fig_flow_fit.json`. Three panels, one row:

- **(a) Fidelity vs control** — grouped bars, log-y: RBF-MMD and mode-hist TV for
  trained flow vs random-init control, per env. D1 predates report persistence, so
  first re-run `tools/validate_flow_fidelity.py` for {pointmaze, cube, antmaze} ×
  {trained, random control} with `report_out=logs/d1_<env>{,_random}.json` (mind the
  known wart: `mmd_rbf` hardcodes 1024 rows — fine, just note it in the JSON).
  Expected: cube ~1.5e-4 vs 1.46e-1; pointmaze 3.0e-4 vs 3.97e-3 (honest thin margin).
- **(b) Inversion typicality + ESS** — per env: histogram of ‖u*‖² over the full 1M-row
  npz with the χ²_{d_a} density overlaid; inset or twin panel: final-step ESS histogram
  with the gate line at 20. Read straight from the npz + `.analysis.json`. Antmaze's
  mixture arm MUST appear with its true ESS 7.6 / 6%>20 and be labeled "mixture arm
  unusable at d_a=8; point arm used throughout" — this caveat is already published on
  the HF card, the paper must match it.
- **(c) Roundtrip error CDF** — `preimage_roundtrip` per env, one line each (~1e-4).

Also regenerate the existing paper figure: `tools/plot_preimage_intuition.py` against
the alpha=20/N=200 cube npz (it reads the flow from the npz sidecar), replacing the
stale `PAPER/ICLR/figures/preimage_intuition.png` whose panels still say "alpha=1 as
run". Fix the stale panel titles in the tool while there (flagged in HANDOFF 08-04 §3).
GPU, ~5 min.

**Caption answer:** the flow reproduces the behavior distribution (vs a random-flow
control) and its inversion produces latents that are typical under the prior with
faithful decode — the preimages are *valid*; whatever fails downstream is not inversion
quality.

## 5. WP3 — Figure F1: "the preimages contain no reachable skill" (headline)

Script: `tools/fig_reachability.py` → `PAPER/ICLR/figures/fig_reachability.pdf` +
`logs/fig_reachability.json`. Three panels + one inset table. All inputs exist except
panel (c)'s statistics, which are recomputed from the npz (cheap, CPU).

- **(a) Exhaustive reachability map (pointmaze).** d_a=2, so the 13×13 grid covers the
  ENTIRE latent box: scatter/heatmap over [-3,3]² colored by min agent–goal distance over
  a full 1000-step rollout (from `logs/latent_reachability_pointmaze.json`), with dataset
  preimages and goal-transition preimages overlaid as distinct markers. Annotate: best
  candidate 1.32 (never reaches), 75% never closer than ~23.5, maze span ~30. The point:
  this is exhaustive — no selection mechanism over u can succeed.
- **(b) The two degenerate regimes.** Two small maze insets from the
  `tools/viz_fixed_u_field.py` outputs: typical-u orbit (path length 164, net
  displacement 1.4) and saturated-corner constant heading sliding past the goal. Re-run
  the tool if the repo-root PNG's layout doesn't decompose; it writes its JSON to
  `logs/viz_fixed_u_field_pointmaze.json`.
- **(c) Routes are latent white noise.** From `preimages_pointmaze_medium_a20_n200.npz`
  alone: autocorrelation of u*_t vs lag within episodes (lag-1 ≈ 0.27, ~0 by lag 50)
  and the within-episode/marginal variance ratio (0.99) as an annotation. Episode
  boundaries: use the dataset's terminal/valid structure as
  `tools/latent_reachability.py` does — check its episode-slicing code and reuse it.
- **Inset table:** family ceiling — pointmaze 0/233, cube 2/128 (~1% of
  latent-episodes), antmaze 5/64 (~4%) — next to the observed Rung-1 GPI results (0.0 /
  0.02–0.06 / n.a.). The match between ceiling and observed GPI floor is the sentence
  that carries the section: GPI was already extracting everything the family contains.

**Caption answer:** per-transition preimages are valid (F2) but temporally white — no
fixed u encodes a route, so the fixed-index family contains no goal-reaching member.
Not a training failure; a property of the object.

## 6. WP4 — Figure F3: actor comparison (flow-GPI vs flowBC latent actor vs FQL)

Script: `tools/fig_actor_comparison.py` → `PAPER/ICLR/figures/fig_actor_comparison.pdf`
+ `logs/fig_actor_comparison.json`. Two panels. Blocked on WP0 + WP1.

- **(a) Cube-single bars, 500-ep evals, Wilson 95% whiskers:**
  1. BC control (frozen flow, per-step prior latent) — 0.068
  2. Rung-1 fixed-u flow-GPI — 0.02–0.06 (from `audit_psmflow_cube_204354` eval.csv;
     if no 500-ep eval exists for it, run one — same template, its own flags.json)
  3. LatentFlowPSM, `acting=gpi` (same ckpts as 4 — actor-free ablation) — WP0.2
  4. LatentFlowPSM, `acting=actor` — 5 seeds pooled + per-seed dots — WP0.1
  5. FQL per-task topline — WP1, visually separated (different shade + dashed divider)
     and labeled "per-task (sees reward at train time)"; it is a ceiling, not a peer.
  Draw a horizontal line at the family ceiling (~1%) through bars 1–2: the fixed-u
  bars sit AT the ceiling; the latent-actor bars sit far above it.
- **(b) Training curves, cube:** success vs steps, mean ± 95% CI band across the 5
  LatentFlowPSM seeds (reuse `scripts/compare_zeroshot.py` aggregation) and the 3 FQL
  seeds; BC control as a flat reference line. Honest shape: LatentFlowPSM is flat-noisy
  after 50k — say so in the caption rather than hiding it.

**Caption answer:** selection over fixed latents (GPI) cannot beat the family ceiling;
moving policy identity to w with an amortized flowBC latent actor lifts cube success
~4x over the BC control (non-overlapping intervals); FQL's per-task score bounds what
remains. Include the pointmaze caveat sentence: the action-space PSM control also reads
0.0 on pointmaze-medium task1 (`audit_psm_pointmaze_204448`), so pointmaze zeros do not
discriminate between actors.

## 7. WP5 — Figure F4: what LatentFlowPSM policies actually do

One NEW tool: `tools/viz_policy_rollouts.py` (audit item; keep it small) — load a
psmflow checkpoint exactly as `tools/eval_checkpoint.py` does (infer w the same way),
roll N episodes, and dump per-step (state, u_actor, action, reward) to npz/JSON; render
xy-trajectory overlays for maze envs and success/failure flags. Seeded; `report_out`
JSON like the other tools.

Script: `tools/fig_policy_anatomy.py` → `PAPER/ICLR/figures/fig_policy_anatomy.pdf` +
`logs/fig_policy_anatomy.json`. Panels:

- **(a) Rollout gallery, antmaze** (xy plottable and it's the env with a positive
  1-seed result): trajectories over the maze, colored by time, N≈20 episodes,
  successes/failures distinguishable. Cube: a small strip of per-episode object
  displacement or success timeline (no xy plot available; keep minimal).
- **(b) The support claim, measured:** distribution of ‖u_actor‖ emitted during those
  rollouts vs (i) the dataset preimage ‖u*‖ distribution and (ii) the χ²_{d_a} prior
  curve, per env. This turns "every action is an in-support flow decode" from an
  assertion into a measurement. Flag the u_clip=3 boundary; report the fraction of
  actor latents within the typical set in the JSON.
- **(c) Calibration for the living agent.** Extend `tools/calibration_check.py` with an
  `actor` mode: at K eval start states, predicted diagonal score ψ(s, w, u_actor(s,w))ᵀw
  vs realized discounted return of the ACTOR policy rollout (the current tool rolls
  fixed-u candidates — meaningless post-pivot). Report Spearman across states. Run on
  cube sd0–sd2. Rung-1 read −0.13 with all-zero returns; whatever this reads, it goes
  in the paper — if it is still ~0, that is the stated open problem, not a thing to bury.

## 8. WP6 — paper + docs integration (after figures exist)

Minimal edits to `PAPER/ICLR/main.tex` (do not rewrite the paper; that is a separate
task):

1. Insert F1–F4 as `\begin{figure}` floats with the captions drafted above; wire
   `fig:preimage` to the regenerated PNG. F1 goes in the Findings section; F2 next to
   the gates table; F3+F4 in a new "LatentFlowPSM" results subsection.
2. Port the LatentFlowPSM definition paragraph from `PAPER/main.tex` §LatentFlowPSM
   (lines ~1063–1163) in compressed form; state the cube result as mean ± 95% CI over
   5 seeds (from WP0.1) next to the BC control, per protocol.
3. Correct Finding 3's diagnosis to the reachability root cause (currently blames TD
   grounding); add one sentence to the D2 row of `tab:gates` noting D2 measures
   coherence+diversity, not usefulness.
4. `\iclrfinalcopy` off for the anonymized build; check `main.pdf` compiles.

Docs hygiene (same PR):
- New HANDOFF entry (2026-08-10): the LatentFlowPSM results (cube 5-seed numbers, the
  pointmaze kill at 100–150k reading 0.0/0.02, antmaze 1-seed 0.32, the PSM pointmaze
  control landing 0.0) — currently recorded NOWHERE, and the repo has already lost
  results once (08-03 D2) to exactly this gap.
- Fill the HF repo ID `amsks/psmflows-preimages` into `README.md` and
  `docs/PREIMAGES.md` (both still say `<user>/<name>`); add `huggingface_hub` to
  `requirements.txt`.

## 9. Ordering & schedule

```
Day 1: WP0 evals (share 1–2 GPUs) ─┐        WP1 FQL cube (3 GPUs, overnight)
       WP2 fig_flow_fit + D1 reruns │  parallel
       WP5 tool skeleton            ┘
Day 2: WP3 F1 assembly · WP4 F3 (needs WP0/WP1) · WP5 F4 runs+assembly
Day 3: WP6 paper integration, HANDOFF entry, commit
```

Commit in coherent units: (1) eval reports + fig tools, (2) figures + paper edits,
(3) docs. Nothing under `/data-local` or `/var/local` is committed; JSON reports that
figures depend on should ALSO be copied into `PAPER/ICLR/figures/data/` so the paper
build is reproducible off-machine.

## 10. Acceptance checklist

- [ ] `eval500_latentpsm_cube_sd{2,3,4}.json` exist; 5-seed actor mean ± 95% CI computed
- [ ] `eval500_latentpsm_cube_sd{0..4}_gpi.json` exist; actor-vs-gpi delta reported
- [ ] `eval500_latentpsm_antmaze_sd0.json` + `eval500_bcflow_antmaze.json` exist
- [ ] cube FQL: 3 seeds trained, peak/final ± CI + 500-ep evals
- [ ] F1–F4 PDFs in `PAPER/ICLR/figures/` with sidecar JSONs; every number in a figure
      traceable to a JSON report
- [ ] `preimage_intuition.png` regenerated at alpha=20/N=200; stale titles fixed
- [ ] ICLR main.tex compiles with all four figures + corrected Finding 3
- [ ] HANDOFF 08-10 entry; HF repo ID in README + PREIMAGES.md
- [ ] tmux server killed when runs complete

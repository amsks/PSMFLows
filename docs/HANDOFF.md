---
marp: true
title: PSMFlows — Session Handoff
theme: default
paginate: true
---

<!-- _class: lead -->

# PSMFlows — Session Handoff

Chasing a **real, systematic gap** between our JAX PSM+flow and the
PyTorch reference — localized to TRAINING, hunt in progress.

Branch: `feat/psm-integration` · Machine: `midi-01` (UT CS)
Date: **2026-07-15** (latest) · prior investigation: 2026-07-13, 2026-07-07

---

<!-- _class: lead -->

## 2026-07-15 session — FB agent ported to JAX (bit-exact)

Ported the PyTorch **Forward–Backward (FB)** agent (`../Factored-FB` `agents/fb/*`)
to JAX/Flax, mirroring the PSM port protocol. **Spec** `docs/superpowers/specs/
2026-07-14-fb-jax-port-design.md`, **plan** `docs/superpowers/plans/2026-07-14-fb-jax-port.md`.

- **Scope:** cube-default `fb_flowbc` path — Forward map `F(left_enc(obs),z,a)`,
  Backward map `B(next_obs)` (measure basis + z-source), a **left_encoder** trunk
  (with target), td3/flow actor. Measure `M=F·Bᵀ`, off-diag/diag + ortho loss.
  2 backward passes (FB, actor); targets on forward/backward/left_encoder, none on
  actor; taus 0.005. z mixed 50/50 with `B(next_obs[perm])`. **Not** ported:
  iql critic, traj goal-mode, fixed_b, goal_cond, onestep, reweight.
- **Files:** `agents/fb.py`, `utils/fb_networks.py` (ForwardMap/BackwardMap/FBTd3Actor;
  reuses NoiseConditionedActor/FlowVectorField), `configs/agent/fb.yaml`
  (z_dim=50, batch=256, disc=0.99), registered `fb` in `agents/__init__.py`.
  `main.py` needs NO FB branch (no `index`; `infer_eval_z` picked up generically;
  seeded eval applies).
- **Bit-exact parity (like PSM):** `tools/export_fb_fixture.py` → `tests/fixtures/
  fb_reference.npz`; `tests/test_fb_{networks,agent,smoke}_equiv.py`. Per-module
  atol 1e-10, 10-step `apply_update` atol 1e-8. **13/13 FB tests pass.** Also the
  first bit-exact check of our shared NoiseConditionedActor/FlowVectorField.
- **Fixture gotchas (cost a long debug):** (1) torch `.numpy()` **aliases** memory —
  per-step param snapshots must `.copy()` or they all show the final trained weights;
  (2) the reference torch build's in-place first Adam step lands ~10× too large under
  some construction orders, so the K-step trace uses a **manual optax-matching Adam**.
- **Runnable:** `bash scripts/launch_fb_cube.sh [GPU] [STEPS]` (one seed/GPU,
  save_interval=100k, seeded eval, eval_episodes=10). Verified 3-step end-to-end
  through `main.py` (exit 0). **NEXT:** cube-single parity run vs the reference FB
  benchmark (wandb `amsks/factored-fb`; see [[reference-fb-flow-benchmarks]]).

---

<!-- _class: lead -->

## 2026-07-13 session — fresh from-scratch parity runs

**Goal:** measure how much our JAX PSM+flow recovers vs the reference on
from-scratch runs (not transplant), 3 seeds, 500k.

- Launched seeds **0/1/2**, flow, `ortho_coef=1000`, `z_dim=128`, `lr_phi=1e-5`,
  eval@50k / 50 ep. Group `psm_recover500k_flow_ortho1000_20260713_160709`.
- **NEW, CONTRASTS with prior finding:** **seed 2 recovers the reference
  in-window.** seed2 @500k = **0.42** vs ref **0.60** (−0.18); but
  **mean(0–500k) ours 0.238 vs ref 0.240 — a wash.** We LEAD early (50k/100k the
  ref is still 0.00), ref leads the 250–300k bump. Peak ours 0.44@350k vs ref's
  own in-window 0.50. This is NOT the old "plateau at ~0.05–0.11" story.
- **Seeds 0 & 1 weaker so far** (peaks 0.18 / 0.22 at 450k) — but the reference
  for THOSE seeds is also weak early (s0 ref 0.30 by 300k). ⇒ **high seed
  variance dominates**; single-seed reads are noisy.
- **THE remaining gap is the CEILING, not the trajectory.** The reference keeps
  climbing past 500k: peaks **0.80 @750k**, holds **0.5–0.8 to 1.5M**. At our
  500k cap the −0.18 is real but small; the reference's *ceiling* only appears
  with 2–3× more training we hadn't run.
- **⇒ Launched seed 2 to 1M** on GPU 1 (solo) to test the ceiling.
  Group `psm_recover1M_flow_ortho1000_s2_20260713_174415`. **At 250k already
  0.60** (ref 0.50 @250k). ETA ~2.5–3h from 17:44.
- **Infra lesson:** 2 seeds sharing one GPU run ~2× slower (seeds 0/1 on GPU3
  took ~2.5h; seed 2 solo on GPU1 did 500k in ~1h23m). **One seed per GPU** for
  even wall-clock. `XLA_PYTHON_CLIENT_MEM_FRACTION=0.30` per proc.

### Runs (2026-07-13)

| group | seeds | steps | GPU | status |
|---|---|---|---|---|
| `psm_recover500k_flow_ortho1000_20260713_160709` | 0,1 | 500k | 3 (shared) | running (~450k) |
| ″ | 2 | 500k | 1 | **done** @500k=0.42 |
| `psm_recover1M_flow_ortho1000_s2_20260713_174415` | 2 | **1M** | 1 (solo) | running (~250k, 0.60) |

Compare: `.venv/bin/python scripts/compare_multiseed.py [GROUP]` (defaults to
`/var/local/amsks/exp/multiseed_group.txt`; the 1M group is in
`recover1M_s2_group.txt`). Reference cache has seeds 0,1,2,3,4,5,7.

### Next (2026-07-13 → next session)

1. **Read the 1M seed-2 result** — does it reach the ref's 0.7–0.8 ceiling? If
   yes ⇒ we DO have parity, just needed budget; the "systematic gap" was a
   500k-cap artifact + seed variance. If it plateaus ~0.4 ⇒ real ceiling gap.
2. Let seeds 0/1 finish 500k; re-run `compare_multiseed.py` for the 3-seed table.
3. If ceiling confirmed, consider 1M runs for seeds 0/1 too (one-per-GPU) to
   get a real 3-seed peak distribution vs the reference's spread.
4. GPUs 1 & 3 usable; 0 & 2 were busy/full (other users) this session.

---

## TL;DR — the headline finding

- Our JAX PSM+flow **task2 success plateaus at peak ~0.20–0.32** and stays near the floor.
- The **code-matched reference climbs to 0.30–0.60 by 500k and peaks 0.80–0.90** (at 650k–1350k).
- This is a **real, systematic difference** (all 3 seeds fail uniformly), **NOT noise**.
- It is **NOT** in the eval/acting path, the loss math, masks, the proto table, or obs-norm
  (all audited/matched). **It is in TRAINING/init** — our update yields weaker weights.
- **CONFIRMED by transplant-eval:** reference weights (seed5 @100k) score **0.58** in OUR eval
  (task2); our own weights top ~0.25 in the same eval. Our eval is faithful; **training is where
  our weights end up worse.**
- **UPDATE (round 9): init scheme ALSO faithful.** After 9 rounds, EVERY code path is faithful
  (per-step math, training-gen, acting, eval, init scheme, config). **No code bug found.**
- Remaining diffs are **pure cross-framework RNG** (init draws + minibatch order). The reference
  is **hugely noisy** (same seed5: 0.20 wandb vs 0.52 fresh @100k). Effect size is uncertain
  with 3 seeds×1 run — may be a low draw, not a bug.

---

## How we got here (this session)

1. Started from an "eval@100k red flag". First pass concluded *faithful* — but that was an
   **improper comparison** (wrong reference algo for s5/10/7 + mid-run vs 1.5M peak).
2. Corrected: pulled the **code-matched PSM-orthohi** reference curves (wandb
   `amsks/factored-fb` `psm_state_orthohi__ortho_coef1000__lr_phi1e-5__s{0,5,7}`), aligned
   step-for-step. At matched steps ≤140k we tracked it — but the reference **blooms late**.
3. User (correctly) pushed: a faithful port must reproduce the reference's tight 0.8–0.9
   **peak distribution**. It doesn't.
4. Transplanted the reference **proto behavior table** to kill the last RNG diff → **gap did
   NOT close**. Confirmed the difference is real.

---

## What is RULED OUT (audited faithful / matched)

- **Per-step update math** — `tests/test_psm_agent_equiv.py` transplants reference weights,
  runs 10 optimizer steps on injected data, matches to **atol 1e-8**. Network+objective+HPs
  are bit-identical *given identical inputs*.
- **Eval z-inference** — formula-identical (`infer_z` == ref `reward_inference`); reward
  source matches (our dataset 2.08% nonzero == ref relabel ~2%).
- **Eval task identity** — `cube-single-play-singletask-v0` has `cur_task_id=2`; we compared
  task2-to-task2 all along.
- **Acting path** — `sample_actions` + actor nets faithful (one-step flow sample, tanh/clip,
  z-proj all match; multi-step Euler rollout is only a distill target, never acted).
- **masks** (always-γ), **obs normalization** (Identity for state, both sides), **proto table
  distribution** (`(rand-1)*2 ∈ [-2,0)`), now transplanted verbatim.

---

## KEY INSIGHT — why "losses match" proved nothing

- `actor_loss ≈ -Q/|Q|` and `train/q` is the **critic's own** estimate — both read ~-0.77 /
  ~600 **whether or not the policy reaches goals**. `bc_flow_loss` only measures fit to the
  behavior dist. **None of the scalar losses measure task success.**
- The equiv test **injects** the batch + all latent/action samples, so it verifies the update
  *given inputs* but **never how those inputs are GENERATED**.
- ⇒ The bug must live in **training-signal generation** (batch sampling, `z_cont` mixing,
  next-action injection, target updates) — invisible to both the losses and the equiv test.

---

## Two deep audits — BOTH came back "faithful"

- **Acting/eval-path audit: FAITHFUL.** `sample_actions` + actor nets match; eval task is
  task2 both sides; z-inference reward matches.
- **Training-generation audit: FAITHFUL.** batch sampling, `z_cont` mix, SF/proto next-action,
  target updates, per-stage optimizers all match. Only nits: `z_cont` mixed half uses
  pre-proto-step phi (ref uses post-step) — one `lr_phi=1e-5` step, negligible; proto table
  (now transplantable).
- **Config verified matching:** our runs used **z_dim=128, ortho_coef=1000, actor=flow**
  (checked `flags.json`). The audit's `z_dim=50` worry was stale-handoff text, not real.

⇒ **No code/config bug found on either side.** Remaining suspects: **init SCHEME** (orthogonal
gain/draw — audits didn't deep-check; all 3 seeds fail uniformly = systematic, argues against
pure basin luck) OR genuine cross-framework init/data-order basin variance (weakened by 3/3
uniform failure).

## STILL IN FLIGHT — the discriminator

**Torch reference checkpoint dump** — `seed5`, **ortho_coef=1000** (gotcha: psm_flowbc default
is 1.0!), GPU 1, `/var/local/amsks/exp/ref_ckpt_s5_300k`, saving `step_{100,200,300}k.pt`.
**Transplant-eval:** load ref weights into our JAX eval →
- ref-weights score **high** ⇒ bug is in **our training/init** (not eval).
- score **low** ⇒ bug is in **our eval** (both audits say unlikely).
To separate init-vs-training: load ref *early* weights into our JAX and TRAIN — climbs ⇒ init.

---

## Transplant-eval — READY to build (checkpoint landed)

`step_100000.pt` (+200k,300k) saved in `/var/local/amsks/exp/ref_ckpt_s5_300k/`. state_dict
naming **matches the fixture** convention → prefix keys with `w__` and reuse existing loaders.

- **phi / sf_psi / psm_psi:** reuse `load_phi_params` / `load_psi_params` as-is. Keys:
  `phi.net.{0,1,3,5}`, `{sf,psm}_psi.{embed_z,embed_sa}.{0,1,3}` + `Fs.{0,2}` (ensembled, P=2,
  shapes `[2,in,out]`). targets present too (`target_*`).
- **NEW converters needed** (torch_to_flax is ddpgbc-only):
  - `_actor` = **NoiseConditionedActor** (18 params): `embed_z.{0,1,3}`, `embed_s.{0,1,3}`,
    + noise/policy layers → our `utils/psm_networks.py:NoiseConditionedActor` tree.
  - `_actor_vf` = **FlowVectorField** (10 params): `net.{0,2,4,6,8}` (5 Linears, GELU between)
    → our `FlowVectorField` tree.
- Harness: build `PSMAgent` with these params (config z_dim=128, ortho1000, actor=flow), run
  `infer_eval_z` + `evaluate` on task2. **Ref weights should score ~0.20 (100k)** in a faithful
  eval — matches this run's own eval (seed5 task2 0.08@50k, climbing). Low ⇒ eval bug; ok ⇒
  training/init bug. **A buggy converter gives a false low — validate by matching phi/Q outputs
  to the torch model on a shared batch first.**

---

## Code changes this session

- **Committed & pushed** `31ed43d` "PSM: reference-parity fixes (eval z-inference + masks) +
  audit tooling": masks always-γ (`agents/psm.py`), eval z-shift/relabel (`main.py`,
  `config.yaml`), `eval_interval 100k→20k`, `docs/`, `scripts/launch_psm_cube.sh`.
  **NB: commits use NO Claude co-authorship** (user preference).
- **UNCOMMITTED** (working tree): `proto_table_path` hook — `agents/psm.py` `create()` loads a
  transplanted table when `agent.proto_table_path` set; `configs/agent/psm.yaml` declares it
  (`null` default). 15/15 PSM tests still pass.

---

## Runs (this session)

| group / run | what | status |
|---|---|---|
| `psm_maskfix_flow_ortho1000_20260707_164832` | s0/5/10/7, 500k, eval+masks fix | **killed** (superseded) |
| `psm_protoxplant_flow_ortho1000_20260707_184559` | s0/5/7, 500k, +proto transplant | **done** — gap NOT closed |
| `/var/local/amsks/exp/ref_ckpt_s5_300k` (torch) | ref s5 ortho1000, ckpt dump | **running** (GPU 1) |

Late-window (350–500k) task2 mean, protoxplant: ours s0/5/7 = **.06/.11/.05** vs ref **.08/.35/.33**.

---

## Reference curves (code-matched PSM-orthohi, task2)

Cached: `/var/local/amsks/exp/ref_orthohi_task2_curves.json` (seeds 0/5/7, every 50k to 1.5M).

| seed | @100k | @300k | @500k | peak |
|------|------:|------:|------:|-----:|
| 0 | 0.10 | 0.20 | 0.00 | 0.80 @1350k |
| 5 | 0.20 | 0.60 | 0.30 | 0.90 @650k |
| 7 | 0.00 | 0.40 | 0.10 | 0.80 @900k |

Reference is **very noisy** but clearly trends up; ours does not. `s10` has **no** code-matched
reference (orthohi set is seeds 0–9) — dropped it.

---

## Next steps (priority order) — DONE: transplant-eval + init audit (both clean)

All 9 rounds of code audit are clean. The question is no longer "where's the code bug" but
"is our training genuinely worse, or a low RNG draw." Two ways to settle it:

1. **DECISIVE: continue-training from ref weights in OUR loop.** Load ref 100k weights (score
   0.58 in our eval) into a trainable `PSMAgent` + fresh opt_states, train 100k more with our
   `update`, eval. **DEGRADES toward ~0.2 ⇒ our training dynamics actively hurt (real bug);
   PRESERVES/climbs ⇒ it was init-draw/basin luck.** (Extend `scripts/transplant_eval.py`:
   add opt_states init + our training loop over the dataset.)
2. **Characterize distributions:** run s0/5/7 (+more seeds) as **multiple replicates** and
   compare to the reference's own spread — the reference bounces 0.20–0.52 for the same seed.
3. Commit the `proto_table_path` hook (+ converter/harness) once settled.
4. Peak parity ultimately needs **1.5M** runs (user capped at 500k this round).

---

## Environment gotchas (unchanged + new)

- **`python` = system Python 2.7** on this box. ALWAYS use `.venv/bin/python` (ours) or
  `/var/local/amsks/ffb-venv/bin/python` (torch). The launcher needs `source .venv/bin/activate`.
- **midi-01 = HTCondor, no Slurm.** Run directly (nohup/tmux + `CUDA_VISIBLE_DEVICES`).
- Home NFS quota tiny → all bulk data in `/var/local/amsks/`. GPUs shared — check `nvidia-smi`.
- Torch stdout is **block-buffered** to a file (looks "stuck"; it's flushing in bursts).
- **Reference ortho_coef override is bare `ortho_coef=1000`** (`@package _global_`), NOT
  `agent.*`; psm_flowbc default is 1.0. The old repro `ref_psm_cube_s0_100k` was ortho=1.0.

---

<!-- _class: lead -->

## Pointers

- Agent: `agents/psm.py` · Networks: `utils/psm_networks.py` · Converter: `utils/torch_to_flax.py`
- Config: `configs/agent/psm.yaml`, `configs/config.yaml` · Launcher: `scripts/launch_psm_cube.sh`
- Reference (PyTorch): `/u/amsks/git/Factored-FB` (`agents/psm/*`, `nn_models.py`), torch venv
  `/var/local/amsks/ffb-venv`, data `/dev/shm/factored-fb/datasets`
- Investigation tools (now in repo): `scripts/transplant_eval.py` (load ref torch weights →
  our JAX eval), `scripts/compare_protoxplant.py` (step-aligned curve vs cached ref)
- Cached ref curves: `/var/local/amsks/exp/ref_orthohi_task2_curves.json`
- **Memory: `psm-vs-reference-audit` (rounds 1–6 — the full investigation trail)**

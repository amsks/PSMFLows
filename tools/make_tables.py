"""Assemble the two results tables from the 500-episode eval JSONs.

Hand-typed tables go stale the moment a queued seed lands, and this project now has three
ACTING MODES per checkpoint (deployed / decode-only / lambda-rank) whose numbers differ by
more than the effects being reported. So the mapping from "table cell" to "which eval JSONs"
is written out explicitly below, and everything else is derived.

Aggregation follows the project convention: more than one seed -> mean +/- 95% CI across
seeds (t interval); a single seed -> that run's Wilson interval, marked. A cell whose files
are missing prints as "--" and is listed under `missing` in the JSON, so a half-finished
table can never read as a complete one.

Writes:
  PAPER/ICLR/tables/table_headline.tex, table_fraction.tex   (booktabs, \\input-able)
  docs/tables/results.md                                     (same numbers, markdown)
  <logs>/table_dataset_fraction.json                         (machine-readable, incl. gaps)

Run: .venv/bin/python tools/make_tables.py [--logs DIR]
"""
import argparse
import glob
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LOGS = "/data-local/amsks/PSMFLows/logs"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---- cell -> eval files. Globs are resolved; every match is one seed. -----------------
# Kept explicit: filename conventions have drifted (lambdarank/decodeonly/1M suffixes), and
# a glob like eval500_hybrid_*.json would silently mix three acting modes into one number.
HEADLINE = [
    ("FQL (per-task reference, raw actions)", "per-task", ["eval500_fql_cube_sd?.json"]),
    ("Latent RL, per-task, eps=0.05 @peak", "per-task", ["eval500_l1stab_res0.05_*_peak*.json"]),
    ("Latent RL, per-task, eps=0 (pure decode)", "per-task", ["eval500_w4_res0.0_sd?_final.json"]),
    # 09-03 DSRL-SAC: scalar critic over LATENTS (critic_input=latent), flow never called in
    # training. Two u_clip arms, 2 seeds each. Both must be evaluated with the training
    # u_clip repeated (agent.u_clip, agent.critic_input=latent) -- the JSONs record both.
    ("Latent RL, DSRL-SAC (latent critic), u_clip=3", "per-task",
     ["eval500_latsac_uclip3_sd?.json"]),
    ("Latent RL, DSRL-SAC (latent critic), u_clip=1", "per-task",
     ["eval500_latsac_uclip1_sd?.json"]),
    ("FB (zero-shot, raw actions)", "zero-shot", ["eval500_fb_cube_sd?.json"]),
    ("PSMFlow (zero-shot, latent -> frozen decode)", "zero-shot", ["eval500_latentpsm_cube_sd?.json"]),
    ("PSMFlow, HP-matched to FB", "zero-shot", ["eval500_hpmatch_sd00?.json"]),
    ("Hybrid (action critic + residual), deployed", "zero-shot",
     ["eval500_hybrid_sd?_final.json", "eval500_hybrid_1M_sd?.json"]),
    ("Hybrid, decode-only control", "zero-shot",
     ["eval500_hybrid_decodeonly_sd00?.json", "eval500_hybrid_1M_decodeonly_sd00?.json"]),
    ("Hybrid, lambda-rank (K=32, no residual)", "zero-shot",
     ["eval500_hybrid_lambdarank_sd00?.json"]),
    ("Hybrid + FB graft, deployed", "zero-shot", ["eval500_fbgraft_sd?_deployed.json"]),
    ("Hybrid + FB graft, decode-only control", "zero-shot",
     ["eval500_fbgraft_sd?_decodeonly.json"]),
    # The BC controls were produced on two machines with different basenames and
    # directories (midi-01: <logs>/eval500_bcflow_cube.json; KISSKI: $PSM_DATA/evals/
    # bc_cube.json). Both spellings are listed; _load de-duplicates identical evals, so
    # a machine that happens to hold both does not report the control as "2 seeds".
    ("Behavior-cloning control (per-step prior)", "control",
     ["eval500_bcflow_cube.json", "../evals/bc_cube.json"]),
    # E2 (08-31): same 5 checkpoints re-evaluated post-P0.2 seeding, one-step vs ODE-100.
    ("PSMFlow re-eval, actor, one-step decode", "zero-shot", ["e2_onestep_actor_sd?.json"]),
    ("PSMFlow re-eval, actor, exact ODE-100 decode", "zero-shot", ["e2_ode_actor_sd?.json"]),
    ("PSMFlow re-eval, gpi, one-step decode", "zero-shot", ["e2_onestep_gpi_sd?.json"]),
    ("PSMFlow re-eval, gpi, exact ODE-100 decode", "zero-shot", ["e2_ode_gpi_sd?.json"]),
    # E3 (08-31): paper-faithful arms; epoch recorded per-file in table_dataset_fraction.json.
    # `sd?` only, NOT `sd?*`: the mid-training `*_ep250000.json` evals of the runs the
    # 08-31 disk-full killed must not be pooled with the 500k results that replaced them.
    ("Paper-faithful Arm A (u'~p0 bootstrap)", "zero-shot", ["eval500_paperfaith_armA_sd?.json"]),
    ("Paper-faithful Arm B (psi(s,u,u'), no actor, gpi)", "zero-shot",
     ["eval500_paperfaith_armB_sd?.json"]),
    # 09-03 point-vs-mixture ablation (audit discrepancy #3: the write-up's q_alpha is the
    # MIXTURE, every reported number was the point preimage). Shipped defaults otherwise
    # (acting=actor, policy_index=task_vector, train_actor=true, u_clip=3.0), 2 seeds each.
    # The mixture arm CANNOT use the canonical npz -- its sidecar records no prior_scale, so
    # main.py refuses use_point_preimage=false against the legacy likelihood-only target --
    # so it runs on the HPO-corrected npz (cube a20.6/ps0.69/ns12, antmaze a26.5/ps0.60/ns5).
    # `pointps` is the confound control: POINT preimages read from that SAME corrected npz,
    # which separates "mixture vs point" from "one npz vs the other". Antmaze rows are the
    # first 500-episode psmflow numbers on that env; they do not pool with the cube rows.
    ("PSMFlow point preimage, cube (canonical npz)", "zero-shot",
     ["eval500_psmflow_cube_point_sd?.json"]),
    ("PSMFlow mixture q_alpha, cube (corrected npz)", "zero-shot",
     ["eval500_psmflow_cube_mix_sd?.json"]),
    ("PSMFlow point preimage, cube (corrected npz control)", "zero-shot",
     ["eval500_psmflow_cube_pointps_sd?.json"]),
    ("PSMFlow point preimage, antmaze (canonical npz)", "zero-shot",
     ["eval500_psmflow_antmaze_point_sd?.json"]),
    ("PSMFlow mixture q_alpha, antmaze (corrected npz)", "zero-shot",
     ["eval500_psmflow_antmaze_mix_sd?.json"]),
    ("PSMFlow point preimage, antmaze (corrected npz control)", "zero-shot",
     ["eval500_psmflow_antmaze_pointps_sd?.json"]),
    ("Behavior-cloning control, antmaze (per-step prior)", "control",
     ["bc_control_antmaze.json", "../evals/bc_antmaze.json"]),
    # 09-04/09-05 affine measure head (docs/design/2026-09-04-affine-psi.md):
    # psi(s,u,u') = A(s,u)^T w(u') + beta(s,u) with a learned policy encoder, i.e. Prop.
    # bilinear made explicit instead of Rem. tradeoff's free psi. Both arms are
    # policy_index=latent; `strict` is Arm B's settings (train_actor=false, acting=gpi),
    # `actor` is the DSRL-style latent actor on the same substrate.
    #
    # ONE ROW PER (arm, env, CHECKPOINT). The filename carries the epoch --
    # eval500_affine<N>k_<arm>_<env>_sd<S>.json -- EXCEPT the first batch (09-04), whose
    # basenames have no epoch token at all and are all restore_epoch=100000; those are the
    # `@100k` rows below and must not be read as 500k numbers, which is exactly the bug
    # this layout replaces. Epochs never pool with each other and cube never pools with
    # antmaze; the only cross-epoch cells are the explicitly labelled `late-ckpt mean`
    # rows, which exist because this arm oscillates by +/-0.3 between 50k checkpoints and
    # no single checkpoint of it may be quoted on its own (see the 09-05 handoff entry).
    # Cube ladder, 09-06: every 50k checkpoint from 50k to 500k, three seeds. `sd?`
    # already spans sd0/sd1/sd2 -- a new seed needs no glob change, only a run.
    ("Affine psi, strict (no actor, gpi), cube @50k", "zero-shot",
     ["eval500_affine50k_strict_cube_sd?.json"]),
    ("Affine psi, strict (no actor, gpi), cube @100k", "zero-shot",
     ["eval500_affine_strict_cube_sd?.json", "eval500_affine100k_strict_cube_sd?.json"]),
    ("Affine psi, strict, cube @150k", "zero-shot",
     ["eval500_affine150k_strict_cube_sd?.json"]),
    ("Affine psi, strict, cube @200k", "zero-shot",
     ["eval500_affine200k_strict_cube_sd?.json"]),
    ("Affine psi, strict, cube @250k", "zero-shot",
     ["eval500_affine250k_strict_cube_sd?.json"]),
    ("Affine psi, strict, cube @300k", "zero-shot",
     ["eval500_affine300k_strict_cube_sd?.json"]),
    ("Affine psi, strict, cube @350k", "zero-shot",
     ["eval500_affine350k_strict_cube_sd?.json"]),
    ("Affine psi, strict, cube @400k", "zero-shot",
     ["eval500_affine400k_strict_cube_sd?.json"]),
    ("Affine psi, strict, cube @450k", "zero-shot",
     ["eval500_affine450k_strict_cube_sd?.json"]),
    ("Affine psi, strict, cube @500k", "zero-shot",
     ["eval500_affine500k_strict_cube_sd?.json"]),
    ("Affine psi, strict, cube -- late-ckpt mean (250k-500k x seeds)", "zero-shot",
     ["eval500_affine250k_strict_cube_sd?.json", "eval500_affine300k_strict_cube_sd?.json",
      "eval500_affine350k_strict_cube_sd?.json", "eval500_affine400k_strict_cube_sd?.json",
      "eval500_affine450k_strict_cube_sd?.json", "eval500_affine500k_strict_cube_sd?.json"]),
    ("Affine psi, strict, cube -- late-ckpt mean (300k-500k x seeds)", "zero-shot",
     ["eval500_affine300k_strict_cube_sd?.json", "eval500_affine350k_strict_cube_sd?.json",
      "eval500_affine400k_strict_cube_sd?.json", "eval500_affine450k_strict_cube_sd?.json",
      "eval500_affine500k_strict_cube_sd?.json"]),
    ("Affine psi, latent actor, cube @100k", "zero-shot",
     ["eval500_affine_actor_cube_sd?.json"]),
    ("Affine psi, latent actor, cube @250k", "zero-shot",
     ["eval500_affine250k_actor_cube_sd?.json"]),
    ("Affine psi, latent actor, cube @500k", "zero-shot",
     ["eval500_affine500k_actor_cube_sd?.json"]),
    ("Affine psi, strict (no actor, gpi), antmaze @50k", "zero-shot",
     ["eval500_affine50k_strict_antmaze_sd?.json"]),
    ("Affine psi, strict, antmaze @250k", "zero-shot",
     ["eval500_affine250k_strict_antmaze_sd?.json"]),
    ("Affine psi, strict, antmaze @300k", "zero-shot",
     ["eval500_affine300k_strict_antmaze_sd?.json"]),
    ("Affine psi, strict, antmaze @350k", "zero-shot",
     ["eval500_affine350k_strict_antmaze_sd?.json"]),
    ("Affine psi, strict, antmaze @400k", "zero-shot",
     ["eval500_affine400k_strict_antmaze_sd?.json"]),
    ("Affine psi, strict, antmaze @500k", "zero-shot",
     ["eval500_affine500k_strict_antmaze_sd?.json"]),
    ("Affine psi, strict, antmaze -- late-ckpt mean (250k-500k x seeds)", "zero-shot",
     ["eval500_affine250k_strict_antmaze_sd?.json",
      "eval500_affine300k_strict_antmaze_sd?.json",
      "eval500_affine350k_strict_antmaze_sd?.json",
      "eval500_affine400k_strict_antmaze_sd?.json",
      "eval500_affine500k_strict_antmaze_sd?.json"]),
    ("Affine psi, latent actor, antmaze @50k", "zero-shot",
     ["eval500_affine50k_actor_antmaze_sd?.json"]),
    ("Affine psi, latent actor, antmaze @100k", "zero-shot",
     ["eval500_affine_actor_antmaze_sd?.json"]),
    ("Affine psi, latent actor, antmaze @500k", "zero-shot",
     ["eval500_affine500k_actor_antmaze_sd?.json"]),
    # Pointmaze ladder, 09-06: the third env with published Stage-A/B artifacts, run at the
    # repo-default affine strict settings. env is pointmaze-medium-navigate-singletask-TASK1
    # (the maze default, ogbench/locomaze/maze.py:348), not the bare suffix. COMPENDIUM 4.11
    # records every earlier Stage-C variant reading 0.0 here with a diagnosed structural
    # cause, so these rows exist to record whether the affine head changes that -- read a
    # zero as the expected outcome, not as a broken run.
    ("Affine psi, strict (no actor, gpi), pointmaze @50k", "zero-shot",
     ["eval500_affine50k_strict_pointmaze_sd?.json"]),
    ("Affine psi, strict, pointmaze @100k", "zero-shot",
     ["eval500_affine100k_strict_pointmaze_sd?.json"]),
    ("Affine psi, strict, pointmaze @150k", "zero-shot",
     ["eval500_affine150k_strict_pointmaze_sd?.json"]),
    ("Affine psi, strict, pointmaze @200k", "zero-shot",
     ["eval500_affine200k_strict_pointmaze_sd?.json"]),
    ("Affine psi, strict, pointmaze @250k", "zero-shot",
     ["eval500_affine250k_strict_pointmaze_sd?.json"]),
    ("Affine psi, strict, pointmaze @300k", "zero-shot",
     ["eval500_affine300k_strict_pointmaze_sd?.json"]),
    ("Affine psi, strict, pointmaze @350k", "zero-shot",
     ["eval500_affine350k_strict_pointmaze_sd?.json"]),
    ("Affine psi, strict, pointmaze @400k", "zero-shot",
     ["eval500_affine400k_strict_pointmaze_sd?.json"]),
    ("Affine psi, strict, pointmaze @450k", "zero-shot",
     ["eval500_affine450k_strict_pointmaze_sd?.json"]),
    ("Affine psi, strict, pointmaze @500k", "zero-shot",
     ["eval500_affine500k_strict_pointmaze_sd?.json"]),
    ("Affine psi, strict, pointmaze -- late-ckpt mean (300k-500k x seeds)", "zero-shot",
     ["eval500_affine300k_strict_pointmaze_sd?.json",
      "eval500_affine350k_strict_pointmaze_sd?.json",
      "eval500_affine400k_strict_pointmaze_sd?.json",
      "eval500_affine450k_strict_pointmaze_sd?.json",
      "eval500_affine500k_strict_pointmaze_sd?.json"]),
    ("Behavior-cloning control, pointmaze (per-step prior)", "control",
     ["bc_control_pointmaze.json"]),
    # 09-07 DISCOUNT SWEEP (docs/design/2026-09-07-discount-sweep.md). Same repo-default
    # affine-strict agent, same flow/preimages/seeds/budget, ONE key changed:
    # agent.discount 0.98 -> 0.99 (effective horizon 50 -> 100 steps). Motivated by the
    # 09-06 antmaze H2 test, where 0.99 read 0.534 @100k against a 0.081 late-ckpt mean at
    # 0.98, and 0.995 read 0.678 and then collapsed to 0.000 in-loop on all three seeds.
    # The `_g99` infix is what keeps these disjoint from the rows above -- the gamma=0.98
    # basenames carry no discount token, so a glob without the infix would pool the two
    # discounts into one number. Rows are per (env, checkpoint) exactly like the 0.98
    # ladders, plus the labelled 300k-500k pooled means; epochs never pool with each other.
    ("Affine psi, strict, cube g=0.99 @50k", "zero-shot",
     ["eval500_affine50k_strict_cube_g99_sd?.json"]),
    ("Affine psi, strict, cube g=0.99 @100k", "zero-shot",
     ["eval500_affine100k_strict_cube_g99_sd?.json"]),
    ("Affine psi, strict, cube g=0.99 @150k", "zero-shot",
     ["eval500_affine150k_strict_cube_g99_sd?.json"]),
    ("Affine psi, strict, cube g=0.99 @200k", "zero-shot",
     ["eval500_affine200k_strict_cube_g99_sd?.json"]),
    ("Affine psi, strict, cube g=0.99 @250k", "zero-shot",
     ["eval500_affine250k_strict_cube_g99_sd?.json"]),
    ("Affine psi, strict, cube g=0.99 @300k", "zero-shot",
     ["eval500_affine300k_strict_cube_g99_sd?.json"]),
    ("Affine psi, strict, cube g=0.99 @350k", "zero-shot",
     ["eval500_affine350k_strict_cube_g99_sd?.json"]),
    ("Affine psi, strict, cube g=0.99 @400k", "zero-shot",
     ["eval500_affine400k_strict_cube_g99_sd?.json"]),
    ("Affine psi, strict, cube g=0.99 @450k", "zero-shot",
     ["eval500_affine450k_strict_cube_g99_sd?.json"]),
    ("Affine psi, strict, cube g=0.99 @500k", "zero-shot",
     ["eval500_affine500k_strict_cube_g99_sd?.json"]),
    ("Affine psi, strict, cube g=0.99 -- late-ckpt mean (300k-500k x seeds)", "zero-shot",
     ["eval500_affine300k_strict_cube_g99_sd?.json",
      "eval500_affine350k_strict_cube_g99_sd?.json",
      "eval500_affine400k_strict_cube_g99_sd?.json",
      "eval500_affine450k_strict_cube_g99_sd?.json",
      "eval500_affine500k_strict_cube_g99_sd?.json"]),
    ("Affine psi, strict, pointmaze g=0.99 @50k", "zero-shot",
     ["eval500_affine50k_strict_pointmaze_g99_sd?.json"]),
    ("Affine psi, strict, pointmaze g=0.99 @100k", "zero-shot",
     ["eval500_affine100k_strict_pointmaze_g99_sd?.json"]),
    ("Affine psi, strict, pointmaze g=0.99 @150k", "zero-shot",
     ["eval500_affine150k_strict_pointmaze_g99_sd?.json"]),
    ("Affine psi, strict, pointmaze g=0.99 @200k", "zero-shot",
     ["eval500_affine200k_strict_pointmaze_g99_sd?.json"]),
    ("Affine psi, strict, pointmaze g=0.99 @250k", "zero-shot",
     ["eval500_affine250k_strict_pointmaze_g99_sd?.json"]),
    ("Affine psi, strict, pointmaze g=0.99 @300k", "zero-shot",
     ["eval500_affine300k_strict_pointmaze_g99_sd?.json"]),
    ("Affine psi, strict, pointmaze g=0.99 @350k", "zero-shot",
     ["eval500_affine350k_strict_pointmaze_g99_sd?.json"]),
    ("Affine psi, strict, pointmaze g=0.99 @400k", "zero-shot",
     ["eval500_affine400k_strict_pointmaze_g99_sd?.json"]),
    ("Affine psi, strict, pointmaze g=0.99 @450k", "zero-shot",
     ["eval500_affine450k_strict_pointmaze_g99_sd?.json"]),
    ("Affine psi, strict, pointmaze g=0.99 @500k", "zero-shot",
     ["eval500_affine500k_strict_pointmaze_g99_sd?.json"]),
    ("Affine psi, strict, pointmaze g=0.99 -- late-ckpt mean (300k-500k x seeds)",
     "zero-shot",
     ["eval500_affine300k_strict_pointmaze_g99_sd?.json",
      "eval500_affine350k_strict_pointmaze_g99_sd?.json",
      "eval500_affine400k_strict_pointmaze_g99_sd?.json",
      "eval500_affine450k_strict_pointmaze_g99_sd?.json",
      "eval500_affine500k_strict_pointmaze_g99_sd?.json"]),
    # 09-05 GPI EVAL-TIME ABLATIONS (docs/design/2026-09-05-gpi-ablations.md). One run only
    # -- affine strict cube sd0 -- restored at the 0.704 checkpoint (350k) and, for the two
    # arms that mattered, at the 0.086 one (500k), with the acting rule swapped at EVAL
    # time and nothing retrained. These are SINGLE-SEED, SINGLE-CHECKPOINT diagnostics of
    # the acting rule, not agent results: they never pool with each other, never pool with
    # the rows above, and each carries its own Wilson interval.
    ("GPI abl @350k: argmax (the shipped rule, reference)", "ablation",
     ["eval500_gpiabl_argmax_350000.json"]),
    ("GPI abl @350k: ensemble mean instead of min", "ablation",
     ["eval500_gpiabl_mean_350000.json"]),
    ("GPI abl @350k: small_ball", "ablation", ["eval500_gpiabl_small_ball_350000.json"]),
    ("GPI abl @500k: small_ball", "ablation", ["eval500_gpiabl_small_ball_500000.json"]),
    ("GPI abl @350k: soft_topm", "ablation", ["eval500_gpiabl_soft_topm_350000.json"]),
    ("GPI abl @500k: soft_topm", "ablation", ["eval500_gpiabl_soft_topm_500000.json"]),
    ("GPI abl @350k: top_quartile", "ablation",
     ["eval500_gpiabl_top_quartile_350000.json"]),
    ("GPI abl @350k: max_norm (critic-free, large-norm select)", "ablation",
     ["eval500_gpiabl_max_norm_350000.json"]),
    ("GPI abl @350k: random_draw (one prior u', no selection)", "ablation",
     ["eval500_gpiabl_random_draw_350000.json"]),
    ("GPI abl @350k: fixed_index s0 (one u' held all episode)", "ablation",
     ["eval500_gpiabl_fixed_index_s0_350000.json"]),
    ("GPI abl @350k: fixed_index s1 (one u' held all episode)", "ablation",
     ["eval500_gpiabl_fixed_index_s1_350000.json"]),
    # 09-06 H1: the same fixed_index ablation on ANTMAZE (docs/design/
    # 2026-09-06-antmaze-failure-tests.md). Three affine-strict antmaze checkpoints x four
    # pinned index draws. Like the cube ablations above these are single-seed,
    # single-checkpoint acting-rule diagnostics: they never pool -- the four index seeds of
    # one checkpoint span 0.000-0.142, so a mean over them would hide the whole finding.
    ("H1 antmaze fixed_index: sd1 @250k, index seed 0 (argmax ref 0.178)", "ablation",
     ["eval500_antmaze_fixedidx_sd1_250k_is0.json"]),
    ("H1 antmaze fixed_index: sd1 @250k, index seed 1 (argmax ref 0.178)", "ablation",
     ["eval500_antmaze_fixedidx_sd1_250k_is1.json"]),
    ("H1 antmaze fixed_index: sd1 @250k, index seed 2 (argmax ref 0.178)", "ablation",
     ["eval500_antmaze_fixedidx_sd1_250k_is2.json"]),
    ("H1 antmaze fixed_index: sd1 @250k, index seed 3 (argmax ref 0.178)", "ablation",
     ["eval500_antmaze_fixedidx_sd1_250k_is3.json"]),
    ("H1 antmaze fixed_index: sd2 @300k, index seed 0 (argmax ref 0.088)", "ablation",
     ["eval500_antmaze_fixedidx_sd2_300k_is0.json"]),
    ("H1 antmaze fixed_index: sd2 @300k, index seed 1 (argmax ref 0.088)", "ablation",
     ["eval500_antmaze_fixedidx_sd2_300k_is1.json"]),
    ("H1 antmaze fixed_index: sd2 @300k, index seed 2 (argmax ref 0.088)", "ablation",
     ["eval500_antmaze_fixedidx_sd2_300k_is2.json"]),
    ("H1 antmaze fixed_index: sd2 @300k, index seed 3 (argmax ref 0.088)", "ablation",
     ["eval500_antmaze_fixedidx_sd2_300k_is3.json"]),
    ("H1 antmaze fixed_index: sd0 @350k, index seed 0 (argmax ref 0.082)", "ablation",
     ["eval500_antmaze_fixedidx_sd0_350k_is0.json"]),
    ("H1 antmaze fixed_index: sd0 @350k, index seed 1 (argmax ref 0.082)", "ablation",
     ["eval500_antmaze_fixedidx_sd0_350k_is1.json"]),
    ("H1 antmaze fixed_index: sd0 @350k, index seed 2 (argmax ref 0.082)", "ablation",
     ["eval500_antmaze_fixedidx_sd0_350k_is2.json"]),
    ("H1 antmaze fixed_index: sd0 @350k, index seed 3 (argmax ref 0.082)", "ablation",
     ["eval500_antmaze_fixedidx_sd0_350k_is3.json"]),
    # 09-06 H2 (docs/design/2026-09-06-antmaze-failure-tests.md): the antmaze DISCOUNT
    # sweep. Same repo-default affine-strict agent, three seeds per discount, everything
    # but `agent.discount` byte-identical. ONE ROW PER (discount, checkpoint), pooling the
    # three seeds -- never across checkpoints, because gamma=0.995 wins early (0.678 @100k)
    # and then collapses to the floor from 300k, so a late-checkpoint mean and an early one
    # describe different agents. Full per-cell ladder with Wilson intervals:
    # docs/tables/affine_antmaze_discount_ladder.md.
    ("Affine psi, strict, antmaze gamma=0.99 @50k", "zero-shot",
     ["eval500_affine50k_strict_antmaze_g99_sd?.json"]),
    ("Affine psi, strict, antmaze gamma=0.99 @100k", "zero-shot",
     ["eval500_affine100k_strict_antmaze_g99_sd?.json"]),
    ("Affine psi, strict, antmaze gamma=0.99 @150k", "zero-shot",
     ["eval500_affine150k_strict_antmaze_g99_sd?.json"]),
    ("Affine psi, strict, antmaze gamma=0.99 @200k", "zero-shot",
     ["eval500_affine200k_strict_antmaze_g99_sd?.json"]),
    ("Affine psi, strict, antmaze gamma=0.99 @250k", "zero-shot",
     ["eval500_affine250k_strict_antmaze_g99_sd?.json"]),
    ("Affine psi, strict, antmaze gamma=0.99 @300k", "zero-shot",
     ["eval500_affine300k_strict_antmaze_g99_sd?.json"]),
    ("Affine psi, strict, antmaze gamma=0.99 @350k", "zero-shot",
     ["eval500_affine350k_strict_antmaze_g99_sd?.json"]),
    ("Affine psi, strict, antmaze gamma=0.99 @400k", "zero-shot",
     ["eval500_affine400k_strict_antmaze_g99_sd?.json"]),
    ("Affine psi, strict, antmaze gamma=0.99 @450k", "zero-shot",
     ["eval500_affine450k_strict_antmaze_g99_sd?.json"]),
    ("Affine psi, strict, antmaze gamma=0.99 @500k", "zero-shot",
     ["eval500_affine500k_strict_antmaze_g99_sd?.json"]),
    ("Affine psi, strict, antmaze gamma=0.995 @50k", "zero-shot",
     ["eval500_affine50k_strict_antmaze_g995_sd?.json"]),
    ("Affine psi, strict, antmaze gamma=0.995 @100k", "zero-shot",
     ["eval500_affine100k_strict_antmaze_g995_sd?.json"]),
    ("Affine psi, strict, antmaze gamma=0.995 @150k", "zero-shot",
     ["eval500_affine150k_strict_antmaze_g995_sd?.json"]),
    ("Affine psi, strict, antmaze gamma=0.995 @200k", "zero-shot",
     ["eval500_affine200k_strict_antmaze_g995_sd?.json"]),
    ("Affine psi, strict, antmaze gamma=0.995 @250k", "zero-shot",
     ["eval500_affine250k_strict_antmaze_g995_sd?.json"]),
    ("Affine psi, strict, antmaze gamma=0.995 @300k", "zero-shot",
     ["eval500_affine300k_strict_antmaze_g995_sd?.json"]),
    ("Affine psi, strict, antmaze gamma=0.995 @500k", "zero-shot",
     ["eval500_affine500k_strict_antmaze_g995_sd?.json"]),
    ("Affine psi, strict, antmaze gamma=0.99 -- early-ckpt mean (50k-250k x seeds)", "zero-shot",
     ["eval500_affine50k_strict_antmaze_g99_sd?.json",
      "eval500_affine100k_strict_antmaze_g99_sd?.json",
      "eval500_affine150k_strict_antmaze_g99_sd?.json",
      "eval500_affine200k_strict_antmaze_g99_sd?.json",
      "eval500_affine250k_strict_antmaze_g99_sd?.json"]),
    ("Affine psi, strict, antmaze gamma=0.995 -- early-ckpt mean (50k-250k x seeds)", "zero-shot",
     ["eval500_affine50k_strict_antmaze_g995_sd?.json",
      "eval500_affine100k_strict_antmaze_g995_sd?.json",
      "eval500_affine150k_strict_antmaze_g995_sd?.json",
      "eval500_affine200k_strict_antmaze_g995_sd?.json",
      "eval500_affine250k_strict_antmaze_g995_sd?.json"]),
]

# Static provenance notes appended to the markdown table.
NOTES = [
    "The FB-graft rows aggregate every `eval500_fbgraft_sd?_*.json` present; the "
    "previously quoted single-seed 0.064 predates sd0's JSONs landing (08-14 23:05).",
    "sd0 headline evals recorded before the P0.2 eval-seeding fix are not exactly "
    "reproducible (sd0 re-eval 0.240 vs recorded 0.318); the E2 re-eval rows are the "
    "post-fix measurement of the same checkpoints and supersede the PSMFlow headline "
    "row for comparisons.",
    "E1 oracle-aim (below) is a diagnostic, not an agent: an oracle picks among K=512 "
    "decoded prior latents using a frozen FQL expert's action.",
]


# ---- Zero-shot across the five cube tasks (2026-09-06). ------------------------------
# EVERY cube number above this line is OGBench **task 2**: `cube-single-play-singletask-v0`
# carries no task token and `cube_env.py` defaults `reward_task_id` to 2, while
# `utils/evaluation.py` never passes `options={'task_id': ...}`. (The maze envs default to
# task 1, so every antmaze and pointmaze number is task 1.) Training reads rewards nowhere --
# `agents/psmflow.py` touches them only in `infer_z`/`infer_eval_z` at eval -- and every task
# shares one `cube-single-play-v0.npz` with only the reward column relabelled, so the SAME
# checkpoints evaluate zero-shot on all five tasks by changing `env_name` alone.
# Epochs are listed explicitly because the window is 300k-500k and a glob cannot filter on
# `restore_epoch`; each row therefore pools 5 checkpoints x 3 seeds = 15 measurements.
MULTITASK_EPOCHS = [300, 350, 400, 450, 500]


def _mt_pats(task):
    if task == 2:  # the default id: no task token in the basename
        return [f"eval500_affine{n}k_strict_cube_sd?.json" for n in MULTITASK_EPOCHS]
    return [f"eval500_affine{n}k_strict_cube_task{task}_sd?.json" for n in MULTITASK_EPOCHS]


MULTITASK = [(t, _mt_pats(t),
              ["../evals/bc_cube.json"] if t == 2 else [f"../evals/bc_cube_task{t}.json"])
             for t in (1, 2, 3, 4, 5)]


def _md(text):
    """LaTeX +/- -> the markdown spelling (`md` in main() is nested; this is the same map)."""
    return text.replace("$\\pm$", "\u00b1")


def multitask_section(logs):
    """-> markdown block for the five-task zero-shot table, or '' if nothing has landed."""
    rows, means, bc_means = [], [], []
    for task, pats, bc_pats in MULTITASK:
        txt, meta = cell(logs, pats)
        bc_txt, bc_meta = cell(logs, bc_pats)
        if meta["n_seeds"]:
            means.append(meta["mean"])
        if bc_meta["n_seeds"]:
            bc_means.append(bc_meta["mean"])
        ratio = "--"
        if meta["n_seeds"] and bc_meta["n_seeds"] and bc_meta["mean"] > 0:
            ratio = f"{meta['mean'] / bc_meta['mean']:.1f}x"
        label = f"task {task}" + (" (the default id)" if task == 2 else "")
        txt_md, bc_md = _md(txt), _md(bc_txt)
        rows.append(f"| {label} | {meta['n_seeds']} | {txt_md} | {bc_md} | {ratio} |")
    if not means:
        return ""
    lines = ["\n## Zero-shot across the five cube tasks (affine PSMFlow, 300k-500k)\n",
             "One representation per seed; the reward function is inferred in closed form at "
             "eval (`infer_eval_z`), so all five tasks are read off the SAME checkpoints with "
             "no retraining. Every other cube row in this file is task 2 alone.\n",
             "| task | n (ckpt x seed) | success, mean ± 95% CI | BC control | ratio vs BC |",
             "|---|---|---|---|---|"]
    lines += rows
    avg = f"{sum(means) / len(means):.3f}" if means else "--"
    bavg = f"{sum(bc_means) / len(bc_means):.3f}" if bc_means else "--"
    lines.append(f"| **average over {len(means)} task(s)** | -- | **{avg}** | **{bavg}** | "
                 + ("--" if not bc_means or float(bavg) <= 0
                    else f"**{float(avg) / float(bavg):.1f}x**") + " |")
    lines.append("\nFull per-checkpoint breakdown and the figure: "
                 "`docs/tables/affine_cube_multitask.md`, "
                 "`docs/figures/2026-09-06-affine-cube-multitask.png` "
                 "(`tools/fig_affine_multitask.py`).")
    return "\n".join(lines) + "\n"


def e4_section(logs):
    """-> markdown block for E4a (fql-critic scorer) + E4b (mixture probes), or ''."""
    p = os.path.join(logs, "e4a_fql_critic_aim_cube_sd0.json")
    if not os.path.exists(p):
        return ""
    with open(p) as f:
        d = json.load(f)
    r = d["arms"]["fql_critic_aim"]
    rho = d["spearman_vs_oracle"]
    lines = ["\n## E4a: FQL-critic-as-scorer (same K=512 candidates + ODE decode as E1)\n",
             f"Success {r['success']:.3f} [{r['wilson95'][0]:.3f}, {r['wilson95'][1]:.3f}] "
             f"— below the one-step random floor (0.086). Per-step Spearman vs the oracle "
             f"ranking: mean {rho['mean']:.3f}, median {rho['median']:.3f}, "
             f"{rho['frac_above_0.3']:.0%} of steps above 0.3 — the expert's critic ranks "
             f"moderately, but argmax over 512 candidates picks "
             f"{d['picked_dist_to_expert']['mean']:.3f} from the expert action when "
             f"{d['best_available_dist']['mean']:.3f} was available. Verdict: "
             f"{d['fork_branch']}."]
    p2 = os.path.join(logs, "d1a_latent_ranking_mixhpo_ep500k.json")
    if os.path.exists(p2):
        with open(p2) as f:
            m = json.load(f)
        lines.append(f"\nE4b (mixture-trained checkpoint, 500k): ranking Spearman "
                     f"{m['spearman_mean']:.3f}, Q spread 0.86% of |Q| — same band as the "
                     f"point arm (0.10 / 1.1%) and Arm B (0.079 / 0.9%). The mixture does "
                     f"not create ranking signal.")
    return "\n".join(lines) + "\n"


def e1_section(logs):
    """-> markdown block for the E1 oracle-aim report, or '' if absent."""
    path = os.path.join(logs, "e1_oracle_aim_cube_sd0.json")
    if not os.path.exists(path):
        return ""
    with open(path) as f:
        d = json.load(f)
    lines = ["\n## E1 oracle-aim (tools/diag_oracle_aim.py, 500 ep, K=512, ODE-100 decode)\n",
             "| Arm | Success | Wilson 95% |", "|---|---|---|"]
    for arm, r in d.get("arms", {}).items():
        lo, hi = r.get("wilson95", [None, None])
        lines.append(f"| {arm} | {r['success']:.3f} | [{lo:.3f}, {hi:.3f}] |")
    md = d.get("aim_distance", {}).get("min_over_K", {})
    if md:
        lines.append(f"\nMean min-distance to the oracle action over K: {md['mean']:.3f} "
                     f"(p90 {md['p90']:.3f}). Verdict: {d.get('fork_branch', 'n/a')}.")
    return "\n".join(lines) + "\n"

FRACTIONS = ["10\\%", "50\\%", "100\\%"]
FRACTION = [
    ("Behavior flow (BC control)",
     ["eval500_bcflow_frac10.json"], ["eval500_bcflow_frac50.json"], ["eval500_bcflow_cube.json"]),
    ("FQL (per-task)",
     ["eval500_fql_frac10_sd?.json"], ["eval500_fql_frac50_sd?.json"], ["eval500_fql_cube_sd?.json"]),
    ("FB (zero-shot)",
     ["eval500_fb_frac10_sd?.json"], ["eval500_fb_frac50_sd?.json"], ["eval500_fb_cube_sd?.json"]),
    ("PSMFlow (zero-shot)",
     ["eval500_psmflow_frac10_sd?.json"], ["eval500_psmflow_frac50_sd?.json"],
     ["eval500_latentpsm_cube_sd?.json"]),
    ("Hybrid, deployed",
     ["eval500_hybrid_frac10_sd?.json"], ["eval500_hybrid_frac50_sd?.json"],
     ["eval500_hybrid_sd?_final.json", "eval500_hybrid_1M_sd?.json"]),
    ("Hybrid, decode-only control",
     ["eval500_hybrid_frac10_decodeonly_sd?.json"], ["eval500_hybrid_frac50_decodeonly_sd?.json"],
     ["eval500_hybrid_decodeonly_sd00?.json", "eval500_hybrid_1M_decodeonly_sd00?.json"]),
    ("Latent RL (per-task)",
     ["eval500_latrl_frac10_sd?.json"], [], ["eval500_l1stab_res0.05_*_peak*.json"]),
]


def _load(logs, patterns):
    """-> [(basename, report)]. Identical evals reached twice count once.

    A row may list the same eval under two names/directories (a control copied between
    machines). Two files agreeing on env, checkpoint, episode count and success are the
    same measurement, and pooling them would print a 1-seed control as 2 seeds.
    """
    out, seen = [], set()
    for pat in patterns:
        for p in sorted(glob.glob(os.path.join(logs, pat))):
            with open(p) as f:
                d = json.load(f)
            r = d.get("report", d)
            if "success" not in r or r.get("success") is None:
                continue
            key = (r.get("env"), r.get("agent"), str(r.get("restore_path")),
                   r.get("restore_epoch"), r.get("num_episodes"), r.get("num_success"),
                   r.get("success"))
            if key in seen:
                continue
            seen.add(key)
            out.append((os.path.basename(p), r))
    return out


def _t95(n):
    # two-sided 95% t quantiles, n-1 dof; avoids a scipy import for a 6-entry lookup
    return {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447,
            8: 2.365, 9: 2.306, 10: 2.262}.get(n, 1.96)


def cell(logs, patterns):
    """-> (text, dict). Mean +/- 95% CI across seeds, or a single run's Wilson interval."""
    runs = _load(logs, patterns)
    if not runs:
        return "--", {"n_seeds": 0, "files": [], "patterns": patterns}
    vals = [float(r["success"]) for _, r in runs]
    meta = {"n_seeds": len(vals), "files": [f for f, _ in runs], "values": vals,
            "modes": sorted({r.get("acting_mode", "unrecorded") for _, r in runs}),
            "epochs": sorted({r.get("restore_epoch") for _, r in runs})}
    if len(vals) == 1:
        lo, hi = runs[0][1].get("wilson95", [None, None])
        meta.update(mean=vals[0], ci_type="wilson", lo=lo, hi=hi)
        return (f"{vals[0]:.3f} [{lo:.3f}, {hi:.3f}]" if lo is not None
                else f"{vals[0]:.3f}"), meta
    m = sum(vals) / len(vals)
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))
    h = _t95(len(vals)) * sd / math.sqrt(len(vals))
    meta.update(mean=m, ci_type="t95_across_seeds", half_width=h)
    return f"{m:.3f} $\\pm$ {h:.3f}", meta



# ---------------------------------------------------------------------------
# Actor ablation (2026-09-07): actor-free GPI vs the DSRL-style latent actors.
# APPEND-ONLY addition; nothing above is modified. Writes its own
# docs/tables/actor_ablation.md because the rows are pooled over a checkpoint
# WINDOW (300k-500k) x seeds rather than being one glob per row, which the
# HEADLINE/cell() machinery is not shaped for.
# See docs/design/2026-09-06-dsrl-actor-audit.md.
# ---------------------------------------------------------------------------

#: (env label, arm label, glob, BC control). None glob = a literal recorded elsewhere.
ACTOR_ABLATION = [
    ("cube-single-play", "actor-free (GPI)", "eval500_affine*_strict_cube_sd?.json", 0.072),
    ("cube-single-play", "dsrl_na", "eval500_affine*_dsrlna_cube_sd*.json", 0.072),
    ("cube-single-play", "dsrl_sac", "eval500_affine*_dsrlsac_cube_sd*.json", 0.072),
    ("cube-single-play", "dsrl_sac (corrected target_entropy)",
     "eval500_affine*_dsrlsacte_cube_sd*.json", 0.072),
    ("antmaze-medium g=0.98", "actor-free (GPI)",
     "eval500_affine*_strict_antmaze_sd?.json", 0.072),
    ("antmaze-medium g=0.98", "dsrl_sac",
     "eval500_affine500k_dsrlsac_antmaze_sd?.json", 0.072),
    ("antmaze-medium g=0.99", "actor-free (GPI)",
     "eval500_affine*_strict_antmaze_g99_sd?.json", 0.072),
    ("antmaze-medium g=0.99", "dsrl_na + advantage weighting",
     "eval500_affine*_dsrlnaadv_antmaze_g99_sd?.json", 0.072),
    ("antmaze-medium g=0.99", "dsrl_sac",
     "eval500_affine*_dsrlsac_antmaze_g99_sd?.json", 0.072),
    ("pointmaze-medium", "actor-free (GPI)",
     "eval500_affine*_strict_pointmaze_sd?.json", 0.002),
    ("pointmaze-medium", "dsrl_sac",
     "eval500_affine500k_dsrlsac_pointmaze_sd?.json", 0.002),
]

#: Rows whose number does not come from a 300k-500k glob in this logs dir.
ACTOR_ABLATION_LITERALS = [
    ("cube-single-play", "ddpg latent actor (pre-existing arm)", "0.146", "2", "0.130-0.162",
     0.072),
    ("cube-single-play", "prior_shrunk (critic-free control)", "0.059 [0.048, 0.072]",
     "1500 ep", "-", 0.072),
]

ACTOR_WINDOW = (300000, 500000)


def _actor_ablation_agg(logs, pattern, lo, hi):
    """Pooled mean +/- 95% CI over every eval JSON in the checkpoint window."""
    vals = []
    for path in glob.glob(os.path.join(logs, pattern)):
        try:
            with open(path) as f:
                d = json.load(f)
            if lo <= int(d["restore_epoch"]) <= hi:
                vals.append(float(d["success"]))
        except (OSError, ValueError, KeyError):
            continue
    if not vals:
        return None
    n = len(vals)
    mean = sum(vals) / n
    if n > 1:
        sd = (sum((v - mean) ** 2 for v in vals) / (n - 1)) ** 0.5
        ci = 1.96 * sd / math.sqrt(n)
    else:
        ci = 0.0
    return n, mean, ci, min(vals), max(vals)


def actor_ablation_table(logs):
    """Write docs/tables/actor_ablation.md and return it as a markdown string."""
    lo, hi = ACTOR_WINDOW
    out = [
        "# Actor ablation: actor-free GPI vs DSRL-style latent actors",
        "",
        "Generated by `tools/make_tables.py`. **Do not hand-edit.**",
        "",
        "500-episode evaluations (`tools/eval_checkpoint.py`, `EVAL_WORKERS=1`), pooled over",
        f"the **{lo//1000}k-{hi//1000}k checkpoints x 3 training seeds**, mean +/- 95% CI.",
        "Never a peak, never a single seed: see docs/design/2026-09-06-dsrl-actor-audit.md",
        "§4g for two errors made by reading partial ladders while these very numbers landed.",
        "",
        "Substrate identical across arms (`psi_form=affine policy_index=latent`,",
        "`index_agg=max`, `u_clip=3.0`, same frozen flow and preimages); the arms differ only",
        "in the actor and the acting rule.",
        "",
        "| env | arm | late mean +/- 95% CI | n | range | BC |",
        "|---|---|---|---|---|---|",
    ]
    for env, arm, pat, bc in ACTOR_ABLATION:
        r = _actor_ablation_agg(logs, pat, lo, hi)
        if r is None:
            out.append(f"| {env} | {arm} | (not yet evaluated) | - | - | {bc:.3f} |")
        else:
            n, m, ci, mn, mx = r
            out.append(f"| {env} | {arm} | {m:.3f} +/- {ci:.3f} | {n} | "
                       f"{mn:.3f}-{mx:.3f} | {bc:.3f} |")
    for env, arm, val, n, rng, bc in ACTOR_ABLATION_LITERALS:
        out.append(f"| {env} | {arm} | {val} | {n} | {rng} | {bc:.3f} |")
    out += [
        "",
        "Reading notes (full argument in the design doc):",
        "",
        "- `dsrl_na` reaches ~74% of the actor-free ceiling on cube and more than doubles the",
        "  previous best amortized actor, but its spread is LARGER, not smaller (per-seed",
        "  means 0.205 / 0.237 / 0.479). The reason to prefer an amortized actor over",
        "  per-step GPI was reproducibility; it does not deliver it.",
        "- `prior_shrunk` is the critic-free control: one prior draw scaled to the NA actor's",
        "  own operating radius, reading neither psi nor task_z (verified: identical success",
        "  count at 300k and 500k). At BC, so `dsrl_na`'s value is genuine state-dependent",
        "  behaviour, not 'shrink your latents'.",
        "- The plain `dsrl_sac` rows measure a hyperparameter PORT, not DSRL-SAC:",
        "  `target_entropy=0` is DSRL's value for a noise box of b=1.5 and ours is u_clip=3.0,",
        "  which leaves the actor at ~2.0x the prior radius on all three envs. The corrected",
        "  row is the one to read once it lands.",
        "- pointmaze is a null for every arm including actor-free; it discriminates nothing.",
        "- antmaze g=0.98 is a broken baseline (actor-free below its own BC control); the",
        "  g=0.99 block is the live comparison.",
        "",
    ]
    md = "\n".join(out)
    os.makedirs(os.path.join(REPO, "docs/tables"), exist_ok=True)
    with open(os.path.join(REPO, "docs/tables/actor_ablation.md"), "w") as f:
        f.write(md)
    return md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default=LOGS)
    args = ap.parse_args()
    logs = args.logs
    missing, data = [], {"headline": {}, "fraction": {}}

    head_rows = []
    for label, kind, pats in HEADLINE:
        txt, meta = cell(logs, pats)
        data["headline"][label] = {**meta, "setting": kind}
        if meta["n_seeds"] == 0:
            missing.append({"table": "headline", "row": label, "patterns": pats})
        head_rows.append((label, kind, txt, meta["n_seeds"]))

    frac_rows = []
    for label, *cols in FRACTION:
        texts, metas = [], []
        for frac, pats in zip(FRACTIONS, cols):
            txt, meta = cell(logs, pats)
            texts.append(txt)
            metas.append(meta)
            data["fraction"].setdefault(label, {})[frac.replace("\\", "")] = meta
            if meta["n_seeds"] == 0 and pats:
                missing.append({"table": "fraction", "row": label, "col": frac,
                                "patterns": pats})
        frac_rows.append((label, texts, metas))

    os.makedirs(os.path.join(REPO, "PAPER/ICLR/tables"), exist_ok=True)
    os.makedirs(os.path.join(REPO, "docs/tables"), exist_ok=True)

    # 2026-09-07 actor ablation. Appended; writes its own file and does not touch the
    # headline/fraction tables above.
    actor_ablation_table(logs)

    with open(os.path.join(REPO, "PAPER/ICLR/tables/table_headline.tex"), "w") as f:
        f.write("% generated by tools/make_tables.py -- do not edit by hand\n")
        f.write("\\begin{tabular}{llr}\n\\toprule\nMethod & Setting & "
                "Success (500 ep) \\\\\n\\midrule\n")
        for label, kind, txt, n in head_rows:
            f.write(f"{label} & {kind} & {txt} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    with open(os.path.join(REPO, "PAPER/ICLR/tables/table_fraction.tex"), "w") as f:
        f.write("% generated by tools/make_tables.py -- do not edit by hand\n")
        f.write("\\begin{tabular}{lrrr}\n\\toprule\nMethod & " +
                " & ".join(FRACTIONS) + " \\\\\n\\midrule\n")
        for label, texts, _ in frac_rows:
            f.write(f"{label} & " + " & ".join(texts) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    def md(t):
        return t.replace("$\\pm$", "±")

    with open(os.path.join(REPO, "docs/tables/results.md"), "w") as f:
        f.write("# Results tables (generated by `tools/make_tables.py`)\n\n"
                "cube-single-play-singletask-v0 -- which is OGBench **task 2**, not all "
                "five: the bare id has no task token and `cube_env.py` defaults "
                "`reward_task_id` to 2, so every row below except the five-task section is "
                "task 2 alone (antmaze/pointmaze rows are task 1, the mazes' default). "
                "500-episode evals. Multiple seeds are "
                "mean ± 95% CI across seeds; a single seed shows its Wilson interval.\n\n"
                "## Headline\n\n| Method | Setting | Success (500 ep) | seeds |\n"
                "|---|---|---|---|\n")
        for label, kind, txt, n in head_rows:
            f.write(f"| {label} | {kind} | {md(txt)} | {n} |\n")
        f.write("\n## Data fraction\n\n| Method | " + " | ".join(
            x.replace("\\", "") for x in FRACTIONS) + " |\n|---|---|---|---|\n")
        for label, texts, _ in frac_rows:
            f.write(f"| {label} | " + " | ".join(md(t) for t in texts) + " |\n")
        f.write(multitask_section(logs))
        f.write(e1_section(logs))
        f.write(e4_section(logs))
        f.write("\n## Provenance notes\n\n")
        for note in NOTES:
            f.write(f"- {note}\n")
        if missing:
            f.write("\n## Cells with no data yet\n\n")
            for m in missing:
                f.write(f"- {m['table']}: {m['row']}"
                        + (f" @ {m['col']}" if "col" in m else "")
                        + f"  (`{'`, `'.join(m['patterns'])}`)\n")

    data["missing"] = missing
    with open(os.path.join(logs, "table_dataset_fraction.json"), "w") as f:
        json.dump(data, f, indent=2)

    print(open(os.path.join(REPO, "docs/tables/results.md")).read())
    print(f"\n{len(missing)} empty cell(s); wrote PAPER/ICLR/tables/*.tex, "
          f"docs/tables/results.md, {logs}/table_dataset_fraction.json")


if __name__ == "__main__":
    main()

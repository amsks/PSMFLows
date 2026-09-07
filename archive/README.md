# archive/ — frozen, not deleted

**2026-09-04.** The paper-strict **affine LatentFlowPSM** became THE algorithm of this
repo: `agent=psmflow` now defaults to `psi_form=affine policy_index=latent
train_actor=false acting=gpi` (cube-single-play @250k, 500 episodes: 0.532 / 0.620 over two
seeds, against 0.083 for the free-psi arm, 0.230 for the latent-actor agent and 0.072 BC).
Everything that is not on the `fql` -> preimages -> `psmflow` path moved here.

**Ignore this tree.** It is excluded from `ruff check .` and from `pytest` collection
(`pyproject.toml`). Nothing in the live tree imports it. It is kept because these are the
*negative results of record*: the compendium cites them, and deleting them invites
re-running settled experiments.

**To revive something:** move the agent back to `agents/`, its yaml back to
`configs/agent/`, its tests back to `tests/`, re-add it to `agents/__init__.py`, and
re-point the `archive.*` imports (see "one edit was needed" below). Layout mirrors the
original tree, so every move is the inverse of a `git mv`.

## One edit was needed on the way in

`agents/psm.py` was never only an agent: `agents/psmflow.py` imported seven pure helpers
from it. Those helpers (`contrastive_loss`, `ortho_loss`, `targets_uncertainty`,
`proto_sample`, `project_z`, `off_diagonal_mask`, `polyak_update`, `_plain_config`,
`_HashableDict`, `_step`, `_soft`) now live in **`utils/psm_common.py`**, in the live tree.
`archive/agents/psm.py` re-exports them so it reads as it always did; the other archived
agents import them from `utils.psm_common` directly. `utils/fb_networks.py` and
`utils/torch_to_flax.py` came here too — their only consumers were archived.

## agents/ (+ the matching `configs/agent/*.yaml`)

| what | why |
|---|---|
| `psm.py` | PSM (arXiv 2411.19418) JAX port. Peer baseline; the parity hunt is CLOSED (HANDOFF 07-13). |
| `affine_psm.py` | Full affine/LP measure PSM. Cube push PARKED 07-26, never resumed. |
| `latent_affine_psm.py` | `affine_psm` over flow latents. Subsumed by `psmflow`. |
| `latentrl.py` | Per-task latent probe / offline DSRL-SAC arm. Result settled negative (09-03 plan). |
| `fb.py` | Forward-Backward. The strongest comparator (cube 0.721) — see the note below. |
| `ifql.py`, `iql.py`, `rebrac.py`, `sac.py` | Off-the-shelf baselines inherited from the FQL codebase; never part of this method. |

**On `fb`.** Archived, not kept. `tools/make_tables.py` is pure JSON I/O (`glob` + `json`,
no `subprocess`, no agent import): the FB rows come from `eval500_fb_*.json` files already
on disk, so the tables keep printing 0.721 with `agents/fb.py` gone. Re-running FB is the
only thing that needs it, and that means un-archiving it.

## utils/

| what | why |
|---|---|
| `fb_networks.py` | `ForwardMap`/`BackwardMap`/`FBTd3Actor`; only `agents/fb.py` used them. |
| `torch_to_flax.py` | Torch->Flax weight loader for the PSM/FB parity fixtures only. |

## tools/

Diagnostics for hypotheses the compendium records as **settled negative**, plus everything
built on an archived agent and the dropped ICLR F1-F4 figure set.

| what | why |
|---|---|
| `diag_oracle_aim.py`, `diag_action_coverage.py`, `diag_geometry_robustness.py` | E1/W1/L0 "the action interface is the ceiling" — refuted (oracle aim 0.934). |
| `diag_q_landscape.py`, `diag_policy_ranking.py`, `diag_task_projection.py` | D1-D3 critic forensics on the free head; superseded by `diag_latent_ranking_oracle.py`. |
| `diag_flow_jacobian.py` | Local-injectivity hypothesis, refuted (COMPENDIUM §4.8). |
| `diag_preimage_posterior_width.py`, `diag_preimage_sampling_fidelity.py` | "The mixture preimage is a set" — refuted on all three envs. |
| `diag_eps_ball_fit.py` | Superseded two minutes after it landed by `diag_flow_fit.py`. |
| `diag_fb_basis_probe.py`, `diag_policy_differentiation.py` | FB-specific probes. |
| `diag_action_distance.py`, `diag_fql_calibration.py`, `diag_generated_pair_support.py` | `agent=latentrl` support/collapse forensics. |
| `calibration_check.py`, `eval_fixed_u_rollouts.py`, `latent_reachability.py`, `viz_fixed_u_field.py` | The fixed-`u` policy family — settled negative (0/233 latents reach the goal). |
| `latent_q_sanity.py`, `viz_latent_value.py`, `viz_policy_rollouts.py` | Superseded by `eval_checkpoint.py` / `diag_latent_smoothness.py`. |
| `fig_reachability.py`, `fig_flow_fit.py`, `fig_actor_comparison.py`, `fig_policy_anatomy.py` | ICLR F1-F4; that figure set is dropped (`fig_diag_flow_fit.py` and `fig_flow_fit_overlap.py` replaced F2). |
| `plot_preimage_intuition.py` | Preimage-basin figure; the basin claim it illustrates is refuted. |
| `export_psm_fixture.py`, `export_fb_fixture.py` | Write the torch parity fixtures in `archive/tests/fixtures/`. |

## scripts/

| what | why |
|---|---|
| `launch_psm_cube.sh`, `launch_fb_cube.sh`, `launch_affine_psm_cube.sh`, `launch_baselines.sh` | Launchers for archived agents. |
| `compare_multiseed.py`, `compare_fb_multiseed.py`, `compare_protoxplant.py`, `reeval_checkpoints.py` | Comparisons against the PSM/FB reference curves. |
| `reeval_ode_e2.sh` | E2 ODE-decode re-eval; the effect reversed sign (COMPENDIUM §6). |
| `transplant_eval.py` | Imports `PSMAgent` against a pre-refactor API; already stale before the move. |

## tests/

The suites of the archived agents, moved with them: `test_psm_*`, `test_fb_*`,
`test_affine_psm_*`, `test_latent_affine_psm.py`, `test_latentrl_smoke.py`, plus
`fixtures/psm_reference.npz` and `fixtures/fb_reference.npz` (torch parity fixtures whose
only consumers were `test_psm_networks_equiv.py` / `test_fb_networks_equiv.py`).

## docs/

`plans/` and `design/` for the archived agents (FB port, PSM refactor, affine PSM) and the
plan queue whose experiments are now settled: the ICLR F1-F4 figure plan, the
critic-diagnosis / PSM-fix / boost-lever / priority-stack roadmaps, the E1-E3 interface
fork, and the latentrl DSRL-SAC plan. `audits/2026-08-12-psmflow-fb-hp-diff.md` —
"HP-matching psmflow to FB helps" is a settled no.

Still live in `docs/`: `COMPENDIUM.md`, `HANDOFF.md`, `PREIMAGES.md`, `TUNING_INVERSION.md`,
`README-fql.md`, the psmflow v1 plan + design, the formal-writeup design, the two 09-03
audits, the 09-04 affine-psi design, and the 09-03 latent-coherence plan (S2/S3 still live).

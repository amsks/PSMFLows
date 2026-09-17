# Affine action-conditioning implementation plan

Goal: implement and evaluate one default-off change to affine PSMFlow, preserving existing
dirty-tree work and old checkpoints. No training outcome is guaranteed by implementation.

Spec: [design](../design/2026-09-11-affine-action-conditioning.md).
Stack: existing JAX/Flax/Hydra, pytest, Slurm; no new dependencies.

- [x] Add meaningful failing tests in `tests/test_psmflow_action_input.py`: raw-action fit,
  latent-equivalence under a controlled decoder, factored/explicit scores, frozen flow,
  finite update and old-config default.
- [x] Add `measure_action_input` default/backfill/validation in `agents/psmflow.py` and
  `configs/agent/psmflow.yaml`. Require affine + latent policy index in action mode and
  reject `measure_u_samples != 1` there. Keep default behavior unchanged.
- [x] Route all bounded psi queries through decoded actions in action mode; the online
  measure fit uses `batch['actions']` directly. Adapt the affine `sa_terms` fast path too.
- [x] Run targeted regression tests, inspect the diff and get an independent review.
- [x] Add `tools/report_seed_comparison.py` with strict report provenance, complete task matrices,
  and across-training-seed Student-t intervals; focused tests pass.
- [x] Prepare immutable experiment snapshot, hyperparameter manifest and exact Slurm smoke
  commands under workspace outputs. Do not alter existing jobs or checkpoints.
- [x] Run two 200-step production smokes and re-read saved flags. Only on success advance
  to the three-seed, 500k experiments and final five-task evaluation.
- [x] Complete the six training jobs and their final five-task evaluations.
- [x] Persist raw reports and across-seed Student-t intervals; add dated handoff evidence.

Progress: implementation and independent review complete. Tests: 9 action/parity, 11 config,
12 legacy affine, 18 core passed (2 checkpoint-dependent skips), and 5 reporter tests passed.
Both 200-step GPU smokes completed with exit 0 and verified saved configurations.
Training jobs2493455–2493460 and aggregate job2493461 completed0:0.
Artifacts: `outputs/affine_action_20260911/`; code snapshot has 159 SHA-256 entries.
All30 final reports validated. Cube43.92% ±38.02points; AntMaze1.08% ±1.46points,
across-three-seed95% t intervals of five-task means. The experiment is complete; the
requested performance target was not met. The freeze-phi diagnostic is the next study.

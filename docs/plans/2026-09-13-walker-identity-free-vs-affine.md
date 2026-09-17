# Walker identity-decoder control: free vs affine psi, raw actions

Date: 2026-09-13. Branch `feat/inversion-integration`, source HEAD `daa0268`.
This is gate 3 of the isolation protocol in
[`docs/design/2026-09-13-psm-interface-audit.md`](../design/2026-09-13-psm-interface-audit.md):
"matched free/affine control without a learned flow".

## Question

The affine PSMFlow reaches 0.532/0.620 on cube and 0.294 on antmaze against a BC control of
0.072, and the failure is measured as latent *selection*: oracle-aim reaches 0.934 on cube
(`COMPENDIUM` §4.3) and **0.966 on antmaze** (`HANDOFF` 2026-09-13), so the reachable action
set contains near-expert behaviour at nearly every step and the critic cannot find it.

On 2026-09-13 the archived raw-action PSM reproduced the published Walker average:
**694.13 ± 62.03** against Table 1's 689.07. That run differs from the failing runs in four
ways at once — agent, action space, data type and reward density — so it does not say which
one carries the failure.

This experiment removes two of the four. Same ExORL Walker RND data, same dense rewards,
same 2M budget, but our own PSMFlow measure objective, and the decoder replaced by the
identity so no flow is involved. Only the psi head differs between the two arms.

## What is held identical between the arms

| component | value |
|---|---|
| data | ExORL Walker RND, 5000 episodes, 5M transitions, the validated `outputs/walker_psm_20260913/data/` |
| decoder | identity: `a = u`, so the action latent IS the action |
| action prior `p0` | uniform on `[-1, 1]^6`, the Walker action box; the same draw feeds training and acting |
| policy index family | `u' ~ p0`, one per row, the same bounded prior |
| bootstrap | `psi_bar(s', u', u')^T phi_bar(s'_j)`, i.e. `u_next = u_index`, matching `policy_index=latent` |
| ensemble reduction | `mean - 0.5 * spread` over `P=2`, exact min |
| losses | `utils.psm_common.contrastive_loss` + `ortho_coef * ortho_loss`, the live helpers |
| acting | GPI argmax over `K=64` actions x `K=64` indices, redrawn every step |
| reward inference | `w = E[r phi]` over 10k relabel rows, per task |
| budget | B=1024, z=128, gamma=0.98, tau=0.01, ortho_coef=1, all LR 1e-4, 2M updates |
| seeds | 0, 1, 2 |
| tasks | stand, walk, run, flip |

`ortho_coef=1` follows the audit's declared Walker match, not the cube default of 1000.

**The only difference:** `psi_form=free` builds `PsiMap`, `psi_form=affine` builds
`AffinePsiMap` with `psi(s,u,u') = A(s,u)^T w(u') + beta(s,u)` and `norm_w=true`. Both come
from the live `utils/psm_networks.py`, unmodified.

## What this can and cannot show

It isolates the psi factorization and the presence of a learned flow, holding data and
reward fixed. It does **not** isolate the data type (ExORL RND is exploratory, OGBench play
is behaviour data) or the reward density. Comparing either arm against the archived PSM's
694 does not isolate factorization either, because that agent's policy family and feature
extractor also differ — the audit says so explicitly, and this plan does not claim otherwise.

## Pre-registered outcomes

Stated before any number exists.

1. **Both arms clear the random floor.** A uniform action policy on Walker scores well under
   200 on the task average. If an arm lands there, the port is broken and the comparison is
   void, not informative.
2. **Expected result: both arms land well below 694.** The GPI acting rule scans 64 prior
   draws per step, and on cube and antmaze that rule is what fails. If it fails here too,
   with no flow in the path and with dense reward, the failure is not the flow interface.
3. **Affine is expected to be at or below free.** On cube the affine head measured 0.532
   against free psi's 0.083, so affine is not expected to be the weaker arm; but the audit's
   counterexample — independent training slots, diagonal coupling at bootstrap — applies to
   both heads, so no separation between them is the most likely outcome.
4. **The informative case is either arm approaching 694.** That would mean the measure
   objective plus GPI works on this substrate, and would put the OGBench failure on the data
   type or the flow, not on the objective.

Failure of any pre-registered expectation is recorded as such; none of them is a gate that
cancels the run.

## Cost and the eval-budget decision

Walker episodes are exactly 1000 steps. Four tasks x 500 episodes is 2M environment steps
per evaluation, about 20x a cube eval500. GPI with `K=64` costs 4096 psi rows per step for
the free head; the affine head computes `A`, `beta` and `w(u')` once and finishes in an
einsum, so it is far cheaper at acting. Two arms cannot share an episode budget that is
chosen before this is measured.

**Decision rule, fixed here:** the smoke measures acting throughput for both heads. The
final evaluation uses the largest episode count per task, equal across both arms, that fits
an 8 h job at the measured rate, capped at 500. The chosen number is recorded in the run's
own manifest before the production jobs are submitted. Periodic in-training evals use 10
episodes per task, matching the sibling campaign.

## Protocol

1. Snapshot the agent, harness and the live `utils/` into
   `outputs/walker_identity_20260913/code/`, with a source manifest and SHA256, as the
   sibling campaign did. Nothing is added to the live agent registry.
2. Smoke: 200 updates, both heads, 2 episodes per task, on GPU. Record updates/s, acting
   rate, finite losses, and the restored-checkpoint eval path.
3. Verify the smoke's own `flags.json` carries the intended settings before production.
4. Production: 6 jobs, 2 heads x 3 seeds, 2M updates, checkpoints at 50k/100k/250k/500k/1M/2M.
5. Report: average tasks within a seed, then mean and 95% Student-t interval across the three
   seeds, the same aggregation as the PSM campaign, so the two are comparable.

## Files

- Campaign: `outputs/walker_identity_20260913/`
- Agent: `outputs/walker_identity_20260913/code/identity_psm_agent.py`
- Harness: reused from `outputs/walker_psm_20260913/code_v2/tools/walker_psm/`, with the
  agent construction and the acting call swapped
- Report: `outputs/walker_identity_20260913/aggregate.json`

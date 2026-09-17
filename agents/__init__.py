"""Agent registry — what `main.py agent=<name>` and `tools/eval_checkpoint.py` can build.

Three agents, one pipeline. `fql` is Stage A (the behaviour flow `G(s,u)` fit by flow
matching, and its inverse: `compute_full_proposal_distribution_em` is Stage B's core) and
also the BC control every Stage-C number is quoted beside. `f_psmflow` (renamed from
`psmflow` on 2026-09-17) is the method: the paper-strict affine / factorized LatentFlowPSM,
primary since 2026-09-04; `psmflow` remains a backward-compat alias to the same class, so
existing checkpoints (`flags.json` records `agent_name: psmflow`) and `agent=psmflow`
scripts keep working. `psmgoal` (2026-09-17) is the goal-indexed measure of
docs/design/2026-09-17-psmgoal.md, on the same frozen flow and preimage dataset.

Every other agent this repo has carried (psm, affine_psm, latent_affine_psm, latentrl, fb,
ifql, iql, rebrac, sac) moved to `archive/agents/` on 2026-09-04 and is deliberately NOT
importable here -- `agent=fb` etc. now fail at the hydra config group, which is the point.
See `archive/README.md` for what each one was and how to revive it.
"""

from agents.fql import FQLAgent
from agents.f_psmflow import PSMFlowAgent
from agents.psmgoal import PSMGoalAgent

agents = dict(
    fql=FQLAgent,
    f_psmflow=PSMFlowAgent,
    psmflow=PSMFlowAgent,  # backward-compat alias (renamed to f_psmflow on 2026-09-17)
    psmgoal=PSMGoalAgent,
)

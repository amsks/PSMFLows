"""Agent registry — what `main.py agent=<name>` and `tools/eval_checkpoint.py` can build.

Three agents, one pipeline. `fql` is Stage A (the behaviour flow `G(s,u)` fit by flow
matching, and its inverse: `compute_full_proposal_distribution_em` is Stage B's core) and
also the BC control every Stage-C number is quoted beside. `psmgoal` is the method (main
PSMFlows implementation since 2026-09-22): the goal-indexed measure
M(s,u,s+) = phi(s,u,s+)^T w + b(s,u,s+) with Lagrangian coefficient inference, on the frozen
flow and preimage dataset (docs/design/2026-09-17-psmgoal.md). `f_psmflow` (renamed from
`psmflow` on 2026-09-17) is the earlier paper-strict affine LatentFlowPSM, kept as the
comparator; `psmflow` remains a backward-compat alias to the same class, so existing
checkpoints (`flags.json` records `agent_name: psmflow`) and `agent=psmflow` scripts keep
working.

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

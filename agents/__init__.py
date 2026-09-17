"""Agent registry — what `main.py agent=<name>` and `tools/eval_checkpoint.py` can build.

Two agents, one pipeline. `fql` is Stage A (the behaviour flow `G(s,u)` fit by flow
matching, and its inverse: `compute_full_proposal_distribution_em` is Stage B's core) and
also the BC control every Stage-C number is quoted beside. `psmflow` is the method: the
paper-strict affine LatentFlowPSM, primary since 2026-09-04. `psmgoal` (2026-09-17) is the
goal-indexed affine measure of docs/design/2026-09-17-psmgoal.md, on the same frozen flow
and preimage dataset.

Every other agent this repo has carried (psm, affine_psm, latent_affine_psm, latentrl, fb,
ifql, iql, rebrac, sac) moved to `archive/agents/` on 2026-09-04 and is deliberately NOT
importable here -- `agent=fb` etc. now fail at the hydra config group, which is the point.
See `archive/README.md` for what each one was and how to revive it.
"""

from agents.fql import FQLAgent
from agents.psmflow import PSMFlowAgent
from agents.psmgoal import PSMGoalAgent

agents = dict(
    fql=FQLAgent,
    psmflow=PSMFlowAgent,
    psmgoal=PSMGoalAgent,
)

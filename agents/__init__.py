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

Every other agent this repo has carried (psm, affine_psm, latent_affine_psm, latentrl,
ifql, iql, rebrac, sac) moved to `archive/agents/` on 2026-09-04 and is deliberately NOT
importable here. `fb` was brought back on 2026-11-02 as the raw-action baseline, with
`fb_latent` its port onto the frozen flow's latents. See `archive/README.md` for the rest.
"""

from agents.f_psmflow import PSMFlowAgent
from agents.density_fb_latent import DensityFBLatentAgent
from agents.fb import FBAgent
from agents.fb_latent import FBLatentAgent
from agents.fql import FQLAgent
from agents.psmgoal import PSMGoalAgent

agents = dict(
    fql=FQLAgent,
    f_psmflow=PSMFlowAgent,
    psmflow=PSMFlowAgent,  # backward-compat alias (renamed to f_psmflow on 2026-09-17)
    # f_psmgoal2p (2026-09-21): the same PSMFlowAgent run 2P-DSRL, but with the frozen phi
    # replaced by a state feature projected from a psmgoal RLUMeasure basis. It differs from
    # f_psmflow only by config (basis_restore_path + the 2P-DSRL flags); see
    # configs/agent/f_psmgoal2p.yaml and docs/design/2026-09-21-psmgoal-2p-dsrl.md.
    f_psmgoal2p=PSMFlowAgent,
    psmgoal=PSMGoalAgent,
    # psmz (2026-11-10): the z-conditioned PSM, the same PSMGoalAgent class with configs/agent/psmz.yaml
    # (factorised measure, reward-sample ridge index, FlowBC latent actor at bc 0.1). psmgoal is the
    # goal-conditioned one. Cube-single actor alone 0.712 at 500k, 3 seeds (docs/HANDOFF.md 2026-11-09).
    psmz=PSMGoalAgent,
    # fb (2026-11-02): the archived raw-action Forward-Backward agent restored as the baseline
    # (cube-single 0.721 under its own flow actor). fb_latent: the same FB on the frozen flow's
    # latent u in place of the action, with the Density FB seams (docs/design/2026-11-02-fb-latent-port.md).
    fb=FBAgent,
    fb_latent=FBLatentAgent,
    # density_fb_latent (2026-11-02): the paper's Density FB softmax measure + FlowBC actor with the
    # action replaced by the flow latent (docs/design/2026-11-02-density-fb-latent-port.md).
    density_fb_latent=DensityFBLatentAgent,
)

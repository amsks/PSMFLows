"""agents/psm/flow_psm/flow_inversion.py — flow inversion a -> u0 (SCAFFOLD STUB).

The PSMFlows policy class is indexed by the behavior flow's initial noise u0:
Pi = {G_theta(s, u0) | u0 ~ p0}. Learning phi(s, u0, u0') over this class requires
recovering u0 from dataset (s, a). This is the deterministic inverse of the flow
G_theta(s, .).

OPEN DESIGN POINTS (not implemented here):
  - A single deterministic inverse yields ONE u0 per action. We actually want the
    SET of u0 that map to ~the same action, so the real implementation will need
    added noise and/or a forward-correction pass.
  - Inversion technique reference: the inversion method cited in the PSMFlows note.
"""

from __future__ import annotations

import torch


def invert_flow(model, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """Return u0 such that G_theta(s, u0) ~= a. NOT IMPLEMENTED (scaffold)."""
    raise NotImplementedError(
        "flow inversion (a -> u0) is not implemented in the FlowPSM scaffold; "
        "see module docstring for the intended design."
    )

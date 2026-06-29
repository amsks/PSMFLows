"""tests/test_psm_ortho_coef_live.py — proves PSM's ortho_coef actually scales the
phi orthonormality term in _update_psm (was hardcoded `1 * orth_loss`; made
configurable for the ablation sweep). Self-contained: no baselines reference needed.
"""
import torch

from tests.test_psm_reference_equivalence import _build_agent, _fixed_inputs, B, LR, BETAS, EPS, WD


def _run_update(coef):
    torch.manual_seed(0)              # identical weight init across calls
    agent = _build_agent()
    agent.ortho_coef = coef
    agent.optim_phi = torch.optim.Adam(agent.model.phi.parameters(), lr=LR, betas=BETAS, eps=EPS, weight_decay=WD)
    agent.optim_psm_psi = torch.optim.Adam(agent.model.psm_psi.parameters(), lr=LR, betas=BETAS, eps=EPS, weight_decay=WD)
    obs, action, next_obs, goal, next_action, z_psm, z_sf, discount = _fixed_inputs()
    return agent._update_psm(obs, action, discount, next_obs,
                             torch.arange(B), goal, z_psm, next_action=next_action)


def test_ortho_coef_scales_psm_loss():
    m1 = _run_update(1.0)
    m10 = _run_update(10.0)
    # same weights+inputs => the orthonormality term itself is identical
    assert torch.equal(m1["orth_loss"], m10["orth_loss"])
    # total psm_loss must reflect the coefficient on the ortho term (exact: float64,
    # simple addition of detached scalars => torch.equal holds, matching the style
    # of test_psm_reference_equivalence.py)
    assert torch.equal(m1["psm_loss"],  m1["psm_diag"]  + m1["psm_offdiag"]  + 1.0  * m1["orth_loss"])
    assert torch.equal(m10["psm_loss"], m10["psm_diag"] + m10["psm_offdiag"] + 10.0 * m10["orth_loss"])
    # and the knob actually changes the loss (fails while hardcoded to 1)
    assert not torch.equal(m1["psm_loss"], m10["psm_loss"])

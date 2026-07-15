"""FB agent smoke tests. The bit-exact network/agent equivalence tests live in
test_fb_networks_equiv.py and test_fb_agent_equiv.py; this file holds the fixture
schema check plus (added in later tasks) registration/update/eval smoke tests."""
import numpy as np


def test_fb_fixture_schema():
    fix = np.load("tests/fixtures/fb_reference.npz")
    keys = set(fix.keys())
    for k in ["in__observations", "in__actions", "in__next_observations",
              "in__z_gauss", "in__mix_mask", "in__perm", "in__z_mixed",
              "out__F", "out__B", "out__left_enc", "out__fb_loss", "out__actor_loss",
              "out__actor_mu", "out__vf"]:
        assert k in keys, f"missing {k}"
    for net in ["_forward_map", "_backward_map", "_left_encoder",
                "_target_forward_map", "_target_backward_map", "_target_left_encoder",
                "_actor", "_actor_vf"]:
        assert any(k.startswith(f"w__{net}.") for k in keys), f"no params for {net}"
    for i in range(10):
        assert any(k.startswith(f"step__{i}__") for k in keys), f"no step {i}"
        assert any(k.startswith(f"step_in__{i}__") for k in keys), f"no step_in {i}"
    assert fix["out__F"].ndim == 3 and fix["out__F"].shape[0] == 2  # [P,B,z]

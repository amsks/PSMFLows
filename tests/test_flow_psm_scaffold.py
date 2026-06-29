import math

import gymnasium as gym
import pytest
import torch

from agents.psm.flow_psm.flow_inversion import invert_flow


def test_invert_flow_is_stub():
    with pytest.raises(NotImplementedError):
        invert_flow(None, torch.randn(8, 40), torch.rand(8, 5) * 2 - 1)

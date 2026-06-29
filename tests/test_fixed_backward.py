import torch
from agents.fb.fixed_backward import FixedCubeBackward, CUBE_OBS_SLICE


def test_extracts_scaled_cube_xyz_and_has_no_params():
    b = FixedCubeBackward(cube_slice=CUBE_OBS_SLICE)
    obs = torch.zeros(4, 40)
    obs[:, CUBE_OBS_SLICE] = torch.tensor([1.0, 2.0, 3.0])
    out = b(obs)
    assert out.shape == (4, 3)
    assert torch.allclose(out, torch.tensor([1.0, 2.0, 3.0]).expand(4, 3))
    assert list(b.parameters()) == []   # parameter-free


def test_output_space_is_3():
    b = FixedCubeBackward(cube_slice=CUBE_OBS_SLICE)
    assert b.output_space.shape == (3,)

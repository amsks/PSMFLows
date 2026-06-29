import numpy as np

from data.ogbench import load_transitions


def _write_state_ep(path, T=6, obs_dim=4, act_dim=2):
    np.savez(
        path,
        observation=np.random.rand(T, obs_dim).astype(np.float32),
        action=np.random.rand(T, act_dim).astype(np.float32),
        physics=np.random.rand(T, 3).astype(np.float32),
        discount=np.ones(T, dtype=np.float32),
    )


def _write_pixel_ep(path, T=6, H=8, W=8, C=3, act_dim=2):
    np.savez(
        path,
        pixels=(np.random.rand(T, H, W, C) * 255).astype(np.uint8),
        action=np.random.rand(T, act_dim).astype(np.float32),
        physics=np.random.rand(T, 3).astype(np.float32),
        discount=np.ones(T, dtype=np.float32),
    )


def test_state_path_unchanged(tmp_path):
    f = tmp_path / "ep0.npz"
    _write_state_ep(f, T=6, obs_dim=4)
    s = load_transitions([f])  # default obs_type="state"
    assert s["observation"].dtype == np.float32
    assert s["observation"].shape == (5, 4)
    assert "timestep" not in s


def test_pixel_path_chw_uint8_with_timestep(tmp_path):
    f0, f1 = tmp_path / "ep0.npz", tmp_path / "ep1.npz"
    _write_pixel_ep(f0, T=6, H=8, W=8, C=3)
    _write_pixel_ep(f1, T=4, H=8, W=8, C=3)
    s = load_transitions([f0, f1], obs_type="pixels")
    assert s["observation"].shape == (8, 3, 8, 8)
    assert s["observation"].dtype == np.uint8
    assert s["next"]["observation"].shape == (8, 3, 8, 8)
    np.testing.assert_array_equal(
        s["timestep"], np.array([0, 1, 2, 3, 4, 0, 1, 2], dtype=np.int32)
    )


def test_pixel_missing_key_raises(tmp_path):
    f = tmp_path / "ep0.npz"
    _write_state_ep(f)  # has "observation", not "pixels"
    try:
        load_transitions([f], obs_type="pixels")
        assert False, "expected KeyError-style failure for missing pixels key"
    except (KeyError, ValueError) as e:
        assert "pixels" in str(e)

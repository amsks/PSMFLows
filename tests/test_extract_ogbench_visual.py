import importlib.util
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "extract_ogbench", REPO / "scripts" / "extract_ogbench.py"
)
extract_ogbench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(extract_ogbench)


def _write_monolithic(path: Path, env_name: str, n_eps=2, ep_len=3, h=4, w=4, c=3, act=2):
    """Two episodes; terminals mark the last step of each episode."""
    T = n_eps * ep_len
    if "visual" in env_name:
        observations = (np.random.rand(T, h, w, c) * 255).astype(np.uint8)  # HWC
    else:
        observations = np.random.rand(T, 7).astype(np.float32)
    actions = np.random.rand(T, act).astype(np.float32)
    qpos = np.random.rand(T, 5).astype(np.float32)
    terminals = np.zeros(T, dtype=np.float32)
    for e in range(n_eps):
        terminals[(e + 1) * ep_len - 1] = 1.0
    np.savez(
        path / (env_name + ".npz"),
        observations=observations,
        actions=actions,
        qpos=qpos,
        terminals=terminals,
    )
    return observations, actions


def test_visual_branch_writes_hwc_pixels_and_dummy_obs_to_out_env(tmp_path):
    src = tmp_path / "cache"
    src.mkdir()
    out = tmp_path / "out"
    env = "visual-cube-single-play-v0"
    observations, actions = _write_monolithic(src, env, n_eps=2, ep_len=3, h=4, w=4, c=3)

    extract_ogbench.extract(
        env_name=env,
        output_folder=str(out),
        dataset_path=str(src),
        out_env="cube-single-play-v0",
    )

    buf = out / "cube-single-play-v0" / "buffer"   # --out_env override honored
    files = sorted(buf.glob("episode_*.npz"))
    assert len(files) == 2, f"expected 2 episodes, got {files}"
    assert files[0].name == "episode_000000_2.npz"  # length = end - start = 2

    ep0 = np.load(files[0])
    # episode 0 spans source rows [0, 1, 2] (3 frames)
    assert ep0["observation"].shape == (3, 1)            # dummy zeros, (L+1, 1)
    assert not ep0["observation"].any()
    assert ep0["pixels"].shape == (3, 4, 4, 3)           # HWC, NO moveaxis
    np.testing.assert_array_equal(ep0["pixels"], observations[0:3])
    assert ep0["action"][0].tolist() == [0.0, 0.0]       # action shift applied
    np.testing.assert_array_equal(ep0["action"][1:], actions[0:2])
    assert ep0["physics"].shape == (3, 5)


def test_state_path_unchanged_no_pixels_key(tmp_path):
    src = tmp_path / "cache"
    src.mkdir()
    out = tmp_path / "out"
    env = "cube-single-play-v0"
    observations, _ = _write_monolithic(src, env, n_eps=1, ep_len=4)

    extract_ogbench.extract(env_name=env, output_folder=str(out), dataset_path=str(src))

    buf = out / "cube-single-play-v0" / "buffer"   # default out_env == env_name
    files = sorted(buf.glob("episode_*.npz"))
    assert len(files) == 1
    ep = np.load(files[0])
    assert "pixels" not in ep.files
    assert ep["observation"].shape == (4, 7)
    np.testing.assert_array_equal(ep["observation"], observations[0:4])

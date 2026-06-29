import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from gymnasium.spaces import Box
from hydra import compose, initialize_config_dir

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals import analysis as A
from train import make_agent


def _cfg(domain="cube_single"):
    with initialize_config_dir(config_dir=str(REPO_ROOT / "configs"),
                               version_base="1.3"):
        return compose(config_name="train",
                       overrides=[f"domain={domain}", "device=cpu",
                                  "batch_size=16"])


def test_load_checkpoint_roundtrip(tmp_path):
    cfg = _cfg()
    obs_space = Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)
    a1 = make_agent(cfg, obs_space, action_dim=4)
    ckpt = tmp_path / "step_10.pt"
    torch.save(a1.state_dict(), ckpt)

    a2 = make_agent(cfg, obs_space, action_dim=4)
    A.load_checkpoint(a2, ckpt, map_location="cpu")
    # a parameter tensor should now match
    p1 = next(iter(a1.state_dict()["model"].values()))
    p2 = next(iter(a2.state_dict()["model"].values()))
    assert torch.allclose(p1, p2)


def test_load_checkpoint_clear_error_on_mismatch(tmp_path):
    junk = tmp_path / "junk.pt"
    torch.save({"model": {"not_a_real_key": torch.zeros(1)}}, junk)
    cfg = _cfg()
    obs_space = Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)
    agent = make_agent(cfg, obs_space, action_dim=4)
    with pytest.raises(RuntimeError, match="checkpoint does not match"):
        A.load_checkpoint(agent, junk, map_location="cpu")


class _StubEnv:
    """Episode of fixed length 3, then done."""
    def __init__(self):
        self.t = 0

    def reset(self):
        self.t = 0
        return np.zeros(15, dtype=np.float32), {"success": False}

    def step(self, action):
        self.t += 1
        done = self.t >= 3
        return (np.full(15, self.t, dtype=np.float32), 1.0, done, False,
                {"success": done})

    def render(self):
        return np.zeros((4, 4, 3), dtype=np.uint8)


class _StubAgent:
    device = "cpu"

    def act(self, obs=None, z=None):
        return torch.zeros((1, 4))


def test_rollout_with_trajectory_shapes():
    res = A.rollout_with_trajectory(_StubEnv(), _StubAgent(), num_episodes=2,
                                    z=torch.zeros(4), record=False)
    assert res["observations"].shape == (2, 4, 15)  # T_max = 3 steps + reset
    assert res["actions"].shape == (2, 4, 4)
    assert list(res["lengths"]) == [4, 4]
    assert res["success"].shape == (2,)
    assert bool(res["success"][0]) is True


def test_save_trajectories_writes_npz_and_summary(tmp_path):
    res = A.rollout_with_trajectory(_StubEnv(), _StubAgent(), num_episodes=2,
                                    z=torch.zeros(4), record=False)
    A.save_trajectories(res, task="taskX", out_dir=tmp_path)
    assert (tmp_path / "trajectories" / "taskX.npz").exists()
    assert (tmp_path / "trajectories" / "trajectory_summary.parquet").exists()
    import pandas as pd
    summ = pd.read_parquet(tmp_path / "trajectories" / "trajectory_summary.parquet")
    assert len(summ) == 2
    assert set(["task", "episode", "return", "length", "success"]).issubset(
        summ.columns)


def test_z_probe_cross_task_matrix_shape():
    tasks = ["t1", "t2", "t3"]

    def infer_z(task):
        return torch.zeros(4)

    def make_env(task):
        return _StubEnv()

    df = A.z_probe_cross_task(tasks, _StubAgent(), infer_z, make_env,
                              n_episodes=2)
    assert set(df.columns) == {"env_task", "z_task", "success", "reward"}
    assert len(df) == 9  # 3x3
    assert set(df["env_task"]) == set(tasks)


def test_z_interp_sweeps_alpha():
    def make_env(task):
        return _StubEnv()

    df = A.z_interp(torch.zeros(4), torch.ones(4), _StubAgent(),
                    make_env("t2"), n_alpha=5, n_episodes=1)
    assert len(df) == 5
    assert list(df["alpha"]) == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert {"alpha", "success", "reward"}.issubset(df.columns)

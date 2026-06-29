import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals import phase_probe as pp


def _sig(eff, cube, grip, success, goal=(1.0, 1.0, 0.0), table_z=0.02):
    return {
        "eff": np.asarray(eff, dtype=np.float32),
        "cube": np.asarray(cube, dtype=np.float32),
        "grip": np.asarray(grip, dtype=np.float32),
        "goal": np.asarray(goal, dtype=np.float32),
        "success": bool(success),
        "table_z": float(table_z),
    }


def test_full_success_episode():
    T = 8
    eff = np.zeros((T, 3)); cube = np.zeros((T, 3))
    grip = np.ones(T)            # closed
    cube[:, 2] = 0.10            # lifted well above table_z=0.02
    out = pp.classify_phases(_sig(eff, cube, grip, success=True), pp.Thresholds())
    assert out["reached"] and out["secured"] and out["success"]
    assert out["furthest_phase"] == "success"
    assert out["fail_phase"] == "none"
    assert out["length"] == T


def test_reach_but_no_secure():
    T = 6
    eff = np.zeros((T, 3)); cube = np.zeros((T, 3))
    grip = np.zeros(T)           # never closes
    out = pp.classify_phases(_sig(eff, cube, grip, success=False), pp.Thresholds())
    assert out["reached"] and not out["secured"] and not out["success"]
    assert out["furthest_phase"] == "reached"
    assert out["fail_phase"] == "grasp"


def test_never_reach():
    T = 5
    eff = np.zeros((T, 3)); cube = np.full((T, 3), 9.0)
    grip = np.zeros(T)
    out = pp.classify_phases(_sig(eff, cube, grip, success=False), pp.Thresholds())
    assert not out["reached"]
    assert out["furthest_phase"] == "none"
    assert out["fail_phase"] == "reach"


def test_secure_but_no_success():
    T = 7
    eff = np.zeros((T, 3)); cube = np.zeros((T, 3))
    grip = np.ones(T); cube[:, 2] = 0.10
    out = pp.classify_phases(_sig(eff, cube, grip, success=False), pp.Thresholds())
    assert out["secured"] and not out["success"]
    assert out["furthest_phase"] == "secured"
    assert out["fail_phase"] == "transport"


def test_monotone_forcing_success_implies_lower():
    T = 4
    eff = np.full((T, 3), 9.0); cube = np.zeros((T, 3))   # raw reach False
    grip = np.zeros(T)                                    # raw secure False
    out = pp.classify_phases(_sig(eff, cube, grip, success=True), pp.Thresholds())
    assert out["reached_raw"] is False and out["secured_raw"] is False
    assert out["reached"] and out["secured"] and out["success"]
    assert out["furthest_phase"] == "success"


def test_k_steps_window_required():
    T = 10
    eff = np.zeros((T, 3)); cube = np.zeros((T, 3)); grip = np.ones(T)
    cube[:, 2] = 0.02            # at table; not lifted
    cube[3:5, 2] = 0.10          # lifted only 2 steps (< k=5)
    out = pp.classify_phases(_sig(eff, cube, grip, success=False), pp.Thresholds())
    assert out["secured"] is False


def test_zero_length_episode():
    z = np.zeros((0, 3))
    out = pp.classify_phases(_sig(z, z, np.zeros(0), success=False),
                             pp.Thresholds())
    assert out["length"] == 0
    assert out["furthest_phase"] == "none"
    assert out["fail_phase"] == "reach"
    assert np.isnan(out["min_eff_cube_dist"])
    assert np.isnan(out["final_cube_lift"])
    assert np.isnan(out["final_grip"])


def test_final_step_signals_when_held():
    T = 10
    eff = np.zeros((T, 3)); cube = np.zeros((T, 3)); grip = np.ones(T)
    cube[:, 2] = 0.10            # cube lifted whole episode (above table_z=0.02)
    out = pp.classify_phases(_sig(eff, cube, grip, success=False), pp.Thresholds())
    assert out["secured"] and not out["success"]
    assert out["final_cube_lift"] == pytest.approx(0.10 - 0.02)
    assert out["final_grip"] == pytest.approx(1.0)


def test_final_step_signals_when_dropped():
    T = 10
    eff = np.zeros((T, 3)); cube = np.zeros((T, 3)); grip = np.ones(T)
    cube[:5, 2] = 0.10           # lifted early
    cube[5:, 2] = 0.02           # then fell back to table at the end
    grip[8:] = 0.0               # gripper opened at the end
    out = pp.classify_phases(_sig(eff, cube, grip, success=False), pp.Thresholds())
    assert out["secured"] and not out["success"]
    assert out["final_cube_lift"] == pytest.approx(0.0, abs=1e-6)  # back on table
    assert out["final_grip"] == pytest.approx(0.0)                  # released


import pytest


class _FakeUnwrapped:
    def __init__(self):
        self._data = object()
        self._pinch_site_id = 0
        self._target_block = 0
        self.cur_task_info = {
            "goal_xyzs": np.array([[0.4, -0.1, 0.02]]),
            "init_xyzs": np.array([[0.42, 0.10, 0.0199]]),
        }


class _FakeEnv:
    def __init__(self, unwrapped):
        self.unwrapped = unwrapped


def test_step_signals_reads_info_keys():
    info = {
        "proprio/effector_pos": np.array([0.1, 0.2, 0.3]),
        "proprio/gripper_opening": np.array([0.9]),
        "privileged/block_0_pos": np.array([0.4, 0.5, 0.02]),
        "success": False,
    }
    s = pp.step_signals(info)
    assert np.allclose(s["eff"], [0.1, 0.2, 0.3])
    assert np.allclose(s["cube"], [0.4, 0.5, 0.02])
    assert s["grip"] == pytest.approx(0.9)


def test_episode_goal_and_table_z_from_cur_task_info():
    env = _FakeEnv(_FakeUnwrapped())
    assert np.allclose(pp.episode_goal(env), [0.4, -0.1, 0.02])
    assert pp.episode_table_z(env) == pytest.approx(0.0199)


def test_ensure_manip_env_rejects_non_manip():
    class Bare:
        pass

    with pytest.raises(RuntimeError, match="cube-only"):
        pp.ensure_manip_env(_FakeEnv(Bare()))
    # a manip-like env passes silently
    pp.ensure_manip_env(_FakeEnv(_FakeUnwrapped()))


class _FakeJoint:
    def __init__(self):
        self.qpos = np.zeros(7)


class _FakeData:
    def __init__(self):
        self.site_xpos = np.array([[0.5, 0.6, 0.7]])  # pinch site id 0
        self.ctrl = np.zeros(8)
        self._joint = _FakeJoint()

    def joint(self, name):
        assert name == "object_joint_0"
        return self._joint


class _FakeManip:
    def __init__(self):
        self._model = object()
        self._data = _FakeData()
        self._pinch_site_id = 0
        self._gripper_actuator_ids = np.array([6, 7])
        self._target_block = 0
        self.cur_task_info = {
            "goal_xyzs": np.array([[0.4, -0.1, 0.02]]),
            "init_xyzs": np.array([[0.42, 0.10, 0.0199]]),
        }


class _FakeMujoco:
    def __init__(self):
        self.forward_calls = 0
        self.step_calls = 0

    def mj_forward(self, model, data):
        self.forward_calls += 1

    def mj_step(self, model, data):
        self.step_calls += 1


def test_apply_scenario_s0_is_noop(monkeypatch):
    fake_mj = _FakeMujoco()
    monkeypatch.setattr(pp, "mujoco", fake_mj)
    env = _FakeEnv(_FakeManip())
    before = env.unwrapped._data._joint.qpos.copy()
    pp.apply_scenario(env, "S0")
    assert np.allclose(env.unwrapped._data._joint.qpos, before)
    assert fake_mj.step_calls == 0


def test_apply_scenario_s1_moves_cube_to_hand_xy(monkeypatch):
    fake_mj = _FakeMujoco()
    monkeypatch.setattr(pp, "mujoco", fake_mj)
    env = _FakeEnv(_FakeManip())
    pp.apply_scenario(env, "S1")
    q = env.unwrapped._data._joint.qpos
    assert q[0] == pytest.approx(0.5) and q[1] == pytest.approx(0.6)
    assert q[2] == pytest.approx(0.0199)        # table_z, not hand z
    assert fake_mj.forward_calls == 1
    assert fake_mj.step_calls == 0


def test_apply_scenario_s2_grasps_and_settles(monkeypatch):
    fake_mj = _FakeMujoco()
    monkeypatch.setattr(pp, "mujoco", fake_mj)
    env = _FakeEnv(_FakeManip())
    pp.apply_scenario(env, "S2")
    q = env.unwrapped._data._joint.qpos
    assert np.allclose(q[:3], [0.5, 0.6, 0.7])  # cube at effector
    assert np.all(env.unwrapped._data.ctrl[[6, 7]] == 255.0)
    assert fake_mj.step_calls == pp.N_SETTLE
    assert fake_mj.forward_calls >= 1


def test_apply_scenario_unknown_raises(monkeypatch):
    monkeypatch.setattr(pp, "mujoco", _FakeMujoco())
    with pytest.raises(ValueError, match="unknown scenario"):
        pp.apply_scenario(_FakeEnv(_FakeManip()), "SX")


import torch


class _RolloutEnv:
    """3-step episodes; cube fixed, effector reaches it; manip-like."""

    def __init__(self):
        self.unwrapped = _FakeManip()
        self.t = 0

    def _info(self, eff):
        return {
            "proprio/effector_pos": np.array(eff, dtype=np.float64),
            "proprio/gripper_opening": np.array([1.0]),
            "privileged/block_0_pos": np.array([0.0, 0.0, 0.10]),
            "success": self.t >= 3,
        }

    def reset(self, seed=None):
        self.t = 0
        return np.zeros(4, dtype=np.float32), self._info([9.0, 9.0, 9.0])

    def step(self, action):
        self.t += 1
        done = self.t >= 3
        return (np.zeros(4, dtype=np.float32), -1.0, done, False,
                self._info([0.0, 0.0, 0.10]))

    def close(self):
        pass


class _RolloutAgent:
    device = "cpu"

    def act(self, obs=None, z=None):
        return torch.zeros((1, 4))


def test_rollout_with_phase_signals_shapes(monkeypatch):
    monkeypatch.setattr(pp, "mujoco", _FakeMujoco())
    eps = pp.rollout_with_phase_signals(
        _RolloutEnv(), _RolloutAgent(), z=torch.zeros(4),
        n_episodes=2, thr=pp.Thresholds(), scenario="S0")
    assert len(eps) == 2
    e = eps[0]
    assert e["eff"].shape[1] == 3 and e["cube"].shape[1] == 3
    assert e["grip"].ndim == 1
    assert e["goal"].shape == (3,)
    assert e["table_z"] == pytest.approx(0.0199)
    assert e["success"] is True            # reaches t>=3
    assert e["length"] == e["eff"].shape[0]


def test_run_phase_probe_aggregates(monkeypatch, tmp_path):
    monkeypatch.setattr(pp, "mujoco", _FakeMujoco())

    def infer_z(task):
        return torch.zeros(4)

    def make_env(task):
        return _RolloutEnv()

    per_ep, summary, hist = pp.run_phase_probe(
        agent=_RolloutAgent(), infer_z=infer_z, make_env=make_env,
        tasks=["t1", "t2"], scenarios=["S0", "S2"], n_episodes=3,
        thr=pp.Thresholds())
    assert set(per_ep.columns) >= {
        "task", "scenario", "episode", "reached", "secured", "success",
        "reached_raw", "secured_raw", "fail_phase", "min_eff_cube_dist",
        "max_cube_lift", "final_cube_goal_dist", "length"}
    assert len(per_ep) == 2 * 2 * 3
    row = summary[(summary.task == "t1") & (summary.scenario == "S0")].iloc[0]
    assert row["n"] == 3
    assert row["success_rate"] == pytest.approx(1.0)  # _RolloutEnv succeeds
    assert "t1" in hist


def test_plots_and_summary_md(tmp_path):
    import pandas as pd
    per_ep = pd.DataFrame([
        {"task": "t1", "scenario": "S0", "episode": 0, "furthest_phase": "reached"},
        {"task": "t1", "scenario": "S0", "episode": 1, "furthest_phase": "secured"},
    ])
    summary = pd.DataFrame([
        {"task": "t1", "scenario": "S0", "reached_rate": 1.0,
         "secured_rate": 0.5, "success_rate": 0.0, "n": 2},
        {"task": "t1", "scenario": "S2", "reached_rate": 1.0,
         "secured_rate": 1.0, "success_rate": 0.9, "n": 2},
    ])
    pp.plot_phase_histogram(per_ep, tmp_path / "hist.png")
    pp.plot_scenario_success(summary, tmp_path / "succ.png")
    pp.write_summary_md(summary, tmp_path / "summary.md")
    assert (tmp_path / "hist.png").exists()
    assert (tmp_path / "succ.png").exists()
    txt = (tmp_path / "summary.md").read_text()
    assert "Hypothesis" in txt and "t1" in txt

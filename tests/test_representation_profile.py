import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals import representation_profile as rp


class _StubModel:
    """B = first z_dim obs dims; F = (obs0 + action0) broadcast, 2 heads."""
    z_dim = 3

    def backward_map(self, obs):
        return obs[:, : self.z_dim].clone()

    def forward_map(self, obs, z, action):
        base = (obs[:, 0:1] + action[:, 0:1]).expand(-1, self.z_dim)  # [B,zd]
        return torch.stack([base, base + 1.0], dim=0)                 # [2,B,zd]


class _StubAgent:
    device = "cpu"

    def act(self, obs, z):
        return torch.zeros((obs.shape[0], 2))


def test_ensemble_q_means_over_heads():
    F = torch.tensor([[[1.0, 0.0, 0.0]], [[3.0, 0.0, 0.0]]])  # [2,1,3]
    z = torch.tensor([2.0, 0.0, 0.0])
    q = rp._ensemble_q(F, z)
    assert q.shape == (1,)
    assert q[0] == 2.0 * ((1.0 + 3.0) / 2)  # mean over heads of (F·z)


def test_q_values_monotone_in_action():
    m = _StubModel()
    obs = torch.zeros((4, 5))
    z = torch.ones(3)
    a_lo = torch.zeros((4, 2))
    a_hi = torch.ones((4, 2))
    q_lo = rp.q_values(m, obs, a_lo, z)
    q_hi = rp.q_values(m, obs, a_hi, z)
    assert q_lo.shape == (4,)
    assert np.all(q_hi > q_lo)


def test_v_values_uses_policy_action():
    m, ag = _StubModel(), _StubAgent()
    obs = torch.zeros((3, 5))
    z = torch.ones(3)
    v = rp.v_values(m, ag, obs, z)
    # policy action is zeros => same as q_values with zero action
    assert np.allclose(v, rp.q_values(m, obs, torch.zeros((3, 2)), z))


from evals import phase_probe as pp


class _PhaseEnv:
    """3-step episode; manip-like; cube approaches goal so it 'succeeds'."""
    def __init__(self):
        self.unwrapped = _PhaseUnwrapped()
        self.t = 0

    def _info(self, eff, cube):
        return {
            "proprio/effector_pos": np.array(eff, np.float64),
            "proprio/gripper_opening": np.array([1.0]),
            "privileged/block_0_pos": np.array(cube, np.float64),
            "success": self.t >= 3,
        }

    def reset(self, seed=None):
        self.t = 0
        return np.zeros(5, np.float32), self._info([9, 9, 9], [0, 0, 0.02])

    def step(self, a):
        self.t += 1
        done = self.t >= 3
        return (np.full(5, self.t, np.float32), -1.0, done, False,
                self._info([0, 0, 0.20], [0, 0, 0.20]))

    def close(self):
        pass


class _PhaseUnwrapped:
    _target_block = 0
    # _data / _pinch_site_id only satisfy ensure_manip_env; never
    # dereferenced because every Task-2 rollout uses S0 (a no-op).
    _data = object()
    _pinch_site_id = 0
    cur_task_info = {"goal_xyzs": np.array([[0.0, 0.0, 0.20]]),
                     "init_xyzs": np.array([[0.0, 0.0, 0.02]])}


def test_phase_rollout_records_obs_when_requested():
    eps = pp.rollout_with_phase_signals(
        _PhaseEnv(), _StubAgent(), z=torch.zeros(3), n_episodes=1,
        thr=pp.Thresholds(), scenario="S0", record_obs=True)
    e = eps[0]
    assert "obs" in e and "action" in e
    assert e["obs"].shape[0] == e["length"]
    assert e["action"].shape[0] == e["length"] - 1  # actions between steps


def test_rollout_for_profile_attaches_outcome_and_d():
    eps = rp.rollout_for_profile(
        _PhaseEnv(), _StubAgent(), z=torch.zeros(3), n_episodes=1,
        thr=pp.Thresholds())
    e = eps[0]
    assert e["outcome"] in ("success", "transport_fail", "other")
    assert e["d"].shape[0] == e["obs"].shape[0]      # ||cube-goal|| per step
    assert e["transport_mask"].shape[0] == e["obs"].shape[0]


def _fake_ep(outcome, d, v_at_d):
    """Build a profile-episode whose V(s_t) = v_at_d(d_t) via stub model."""
    T = len(d)
    obs = np.zeros((T, 5), np.float32)
    obs[:, 0] = [v_at_d(x) for x in d]      # _StubModel Q ~ obs[:,0]
    return {
        "obs": obs, "action": np.zeros((max(T - 1, 0), 2), np.float32),
        "d": np.asarray(d, np.float64),
        "cube": np.zeros((T, 3), np.float64),
        "transport_mask": np.ones(T, bool),
        "grip": np.ones(T, np.float64),   # real episodes always carry grip
        "eff": np.zeros((T, 3), np.float64),  # and eff
        "outcome": outcome, "length": T,
    }


def test_value_landscape_rho_success_vs_fail():
    m, ag = _StubModel(), _StubAgent()
    z = torch.tensor([1.0, 0.0, 0.0])
    # >=5 transport steps so the guard passes; success V strictly rises
    # as d -> 0 (monotone in -d => rho ~ +1); fail V is flat => nan.
    eps = [
        _fake_ep("success",
                 [0.30, 0.25, 0.20, 0.15, 0.10, 0.02], lambda x: 1.0 - x),
        _fake_ep("transport_fail",
                 [0.30, 0.25, 0.20, 0.15, 0.10, 0.05], lambda x: 0.5),
    ]
    per_ep, per_step = rp.probe_value_landscape(m, ag, eps, z)
    s = per_ep.set_index("outcome")["rho_V_negd"]
    assert s.loc["success"] > 0.99            # strictly monotone -> ~1
    assert np.isnan(s.loc["transport_fail"])  # flat V -> degenerate -> nan
    assert set(per_step.columns) >= {"outcome", "d", "V"}
    assert {"cube_x", "cube_y"} <= set(per_step.columns)


def test_value_landscape_rho_guard_short_episode():
    m, ag = _StubModel(), _StubAgent()
    z = torch.tensor([1.0, 0.0, 0.0])
    # only 4 transport steps (< 5) -> guarded -> nan even though monotone
    eps = [_fake_ep("success", [0.30, 0.20, 0.10, 0.02], lambda x: 1.0 - x)]
    per_ep, _ = rp.probe_value_landscape(m, ag, eps, z)
    s = per_ep.set_index("outcome")["rho_V_negd"]
    assert np.isnan(s.loc["success"])


def test_z_decoding_topk_at_goal():
    m = _StubModel()  # B(next_obs) = first 3 dims
    N = 20
    next_obs = torch.zeros((N, 5))
    # 5 states aligned with z=[1,0,0]; of those, 3 are at goal (d<0.04)
    next_obs[:5, 0] = 10.0
    physics = np.zeros((N, 20), np.float32)
    goal = np.array([0.0, 0.0, 0.02], np.float64)
    physics[:, 14:17] = goal              # all at goal by default
    physics[3:, 14] = 1.0                 # indices 3.. are FAR from goal
    z = torch.tensor([1.0, 0.0, 0.0])
    relabel_metrics = {"relabel_reward#nonzero": 3.0,
                       "relabel_reward#num_samples": 20.0}
    out = rp.probe_z_decoding(m, next_obs, physics, goal, z,
                              relabel_metrics, topk=5)
    assert out["relabel_pos_frac"] == 3.0 / 20.0
    assert 0.0 <= out["topk_pct_at_goal"] <= 1.0
    # top-5 by B·z are indices 0..4; of these, 0,1,2 at goal => 60%
    assert abs(out["topk_pct_at_goal"] - 0.6) < 1e-9
    assert out["topk_mean_d"] >= 0.0


def test_b_resolution_linear_vs_random():
    rng = np.random.default_rng(0)
    N = 400
    B_lin = rng.normal(size=(N, 4)).astype(np.float32)
    d_lin = (B_lin @ np.array([1.0, -2.0, 0.5, 0.0]) + 0.1).astype(np.float64)
    out_lin = rp.probe_b_resolution(B_lin, d_lin, seed=0)
    assert out_lin["r2"] > 0.95

    B_rnd = rng.normal(size=(N, 4)).astype(np.float32)
    d_rnd = rng.normal(size=N).astype(np.float64)
    out_rnd = rp.probe_b_resolution(B_rnd, d_rnd, seed=0)
    assert out_rnd["r2"] < 0.2

    # separability: placed (d<0.04) vs near (0.04..0.12) clearly offset
    B_sep = np.vstack([np.zeros((100, 4)), np.ones((100, 4))]).astype(np.float32)
    d_sep = np.concatenate([np.full(100, 0.01), np.full(100, 0.08)])
    out_sep = rp.probe_b_resolution(B_sep, d_sep, seed=0)
    assert out_sep["placed_vs_near_acc"] > 0.95


def test_coverage_nn_distance_by_region():
    ref = np.zeros((50, 4), np.float32)            # support at origin
    eps = [{
        "obs": np.array([[0, 0, 0, 0],             # on-support (reach)
                          [0, 0, 0, 0],             # on-support (lift)
                          [9, 9, 9, 9]], np.float32),  # off-support (transport)
        "transport_mask": np.array([False, False, True]),
        "grip": np.array([0.0, 1.0, 1.0]),
        "cube": np.array([[0, 0, 0.02], [0, 0, 0.10], [0, 0, 0.20]]),
        "table_z": 0.02, "outcome": "transport_fail", "length": 3,
    }]
    df = rp.probe_coverage(eps, ref, pp.Thresholds())
    assert set(df.columns) >= {"outcome", "region", "nn_dist"}
    far = df[df.region == "transport"]["nn_dist"].mean()
    near = df[df.region == "reach"]["nn_dist"].mean()
    assert far > near


def test_run_representation_profile_returns_four_frames(monkeypatch):
    m, ag = _StubModel(), _StubAgent()

    def infer_z(task):
        return torch.tensor([1.0, 0.0, 0.0]), {
            "relabel_reward#nonzero": 2.0,
            "relabel_reward#num_samples": 10.0}

    def make_env(task):
        return _PhaseEnv()

    def sample_buffer(n):
        return {
            "next_obs": torch.zeros((n, 5)),
            "physics": np.zeros((n, 20), np.float32),
            "action": np.zeros((n, 2), np.float32),
        }

    def relabel_fn(phys, act):
        return np.zeros((len(phys), 1), np.float32)

    out = rp.run_representation_profile(
        model=m, agent=ag, infer_z=infer_z, make_env=make_env,
        sample_buffer=sample_buffer, relabel_fn_for=lambda t: relabel_fn,
        goal_for=lambda t: np.array([0.0, 0.0, 0.20]),
        tasks=["t1", "t2"], n_episodes=1, thr=pp.Thresholds(),
        buffer_sample=8, topk=4, seed=0)
    assert set(out) == {"value_landscape", "value_steps",
                        "z_decoding", "b_resolution", "coverage"}
    assert {"task"}.issubset(out["z_decoding"].columns)
    assert out["z_decoding"]["task"].nunique() == 2


def test_run_representation_profile_cube_coverage():
    m, ag = _StubModel(), _StubAgent()

    def infer_z(task):
        return torch.tensor([1.0, 0.0, 0.0]), {
            "relabel_reward#nonzero": 2.0,
            "relabel_reward#num_samples": 10.0}

    def make_env(task):
        return _PhaseEnv()

    def sample_buffer(n):
        phys = np.zeros((n, 20), np.float32)   # cube ref at origin
        return {"next_obs": torch.zeros((n, 5)), "physics": phys,
                "action": np.zeros((n, 2), np.float32)}

    out = rp.run_representation_profile(
        model=m, agent=ag, infer_z=infer_z, make_env=make_env,
        sample_buffer=sample_buffer,
        relabel_fn_for=lambda t: (lambda p, a: np.zeros((len(p), 1),
                                                        np.float32)),
        goal_for=lambda t: np.array([0.0, 0.0, 0.20]),
        tasks=["t1"], n_episodes=1, thr=pp.Thresholds(),
        buffer_sample=8, topk=4, seed=0, coverage_feature="cube")
    assert set(out["coverage"].columns) >= {"task", "outcome",
                                            "region", "nn_dist"}


def test_aggregate_builds_tables_and_story(tmp_path):
    import importlib
    import pandas as pd
    agg = importlib.import_module("scripts.probes.representation_profile_aggregate")
    for s in (3, 7):
        d = tmp_path / f"s{s}_final"
        d.mkdir()
        pd.DataFrame([{"task": "t1", "outcome": "success",
                       "rho_V_negd": 0.9, "V_at_secure": 0.1,
                       "V_at_end": 0.9, "n_transport_steps": 5},
                      {"task": "t1", "outcome": "transport_fail",
                       "rho_V_negd": 0.0, "V_at_secure": 0.5,
                       "V_at_end": 0.5, "n_transport_steps": 5}]
                     ).to_parquet(d / "value_landscape.parquet")
        pd.DataFrame([{"task": "t1", "outcome": "success", "d": 0.1,
                       "V": 0.5, "transport": True}]
                     ).to_parquet(d / "value_steps.parquet")
        pd.DataFrame([{"task": "t1", "relabel_pos_frac": 0.02,
                       "topk_mean_d": 0.3, "topk_pct_at_goal": 0.1,
                       "topk": 4}]).to_parquet(d / "z_decoding.parquet")
        pd.DataFrame([{"task": "t1", "r2": 0.2,
                       "placed_vs_near_acc": 0.6, "n_placed": 5,
                       "n_near": 9}]).to_parquet(d / "b_resolution.parquet")
        pd.DataFrame([{"task": "t1", "outcome": "transport_fail",
                       "region": "transport", "nn_dist": 3.0}]
                     ).to_parquet(d / "coverage.parquet")
    out = tmp_path / "aggregate"
    agg.aggregate(tmp_path, out)
    assert (out / "story.md").exists()
    assert (out / "value_vs_dist.png").exists()
    assert (out / "T1_value_gradient.parquet").exists()
    txt = (out / "story.md").read_text()
    assert "## Readout" in txt
    assert "0.02" in txt          # relabel_pos_frac mean filled in
    assert "T1 value gradient" in txt and "Synthesis:" in txt
    # rho_succ=0.9, rho_fail=0.0 -> SUPPORTS; frac 0.02 & topk 0.1 -> SUPPORTS
    assert "SUPPORTS" in txt
    assert "Reach ~100%" not in txt   # hardcoded template is gone


class _ContractModel:
    """Enforces the real FB contract: forward_map needs z batched to
    obs (ForwardMap concatenates [obs, z] on the feature axis)."""
    z_dim = 3

    def forward_map(self, obs, z, action):
        assert z.dim() == 2 and z.shape[0] == obs.shape[0], \
            (tuple(obs.shape), tuple(z.shape))
        base = (obs[:, 0:1] + action[:, 0:1]).expand(-1, self.z_dim)
        return torch.stack([base, base + 1.0], dim=0)             # [2,B,zd]


class _ContractAgent:
    """Enforces the real agent.act contract: z must be [B, z_dim]
    (model.actor draws noises = randn((z.shape[0], action_dim)))."""
    device = "cpu"

    def act(self, obs, z):
        assert z.dim() == 2 and z.shape[0] == obs.shape[0], \
            (tuple(obs.shape), tuple(z.shape))
        return torch.zeros((obs.shape[0], 2))


def test_q_values_broadcasts_global_z_over_batch():
    obs = torch.zeros((5, 4))
    action = torch.zeros((5, 2))
    z = torch.tensor([1.0, 0.0, 0.0])          # one global task vector
    q = rp.q_values(_ContractModel(), obs, action, z)
    assert q.shape == (5,)


def test_v_values_broadcasts_global_z_over_batch():
    obs = torch.zeros((5, 4))
    z = torch.tensor([1.0, 0.0, 0.0])          # one global task vector
    v = rp.v_values(_ContractModel(), _ContractAgent(), obs, z)
    assert v.shape == (5,)


def test_aggregate_verdict_helpers():
    import importlib
    agg = importlib.import_module("scripts.probes.representation_profile_aggregate")
    # T1: success monotone, fail flat -> SUPPORTS
    assert agg._verdict_t1(0.40, 0.02)[0] == "SUPPORTS"
    # T1: fail rises more than success -> CONTRADICTS
    assert agg._verdict_t1(0.05, 0.40)[0] == "CONTRADICTS"
    # T1: nan in -> INSUFFICIENT DATA
    assert agg._verdict_t1(float("nan"), 0.1)[0] == "INSUFFICIENT DATA"
    # T2: sparse + z mis-pointed -> SUPPORTS
    assert agg._verdict_t2(0.02, 0.20)[0] == "SUPPORTS"
    assert agg._verdict_t2(0.30, 0.80)[0] == "WEAK"
    # T3: weak resolution -> SUPPORTS; strong -> RESOLVES
    assert agg._verdict_t3(0.30, 0.65)[0] == "SUPPORTS"
    assert agg._verdict_t3(0.80, 0.95)[0] == "RESOLVES"
    # T4: fail far more off-support -> SUPPORTS; similar -> NEUTRAL
    assert agg._verdict_t4(3.0, 1.0)[0] == "SUPPORTS"
    assert agg._verdict_t4(2.2, 2.1)[0] == "NEUTRAL"
    assert agg._verdict_t4(float("nan"), 2.0)[0] == "INSUFFICIENT DATA"
    # synthesis names supporting probes
    syn = agg._synthesis({"T1": ("SUPPORTS", ""), "T2": ("SUPPORTS", ""),
                           "T3": ("WEAK", ""), "T4": ("NEUTRAL", "")})
    assert "T1" in syn and "T2" in syn and syn.startswith("Synthesis:")


def test_probe_value_landscape_emits_t_and_region(monkeypatch):
    from evals.phase_probe import Thresholds

    T = 5
    ep = {
        "obs": np.zeros((T, 3), np.float32),
        "d": np.linspace(1.0, 0.1, T),
        "transport_mask": np.array([False, False, False, True, True]),
        "grip": np.array([0.0, 0.0, 0.9, 0.9, 0.9]),
        "cube": np.column_stack([np.linspace(0, 0.2, T), np.zeros(T),
                                 np.array([0.02, 0.02, 0.02,
                                           0.10, 0.10])]),
        "eff": np.column_stack([np.linspace(0.1, 0.3, T),
                                np.linspace(-0.1, 0.1, T),
                                np.full(T, 0.4)]),
        "table_z": 0.02,
        "outcome": "success",
    }
    monkeypatch.setattr(rp, "v_values",
                         lambda *a, **k: np.linspace(0.0, 1.0, T))
    _, per_step = rp.probe_value_landscape(None, None, [ep], None,
                                           thr=Thresholds())
    assert list(per_step["t"]) == list(range(T))
    assert list(per_step["region"]) == ["reach", "reach", "lift",
                                        "transport", "transport"]
    assert {"eef_x", "eef_y"} <= set(per_step.columns)
    assert float(per_step.iloc[0]["eef_x"]) == 0.1

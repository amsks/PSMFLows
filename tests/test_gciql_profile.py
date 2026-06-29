import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.profiles import gciql_profile as gp
from evals.phase_probe import Thresholds


def test_parse_flags_reads_env_and_seed(tmp_path):
    (tmp_path / "flags.json").write_text(json.dumps(
        {"env_name": "cube-single-play-v0", "seed": 10,
         "agent": "agents/gciql.py"}))
    info = gp.parse_flags(tmp_path)
    assert info["env_name"] == "cube-single-play-v0"
    assert info["seed"] == 10


def _ep(outcome, d, v, grip, cube_z, table_z=0.02):
    T = len(d)
    cube = np.column_stack([np.zeros(T), np.zeros(T), np.asarray(cube_z)])
    return {
        "obs": np.zeros((T, 4), np.float32),
        "d": np.asarray(d, np.float64),
        "V": np.asarray(v, np.float64),
        "grip": np.asarray(grip, np.float64),
        "cube": cube,
        "eff": cube.copy(),                 # effector on the cube
        "goal": np.array([0.0, 0.0, 0.02]),
        "table_z": table_z, "success": outcome == "success",
        "task": "task1", "episode": 0,
    }


def test_episodes_to_frames_t1_t4_funnel():
    # 6 transport steps, V strictly rises as d -> 0  => rho ~ +1
    d = [0.30, 0.25, 0.20, 0.15, 0.10, 0.02]
    v = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    grip = [1.0] * 6
    cube_z = [0.20] * 6                    # lifted (table_z 0.02)
    eps = [_ep("success", d, v, grip, cube_z)]
    vl, cov, fun, vs = gp.episodes_to_frames(eps, Thresholds(),
                                             ref_obs=np.zeros((10, 4)))
    assert set(vl.columns) >= {"task", "episode", "outcome",
                               "rho_V_negd", "n_transport_steps"}
    assert vl.iloc[0]["rho_V_negd"] > 0.99
    assert vl.iloc[0]["outcome"] == "success"
    assert set(cov.columns) >= {"task", "outcome", "region", "nn_dist"}
    assert set(fun.columns) >= {"task", "episode", "furthest_phase",
                                "outcome"}
    assert set(vs.columns) == {"task", "episode", "t", "outcome",
                               "region", "d", "V", "transport",
                               "cube_x", "cube_y", "eef_x", "eef_y"}
    assert vs["cube_x"].iloc[0] == eps[0]["cube"][0][0]
    assert vs["cube_y"].iloc[0] == eps[0]["cube"][0][1]
    assert len(vs) == 6                       # one row per step
    assert vs["transport"].dtype == bool


def test_episodes_to_frames_attributes_transport_fail_with_real_eff():
    # hand on the cube, cube lifted+gripped >=5 steps, never reaches
    # goal, env success False  ->  classify_phases must say transport_fail
    # (regression: zero-filled eff mislabels this as reach/other)
    T = 6
    cube = np.tile([0.40, 0.10, 0.20], (T, 1)).astype(np.float64)
    eff = cube.copy()                       # effector reaches+secures
    grip = np.ones(T)                       # gripper closed
    d = np.linspace(0.30, 0.10, T)          # cube never delivered
    V = np.linspace(0.0, 1.0, T)
    ep = {
        "obs": np.zeros((T, 4), np.float32),
        "eff": eff, "cube": cube, "grip": grip,
        "d": d, "V": V, "table_z": 0.02,
        "goal": np.array([0.0, 0.0, 0.02]),
        "success": False, "task": "task1", "episode": 0,
    }
    vl, _, fun, _ = gp.episodes_to_frames([ep], Thresholds(),
                                          ref_obs=np.zeros((10, 4)))
    assert vl.iloc[0]["outcome"] == "transport_fail"
    assert fun.iloc[0]["furthest_phase"] == "transport"


def test_build_argparser_has_required_flags():
    ap = gp.build_argparser()
    ns = ap.parse_args([
        "--run-dir", "/x/sd001", "--step", "1000000",
        "--out", "/tmp/o", "--tasks", "1,2,3", "--n-episodes", "4",
        "--dataset-path", "datasets/cube-single-play-v0"])
    assert ns.run_dir == "/x/sd001" and ns.step == 1000000
    assert ns.tasks == "1,2,3" and ns.n_episodes == 4


def test_build_argparser_has_obs_type():
    ap = gp.build_argparser()
    ns = ap.parse_args(["--run-dir", "/x", "--out", "/o",
                        "--obs-type", "pixels"])
    assert ns.obs_type == "pixels"


def test_load_ref_obs_cube_reads_physics(tmp_path):
    buf = tmp_path / "buffer"
    buf.mkdir(parents=True)
    T = 5
    phys = np.zeros((T, 21), np.float32)
    phys[:, 14:17] = np.arange(T * 3).reshape(T, 3)
    np.savez(buf / "episode_000000_5.npz",
             observation=np.zeros((T, 28), np.float32),
             physics=phys, action=np.zeros((T, 5), np.float32))
    ref = gp._load_ref_obs(str(tmp_path), feature="cube")
    assert ref.shape[1] == 3
    assert np.allclose(np.sort(ref.ravel()),
                       np.sort(phys[:, 14:17].ravel()))


def test_episodes_to_frames_cube_coverage():
    d = [0.30, 0.25, 0.20, 0.15, 0.10, 0.02]
    v = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    grip = [1.0] * 6
    cube_z = [0.20] * 6
    eps = [_ep("success", d, v, grip, cube_z)]
    cube_ref = np.zeros((10, 3), np.float32)
    vl, cov, fun, vs = gp.episodes_to_frames(
        eps, Thresholds(), cube_ref, feature="cube")
    assert set(cov.columns) >= {"task", "outcome", "region", "nn_dist"}


def _seed_dir(root, s):
    d = root / f"s{s}_final"
    d.mkdir(parents=True)
    pd.DataFrame([
        {"task": "task1", "episode": 0, "outcome": "success",
         "rho_V_negd": 0.6, "V_at_secure": 0.1, "V_at_end": 0.9,
         "n_transport_steps": 6},
        {"task": "task1", "episode": 1, "outcome": "transport_fail",
         "rho_V_negd": 0.0, "V_at_secure": 0.5, "V_at_end": 0.5,
         "n_transport_steps": 6},
    ]).to_parquet(d / "value_landscape.parquet")
    pd.DataFrame([
        {"task": "task1", "outcome": "transport_fail",
         "region": "transport", "nn_dist": 3.0},
        {"task": "task1", "outcome": "success",
         "region": "transport", "nn_dist": 1.0}],
    ).to_parquet(d / "coverage.parquet")
    pd.DataFrame([
        {"task": "task1", "episode": 0, "furthest_phase": "success",
         "outcome": "success"},
        {"task": "task1", "episode": 1, "furthest_phase": "transport",
         "outcome": "transport_fail"}],
    ).to_parquet(d / "phase_funnel.parquet")


def test_gciql_aggregate_and_comparison(tmp_path):
    import importlib
    agg = importlib.import_module("scripts.profiles.gciql_profile_aggregate")
    groot = tmp_path / "gciql"
    for s in (1, 2):
        _seed_dir(groot, s)
    # synthetic FB aggregate for the comparison
    fbagg = tmp_path / "fb_aggregate"
    fbagg.mkdir(parents=True)
    pd.DataFrame([
        {"task": "task1", "outcome": "success", "rho_V_negd": 0.13},
        {"task": "task1", "outcome": "transport_fail",
         "rho_V_negd": 0.23}]).to_parquet(
        fbagg / "T1_value_gradient.parquet")
    out = tmp_path / "gagg"
    agg.aggregate(groot, out, fb_aggregate=fbagg)
    assert (out / "story.md").exists()
    assert (out / "T1_value_gradient.parquet").exists()
    cmp = out / "comparison"
    assert (cmp / "fb_vs_gciql.parquet").exists()
    assert (cmp / "comparison.md").exists()
    txt = (cmp / "comparison.md").read_text()
    assert "FB" in txt and "GCIQL" in txt


def _seed_steps(_P, root, s, outcome, n=8):
    import numpy as np
    base = root / f"s{s}_final"
    base.mkdir(parents=True, exist_ok=True)
    ep = 0 if outcome == "success" else 1
    rho = 0.5 if outcome == "success" else 0.0

    def _app(name, df):
        p = base / name
        if p.exists():
            df = pd.concat([pd.read_parquet(p), df], ignore_index=True)
        df.to_parquet(p)

    _app("value_landscape.parquet", pd.DataFrame([
        {"task": "task1", "episode": ep, "outcome": outcome,
         "rho_V_negd": rho, "V_at_secure": 0.1, "V_at_end": 0.9,
         "n_transport_steps": n}]))
    _app("value_steps.parquet", pd.DataFrame(
        {"task": ["task1"] * n, "episode": [ep] * n,
         "outcome": [outcome] * n, "d": np.linspace(0.3, 0.05, n),
         "V": np.linspace(0.0, 1.0, n), "transport": [True] * n}))
    _app("coverage.parquet", pd.DataFrame([
        {"task": "task1", "outcome": outcome, "region": "transport",
         "nn_dist": 2.0}]))
    _app("phase_funnel.parquet", pd.DataFrame([
        {"task": "task1", "episode": ep, "furthest_phase": outcome,
         "outcome": outcome}]))


def test_zscore_v_is_per_call_standardised_and_order_preserving():
    import numpy as np
    from scripts.profiles.gciql_profile_aggregate import _zscore_v
    df = pd.DataFrame({"V": [2000.0, 2100.0, 1900.0, 2050.0, 1950.0]})
    out = _zscore_v(df)
    assert abs(float(out["Vz"].mean())) < 1e-9
    assert abs(float(out["Vz"].std(ddof=0)) - 1.0) < 1e-9
    # rank order preserved (so the curve shape is unchanged, only scale)
    assert list(out["Vz"].rank()) == list(df["V"].rank())
    # constant V -> all zeros, no div-by-zero
    z0 = _zscore_v(pd.DataFrame({"V": [5.0, 5.0, 5.0]}))
    assert (z0["Vz"] == 0.0).all()


def test_comparison_emits_four_plots(tmp_path):
    import importlib
    agg = importlib.import_module("scripts.profiles.gciql_profile_aggregate")
    groot = tmp_path / "gciql"
    froot = tmp_path / "repr"
    for s in (1, 2):
        _seed_steps(__import__("pathlib").Path, groot, s, "success")
        _seed_steps(__import__("pathlib").Path, groot, s,
                    "transport_fail")
    for s in (1, 2):
        _seed_steps(__import__("pathlib").Path, froot, s, "success")
        _seed_steps(__import__("pathlib").Path, froot, s,
                    "transport_fail")
    fbagg = tmp_path / "fb_aggregate"
    fbagg.mkdir()
    pd.DataFrame([{"task": "task1", "outcome": "success",
                   "rho_V_negd": 0.13},
                  {"task": "task1", "outcome": "transport_fail",
                   "rho_V_negd": 0.23}]).to_parquet(
        fbagg / "T1_value_gradient.parquet")
    out = tmp_path / "gagg"
    agg.aggregate(groot, out, fb_aggregate=fbagg, fb_seed_root=froot)
    cmp = out / "comparison"
    for fn in ("cmp_value_vs_dist.png", "cmp_rho_box.png",
               "cmp_funnel.png", "cmp_coverage.png",
               "cmp_value_rho.png", "fb_vs_gciql.parquet",
               "comparison.md"):
        assert (cmp / fn).exists(), fn


def test_bin_vz_grid_masks_empty_cells():
    import numpy as np
    from scripts.profiles.gciql_profile_aggregate import _bin_vz_grid
    df = pd.DataFrame({
        "cube_x": [0.31, 0.31, 0.54],
        "cube_y": [-0.24, -0.24, 0.24],
        "Vz": [1.0, 3.0, -2.0],
    })
    grid, xe, ye = _bin_vz_grid(df, xmin=0.30, xmax=0.55,
                                ymin=-0.25, ymax=0.25, nbins=5)
    assert grid.shape == (5, 5)
    # the (0.31,-0.24) cell holds mean(1,3)=2.0; far corner holds -2.0
    assert np.isfinite(grid).sum() == 2
    assert np.nanmin(grid) == -2.0 and np.nanmax(grid) == 2.0


def test_medoid_episode_picks_closest_to_mean_path():
    import numpy as np
    from scripts.profiles.gciql_profile_aggregate import _medoid_episode
    # ep 0 & 1 near y=0; ep 2 far at y=1 -> medoid is 0 or 1, not 2
    rows = []
    for ep, y in ((0, 0.0), (1, 0.01), (2, 1.0)):
        for t in range(4):
            rows.append({"episode": ep, "cube_x": 0.4 + 0.01 * t,
                         "cube_y": y})
    df = pd.DataFrame(rows)
    assert _medoid_episode(df) in (0, 1)


def test_episodes_to_frames_emits_t_and_region():
    from scripts.profiles import gciql_profile as gp
    from evals.phase_probe import Thresholds
    import numpy as np
    T = 6
    ep = {
        "obs": np.zeros((T, 4), np.float32),
        "d": np.linspace(1.0, 0.1, T),
        "V": np.linspace(0.0, 1.0, T),
        "grip": np.array([0.0, 0.0, 0.9, 0.9, 0.9, 0.9]),
        "cube": np.column_stack([np.linspace(0, 0.3, T),
                                 np.zeros(T),
                                 np.array([0.02, 0.02, 0.02,
                                           0.10, 0.10, 0.10])]),
        "eff": np.zeros((T, 3), np.float64),
        "goal": np.array([0.3, 0.0, 0.02]),
        "table_z": 0.02,
        "success": True, "task": "task1", "episode": 0,
    }
    _, _, _, vs = gp.episodes_to_frames([ep], Thresholds(),
                                        np.zeros((2, 4), np.float32))
    assert list(vs["t"]) == list(range(T))
    assert set(vs["region"]) <= {"reach", "lift", "transport"}
    assert list(vs["region"]) == ["reach", "reach", "lift",
                                  "transport", "transport", "transport"]
    # eef_x/eef_y persisted from ep["eff"][:, :2]
    assert set({"eef_x", "eef_y"}) <= set(vs.columns)
    assert float(vs.iloc[0]["eef_x"]) == 0.0
    assert float(vs.iloc[0]["eef_y"]) == 0.0

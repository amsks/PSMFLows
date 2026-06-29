import stat
from pathlib import Path

from hydra import compose, initialize

from run_gciql import build_ogbench_argv

REPO = Path(__file__).resolve().parents[1]


def test_visual_gciql_config_builds_expected_argv():
    with initialize(version_base="1.3", config_path="../configs/gciql"):
        cfg = compose(config_name="cube_single_visual")
    cfg_d = dict(cfg)
    cfg_d["output_root"] = "/tmp/out"
    argv = build_ogbench_argv(cfg_d)
    assert "--env_name=visual-cube-single-play-v0" in argv
    assert "--agent=agents/gciql.py" in argv
    assert "--agent.encoder=impala_small" in argv
    assert "--agent.p_aug=0.5" in argv
    assert "--agent.batch_size=256" in argv


def test_runner_script_exists_and_is_executable():
    p = REPO / "scripts" / "run_gciql_visual.sh"
    assert p.exists(), "scripts/run_gciql_visual.sh must exist"
    mode = p.stat().st_mode
    assert mode & stat.S_IXUSR, "script must be executable"
    text = p.read_text()
    # Runner is CONFIG-selectable (GCIQL vs GCIVL vs DrQ front-end); it passes
    # --config-name "$CONFIG" and defaults CONFIG to the GCIQL visual config.
    assert '--config-name "$CONFIG"' in text
    assert 'CONFIG="${CONFIG:-cube_single_visual}"' in text

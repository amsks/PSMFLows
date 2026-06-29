from hydra import compose, initialize

from run_gciql import build_ogbench_argv


def _argv(name):
    with initialize(version_base="1.3", config_path="../configs/gciql"):
        cfg = compose(config_name=name)
    cfg_d = dict(cfg)
    cfg_d["output_root"] = "/tmp/out"
    return build_ogbench_argv(cfg_d)


def test_gciql_drq_config_argv():
    argv = _argv("cube_single_visual_drq")
    assert "--agent=agents/gciql.py" in argv
    assert "--agent.encoder=drq" in argv
    assert "--agent.frame_stack=3" in argv
    assert "--agent.p_aug=1.0" in argv
    assert "--agent.alpha=1.0" in argv
    assert "--run_group=factored-fb-gciql-pixel-drq" in argv
    assert "--env_name=visual-cube-single-play-v0" in argv


def test_gcivl_drq_config_argv():
    argv = _argv("cube_single_visual_gcivl_drq")
    assert "--agent=agents/gcivl.py" in argv
    assert "--agent.encoder=drq" in argv
    assert "--agent.frame_stack=3" in argv
    assert "--agent.p_aug=1.0" in argv
    assert "--agent.alpha=10.0" in argv
    assert "--run_group=factored-fb-gcivl-pixel-drq" in argv

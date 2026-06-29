from hydra import compose, initialize

from run_gciql import build_ogbench_argv


def _argv(name):
    with initialize(version_base="1.3", config_path="../configs/gciql"):
        cfg = compose(config_name=name)
    cfg_d = dict(cfg)
    cfg_d["output_root"] = "/tmp/out"
    return build_ogbench_argv(cfg_d)


def test_crl_flowbc_state_config_argv():
    argv = _argv("cube_single_state_crl_flowbc")
    assert any(a.startswith("--agent=") and a.endswith("tools/wandb_mode_shim/crl_flowbc.py") for a in argv)
    assert "--agent.bc_coeff=3.0" in argv
    assert "--run_group=factored-fb-crl-flowbc" in argv
    assert "--env_name=cube-single-play-v0" in argv
    assert not any(a.startswith("--agent.encoder=") for a in argv)


def test_crl_flowbc_pixel_config_argv():
    argv = _argv("cube_single_visual_crl_flowbc_drq")
    assert any(a.startswith("--agent=") and a.endswith("tools/wandb_mode_shim/crl_flowbc.py") for a in argv)
    assert "--agent.encoder=drq" in argv
    assert "--agent.frame_stack=3" in argv
    assert "--agent.p_aug=1.0" in argv
    assert "--agent.bc_coeff=3.0" in argv
    assert "--run_group=factored-fb-crl-flowbc-pixel-drq" in argv
    assert "--env_name=visual-cube-single-play-v0" in argv

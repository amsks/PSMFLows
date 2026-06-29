from run_gciql import build_child_env

_BASE = dict(
    env_name="visual-cube-single-play-v0",
    wandb_mode="online",
    agent_overrides={"encoder": "drq", "frame_stack": 3, "p_aug": 1.0},
)


def test_drq_encoder_sets_frontend_flag():
    env = build_child_env(_BASE, {})
    assert env["OGBENCH_DRQ_FRONTEND"] == "1"


def test_impala_encoder_does_not_set_frontend_flag():
    plain = {**_BASE, "agent_overrides": {"encoder": "impala_small", "p_aug": 0.5}}
    env = build_child_env(plain, {})
    assert "OGBENCH_DRQ_FRONTEND" not in env


def test_build_child_env_still_sets_core_shim_vars():
    env = build_child_env(_BASE, {})
    assert env["GCIQL_FB_ENV"] == "visual-cube-single-play-v0"
    assert "tools/wandb_mode_shim" in env["PYTHONPATH"]
    assert env["MUJOCO_GL"] == "egl"


def test_pythonpath_has_shim_first_then_impls():
    # The impls dir MUST be on PYTHONPATH so sitecustomize can import utils.* at
    # interpreter startup (the script dir is not yet on sys.path then); the shim
    # dir must come first so Python imports our sitecustomize.
    env = build_child_env(_BASE, {})
    parts = env["PYTHONPATH"].split(":")
    assert parts[0].endswith("tools/wandb_mode_shim")
    assert any(p.endswith("third_party/ogbench/impls") for p in parts)

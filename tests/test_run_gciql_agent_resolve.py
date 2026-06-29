import os

from run_gciql import build_ogbench_argv

_BASE = dict(
    env_name="cube-single-play-v0", seed=0, train_steps=10, log_interval=5,
    eval_interval=5, save_interval=5, eval_episodes=2, eval_on_cpu=1,
    run_group="g", output_root="/tmp/out", agent_overrides={},
)


def test_repo_relative_agent_file_made_absolute():
    cfg = {**_BASE, "agent_file": "tools/wandb_mode_shim/crl_flowbc.py"}
    argv = build_ogbench_argv(cfg)
    flag = [a for a in argv if a.startswith("--agent=")][0]
    path = flag.split("=", 1)[1]
    assert os.path.isabs(path)
    assert path.endswith("tools/wandb_mode_shim/crl_flowbc.py")
    assert os.path.isfile(path)


def test_vendored_agent_file_passthrough():
    cfg = {**_BASE, "agent_file": "agents/crl.py"}
    argv = build_ogbench_argv(cfg)
    assert "--agent=agents/crl.py" in argv

from agents.tdmpc2.config import build_tdmpc2_cfg


def test_cfg_folds_goal_into_obs_shape():
    cfg = build_tdmpc2_cfg(obs_dim=19, action_dim=5, device="cpu")
    assert cfg.obs_shape["state"][0] == 19 + 3   # GOAL_DIM
    assert cfg.task_dim == 0
    assert cfg.multitask is False
    assert abs(cfg.bin_size - (10 - (-10)) / 100) < 1e-9


def test_overrides_apply():
    cfg = build_tdmpc2_cfg(obs_dim=10, action_dim=4, device="cpu", horizon=5, num_q=3)
    assert cfg.horizon == 5 and cfg.num_q == 3

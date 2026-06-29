"""One-step FB (https://github.com/chongyi-zheng/onestep-fb), integrated with FlowBC.

One-step FB learns the BEHAVIOR-policy successor features: F is NOT conditioned on z
(_fb_z feeds zeros) and the next action is the dataset's own (SARSA, batch["next"]["action"]),
not sampled from the policy. The FlowBC actor then does one improvement step on <F(s,a), z>.
Flag-gated on `onestep` (default False => standard FB, byte-identical).
"""
import numpy as np
import torch


def _tiny(**extra):
    import gymnasium
    from agents.fb.agent import FBAgent
    from nn_models import IdentityNNConfig
    from normalizers import IdentityNormalizerConfig
    return FBAgent(
        obs_space=gymnasium.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32),
        action_dim=2, z_dim=8, L_dim=8, batch_size=16,
        obs_normalizer_cfg=IdentityNormalizerConfig(),
        rgb_encoder_cfg=IdentityNNConfig(),
        augmentator_cfg=IdentityNNConfig(),
        device="cpu", **extra,
    )


def _tiny_flowbc(**extra):
    import gymnasium
    from agents.fb.flow_bc.agent import FBFlowBCAgent
    from nn_models import IdentityNNConfig, NoiseConditionedActorArchiConfig, SimpleVectorFieldArchiConfig
    from normalizers import IdentityNormalizerConfig
    return FBFlowBCAgent(
        obs_space=gymnasium.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32),
        action_dim=2, z_dim=8, L_dim=8, batch_size=16,
        actor_cfg=NoiseConditionedActorArchiConfig(),
        actor_vf_cfg=SimpleVectorFieldArchiConfig(),
        obs_normalizer_cfg=IdentityNormalizerConfig(),
        rgb_encoder_cfg=IdentityNNConfig(),
        augmentator_cfg=IdentityNNConfig(),
        device="cpu", **extra,
    )


# --- data: next-action (SARSA) is stored and aligned -------------------------------

def test_next_action_is_sarsa(tmp_path):
    from data.ogbench import load_transitions
    T = 5
    obs = np.arange(T * 2, dtype=np.float32).reshape(T, 2)
    act = (np.arange(T, dtype=np.float32) * 10.0).reshape(T, 1)  # 0,10,20,30,40
    phys = np.zeros((T, 20), dtype=np.float32)
    f = tmp_path / "episode_000000_4.npz"
    np.savez(f, observation=obs, action=act, physics=phys)
    st = load_transitions([str(f)], obs_type="state")
    assert "action" in st["next"], "next/action missing from storage"
    a, na = st["action"], st["next"]["action"]
    assert a.shape == na.shape                       # aligned with the other arrays
    assert np.allclose(na[:-1], a[1:])               # SARSA: next-action[t] == action[t+1]


# --- agent: onestep flag + z-zeroing for F -----------------------------------------

def test_onestep_flag_default_off():
    a = _tiny()
    assert a.onestep is False
    z = torch.ones(4, a.z_dim)
    assert torch.equal(a._fb_z(z), z)                # standard FB: F sees z


def test_onestep_zeros_z_for_F():
    a = _tiny(onestep=True)
    assert a.onestep is True
    z = torch.randn(4, a.z_dim)
    zf = a._fb_z(z)
    assert zf.shape == z.shape and torch.count_nonzero(zf) == 0   # F is z-independent


# --- integration: one-step FB + FlowBC update() runs end-to-end --------------------

def test_onestep_flowbc_update_runs():
    a = _tiny_flowbc(onestep=True)
    B = a.batch_size
    batch = {
        "observation": torch.randn(B, 4),
        "action": torch.rand(B, 2) * 2 - 1,
        "next": {
            "observation": torch.randn(B, 4),
            "terminated": torch.zeros(B, 1),
            "physics": torch.zeros(B, 20),
            "action": torch.rand(B, 2) * 2 - 1,     # SARSA next-action from data
        },
    }
    m = a.update(batch, step=0)
    assert np.isfinite(m["fb_loss"]) and np.isfinite(m["actor_loss"])

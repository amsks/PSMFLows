"""Latent affine PSM: the measure reads the flow's latent, never the raw action."""
import ml_collections
import numpy as np
from hydra import compose, initialize
from omegaconf import OmegaConf

from agents import agents

OB, ACT = 8, 2


def _config(**overrides):
    with initialize(version_base="1.3", config_path="../configs/agent"):
        cfg = compose(config_name="latent_affine_psm")
    config = ml_collections.ConfigDict(OmegaConf.to_container(cfg, resolve=True))
    config["allow_untrained_flow"] = True   # no Stage-A checkpoint in the test env
    for k, v in overrides.items():
        config[k] = v
    return config


def _batch(n=32, seed=0):
    rng = np.random.default_rng(seed)
    return dict(
        observations=rng.standard_normal((n, OB)).astype(np.float32),
        actions=np.clip(rng.standard_normal((n, ACT)), -1, 1).astype(np.float32),
        next_observations=rng.standard_normal((n, OB)).astype(np.float32),
        noise_preimage=rng.standard_normal((n, ACT)).astype(np.float32),
        index=np.arange(n, dtype=np.int64),
        masks=np.ones((n,), np.float32),
    )


def _agent(config=None):
    config = _config() if config is None else config
    return agents["latent_affine_psm"].create(
        0, np.zeros((1, OB), np.float32), np.zeros((1, ACT), np.float32), config)


def test_registered_and_creates():
    config = _config()
    assert config["agent_name"] == "latent_affine_psm"
    agent = _agent(config)
    assert agent.w_inf.shape == (config["d_dim"],)
    assert agent.flow_vf is not None and agent.flow_onestep is not None


def test_update_runs_and_is_finite():
    agent = _agent()
    agent, info = agent.update(_batch())
    for k, v in info.items():
        assert np.isfinite(np.asarray(v)).all(), f"{k} is not finite"


def test_measure_reads_the_latent_not_the_action():
    """The whole point of the agent: perturbing `actions` must not move the loss, and
    perturbing `noise_preimage` must."""
    # ortho_coef=0: the regularizer is a function of the basis at x only, and at 1000 it
    # dominates the loss, so the TD terms are the ones to compare.
    agent = _agent(_config(ortho_coef=0.0))
    batch = _batch()
    sampled = agent.sample_step_inputs(batch, agent.rng)

    def td(b):
        return float(agent.measure_loss(b, sampled, agent.measure.params, agent.w.params)[0])

    base = td(batch)
    same = td(dict(batch, actions=-batch["actions"]))
    assert same == base, "measure still reads batch['actions']"

    moved = td(dict(batch, noise_preimage=batch["noise_preimage"] + 1.0))
    assert moved != base, "measure ignores the latent"


def test_slot_is_clipped_to_u_clip():
    config = _config()
    agent = _agent(config)
    batch = _batch()
    batch["noise_preimage"] = batch["noise_preimage"] * 100.0
    u = np.asarray(agent._slot(batch))
    assert np.abs(u).max() <= config["u_clip"] + 1e-6


def test_codebook_is_latent_prior_within_the_box():
    config = _config()
    agent = _agent(config)
    table = np.asarray(agent.proto[0])
    assert np.abs(table).max() <= config["u_clip"] + 1e-6
    # A prior draw, not RLU's uniform-in-[-2,0) action table.
    assert table.min() < 0 < table.max()
    assert abs(float(table.mean())) < 0.1


def test_sample_actions_decodes_through_the_flow():
    agent = _agent()
    obs = _batch(n=1)["observations"][0]
    a = np.asarray(agent.sample_actions(obs, seed=agent.rng))
    assert a.shape == (ACT,)
    assert np.isfinite(a).all() and np.abs(a).max() <= 1.0 + 1e-6


def test_zeroshot_inference_uses_latents():
    """infer_w_zeroshot must go through the slot, so a dataset without latents fails
    loudly rather than silently inferring w from raw actions."""
    config = _config()
    config["inference"]["mode"] = "zero_shot"
    agent = _agent(config)

    class _DS:
        def __init__(self, b):
            self._b = b

        def sample(self, n):
            return {k: v[:n] for k, v in self._b.items()}

    batch = _batch(n=64)
    inferred = agent.infer_w_zeroshot(_DS(batch), batch["next_observations"][0], num_samples=32)
    assert inferred.w_inf.shape == (config["d_dim"],)
    assert np.isfinite(np.asarray(inferred.w_inf)).all()

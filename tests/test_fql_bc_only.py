"""FQL `bc_only` mode — reward-free behaviour-flow pretraining (psmflow plan Task 1).

PSMFlows needs a behaviour flow G_theta(s,u) trained on the dataset with NO reward
signal, then frozen. FQL already trains exactly that flow (`actor_bc_flow`) as a
by-product of its actor loss, so `bc_only` strips the reward-dependent terms and leaves
`bc_flow_loss + alpha * distill_loss`.

Two properties matter downstream:
  * the update must never touch batch['rewards'] / batch['masks'] — the pretraining
    dataset is reward-free by construction;
  * the checkpoint must keep the SAME param tree (critic present, just untrained), so
    restore_agent round-trips into a default-shaped FQL agent in Tasks 2/3.
"""
import jax
import ml_collections
import numpy as np
from hydra import compose, initialize
from omegaconf import OmegaConf

from agents import agents


def _config(**overrides):
    with initialize(version_base="1.3", config_path="../configs/agent"):
        cfg = compose(config_name="fql")
    c = ml_collections.ConfigDict(OmegaConf.to_container(cfg, resolve=True))
    with c.unlocked():
        c["actor_hidden_dims"] = tuple(c["actor_hidden_dims"])
        c["value_hidden_dims"] = tuple(c["value_hidden_dims"])
        for k, v in overrides.items():
            c[k] = v
    return c


def _rewardfree_batch(n=16, obs=4, act=2, seed=0):
    rng = np.random.default_rng(seed)
    return dict(  # deliberately NO rewards / masks
        observations=rng.standard_normal((n, obs)).astype(np.float32),
        actions=np.clip(rng.standard_normal((n, act)), -1, 1).astype(np.float32),
        next_observations=rng.standard_normal((n, obs)).astype(np.float32),
    )


def _agent(**overrides):
    config = _config(**overrides)
    return agents["fql"].create(0, np.zeros((1, 4), np.float32),
                                np.zeros((1, 2), np.float32), config), config


def _leaves(tree):
    return [np.asarray(x) for x in jax.tree_util.tree_leaves(tree)]


def test_bc_only_updates_without_rewards_or_masks():
    agent, _ = _agent(bc_only=True)
    agent, info = agent.update(_rewardfree_batch())   # would KeyError if rewards were read
    for k in ["actor/actor_loss", "actor/bc_flow_loss", "actor/distill_loss"]:
        assert k in info and np.isfinite(float(info[k])), k
    # the reward-dependent terms must be gone, not merely zero
    assert "critic/critic_loss" not in info
    assert "actor/q_loss" not in info


def test_bc_only_freezes_the_critic():
    agent, _ = _agent(bc_only=True)
    before = _leaves(agent.network.params["modules_critic"])
    agent, _ = agent.update(_rewardfree_batch())
    after = _leaves(agent.network.params["modules_critic"])
    assert all(np.allclose(x, y) for x, y in zip(before, after)), "critic moved in bc_only"


def test_bc_only_keeps_the_param_tree_shape():
    # Checkpoints must round-trip into a default-shaped FQL agent (Tasks 2/3 restore them).
    bc_agent, _ = _agent(bc_only=True)
    rl_agent, _ = _agent(bc_only=False)
    bc_tree = jax.tree_util.tree_structure(bc_agent.network.params)
    rl_tree = jax.tree_util.tree_structure(rl_agent.network.params)
    assert bc_tree == rl_tree, "bc_only changed the param tree; checkpoints will not restore"


def test_bc_only_trains_the_flow():
    agent, _ = _agent(bc_only=True)
    b = _rewardfree_batch()
    losses = []
    for _ in range(60):
        agent, info = agent.update(b)
        losses.append(float(info["actor/bc_flow_loss"]))
    assert np.mean(losses[-10:]) < np.mean(losses[:10]), (losses[:3], losses[-3:])


def test_default_is_off_and_full_fql_still_trains():
    config = _config()
    assert config.get("bc_only", False) is False, "bc_only must default to False"
    agent, _ = _agent()
    b = _rewardfree_batch()
    b["rewards"] = np.zeros((16,), np.float32)
    b["masks"] = np.ones((16,), np.float32)
    agent, info = agent.update(b)
    assert "critic/critic_loss" in info and np.isfinite(float(info["critic/critic_loss"]))
    assert np.isfinite(float(info["actor/q_loss"]))

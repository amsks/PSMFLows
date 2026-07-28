import math

import ml_collections
import numpy as np
from hydra import compose, initialize
from omegaconf import OmegaConf

from agents import agents


def _config():
    with initialize(version_base="1.3", config_path="../configs/agent"):
        cfg = compose(config_name="affine_psm")
    return ml_collections.ConfigDict(OmegaConf.to_container(cfg, resolve=True))


def _batch(n=32, obs=8, act=2):
    rng = np.random.default_rng(0)
    return dict(
        observations=rng.standard_normal((n, obs)).astype(np.float32),
        actions=np.clip(rng.standard_normal((n, act)), -1, 1).astype(np.float32),
        next_observations=rng.standard_normal((n, obs)).astype(np.float32),
        index=np.arange(n, dtype=np.int64),
        masks=np.ones((n,), np.float32),
    )


def _agent():
    config = _config()
    return agents["affine_psm"].create(0, np.zeros((1, 8), np.float32),
                                       np.zeros((1, 2), np.float32), config)


def test_affine_psm_registered_and_creates():
    config = _config()
    assert config["agent_name"] == "affine_psm"
    cls = agents["affine_psm"]
    agent = cls.create(0, np.zeros((1, 8), np.float32), np.zeros((1, 2), np.float32), config)
    assert agent.w_inf.shape == (config["d_dim"],)
    assert agent.actor is not None


def test_update_runs_and_is_finite():
    agent = _agent()
    agent, info = agent.update(_batch())
    for k in ["psm_loss", "psm_diag", "psm_offdiag"]:
        assert math.isfinite(float(info[k])), (k, info[k])


def test_training_decreases_loss():
    # The objective is stochastic per step (a fresh codebook code z is sampled each
    # update, so the target shifts), so compare windowed averages rather than single
    # steps. Training over ~80 steps must reduce the mean contrastive loss.
    agent = _agent()
    b = _batch()
    losses = []
    for _ in range(80):
        agent, info = agent.update(b)
        losses.append(float(info["psm_loss"]))
    assert np.mean(losses[-15:]) < np.mean(losses[:15])


def test_measure_and_w_receive_gradient():
    import jax
    agent = _agent()
    a0 = agent
    a1, _ = agent.update(_batch())

    def changed(t0, t1):
        leaves0 = jax.tree_util.tree_leaves(t0)
        leaves1 = jax.tree_util.tree_leaves(t1)
        return any(not np.allclose(np.asarray(x), np.asarray(y)) for x, y in zip(leaves0, leaves1))

    assert changed(a0.measure.params, a1.measure.params)
    assert changed(a0.w.params, a1.w.params)


def test_proto_branch_consumes_batch_index():
    # The codebook policy pi_z must key on the GLOBAL buffer row index, not on batch
    # position. Without batch['index'] the agent falls back to arange(B), so pi_z becomes
    # a function of where a transition lands in the batch and the TD bootstrap target is
    # re-randomized on every resample — there is then no policy whose measure is learned.
    # main.py must set dataset.return_index for affine_psm (it once set it only for psm).
    agent = _agent()
    b = _batch()
    b_noidx = {k: v for k, v in b.items() if k != "index"}
    b_idx = dict(b, index=np.arange(1000, 1000 + b["observations"].shape[0], dtype=np.int64))
    _, i1 = agent.update(b_noidx)          # arange(B) fallback
    _, i2 = agent.update(b_idx)            # global-index keyed
    assert math.isfinite(float(i2["psm_offdiag"]))
    # Different proto keys -> different bootstrap actions -> different TD residual.
    # Compare psm_offdiag: psm_diag reads only the prediction M (no proto action), and
    # the psm_loss total is swamped by the index-independent ortho term.
    assert not np.isclose(float(i1["psm_offdiag"]), float(i2["psm_offdiag"]))


def test_main_sets_return_index_for_affine_psm():
    # Guards the main.py wiring itself, which is where the bug actually lived.
    import re
    src = open("main.py").read()
    m = re.search(r"if config\['agent_name'\] in \(([^)]*)\):\s*\n(?:\s*#.*\n)*\s*dataset\.return_index = True", src)
    assert m is not None, "main.py no longer gates return_index on an agent-name tuple"
    assert "'affine_psm'" in m.group(1), "affine_psm missing from the return_index agents"

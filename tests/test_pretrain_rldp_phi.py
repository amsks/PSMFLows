"""tools/pretrain_rldp_phi.py: the policy-free RLDP basis and its hand-off to psmflow.

Run module-per-process: `pytest tests/test_pretrain_rldp_phi.py`.
"""
import json
import os

import jax
import jax.numpy as jnp
import numpy as np

from tests.test_psmflow_agent import ACT, OBS, _agent
from tools.pretrain_rldp_phi import (
    SegmentSampler,
    episode_ends,
    gram_deviation,
    init_state,
    make_update,
    parse_args,
    save_checkpoint,
    train,
)


def _episodic_dataset(lengths, obs_dim=OBS, act_dim=ACT, seed=0):
    """observations[:, 0] = episode id, [:, 1] = step within the episode; the rest noise."""
    rng = np.random.default_rng(seed)
    n = int(sum(lengths))
    obs = rng.standard_normal((n, obs_dim)).astype(np.float32)
    terminals = np.zeros(n, np.float32)
    row = 0
    for ep, length in enumerate(lengths):
        obs[row:row + length, 0] = ep
        obs[row:row + length, 1] = np.arange(length)
        terminals[row + length - 1] = 1.0
        row += length
    next_obs = np.roll(obs, -1, axis=0).copy()
    # the terminal row's successor is its own final state, not the next episode's first row
    next_obs[terminals > 0.5] = obs[terminals > 0.5] + np.array([0, 1] + [0] * (obs_dim - 2),
                                                                np.float32)
    actions = np.clip(rng.standard_normal((n, act_dim)), -1, 1).astype(np.float32)
    return obs, actions, next_obs, terminals


def test_episode_ends_marks_the_last_row_of_each_episode_and_the_final_row():
    term = np.array([0, 0, 1, 0, 1, 0, 0], np.float32)   # last episode unterminated
    np.testing.assert_array_equal(episode_ends(term), [2, 2, 2, 4, 4, 6, 6])


def test_segment_sampler_never_crosses_an_episode_boundary():
    lengths = [7, 3, 12, 5, 4]                            # 3 and 4 are shorter than H=5
    obs, act, next_obs, term = _episodic_dataset(lengths)
    H = 5
    sampler = SegmentSampler(obs, act, next_obs, term, H, seed=1)
    # exactly the rows with >= H transitions left in their episode can start a segment
    assert sampler.starts.size == (7 - H + 1) + (12 - H + 1) + (5 - H + 1)
    batch = sampler.sample(2000)
    ep0, t0 = batch["s0"][:, 0], batch["s0"][:, 1]
    ep_t, t_t = batch["targets"][:, :, 0], batch["targets"][:, :, 1]
    assert np.all(ep_t == ep0[:, None]), "a target row belongs to another episode"
    assert np.all(t_t == t0[:, None] + np.arange(1, H + 1)[None, :]), "targets are not consecutive"
    assert not np.any(np.isin(ep0, [1, 4])), "a segment started in an episode shorter than H"
    assert batch["actions"].shape == (2000, H, ACT)
    # the rows the segments started from are exactly (a subset of) the valid starts
    row_of = {(int(e), int(t)): r for r, (e, t) in enumerate(obs[:, :2])}
    rows = {row_of[(int(e), int(t))] for e, t in zip(ep0, t0)}
    assert rows <= set(sampler.starts.tolist())
    assert len(rows) == sampler.starts.size, "2000 draws over 14 starts should hit every start"


def test_loss_is_finite_and_decreases_over_20_steps():
    rng = np.random.default_rng(0)
    n, obs_dim, act_dim = 4000, OBS, ACT
    # linear dynamics with a small noise term, episodes of 50
    A = 0.9 * np.eye(obs_dim, dtype=np.float32)
    Bm = rng.standard_normal((act_dim, obs_dim)).astype(np.float32) * 0.5
    obs = np.zeros((n, obs_dim), np.float32)
    act = np.clip(rng.standard_normal((n, act_dim)), -1, 1).astype(np.float32)
    term = np.zeros(n, np.float32)
    for i in range(n):
        if i % 50 == 0:
            obs[i] = rng.standard_normal(obs_dim)
        elif i > 0:
            obs[i] = obs[i - 1] @ A + act[i - 1] @ Bm + 0.01 * rng.standard_normal(obs_dim)
        if i % 50 == 49:
            term[i] = 1.0
    next_obs = np.roll(obs, -1, axis=0).copy()
    next_obs[term > 0.5] = obs[term > 0.5] @ A + act[term > 0.5] @ Bm
    sampler = SegmentSampler(obs, act, next_obs, term, 5, seed=0)
    state = init_state(obs_dim, act_dim, z_dim=8, phi_hidden_dim=32, phi_hidden_layers=2,
                       pred_hidden_dim=32, pred_hidden_layers=2, lr=1e-3, seed=0)
    update = make_update(state["phi_def"], state["pred_def"], state["tx"], lam=1.0, tau=0.01)
    params, target, opt = state["params"], state["target_phi"], state["opt_state"]
    losses = []
    for _ in range(20):
        batch = {k: jnp.asarray(v) for k, v in sampler.sample(256).items()}
        params, target, opt, info = update(params, target, opt, batch)
        losses.append(float(info["loss"]))
        assert np.isfinite(float(info["dyn_loss"])) and np.isfinite(float(info["orth_loss"]))
    assert np.all(np.isfinite(losses))
    assert np.mean(losses[-5:]) < np.mean(losses[:5]), losses
    # the target tracks the online phi: after 20 Polyak steps it has moved off its init
    assert not all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(
        jax.tree_util.tree_leaves(target), jax.tree_util.tree_leaves(state["target_phi"])))


def test_saved_pickle_loads_through_phi_restore_path_and_reproduces_phi(tmp_path):
    """The checkpoint carries phi at the pytree path `_load_phi_params` reads, with the
    architecture of the test agent's phi (z_dim 16, hidden 256 x 2, sqrt(d) sphere)."""
    obs, act, next_obs, term = _episodic_dataset([40, 40, 40])
    run_dir = str(tmp_path / "rldp")
    args = parse_args([
        "--out_dir", run_dir, "--steps", "3", "--batch_size", "16", "--horizon", "2",
        "--z_dim", "16", "--phi_hidden_dim", "256", "--phi_hidden_layers", "2",
        "--pred_hidden_dim", "16", "--pred_hidden_layers", "1", "--save_interval", "3",
        "--log_interval", "1",
    ])
    npz = tmp_path / "data.npz"
    np.savez(npz, observations=obs, actions=act, next_observations=next_obs, terminals=term)
    args.dataset_npz = str(npz)
    params, _, curve = train(args)
    assert os.path.isfile(os.path.join(run_dir, "params_3.pkl"))
    assert os.path.isfile(os.path.join(run_dir, "flags.json"))
    with open(os.path.join(run_dir, "rldp_curve.json")) as f:
        assert json.load(f)["curve"][-1]["step"] == 3
    assert len(curve) >= 2

    agent = _agent(phi_restore_path=run_dir, phi_restore_epoch=3, train_phi=False)
    probe = np.random.default_rng(3).standard_normal((32, OBS)).astype(np.float32)
    from utils.psm_networks import PhiMap
    phi_def = PhiMap(z_dim=16, hidden_dim=256, hidden_layers=2, norm=True)
    want = np.asarray(phi_def.apply({"params": params["phi"]}, probe))
    got = np.asarray(agent.phi.apply_fn({"params": agent.phi.params}, probe))
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6)
    got_target = np.asarray(agent.phi.apply_fn({"params": agent.target_phi}, probe))
    np.testing.assert_allclose(got_target, want, rtol=1e-6, atol=1e-6)
    # the loaded phi sits on the sqrt(d) sphere, like the agent's own
    np.testing.assert_allclose(np.linalg.norm(got, axis=-1), np.sqrt(16), rtol=1e-4)
    dev, devn = gram_deviation(phi_def, params["phi"], probe)
    assert np.isfinite(dev) and devn == dev / np.sqrt(16)


def test_save_checkpoint_layout_matches_save_agent():
    import pickle
    state = init_state(OBS, ACT, z_dim=4, phi_hidden_dim=8, phi_hidden_layers=1,
                       pred_hidden_dim=8, pred_hidden_layers=1, lr=1e-3, seed=0)
    d = "/tmp/psmflow_rldp_test_ckpt"
    os.makedirs(d, exist_ok=True)
    path = save_checkpoint(d, 5, state["params"], state["target_phi"])
    with open(path, "rb") as f:
        saved = pickle.load(f)
    assert set(saved["agent"]) == {"phi", "target_phi", "predictor"}
    assert "params" in saved["agent"]["phi"] and saved["step"] == 5
    assert set(saved["agent"]["phi"]["params"]) == set(state["params"]["phi"])

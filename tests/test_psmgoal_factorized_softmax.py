"""psmgoal measure_form=factorized under measure_loss=softmax (2026-11-06): the ladder arm
"BASE + the factorised net only" (gc_scripts/ladder, BASE = reward index, ridge, softmax, flowbc actor,
acting=distill).

Checks: BASE (joint, softmax) and measure_form=factorized with the squared loss are bit-identical to the
snapshot taken before the wiring (tests/fixtures/psmgoal_factorized_softmax_golden.npz,
tests/psmgoal_factorized_golden.py); the softmax logits equal an explicit einsum over A, beta, f and the index
w, and the loss equals the joint path's cross-entropy on them; the gradient reaches A, beta and f; f is
evaluated once per mesh state, never per (row, state) pair; the loss is finite and falls over 50 steps; the
index feature is f(s+) at the slow tower; the gpi (ridge_score=goal, pool_softmax) and distill readouts act;
the flowbc actor's Q reads the factored M; the joint-only seams assert. Optionally a 200-step smoke through
main.py (skipped without the cube artifacts).

Run in its own process: JAX_PLATFORMS=cpu .venv/bin/python -m pytest
tests/test_psmgoal_factorized_softmax.py -q -p no:cacheprovider
"""
import glob
import json
import os
import re
import subprocess
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agents.psmgoal import PSMGoalAgent, sample_z_bin, softmax_td_loss
from tests.psmgoal_factorized_golden import ARMS, BASE, GOLDEN, snapshot
from tests.psmgoal_golden import DA, OB, N, Z, tiny_batch, tiny_cfg
from utils.psm_networks import FactorizedMeasure

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PSM_DATA = os.environ.get("PSM_DATA", "/mnt/home/amohan/psm-data")
# the arm: BASE + the factored basis; f_dim = z_dim so the ridge index reads f(s+) (tiny: Z)
ARM = {**BASE, "measure_form": "factorized", "f_dim": Z}


def _agent(seed=0, **over):
    return PSMGoalAgent.create(seed, np.zeros((2, OB), np.float32), np.zeros((2, DA), np.float32),
                               tiny_cfg(**{**ARM, **over}))


def _rows(ag, batch):
    """The batch with the reward-index rows, plus (z, u_next) as `update` builds them."""
    batch = {**batch, **ag.reward_index_rows(batch)}
    z = sample_z_bin(jax.random.PRNGKey(5), N, int(ag.config["max_log_seed"]))
    return batch, z, ag._select_u_next(batch, z)


def _explicit_M(ag, params, obs, u, goals, w):
    """M_ij = (A(s_i,u_i)^T f(g_j))^T w_i + beta(s_i,u_i)^T f(g_j), written out."""
    A, beta = ag.basis(obs, u, params=params, method="operators")       # (N, K, D), (N, K)
    f = ag.basis(goals, params=params, method="features")               # (G, K)
    return jnp.einsum("nkd,gk,nd->ng", A, f, w) + jnp.einsum("nk,gk->ng", beta, f)


# ------------------------------------------------------------------ golden
@pytest.mark.parametrize("arm", list(ARMS))
def test_golden_bit_identical(arm):
    ref = np.load(GOLDEN)
    now = snapshot(arm)
    keys = sorted(k for k in ref.files if k.startswith(arm + "/"))
    assert keys == sorted(now), set(keys) ^ set(now)
    for k in keys:
        assert np.array_equal(ref[k], now[k]), k


# ------------------------------------------------------------------ construction
def test_arm_creates_and_joint_only_seams_assert():
    ag = _agent()
    assert isinstance(ag.basis.model_def, FactorizedMeasure)
    assert ag._factorized() and ag._softmax() and ag._reward_index()
    bad = [
        ({"f_dim": Z + 1}, "f_dim must equal z_dim"),
        # critic_heads > 1 is wired for the factorized form since 2026-11-16 (tests/test_psmgoal_factorized_heads_index_input.py)
        ({"critic_action_input": "latent_and_action"}, "critic_action_input=latent"),
        ({"measure": {"hidden_dim": 16, "hidden_layers": 2, "layer_norm": True}}, "layer_norm"),
        ({"policy_index": "code", "coef_source": "lp", "train_actor": False, "actor_kind": "tanh",
          "acting": "gpi", "measure_loss": "pool"}, "measure_loss=pool"),
    ]
    for over, msg in bad:
        with pytest.raises(AssertionError, match=msg):
            _agent(**over)
    # actor_value (the Factored-FB actor) stays unwired for the factored basis
    with pytest.raises(AssertionError, match="actor_value"):
        _agent(policy_index="code", coef_source="ridge", actor_kind="tanh", acting="gpi", measure_loss="squared",
               actor_value="measure_reward_softmax")


# ------------------------------------------------------------------ the logits and the loss
def test_logits_are_the_explicit_einsum_and_loss_is_the_joint_cross_entropy():
    ag = _agent()
    batch, z, u_next = _rows(ag, tiny_batch())
    c = ag.config
    obs, nxt = jnp.asarray(batch["observations"]), jnp.asarray(batch["next_observations"])
    u = jnp.clip(jnp.asarray(batch["noise_preimage"]), -c["u_clip"], c["u_clip"])
    w = jnp.asarray(batch["index_w"])
    M_ref = _explicit_M(ag, ag.basis.params, obs, u, nxt, w)
    assert np.allclose(np.asarray(ag.mesh_M(obs, u, nxt, w)), np.asarray(M_ref), atol=1e-5)
    # the loss is softmax_td_loss on these logits and the target nets' logits at (s', u', w)
    M_bar = _explicit_M(ag, ag.target_basis, nxt, u_next, nxt, w)
    want = softmax_td_loss(M_ref, M_bar, float(c["discount"]), float(c["measure_temp"]))
    loss, info = ag.measure_loss(ag.basis.params, ag.w.params, batch, z, u_next)
    assert np.isfinite(float(loss)) and abs(float(loss) - float(want)) < 1e-5
    assert "f_ortho_loss" in info and np.isfinite(float(info["f_ortho_loss"]))
    # f_ortho_coef does not enter the softmax loss
    ag2 = _agent(f_ortho_coef=0.0)
    loss2, _ = ag2.measure_loss(ag.basis.params, ag.w.params, batch, z, u_next)
    assert float(loss2) == float(loss)


def test_gradient_reaches_A_beta_and_f():
    ag = _agent()
    batch, z, u_next = _rows(ag, tiny_batch())
    g = jax.grad(lambda bp: ag.measure_loss(bp, ag.w.params, batch, z, u_next)[0])(ag.basis.params)
    for name in ("A_out", "beta_out", "f_net"):
        norm = float(jnp.sqrt(sum(jnp.sum(x ** 2) for x in jax.tree_util.tree_leaves(g[name]))))
        assert np.isfinite(norm) and norm > 0.0, (name, norm)


def test_f_is_evaluated_once_per_mesh_state(monkeypatch):
    calls = []
    orig = FactorizedMeasure.features

    def spy(self, g):
        calls.append(tuple(g.shape))
        return orig(self, g)

    monkeypatch.setattr(FactorizedMeasure, "features", spy)
    ag = _agent()
    batch, z, u_next = _rows(ag, tiny_batch())
    calls.clear()
    ag.measure_loss(ag.basis.params, ag.w.params, batch, z, u_next)
    # the online mesh, the target mesh, and the f_ortho diagnostic: each on the S = N mesh states
    assert len(calls) == 3, calls
    assert all(s == (N, OB) for s in calls), calls          # never (N * S, OB)


def test_loss_finite_and_decreasing_over_50_steps():
    ag = _agent()
    batch = tiny_batch()
    losses = []
    for _ in range(50):
        ag, info = ag.update(batch)
        losses.append(float(info["psm_loss"]))
    assert np.all(np.isfinite(losses))
    assert np.mean(losses[-10:]) < np.mean(losses[:10]), (losses[:10], losses[-10:])
    assert np.isfinite(float(info["actor_q"])) and np.isfinite(float(info["index_proxy_corr"]))


# ------------------------------------------------------------------ the reward index and the readouts
def test_index_feature_is_f_at_the_slow_tower():
    ag = _agent()
    ag, _ = ag.update(tiny_batch())
    nxt = jnp.asarray(tiny_batch(seed=1)["next_observations"])
    f = ag.fb_feature(nxt, ag.fb_zenc, ag.fb_anchor)
    assert f.shape == (N, Z)
    assert np.allclose(np.asarray(f), np.asarray(ag._features_fwd(nxt, params=ag.fb_zenc)), atol=1e-6)
    assert np.allclose(np.linalg.norm(np.asarray(f), axis=-1), np.sqrt(Z), atol=1e-4)
    # the index rows are ridge solves on it: sqrt(D) sphere
    rows = ag.reward_index_rows(tiny_batch(seed=1))
    assert np.allclose(np.linalg.norm(np.asarray(rows["index_w"]), axis=-1), np.sqrt(Z), atol=1e-3)


@pytest.mark.parametrize("over", [
    {"acting": "gpi", "ridge_score": "goal"},
    {"acting": "gpi", "ridge_score": "pool_softmax", "index_pool_m": 4},
    {"acting": "distill"},
])
def test_readouts_act(over):
    ag = _agent(**over)
    ag, _ = ag.update(tiny_batch())
    b = tiny_batch(seed=2)
    r = np.zeros((N,), np.float32)
    r[:3] = 1.0                      # rewarding rows (> REWARDING_THRESHOLD)
    ag = ag.infer_eval_goals(b, r)
    assert np.allclose(np.linalg.norm(np.asarray(ag.eval_w_star)), np.sqrt(Z), atol=1e-3)
    a = ag.sample_actions(np.ones((OB,), np.float32), seed=jax.random.PRNGKey(1))
    assert a.shape == (DA,) and np.all(np.isfinite(np.asarray(a)))


def test_flowbc_actor_q_reads_the_factored_M():
    ag = _agent()
    batch, _, _ = _rows(ag, tiny_batch())
    c = ag.config
    obs, pool = jnp.asarray(batch["observations"]), jnp.asarray(batch["next_observations"])
    w_a, f_pool = jnp.asarray(batch["index_w_actor"]), jnp.asarray(batch["index_f"])
    r = ag.reward_actor_rows(batch, w_a, f_pool - ag.fb_mu[None])
    eps = jax.random.normal(jax.random.PRNGKey(9), (N, DA))
    u_data = jnp.clip(jnp.asarray(batch["noise_preimage"]), -c["u_clip"], c["u_clip"])
    _, info = ag.actor_loss_flowbc_reward(ag.delta.params, obs, pool, w_a, r, eps, u_data)
    u = ag.flowbc_latent(obs, w_a, eps, params=ag.delta.params)[0]
    M = _explicit_M(ag, ag.basis.params, obs, u, pool, w_a)
    q = jnp.sum(jax.nn.softmax(M / float(c["measure_temp"]), axis=1) * r, axis=1)
    assert abs(float(info["actor_q"]) - float(jnp.mean(q))) < 1e-5


# ------------------------------------------------------------------ smoke through main.py (gated)
FLOW = os.path.join(PSM_DATA, "flow", "cube-single-play")
PREIMAGES = os.path.join(PSM_DATA, "preimages", "cube-single-play.npz")
LADDER = os.path.join(PSM_DATA, "gc_scripts", "ladder")


@pytest.mark.skipif(not (os.path.isfile(os.path.join(FLOW, "params_500000.pkl")) and os.path.isfile(PREIMAGES)
                         and os.path.isfile(os.path.join(LADDER, "check_flags.py"))),
                    reason="cube Stage-A flow / preimages / ladder scripts not on this machine")
def test_smoke_main_py_factorized_arm():
    extra = None
    with open(os.path.join(LADDER, "arms.tsv")) as fh:
        for line in fh:
            name, _, rest = line.rstrip("\n").partition("\t")
            if name == "factorized":
                extra = rest
    assert extra, "add the factorized arm to arms.tsv"
    assert "," not in extra and "agent.measure_form=factorized" in extra.split()
    save_dir = os.path.join(PSM_DATA, "exp", "smoke_ladder")
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "MUJOCO_GL": "egl", "WANDB_MODE": "offline",
           "WANDB_DIR": os.path.join(save_dir, "wandb"), "PSM_DATA": PSM_DATA}
    env.setdefault("OGBENCH_DATASET_DIR", "/mnt/home/amohan/.ogbench/data")
    cmd = [sys.executable, "main.py", "agent=psmgoal", "env_name=cube-single-play-singletask-v0",
           f"agent.flow_ckpt_path={FLOW}", "agent.flow_ckpt_epoch=500000", f"agent.preimage_path={PREIMAGES}",
           *extra.split(), "offline_steps=200", "eval_interval=100", "eval_episodes=2", "log_interval=50",
           "save_interval=200", f"save_dir={save_dir}", "run_group=ladder_factorized", "seed=0"]
    t0 = time.time()
    out = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=3600, check=False)
    wall = time.time() - t0
    assert out.returncode == 0, out.stderr[-4000:]
    runs = sorted(glob.glob(os.path.join(save_dir, "PSMFLows", "ladder_factorized", "sd000*")), key=os.path.getmtime)
    assert runs, "no run dir written"
    flags = os.path.join(runs[-1], "flags.json")
    chk = subprocess.run([sys.executable, os.path.join(LADDER, "check_flags.py"), "factorized", flags],
                         cwd=REPO, capture_output=True, text=True, check=False)
    assert chk.returncode == 0, chk.stdout + chk.stderr
    with open(flags) as fh:
        agent_flags = json.load(fh)["agent"]
    assert agent_flags["measure_form"] == "factorized" and agent_flags["measure_loss"] == "softmax"
    assert agent_flags["f_dim"] == agent_flags["z_dim"]
    rate = re.findall(r"epoch_time[^\d]*([\d.]+)", out.stdout + out.stderr)
    print(f"smoke wall {wall:.0f}s, epoch_time {rate[-1] if rate else 'n/a'} s/step, run {runs[-1]}")

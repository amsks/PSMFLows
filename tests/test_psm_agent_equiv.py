import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import flax  # noqa: E402

from agents.psm import PSMAgent, contrastive_loss, ortho_loss, proto_sample  # noqa: E402
from utils.torch_to_flax import (  # noqa: E402
    load_actor_params, load_phi_params, load_psi_params,
)

FIX = np.load("tests/fixtures/psm_reference.npz")
B = FIX["in__obs"].shape[0]
OFF = 1 - jnp.eye(B, dtype=jnp.float64)
OFF_SUM = OFF.sum()
DISC = 0.98 * jnp.ones((B, 1), jnp.float64)
MAX_SEED = 2 ** 4 + 20000  # max_log_seed=4 (proto_sampler: 2**L + 20000)


def test_contrastive_matches_fixture():
    M = jnp.asarray(FIX["out__M_psm"], jnp.float64)
    tM = jnp.asarray(FIX["out__target_M_psm"], jnp.float64)
    _, diag, offd = contrastive_loss(M, tM, DISC, OFF, OFF_SUM)
    assert np.allclose(float(offd), FIX["out__psm_offdiag"], atol=1e-10, rtol=0)
    assert np.allclose(float(diag), FIX["out__psm_diag"], atol=1e-10, rtol=0)


def test_ortho_matches_fixture():
    phi = jnp.asarray(FIX["out__phi"], jnp.float64)
    _, diag, offd = ortho_loss(phi, OFF, OFF_SUM)
    assert np.allclose(float(offd), FIX["out__orth_offdiag"], atol=1e-10, rtol=0)
    assert np.allclose(float(diag), FIX["out__orth_diag"], atol=1e-10, rtol=0)


def test_proto_sample_matches_fixture():
    table = jnp.asarray(FIX["proto_seed_to_action"], jnp.float64)
    powers = jnp.asarray(FIX["proto_powers"], jnp.float64)
    obs_hash = jnp.asarray(FIX["in__next_obs_hash"], jnp.float64)
    z = jnp.asarray(FIX["in__z_psm"], jnp.float64)
    out = proto_sample(table, powers, obs_hash, z, MAX_SEED)
    assert np.allclose(np.asarray(out), FIX["in__proto_next_action"], atol=1e-10, rtol=0)


CONFIG = dict(
    agent_name="psm", z_dim=8, max_log_seed=4, num_parallel=2, discount=0.98, tau=0.01,
    ortho_coef=1.0, mix_ratio=0.5, pessimism_penalty=0.0, actor_pessimism_penalty=0.5,
    actor_std=0.2, stddev_clip=0.3, norm_z=True,
    lr_phi=1e-4, lr_sf=1e-4, lr_actor=1e-4,
    phi=dict(hidden_dim=32, hidden_layers=2),
    sf=dict(hidden_dim=32, hidden_layers=1, embedding_layers=2),
    actor=dict(hidden_dim=32, hidden_layers=1, embedding_layers=2),
)


def _mapped_agent():
    ex_obs = jnp.asarray(FIX["in__obs"], jnp.float64)
    ex_act = jnp.asarray(FIX["in__action"], jnp.float64)
    agent = PSMAgent.create(0, ex_obs, ex_act, CONFIG)
    phi_p = load_phi_params(FIX)
    sf_p = load_psi_params(FIX, "sf_psi")
    psm_p = load_psi_params(FIX, "psm_psi")
    act_p = load_actor_params(FIX)
    params = flax.core.freeze({
        "phi": phi_p, "sf_psi": sf_p, "psm_psi": psm_p, "actor": act_p,
        "target_phi": phi_p, "target_sf_psi": sf_p, "target_psm_psi": psm_p,
    })
    # init optimizer states on the FROZEN params so pytree node types match.
    opt = {k: agent.txs[k].init(params[k]) for k in ["phi", "psm_psi", "sf_psi", "actor"]}
    return agent.replace(params=params, opt_states=opt)


def _batch():
    return dict(
        observations=jnp.asarray(FIX["in__obs"], jnp.float64),
        actions=jnp.asarray(FIX["in__action"], jnp.float64),
        next_observations=jnp.asarray(FIX["in__next_obs"], jnp.float64),
        masks=jnp.ones((B,), jnp.float64),  # mask=1 (terminated=0), matches the fixture disc
    )


def _inj(prefix):
    return {n: jnp.asarray(FIX[f"{prefix}{n}"], jnp.float64) for n in
            ["z_psm", "z_cont", "proto_next_action", "actor_next_action", "actor_sample"]}


def test_agent_static_equiv():
    agent = _mapped_agent()
    info, _ = agent.compute_static(_batch(), _inj("in__"))
    checks = {
        "psm_diag": "out__psm_diag", "psm_offdiag": "out__psm_offdiag",
        "orth_loss": "out__orth_loss", "orth_diag": "out__orth_diag",
        "orth_offdiag": "out__orth_offdiag", "psm_loss": "out__psm_total",
        "sf_loss": "out__sf_loss", "sf_diag": "out__sf_diag", "sf_offdiag": "out__sf_offdiag",
        "actor_loss": "out__actor_loss", "q": "out__q",
    }
    for ik, fk in checks.items():
        assert np.allclose(float(info[ik]), float(FIX[fk]), atol=1e-10, rtol=0), \
            (ik, float(info[ik]), float(FIX[fk]))


def test_agent_perstep_equiv():
    agent = _mapped_agent()
    batch = _batch()
    keys = ["psm_loss", "psm_diag", "psm_offdiag", "orth_loss",
            "sf_loss", "sf_diag", "sf_offdiag", "actor_loss", "q"]
    for i in range(10):
        agent, info = agent.apply_update(batch, _inj(f"step_in__{i}__"))
        for k in keys:
            fk = f"step__{i}__{k}"
            if fk in FIX:
                assert np.allclose(float(info[k]), float(FIX[fk]), atol=1e-8, rtol=0), \
                    (i, k, float(info[k]), float(FIX[fk]))

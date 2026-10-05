"""The Factored-FB induced-reward actor (actor_value != none) is OFF by default, and with it
off the psmgoal update is byte-identical to the code before the port.

The golden digests below were computed with agents/psmgoal.py as it stood before the
actor_value port (2026-10-01), on CPU, from this file's own `_digest`. They cover the
default agent and the train_actor=true (distill) agent, since the port touches both
`_select_u_next` and `apply_update`. If jax/flax are upgraded the digests can change for
reasons unrelated to this code; regenerate them from the pre-port file in that case.

Run in its own process: JAX_PLATFORMS=cpu .venv/bin/python -m pytest
tests/test_psmgoal_fbtrick_defaults_off.py -q -p no:cacheprovider
"""
import hashlib
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import numpy as np

from agents.psmgoal import PSMGoalAgent, get_config

OB, DA, Z, CODE, N = 6, 3, 8, 4, 8

# sha256 of every pre-port field's leaves + every info value after 3 updates.
GOLDEN = {
    "default": "f9156145f0046205f84b1fd181523a25e870bd12f521164bcc1a78398e642f99",
    "train_actor": "f58b74d5c915d2e3efcc495290c72d48ea98eb148690996f58283a3b872ac361",
}


def _cfg(**over):
    c = get_config()
    c.z_dim = Z
    c.max_log_seed = CODE
    c.batch_size = N
    c.k_goals = 4
    c.gpi_num_u = 5
    c.num_inference_steps = 2
    c.infer_batch = N
    c.allow_untrained_flow = True
    c.measure.hidden_dim = 16
    c.w.hidden_dim = 16
    c.l.hidden_dim = 16
    c.actor.hidden_dim = 16
    c.flow.hidden_dims = (16, 16)
    for k, v in over.items():
        setattr(c, k, v)
    return c


def _batch(n=N, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "observations": rng.standard_normal((n, OB)).astype(np.float32),
        "next_observations": rng.standard_normal((n, OB)).astype(np.float32),
        "noise_preimage": rng.standard_normal((n, DA)).astype(np.float32),
        "index": np.arange(n).astype(np.int32),
        "rewards": -np.ones((n,), np.float32),
    }


# The agent fields that existed before the port; new fields are excluded on purpose.
_OLD_FIELDS = ("rng", "basis", "w", "l", "actor", "w_star", "target_basis", "target_w",
               "eval_goals", "eval_w_star")


def _digest(**over):
    ag = PSMGoalAgent.create(0, np.zeros((2, OB), np.float32), np.zeros((2, DA), np.float32),
                             _cfg(**over))
    h = hashlib.sha256()
    for step in range(3):
        ag, info = ag.update(_batch(seed=step))
        for k in sorted(info):
            h.update(k.encode())
            h.update(np.asarray(info[k], np.float32).tobytes())
    for name in _OLD_FIELDS:
        for leaf in jax.tree_util.tree_leaves(getattr(ag, name)):
            h.update(np.asarray(leaf).tobytes())
    return h.hexdigest()


def test_actor_value_defaults_off():
    c = get_config()
    assert c.actor_value == "none"


def test_default_update_byte_identical():
    assert _digest() == GOLDEN["default"]


def test_train_actor_update_byte_identical():
    assert _digest(train_actor=True) == GOLDEN["train_actor"]


if __name__ == "__main__":
    print({"default": _digest(), "train_actor": _digest(train_actor=True)})

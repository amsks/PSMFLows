"""Old checkpoints must keep restoring after `agent` config keys are added.

WHY THIS FILE EXISTS. On 2026-09-07 a few minutes of skew between `agents/psmflow.py` and
`configs/agent/psmflow.yaml` -- the module asserted on `config["actor"]["entropy"]` before
the yaml defined it -- killed twelve queued 500-episode evaluations of OLDER checkpoints at
startup (SLURM 2491907, 2491909-2491916, 2491919-2491921) with

    File "agents/psmflow.py", line 1056, in create
      assert a_cfg["entropy"] in ("auto", "fixed")
    KeyError: "'entropy'"

reached through `tools/eval_checkpoint.py` -> `_evaluate_shard` -> `create`. Nothing caught
it because the whole test suite builds configs from THIS checkout's `get_config()`, which
always has every key. A restored checkpoint's `flags.json` is by construction older than the
code restoring it, so the gap was invisible to every existing test.

Two invariants are pinned here, and both must hold for any future `agent` key:

  1. REAL ARCHIVED CONFIGS STILL BUILD. `tests/fixtures/flags_*.json` are verbatim copies of
     `flags.json` files from runs on disk -- pre-affine (`policy_index=task_vector`, no
     `psi_form`), affine strict, and affine + latent actor. Each goes through the eval tool's
     own `merge_run_config` and then `PSMFlowAgent.create`, on CPU, with no checkpoint. A new
     required key breaks these immediately.
  2. YAML <-> get_config PARITY. The hydra group and the importable defaults must carry the
     same key paths. That single assertion is what would have caught the outage: the module
     read a key the yaml did not define.

Run module-per-process: `pytest tests/test_psmflow_config_compat.py`.
"""
import glob
import json
import os

import ml_collections
import numpy as np
import pytest
import yaml

from agents.psmflow import (
    ACTOR_DEFAULTS,
    STABILITY_DEFAULTS,
    PSMFlowAgent,
    fill_actor_defaults,
    fill_stability_defaults,
    get_config,
)
from main import _lists_to_tuples
from tools.eval_checkpoint import merge_run_config

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURES = os.path.join(REPO, "tests", "fixtures")
YAML_PATH = os.path.join(REPO, "configs", "agent", "psmflow.yaml")

# (fixture, obs dim, action dim, what the run WAS, so the restore is checked not just for
# absence-of-crash but for landing on the old behaviour).
CASES = [
    ("flags_psmflow_task_vector_cube_20260902.json", 28, 5,
     dict(policy_index="task_vector", psi_form="free", train_actor=True, acting="actor")),
    ("flags_affine_strict_cube_20260904.json", 28, 5,
     dict(policy_index="latent", psi_form="affine", train_actor=False, acting="gpi")),
    ("flags_affine_actor_cube_20260904.json", 28, 5,
     dict(policy_index="latent", psi_form="affine", train_actor=True, acting="actor")),
]


def _yaml_agent():
    with open(YAML_PATH) as fh:
        return yaml.safe_load(fh)


def _leaf_paths(d, prefix=""):
    out = set()
    for k, v in d.items():
        if isinstance(v, dict):
            out |= _leaf_paths(v, prefix + k + ".")
        else:
            out.add(prefix + k)
    return out


def _build(merged, ob, act):
    cfg = ml_collections.ConfigDict(_lists_to_tuples(merged))
    cfg["allow_untrained_flow"] = True   # no Stage-A checkpoint on a CPU test box
    cfg["flow_ckpt_path"] = None
    return PSMFlowAgent.create(0, np.zeros((1, ob), np.float32),
                               np.zeros((1, act), np.float32), cfg)


# ------------------------------------------------------- 1. archived configs still build


@pytest.mark.parametrize("fixture,ob,act,expect", CASES)
def test_archived_flags_json_restores_through_the_eval_tool(fixture, ob, act, expect, tmp_path):
    """The exact path the twelve failed evals took: flags.json -> merge -> create."""
    with open(os.path.join(FIXTURES, fixture)) as fh:
        flags = json.load(fh)
    # merge_run_config reads `<restore_path>/flags.json`, so stage the fixture as one.
    with open(tmp_path / "flags.json", "w") as fh:
        json.dump(flags, fh)

    merged, _prov = merge_run_config(_yaml_agent(), str(tmp_path), set())
    agent = _build(merged, ob, act)          # must not raise

    for key, want in expect.items():
        assert agent.config[key] == want, (fixture, key, agent.config[key], want)
    # The new keys land on the values that reproduce the run's original behaviour.
    assert agent.config["actor_mode"] == "ddpg"
    assert agent.config["actor"]["index_panel"] == 0
    for k, v in ACTOR_DEFAULTS.items():
        assert agent.config["actor"][k] == v, (fixture, k)
    # Same contract for the 2026-09-07 loss stabilisers: a run written before they existed
    # restores onto the OFF values, i.e. onto the loss it was actually trained with.
    for k, v in STABILITY_DEFAULTS.items():
        assert agent.config[k] == v, (fixture, k, agent.config[k], v)
        assert k not in flags["agent"], (fixture, k)
    # And the fixture really is an old one -- otherwise this test proves nothing.
    assert "entropy" not in flags["agent"]["actor"]
    assert "actor_mode" not in flags["agent"]


def test_a_bare_old_actor_dict_builds_without_the_eval_tool():
    """Any construction path, not only merge_run_config: an `actor` dict with ONLY the
    pre-2026-09-06 keys must build. main.py hands `create` whatever the hydra group holds."""
    cfg = get_config()
    cfg["actor"] = ml_collections.ConfigDict(dict(
        hidden_dim=512, hidden_layers=2, embedding_layers=2,
        vf_hidden_dim=512, vf_hidden_layers=4, flow_steps=10, bc_coeff=1.0))
    del cfg["actor_mode"]
    cfg["allow_untrained_flow"] = True
    agent = PSMFlowAgent.create(0, np.zeros((1, 6), np.float32),
                                np.zeros((1, 2), np.float32), cfg)
    assert agent.config["actor_mode"] == "ddpg"
    assert agent.config["actor"]["entropy"] == "auto"


def test_fill_actor_defaults_is_idempotent_and_does_not_override():
    cfg = ml_collections.ConfigDict(dict(actor=ml_collections.ConfigDict(
        dict(hidden_dim=512, entropy="fixed", index_panel=7))))
    fill_actor_defaults(cfg)
    fill_actor_defaults(cfg)
    assert cfg["actor"]["entropy"] == "fixed", "an explicit value was overwritten"
    assert cfg["actor"]["index_panel"] == 7
    assert cfg["actor"]["q_coeff"] == ACTOR_DEFAULTS["q_coeff"]
    assert cfg["actor_mode"] == "ddpg"


def test_fill_stability_defaults_is_idempotent_and_does_not_override():
    cfg = ml_collections.ConfigDict({"ortho_mode": "relative", "z_dim": 8})
    fill_stability_defaults(cfg)
    fill_stability_defaults(cfg)
    assert cfg["ortho_mode"] == "relative", "an explicit value was overwritten"
    assert cfg["psi_bound"] == STABILITY_DEFAULTS["psi_bound"]
    assert cfg["psi_bound_scale"] == STABILITY_DEFAULTS["psi_bound_scale"]


def test_every_stability_default_is_declared_in_both_places():
    yaml_agent, py_agent = _yaml_agent(), get_config()
    for k, v in STABILITY_DEFAULTS.items():
        assert k in yaml_agent, f"STABILITY_DEFAULTS[{k!r}] missing from the yaml"
        assert yaml_agent[k] == v, (k, yaml_agent[k], v)
        assert py_agent[k] == v, (k, py_agent[k], v)


def test_fill_actor_defaults_survives_a_config_without_an_actor_block():
    """fql and the other agents have no `actor` sub-dict; the helper must not throw."""
    fill_actor_defaults(ml_collections.ConfigDict(dict(agent_name="fql")))
    fill_actor_defaults({})


# ------------------------------------------------------------------- 2. yaml <-> py parity


def test_yaml_and_get_config_carry_the_same_keys():
    """The assertion that would have caught the 2026-09-07 outage.

    `agents/psmflow.py` reads keys the hydra group must supply. If the module gains a key
    the yaml lacks, every hydra-launched run and every eval dies on a KeyError -- which is
    exactly what happened. Compared as leaf paths so a nested block counts key by key.
    """
    yaml_keys = _leaf_paths(_yaml_agent())
    py_keys = _leaf_paths(get_config().to_dict())
    assert not (py_keys - yaml_keys), (
        f"in get_config() but not configs/agent/psmflow.yaml: {sorted(py_keys - yaml_keys)}")
    assert not (yaml_keys - py_keys), (
        f"in configs/agent/psmflow.yaml but not get_config(): {sorted(yaml_keys - py_keys)}")


def test_every_actor_default_is_declared_in_both_places():
    yaml_actor = _yaml_agent()["actor"]
    py_actor = get_config()["actor"]
    for k, v in ACTOR_DEFAULTS.items():
        assert k in yaml_actor, f"ACTOR_DEFAULTS[{k!r}] missing from the yaml"
        assert yaml_actor[k] == v, (k, yaml_actor[k], v)
        assert py_actor[k] == v, (k, py_actor[k], v)


def test_fixtures_are_present():
    """A silently-empty fixture dir would make the archived-config tests vacuous."""
    found = sorted(os.path.basename(p) for p in glob.glob(os.path.join(FIXTURES, "flags_*.json")))
    assert found == sorted(c[0] for c in CASES), found

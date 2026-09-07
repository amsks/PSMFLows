"""tools/eval_checkpoint.py takes its agent config from the RUN's flags.json.

Why (2026-09-03 latent-actor audit §7): the tool used to build the agent config from the
hydra `agent` group plus whatever was typed on the eval line, and never looked at the run
it was restoring. A run trained off-default -- `policy_index=latent`, `train_actor=false`,
a non-default `u_clip`, `acting=gpi` -- then evaluated a
DIFFERENT policy unless every flag was re-typed. `restore_agent` replaces the parameter
tree without a shape check, so that is loud for a width change and SILENT for `u_clip`
(the actor is tanh * u_clip: every action scaled 3x, no error, a plausible-looking number).

`merge_run_config` is a pure function so this needs no GPU, env, hydra job or checkpoint.
Pinned here:
  - flags.json supplies the defaults and they actually land;
  - an explicitly typed CLI override still wins (deliberate off-config evals stay possible);
  - only keys the current config already has are inherited, so an old or newer flags.json
    cannot add or drop a field;
  - a key this checkout has that the run's flags.json does NOT falls back to that key's
    PRE-2026-09-04 value, not to today's default: a flags.json written before the affine
    flip describes a free-psi, task-vector, actor-driven run, and inheriting today's
    defaults into it would evaluate a different agent (and trip create()'s guard);
  - a checkpoint from a different agent is refused with the reason;
  - no flags.json, an unreadable one, or one without an `agent` block is a no-op, which is
    what keeps every previously recorded eval500 number reproducible.
"""
import json
import os

import pytest

from tools.eval_checkpoint import merge_run_config

# What configs/agent/psmflow.yaml gives before anything is typed on the command line: the
# paper-strict affine agent (default since 2026-09-04).
CLI = {
    "agent_name": "psmflow",
    "u_clip": 3.0,
    "acting": "gpi",
    "policy_index": "latent",
    "psi_form": "affine",
    "train_actor": False,
    "use_point_preimage": True,
    "flow_ckpt_path": None,
    "action_critic": {"enabled": False, "eval_rank_k": 0},
}

# A pre-flip run (task-vector index, free psi, latent actor, u_clip=1.0): every one of
# these differs from the config default above, and it is the case the tool must not get
# wrong -- those checkpoints are still on disk and still get re-evaluated.
RUN_AGENT = {
    "agent_name": "psmflow",
    "u_clip": 1.0,
    "acting": "actor",
    "policy_index": "task_vector",
    "psi_form": "free",
    "train_actor": True,
    "use_point_preimage": False,
    "flow_ckpt_path": "/somewhere/else/flow",
    "action_critic": {"enabled": True, "eval_rank_k": 4},
    "a_key_this_checkout_does_not_have": 7,
}


def _run_dir(tmp_path, agent=RUN_AGENT, **extra):
    d = tmp_path / "sd000_s_1.0"
    d.mkdir()
    flags = {"seed": 0, "env_name": "antmaze-medium-navigate-singletask-v0"}
    if agent is not None:
        flags["agent"] = agent
    flags.update(extra)
    (d / "flags.json").write_text(json.dumps(flags))
    return str(d)


def test_run_flags_supply_the_defaults(tmp_path):
    merged, prov = merge_run_config(CLI, _run_dir(tmp_path), cli_keys=set())
    assert merged["u_clip"] == 1.0
    assert merged["acting"] == "actor"
    assert merged["policy_index"] == "task_vector"
    assert merged["psi_form"] == "free"
    assert merged["train_actor"] is True
    assert merged["action_critic"] == {"enabled": True, "eval_rank_k": 4}
    assert prov["flags_json"].endswith("flags.json")
    assert set(prov["inherited"]) >= {"u_clip", "acting", "policy_index", "psi_form",
                                      "train_actor", "action_critic.enabled"}
    # The caller's own dict is not mutated.
    assert CLI["u_clip"] == 3.0 and CLI["action_critic"]["enabled"] is False


def test_an_explicit_cli_override_still_wins(tmp_path):
    cli = dict(CLI, u_clip=2.5)
    merged, prov = merge_run_config(cli, _run_dir(tmp_path), cli_keys={"u_clip"})
    assert merged["u_clip"] == 2.5, "a typed override must beat the run's flags.json"
    assert "u_clip" not in prov["inherited"]
    assert merged["acting"] == "actor", "untouched keys still come from the run"
    assert prov["cli_overrides"] == ["u_clip"]


def test_a_nested_cli_override_wins(tmp_path):
    cli = json.loads(json.dumps(CLI))
    cli["action_critic"]["eval_rank_k"] = 16
    merged, _ = merge_run_config(cli, _run_dir(tmp_path), cli_keys={"action_critic.eval_rank_k"})
    assert merged["action_critic"]["eval_rank_k"] == 16
    assert merged["action_critic"]["enabled"] is True, "the sibling still comes from the run"


def test_schema_is_the_checkouts_not_the_runs(tmp_path):
    merged, prov = merge_run_config(CLI, _run_dir(tmp_path), cli_keys=set())
    assert "a_key_this_checkout_does_not_have" not in merged
    assert "a_key_this_checkout_does_not_have" in prov["ignored_run_only_keys"]
    assert set(merged) == set(CLI), "flags.json must not add or drop fields"


def test_a_different_agent_is_refused(tmp_path):
    run = dict(RUN_AGENT, agent_name="latentrl")
    with pytest.raises(AssertionError, match="latentrl"):
        merge_run_config(CLI, _run_dir(tmp_path, agent=run), cli_keys=set())


@pytest.mark.parametrize("case", ["missing_dir", "no_flags", "no_agent_block", "unparseable"])
def test_absent_or_unusable_flags_is_a_no_op(tmp_path, case):
    """Every recorded eval500 number must stay reproducible when there is nothing to read."""
    if case == "missing_dir":
        path = str(tmp_path / "nope")
    elif case == "no_flags":
        (tmp_path / "empty").mkdir()
        path = str(tmp_path / "empty")
    elif case == "no_agent_block":
        path = _run_dir(tmp_path, agent=None)
    else:
        path = _run_dir(tmp_path, agent=None)
        with open(os.path.join(path, "flags.json"), "w") as fh:
            fh.write("{not json")
    merged, prov = merge_run_config(CLI, path, cli_keys=set())
    assert merged == CLI
    assert prov["flags_json"] is None


def test_no_restore_path_is_a_no_op():
    merged, prov = merge_run_config(CLI, None, cli_keys=set())
    assert merged == CLI and prov["flags_json"] is None


def test_bc_control_semantics_are_unchanged(tmp_path):
    """`eval500.sh bc` points restore_path at the Stage-A FLOW dir, whose flags.json is
    the fql bc_only run's. Its values already equal the eval defaults, so the BC control
    number (cube 0.072) is not moved by this feature."""
    cli = {"agent_name": "fql", "bc_only": True, "flow_steps": 10, "alpha": 10.0}
    run = {"agent_name": "fql", "bc_only": True, "flow_steps": 10, "alpha": 10.0}
    merged, prov = merge_run_config(cli, _run_dir(tmp_path, agent=run), cli_keys={"bc_only"})
    assert merged == cli
    assert prov["inherited"] == {}


def test_keys_absent_from_flags_json_fall_back_to_the_pre_flip_defaults(tmp_path):
    """A run written before `psi_form` existed has no such key. Leaving this checkout's
    `affine` in place would build a DIFFERENT network from the checkpoint's -- and, with
    the run's `policy_index=task_vector` inherited on top, would trip create()'s guard.
    The pre-2026-09-04 value is what that run actually trained with."""
    old_run = {k: v for k, v in RUN_AGENT.items()
               if k not in ("psi_form", "acting", "policy_index", "train_actor")}
    merged, prov = merge_run_config(CLI, _run_dir(tmp_path, agent=old_run), cli_keys=set())
    assert merged["psi_form"] == "free"
    assert merged["policy_index"] == "task_vector"
    assert merged["acting"] == "actor"
    assert merged["train_actor"] is True
    assert set(prov["legacy_defaults"]) == {"psi_form", "policy_index", "acting", "train_actor"}
    assert merged["u_clip"] == 1.0, "keys the run DOES carry still come from the run"


def test_a_cli_override_beats_the_pre_flip_fallback(tmp_path):
    """Deliberately evaluating an old checkpoint under the new head stays possible."""
    cli = dict(CLI, psi_form="affine")
    old_run = {k: v for k, v in RUN_AGENT.items() if k != "psi_form"}
    merged, prov = merge_run_config(cli, _run_dir(tmp_path, agent=old_run),
                                    cli_keys={"psi_form"})
    assert merged["psi_form"] == "affine"
    assert "psi_form" not in prov["legacy_defaults"]


def test_a_current_run_gets_no_legacy_fallback(tmp_path):
    """Every key present => nothing to back-fill."""
    _, prov = merge_run_config(CLI, _run_dir(tmp_path), cli_keys=set())
    assert prov["legacy_defaults"] == {}


def test_the_training_seed_is_recovered_from_flags_json(tmp_path):
    """`seed` in the report is the EVAL seed (0 on every eval500 line ever run), so a
    checkpoint's own training seed appeared nowhere in its JSON -- only in `restore_path`
    and the filename. Pooling across seeds must be checkable from the report itself."""
    _, prov = merge_run_config(CLI, _run_dir(tmp_path, seed=2), cli_keys=set())
    assert prov["train_seed"] == 2


def test_the_training_seed_is_read_even_without_an_agent_block(tmp_path):
    _, prov = merge_run_config(CLI, _run_dir(tmp_path, agent=None, seed=1), cli_keys=set())
    assert prov["train_seed"] == 1


@pytest.mark.parametrize("case", ["missing_dir", "unparseable"])
def test_the_training_seed_is_none_when_unreadable(tmp_path, case):
    if case == "missing_dir":
        path = str(tmp_path / "nope")
    else:
        path = _run_dir(tmp_path, agent=None)
        with open(os.path.join(path, "flags.json"), "w") as fh:
            fh.write("{not json")
    _, prov = merge_run_config(CLI, path, cli_keys=set())
    assert prov["train_seed"] is None

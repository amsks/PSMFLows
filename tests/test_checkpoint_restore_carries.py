"""A restored checkpoint must carry the OPTIMISER state and the TARGET networks.

Continuing a run past its original budget -- restore a 500k checkpoint, train 500k more --
is only equivalent to training straight through if Adam's moments and the Polyak targets
come back with the weights. If either is silently re-initialised, the continuation restarts
the optimiser and the bootstrap, and the result is a different experiment from a fresh
long run while looking like the same one.

`save_agent` pickles `flax.serialization.to_state_dict(agent)` over the whole PyTreeNode,
so `TrainState.opt_state` and every `target_*` field are included by construction. This
test is what stops that construction changing quietly. See `docs/HANDOFF.md` 6d for the
save_dir trap that goes with it.
"""
import tempfile

import numpy as np
import pytest

import utils.xla_guard  # noqa: F401  -- MUST precede jax
from tests.test_psmflow_dsrl_na import _na_agent, _na_batch
from utils.flax_utils import restore_agent, save_agent


def _identical(a, b):
    """Exact equality, leaf by leaf. A restore is a copy, so no tolerance is warranted.

    jax is imported HERE, not at module scope: ruff's import sort would hoist a top-level
    `import jax` above `utils.xla_guard`, and that ordering is the invariant conftest.py
    exists to enforce. The diag tools use the same dodge.
    """
    import jax

    la = [np.asarray(x) for x in jax.tree_util.tree_leaves(a)]
    lb = [np.asarray(x) for x in jax.tree_util.tree_leaves(b)]
    if len(la) != len(lb):
        return False
    return all(np.array_equal(x, y) for x, y in zip(la, lb))


@pytest.fixture(scope="module")
def trained_and_restored():
    """An agent moved off its init, saved, and restored into a freshly built one."""
    agent = _na_agent()
    batch = _na_batch(0)
    for _ in range(5):
        agent, _ = agent.update(batch)
    with tempfile.TemporaryDirectory() as d:
        save_agent(agent, d, 100)
        restored = restore_agent(_na_agent(), d, 100)
    return agent, restored


@pytest.mark.parametrize("field", [
    "qa.params", "qa.opt_state", "qa.step",
    "phi.params", "phi.opt_state",
    "psi.params", "psi.opt_state",
    "target_phi", "target_psi", "target_qa",
])
def test_restore_carries(trained_and_restored, field):
    saved, restored = trained_and_restored

    def get(a):
        for part in field.split("."):
            a = getattr(a, part)
        return a

    assert _identical(get(saved), get(restored)), f"restore lost {field}"


@pytest.mark.parametrize("field", ["qa.opt_state", "target_phi"])
def test_the_comparison_is_not_vacuous(trained_and_restored, field):
    """Guard the guard: these fields must actually differ from a fresh agent's.

    Without this, a restore that re-initialised everything would pass the test above on any
    field the five updates happened not to move.
    """
    saved, _ = trained_and_restored
    fresh = _na_agent()

    def get(a):
        for part in field.split("."):
            a = getattr(a, part)
        return a

    assert not _identical(get(saved), get(fresh)), (
        f"{field} is unchanged by 5 updates, so the restore test on it proves nothing")

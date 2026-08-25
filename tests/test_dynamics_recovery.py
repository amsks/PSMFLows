"""Simulator-restore plumbing for tools/validate_dynamics_recovery.py.

The tool's numbers need a real OGBench env and a ~250 MB dataset, so what is checked here
is the part that decides whether they mean anything: that every stepped action starts from
a freshly restored state, and that button-latch envs get their third `set_state` argument.
"""
import numpy as np
import pytest

from tools.validate_dynamics_recovery import _observe, _restore, _step_from


class _FakeSim:
    """Stands in for the unwrapped MuJoCo env: records the calls the tool makes."""

    def __init__(self, obs_fn='compute_observation'):
        self.calls = []
        self.state = None
        setattr(self, obs_fn, self._observe)

    def _observe(self):
        return np.asarray(self.state, np.float32)

    def set_state(self, qpos, qvel, button_states=None):
        self.calls.append(('set_state', np.array(qpos), np.array(qvel), button_states))
        self.state = np.concatenate([qpos, qvel])


class _FakeEnv:
    def __init__(self, sim):
        self.unwrapped = sim

    def reset(self):
        self.unwrapped.calls.append(('reset',))
        self.unwrapped.state = None
        return None, {}

    def step(self, action):
        self.unwrapped.calls.append(('step', np.array(action)))
        return self.unwrapped.state + action.sum(), 0.0, False, False, {}


def _sim_state(n=3, dim=2, buttons=False):
    rng = np.random.default_rng(0)
    out = dict(qpos=rng.standard_normal((n, dim)).astype(np.float32),
               qvel=rng.standard_normal((n, dim)).astype(np.float32))
    if buttons:
        out['button_states'] = np.zeros((n, 2), np.int32)
    return out


@pytest.mark.parametrize('obs_fn', ['compute_observation', 'get_ob'])
def test_observe_accepts_either_env_family(obs_fn):
    """manipspace exposes compute_observation, locomaze get_ob."""
    env = _FakeEnv(_FakeSim(obs_fn))
    state = _sim_state()
    _restore(env, state, 1)
    np.testing.assert_allclose(_observe(env),
                               np.concatenate([state['qpos'][1], state['qvel'][1]]))


def test_restore_resets_first_and_sets_the_requested_row():
    env = _FakeEnv(_FakeSim())
    state = _sim_state()
    _restore(env, state, 2)

    kinds = [c[0] for c in env.unwrapped.calls]
    assert kinds == ['reset', 'set_state'], 'the episode clock must be reset before set_state'
    _, qpos, qvel, buttons = env.unwrapped.calls[1]
    np.testing.assert_array_equal(qpos, state['qpos'][2])
    np.testing.assert_array_equal(qvel, state['qvel'][2])
    assert buttons is None


def test_button_states_are_passed_when_the_dataset_has_them():
    """Puzzle/scene latches live outside qpos/qvel; without them the restore is partial."""
    env = _FakeEnv(_FakeSim())
    state = _sim_state(buttons=True)
    _restore(env, state, 0)
    assert env.unwrapped.calls[1][3] is not None


def test_every_stepped_action_starts_from_a_fresh_restore():
    """Two latent sources must be stepped from the same state, not from each other's
    outcome."""
    env = _FakeEnv(_FakeSim())
    state = _sim_state()
    a, b = np.float32([1.0, 0.0]), np.float32([0.0, 1.0])
    out_a = _step_from(env, state, 1, a)
    out_b = _step_from(env, state, 1, b)

    assert [c[0] for c in env.unwrapped.calls] == ['reset', 'set_state', 'step'] * 2
    np.testing.assert_allclose(out_a, out_b)  # same start, equal action sums

"""bootstrap_source=frozen_delta (2026-10-17): the TD target latent is built from a FROZEN flowbc
delta (plus / minus / random direction of the same length / zero). CPU, allow_untrained_flow."""
import jax
import numpy as np
import pytest

import utils.xla_guard  # noqa: F401  (must precede jax)
from tests.test_psmgoal_bootstrap_delta_scale import OVER, _nonzero_delta, _z
from tests.test_psmgoal_joint_actor import _agent, _batch


def _frozen(base, mode):
    """A frozen_delta agent whose frozen params are base's (non-zero) delta params and whose rng is base's."""
    over = {**OVER, 'bootstrap_source': 'frozen_delta', 'frozen_delta_mode': mode}
    ag = _agent(**over)
    return ag.replace(frozen_delta_params=base.delta.params, rng=base.rng)


def _u(agent, batch):
    return np.asarray(agent._select_u_next(batch, _z(agent, batch)))


def test_plus_equals_the_live_actor_bootstrap():
    base = _nonzero_delta(_agent(**OVER))
    batch = _batch()
    np.testing.assert_allclose(_u(_frozen(base, 'plus'), batch), _u(base, batch), atol=1e-6)


def test_zero_equals_scale_zero():
    base = _nonzero_delta(_agent(**OVER))
    zero = _agent(bootstrap_delta_scale=0.0, **OVER).replace(delta=base.delta, rng=base.rng)
    batch = _batch()
    np.testing.assert_allclose(_u(_frozen(base, 'zero'), batch), _u(zero, batch), atol=1e-6)


def test_minus_mirrors_plus_about_the_draw():
    base = _nonzero_delta(_agent(**OVER))
    batch = _batch()
    u_plus, u_minus, u_eps = _u(_frozen(base, 'plus'), batch), _u(_frozen(base, 'minus'), batch), _u(_frozen(base, 'zero'), batch)
    clip = float(base.config['u_clip'])
    free = np.all(np.abs(u_plus) < clip - 1e-6, axis=1) & np.all(np.abs(u_minus) < clip - 1e-6, axis=1)
    assert free.sum() > 0
    np.testing.assert_allclose(u_minus[free] - u_eps[free], -(u_plus[free] - u_eps[free]), atol=1e-5)


def test_random_has_the_steps_length_and_a_different_direction():
    base = _nonzero_delta(_agent(**OVER))
    batch = _batch()
    u_plus, u_rand, u_eps = _u(_frozen(base, 'plus'), batch), _u(_frozen(base, 'random'), batch), _u(_frozen(base, 'zero'), batch)
    clip = float(base.config['u_clip'])
    free = (np.all(np.abs(u_plus) < clip - 1e-6, axis=1) & np.all(np.abs(u_rand) < clip - 1e-6, axis=1)
            & np.all(np.abs(u_eps) < clip - 1e-6, axis=1))
    assert free.sum() > 0
    d = u_plus[free] - u_eps[free]
    r = u_rand[free] - u_eps[free]
    np.testing.assert_allclose(np.linalg.norm(r, axis=1), np.linalg.norm(d, axis=1), rtol=1e-4, atol=1e-6)
    cos = (d * r).sum(1) / np.maximum(np.linalg.norm(d, axis=1) * np.linalg.norm(r, axis=1), 1e-12)
    assert np.abs(cos).mean() < 0.9          # not the same direction


def test_frozen_params_are_not_updated_by_an_actor_step():
    base = _nonzero_delta(_agent(**OVER))
    ag = _frozen(base, 'plus')
    before = jax.tree_util.tree_leaves(ag.frozen_delta_params)
    new_ag, _ = ag.update(_batch())
    after = jax.tree_util.tree_leaves(new_ag.frozen_delta_params)
    for a, b in zip(before, after):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    # the live actor did move
    live_before = jax.tree_util.tree_leaves(ag.delta.params)
    live_after = jax.tree_util.tree_leaves(new_ag.delta.params)
    assert any(not np.allclose(np.asarray(a), np.asarray(b)) for a, b in zip(live_before, live_after))


def test_bad_mode_is_refused():
    with pytest.raises(AssertionError):
        _agent(**{**OVER, 'bootstrap_source': 'frozen_delta', 'frozen_delta_mode': 'sideways'})

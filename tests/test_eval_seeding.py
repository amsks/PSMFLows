"""Eval-env seeding regression test.

Root cause of run-to-run eval irreproducibility (our two "seed 2" runs gave
different curves for identical weights): the eval env was never seeded. The
reference (`evals/ogbench.py`) re-seeds the eval env with `cfg.seed` on every
eval; ours reset it unseeded, so episode-init states came from OS entropy and
differed every run.

These tests use a tiny mock env whose episode-init RNG mimics gymnasium: a
`reset(seed=s)` pins the generator; an unseeded `reset()` advances it (and, on
the very first unseeded reset, self-seeds from per-instance "entropy"). Success
is a deterministic function of the init draw, so eval stats are reproducible iff
the init RNG is pinned.
"""
import numpy as np

from utils.evaluation import evaluate


class _MockEnv:
    """Episode-init RNG mirrors gymnasium: reset(seed) pins, reset() advances."""

    def __init__(self, entropy):
        self._entropy = entropy          # stands in for OS entropy per process/run
        self._rng = None
        self._t = 0
        self._init = 0.0
        self.inits = []                  # every episode's init draw, for assertions

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        elif self._rng is None:
            self._rng = np.random.default_rng(self._entropy)  # first unseeded reset
        self._init = float(self._rng.random())  # this episode's init state
        self.inits.append(self._init)
        self._t = 0
        return np.zeros(4, np.float32), {"success": float(self._init > 0.5)}

    def step(self, action):
        self._t += 1
        done = self._t >= 3
        info = {"success": float(self._init > 0.5)}
        return np.zeros(4, np.float32), 0.0, done, False, info


class _MockAgent:
    def sample_actions(self, observations, seed=None, temperature=1.0):
        return np.zeros(2, np.float32)


class _StochasticAgent:
    """Records the action-sampling keys it is handed, so the stream can be compared."""

    def __init__(self):
        self.keys = []

    def sample_actions(self, observations, seed=None, temperature=1.0):
        self.keys.append(np.asarray(seed).tolist())
        return np.zeros(2, np.float32)


def _run(env, seed):
    stats, _, _ = evaluate(_MockAgent(), env, num_eval_episodes=8, seed=seed)
    return stats["success"], env.inits


def test_seeded_eval_is_reproducible_across_runs():
    """Same eval seed on envs with DIFFERENT entropy -> identical init sequence
    (hence identical success). This is the fix: pinning the init RNG each eval
    makes eval a function of the weights alone, not of the env's entropy."""
    (succ_a, inits_a) = _run(_MockEnv(entropy=111), seed=42)
    (succ_b, inits_b) = _run(_MockEnv(entropy=999), seed=42)
    assert inits_a == inits_b      # full episode-init stream matches
    assert succ_a == succ_b


def test_unseeded_eval_is_NOT_reproducible():
    """No seed -> entropy leaks in -> different init sequences. Documents the bug
    the fix removes (guards against silently dropping the seed again). Compares
    the full float init stream, not the collapsed mean, so it can't coincide."""
    (_, inits_a) = _run(_MockEnv(entropy=111), seed=None)
    (_, inits_b) = _run(_MockEnv(entropy=999), seed=None)
    assert inits_a != inits_b


def test_seeded_eval_pins_the_action_key_stream():
    """The other half of eval reproducibility: the ACTION noise. It was seeded from OS
    entropy (`np.random.randint`) whatever `seed` said, so a stochastic actor took a
    different action stream on every re-eval of identical weights."""
    a, b, c = _StochasticAgent(), _StochasticAgent(), _StochasticAgent()
    evaluate(a, _MockEnv(entropy=111), num_eval_episodes=4, seed=42)
    evaluate(b, _MockEnv(entropy=999), num_eval_episodes=4, seed=42)
    evaluate(c, _MockEnv(entropy=111), num_eval_episodes=4, seed=7)
    assert a.keys == b.keys        # same eval seed -> same action keys
    assert a.keys != c.keys        # different eval seed -> different action keys


def test_unseeded_eval_does_not_pin_the_action_key_stream():
    a, b = _StochasticAgent(), _StochasticAgent()
    evaluate(a, _MockEnv(entropy=111), num_eval_episodes=4, seed=None)
    evaluate(b, _MockEnv(entropy=111), num_eval_episodes=4, seed=None)
    assert a.keys != b.keys


def test_seed_overrides_env_history():
    """Re-seeding each eval is immune to prior resets having advanced the RNG."""
    env = _MockEnv(entropy=7)
    (_, fresh) = _run(env, seed=42)
    for _ in range(5):          # advance the env's RNG with unseeded resets
        env.reset()
    (_, again) = _run(env, seed=42)
    assert again[-8:] == fresh[-8:]   # last eval's 8 episode inits reproduce

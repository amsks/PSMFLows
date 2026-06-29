"""Regression: OGBenchEvaluator must build its rollout env with the same
frame_stack as training, else the pixel encoder (built for frame_stack*3
channels) gets a 3-channel eval observation and conv2d raises a channel
mismatch. See systematic-debugging of the FB-pixel smoke crash.
"""
import pytest

from evals.ogbench import OGBenchEvaluator


class _Stop(Exception):
    pass


def _patch_capture(monkeypatch):
    """Patch create_ogbench_env to record kwargs then abort run() early
    (before _infer_z / rollout, which need a real env + buffer)."""
    captured = {}

    def fake_create_ogbench_env(task, **kwargs):
        captured["task"] = task
        captured["kwargs"] = kwargs
        raise _Stop

    monkeypatch.setattr("evals.ogbench.create_ogbench_env", fake_create_ogbench_env)
    return captured


def test_evaluator_forwards_frame_stack_to_eval_env(monkeypatch):
    captured = _patch_capture(monkeypatch)
    ev = OGBenchEvaluator(
        domain="cube-single-play-v0",
        agent=None,
        offline_buffer=[],
        obs_type="pixels",
        frame_stack=3,
    )
    with pytest.raises(_Stop):
        ev.run(step=0)
    assert captured["kwargs"].get("frame_stack") == 3, (
        f"eval env must be built with frame_stack=3, got "
        f"{captured['kwargs'].get('frame_stack')!r}"
    )


def test_evaluator_frame_stack_defaults_to_one_state_path(monkeypatch):
    captured = _patch_capture(monkeypatch)
    ev = OGBenchEvaluator(
        domain="cube-single-play-v0",
        agent=None,
        offline_buffer=[],
        obs_type="state",
    )
    with pytest.raises(_Stop):
        ev.run(step=0)
    assert captured["kwargs"].get("frame_stack", 1) == 1, (
        "state path must default frame_stack to 1 (regression-safe)"
    )

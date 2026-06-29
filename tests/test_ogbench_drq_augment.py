from tests.jax_subprocess import run_jax


def test_random_shifts_identity_at_center_shift():
    """shift == pad recovers the original image exactly (replicate-pad then crop
    [pad:pad+h] == identity)."""
    code = """
        import numpy as np
        from ogbench_drq import _shift_batch_np
        rng = np.random.default_rng(0)
        imgs = rng.integers(0, 256, (3, 8, 8, 9), dtype=np.uint8)
        shifts = np.full((3, 2), 2, dtype=np.int32)  # shift == pad -> identity
        out = _shift_batch_np(imgs, shifts, 2)
        assert out.shape == imgs.shape, out.shape
        assert out.dtype == imgs.dtype, out.dtype
        assert np.array_equal(out, imgs)
        print("OK")
    """
    proc = run_jax(code)
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout


def test_random_shifts_batch_shape_determinism_passthrough():
    code = """
        import numpy as np
        from ogbench_drq import random_shifts_batch

        def make_batch():
            img = np.broadcast_to(
                np.arange(8 * 8 * 9, dtype=np.uint8).reshape(8, 8, 9), (4, 8, 8, 9)
            ).copy()
            return {
                "observations": img.copy(),
                "value_goals": img.copy(),
                "actions": np.zeros((4, 5), dtype=np.float32),  # non-image (2D)
            }

        keys = ["observations", "value_goals", "actions"]
        ba, bb = make_batch(), make_batch()
        random_shifts_batch(ba, keys, pad=2, rng=np.random.default_rng(7))
        random_shifts_batch(bb, keys, pad=2, rng=np.random.default_rng(7))

        # shapes preserved
        assert ba["observations"].shape == (4, 8, 8, 9)
        # deterministic under equal seeds
        assert np.array_equal(ba["observations"], bb["observations"])
        assert np.array_equal(ba["value_goals"], bb["value_goals"])
        # non-image key untouched
        assert np.array_equal(ba["actions"], np.zeros((4, 5), dtype=np.float32))
        # independent per-key shifts: obs and goals differ (very likely)
        assert not np.array_equal(ba["observations"], ba["value_goals"])
        print("OK")
    """
    proc = run_jax(code)
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout

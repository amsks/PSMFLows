from tests.jax_subprocess import run_jax


def test_drq_encoder_forward_shape():
    code = """
        import jax, jax.numpy as jnp
        from ogbench_drq import DrQEncoder
        enc = DrQEncoder(feature_dim=256)
        x = jnp.zeros((4, 64, 64, 9), dtype=jnp.uint8)   # NHWC, frame_stack=3 RGB
        params = enc.init(jax.random.PRNGKey(0), x)["params"]
        out = enc.apply({"params": params}, x)
        assert out.shape == (4, 256), out.shape
        # conv trunk spatial reduction 64->31->29->27->25, 25*25*32 = 20000
        kshape = params["proj"]["kernel"].shape
        assert kshape == (20000, 256), kshape
        print("OK")
    """
    proc = run_jax(code)
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout

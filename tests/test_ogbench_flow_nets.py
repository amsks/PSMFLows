from tests.jax_subprocess import run_jax


def test_flow_nets_shapes_and_determinism():
    code = """
        import jax, jax.numpy as jnp
        from ogbench_flow import VectorField, NoiseConditionedActor, compute_flow_actions
        B, obs_dim, z_dim, act = 4, 11, 7, 5
        obs = jnp.zeros((B, obs_dim)); a = jnp.zeros((B, act)); t = jnp.zeros((B, 1)); z = jnp.zeros((B, z_dim)); n = jnp.zeros((B, act))
        vf = VectorField(hidden_dim=512, hidden_layers=4)
        pv = vf.init(jax.random.PRNGKey(0), obs, a, t)["params"]
        ov = vf.apply({"params": pv}, obs, a, t)
        assert ov.shape == (B, act), ov.shape
        ac = NoiseConditionedActor(hidden_dim=512, hidden_layers=2, embedding_layers=2)
        pa = ac.init(jax.random.PRNGKey(1), obs, z, n)["params"]
        oa = ac.apply({"params": pa}, obs, z, n)
        assert oa.shape == (B, act), oa.shape
        assert float(jnp.max(jnp.abs(oa))) <= 1.0  # tanh output
        roll = compute_flow_actions(vf, pv, obs, n, flow_steps=10)
        assert roll.shape == (B, act)
        # determinism: same params+input -> same output
        oa2 = ac.apply({"params": pa}, obs, z, n)
        assert float(jnp.max(jnp.abs(oa - oa2))) == 0.0
        print("OK")
    """
    proc = run_jax(code)
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "OK" in proc.stdout

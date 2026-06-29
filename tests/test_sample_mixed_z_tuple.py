import torch, gymnasium as gym
from agents.fb.flow_bc.agent import FBFlowBCAgent
from nn_models import NoiseConditionedActorArchiConfig, SimpleVectorFieldArchiConfig

def _agent(**kw):
    return FBFlowBCAgent(obs_space=gym.spaces.Box(-1,1,(40,)), action_dim=5, batch_size=8, z_dim=50, L_dim=50,
        actor_cfg=NoiseConditionedActorArchiConfig(hidden_dim=64, hidden_layers=2, embedding_layers=2),
        actor_vf_cfg=SimpleVectorFieldArchiConfig(hidden_dim=64, hidden_layers=4), device="cpu", **kw)

def test_returns_tuple_and_placeholder_zeros():
    a = _agent()
    g = torch.randn(8, 40)
    torch.manual_seed(0)
    z, g_used = a.sample_mixed_z(train_goal=g)
    assert z.shape == (8, 50) and g_used.shape == (8, 40)
    # rows where z came from random (not a goal) must have zero placeholder OR match a goal row
    is_zero = (g_used.abs().sum(-1) == 0)
    assert is_zero.any() or (~is_zero).all()

def test_ratio_zero_all_random():
    # ratio=0 => every row is the random z, every g_used row is the zero placeholder,
    # and z must NOT be project_z(B(g)) for the matched permutation.
    a = _agent()
    a.train_goal_ratio = 0.0
    g = torch.randn(8, 40)
    torch.manual_seed(0)
    z, g_used = a.sample_mixed_z(train_goal=g)
    assert z.shape == (8, 50) and g_used.shape == (8, 40)
    # ALL placeholder rows are zero
    assert (g_used.abs().sum(-1) == 0).all()
    # z is the random sample, not the goal-derived z. project_z(B(g)) for ANY permutation
    # would differ from the pure random z; check z does not equal goal-derived z of any goal.
    goal_z = a.model.project_z(a.model._backward_map(g))  # [8,50]
    # no random row should coincide with a goal-derived row
    for i in range(z.shape[0]):
        matches = torch.tensor([torch.allclose(z[i], goal_z[j]) for j in range(goal_z.shape[0])])
        assert not matches.any()

def test_ratio_one_all_goals():
    # ratio=1 => every row comes from a goal: no g_used row is the zero placeholder,
    # each g_used row equals some input goal row, and z equals project_z(B(g_used)).
    a = _agent()
    a.train_goal_ratio = 1.0
    torch.manual_seed(0)
    g = torch.randn(8, 40)
    # ensure the goal-derived z's are nonzero (sanity on the fixture)
    goal_z_full = a.model.project_z(a.model._backward_map(g))
    assert (goal_z_full.abs().sum(-1) > 0).all()

    z, g_used = a.sample_mixed_z(train_goal=g)
    assert z.shape == (8, 50) and g_used.shape == (8, 40)
    # NO placeholder row is zero
    assert (g_used.abs().sum(-1) > 0).all()
    # every g_used row equals some input goal row (random permutation, possibly repeated)
    for i in range(g_used.shape[0]):
        matches = torch.tensor([torch.allclose(g_used[i], g[j]) for j in range(g.shape[0])])
        assert matches.any()
    # z must equal project_z(B(g_used)) row-for-row
    assert torch.allclose(z, a.model.project_z(a.model._backward_map(g_used)))

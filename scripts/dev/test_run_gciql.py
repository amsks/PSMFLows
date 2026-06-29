"""Verification for run_gciql.build_ogbench_argv (no Hydra/jax needed)."""
import sys
from pathlib import Path

# Make the repo root importable when run from anywhere (cf. scripts/dev/smoke.py)
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from run_gciql import build_ogbench_argv

STATE = dict(
    variant="state", env_name="cube-single-play-v0", agent_file="agents/gciql.py",
    seed=0, train_steps=1000000, log_interval=5000, eval_interval=100000,
    save_interval=1000000, eval_episodes=50, eval_on_cpu=1,
    run_group="factored-fb-gciql", agent_overrides={"alpha": 1.0},
    output_root="/mnt/scratch1/gciql_outputs",
)
argv = build_ogbench_argv(STATE)
assert argv == [
    "--env_name=cube-single-play-v0",
    "--agent=agents/gciql.py",
    "--seed=0",
    "--train_steps=1000000",
    "--log_interval=5000",
    "--eval_interval=100000",
    "--save_interval=1000000",
    "--eval_episodes=50",
    "--eval_on_cpu=1",
    "--run_group=factored-fb-gciql",
    "--save_dir=/mnt/scratch1/gciql_outputs",
    "--agent.alpha=1.0",
], argv

VISUAL = dict(
    variant="visual", env_name="visual-cube-single-play-v0",
    agent_file="agents/gciql.py", seed=3, train_steps=500000, log_interval=5000,
    eval_interval=100000, save_interval=500000, eval_episodes=50, eval_on_cpu=0,
    run_group="factored-fb-gciql",
    agent_overrides={"alpha": 1.0, "batch_size": 256,
                     "encoder": "impala_small", "p_aug": 0.5},
    output_root="/mnt/scratch1/gciql_outputs",
)
v = build_ogbench_argv(VISUAL)
assert "--env_name=visual-cube-single-play-v0" in v
assert "--seed=3" in v
assert "--agent.batch_size=256" in v
assert "--agent.encoder=impala_small" in v
assert "--agent.p_aug=0.5" in v
assert "--agent.alpha=1.0" in v
# agent overrides come last, sorted by key for determinism
assert v.index("--agent.alpha=1.0") < v.index("--agent.batch_size=256")

# Isolation: the launcher module must NOT import jax/flax.
assert "jax" not in sys.modules and "flax" not in sys.modules, sorted(sys.modules)

print("RUN_GCIQL ARGV OK")

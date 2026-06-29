"""Unit tests for scripts/agents/_common.sh helper functions.
Each test sources the script under bash and asserts on the function's stdout/exit code.
"""
import subprocess
from pathlib import Path

COMMON = Path(__file__).resolve().parents[2] / "scripts" / "agents" / "_common.sh"


def _run(snippet: str, **env_extra) -> subprocess.CompletedProcess:
    """Source _common.sh in a fresh bash, run the snippet, return CompletedProcess."""
    env = {"PATH": "/usr/bin:/bin"}
    env.update(env_extra)
    return subprocess.run(
        ["bash", "-c", f"set -u; source {COMMON}; {snippet}"],
        capture_output=True, text=True, env=env,
    )


def test_common_sh_exists():
    assert COMMON.is_file(), f"missing {COMMON}"


def test_gpu_for_seed_round_robin():
    out = _run('for s in 0 1 2 3; do _gpu_for_seed "$s" "0 1 2"; done').stdout.split()
    assert out == ["0", "1", "2", "0"], out


def test_gpu_for_seed_single_gpu():
    out = _run('_gpu_for_seed 5 "7"').stdout.strip()
    assert out == "7", out


def test_log_dir_format():
    out = _run('_log_dir test_group 3').stdout.strip()
    assert out.startswith("/dev/shm/test_group_"), out
    assert out.endswith("/seed_3.log"), out


def test_dry_run_preview_short_circuits():
    r = _run('_dry_run_preview foo bar baz', DRY_RUN="1")
    assert r.returncode == 0
    assert "[DRY] foo bar baz" in r.stdout


def test_dry_run_preview_no_op_when_unset():
    # _dry_run_preview returns 1 when DRY_RUN is unset (caller proceeds).
    # Capture the function's exit code instead of letting bash exit non-zero.
    r = _run('_dry_run_preview foo bar baz || echo "rc=$?"')
    assert r.returncode == 0
    assert "[DRY]" not in r.stdout
    assert "rc=1" in r.stdout


def test_assert_data_passes_when_dir_exists(tmp_path):
    d = tmp_path / "buf"
    d.mkdir()
    r = _run(f'_assert_data {d}')
    assert r.returncode == 0


def test_assert_data_fails_when_missing(tmp_path):
    r = _run(f'_assert_data {tmp_path / "nope"}')
    assert r.returncode != 0
    assert "MISSING dataset" in r.stderr

"""Integration tests: each scripts/agents/<agent>/<obs>.sh leaf with DRY_RUN=1
must exit 0 and print the expected command tokens. No actual jobs are launched."""
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _dry_run(leaf: str, **env_extra) -> subprocess.CompletedProcess:
    env = {"PATH": "/usr/bin:/bin", "DRY_RUN": "1", "SEEDS": "0", "GPUS": "0"}
    env.update(env_extra)
    return subprocess.run(
        ["bash", str(REPO / "scripts" / "agents" / leaf)],
        capture_output=True, text=True, env=env, cwd=str(REPO),
    )


# ── fb/state.sh ────────────────────────────────────────────────────────────
def test_fb_state_vanilla_dry_run():
    r = _dry_run("fb/state.sh")
    assert r.returncode == 0, r.stderr
    assert "[DRY]" in r.stdout
    assert "train.py" in r.stdout
    assert "domain=cube_single" in r.stdout
    assert "onestep=true" not in r.stdout
    assert "reweight_alpha" not in r.stdout
    assert "CUDA_VISIBLE_DEVICES=0" in r.stdout


def test_fb_state_onestep_mode():
    r = _dry_run("fb/state.sh", MODE="onestep")
    assert r.returncode == 0, r.stderr
    assert "onestep=true" in r.stdout


def test_fb_state_reweight_alpha():
    r = _dry_run("fb/state.sh", REWEIGHT_ALPHA="0.5")
    assert r.returncode == 0, r.stderr
    assert "reweight_alpha=0.5" in r.stdout
    assert "weight_diag=true" in r.stdout
    assert "weight_z=true" in r.stdout
    assert "reweight_density_path=" in r.stdout


def test_fb_state_multi_seed():
    # 3 seeds across 2 GPUs -> 3 [DRY] command lines, one per seed
    r = _dry_run("fb/state.sh", SEEDS="0 1 2", GPUS="0 1")
    assert r.returncode == 0, r.stderr
    assert r.stdout.count("[DRY] CUDA_VISIBLE_DEVICES=") == 3
    assert "seed=0" in r.stdout and "seed=1" in r.stdout and "seed=2" in r.stdout


def test_fb_state_save_eval_videos_default_false():
    r = _dry_run("fb/state.sh")
    assert r.returncode == 0
    assert "save_eval_videos=false" in r.stdout


def test_fb_state_save_eval_videos_true():
    r = _dry_run("fb/state.sh", SAVE_EVAL_VIDEOS="true")
    assert r.returncode == 0
    assert "save_eval_videos=true" in r.stdout


def test_fb_state_help_exits_zero():
    r = subprocess.run(
        ["bash", str(REPO / "scripts/agents/fb/state.sh"), "--help"],
        capture_output=True, text=True, cwd=str(REPO),
        env={"PATH": "/usr/bin:/bin"},
    )
    assert r.returncode == 0
    assert "fb/state.sh" in r.stderr or "fb/state.sh" in r.stdout


# ── fb/pixel.sh ────────────────────────────────────────────────────────────
def test_fb_pixel_vanilla_dry_run():
    r = _dry_run("fb/pixel.sh")
    assert r.returncode == 0, r.stderr
    assert "train.py" in r.stdout
    assert "domain=visual_cube_single" in r.stdout
    assert "[DRY]" in r.stdout


def test_fb_pixel_onestep():
    r = _dry_run("fb/pixel.sh", MODE="onestep")
    assert r.returncode == 0
    assert "onestep=true" in r.stdout
    assert "domain=visual_cube_single" in r.stdout


# ── gciql/state.sh ─────────────────────────────────────────────────────────
def test_gciql_state_dry_run():
    r = _dry_run("gciql/state.sh")
    assert r.returncode == 0, r.stderr
    assert "run_gciql.py" in r.stdout
    assert "--config-name cube_single_state" in r.stdout
    assert "[DRY]" in r.stdout


# ── gciql/pixel.sh ─────────────────────────────────────────────────────────
def test_gciql_pixel_default_impala():
    r = _dry_run("gciql/pixel.sh")
    assert r.returncode == 0, r.stderr
    assert "--config-name cube_single_visual" in r.stdout
    assert "cube_single_visual_drq" not in r.stdout


def test_gciql_pixel_drq_encoder():
    r = _dry_run("gciql/pixel.sh", ENCODER="drq")
    assert r.returncode == 0
    assert "--config-name cube_single_visual_drq" in r.stdout


# ── gcivl/pixel.sh ─────────────────────────────────────────────────────────
def test_gcivl_pixel_default_impala():
    r = _dry_run("gcivl/pixel.sh")
    assert r.returncode == 0, r.stderr
    assert "--config-name cube_single_visual_gcivl" in r.stdout
    assert "cube_single_visual_gcivl_drq" not in r.stdout


def test_gcivl_pixel_drq_encoder():
    r = _dry_run("gcivl/pixel.sh", ENCODER="drq")
    assert r.returncode == 0
    assert "--config-name cube_single_visual_gcivl_drq" in r.stdout


# ── crl/state.sh ───────────────────────────────────────────────────────────
def test_crl_state_dry_run():
    r = _dry_run("crl/state.sh")
    assert r.returncode == 0, r.stderr
    assert "--config-name cube_single_state_crl" in r.stdout
    assert "cube_single_state_crl_flowbc" not in r.stdout


# ── crl_flowbc/state.sh ────────────────────────────────────────────────────
def test_crl_flowbc_state_dry_run():
    r = _dry_run("crl_flowbc/state.sh")
    assert r.returncode == 0, r.stderr
    assert "--config-name cube_single_state_crl_flowbc" in r.stdout


# ── crl_flowbc/pixel.sh ────────────────────────────────────────────────────
def test_crl_flowbc_pixel_dry_run():
    r = _dry_run("crl_flowbc/pixel.sh")
    assert r.returncode == 0, r.stderr
    assert "--config-name cube_single_visual_crl_flowbc_drq" in r.stdout


# ── rldp/state.sh ──────────────────────────────────────────────────────────
def test_rldp_state_dry_run():
    r = _dry_run("rldp/state.sh")
    assert r.returncode == 0, r.stderr
    assert "train.py" in r.stdout
    assert "agent=rldp" in r.stdout
    assert "domain=cube_single" in r.stdout
    assert "[DRY]" in r.stdout


# ── rldp_flowbc/state.sh ───────────────────────────────────────────────────
def test_rldp_flowbc_state_dry_run():
    r = _dry_run("rldp_flowbc/state.sh")
    assert r.returncode == 0, r.stderr
    assert "agent=rldp_flowbc" in r.stdout

import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval import wandb_pull as wp


class FakeRun:
    def __init__(self, rid, obs_type):
        self.id = rid
        self.group = None
        self.tags = []
        self.created_at = datetime.now(timezone.utc)
        self.config = {"domain": "cube-single-play-v0", "obs_type": obs_type}


def test_obs_type_filter_keeps_only_matching():
    runs = [FakeRun("pix", "pixels"), FakeRun("st", "state")]
    kept = wp.filter_runs(runs, groups=None, tags=None, hours=96,
                          domains=None, obs_type="pixels")
    assert [r.id for r in kept] == ["pix"]


def test_obs_type_none_keeps_all():
    runs = [FakeRun("pix", "pixels"), FakeRun("st", "state")]
    kept = wp.filter_runs(runs, groups=None, tags=None, hours=96,
                          domains=None, obs_type=None)
    assert {r.id for r in kept} == {"pix", "st"}

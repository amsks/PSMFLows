import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals import _profile_core as pc


def test_profile_core_is_torch_free():
    import ast
    import importlib
    import evals._profile_core as m
    tree = ast.parse(Path(m.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(a.name.split(".")[0] != "torch"
                       for a in node.names)
        if isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] != "torch"
    importlib.reload(m)  # imports cleanly with numpy/pandas only


def test_spearman_monotone_and_degenerate():
    assert pc._spearman(np.array([1.0, 2, 3, 4, 5]),
                         np.array([1.0, 2, 3, 4, 5])) > 0.99
    assert np.isnan(pc._spearman(np.ones(5), np.arange(5.0)))


def test_verdict_helpers_relocated():
    assert pc._verdict_t1(0.40, 0.02)[0] == "SUPPORTS"
    assert pc._verdict_t4(3.0, 1.0)[0] == "SUPPORTS"
    syn = pc._synthesis({"T1": ("SUPPORTS", ""), "T4": ("NEUTRAL", "")})
    assert syn.startswith("Synthesis:")

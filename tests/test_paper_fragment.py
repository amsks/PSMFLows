import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals import paper_fragment as pf


def test_latex_escape():
    assert pf.latex_escape("a_b 50% #1 & $x") == \
        r"a\_b 50\% \#1 \& \$x"


def test_df_to_booktabs():
    df = pd.DataFrame({"task": ["t1"], "r2": [0.2]})
    tex = pf.df_to_booktabs(df, caption="B res", label="tab:b")
    assert "\\begin{tabular}" in tex
    assert "\\toprule" in tex and "\\bottomrule" in tex
    assert "\\caption{B res}" in tex and "\\label{tab:b}" in tex
    assert "t1" in tex


def test_build_fragment(tmp_path):
    phase = tmp_path / "phase" / "aggregate"
    repr_ = tmp_path / "repr" / "aggregate"
    phase.mkdir(parents=True)
    repr_.mkdir(parents=True)
    (phase / "overall.md").write_text("# phase\n- S0 success 47%")
    for fn in ("funnel.png", "fail_phase_composition.png"):
        (phase / fn).write_bytes(b"\x89PNG")
    (repr_ / "story.md").write_text("# story\n## Readout\nflat V")
    pd.DataFrame({"task": ["t1"], "r2": [0.2]}).to_parquet(
        repr_ / "T3_b_resolution.parquet")
    for fn in ("value_vs_dist.png", "sparsity_bars.png",
               "b_resolution_bars.png", "coverage_curves.png"):
        (repr_ / fn).write_bytes(b"\x89PNG")

    pf.build_fragment(phase, repr_, tmp_path / "PAPER", "fb-cube-failure")
    base = tmp_path / "PAPER" / "fb-cube-failure"
    tex = (base / "fb_cube_failure.tex").read_text()
    # self-contained, compilable LaTeX document
    assert "\\documentclass" in tex
    assert "\\begin{document}" in tex and "\\end{document}" in tex
    assert "\\includegraphics" in tex
    # tables embedded inline, not pulled from external files
    assert "\\begin{tabular}" in tex
    assert "\\input{" not in tex
    assert (base / "figures" / "value_vs_dist.png").exists()
    # user's hand-restructured layout, reproduced by the generator
    assert "\\section{Where and Why FB Fails on Cube Manipulation}" in tex
    assert tex.count("\\subsection") >= 4
    for sub in ("Experiment Design", "Methodology: Representation Probes",
                "Results", "Discussion"):
        assert sub in tex
    # GCIQL comparison folded in when a comparison dir is supplied
    gcmp = tmp_path / "gcmp"
    gcmp.mkdir()
    pd.DataFrame([{"method": "FB", "rho_success": 0.13,
                   "rho_fail": 0.23, "T1_verdict": "WEAK"},
                  {"method": "GCIQL", "rho_success": 0.4,
                   "rho_fail": 0.05, "T1_verdict": "SUPPORTS"}]
                 ).to_parquet(gcmp / "fb_vs_gciql.parquet")
    (gcmp / "comparison.md").write_text(
        "# cmp\n- FB ...\n- GCIQL ...")
    (gcmp / "cmp_value_rho.png").write_bytes(b"\x89PNG")
    for extra in ("cmp_value_vs_dist.png", "cmp_rho_box.png",
                  "cmp_funnel.png", "cmp_coverage.png"):
        (gcmp / extra).write_bytes(b"\x89PNG")
    pf.build_fragment(phase, repr_, tmp_path / "PAPER3", "x",
                      gciql_cmp=gcmp)
    t3 = (tmp_path / "PAPER3" / "x" / "fb_cube_failure.tex").read_text()
    assert "FB vs GCIQL" in t3
    assert "cmp_value_rho.png" in t3
    for extra in ("cmp_value_vs_dist.png", "cmp_rho_box.png",
                  "cmp_funnel.png", "cmp_coverage.png"):
        assert extra in t3
    # missing optional figure handled, not crashed
    pf.build_fragment(phase, repr_, tmp_path / "PAPER2", "x",
                      figures=("nonexistent.png",))
    assert "% [figure missing]" in \
        (tmp_path / "PAPER2" / "x" / "fb_cube_failure.tex").read_text()

"""evals/paper_fragment.py — render an \\input-able LaTeX section from
the phase-probe + representation-profile aggregates. No LaTeX is
compiled here; this only generates text + copies figures.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Sequence

_ESCAPE = {"&": r"\&", "%": r"\%", "#": r"\#", "_": r"\_",
           "$": r"\$", "{": r"\{", "}": r"\}"}

_DEFAULT_FIGS = ("value_vs_dist.png", "sparsity_bars.png",
                 "b_resolution_bars.png", "coverage_curves.png")
_PHASE_FIGS = ("funnel.png", "fail_phase_composition.png")


def latex_escape(s: str) -> str:
    return "".join(_ESCAPE.get(ch, ch) for ch in str(s))


def df_to_booktabs(df, caption: str, label: str) -> str:
    cols = list(df.columns)
    head = " & ".join(latex_escape(c) for c in cols) + r" \\"
    body = []
    for _, row in df.iterrows():
        cells = []
        for v in row:
            cells.append(f"{v:.3g}" if isinstance(v, float)
                         else latex_escape(v))
        body.append(" & ".join(cells) + r" \\")
    return "\n".join([
        r"\begin{table}[t]", r"\centering",
        r"\begin{tabular}{" + "l" * len(cols) + "}",
        r"\toprule", head, r"\midrule", *body, r"\bottomrule",
        r"\end{tabular}",
        r"\caption{" + caption + "}", r"\label{" + label + "}",
        r"\end{table}", ""])


def _fig(figdir_rel: str, name: str, present: bool, caption: str) -> str:
    if not present:
        return f"% [figure missing] {name}\n"
    return "\n".join([
        r"\begin{figure}[t]", r"\centering",
        r"\includegraphics[width=0.7\linewidth]{"
        + f"{figdir_rel}/{name}" + "}",
        r"\caption{" + caption + "}",
        r"\end{figure}", ""])


# story.md uses unicode that plain pdflatex cannot typeset; map to TeX.
_UNICODE = {
    "ρ": r"$\rho$", "→": r"$\to$", "←": r"$\leftarrow$",
    "±": r"$\pm$", "−": "-", "·": r"$\cdot$", "×": r"$\times$",
    "⇒": r"$\Rightarrow$", "≈": r"$\approx$", "≥": r"$\ge$",
    "≤": r"$\le$", "²": r"\textsuperscript{2}",
    "’": "'", "‘": "'", "“": "``", "”": "''",
}


def _tex(s: str) -> str:
    """Escape LaTeX specials, then transliterate unicode to TeX so the
    document compiles with plain pdflatex."""
    out = latex_escape(s)
    for u, r in _UNICODE.items():
        out = out.replace(u, r)
    return out


def _story_to_latex(md: str) -> str:
    """Render the data-driven story.md (headings, bullets, leading-and-
    trailing-underscore italics) as LaTeX prose — not an escaped dump."""
    lines, in_list = [], False

    def _close():
        nonlocal in_list
        if in_list:
            lines.append(r"\end{itemize}")
            in_list = False

    for raw in md.splitlines():
        line = raw.rstrip()
        if not line:
            _close()
            lines.append("")
            continue
        if line.startswith("# "):
            continue  # document title comes from \maketitle
        if line.startswith("## "):
            _close()
            lines.append(r"\subsection*{" + _tex(line[3:]) + "}")
            continue
        if line.startswith("- "):
            if not in_list:
                lines.append(r"\begin{itemize}")
                in_list = True
            lines.append(r"\item " + _tex(line[2:]))
            continue
        _close()
        if len(line) > 2 and line[0] == "_" and line[-1] == "_":
            lines.append(r"\emph{" + _tex(line[1:-1]) + "}")
        else:
            lines.append(_tex(line))
    _close()
    return "\n".join(lines)


def _abstract_from_story(md: str) -> str:
    """One-paragraph abstract built from the computed Synthesis line."""
    for raw in md.splitlines():
        if raw.strip().startswith("Synthesis:"):
            syn = raw.strip()[len("Synthesis:"):].strip()
            return (
                "We profile, across seven seeds, why the Forward--Backward "
                "(FB) agent fails at cube transport. Four representation "
                "probes are scored against the representation-failure "
                "hypothesis with heuristic verdicts. " + _tex(syn[:1].upper() + syn[1:]))
    return ("We profile, across seven seeds, why the Forward--Backward "
            "(FB) agent fails at cube transport, using four representation "
            "probes scored with heuristic verdicts.")


def build_fragment(phase_root, repr_root, out_dir, subdir: str,
                   figures: Sequence[str] = _DEFAULT_FIGS,
                   gciql_cmp=None) -> Path:
    phase_root, repr_root = Path(phase_root), Path(repr_root)
    base = Path(out_dir) / subdir
    figdir = base / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    for missing in (phase_root, repr_root):
        if not missing.exists():
            raise FileNotFoundError(
                f"Aggregate input not found: {missing}. Produce it with "
                f"the *_aggregate scripts before building the fragment.")

    # copy figures (phase + repr); track which exist
    present = {}
    for src_root, names in ((phase_root, _PHASE_FIGS),
                            (repr_root, figures)):
        for n in names:
            src = src_root / n
            if src.exists():
                shutil.copyfile(src, figdir / n)
                present[n] = True
            else:
                present[n] = False

    # tables rendered inline (booktabs) straight from the repr aggregate
    tbls = ""
    try:
        import pandas as pd
        for pq in sorted(repr_root.glob("T*.parquet")):
            df = pd.read_parquet(pq).reset_index()
            tname = pq.stem
            tbls += df_to_booktabs(
                df, caption=_tex(tname.replace("_", " ")),
                label=f"tab:{tname}") + "\n"
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] table generation skipped: {e}")

    story_md = ""
    sp = repr_root / "story.md"
    if sp.exists():
        story_md = sp.read_text()

    figs_phase = "".join(
        _fig("figures", n, present.get(n, False),
             "Phase-probe: " + _tex(n.replace("_", " ").replace(".png", "")))
        for n in _PHASE_FIGS)
    figs_repr = "".join(
        _fig("figures", n, present.get(n, False),
             _tex(n.replace("_", " ").replace(".png", "")))
        for n in figures)

    gcmp_block = ""
    if gciql_cmp is not None:
        gciql_cmp = Path(gciql_cmp)
        cmp_figs = []
        for png in sorted(gciql_cmp.glob("cmp_*.png")):
            shutil.copyfile(png, figdir / png.name)
            cap = _tex(png.stem.replace("cmp_", "").replace("_", " "))
            cmp_figs.append(_fig("figures", png.name, True,
                                 "FB vs GCIQL: " + cap))
        cmp_tbl = ""
        pq = gciql_cmp / "fb_vs_gciql.parquet"
        if pq.exists():
            import pandas as pd
            cmp_tbl = df_to_booktabs(
                pd.read_parquet(pq), caption=_tex("FB vs GCIQL"),
                label="tab:fb-vs-gciql")
        cmp_md = ""
        cmp_md_p = gciql_cmp / "comparison.md"
        if cmp_md_p.exists():
            cmp_md = _story_to_latex(cmp_md_p.read_text())
        gcmp_block = "\n".join([
            r"\subsection{FB vs GCIQL: Shared-Axis Comparison}",
            *cmp_figs, cmp_tbl, cmp_md])

    doc = [
        "% Auto-generated self-contained document; regenerate via "
        "scripts/figures/build_paper_fragment.py",
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{graphicx}",
        r"\usepackage{booktabs}",
        r"\usepackage[hidelinks]{hyperref}",
        r"\title{Where and Why FB Fails on Cube Manipulation}",
        r"\author{FB representation-failure profile (auto-generated)}",
        r"\date{\today}",
        r"\begin{document}",
        r"\maketitle",
        r"\begin{abstract}",
        _abstract_from_story(story_md),
        r"\end{abstract}",
        "",
        r"\section{Where and Why FB Fails on Cube Manipulation}",
        r"\label{sec:fb-cube-failure}",
        "",
        r"\subsection{Experiment Design \& Phase Probing}",
        r"We decompose each cube episode into reach $\to$ grasp/lift "
        r"$\to$ transport, and probe the policy from three start "
        r"states: S0 (baseline), S1 (skip-reach), S2 (pre-grasped, "
        r"physically clamped, no weld). The phase-probe over seeds "
        r"s3--s8, s10 localises where the rollout breaks.",
        figs_phase,
        "",
        r"\subsection{Methodology: Representation Probes}",
        r"Four representation probes on the FB checkpoints: (i) value "
        r"landscape $V(s)=Q_z(s,\pi(s))$, scored as the Spearman "
        r"$\rho$ between $V$ and $-d$ (cube--goal distance) over the "
        r"transport region; (ii) z-decoding and relabel sparsity; "
        r"(iii) ridge $R^2$ of $B(s)\!\to\!d$ and placed-vs-near "
        r"separability; (iv) off-support nearest-neighbour distance. "
        r"Probes are aggregated (mean$\pm$std) over the seven seeds "
        r"and each is assigned a heuristic verdict.",
        "",
        r"\subsection{Results \& Empirical Synthesis}",
        figs_repr,
        tbls,
        _story_to_latex(story_md),
        gcmp_block,
        "",
        r"\subsection{Discussion \& Core Narrative}",
        r"The evidence localises the failure to the \emph{"
        r"representation}, not the value landscape or data coverage. "
        r"The play data almost never shows the cube at the goal and "
        r"the inferred task vector $z$'s highest-scoring states are "
        r"mostly off-goal (T2); the backward map $B$ barely linearly "
        r"resolves placement distance (T3). With a scale-free metric "
        r"the value function provides no gradient that separates "
        r"successes from transport-failures (T1 weak), and failures "
        r"are not measurably further off data support than successes "
        r"(T4 neutral). FB lifts the cube but lacks a learned, "
        r"sharply-encoded target to deliver it.",
        "",
        r"\end{document}",
        "",
    ]
    (base / "fb_cube_failure.tex").write_text("\n".join(doc))
    return base

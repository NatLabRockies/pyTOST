"""Tests guarding the JOSS paper figure and its reference in paper.md.

The editorial bot compiles paper.md directly from the repository, so a missing or
unreferenced figure file breaks the submission silently.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURE = REPO_ROOT / "paper_figure.png"


def test_paper_figure_is_committed_to_the_repository():
    assert FIGURE.exists(), "paper_figure.png is missing from the repository root"
    assert FIGURE.stat().st_size > 0


def test_paper_figure_is_a_png():
    assert FIGURE.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_paper_references_the_figure_with_a_caption_and_label():
    paper = (REPO_ROOT / "paper.md").read_text(encoding="utf-8")

    assert "](paper_figure.png)" in paper, "paper.md does not embed paper_figure.png"
    assert "\\label{fig:engines}" in paper, "figure is missing a citable label"
    assert "\\autoref{fig:engines}" in paper, "figure label is never referenced in the text"


def test_figure_generation_script_exists_for_reproducibility():
    script = REPO_ROOT / "scripts" / "make_paper_figure.py"

    assert script.exists(), "the figure must be reproducible from a committed script"
    assert "paper_figure.png" in script.read_text(encoding="utf-8")

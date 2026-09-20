"""Structural tests for the worked-example documentation.

These guard against the vignette, its figure, and its generating script drifting apart.
The script itself is not executed here because the spatiotemporal bootstrap is slow;
it is the documented reproduction path instead.
"""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DOC = REPO_ROOT / "docs" / "sav_worked_example.md"
FIGURE = REPO_ROOT / "docs" / "sav_worked_example.png"
SCRIPT = REPO_ROOT / "scripts" / "sav_worked_example.py"


def test_worked_example_doc_exists():
    assert DOC.exists()


def test_worked_example_figure_exists_and_is_a_png():
    assert FIGURE.exists()
    assert FIGURE.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_worked_example_script_exists():
    assert SCRIPT.exists()


def test_doc_embeds_the_figure_and_links_the_script():
    text = DOC.read_text(encoding="utf-8")

    assert "](sav_worked_example.png)" in text
    assert "scripts/sav_worked_example.py" in text


def test_doc_and_script_agree_on_the_equivalence_margin():
    """A margin change in the script must not leave stale prose behind."""
    script = SCRIPT.read_text(encoding="utf-8")
    doc = DOC.read_text(encoding="utf-8")

    margin_line = next(
        line for line in script.splitlines() if line.startswith("MARGIN =")
    )
    margin = float(margin_line.split("=")[1].split("#")[0].strip())

    assert f"Δ = {margin:g}" in doc


def test_doc_and_script_agree_on_the_seed():
    script = SCRIPT.read_text(encoding="utf-8")
    doc = DOC.read_text(encoding="utf-8")

    seed_line = next(line for line in script.splitlines() if line.startswith("SEED ="))
    seed = seed_line.split("=")[1].strip()

    assert seed in doc


@pytest.mark.parametrize(
    "heading",
    ["The question", "The data", "The dependence-aware analysis", "What to take away"],
)
def test_doc_covers_the_full_workflow(heading):
    assert heading in DOC.read_text(encoding="utf-8")


def test_readme_links_to_the_worked_example():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    assert "docs/sav_worked_example.md" in readme


class TestDevelopmentStatus:
    """The README must present pyTOST as a released package, not a forthcoming one.

    JOSS requires that the software already be usable and documented at submission
    time, so pre-release phrasing in the README is a submission blocker.
    """

    @staticmethod
    def _status_section() -> str:
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        _, _, after = readme.partition("## Development status")
        assert after, "README is missing a '## Development status' section"
        section, _, _ = after.partition("\n## ")
        return section

    @pytest.mark.parametrize(
        "phrase",
        [
            "being prepared",
            "not yet released",
            "pre-release",
            "coming soon",
            "work in progress",
        ],
    )
    def test_status_avoids_pre_release_phrasing(self, phrase):
        assert phrase not in self._status_section().lower()

    def test_status_states_the_package_is_stable(self):
        assert "stable" in self._status_section().lower()

    def test_status_points_readers_at_the_changelog(self):
        assert "CHANGELOG.md" in self._status_section()

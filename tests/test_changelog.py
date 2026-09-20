"""Tests keeping CHANGELOG.md consistent with the packaged version and git tags."""

import re
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CHANGELOG = REPO_ROOT / "CHANGELOG.md"


def _changelog_text() -> str:
    return CHANGELOG.read_text(encoding="utf-8")


def _released_versions() -> list[str]:
    """Version headings of the form '## [X.Y.Z] — date', excluding Unreleased."""
    return re.findall(r"^## \[(\d+\.\d+\.\d+)\]", _changelog_text(), flags=re.MULTILINE)


def test_changelog_exists():
    assert CHANGELOG.exists()


def test_changelog_has_an_unreleased_section():
    assert "## [Unreleased]" in _changelog_text()


def test_changelog_documents_the_current_packaged_version():
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    version = pyproject["project"]["version"]

    assert version in _released_versions(), (
        f"pyproject version {version} has no CHANGELOG entry; "
        "add one when bumping the version"
    )


def test_changelog_discloses_the_reconstructed_history():
    """JOSS reviewers see 'reconstruct: import' commits; the changelog must explain them."""
    text = _changelog_text()

    assert "reconstruct" in text.lower()
    assert "archive" in text.lower()


def test_changelog_records_the_current_unreleased_features():
    text = _changelog_text()

    for feature in ("summary()", "plot_ci()", "max_lag", "heteroskedastic"):
        assert feature in text, f"{feature} is not recorded in the changelog"


def test_readme_links_to_the_changelog():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    assert "CHANGELOG.md" in readme

"""Consistency checks between `paper.md` and the software it describes.

The JOSS paper is reviewed against the repository, so any claim it makes about
available engines, metadata, or references must stay true as the code changes.
"""

import re
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER = REPO_ROOT / "paper.md"
BIB = REPO_ROOT / "paper.bib"
WORKFLOW = REPO_ROOT / "pyTOST" / "workflow.py"


@pytest.fixture(scope="module")
def paper() -> str:
    return PAPER.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def front_matter(paper: str) -> str:
    _, _, rest = paper.partition("---\n")
    block, _, _ = rest.partition("\n---")
    return block


def _dispatched_engines() -> set[str]:
    """Engine names `run_tost` actually dispatches on."""
    source = WORKFLOW.read_text(encoding="utf-8")
    return set(re.findall(r'eng == "([a-z_]+)"', source))


class TestEngineCoverage:
    def test_dispatch_is_discoverable(self):
        assert len(_dispatched_engines()) >= 6

    @pytest.mark.parametrize("engine", sorted(_dispatched_engines()))
    def test_every_engine_is_described_in_the_paper(self, engine, paper):
        assert engine in paper.lower(), (
            f"engine {engine!r} is available in run_tost but never mentioned in paper.md"
        )


class TestFrontMatter:
    @pytest.mark.parametrize(
        "field", ["title", "authors", "affiliations", "date", "bibliography"]
    )
    def test_required_field_is_present(self, field, front_matter):
        assert f"{field}:" in front_matter

    def test_date_is_iso_formatted(self, front_matter):
        date = re.search(r"^date:\s*(\S+)", front_matter, re.MULTILINE)

        assert date, "paper.md front matter has no date"
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", date.group(1))

    def test_bibliography_file_exists(self, front_matter):
        name = re.search(r"^bibliography:\s*(\S+)", front_matter, re.MULTILINE)

        assert name and (REPO_ROOT / name.group(1)).exists()


class TestRequiredSections:
    @pytest.mark.parametrize(
        "heading",
        ["# Summary", "# Statement of need", "# State of the field", "# References"],
    )
    def test_section_is_present(self, heading, paper):
        assert heading in paper


class TestCitations:
    def test_every_citation_key_resolves_to_a_bib_entry(self, paper):
        defined = set(re.findall(r"^@\w+\{([^,]+),", BIB.read_text(encoding="utf-8"), re.MULTILINE))
        cited = {
            key.strip().lstrip("@")
            for group in re.findall(r"\[([^\]]*@[^\]]*)\]", paper)
            for key in group.split(";")
            if key.strip().startswith("@")
        }

        assert cited, "no citations were found in paper.md"
        assert cited <= defined, f"undefined citation keys: {sorted(cited - defined)}"

    def test_software_is_cited_as_software_not_as_a_manual(self):
        """JOSS asks that software be cited with an entry type that records a version."""
        bib = BIB.read_text(encoding="utf-8")

        assert "@manual{" not in bib
        assert "@software{TOSTERpkg," in bib


class TestMetadataAgreement:
    def test_paper_title_matches_the_zenodo_deposition(self, front_matter):
        import json

        zenodo = json.loads((REPO_ROOT / ".zenodo.json").read_text(encoding="utf-8"))
        title = re.search(r'^title:\s*"?([^"\n]+)"?', front_matter, re.MULTILINE)

        assert title and title.group(1).strip() == zenodo["title"]

    def test_author_affiliation_matches_the_zenodo_deposition(self, front_matter):
        import json

        zenodo = json.loads((REPO_ROOT / ".zenodo.json").read_text(encoding="utf-8"))

        assert zenodo["creators"][0]["affiliation"] in front_matter

    def test_package_name_matches_the_distribution(self, paper):
        metadata = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

        assert metadata["project"]["name"] in paper


class TestArchiveDOI:
    """JOSS requires a citable archive of the released software."""

    CONCEPT_DOI = "10.5281/zenodo.22858610"

    def test_readme_cites_the_concept_doi(self):
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

        assert self.CONCEPT_DOI in readme

    def test_zenodo_metadata_records_the_concept_doi(self):
        import json

        zenodo = json.loads((REPO_ROOT / ".zenodo.json").read_text(encoding="utf-8"))
        identifiers = {r["identifier"] for r in zenodo.get("related_identifiers", [])}

        assert self.CONCEPT_DOI in identifiers

    def test_readme_cites_the_released_version(self):
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        metadata = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

        assert metadata["project"]["version"] in readme

from pathlib import Path

import tomllib


def test_license_header_and_metadata_match_bsd_3_clause():
    repo_root = Path(__file__).resolve().parents[1]

    license_lines = (repo_root / "LICENSE").read_text(encoding="utf-8").splitlines()
    assert license_lines[0] == "Copyright (c) 2026 Alliance for Energy Innovation, LLC"

    pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject["project"]
    assert project["license"] == "BSD-3-Clause"


def test_paper_affiliation_names_the_license_copyright_holder():
    """JOSS reviewers flag a paper affiliation that does not match the LICENSE holder.

    The affiliation must name the legal entity in LICENSE so the relationship
    between NLR and its operating organization is explicit.
    """
    repo_root = Path(__file__).resolve().parents[1]

    license_first_line = (
        (repo_root / "LICENSE").read_text(encoding="utf-8").splitlines()[0]
    )
    holder = license_first_line.split("Copyright (c) 2026 ", 1)[1].strip()

    paper = (repo_root / "paper.md").read_text(encoding="utf-8")
    front_matter = paper.split("---")[1]

    affiliation_lines = [
        line for line in front_matter.splitlines() if line.strip().startswith("name:")
    ]
    assert affiliation_lines, "paper.md front matter declares no affiliation name"
    assert any(
        holder in line for line in affiliation_lines
    ), f"no paper.md affiliation names the LICENSE copyright holder {holder!r}"

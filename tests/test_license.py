from pathlib import Path

import tomllib


def test_license_header_and_metadata_match_bsd_3_clause():
    repo_root = Path(__file__).resolve().parents[1]

    license_lines = (repo_root / "LICENSE").read_text(encoding="utf-8").splitlines()
    assert license_lines[0] == "Copyright (c) 2026 Alliance for Energy Innovation, LLC"

    pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject["project"]
    assert project["license"] == {"text": "BSD-3-Clause"}
    assert "License :: OSI Approved :: BSD License" in project["classifiers"]

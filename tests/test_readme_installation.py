from pathlib import Path


def test_readme_marks_rpy2_as_optional():
    repo_root = Path(__file__).resolve().parents[1]
    readme = (repo_root / "README.md").read_text(encoding="utf-8")

    assert "pip install \"pyTOST[r]\"" in readme
    assert "The `rpy2` integration is optional" in readme

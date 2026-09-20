# pyTOST — JOSS Submission Readiness Action Plan

*Reviewed against [JOSS submission requirements](https://joss.readthedocs.io/en/latest/submitting.html) and
[pre-review screening criteria](https://joss.readthedocs.io/en/latest/submitting.html#pre-review-screening-criteria).
Generated 2026-06-24.*

---

## Summary

The software itself and the paper text are submission-quality. The blocking issues are
structural: the paper files are gitignored, the public commit history is too sparse and
concentrated to pass JOSS's automated iterative-development checks, and there is no CI
workflow. Fix the one-line issues immediately, then invest the next three months in
building a genuine public development record before submitting.

---

## 🔴 Blocking Fixes (will cause desk rejection without these)

### 1. Remove `paper.md` and `paper.bib` from `.gitignore`

**File:** `.gitignore`

The current `.gitignore` contains:

```
# JOSS paper (kept locally, not published in repo)
paper.md
paper.bib
```

JOSS requires: *"Your paper (`paper.md` and BibTeX files, plus any figures) must be hosted
in a Git-based repository together with your software."* The editorial bot compiles the
paper directly from the public GitHub repo and will fail silently if the files are absent.

**Fix:** Remove those two lines from `.gitignore`, then `git add paper.md paper.bib` and commit.

---

### 2. Add a CI workflow for running tests

**Missing file:** `.github/workflows/tests.yml`

The only GitHub Actions workflow is `zenodo-release.yml`. JOSS reviewers explicitly check
for tests + CI as an open-source practice signal (screening criterion 3). Without it,
the submission is materially weaker even if the test suite itself is good.

**Fix:** Add a workflow such as:

```yaml
name: Tests

on:
  push:
    branches: [main]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - run: pip install -e ".[test]"
      - run: pytest -q
```

---

### 3. Obtain at least one external or citable research use

**File:** `paper.md` — "Research impact statement"

The current statement describes only internal use at NLR. JOSS screening criterion 2
requires *demonstrated* research impact, not aspirational statements. Acceptable signals
include a preprint, a published paper, a DOI linking to the software (e.g., a Zenodo
deposit already cited somewhere), or documented adoption by a collaborator at another
institution.

**Fix (fastest path):** Post a preprint to arXiv or ESSOAr describing an SAV equivalence
analysis that uses pyTOST as its inference engine. Then cite that preprint in the paper's
Research Impact section. This simultaneously satisfies the impact criterion and gives the
community a concrete use-case reference.

---

### 4. Build a genuine public development history

**This is the hardest blocker.** JOSS runs automated checks on commit distribution.
The current public log shows meaningful work on exactly 3 calendar days:

| Date | Commits | Content |
|------|---------|---------|
| 2025-10-06 | 1 | Initial commit |
| 2026-03-19 to 20 | 5 | v0.1, repo reorg, README, infrastructure, notebooks |
| 2026-06-22 | 10 | License, packaging, CI, Zenodo, paper fixes (all in one day) |

JOSS explicitly warns: *"A repository where all significant work was added in a
concentrated window is a signal that the project was not developed iteratively."* The
presence of `.git/filter-repo/already_ran` confirms history was rewritten, and
the untracked `old/pyTOSTv0.1.zip` through `pyTOSTv0.18.zip` reveal that the real
development happened outside the public repo.

There are two paths forward:

**Path A — Recover history from the zip archives (preferred):**
See the dedicated section below.

**Path B — Commit ongoing work publicly for 3+ months:**
See the 3-month enhancement plan below. This is the honest path forward even if you
also pursue Path A.

---

## 🟡 Fixes Required During Review

### 5. Update README "Development status" section

The current text says *"pyTOST is being prepared for open-source release and JOSS
submission"* and lists outstanding priorities including packaging, tests, and notebook
cleanup — implying the software isn't ready. JOSS reviewers read the README.

**Fix:** Replace the section with a statement of current stable status, e.g.:

> pyTOST is an actively maintained research package at v0.x. Contributions, bug reports,
> and feature requests are welcome via the issue tracker.

---

### 6. Fix `rpy2` description in README

**File:** `README.md` — "Installation" section

`pyproject.toml` correctly lists `rpy2` as an optional dependency (`[project.optional-dependencies]` → `r`), but the README still lists it under "pyTOST currently declares the following runtime dependencies." This contradiction will confuse reviewers and users.

**Fix:** Update the README to say `rpy2` is optional and show the install command:
```bash
pip install "pyTOST[r]"
```

---

### 7. Resolve the copyright holder / affiliation mismatch

`LICENSE` says: `Copyright (c) 2026 Alliance for Energy Innovation, LLC`  
`paper.md` affiliation says: `National Laboratory of the Rockies (NLR)`

These need to be consistent. Reviewers will flag this. Either update the affiliation
in `paper.md` to match the legal entity in `LICENSE`, or add a brief parenthetical
explaining the relationship (e.g., "NLR, operated by Alliance for Energy Innovation, LLC").

---

### 8. Resolve the version number confusion

`pyproject.toml` declares `version = "0.15.0"`, but untracked files include
`pyTOSTv0.19.zip`, `pyTOSTv0.19_joss_item1_fixed.zip`, and `pyTOSTv0.19_joss_item3_fixed.zip`.
The `old/` directory contains zips through v0.18. The installed package and the zip
versioning scheme are on different tracks.

**Fix:** Decide on the canonical current version and use it everywhere — `pyproject.toml`,
the git tag, and the Zenodo record. Add `old/`, `*.zip`, and `dist/` to `.gitignore` to
prevent stale artifacts from appearing in the working tree.

---

### 9. Justify or move `nbformat` out of runtime dependencies

`nbformat>=5.9` is listed in `[project.dependencies]`. It is a Jupyter notebook
format library, and reviewers will ask why importing `pyTOST` in a plain Python script
requires notebook infrastructure. If it is only used inside `data_gen/` optimization
notebooks, it should be optional or those notebooks should not be part of the installed
package.

**Fix:** Either move `nbformat` to `[project.optional-dependencies]` or remove it
from installed package paths by adjusting `[tool.setuptools.packages.find]`.

---

## 🟢 Minor Polish

### 10. Add a figure to `paper.md`

JOSS papers are not required to have figures, but a single schematic significantly
strengthens the submission and gives reviewers something concrete to engage with.
Good candidates:

- A diagram showing the CI-within-margins equivalence decision (the `(-Δ, Δ)` interval
  graphic used in most TOST tutorials).
- A side-by-side comparison of CI widths across all five engines on the same synthetic
  dataset, showing how ignoring dependence narrows the interval.

Save any figure as `paper_figure.png` (or similar) in the repo root and reference it
in `paper.md` with standard Markdown image syntax.

---

### 11. Fix the `TOSTERpkg` BibTeX entry type

**File:** `paper.bib`

```bibtex
@manual{TOSTERpkg, ...}
```

`@manual` is a TeX legacy type for printed manuals. For R packages, the modern preferred
type is `@software` (BibLaTeX) or `@misc`. JOSS uses Pandoc/BibLaTeX, so `@software`
with a `url` field is cleaner. This is low-stakes but worth correcting.

---

### 12. Clean up untracked clutter

The working tree currently has several untracked items that should not appear in a clean
public repo:

- `old/` — development archive, not part of the package
- `pyTOSTv0.19.zip`, `pyTOSTv0.19_joss_item1_fixed.zip`, etc. — these reveal internal
  versioning practices and may confuse reviewers
- `.ipynb_checkpoints/` in the root — already gitignored but may reappear

Add these to `.gitignore`:
```
old/
*.zip
dist/
*.egg-info/
```

---

## 🗄️ Recovering Git History from the Zip Archives

The `old/` directory contains `pyTOSTv0.1.zip` through `pyTOSTv0.18.zip`, and there is
a `pyTOSTv0.19.zip` at the repo root. These represent the actual iterative development
history that was never committed publicly. Reconstructing commits from these archives
is the most honest and defensible way to establish the iterative history JOSS requires.

### How to reconstruct

1. **Extract each zip in order** and record the file modification timestamps inside each
   archive — these approximate the real development dates.

2. **For each version**, create a commit with `GIT_AUTHOR_DATE` and `GIT_COMMITTER_DATE`
   set to the corresponding timestamp:

   ```bash
   # Example for v0.3
   unzip -o pyTOSTv0.3.zip -d /tmp/pytost_v03
   rsync -a --delete /tmp/pytost_v03/pyTOST/ ./pyTOST/
   git add -A
   GIT_AUTHOR_DATE="2025-11-15T10:00:00" \
   GIT_COMMITTER_DATE="2025-11-15T10:00:00" \
   git commit -m "v0.3: add temporal engine and HAC variance"
   ```

3. **Write meaningful commit messages** for each version based on what actually changed
   between zips (diff the directories to see what was added/modified).

4. **Use `git filter-repo` or `git rebase`** if you need to insert the reconstructed
   history before the current tip.

### What JOSS is looking for

JOSS wants to see that the software was *refined through use and feedback over time*. A
reconstructed history from real artifacts (zip archives with authentic timestamps) is
legitimate — you are not fabricating work, you are surfacing development that genuinely
happened but was not publicly tracked. The key is that the commit contents and dates
should match the actual artifacts, and the commit messages should accurately describe
what changed.

Do not manufacture commits that add trivial whitespace changes on artificially spread
dates — that is the kind of "repo dump" JOSS explicitly rejects.

---

## 📅 3-Month Public Development Plan

This plan assumes you submit no earlier than **late September 2026**, giving JOSS at
least three months of observable public activity from now (June 2026). Commit each item
individually as it is completed — do not batch them. The goal is a steady, substantive
commit stream, not a burst.

Each section below is a self-contained improvement that is genuinely useful to the
package and produces a real, reviewable commit.

---

### Month 1 (July 2026) — Infrastructure and correctness

| Week | Task | Commit message (example) |
|------|------|--------------------------|
| 1 | Add CI test workflow (see fix #2 above) | `ci: add pytest workflow for Python 3.10 and 3.11` |
| 1 | Fix `.gitignore` to include `paper.md`/`paper.bib`; push both files | `paper: add JOSS paper.md and paper.bib to repo` |
| 1 | Fix README rpy2 description (fix #6) | `docs: clarify rpy2 is an optional dependency` |
| 2 | Fix copyright/affiliation mismatch (fix #7) | `docs: align LICENSE copyright holder with paper affiliation` |
| 2 | Resolve version to a single canonical number and tag it | `release: bump version to 0.16.0` |
| 2 | Move `nbformat` to optional or justify it (fix #9) | `packaging: move nbformat to optional notebook dependencies` |
| 3 | Add edge-case tests: single-cluster spatial input, unbalanced spatiotemporal panel with T=2, empty margins list | `tests: add edge case coverage for spatial and spatiotemporal engines` |
| 3 | Add `pytest-cov` to test deps and measure coverage; document it in README | `tests: add coverage reporting; current coverage ~N%` |
| 4 | Improve docstrings in `workflow.py` — add parameter types, return schema, a worked example | `docs: expand run_tost docstring with full parameter reference` |
| 4 | Improve docstrings in `engines/spatial_tost.py` — describe Matérn parameter estimation | `docs: document SpatialTOST covariance estimation procedure` |

---

### Month 2 (August 2026) — Feature enhancements

| Week | Task | Commit message (example) |
|------|------|--------------------------|
| 1 | Add a `summary()` method to the result dict that prints a human-readable equivalence decision table | `feat: add run_tost summary output formatter` |
| 1 | Add `engine="heteroskedastic"` as a first-class `run_tost` option (currently only surfaced via sensitivity) | `feat: expose heteroskedastic engine as a primary run_tost option` |
| 2 | Add `plot_ci()` utility: a matplotlib figure showing CI bars for each engine result, suitable for reports | `feat: add plot_ci visualization for comparing engine CIs` |
| 2 | Use the figure from `plot_ci()` to add a figure to `paper.md` (fix #10) | `paper: add CI comparison figure to JOSS paper` |
| 3 | Add `WorkflowOptions.max_lag` parameter to let users override the automatic Newey–West lag selection | `feat: add max_lag option to temporal engine HAC estimator` |
| 3 | Add input validation with informative error messages when `x`/`ycoord` columns contain NaN | `fix: raise informative error for NaN coordinates in spatial engine` |
| 4 | Write a vignette-style section in the README or a `docs/` folder showing the full SAV use-case workflow (anonymized/synthetic) | `docs: add SAV equivalence analysis worked example` |
| 4 | Add `CHANGELOG.md` and backfill entries for v0.1–current | `docs: add CHANGELOG with history from v0.1 to present` |

---

### Month 3 (September 2026) — Robustness, release, and submission prep

| Week | Task | Commit message (example) |
|------|------|--------------------------|
| 1 | Add parametric tests validating CI coverage against analytical solutions for the IID and cluster engines (Monte Carlo, fixed seed) | `tests: add CI coverage Monte Carlo validation for IID and cluster engines` |
| 1 | Add a `RobustLocationTOST` path for trimmed mean with configurable trim fraction | `feat: support configurable trim fraction in RobustLocationTOST` |
| 2 | Post preprint (arXiv or ESSOAr) describing a concrete use case; add citation to `paper.md` research impact section | `paper: cite preprint in research impact statement` |
| 2 | Tag `v0.17.0` release; trigger Zenodo workflow to mint a DOI; update `.zenodo.json` with DOI | `release: v0.17.0 — mint Zenodo DOI for JOSS submission` |
| 3 | Update `paper.md` date field and do final review of all sections against JOSS review checklist | `paper: final pre-submission review pass` |
| 3 | Update README "Development status" section (fix #5) | `docs: update README development status for v0.17 release` |
| 4 | Ask a colleague at a different institution to install pyTOST from PyPI on a clean environment and file a GitHub issue with their feedback | *(their issue + your response is the external engagement signal)* |
| 4 | Submit to JOSS | — |

---

## Checklist Before Pressing Submit

- [ ] `paper.md` and `paper.bib` are **not** in `.gitignore` and are present on the default branch
- [ ] CI workflow runs and passes on the current `main`
- [ ] At least one tagged release with a minted Zenodo DOI exists
- [ ] Zenodo DOI is referenced in `paper.md` or `.zenodo.json`
- [ ] Research impact section cites at least one preprint or external use
- [ ] README does not describe the package as "being prepared" for anything
- [ ] Copyright holder in `LICENSE` matches (or is explained alongside) the affiliation in `paper.md`
- [ ] `rpy2` is documented as optional in the README
- [ ] Version in `pyproject.toml` matches the git tag and Zenodo record
- [ ] Public commit history shows substantive commits spread across at least 3 months
- [ ] A colleague outside NLR has successfully installed and run the package

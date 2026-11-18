# pyTOST - JOSS submission readiness

Reconciled 2026-11-18 against the live repository and the current
[submission guidance](https://joss.readthedocs.io/en/latest/submitting.html),
[review criteria](https://joss.readthedocs.io/en/latest/review_criteria.html), and
[January 2026 scope announcement](https://blog.joss.theoj.org/2026/01/preparing-joss-for-a-generative-ai-future).
This replaces the obsolete three-month enhancement schedule. It does not certify
editorial eligibility or acceptance.

## Current assessment

The original engineering enhancements are implemented. The repository has paper
sources, automated checks, releases, a worked example, and support and governance
policies. Remaining submission gates are evidence of sufficient public history,
alignment of the submitted release with the manuscript, and final human review of
scientific claims, research-use evidence, and AI disclosure.

Current JOSS screening requires more than six months of public history with active
development across that period. Waiting, imported private history, or spacing issue
timestamps does not supply missing evidence of open development.

For solo projects, the current guidance explicitly says that absence of a PR
workflow alone does not disqualify an otherwise open and well-maintained project.
Multiple indicators still matter: meaningful public commits, releases or a
changelog, tests and CI, documentation, contribution guidance, and support or
governance expectations. External adopters and non-author contributions strengthen
the case but are not independent hard requirements.

## Completed engineering schedule

The original July-September schedule is reconciled below. Commit dates describe
repository history, not independently verified dates of public availability.

| Original task | Evidence / disposition |
|---|---|
| Track paper sources | `paper.md`, `paper.bib`, `paper_figure.png`; `5e810ad` |
| Add CI | `.github/workflows/tests.yml`; `42ff550`, `05ba87d` |
| Clarify optional R dependency | `README.md`, `[project.optional-dependencies]` in `pyproject.toml`; `ad5dea7` |
| Align affiliation and copyright | `paper.md`, `LICENSE`, `.zenodo.json`; `a7ffc63` |
| Establish canonical package version | `pyproject.toml` and latest tag use `0.17.0`; later work is explicitly unreleased |
| Move notebook-only dependency | `nbformat` is in the `notebooks` extra; `dfcbd39` |
| Add engine edge-case tests | `tests/test_edge_cases.py`; `081b149` |
| Add test coverage tooling | `pytest-cov` in the test extra; `pixi run -e test test-cov` |
| Document workflow and spatial estimation | `pyTOST/workflow.py`, `pyTOST/engines/spatial_tost.py`; `e948f18`, `87e2bb8` |
| Add result summary | `TOSTResult.summary()`, `tests/test_summary.py`; `4ecbc04` |
| Expose heteroskedastic primary engine | `run_tost`, `tests/test_heteroskedastic_engine_option.py`; `0f350b2` |
| Add CI plotting | `plot_ci`, `tests/test_plot_ci.py`; `9a8c9e7` |
| Add manuscript figure | `scripts/make_paper_figure.py`, `paper_figure.png`; `9ca8717`, `4bc2312` |
| Expose HAC lag control | `WorkflowOptions.max_lag`, `tests/test_max_lag.py`; `fc4cac5` |
| Validate spatial coordinates | `pyTOST/validation.py`, `tests/test_coordinate_validation.py`; `4258e1e` |
| Add SAV worked example | `docs/sav_worked_example.md`, `scripts/sav_worked_example.py`; `12ba6ee` |
| Add changelog | `CHANGELOG.md`; `36bd4ef` |
| Validate IID and cluster coverage | `tests/test_ci_coverage.py`; `7e9a266` |
| Add configurable trimmed mean | `tests/test_robust_trim.py`; `edb3898` |
| Publish versioned release and archive identifier | [v0.17.0](https://github.com/NatLabRockies/pyTOST/releases/tag/v0.17.0), `.zenodo.json`, README DOI |
| Update development status | README; `f9adcfb` |
| Review manuscript | `paper.md`, `tests/test_paper.py`, `docs/review_register.md`; `20e3d54`, `e89f2b5`; final author approval remains required |
| Add broader engine validation | Temporal, HC3, wild-cluster, and equal-weighted spatiotemporal tests; restricted spatial benchmark in `docs/coverage_benchmark.md` |
| Add conduct, support, and governance policies | `CODE_OF_CONDUCT.md`, [SUPPORT.md](SUPPORT.md), [GOVERNANCE.md](GOVERNANCE.md), [CONTRIBUTING.md](CONTRIBUTING.md) |
| Reconcile temporal user guidance | README now explains GLS, explicit HAC, aggregation, and strict rejection; [#2](https://github.com/NatLabRockies/pyTOST/issues/2) |
| External preprint or adopter issue | Optional strengthening, not an engineering blocker or a fabricated-engagement task |
| Submit to JOSS | Not completed; subject to the gates below |

Coverage results apply to the configurations actually exercised. In particular,
the spatial benchmark fixes `nu_grid=(0.5,)` and both spatial benchmarks use
`point_estimator="equal_weighted"`; they do not establish coverage for every
default GLS/bootstrap configuration. See the artifact's recorded `method` values.

## Public-development eligibility investigation

Investigation for [#3](https://github.com/NatLabRockies/pyTOST/issues/3):
**the public-development start date remains unknown, and submission eligibility is
not established.** Closing the investigation does not certify eligibility.

Verified through the GitHub API on 2026-11-18:

| Record | Timestamp (UTC) | What it establishes |
|---|---|---|
| Repository creation | 2025-10-06 22:14:40 | Creation, not public visibility at that time |
| Current visibility | Public | Current access only |
| [v0.1 release](https://github.com/NatLabRockies/pyTOST/releases/tag/v0.1) | 2026-04-03 02:12:55 | Recorded release publication, not proof it was public then |
| [v0.15.0 release](https://github.com/NatLabRockies/pyTOST/releases/tag/v0.15.0) | 2026-06-22 16:33:33 | Recorded release publication |
| [v0.17.0 release](https://github.com/NatLabRockies/pyTOST/releases/tag/v0.17.0) | 2026-09-20 13:23:22 | Recorded release publication |

`CHANGELOG.md` describes private development followed by incremental public
development at v0.15.0. That narrative is not independent visibility evidence.
Imported private archives and their commit dates cannot establish public history.
No visibility-change event was found in the repository events response; absence
from that limited response does not establish a date either.

The current submission guidance requires **more than six months** of public
history with active development across that period. The January announcement
describes at least six months for privately developed projects. Use the stricter
current screening requirement when planning.

If public development started on 2026-06-22, six calendar months end on 2026-12-22;
2026-12-23 is a conservative planning date beyond that anniversary. This is
conditional, not a verified eligibility date or a promise of editorial acceptance.
The maintainer must supply contemporaneous public evidence or available
visibility-change records before a firm submission date can be set.

Issues [#1](https://github.com/NatLabRockies/pyTOST/issues/1),
[#2](https://github.com/NatLabRockies/pyTOST/issues/2), and
[#3](https://github.com/NatLabRockies/pyTOST/issues/3) were created on 2026-11-18 to
track real follow-up work. Their creation or closure does not demonstrate historical
community engagement.

## Remaining submission work

| Priority | Task | Completion evidence required |
|---|---|---|
| Gate | Verify public-development eligibility | Maintainer-provided visibility or contemporaneous public evidence; sustained activity across the required interval |
| Gate | Reconcile manuscript claims with actual default inference paths | Coverage evidence and wording must identify the estimator/configuration tested; unsupported generalizations must not be presented as validation |
| Gate | Prepare a release matching the manuscript | Version, tag, built distributions, release notes, and archive identify the same code; current post-v0.17.0 changes remain unreleased |
| Gate | Confirm realized research use | Author verifies the NLR use described in the paper and can demonstrate it to editors; synthetic examples support reproducibility but are not proof of realized adoption |
| Gate | Complete accurate AI disclosure | Author checks tools/models and versions actually used, code/paper/docs assistance, scope, and human review; intended model choice is not proof of actual model use |
| Final check | Author reviews manuscript and submission metadata | Date, affiliations, references, figure, research-use claims, acknowledgements, and any required conflict/funding disclosures verified |
| Optional | Obtain independent installation feedback or publish a use-case preprint | Genuine feedback or publication, if available; no requirement to manufacture a non-author issue |

AI-assisted preparation does not establish that the human author reviewed every
output. The author must make that confirmation personally. JOSS also prohibits AI
for author-editor/reviewer conversations except translation; the author must handle
those exchanges.

Implementation follow-ups and issue links are tracked in
[`docs/scope_backlog.md`](docs/scope_backlog.md); verified findings are recorded in
[`docs/review_register.md`](docs/review_register.md). Do not implement new inference
methods solely to fill time before submission.

## Before pressing submit

- [x] Paper, bibliography, and figure are tracked with the source.
- [x] An OSI-approved license and contribution guidance are present.
- [x] Automated tests and CI exist; 290 tests passed locally on 2026-11-18.
- [x] Tagged releases and archive metadata exist.
- [x] Support and governance expectations are documented.
- [x] A reproducible worked example and benchmark artifacts exist.
- [ ] Public-development start and more-than-six-month history are verified.
- [ ] Default inference claims and coverage evidence agree.
- [ ] The submitted release, manuscript, and archived version agree.
- [ ] The author confirms research-use evidence and the complete AI disclosure.
- [ ] Final author review is complete and CI passes at the submission commit.

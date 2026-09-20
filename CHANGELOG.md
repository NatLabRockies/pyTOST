# Changelog

All notable changes to pyTOST are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Note on early history

pyTOST was developed privately from v0.1 through v0.19 as a series of versioned
archives before the project moved to public development. Those archives were later
imported into the Git history as `reconstruct: import pyTOSTvX.Y.zip` commits, so the
entries below for that period describe what the archives actually contain rather than
day-to-day development steps. Public, incremental development begins at v0.15.0.

## [Unreleased]

### Added

- `TOSTResult.summary()` renders a human-readable equivalence decision table covering
  the primary engine, sensitivity analyses, and the bootstrap sanity check. `run_tost`
  now returns a `TOSTResult`, which is a `dict` subclass, so existing dictionary access
  is unchanged.
- `engine="heteroskedastic"` is now a first-class `run_tost` option. It uses HC3 robust
  inference, or cluster-robust wild bootstrap inference when `cluster` is supplied.
  Previously this engine was reachable only through the sensitivity output.
- `plot_ci()` draws a report-ready matplotlib figure comparing confidence intervals
  across engines, with the equivalence margins marked and bars colored by decision.
  It accepts either a `run_tost` result or a `{label: DataFrame}` mapping.
- `WorkflowOptions.max_lag` overrides the Newey–West truncation lag used by the
  temporal engine. It accepts an integer, or `"auto"` to select the lag from the sample
  size via the Newey–West (1994) plug-in rule, exposed as `auto_hac_lags()`.
- Worked example documenting a full SAV method-equivalence analysis, in
  `docs/sav_worked_example.md`, reproducible via `scripts/sav_worked_example.py`.
- Monte Carlo and analytical validation of confidence-interval coverage for the IID and
  cluster engines. The IID interval is verified to match the closed-form Student-t
  interval exactly; simulated coverage is 0.892 (IID engine on IID data) and 0.907
  (cluster engine on clustered data) against a nominal 0.900. The same simulation
  records the motivating failure quantitatively: applying the IID engine to clustered
  data yields only 0.613 coverage.
- Figure in `paper.md` comparing engine confidence intervals on a shared synthetic
  dataset, reproducible via `scripts/make_paper_figure.py`.
- This changelog.

### Changed

- The paper affiliation now names the operating legal entity, matching the copyright
  holder in `LICENSE`.

### Fixed

- The spatial, spatiotemporal, and building-aware engines now raise an informative
  `ValueError` naming the offending column and rows when coordinate columns contain
  NaN or infinite values. Previously such input surfaced as an opaque
  `KeyError: 'theta_log'` from inside the covariance optimizer.

### Notes

- The default temporal truncation lag remains `4`; `max_lag` is opt-in so that results
  from earlier versions stay reproducible.

## [0.15.0] — 2026-06-22

First public release of the packaged project.

### Added

- BSD-3-Clause license, with the Alliance for Energy Innovation, LLC as copyright
  holder.
- Zenodo metadata (`.zenodo.json`) and an automated release workflow for minting DOIs,
  including manual dispatch and backfill for existing release tags.
- Cleaned demonstration notebooks, relocated into `notebooks/`.
- Deterministic synthetic regression baselines in the test suite.
- GitHub Actions workflow running the test suite on Python 3.10 and 3.11, later moved
  to run through Pixi.
- `pixi` manifest so repository environments are managed reproducibly.
- JOSS paper sources (`paper.md`, `paper.bib`) tracked in the repository.

### Changed

- Packaging metadata updated to a PEP 639 SPDX license expression.
- `rpy2` documented and declared as an optional dependency.
- `nbformat` moved from runtime to optional notebook dependencies.

## Archive era (v0.1 – v0.19, reconstructed)

### [0.19] — 2026-03-19

- Reworked the parameter-optimization utilities and parameter I/O for the synthetic
  data generators; added data-generation tests.
- Follow-up archives addressed JOSS readiness items, including further work on the
  spatial engine.

### [0.18] — 2026-03-19

- Added the automated `pytest` suite covering the IID, cluster, temporal, spatial, and
  spatiotemporal engines, the workflow, and the public API surface.

### [0.17] — 2026-03-19

- Large consolidation across every engine and the workflow, and introduction of the
  structured synthetic data generation package (`data_gen`) with per-engine parameter
  optimization utilities.

### [0.13] – [0.15] — 2026-03-05 to 2026-03-17

- Substantial revisions to the bootstrap machinery and supporting notebook tooling.

### [0.2] – [0.12] — 2026-03-03 to 2026-03-04

- Iterative development of the spatiotemporal engine and the bootstrap module,
  including repeated simplification of the spatiotemporal covariance fitting path.

### [0.1] — 2026-03-03

- Initial implementation: IID, cluster, temporal, spatial, and spatiotemporal TOST
  engines; heteroskedastic and robust-location sensitivity engines; bootstrap and
  diagnostics modules; the `run_tost` workflow; and visualization helpers.

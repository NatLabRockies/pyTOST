# Review register

Verified findings from code review, quality audits, and CI failures, with their
disposition. Entries are append-only; update the disposition in place as work lands.

---

## RR-001 — Temporal engine results depend on row order when time values are tied

- **Severity:** High
- **Category:** Correctness / reproducibility
- **Status:** Resolved (2026-09-20)
- **Found:** 2026-09-20, first CI run on `main` (run 35511501676)
- **Affected files:** `pyTOST/engines/temporal_tost.py`,
  `tests/test_deterministic_baselines.py`

### Evidence

`TemporalTOST` orders observations by the `time` column and then forms Newey--West HAC
autocovariances from the resulting residual sequence. When several rows share a time
value, their relative order is whatever the caller's data frame happened to contain, and
the HAC lag windows pair up different residuals depending on that order. The estimate
`mu_hat` is unaffected, but `ci_low` / `ci_high` are not.

Reproduced on the deterministic baseline fixture (120 rows, 40 distinct times, 3 series
per arm), permuting only the input row order:

| input permutation | `ci_low`   |
| ----------------- | ---------- |
| as generated      | 0.10239981 |
| shuffle seed 0    | 0.10147115 |
| shuffle seed 1    | 0.09919320 |
| shuffle seed 2    | 0.09967323 |
| shuffle seed 3    | 0.10096055 |

This surfaced as a cross-platform CI failure:
`tests/test_deterministic_baselines.py::test_deterministic_engine_regression_baseline`
passes on macOS/arm64 and fails on Linux/x86-64 with `ci_low = 0.10120545` against a
pinned `0.10239981`. The platforms resolve different `pandas` versions, which produce a
different row order out of `long_to_diff`, which changes the HAC result.

### Assessment

Two distinct problems sit behind one symptom:

1. **Reproducibility.** Identical data in a different row order yields a different
   confidence interval. Any published result from this engine on tied-time data is not
   reproducible.
2. **Statistical validity.** A Newey--West HAC estimator assumes a single sequence
   indexed by time. Feeding it a panel with several series observed at the same times
   treats cross-sectional replicates as if they were consecutive time points, so the
   estimated autocovariance structure does not correspond to the data-generating
   process. `TemporalTOST` already exposes `require_unique_times` to guard this, but it
   defaults to `False` and `run_tost` never sets it.

Problem 1 is a bug. Problem 2 is a design question about what the temporal engine should
do with panel data: reject it, aggregate over series within each time point, or route it
to a panel-aware estimator. Resolving it changes public behaviour and is therefore a
scope decision, not a bug fix.

### Resolution

Problem 1 (reproducibility) is fixed. `TemporalTOST` now orders observations by
`(time, response)` rather than by time alone. Rows tied on both are interchangeable and
leave the residual sequence unchanged, so the estimate is invariant to the caller's row
order. Covered by `tests/test_temporal_order_invariance.py`.

Problem 2 (validity) is surfaced rather than silently resolved: the engine now emits a
`UserWarning` when time values are tied, naming the assumption that is violated and
pointing at the cluster and spatiotemporal engines. `require_unique_times=True` upgrades
the warning to an error. Deciding what the engine *should* do with panel data --- reject
it, aggregate within time points, or route it to a panel-aware estimator --- remains
open and is tracked in `docs/scope_backlog.md`.

The deterministic baseline fixture was changed from a 3-series panel to a single series
of 120 time points, and asserts that its time values are unique.

### Merge/release impact

Cleared for the v0.17.0 tag once CI is green on `main`.

## RR-002 — JOSS paper overstated validation scope and engine coverage

- **Severity:** medium
- **Category:** accuracy of claims / publication readiness
- **Status:** resolved in `paper.md`
- **Affected files:** `paper.md`, `scripts/make_paper_figure.py`, `tests/test_paper.py`

### Findings

1. **Figure caption claimed completeness it did not have.** The caption described
   "all five `pyTOST` engines" while the package exposes six. The heteroskedastic
   engine is not plotted, because the figure's dataset has constant variance. The
   same stale claim appeared twice in `scripts/make_paper_figure.py` docstrings.

2. **Coverage results were reported without denominators.** The paper quoted 0.892,
   0.907, and 0.613 against a nominal 0.900 with no replicate counts, no simulation
   design, and no Monte Carlo error, so a reader could not judge whether the
   deviations were meaningful. They are 600 and 400 replicates, giving Monte Carlo
   standard errors of 0.012 and 0.015.

3. **Validation scope was not disclosed.** Coverage is simulated for the IID and
   cluster engines only. The paper did not say so, leaving the impression that all
   engines carried the same empirical backing. Tracked in `docs/scope_backlog.md`
   as SB-002.

4. **Evidence was in the wrong section.** The coverage results sat in
   "State of the field", a section about other people's software, burying the
   paper's strongest empirical support for its own central claim.

5. **Engine configuration was described inaccurately.** The text said all five
   plotted engines are "run in equal-weighted mode"; only the spatial and
   spatiotemporal engines accept that option.

6. **Date format violated the JOSS specification,** which requires `%e %B %Y`.
   `tests/test_paper.py::test_date_is_iso_formatted` actively enforced the wrong
   format and had to be rewritten.

7. **API material appeared in the paper.** JOSS directs installation and API usage
   to the software documentation; the `pip install` block and `run_tost(...)`
   example were removed.

### Resolution

All seven are fixed. Coverage evidence moved to a new "Validating the intervals"
subsection under Software design, with replicate counts, simulation design, Monte
Carlo error, and an explicit statement that the temporal, spatial, spatiotemporal,
and heteroskedastic engines have correctness tests but no coverage simulation. The
figure's numbers were re-verified by running `fit_all_engines` directly: n = 192,
width ratios 2.52 and 6.23 versus IID, all five engines returning mu_hat = 0.26, and
equivalence declared by the IID, temporal, and spatial engines only. The paper now
also discloses that the temporal engine is applied to a panel with tied time points
(see RR-001).

`tests/test_paper.py` was corrected to assert the JOSS date format and extended to
guard all eight required JOSS sections.

### Merge/release impact

Paper-only; no library behaviour changed. Does not block a release.

## RR-003 — Temporal engine materially undercovers under serial correlation

- **Severity:** high
- **Category:** statistical validity
- **Status:** Resolved (2026-11-04) — AR GLS is now the default; see Resolution
- **Affected files:** `pyTOST/engines/temporal_tost.py`, `paper.md`

### Evidence

Monte Carlo coverage of the temporal (Newey--West HAC) engine against a nominal
0.900, AR(1) paired differences with known mean, 400--500 replicates:

| rho | n | Temporal (HAC) | IID |
|-----|-----|------|------|
| 0.0 | 120 | 0.868 | 0.886 |
| 0.0 | 400 | 0.880 | 0.894 |
| 0.5 | 120 | 0.798 | 0.618 |
| 0.5 | 400 | 0.838 | 0.650 |
| 0.7 | 120 | **0.720** | 0.468 |
| 0.7 | 400 | **0.786** | 0.486 |

The engine clearly improves on the IID engine, which is the package's core claim, but
it does not attain its stated level. At rho = 0.7, n = 120, a nominally 90% interval
covers 72% of the time.

This is not a lag-selection problem. Sweeping the HAC truncation lag at rho = 0.7
gives at most 0.757 (n = 120) and 0.825 (n = 400):

| lag | auto(4) | 8 | 12 | 16 | 24 | 32 |
|-----|------|------|------|------|------|------|
| n=120 | 0.715 | 0.752 | 0.757 | 0.748 | 0.735 | 0.720 |
| n=400 | 0.772 | 0.805 | 0.818 | 0.823 | 0.825 | 0.825 |

The cause is the combination of downward-biased HAC variance estimation in finite
samples with the standard-normal critical value used in `_hac`.

### Option already present in the codebase

`TemporalTOST._glsar1` implements an AR(1) GLS path with a $t$ critical value. It is
substantially better calibrated:

| rho | n | GLSAR(1) | HAC |
|-----|-----|------|------|
| 0.0 | 400 | 0.890 | 0.880 |
| 0.5 | 400 | 0.882 | 0.838 |
| 0.7 | 120 | 0.838 | 0.720 |
| 0.7 | 400 | 0.877 | 0.786 |
| 0.9 | 400 | 0.868 | n/a |

### Scope decision required

Changing which estimator backs the temporal engine alters public behaviour and is a
scope decision, not a bug fix. Options: make GLSAR(1) the default; expose a `method`
argument defaulting to HAC; adopt fixed-b/KVB critical values; or keep HAC and
document the limitation. Not actioned pending approval.

## RR-004 — Heteroskedastic engine provides no benefit for the intercept-only mean

- **Severity:** medium
- **Category:** accuracy of claims
- **Status:** Resolved (2026-11-04) — `paper.md` now states the equivalence explicitly
  and credits the wild cluster bootstrap rather than HC3
- **Affected files:** `paper.md`, `pyTOST/engines/heteroskedastic_tost.py`

### Evidence

With independent, strongly heteroskedastic paired differences (sigma ramped 0.3 to
3.0 across n = 120), 600 replicates, nominal 0.900:

- Heteroskedastic engine (HC3): 0.930
- IID engine: 0.930

The two are indistinguishable, and this is expected analytically. For an
intercept-only model, Var(ybar) = (1/n^2) * sum(sigma_i^2), and the ordinary OLS
variance estimate s^2/n converges to the same quantity, because E[s^2] approaches the
average of sigma_i^2. Heteroskedasticity-robust covariance matters for *slopes*, not
for an unweighted mean. HC3 is therefore a different finite-sample estimator of the
same target, not a correction.

The engine's real contribution is the cluster-robust wild bootstrap path, which
applies when a cluster variable is supplied. The paper should not imply that HC3
repairs anything for the mean under pure heteroskedasticity.

## RR-005 — AR(1) GLS path is unreachable dead code, and the paper claims otherwise

- **Severity:** medium
- **Category:** accuracy of claims / dead code
- **Status:** Resolved (2026-11-04) — the GLS path is now reachable and is the default
- **Affected files:** `pyTOST/engines/temporal_tost.py`, `paper.md`

`TemporalTOST.fit` calls only `_hac`. `_glsar1` has no caller anywhere in the package,
tests, or workflow. Nevertheless:

- `paper.md` stated "An AR(1) GLS path is available by calling `TemporalTOST`
  directly," which is false; calling the class directly still routes to `_hac`.
- `pyTOST/engines/temporal_tost.py:9` states "the workflow can run both," also false.

The paper sentence has been removed. The code disposition (expose it or delete it)
depends on the RR-003 decision.

## RR-006 — AI usage disclosure does not meet the JOSS policy

- **Severity:** high
- **Category:** publication policy / research integrity
- **Status:** Resolved (2026-11-04) — author supplied tool, version, scope, and
  verification; `paper.md` now names GitHub Copilot CLI v1.0.103 with Claude Opus 5
- **Affected files:** `paper.md`

### Finding

JOSS requires the "AI usage disclosure" statement to contain three specific elements:

1. **Tool use:** the tools/models used **and their versions**, and **where** they were
   used (code, paper text, docs).
2. **Nature and scope of assistance:** e.g. code generation, refactoring, test
   scaffolding, copy-editing, drafting.
3. **Confirmation of review:** an assertion that human authors reviewed, edited and
   validated all AI-assisted outputs and made the core design decisions.

The paper currently says only that "Generative AI tools assisted with documentation
refinement and editing," plus a review confirmation. It names no tool, gives no
version, does not say where AI was used beyond documentation, and understates scope
if AI assisted with code or tests.

JOSS states that failure to provide a complete and accurate disclosure "may be
considered an ethical breach," with consequences including desk rejection and
post-publication withdrawal.

### Required action

Only the author can supply this. The disclosure must name each tool and version, the
areas it touched, and the nature of the assistance. This cannot be inferred from the
repository and must not be guessed.

## RR-007 — Spatial and spatiotemporal coverage remain unvalidated (cost)

- **Severity:** medium
- **Category:** validation coverage
- **Status:** resolved (2026-11-05)
- **Affected files:** `tests/test_ci_coverage.py`, `scripts/coverage_benchmark.py`,
  `docs/coverage_benchmark.md`, `paper.md`

Measured single-fit costs: spatial Matern GP is 6.9 s with the default
`nu_grid=(0.5, 1.5, 2.5)` and per-cluster nugget, 1.9 s with `nu` fixed at 1.5, and
1.2 s at n = 40. A 400-replicate simulation is therefore 8-46 minutes, and the
spatiotemporal engine adds a 400-draw parametric bootstrap on top of that.

These cannot run in CI. They need a separate benchmark script that is run
deliberately, writes a committed results artifact, and is cited by the paper. Until
then the paper must continue to state that these two engines are not coverage
validated. Supersedes the spatial/spatiotemporal portion of SB-002.

### Resolution

The two engines were split by cost rather than treated as one gap.

Re-measurement corrected the premise above. The spatiotemporal engine does **not**
engage a parametric bootstrap in its default configuration: on balanced panels it
takes the joint separable ML path and costs about 0.04 s per fit. A 200-replicate
study therefore runs in roughly 12 s, so it was added to the test suite
(`TestSpatioTemporalCoverage`) and runs on every commit.

The spatial engine's cost is real but is driven by the profile-likelihood optimizer
rather than by sample size, so shrinking the design does not reliably make it
cheaper. It is validated offline by `scripts/coverage_benchmark.py`, which records
configuration, seeds, estimator path, and environment, and writes
`docs/coverage_benchmark.md`.

Measured coverage against a nominal 0.900, 300 replicates, Monte Carlo SE 0.017:

| engine | coverage | mean CI width | s/fit |
|--------|---------:|--------------:|------:|
| spatial | 0.890 | 0.126 | 0.89 |
| spatiotemporal | 0.893 | 0.560 | 0.04 |

Both lie within one Monte Carlo standard error of nominal, and both were measured
under a deliberately misspecified covariance function: the generator draws from a
squared-exponential field while the engines fit a Matern model. The spatial run
fixes `nu_grid=(0.5,)` rather than profiling the default grid, so it validates the
exponential-smoothness configuration rather than the full default search; this is
recorded in the artifact.

The gap was therefore one of measurement, not correctness. `paper.md` no longer
claims these engines lack coverage validation.

---

## RR-003 resolution (2026-11-04)

Four estimators were compared by simulation before any default was changed, 400
replicates per cell, nominal 0.900, AR(1) paired differences:

| estimator | rho=0 | rho=0.5 | rho=0.7 | rho=0.9 |
|-----------|-------|---------|---------|---------|
| Newey--West HAC (old default) | 0.858 | 0.787 | 0.715 | 0.485 |
| AR(1) GLS, corrected (new default) | 0.875 | 0.873 | 0.860 | 0.858 |
| IID | 0.880 | 0.608 | 0.468 | 0.278 |

A lag sweep from 4 to 32 at rho=0.7 topped out at 0.757, so the HAC shortfall is
finite-sample bias plus a normal critical value, not lag selection. Plain AR(1) GLS
was also insufficient on its own (0.718 worst case); the two small-sample corrections
that close the gap are a Marriott--Pope/Kendall bias correction of the autoregressive
coefficient and a Student-t critical value at Bartlett's effective sample size.

Before the default was changed, the candidate was checked against misspecification on
AR(2), MA(1), and ARMA(1,1) processes. It beat HAC on every process and over-covers
(0.93--0.96) on MA(1) and ARMA(1,1), which is the conservative direction for an
equivalence decision. No single autoregressive order dominated across processes, so
`ar_order` is exposed as a lever rather than selected automatically, and `method="hac"`
remains available for dependence that a low-order autoregression does not capture.

**Changes:** `TemporalTOST` gained `method` and `ar_order`; `WorkflowOptions` gained
`temporal_method` and `temporal_ar_order`. Replicated time points are now averaged
before fitting, which also resolves the RR-001 warning path, since the engine models a
single series. Coverage assertions live in `tests/test_ci_coverage.py`, including a
test that keeps HAC's anticonservatism documented.

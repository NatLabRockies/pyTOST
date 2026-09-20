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

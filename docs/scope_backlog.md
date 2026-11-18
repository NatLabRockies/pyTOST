# Scope backlog

Useful work that is out of scope for the slice that discovered it. Items here are not
commitments; they are recorded so the next session can prioritise them from repository
evidence rather than rediscovering them.

---

## SB-001 — Decide how the temporal engine should handle panel data

- **Raised:** 2026-09-20, from `docs/review_register.md` RR-001
- **Status:** Resolved, including documentation follow-up in
  [#2](https://github.com/NatLabRockies/pyTOST/issues/2).
- **Blocking:** No. The immediate reproducibility bug is fixed and the unsound case now
  warns.
- **Affected files:** `pyTOST/engines/temporal_tost.py`, `pyTOST/workflow.py`

`engine="temporal"` applies a Newey--West HAC estimator, which assumes a single sequence
indexed by time. Callers can and do pass panels (several series observed at the same
time points). The engine currently warns and proceeds, treating cross-sectional
replicates as consecutive time points, which does not match the data-generating process.

Options, in increasing order of effort:

1. Reject tied time values by default (`require_unique_times=True`) and direct users to
   the cluster or spatiotemporal engines. Simple and safe, but breaks existing callers.
2. Aggregate to one observation per time point before fitting. Well defined and keeps
   the engine usable on panels, but changes the estimand and discards within-time
   information.
3. Add a panel-aware estimator (Driscoll--Kraay standard errors, or cluster-robust
   inference with time-series correlation within clusters). Statistically the right
   answer for panels, but this is new functionality and needs its own validation.

A decision should also settle whether `run_tost` routes panel data automatically or
requires the user to choose an engine explicitly.

**Reconciled 2026-11-18.** The description above records the original problem, not
current behavior. `TemporalTOST` now defaults to corrected autoregressive GLS,
averages responses within tied time points before fitting, and supports strict
rejection with `require_unique_times=True`. No new panel estimator is required to
close this item. The README now explains the GLS default, explicit HAC selection,
within-time aggregation, its weighting implications, and strict rejection through
the engine class. Engine selection remains explicit.

## SB-002 — Extend coverage validation to the remaining engines

- **Raised:** 2026-09-20, during Month 3 validation work
- **Blocking:** No

`tests/test_ci_coverage.py` validates interval coverage by Monte Carlo simulation for
the IID and cluster engines only. The temporal, spatial, spatiotemporal, and
heteroskedastic engines have correctness tests but no coverage simulation. The spatial
and spatiotemporal engines are the expensive ones to simulate and would need to run
under the `slow` marker, or in a scheduled CI job rather than on every push.

**Resolved 2026-11-05.** Every engine now has coverage validation. The temporal,
heteroskedastic, and spatiotemporal engines run in the test suite; the spatial engine
runs offline via `scripts/coverage_benchmark.py`, which writes
`docs/coverage_benchmark.md`. See RR-003 and RR-007 in `docs/review_register.md` for
the measured numbers and the reason for the offline/CI split.

## Public follow-up tracking

The historical entries above are not new feature commitments. Remaining work:

- [#2](https://github.com/NatLabRockies/pyTOST/issues/2): Resolved. Temporal-engine user
  documentation matches the GLS default and aggregation behavior (SB-001 follow-up).
- [#1](https://github.com/NatLabRockies/pyTOST/issues/1): Reconcile the historical JOSS readiness plan and checklist with completed work
  and current submission criteria.
- [#3](https://github.com/NatLabRockies/pyTOST/issues/3): Verify the start of public development and document the six-month eligibility
  date; repository creation and imported commit dates alone do not prove when
  development became public.

Closed enhancements are not reopened merely to populate the tracker.

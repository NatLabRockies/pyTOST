"""Static guidance content served by the pyTOST MCP server.

Keep this file plain data + light formatting logic; no pyTOST imports so the
MCP server can start even in a minimal environment.
"""

from __future__ import annotations

OVERVIEW = """\
pyTOST overview
================

pyTOST is a Python package for dependence-aware equivalence testing with
validation. It applies the two one-sided tests (TOST) framework to paired
differences when observations may be IID, clustered, temporal, spatial, or
spatiotemporal.

Across all engines the estimand is the same: the mean paired difference.
For a given equivalence margin, pyTOST estimates the mean difference,
constructs a confidence interval, and declares equivalence when the
interval lies entirely inside (-margin, +margin).

Use pyTOST when you need to:
- test equivalence (not just "no significant difference") between two
  measurement methods, models, or conditions
- account for dependence structure (clustering, temporal autocorrelation,
  spatial/spatiotemporal correlation) rather than assuming IID
- run sensitivity analyses (heteroskedastic, robust-location) or bootstrap
  sanity checks alongside the primary TOST result

Do not use pyTOST for: standard hypothesis testing of a difference (use a
plain t-test/TOST library without dependence-awareness); prediction/fitting
tasks unrelated to equivalence testing.

Engines (see `engine_guide`): `iid`, `cluster`, `temporal`, `spatial`,
`spatiotemporal`, plus sensitivity engines `heteroskedastic` and
`robust_location`, and `building_aware` for building-level clustered data.
"""

ENGINE_GUIDE = """\
pyTOST engine selection guide
==============================
- `iid`      -- observations are independent (no clustering/temporal/spatial
                dependence). Fastest, simplest CI.
- `cluster`  -- paired differences are grouped (e.g. by site, subject,
                building) with within-cluster correlation. Use cluster-robust
                variance / cluster bootstrap.
- `building_aware` -- special case of clustered data where clusters are
                buildings with a known aware/robust variance structure.
- `temporal` -- observations form a time series with autocorrelation; use a
                dependence-aware variance estimator (e.g. Newey-West style)
                instead of assuming IID.
- `spatial`  -- observations have spatial coordinates and spatial
                autocorrelation (e.g. Matern covariance); variance is
                estimated accounting for spatial dependence.
- `spatiotemporal` -- both spatial and temporal dependence present.

After choosing an engine, run `run_tost(...)` with the equivalence margin
and engine-specific dependence arguments (cluster ids, timestamps,
coordinates). Follow up with sensitivity analyses
(`heteroskedastic`, `robust_location`) and bootstrap sanity checks before
reporting results, especially near the equivalence boundary.
"""

STRENGTHS_AND_WEAKNESSES = """\
pyTOST strengths & weaknesses
===============================
Strengths: consistent TOST workflow across IID/clustered/temporal/spatial/
spatiotemporal settings; validation-oriented sensitivity analyses and
bootstrap sanity checks built in; structured synthetic data generation for
benchmarking and reproducible examples; a canonical demonstration notebook
comparing all engines on the same synthetic dataset.

Weaknesses: pyTOST is library-first -- CLI-oriented workflows are
intentionally out of scope unless explicitly accepted on the project
roadmap; dependence-structure misspecification (choosing the wrong engine
for the true dependence pattern) can bias the confidence interval and the
equivalence conclusion, so engine selection should be checked against the
data-generating process, not assumed.
"""

DOC_RESOURCES: dict[str, str] = {
    "readme": "README.md",
    "contributing": "CONTRIBUTING.md",
}

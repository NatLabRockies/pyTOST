# pyTOST

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22858610.svg)](https://doi.org/10.5281/zenodo.22858610)

pyTOST is a Python package for **dependence-aware equivalence testing with validation**. It applies the **two one-sided tests (TOST)** framework to paired differences when observations may be **IID**, **clustered**, **temporal**, **spatial**, or **spatiotemporal**.

Across all engines, pyTOST targets the same estimand: the **mean paired difference**. For each equivalence margin `Δ`, the package estimates the mean difference `μ̂`, constructs a confidence interval for `μ`, and declares **equivalence** when the interval lies entirely inside `(-Δ, Δ)`.

The distinguishing feature of pyTOST is that the confidence interval is adapted to the assumed dependence structure rather than relying on an IID approximation when dependence is present.

## Core features

- **Common TOST workflow** across IID, clustered, temporal, spatial, and spatiotemporal settings
- **Validation-oriented sensitivity analyses**, including heteroskedastic and robust-location checks
- **Bootstrap sanity checks** for the mean paired difference
- **Structured synthetic data generation** for benchmarking, testing, calibration, and reproducible examples
- **Canonical demonstration notebook** showing the same synthetic dataset analyzed with all major engines

## Installation

```bash
pip install pyTOST
```

For development from a local clone:

```bash
pip install -e .
```

For local testing during development:

```bash
pip install -e ".[test]"
```

pyTOST currently requires **Python 3.10+** and relies on the scientific Python stack plus spatial tooling. The `rpy2` integration is optional; install it only if you need the R-backed workflow and have a working **R** installation that `rpy2` can bind to.

pyTOST currently declares the following runtime dependencies:

- `numpy`
- `pandas`
- `scipy`
- `statsmodels`
- `matplotlib`
- `nbformat`
- `esda`
- `libpysal`

To include the optional R integration:

```bash
pip install "pyTOST[r]"
```

Some optional functionality in the spatial engine becomes more complete when PySAL and R-related tooling are available, but the core library workflow remains `run_tost(...)`.

## What pyTOST expects

pyTOST works with a `pandas.DataFrame` containing a paired-difference column. In the examples below we use:

- `diff`: paired difference to test
- `cluster_id`: cluster identifier for grouped designs
- `x`: x-coordinate
- `ycoord`: y-coordinate
- `time`: time index

These names are only conventions used in the documentation. The API accepts any column names you provide.

## Quick start

```python
import pandas as pd
from pyTOST import run_tost, WorkflowOptions

df = pd.DataFrame(
    {
        "diff": [0.10, 0.18, 0.05, 0.12, 0.08, 0.15],
        "cluster_id": ["A", "A", "B", "B", "C", "C"],
    }
)

res = run_tost(
    df,
    y="diff",
    margins=[0.5],
    alpha=0.05,
    engine="cluster",
    cluster="cluster_id",
    options=WorkflowOptions(
        do_sensitivity=True,
        bootstrap_B=500,
        seed=42,
    ),
)

print(res["primary"][["delta", "mu_hat", "ci_low", "ci_high", "equivalent"]])
```

## Choosing an engine

Choose the engine based on the dependence structure you want the interval to respect.

### IID

Use `engine="iid"` when rows can reasonably be treated as independent.

```python
res = run_tost(
    df,
    y="diff",
    margins=[1.0],
    engine="iid",
)
```

### Cluster

Use `engine="cluster"` when observations may be dependent within clusters but independent across clusters.

```python
res = run_tost(
    df,
    y="diff",
    margins=[1.0],
    engine="cluster",
    cluster="cluster_id",
)
```

### Temporal

Use `engine="temporal"` when dependence is primarily along time.

```python
res = run_tost(
    df,
    y="diff",
    margins=[1.0],
    engine="temporal",
    time="time",
)
```

### Spatial

Use `engine="spatial"` when paired differences are spatially correlated within clusters.

```python
from pyTOST import SpatialConfig

res = run_tost(
    df,
    y="diff",
    margins=[1.0],
    engine="spatial",
    cluster="cluster_id",
    x="x",
    ycoord="ycoord",
    spatial_config=SpatialConfig(
        nu_grid=(0.5, 1.5, 2.5),
        verbose_diagnostics=False,
    ),
)
```

### Spatiotemporal

Use `engine="spatiotemporal"` when dependence is joint in space and time.

```python
from pyTOST import SpatioTemporalConfig

res = run_tost(
    df,
    y="diff",
    margins=[1.0],
    engine="spatiotemporal",
    cluster="cluster_id",
    time="time",
    x="x",
    ycoord="ycoord",
    spatiotemporal_config=SpatioTemporalConfig(
        nu_grid=(0.5, 1.5, 2.5),
        verbose_diagnostics=False,
    ),
)
```

### Heteroskedastic

Use `engine="heteroskedastic"` when the variance of the paired differences is not
constant across observations. Without `cluster`, it uses HC3 robust inference; with
`cluster`, it uses a cluster-robust wild bootstrap (Rademacher multipliers), which is
better behaved when the number of clusters is small.

```python
res = run_tost(
    df,
    y="diff",
    margins=[0.5],
    engine="heteroskedastic",
    cluster="cluster_id",  # optional
)
```

## Worked example

For a complete, end-to-end analysis — choosing an engine, reading the summary, and
interpreting disagreement between sensitivity analyses — see the
[SAV method equivalence worked example](docs/sav_worked_example.md). It shows a case
where ignoring spatial and temporal dependence reverses the equivalence decision.

## Interpreting results
For each margin `Δ`, pyTOST reports a confidence interval for the mean paired difference `μ`.

- If the interval lies entirely inside `(-Δ, Δ)`, pyTOST declares **equivalence**.
- If the interval crosses either margin, pyTOST does **not** declare equivalence.
- Wider intervals under clustered, temporal, spatial, or spatiotemporal models are often expected and can be more defensible than an IID interval that ignores dependence.

In practice:

- treat the selected engine’s `primary` result as the main inference
- use `sensitivity` results to assess whether the conclusion is fragile to modeling choices
- use the `bootstrap` result as a validation-oriented sanity check on uncertainty

## What `run_tost(...)` returns

`run_tost(...)` returns a dictionary with:

- `engine`: the engine that was run
- `primary`: the main result table for the selected engine
- `sensitivity`: optional sensitivity-analysis result tables
- `bootstrap`: optional validation bootstrap summary

The `primary` table typically includes:

- `delta`: equivalence margin
- `mu_hat`: estimated mean paired difference
- `ci_low`, `ci_high`: confidence interval bounds
- `equivalent`: whether the CI is entirely inside `(-Δ, Δ)`

### Human-readable summary

The returned object is a `dict` subclass (`TOSTResult`), so all dictionary access
shown above keeps working. It also provides `summary()`, which renders the
equivalence decision as a report-ready table:

```python
res = run_tost(df, y="diff", margins=[0.25, 0.5], engine="cluster", cluster="cluster_id")
print(res.summary())
```

```text
pyTOST equivalence summary
==========================

Engine: cluster
Method: OLS + cluster-robust SE (df=2)

Primary result
--------------
Delta  mu_hat  ci_low  ci_high  Decision
-----  ------  ------  -------  ----------
0.25   0.1133  0.0669  0.1598   EQUIVALENT
0.5    0.1133  0.0669  0.1598   EQUIVALENT
```

Sensitivity analyses and the bootstrap sanity check are included in the report when
they were enabled. Use `summary(precision=...)` to control the number of decimals.

### Plotting the decision

`plot_ci()` renders the same decision as a report-ready matplotlib figure. Each engine
becomes a horizontal CI bar; dashed lines mark `-Δ` and `+Δ`, and bars are colored by
the equivalence decision.

```python
from pyTOST import plot_ci

fig, ax = plot_ci(res, margin=0.5)
fig.savefig("equivalence.png", dpi=200, bbox_inches="tight")
```

`plot_ci()` accepts either a `run_tost(...)` result — plotting the primary engine
alongside any sensitivity analyses — or a plain `{label: DataFrame}` mapping to
compare engines you ran yourself. Pass `ax=` to draw into an existing figure.

## Sensitivity analyses and validation

The workflow can optionally include:

- heteroskedastic-robust inference
- robust-location equivalence checks
- bootstrap validation for the mean paired difference

These are controlled through `WorkflowOptions`:

```python
from pyTOST import WorkflowOptions

options = WorkflowOptions(
    do_sensitivity=True,
    bootstrap_B=500,
    robust_location_B=100,
    robust_location_block_len=5,
    robust_location_stat="median",
    seed=42,
    spatial_block_size=1.0,
)
```

Then pass `options=options` into `run_tost(...)`.

### Choosing the robust location statistic

The robust-location check defaults to the median, which is maximally resistant to
outliers but discards most of the sample. Set `robust_location_stat="trimmed_mean"` to
use a trimmed mean instead, and `robust_location_trim` to choose how much of *each*
tail is removed:

```python
options = WorkflowOptions(
    do_sensitivity=True,
    robust_location_stat="trimmed_mean",
    robust_location_trim=0.1,  # drop the lowest and highest 10%
)
```

The trim fraction interpolates between the two familiar estimators: `0.0` reproduces the
sample mean, and values approaching `0.5` approach the median. A light trim (0.05–0.10)
is a reasonable default when you want to discount a few contaminated observations
without giving up the efficiency of the mean. Valid values satisfy
`0.0 <= trim < 0.5`; anything else raises `ValueError`, as does an unrecognized
`robust_location_stat`.

### Controlling the temporal HAC lag

The temporal engine uses a Newey–West HAC variance estimator. By default the truncation
lag is selected from the sample size via the Newey–West (1994) plug-in rule
`floor(4 * (n / 100) ** (2 / 9))`, so it grows with the length of the series. Use
`max_lag` to override it:

```python
options = WorkflowOptions(max_lag=12)       # fixed lag
options = WorkflowOptions(max_lag="auto")   # data-driven lag (the default)
options = WorkflowOptions(max_lag=4)        # reproduce pyTOST < 0.17.0
```

> **Changed in 0.17.0.** Earlier versions used a fixed lag of `4` regardless of sample
> size, which under-smoothed long series and over-smoothed short ones. Pass
> `max_lag=4` explicitly to reproduce results from an earlier version.

The temporal engine expects **one observation per time point**. If several rows share a
time value, HAC inference treats them as consecutive time points, which will not reflect
the real autocorrelation; the engine warns in that case. Aggregate within time points
first, or use the `cluster` or `spatiotemporal` engine, which model cross-sectional
replication explicitly.

Longer lags admit more autocorrelation into the variance estimate and generally widen
the interval. `"auto"` is recommended when the series length varies between analyses;
the default of `4` is retained so results from earlier versions stay reproducible.

## Synthetic data generation

pyTOST includes structured synthetic data generators for reproducible benchmarking, testing, calibration, and examples. These utilities are ancillary to the inference engines but are part of the package and are used in the demonstration notebook and test plan.

Current generators live in:

- `pyTOST.data_gen.synthetic_tost_data`
- `pyTOST.data_gen.params_io`

Example imports:

```python
from pyTOST.data_gen.synthetic_tost_data import (
    generate_iid,
    generate_iid_grouped,
    generate_cluster_groups,
    generate_spatial_clusters,
    generate_temporal_ar1,
    generate_spatiotemporal,
)
from pyTOST.data_gen.params_io import load_params
```

These generators support settings such as:

- IID paired samples
- grouped or clustered dependence
- AR(1) temporal dependence
- spatial Matérn-correlated data
- separable spatiotemporal dependence

## Canonical example notebook

The repository includes demonstration notebooks:

- `examples/pyTOST_basic_demo.ipynb`
- `examples/pyTOST_advanced_demo.ipynb`

This notebook is intended to be the canonical worked example. It uses synthetic data to show how to:

1. generate or load a structured synthetic example,
2. construct a paired-difference analysis table,
3. run the IID, cluster, temporal, spatial, and spatiotemporal engines,
4. compare confidence intervals and equivalence decisions, and
5. interpret differences in uncertainty across engines.

## Public API

The main public imports are:

```python
from pyTOST import (
    run_tost,
    WorkflowOptions,
    IIDTOST,
    ClusterTOST,
    TemporalTOST,
    SpatialTOST,
    SpatialConfig,
    SpatioTemporalTOST,
    SpatioTemporalConfig,
    HeteroskedasticTOST,
    RobustLocationTOST,
)
```

For most users, the recommended entry point is:

```python
from pyTOST import run_tost
```

## Documentation conventions

In the examples and documentation:

- `cluster_id` is the generic grouped-data identifier
- `diff` is the paired-difference column
- `x` and `ycoord` are coordinate columns
- `time` is the temporal index

These are example names only. The package does not require these exact column names.


## Contributing and support

Please use the repository issue tracker to:

- report bugs and installation issues,
- ask usage questions,
- request features, or
- discuss documentation improvements.

See `CONTRIBUTING.md` for the recommended development workflow and local test commands.

## Development status

pyTOST is stable and suitable for research use. The public API — `run_tost`,
`WorkflowOptions`, `TOSTResult`, `plot_ci`, and the individual engine classes — is
settled, and breaking changes will be announced in [CHANGELOG.md](CHANGELOG.md) ahead of
any release that makes them.

What that rests on:

- six inference engines (`iid`, `cluster`, `temporal`, `spatial`, `spatiotemporal`,
  `heteroskedastic`), each covered by automated tests
- confidence-interval coverage validated by Monte Carlo simulation and, for the IID
  engine, checked against the closed-form Student-t interval
- continuous integration runs the full test suite on every push to `main` and every
  pull request
- a reproducible environment specification (`pixi.toml`, `pixi.lock`) and a worked
  example in [`docs/sav_worked_example.md`](docs/sav_worked_example.md)

Current development focuses on broadening the validation suite, documenting additional
applied examples, and responding to user-reported issues. Bug reports and feature
requests are welcome through the GitHub issue tracker; see `CONTRIBUTING.md` for the
development workflow.

Release-by-release changes are recorded in [CHANGELOG.md](CHANGELOG.md).

## Citation

Every pyTOST release is archived on Zenodo. Cite the version you actually used, or the
concept DOI below, which always resolves to the most recent release:

- All versions (concept DOI): [10.5281/zenodo.22858610](https://doi.org/10.5281/zenodo.22858610)
- v0.17.0: [10.5281/zenodo.22858611](https://doi.org/10.5281/zenodo.22858611)

```bibtex
@software{pyTOST,
  author  = {Hettinger, Dylan},
  title   = {pyTOST: Dependence-aware equivalence testing with validation in Python},
  year    = {2026},
  version = {0.17.0},
  doi     = {10.5281/zenodo.22858610},
  url     = {https://github.com/NatLabRockies/pyTOST}
}
```

A JOSS paper describing the package is in preparation; this section will be updated with
its citation once it is published.

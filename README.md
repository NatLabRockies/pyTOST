# pyTOST

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

Once published to PyPI:

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

pyTOST currently requires **Python 3.10+** and relies on the scientific Python stack plus spatial tooling. Because `rpy2` is a declared runtime dependency, a fully fresh installation also requires a working **R** installation that `rpy2` can bind to.

pyTOST currently declares the following runtime dependencies:

- `numpy`
- `pandas`
- `scipy`
- `statsmodels`
- `matplotlib`
- `nbformat`
- `esda`
- `libpysal`
- `rpy2`

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

The repository includes a demonstration notebook:

- `pyTOST_basic_demo.ipynb`
- `pyTOST_advanced_demo.ipynb`

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

pyTOST is being prepared for open-source release and JOSS submission as a library-first package for dependence-aware equivalence testing with validation.

The near-term priorities are:

- packaging and install metadata
- automated tests across all engines
- cleanup of the canonical demo notebook
- harmonization of documentation and JOSS paper materials

## Citation

If you use pyTOST in research, please cite the accompanying JOSS paper once available.

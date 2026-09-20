---
title: "pyTOST: Dependence-aware equivalence testing with validation in Python"
tags:
  - Python
  - statistics
  - hypothesis-testing
  - time-series
  - spatial
  - spatiotemporal
authors:
  - name: Dylan Hettinger
    orcid: 0009-0000-5769-2333
    affiliation: "1"
affiliations:
  - index: 1
    name: "National Laboratory of the Rockies (NLR), operated by Alliance for Energy Innovation, LLC"
date: 2026-09-20
bibliography: paper.bib
repository: https://github.com/NatLabRockies/pyTOST
---

# Summary

`pyTOST` is a Python package for dependence-aware equivalence testing with validation. It applies the two one-sided tests (TOST) framework to paired differences when observations may be independent and identically distributed (IID), clustered, temporal, spatial, or spatiotemporal [@Lakens2017; @Wellek2010]. Across engines, `pyTOST` targets the mean paired difference $\mu$ and declares equivalence at margin $\Delta$ when a $(1-2\alpha)$ confidence interval lies entirely inside $(-\Delta,\Delta)$ [@Lakens2017]. Unlike classical null-hypothesis significance tests, this framework is designed for the validation question "are two methods practically interchangeable at a decision-relevant tolerance?" [@Lakens2017].

The main contribution of `pyTOST` is that it adapts uncertainty quantification to the assumed dependence structure rather than relying on an IID approximation when dependence is present. The package provides inference engines for IID, clustered, temporal, spatial, spatiotemporal, and heteroskedastic settings, together with validation-oriented sensitivity analyses. Clustered inference uses sandwich/cluster-robust variance estimators [@CameronMiller2015]. Temporal inference supports heteroskedasticity-and-autocorrelation-consistent (HAC) variance via Newey--West [@NeweyWest1987]. Spatial inference models within-cluster dependence using a Matérn covariance with nugget, estimates covariance parameters by profile maximum likelihood (with ν selected by profile over a candidate grid), and forms a profile likelihood-ratio confidence interval for $\mu$ [@Stein1999; @GuttorpGneiting2006; @Pawitan2001]. For balanced panels, the spatiotemporal engine fits a separable AR(1) $\otimes$ Matérn covariance model and reports a parametric-bootstrap confidence interval for $\mu$; for unbalanced panels it falls back to per-time spatial fits combined by inverse-variance weighting [@EfronTibshirani1993; @CressieWikle2011].

`pyTOST` also includes structured synthetic data generation utilities and an automated test suite (`pytest`). The package is implemented in the scientific Python stack, relying primarily on `NumPy` [@Harris2020], `pandas` [@McKinney2010], `SciPy` [@Virtanen2020], and `statsmodels` [@SeaboldPerktold2010], with `PySAL`-based spatial diagnostics [@Rey2021] and optional R interoperability through `rpy2` for selected sensitivity paths.

```bash
pip install pyTOST
```

```python
import pandas as pd
from pyTOST import run_tost, WorkflowOptions

df = pd.DataFrame({
    "diff": [0.10, 0.18, 0.05, 0.12, 0.08, 0.15],
    "cluster_id": ["A", "A", "B", "B", "C", "C"],
})

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

# Statement of need

Equivalence testing is required when developers must demonstrate that a new measurement system, algorithm, simulation workflow, or statistical model is *close enough* to a reference method for practical use. Examples include validating that an updated pipeline yields outputs equivalent to an operational baseline, or that a new measurement procedure is equivalent to a trusted instrument under a prespecified tolerance. Standard null-hypothesis significance tests (NHST) cannot establish equivalence: a non-significant difference does not imply practical similarity [@Lakens2017]. TOST addresses this by reversing the null hypothesis and asking whether the true mean difference falls outside a practically acceptable margin [@Wellek2010; @Lakens2017].

In realistic validation settings, however, equivalence decisions are often driven less by the point estimate than by how uncertainty is quantified. That uncertainty depends on the dependence structure in the data. Ignoring clustering, serial correlation, or spatial dependence will often understate the standard error $\mathrm{SE}(\hat\mu)$ and produce confidence intervals that are too narrow, making equivalence appear easier to establish than it really is. Practitioners can assemble partial solutions from existing libraries---for example, basic TOST in independent settings, cluster-robust covariance estimators, or geostatistical covariance models---but these pieces are not presented as a unified equivalence-testing workflow. pyTOST fills this gap by coupling TOST with dependence-aware inference and validation-oriented sensitivity analyses in a single Python package.

# State of the field

**Equivalence testing software.** In Python, `Pingouin` provides general-purpose statistics including TOST for independent or paired samples [@Vallat2018]. `Statsmodels` includes TOST for mean differences, alongside a broad set of estimators and robust covariance options [@SeaboldPerktold2010]. In R, `TOSTER` implements many equivalence tests for t-tests, correlations, and other common settings [@TOSTERpkg]. These tools are valuable for standard designs, but they do not provide a cohesive workflow for equivalence testing when the primary challenge is dependence-aware uncertainty quantification across clustered, temporal, spatial, and spatiotemporal settings.

**Dependence-aware inference components.** Cluster-robust inference is well established in applied econometrics and biostatistics [@CameronMiller2015]. HAC variance estimators such as Newey--West are standard for serial dependence [@NeweyWest1987]. Spatial and spatiotemporal dependence is commonly modeled through Matérn covariance families and their extensions [@Stein1999; @GuttorpGneiting2006; @CressieWikle2011]. Python tooling supports many of these ingredients individually, including robust regression infrastructure in `statsmodels` and spatial diagnostics in `PySAL` [@Rey2021]. However, practitioners still typically need to stitch these pieces together manually if their actual goal is a TOST-based validation decision under dependence.

`pyTOST` fills this gap with a single interface spanning classical IID TOST, clustered and temporal robust covariance estimation, model-based spatial and spatiotemporal inference, and validation-oriented sensitivity analyses. The package includes an automated test suite (`pytest tests/`) covering all engines, the workflow, data generators, and the public API. Interval coverage is checked directly: the IID engine's interval is verified against the closed-form Student-$t$ interval, and Monte Carlo simulation confirms that the IID and cluster engines attain their nominal coverage on data matching their assumptions (0.892 and 0.907 against a nominal 0.900). The same simulation quantifies the failure mode motivating the package: applying the IID engine to clustered data attains only 0.613 coverage.

# Software design

`pyTOST` is organized around inference "engines" sharing a common workflow and output schema. Users call `run_tost(...)` with a data frame, a paired-difference column, equivalence margins, and an engine choice. Each engine estimates $\hat\mu$ and constructs a $(1-2\alpha)$ confidence interval; equivalence is declared when the interval lies inside $(-\Delta,\Delta)$ [@Lakens2017]. Core computation uses `NumPy`, `pandas`, and `SciPy` [@Harris2020; @McKinney2010; @Virtanen2020]; regression and robust covariance estimation use `statsmodels` [@SeaboldPerktold2010].

## Inference engines

- **IID engine.** Intercept-only OLS with a Student-$t$ CI ($df = n-1$).

- **Cluster engine.** Intercept-only OLS with cluster-robust (sandwich) variance and conservative $df = G-1$ degrees of freedom, where $G$ is the number of clusters [@CameronMiller2015].

- **Temporal engine.** Intercept-only OLS with Newey--West HAC variance; the CI uses a standard normal critical value, consistent with the asymptotic justification of HAC estimators [@NeweyWest1987]. An AR(1) GLS path is available by calling `TemporalTOST` directly.

- **Spatial engine.** Gaussian process with Matérn covariance plus nugget [@Stein1999; @GuttorpGneiting2006]. Covariance parameters are estimated by profile maximum likelihood with smoothness $\nu$ selected over a candidate grid; the mean is estimated by GLS and uncertainty is quantified via a profile likelihood-ratio CI [@Pawitan2001].

- **Heteroskedastic engine.** Intercept-only OLS with HC3 heteroskedasticity-robust variance, or cluster-robust wild bootstrap inference when a cluster variable is supplied [@CameronGelbachMiller2008].

- **Spatiotemporal engine.** For balanced panels, fits a separable AR(1)$\otimes$Matérn model by penalized ML (MAP-regularized to stabilize near-unidentifiable solutions) with a parametric-bootstrap CI for $\mu$ [@EfronTibshirani1993; @CressieWikle2011]. For unbalanced panels, fits spatial models per time slice and combines estimates by inverse-variance weighting with a $t$-CI at $df = T-1$, where $T$ is the number of time slices.

The practical consequence of matching the engine to the dependence structure is shown in \autoref{fig:engines}. Five engines are applied to the same synthetic spatiotemporal panel and, run in equal-weighted mode, all target the same estimand, so they return an identical point estimate $\hat{\mu} = 0.26$. Their confidence intervals differ substantially: the IID engine, which ignores the spatial and temporal dependence that is genuinely present in the data, reports an interval roughly 2.5 times narrower than the cluster-robust engine and 6 times narrower than the spatiotemporal engine. At $\Delta = 0.6$ this difference is decision-relevant — the IID engine declares equivalence, while the cluster and spatiotemporal engines do not. Understating dependence therefore does not merely produce optimistic intervals; it can reverse the validation conclusion.

![Confidence intervals for the mean paired difference $\mu$ from all five `pyTOST` engines applied to the same synthetic spatiotemporal dataset ($n = 192$; 24 sites $\times$ 8 times), with equivalence margin $\Delta = 0.6$ shown as dashed lines. Green intervals fall entirely inside $(-\Delta, \Delta)$ and yield an equivalence decision; red intervals do not. Because all engines are run against the same estimand, the figure isolates the effect of the dependence assumption on interval width. The figure is reproduced by `scripts/make_paper_figure.py`.\label{fig:engines}](paper_figure.png)

## Sensitivity analyses and validation

`pyTOST` includes optional checks to assess whether an equivalence decision is stable to modeling choices:

- **Heteroskedastic-robust CIs** (HC3 or cluster-robust), combined conservatively with wild cluster bootstrap [@CameronGelbachMiller2008], computed by `HeteroskedasticTOST` and surfaced through the `run_tost` sensitivity output.
- **Robust-location equivalence** based on the median or a trimmed mean with a user-specified trim fraction, with dependence-aware bootstrap CIs [@EfronTibshirani1993].
- **Validation bootstrap summaries** for the mean, including cluster and spatial block-style variants.

## Structured synthetic data generation

The package includes generators for IID, clustered, AR(1) temporal, spatially correlated (squared-exponential/RBF covariance; note the inference engines fit a Matérn model), and separable spatiotemporal paired samples. Generators support controlled equivalence scenarios and are used by the included test suite and demonstration notebooks.

# Research impact statement

`pyTOST` is used at the National Laboratory of the Rockies (NLR) for validation workflows where equivalence decisions must be made under non-IID dependence. Use cases include model-to-model validation of geospatial and simulation pipelines whose outputs are correlated in space or time, and measurement-method validation such as solar access value (SAV) assessments where paired differences are spatially structured and repeated over time. The package provides reusable infrastructure for equivalence testing in environmental monitoring, remote sensing, manufacturing metrology, and model intercomparison.

# AI usage disclosure

Generative AI tools were used during documentation refinement and editing. Final technical claims, software descriptions, and methodological statements were reviewed against the source code and revised by the author.

# Acknowledgements

The author thanks collaborators at the National Laboratory of the Rockies for feedback on early versions of `pyTOST` and for internal validation use cases that informed the dependence-aware design.

# References


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
    affiliation: "1"
affiliations:
  - index: 1
    name: "National Laboratory of the Rockies (NLR)"
date: 27 February 2026
bibliography: paper.bib
license: MIT
repository: https://github.com/dhetting/pyTOST 
---

# Summary

`pyTOST` is a Python package for dependence-aware equivalence testing with validation. It applies the two one-sided tests (TOST) framework to paired differences when observations may be independent and identically distributed (IID), clustered, temporal, spatial, or spatiotemporal [@Lakens2017; @Wellek2010]. Across engines, `pyTOST` targets the mean paired difference $\mu$ and declares equivalence at margin $\Delta$ when a $(1-2\alpha)$ confidence interval lies entirely inside $(-\Delta,\Delta)$ [@Lakens2017]. Unlike classical null-hypothesis significance tests, this framework is designed for the validation question “are two methods practically interchangeable at a decision-relevant tolerance?” [@Lakens2017].

The main contribution of `pyTOST` is that it adapts uncertainty quantification to the assumed dependence structure rather than relying on an IID approximation when dependence is present. The package provides inference engines for IID, clustered, temporal, spatial, and spatiotemporal settings, together with validation-oriented sensitivity analyses. Clustered inference uses sandwich/cluster-robust variance estimators [@CameronMiller2015]. Temporal inference supports heteroskedasticity-and-autocorrelation-consistent (HAC) variance via Newey--West [@NeweyWest1987]. Spatial inference models within-cluster dependence using a Matérn covariance with nugget, estimates covariance parameters by restricted maximum likelihood (REML), and forms a likelihood-ratio confidence interval for $\mu$ [@Stein1999; @GuttorpGneiting2006; @Harville1977; @Pawitan2001]. For balanced panels, the spatiotemporal engine fits a separable AR(1) $\otimes$ Matérn covariance model and reports a parametric-bootstrap confidence interval for $\mu$; for unbalanced panels it falls back to per-time spatial fits combined by inverse-variance weighting [@EfronTibshirani1993; @CressieWikle2015].

`pyTOST` also includes structured synthetic data generation utilities for IID, clustered, temporal, spatial, and spatiotemporal settings. These generators are ancillary to the main inference engines, but they support reproducible examples, regression tests, benchmarking, and calibration studies. The package is implemented in the scientific Python stack, relying primarily on `NumPy` [@Harris2020], `pandas` [@McKinney2010], `SciPy` [@Virtanen2020], and `statsmodels` [@SeaboldPerktold2010], with `PySAL`-based spatial diagnostics and optional R interoperability through `rpy2` for selected sensitivity paths.

```bash
pip install pyTOST
```

```python
from pyTOST import run_tost, WorkflowOptions

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

**Dependence-aware inference components.** Cluster-robust inference is well established in applied econometrics and biostatistics [@CameronMiller2015]. HAC variance estimators such as Newey--West are standard for serial dependence [@NeweyWest1987]. Spatial and spatiotemporal dependence is commonly modeled through Matérn covariance families and their extensions [@Stein1999; @GuttorpGneiting2006; @CressieWikle2015]. Python tooling supports many of these ingredients individually, including robust regression infrastructure in `statsmodels` and spatial diagnostics in `PySAL`. However, practitioners still typically need to stitch these pieces together manually if their actual goal is a TOST-based validation decision under dependence.

`pyTOST`’s contribution is to provide a single, test-oriented interface that spans (a) classical IID TOST, (b) clustered and temporal robust covariance estimation, (c) model-based spatial and spatiotemporal inference for the mean paired difference, and (d) validation-oriented sensitivity analyses and synthetic benchmarking support.

# Software design

`pyTOST` is organized around inference “engines” that share a common workflow and output schema. Users call `run_tost(...)` with a data frame, a paired-difference column, one or more equivalence margins, and an engine choice. Each engine estimates $\hat\mu$ and constructs a $(1-2\alpha)$ confidence interval tailored to the assumed dependence structure; equivalence is then assessed by interval containment in $(-\Delta,\Delta)$ [@Lakens2017]. Core numerical computation uses `NumPy`, `pandas`, and `SciPy` [@Harris2020; @McKinney2010; @Virtanen2020], while regression and robust covariance estimation rely on `statsmodels` [@SeaboldPerktold2010].

## Inference engines

- **IID engine.** Estimates $\hat\mu=\bar y$ and reports a Student-$t$ confidence interval for the mean paired difference.

- **Cluster engine.** Uses an intercept-only mean estimate with cluster-robust (sandwich) variance and conservative cluster-based degrees of freedom, reflecting grouped dependence [@CameronMiller2015].

- **Temporal engine.** Uses an intercept-only estimate with HAC (Newey--West) variance for serially dependent paired differences [@NeweyWest1987]. An AR(1) generalized least squares sensitivity path is also available for comparison.

- **Spatial engine.** Models within-cluster dependence as Gaussian with Matérn covariance plus nugget [@Stein1999; @GuttorpGneiting2006]. Covariance parameters are estimated by REML [@Harville1977]. The mean is estimated by generalized least squares,
  $$
  \hat\mu = \frac{\mathbf{1}^\top \Sigma^{-1} y}{\mathbf{1}^\top \Sigma^{-1} \mathbf{1}},
  $$
  and uncertainty is summarized through a likelihood-ratio confidence interval obtained by profiling over $\mu$ [@Pawitan2001].

- **Spatiotemporal engine.** For balanced panels observed on a common spatial grid over time, pyTOST fits a separable AR(1)$\otimes$Matérn covariance model by maximum likelihood and reports a parametric-bootstrap confidence interval for $\mu$ [@EfronTibshirani1993; @CressieWikle2015]. For unbalanced panels, it falls back to fitting spatial models by time slice and combines the resulting mean estimates by inverse-variance weighting.

## Sensitivity analyses and validation

`pyTOST` includes optional checks designed to identify settings where an equivalence conclusion may depend strongly on modeling choices:

- **Heteroskedastic-robust confidence intervals**, including HC-type corrections.
- **Wild cluster bootstrap** for improved clustered inference with a small number of clusters [@CameronGelbachMiller2008].
- **Robust-location equivalence** based on the median or trimmed mean with bootstrap confidence intervals [@EfronTibshirani1993].
- **Validation bootstrap summaries** for the mean paired difference, including cluster bootstrap and spatial block-style variants used as workflow sanity checks.

These analyses do not redefine the primary engine-specific result; instead, they are intended to show whether the equivalence decision is stable to alternative uncertainty summaries.

## Structured synthetic data generation

`pyTOST` includes utilities for generating synthetic datasets that match the dependence structures targeted by its engines. These utilities are not the package’s primary methodological contribution, but they are important for reproducible testing and documentation.

1. **Dependence-structured generators.** The package includes generators for (i) IID paired samples, (ii) grouped or clustered samples, (iii) AR(1) temporal dependence, (iv) spatially correlated samples, and (v) separable spatiotemporal processes.

2. **Controlled equivalence scenarios.** Generators can be configured so that the true mean paired difference lies inside, on, or outside a target equivalence margin, supporting power studies and decision-stability checks.

3. **Calibration and regression testing.** The same generators support reproducible examples, unit tests, and calibration exercises in which users tune design characteristics to achieve target interval width or empirical power.

# Research impact statement

`pyTOST` is used internally at the National Laboratory of the Rockies (NLR) to support validation workflows in which equivalence decisions must be made under non-IID dependence structures.

One use case is **model-to-model validation**, including regression, geospatial, and simulation pipelines whose outputs are correlated across time and/or space. In such settings, dependence-aware confidence intervals provide more defensible equivalence conclusions than IID approximations.

A second use case is **measurement-method validation**, including solar access value (SAV) assessments where paired differences are spatially structured and repeated over time. Spatial and spatiotemporal engines allow the mean difference between two methods to be assessed while accounting for redundancy induced by spatial or spatiotemporal autocorrelation.

More broadly, `pyTOST` provides reusable statistical infrastructure for equivalence testing in environmental monitoring, remote sensing validation, manufacturing metrology, model intercomparison, and related settings where practical interchangeability must be assessed under dependence rather than assumed independence.

# AI usage disclosure

Generative AI tools were used during documentation refinement and editing. Final technical claims, software descriptions, and methodological statements were reviewed against the source code and revised by the author.

# Acknowledgements

We thank collaborators at the National Laboratory of the Rockies for feedback on early versions of `pyTOST` and for internal validation use cases that informed the dependence-aware design.

# References



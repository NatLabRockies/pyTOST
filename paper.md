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
date: 4 November 2026
bibliography: paper.bib
repository: https://github.com/NatLabRockies/pyTOST
---

# Summary

Researchers often need to show that two ways of measuring or computing a quantity agree closely enough to be used interchangeably, for example that a new instrument reproduces a trusted one, or that a revised processing pipeline reproduces the operational baseline it replaces. Equivalence testing answers this by asking whether the difference between the two is small enough to be unimportant in practice. An ordinary significance test cannot, because failing to detect a difference is not evidence that none exists [@Lakens2017].

`pyTOST` is a Python package that performs equivalence testing on paired differences using the two one-sided tests (TOST) procedure [@Lakens2017; @Wellek2010]. It estimates the mean paired difference $\mu$ and declares the two methods equivalent at a tolerance $\Delta$, chosen by the analyst from subject-matter knowledge, when a confidence interval for $\mu$ lies entirely inside $(-\Delta,\Delta)$.

The package's distinguishing feature is that it adapts this interval to the way the observations are related to one another. Measurements collected in groups, repeatedly over time, or across space are not statistically independent, and treating them as though they were yields an interval that is too narrow, making equivalence easier to declare than the data warrant. `pyTOST` provides six inference engines covering independent, clustered, temporal, spatial, and spatiotemporal data as well as unequal variances. They share one interface and one result format, so an analyst states the form of dependence believed present and obtains an interval that accounts for it. The package adds sensitivity analyses that test whether a decision survives changes in modeling assumptions, and synthetic data generators used by its test suite.

`pyTOST` builds on the scientific Python stack, relying primarily on `NumPy` [@Harris2020], `pandas` [@McKinney2010], `SciPy` [@Virtanen2020], and `statsmodels` [@SeaboldPerktold2010], with spatial diagnostics from `PySAL` [@Rey2021] and optional R interoperability through `rpy2` for selected sensitivity analyses.

# Statement of need

Equivalence testing arises whenever a new measurement system, algorithm, simulation code, or statistical model must be shown to agree with a reference method closely enough for practical use. TOST handles this by reversing the null hypothesis, taking non-equivalence as the hypothesis to be rejected and asking whether the mean difference falls outside a practically acceptable margin [@Wellek2010; @Lakens2017].

In applied validation work, however, the equivalence decision usually turns less on the point estimate than on how its uncertainty is quantified, and that uncertainty depends on the dependence structure of the data. Ignoring clustering, serial correlation, or spatial dependence understates the standard error of $\hat\mu$ and narrows the confidence interval, so equivalence appears easier to establish than the data support. The consequence is a false positive in exactly the circumstance that matters most, namely a validation study concluding that two methods agree when the evidence does not establish it.

The pieces can be assembled from existing libraries, pairing TOST for independent samples with cluster-robust or geostatistical covariance estimators. That leaves the analyst to reconcile several interfaces, decide how each variance estimator feeds the TOST decision rule, and build the comparison across dependence assumptions by hand. `pyTOST` supplies that combination directly.

# State of the field

**Equivalence testing software.** In Python, `Pingouin` provides general-purpose statistics including TOST for independent or paired samples [@Vallat2018]. `Statsmodels` includes TOST for mean differences, alongside a broad set of estimators and robust covariance options [@SeaboldPerktold2010]. In R, `TOSTER` implements many equivalence tests for t-tests, correlations, and other common settings [@TOSTERpkg]. These tools cover standard designs well, and for independent samples they are the appropriate choice. None of them, however, pairs the TOST decision rule with variance estimation for clustered, temporal, spatial, or spatiotemporal dependence.

**Dependence-aware inference components.** Cluster-robust inference is well established in applied econometrics and biostatistics [@CameronMiller2015]. HAC variance estimators such as Newey--West are standard for serial dependence [@NeweyWest1987]. Spatial and spatiotemporal dependence is commonly modeled through Matérn covariance families and their extensions [@Stein1999; @GuttorpGneiting2006; @CressieWikle2011]. Python tooling supports many of these ingredients individually, including robust regression infrastructure in `statsmodels` and spatial diagnostics in `PySAL` [@Rey2021]. However, practitioners still typically need to stitch these pieces together manually if their actual goal is a TOST-based validation decision under dependence.

The individual variance estimators are therefore already available; what is missing is the decision layer above them. Each engine must expose an interval on a common scale, apply the containment rule against a margin, and remain comparable with the others when an analyst asks whether an equivalence decision depends on the assumed dependence structure. That shared contract is what `pyTOST` provides, and it fits naturally in neither a general statistics library nor a spatial one.

# Software design

`pyTOST` is organized around inference "engines" sharing a common workflow and output schema. Users call `run_tost(...)` with a data frame, a paired-difference column, equivalence margins, and an engine choice. Each engine estimates $\hat\mu$ and constructs a $(1-2\alpha)$ confidence interval; equivalence is declared when that interval lies inside $(-\Delta,\Delta)$ [@Lakens2017].

## Inference engines

- **IID engine.** Intercept-only OLS with a Student-$t$ CI ($df = n-1$).

- **Cluster engine.** Intercept-only OLS with cluster-robust (sandwich) variance and conservative $df = G-1$ degrees of freedom, where $G$ is the number of clusters [@CameronMiller2015].

- **Temporal engine.** Intercept-only OLS with Newey--West HAC variance, whose truncation lag defaults to the sample-size-based plug-in rule $\lfloor 4(n/100)^{2/9}\rfloor$ and can be fixed by the user; the CI uses a standard normal critical value, consistent with the asymptotic justification of HAC estimators [@NeweyWest1987].

- **Spatial engine.** Gaussian process with Matérn covariance plus nugget [@Stein1999; @GuttorpGneiting2006]. Covariance parameters are estimated by profile maximum likelihood with smoothness $\nu$ selected over a candidate grid; the mean is estimated by GLS and uncertainty is quantified via a profile likelihood-ratio CI [@Pawitan2001].

- **Heteroskedastic engine.** Intercept-only OLS with HC3 heteroskedasticity-robust variance, or cluster-robust wild bootstrap inference when a cluster variable is supplied [@CameronGelbachMiller2008].

- **Spatiotemporal engine.** For balanced panels, fits a separable AR(1)$\otimes$Matérn model by penalized ML, MAP-regularized to stabilize near-unidentifiable solutions, with a parametric-bootstrap CI [@EfronTibshirani1993; @CressieWikle2011]. For unbalanced panels, fits spatial models per time slice and combines them by inverse-variance weighting with a $t$-CI at $df = T-1$ for $T$ time slices.

\autoref{fig:engines} shows what this choice does to a validation decision. Five engines analyze one synthetic spatiotemporal panel containing spatial and temporal dependence by construction. The spatial and spatiotemporal engines are configured to estimate the plain sample mean rather than their default generalized least squares estimate, so all five target the same estimand and return $\hat{\mu} = 0.26$; the comparison isolates the effect of the dependence assumption on interval width. The widths range over a factor of six, with the IID engine reporting an interval 2.5 times narrower than the cluster-robust engine and 6.2 times narrower than the spatiotemporal engine. At $\Delta = 0.6$ this decides the outcome, as the IID, temporal, and spatial engines declare equivalence while the cluster and spatiotemporal engines do not. The temporal engine is shown for comparison even though this panel repeats observations within each time point, which its HAC estimator does not model; `pyTOST` warns when it detects this. Understating dependence does not merely produce optimistic intervals; it can reverse the conclusion of a validation study.

![Confidence intervals for the mean paired difference $\mu$ from five of the six `pyTOST` engines, applied to one synthetic spatiotemporal dataset of $n = 192$ observations (24 sites $\times$ 8 times) containing both spatial and temporal dependence. The heteroskedastic engine is omitted because the dataset has constant variance. Dashed lines mark the equivalence margin $\Delta = 0.6$. Green intervals fall entirely inside $(-\Delta, \Delta)$ and give an equivalence decision; red intervals do not. All five engines estimate the same quantity and return $\hat{\mu} = 0.26$, so the differences shown are differences in estimated uncertainty alone. Reproduced by `scripts/make_paper_figure.py`.\label{fig:engines}](paper_figure.png)

## Sensitivity analyses

`pyTOST` includes optional checks on whether an equivalence decision is stable to modeling choices:

- **Heteroskedastic-robust CIs** (HC3 or cluster-robust), combined conservatively with the wild cluster bootstrap [@CameronGelbachMiller2008].
- **Robust-location equivalence** using the median or a trimmed mean, with dependence-aware bootstrap CIs [@EfronTibshirani1993].
- **Validation bootstrap summaries** for the mean, including cluster and spatial block variants.

## Validating the intervals

Because the package's argument rests on interval width, `pyTOST` checks the intervals themselves. The IID engine reproduces the closed-form Student-$t$ interval exactly, and coverage is estimated by simulation, drawing replicate datasets with a known mean and recording how often the interval contains it. Against a nominal level of 0.900, the IID engine covers 0.892 on independent data over 600 replicates of $n = 40$, and the cluster engine covers 0.907 on clustered data over 400 replicates of 30 clusters of 8 observations. Both lie within one Monte Carlo standard error of nominal, 0.012 and 0.015 respectively.

The same simulation measures the error the package exists to prevent. Applied to the clustered data, the IID engine covers 0.613 rather than 0.900, so roughly two in five of its nominally 90% intervals exclude the true mean. Under the TOST rule, intervals that are too narrow convert directly into unsupported equivalence decisions.

Coverage has been validated for the IID and cluster engines. The temporal, spatial, spatiotemporal, and heteroskedastic engines are covered by correctness and regression tests but not yet by coverage simulation, and their interval widths in \autoref{fig:engines} should be read as the behavior of the implemented estimators rather than as verified coverage.

## Structured synthetic data generation

The package includes generators for independent, clustered, AR(1) temporal, spatially correlated, and separable spatiotemporal paired samples, which support controlled equivalence scenarios and supply the data for the test suite and demonstration notebooks. The spatial generator uses a squared-exponential covariance while the spatial engines fit a Matérn model, so the engines are exercised under a covariance function they do not assume, and the tests do not reward a generator and an estimator for sharing the same parametric form.

# Research impact statement

`pyTOST` is used at the National Laboratory of the Rockies for validation work in which equivalence decisions must be defended under dependence. Two cases recur. The first is model-to-model validation of geospatial and simulation pipelines, where a revised pipeline must be shown to agree with the operational one and the outputs are correlated in space and time. The second is measurement-method validation, such as the solar access value assessments distributed with the package as a worked example, where paired differences are spatially structured and repeated across time. In both, the quantity in dispute is interval width, which is exactly what the choice of engine determines.

The package is built to be checked rather than taken on trust. The worked example in `docs/sav_worked_example.md` runs end to end from a script in the repository, the figure in this paper is regenerated by `scripts/make_paper_figure.py` from a fixed seed, and the coverage results reported above are produced by the test suite rather than quoted from a separate study. Releases are versioned, installable from PyPI, and archived on Zenodo, so a published analysis can cite the exact version that produced it.

# AI usage disclosure

Generative AI tools assisted with documentation refinement and editing. All technical claims, software descriptions, and methodological statements were checked against the source code and revised by the author.

# Acknowledgements

The author thanks collaborators at the National Laboratory of the Rockies for feedback on early versions of `pyTOST` and for the internal validation use cases that shaped its design.

# References


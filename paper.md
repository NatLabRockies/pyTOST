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

Researchers often need to show that two ways of measuring a quantity agree closely enough to be used interchangeably, for example that a new instrument reproduces a trusted one. Equivalence testing asks whether the difference between them is small enough to be unimportant in practice. An ordinary significance test cannot answer this, because failing to detect a difference is not evidence that none exists [@Lakens2017].

`pyTOST` is a Python package that performs equivalence testing on paired differences using the two one-sided tests (TOST) procedure [@Lakens2017; @Wellek2010]. It estimates the mean paired difference $\mu$ and declares equivalence at a tolerance $\Delta$, chosen by the analyst from subject-matter knowledge, when a confidence interval for $\mu$ lies entirely inside $(-\Delta,\Delta)$.

The package's distinguishing feature is that it adapts this interval to the way observations are related to one another. Measurements collected in groups, repeatedly over time, or across space are not independent, and treating them as though they were yields an interval that is too narrow, making equivalence easier to declare than the data warrant. `pyTOST` provides six inference engines covering independent, clustered, temporal, spatial, and spatiotemporal data as well as unequal variances. They share one interface and one result format, so an analyst states the form of dependence believed present and obtains an interval that accounts for it.

`pyTOST` builds on `NumPy` [@Harris2020], `pandas` [@McKinney2010], `SciPy` [@Virtanen2020], and `statsmodels` [@SeaboldPerktold2010], with spatial diagnostics from `PySAL` [@Rey2021] and optional R interoperability through `rpy2`.

# Statement of need

Equivalence testing arises whenever a new measurement system, algorithm, simulation code, or statistical model must agree with a reference method closely enough for practical use [@Wellek2010]. In applied validation work the decision usually turns less on the point estimate than on how its uncertainty is quantified, and that uncertainty depends on the dependence structure of the data. Ignoring clustering, serial correlation, or spatial dependence understates the standard error of $\hat\mu$ and narrows the interval, so equivalence appears easier to establish than the data support. The consequence is a false positive in the circumstance that matters most: a validation study concluding that two methods agree when the evidence does not establish it.

# State of the field

**Equivalence testing software.** In Python, `Pingouin` provides TOST for independent or paired samples [@Vallat2018], and `statsmodels` includes TOST for mean differences alongside robust covariance options [@SeaboldPerktold2010]. In R, `TOSTER` implements many equivalence tests for t-tests, correlations, and other settings [@TOSTERpkg]. These tools cover standard designs well, and for independent samples they are the appropriate choice. None, however, pairs the TOST decision rule with variance estimation for clustered, temporal, spatial, or spatiotemporal dependence.

**Dependence-aware inference components.** Cluster-robust inference is well established in econometrics and biostatistics [@CameronMiller2015], HAC estimators such as Newey--West are standard for serial dependence [@NeweyWest1987], and spatial dependence is commonly modeled through Matérn covariance families [@Stein1999; @GuttorpGneiting2006; @CressieWikle2011]. Python supports these ingredients individually, through `statsmodels` and `PySAL` [@Rey2021], but practitioners must still stitch them together manually if the goal is a TOST decision under dependence.

The variance estimators are therefore already available; what is missing is the decision layer above them. Each engine must expose an interval on a common scale, apply the containment rule against a margin, and remain comparable with the others when an analyst asks whether a decision depends on the assumed dependence structure. That shared contract is what `pyTOST` provides, and it fits naturally in neither a general statistics nor a spatial library.

# Software design

`pyTOST` is organized around inference "engines" sharing a common workflow and output schema. Each engine estimates $\hat\mu$ and constructs a $(1-2\alpha)$ confidence interval; equivalence is declared when it lies inside $(-\Delta,\Delta)$ [@Lakens2017]. The engines differ only in how that interval is built.

## Inference engines

- **IID engine.** Intercept-only OLS with a Student-$t$ CI ($df = n-1$).

- **Cluster engine.** Intercept-only OLS with cluster-robust (sandwich) variance and conservative $df = G-1$ for $G$ clusters [@CameronMiller2015].

- **Temporal engine.** Intercept-only autoregressive GLS, first order by default and user-settable. Two small-sample corrections are applied, without which the engine does not hold its nominal level at validation-study sample sizes: the autoregressive coefficient is bias-corrected for the known downward bias of its least-squares estimator in short series [@MarriottPope1954; @Kendall1954], and the interval uses a Student-$t$ critical value at Bartlett's effective sample size [@Bartlett1935]. Newey--West HAC variance remains available for dependence not well approximated by a low-order autoregression [@NeweyWest1987]. The engine models one series, so replicated time points are averaged first.

- **Spatial engine.** Gaussian process with Matérn covariance plus nugget [@Stein1999; @GuttorpGneiting2006]. Covariance parameters are estimated by profile maximum likelihood with smoothness $\nu$ selected over a grid; the mean is estimated by GLS with a profile likelihood-ratio CI [@Pawitan2001].

- **Heteroskedastic engine.** Intercept-only OLS with HC3 variance, or cluster-robust wild bootstrap when a cluster variable is supplied [@CameronGelbachMiller2008]. For an intercept-only mean the HC3 and IID intervals are asymptotically equivalent, since the usual variance estimator is already consistent under pure heteroskedasticity; the engine's value is the wild cluster bootstrap, which also handles within-cluster correlation.

- **Spatiotemporal engine.** For balanced panels, a separable AR(1)$\otimes$Matérn model by penalized ML, MAP-regularized to stabilize near-unidentifiable solutions, with a parametric-bootstrap CI [@EfronTibshirani1993; @CressieWikle2011]. For unbalanced panels, per-time-slice spatial models combined by inverse-variance weighting with a $t$-CI at $df = T-1$.

\autoref{fig:engines} shows what this choice does to a validation decision. In panel A, four engines analyze one synthetic panel with spatial and temporal dependence. The spatial and spatiotemporal engines estimate the plain sample mean rather than their default GLS estimate, so all four target the same estimand and return $\hat{\mu} = 0.26$, isolating the effect of the dependence assumption on width. The IID interval is 2.5 times narrower than the cluster-robust one and 6.2 times narrower than the spatiotemporal one, and at $\Delta = 0.6$ this decides the outcome: the IID and spatial engines declare equivalence, the cluster and spatiotemporal engines do not. Panels B and C repeat the point for serial dependence and for unequal cluster variances, where modeling the structure widens the interval 2.8-fold and about 1.4-fold and reverses the decision in both; in panel C the cluster-robust and wild cluster bootstrap intervals nearly coincide. Understating dependence does not merely produce optimistic intervals; it can reverse a validation study's conclusion.

![Confidence intervals for the mean paired difference $\mu$. **A**: four engines on one synthetic dataset of $n = 192$ (24 sites $\times$ 8 times) with spatial and temporal dependence. **B**: the temporal engine, which models a single series, on 120 time points with AR(1) correlation $\rho = 0.75$, against the IID engine. **C**: the heteroskedastic engine on 14 clusters of 10 whose standard deviations span 0.4 to 2.0, with the cluster-robust engine as a control. It is separate because heteroskedasticity-robust and IID intervals coincide for a mean under constant variance. Dashed lines mark $\Delta = 0.6$; green intervals fall inside $(-\Delta, \Delta)$, red do not. Generated by `scripts/make_paper_figure.py`.\label{fig:engines}](paper_figure.png)

## Sensitivity analyses

`pyTOST` includes optional checks on whether a decision is stable to modeling choices: heteroskedastic-robust intervals combined conservatively with the wild cluster bootstrap [@CameronGelbachMiller2008]; robust-location equivalence using the median or a trimmed mean [@EfronTibshirani1993]; and bootstrap summaries for the mean, including cluster and spatial block variants.

## Validating the intervals

Because the package's argument rests on interval width, `pyTOST` checks the intervals themselves by simulation, drawing replicate datasets with a known mean and recording how often the interval contains it. Against a nominal 0.900, the IID engine covers 0.892 on independent data over 600 replicates of $n = 40$, and the cluster engine covers 0.907 on clustered data over 400 replicates of 30 clusters of 8. Both lie within one Monte Carlo standard error of nominal, 0.012 and 0.015 respectively.

The same simulation measures the error the package exists to prevent: applied to the clustered data, the IID engine covers 0.613 rather than 0.900, so roughly two in five of its nominally 90% intervals exclude the true mean. Under the TOST rule, intervals that are too narrow convert directly into unsupported equivalence decisions.

Simulation also drove a design change. On AR(1) series of $n = 120$ over 400 replicates, a Newey--West HAC interval covered 0.858, 0.787, 0.715, and 0.485 as $\rho$ rose through 0, 0.5, 0.7, and 0.9, a shortfall a lag sweep attributed to finite-sample bias rather than lag selection. The autoregressive GLS estimator now used by default covers 0.875, 0.873, 0.860, and 0.858 under the same conditions, within 0.04 of nominal throughout, where the IID engine falls to 0.278. HAC remains available, and the comparison is kept in the test suite so the default is auditable.

The remaining engines are validated the same way, though the spatial study runs offline because each replicate fits a Gaussian process by profile likelihood. Over 300 replicates with the covariance function deliberately misspecified, the spatial engine covers 0.890 and the spatiotemporal 0.893, both within one Monte Carlo standard error of nominal. The heteroskedastic engine is likewise validated. The benchmark script and its results are in the repository.

These simulations draw on generators for independent, clustered, temporal, spatial, and spatiotemporal paired samples that also supply the test suite.

# Research impact statement

`pyTOST` is used at the National Laboratory of the Rockies for validation work in which equivalence decisions must be defended under dependence. Two cases recur: model-to-model validation of geospatial and simulation pipelines, where a revised pipeline must agree with the operational one and outputs are correlated in space and time; and measurement-method validation, such as the solar access value assessments distributed as a worked example. In both, the quantity in dispute is interval width, which is what the choice of engine determines.

The package is built to be checked rather than taken on trust. The worked example in `docs/sav_worked_example.md` runs end to end from a repository script, the figure is regenerated by `scripts/make_paper_figure.py` from a fixed seed, and the coverage results above come from the test suite rather than a separate study. Releases are versioned, installable from PyPI, and archived on Zenodo, so a published analysis can cite the exact version that produced it.

# AI usage disclosure

GitHub Copilot CLI v1.0.103, using the Claude Opus 5 model (`claude-opus-5`), was used in three places: reviewing source code, reviewing and editing this manuscript, and generating the script that produces \autoref{fig:engines}. Its contribution was limited to drafting, refactoring, and editorial revision of material specified by the author; it did not originate the statistical methods, the software design, or the research claims. All AI-assisted code was reviewed and tested by the author, every numerical value reported here was regenerated from the source code, and the author takes full responsibility for the software and the paper.

# Acknowledgements

The author thanks collaborators at the National Laboratory of the Rockies for feedback on early versions of `pyTOST` and for the internal validation use cases that shaped its design.

# References


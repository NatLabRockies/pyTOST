# Coverage benchmark: spatial and spatiotemporal engines

Generated 5 November 2026 by `scripts/coverage_benchmark.py`.

Regenerate with:

    pixi run -e test python scripts/coverage_benchmark.py

## Result

Nominal coverage is 0.900, the $(1 - 2\alpha)$ level the TOST decision rule uses at $\alpha = 0.05$. The true mean paired difference is 0.15 by construction.

| engine | replicates | coverage | Monte Carlo SE | mean CI width | s/fit |
|--------|-----------:|---------:|---------------:|--------------:|------:|
| spatial | 300 | 0.890 | 0.017 | 0.126 | 0.89 |
| spatiotemporal | 300 | 0.893 | 0.017 | 0.560 | 0.04 |

Estimator path exercised, taken from the `method` column:

- spatial: Matérn covariance + equal-weighted mean (Wald)
- spatiotemporal: Joint separable spatiotemporal ML (AR1 ⊗ Matérn) + equal-weighted mean (Wald)

## Design

- Replicates per engine: 300, seeds `0 .. 299`.
- Spatial: 10 clusters of 6 points (n = 60), length scale 0.5, field SD 0.8, nugget SD 0.2.
- Spatiotemporal: 18 sites x 8 times (n = 144), length scale 0.6, AR(1) rho = 0.5, spatial SD 0.8, observation SD 0.2.
- Both engines run in `equal_weighted` mode, so the estimand is the mean
  paired difference rather than a GLS-reweighted mean.
- The spatial generator uses a squared-exponential covariance while the engines
  fit a Matern model, so coverage is measured under a misspecified covariance
  function rather than a matched one.
- The spatial run fixes Matern smoothness at `nu_grid=(0.5,)` rather than
  profiling the default `(0.5, 1.5, 2.5)`. Profiling over the grid roughly
  triples the per-fit cost without changing the estimand, so the figure below
  validates the exponential-smoothness configuration rather than the full
  default search.

## Environment

- pyTOST 0.17.0
- Python 3.14.3 on Darwin arm64
- NumPy 2.5.1, pandas 3.0.3

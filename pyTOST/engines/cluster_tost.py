"""engines/cluster_tost.py
=======================

Cluster-aware TOST using **cluster-robust OLS only**.

Rationale
---------
For equivalence testing of a mean difference with clustered observations (e.g., multiple
samples within clusters), a random-intercept MixedLM fit can become numerically unstable
when the random-effect variance is near zero or the number of clusters is small, often
triggering warnings such as singular random-effects covariance or boundary solutions.
In these regimes, MixedLM-based standard errors and confidence intervals for the mean can
be unreliable.

This engine therefore uses an intercept-only OLS model with a **cluster-robust (sandwich)
covariance estimator**, paired with a conservative small-sample degrees-of-freedom choice
(df = G-1 clusters). This aligns well with cluster bootstrap sensitivity checks and is
a standard approach for inference with intra-cluster correlation.

Decision rule (CI-in-TOST)
--------------------------
For each equivalence margin Δ, we declare equivalence if and only if:
    CI_low(μ) > -Δ   and   CI_high(μ) < +Δ.

References
----------
- Cameron, A. C., & Miller, D. L. (2015). A practitioner's guide to cluster-robust inference.
  *Journal of Human Resources*, 50(2), 317–372.
- Bell, R. M., & McCaffrey, D. F. (2002). Bias reduction in standard errors for linear regression
  with multi-stage samples. *Survey Methodology*, 28(2), 169–181.
- Pustejovsky, J. E., & Tipton, E. (2018). Small-sample methods for cluster-robust variance estimation
  and hypothesis testing in fixed effects models. *Journal of Business & Economic Statistics*, 36(4), 672–683.
- Schuirmann, D. J. (1987). A comparison of the Two One-Sided Tests procedure and the power approach for
  assessing the equivalence of average bioavailability. *Journal of Pharmacokinetics and Biopharmaceutics*, 15, 657–680.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


class ClusterTOST:
    """Cluster-aware TOST via intercept-only OLS with cluster-robust SEs.

    Parameters
    ----------
    y : str
        Name of the response column containing paired differences.
    cluster : str
        Name of the column identifying clusters or grouped observational units.

    Notes
    -----
    This engine estimates the mean paired difference using an intercept-only
    ordinary least squares model and computes a cluster-robust sandwich
    covariance estimate for inference. Equivalence is assessed by checking
    whether the confidence interval for the mean difference lies entirely
    within each user-specified equivalence margin.
    """

    def __init__(self, y: str, cluster: str):
        self.y = y
        self.cluster = cluster

    def _cluster_robust_ci(self, df: pd.DataFrame, alpha: float) -> Tuple[float, Tuple[float, float], int, str]:
        """Compute a cluster-robust confidence interval for the mean difference.

        Parameters
        ----------
        df : pandas.DataFrame
            Input analysis table containing the paired-difference column and
            the cluster identifier column.
        alpha : float
            One-sided Type I error rate used to form the two-sided
            ``(1 - 2 * alpha)`` confidence interval.

        Returns
        -------
        mu_hat : float
            Estimated mean paired difference.
        ci : tuple of float
            Two-element tuple ``(ci_low, ci_high)`` giving the confidence
            interval bounds for the mean paired difference.
        dfree : int
            Degrees of freedom used for the critical value calculation.
        label : str
            Human-readable description of the estimation method.
        """
        if self.y not in df.columns:
            raise ValueError(f"ClusterTOST requires y column {self.y!r}. Available: {list(df.columns)}")
        if self.cluster not in df.columns:
            raise ValueError(f"ClusterTOST requires cluster column {self.cluster!r}. Available: {list(df.columns)}")

        yv = df[self.y].to_numpy(float)
        X = np.ones((len(df), 1), dtype=float)
        groups = df[self.cluster].to_numpy()

        fit = sm.OLS(yv, X).fit(cov_type="cluster", cov_kwds={"groups": groups})

        mu_hat = float(fit.params[0])
        se = float(fit.bse[0])

        G = int(pd.Series(groups).nunique())
        dfree = max(G - 1, 1)  # conservative, commonly used with cluster-robust inference
        tcrit = stats.t.ppf(1 - alpha, df=dfree)

        ci_low = mu_hat - tcrit * se
        ci_high = mu_hat + tcrit * se
        return mu_hat, (float(ci_low), float(ci_high)), dfree, f"OLS + cluster-robust SE (df={dfree})"

    def fit(self, df: pd.DataFrame, alpha: float, margins: List[float]) -> pd.DataFrame:
        """Evaluate TOST equivalence decisions under clustered dependence.

        Parameters
        ----------
        df : pandas.DataFrame
            Input analysis table containing paired differences and cluster
            identifiers.
        alpha : float
            One-sided Type I error rate used to form the two-sided
            ``(1 - 2 * alpha)`` confidence interval.
        margins : list of float
            Equivalence margins to evaluate. A separate row is returned for
            each margin.

        Returns
        -------
        pandas.DataFrame
            Data frame with one row per equivalence margin and the following
            columns:

            ``delta``
                Equivalence margin.
            ``mu_hat``
                Estimated mean paired difference.
            ``ci_low``, ``ci_high``
                Confidence interval bounds for the mean paired difference.
            ``equivalent``
                Indicator for whether the confidence interval lies entirely
                within ``(-delta, delta)``.
            ``method``
                Human-readable description of the estimation method.
            ``df``
                Degrees of freedom used for the critical value.
        """
        mu, ci, dfree, label = self._cluster_robust_ci(df, alpha)

        out = []
        for d in margins:
            d = float(d)
            out.append(
                dict(
                    delta=d,
                    mu_hat=mu,
                    ci_low=ci[0],
                    ci_high=ci[1],
                    equivalent=(ci[0] > -d and ci[1] < d),
                    method=label,
                    df=int(dfree),
                )
            )
        return pd.DataFrame(out)

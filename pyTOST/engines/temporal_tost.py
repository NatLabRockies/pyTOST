"""Temporal-aware TOST engines for ordered observations.

This module provides a temporal TOST implementation for paired-difference data
with serial dependence. The primary workflow uses an intercept-only mean model
with a heteroskedasticity-and-autocorrelation-consistent (HAC) covariance
estimator based on the Newey--West approach. A GLSAR(1) estimator is also
implemented as a model-based sensitivity option.

Notes
-----
The main :class:`TemporalTOST` workflow reports HAC-based intervals because they
are robust to serial correlation under mild regularity conditions. The
GLSAR(1) method is available as a model-based alternative when an AR(1)
structure is a reasonable approximation.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from typing import List

class TemporalTOST:
    """TOST inference for temporally ordered paired differences.

    Parameters
    ----------
    y : str
        Name of the column containing paired differences.
    time : str
        Name of the column defining temporal order.
    hac_lags : int, default=4
        Maximum lag used in the Newey--West HAC covariance estimator.

    Notes
    -----
    The primary estimator is an intercept-only OLS model with HAC standard
    errors. Confidence intervals are constructed using a normal critical value
    because the HAC estimator is treated asymptotically.
    """
    def __init__(self, y: str, time: str, hac_lags: int = 4):
        self.y = y
        self.time = time
        self.hac_lags = hac_lags

    def _hac(self, df, alpha):
        """Fit an intercept-only mean model with HAC standard errors.

        Parameters
        ----------
        df : pandas.DataFrame
            Input data containing the paired-difference column and temporal
            index column.
        alpha : float
            One-sided significance level used to construct the
            ``(1 - 2 * alpha)`` confidence interval.

        Returns
        -------
        tuple
            Four-element tuple containing the mean estimate, confidence
            interval, effective degrees of freedom placeholder, and method
            label.

        Notes
        -----
        The returned degrees-of-freedom value is ``np.inf`` because the HAC
        interval is treated as asymptotic and uses a normal critical value.
        """
        df2 = df.sort_values(self.time)
        X = np.ones((len(df2),1))
        fit = sm.OLS(df2[self.y].to_numpy(float), X).fit(cov_type="HAC", cov_kwds={"maxlags": self.hac_lags})
        mu = float(fit.params[0]); se = float(fit.bse[0])
        zcrit = stats.norm.ppf(1-alpha)  # HAC is asymptotic
        return mu, (mu - zcrit*se, mu + zcrit*se), np.inf, f"IID mean with Newey–West HAC (lags={self.hac_lags})"

    def _glsar1(self, df, alpha):
        """Fit a GLSAR(1) sensitivity model for the mean paired difference.

        Parameters
        ----------
        df : pandas.DataFrame
            Input data containing the paired-difference column and temporal
            index column.
        alpha : float
            One-sided significance level used to construct the
            ``(1 - 2 * alpha)`` confidence interval.

        Returns
        -------
        tuple
            Four-element tuple containing the mean estimate, confidence
            interval, finite-sample degrees of freedom, and method label.

        Notes
        -----
        This method is model-based and assumes an AR(1) correlation structure.
        It is useful as a sensitivity analysis rather than the default robust
        inference path.
        """
        df2 = df.sort_values(self.time)
        X = np.ones((len(df2),1))
        model = sm.GLSAR(df2[self.y].to_numpy(float), X, rho=1)
        res = model.iterative_fit(maxiter=10)
        mu = float(res.params[0]); se = float(res.bse[0])
        dfree = max(len(df2)-2, 1)
        tcrit = stats.t.ppf(1-alpha, dfree)
        return mu, (mu - tcrit*se, mu + tcrit*se), dfree, "GLSAR(1)"

    def fit(self, df: pd.DataFrame, alpha: float, margins: List[float]) -> pd.DataFrame:
        """Run temporal TOST inference for one or more equivalence margins.

        Parameters
        ----------
        df : pandas.DataFrame
            Input data containing paired differences and a temporal ordering
            variable.
        alpha : float
            One-sided significance level used to construct the
            ``(1 - 2 * alpha)`` confidence interval.
        margins : list of float
            Equivalence margins to evaluate.

        Returns
        -------
        pandas.DataFrame
            Result table with one row per margin. The output includes the
            estimated mean paired difference, confidence interval bounds,
            equivalence decision, method label, and degrees-of-freedom field.

        Notes
        -----
        The current implementation reports results from the HAC-based path.
        The GLSAR(1) method is available separately as a model-based
        sensitivity estimator.
        """
        mu, ci, dfree, label = self._hac(df, alpha)
        out = []
        for d in margins:
            out.append(dict(delta=float(d), mu_hat=mu,
                            ci_low=ci[0], ci_high=ci[1],
                            equivalent=(ci[0] > -d and ci[1] < d),
                            method=label, df=dfree))
        return pd.DataFrame(out)

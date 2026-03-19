"""
engines/iid_tost.py
===================

IID equivalence testing for paired differences using an intercept-only OLS
model and a t-based confidence interval for the mean difference.

Notes
-----
This engine assumes observations are independent and identically distributed.
It is the baseline TOST engine for settings where dependence across rows is not
substantively important.

References
----------
- Schuirmann (1987) Journal of Pharmacokinetics and Biopharmaceutics.
- Westlake (1976) Biometrics.
- Wellek (2010) CRC Press.
- Lakens (2017) Social Psychological and Personality Science.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from typing import List

class IIDTOST:
    """
    Perform IID equivalence testing for the mean paired difference.

    Parameters
    ----------
    y : str
        Name of the column containing paired differences.

    Attributes
    ----------
    y : str
        Response column used for inference.
    mu_hat_ : float or None
        Placeholder for the fitted mean estimate.
    se_ : float or None
        Placeholder for the fitted standard error.
    df_ : float or None
        Placeholder for the fitted degrees of freedom.
    """
    def __init__(self, y: str):
        self.y = y
        self.mu_hat_ = None
        self.se_ = None
        self.df_ = None

    def fit(self, df: pd.DataFrame, alpha: float, margins: List[float]) -> pd.DataFrame:
        """
        Fit the IID TOST model and evaluate equivalence margins.

        Parameters
        ----------
        df : pandas.DataFrame
            Input analysis table containing the paired-difference column.
        alpha : float
            One-sided significance level used to construct the
            ``(1 - 2 * alpha)`` confidence interval.
        margins : list of float
            Equivalence margins to evaluate.

        Returns
        -------
        pandas.DataFrame
            Result table with one row per margin. The returned table includes
            the tested margin, estimated mean difference, confidence interval
            bounds, the equivalence decision, and method metadata.

        Notes
        -----
        The model is an intercept-only ordinary least squares fit, so the point
        estimate is the sample mean of the paired differences and the interval
        is based on the Student-t distribution.
        """
        X = np.ones((len(df),1))
        fit = sm.OLS(df[self.y].to_numpy(float), X).fit()
        mu = float(fit.params[0])
        se = float(fit.bse[0])
        dfree = max(len(df)-1, 1)
        tcrit = stats.t.ppf(1-alpha, dfree)
        ci = (mu - tcrit*se, mu + tcrit*se)
        out = []
        for d in margins:
            out.append(dict(delta=float(d), mu_hat=mu,
                            ci_low=ci[0], ci_high=ci[1],
                            equivalent=(ci[0] > -d and ci[1] < d),
                            method="IID OLS (t-CI)", df=dfree))
        return pd.DataFrame(out)

"""
engines/temporal_tost.py
========================
Temporal-aware TOST for serial correlation when observations are ordered in time.

Implements
---------
(a) IID mean with HAC (Newey–West) SE (robust, asymptotic)
(b) GLSAR(1) as a model-based sensitivity (the workflow can run both)

References
----------
- Newey & West (1987) Econometrica 55:703–708.
- Box & Jenkins (1970) Time Series Analysis.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from typing import List

class TemporalTOST:
    def __init__(self, y: str, time: str, hac_lags: int = 4):
        self.y = y
        self.time = time
        self.hac_lags = hac_lags

    def _hac(self, df, alpha):
        df2 = df.sort_values(self.time)
        X = np.ones((len(df2),1))
        fit = sm.OLS(df2[self.y].to_numpy(float), X).fit(cov_type="HAC", cov_kwds={"maxlags": self.hac_lags})
        mu = float(fit.params[0]); se = float(fit.bse[0])
        zcrit = stats.norm.ppf(1-alpha)  # HAC is asymptotic
        return mu, (mu - zcrit*se, mu + zcrit*se), np.inf, f"IID mean with Newey–West HAC (lags={self.hac_lags})"

    def _glsar1(self, df, alpha):
        df2 = df.sort_values(self.time)
        X = np.ones((len(df2),1))
        model = sm.GLSAR(df2[self.y].to_numpy(float), X, rho=1)
        res = model.iterative_fit(maxiter=10)
        mu = float(res.params[0]); se = float(res.bse[0])
        dfree = max(len(df2)-2, 1)
        tcrit = stats.t.ppf(1-alpha, dfree)
        return mu, (mu - tcrit*se, mu + tcrit*se), dfree, "GLSAR(1)"

    def fit(self, df: pd.DataFrame, alpha: float, margins: List[float]) -> pd.DataFrame:
        mu, ci, dfree, label = self._hac(df, alpha)
        out = []
        for d in margins:
            out.append(dict(delta=float(d), mu_hat=mu,
                            ci_low=ci[0], ci_high=ci[1],
                            equivalent=(ci[0] > -d and ci[1] < d),
                            method=label, df=dfree))
        return pd.DataFrame(out)


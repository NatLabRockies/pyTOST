"""
engines/iid_tost.py
===================
Classic IID TOST using intercept-only OLS and t-based CI.

References
----------
- Schuirmann (1987) J. Pharmacokinetics & Biopharmaceutics.
- Westlake (1976) Biometrics; Wellek (2010) CRC; Lakens (2017) SPPS.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from typing import List

class IIDTOST:
    """
    IID equivalence testing for the mean difference y.

    Parameters
    ----------
    y : str
        Response column (e.g., SAV difference).
    """
    def __init__(self, y: str):
        self.y = y
        self.mu_hat_ = None
        self.se_ = None
        self.df_ = None

    def fit(self, df: pd.DataFrame, alpha: float, margins: List[float]) -> pd.DataFrame:
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


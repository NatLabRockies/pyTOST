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
from typing import List, Union

LagSpec = Union[int, str]


def auto_hac_lags(n: int) -> int:
    """Data-driven Newey--West truncation lag.

    Uses the common plug-in rule ``floor(4 * (n / 100) ** (2 / 9))`` (Newey & West,
    1994), which grows slowly with the sample size. Returns 0 for empty input.

    Parameters
    ----------
    n : int
        Number of time-ordered observations.

    Returns
    -------
    int
        Non-negative truncation lag.
    """
    if n <= 0:
        return 0
    return int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def _validate_lag_spec(hac_lags: LagSpec) -> LagSpec:
    if isinstance(hac_lags, str):
        if hac_lags.strip().lower() != "auto":
            raise ValueError(
                f"max_lag={hac_lags!r} is not valid; use an integer >= 0 or 'auto'."
            )
        return "auto"

    if isinstance(hac_lags, bool) or not isinstance(hac_lags, (int, np.integer)):
        raise ValueError(
            f"max_lag={hac_lags!r} is not valid; use an integer >= 0 or 'auto'."
        )
    if hac_lags < 0:
        raise ValueError(f"max_lag={hac_lags!r} is not valid; hac_lags must be >= 0.")
    return int(hac_lags)


class TemporalTOST:
    def __init__(
        self,
        y: str,
        time: str,
        hac_lags: LagSpec = "auto",
        require_unique_times: bool = False,
    ):
        """
        Parameters
        ----------
        y : str
            Response column (paired difference).
        time : str
            Time-ordering column.
        hac_lags : int or "auto", default "auto"
            Newey--West truncation lag. ``"auto"`` selects the lag from the sample size
            using :func:`auto_hac_lags`, the Newey--West (1994) plug-in rule
            ``floor(4 * (n / 100) ** (2 / 9))``, so the lag grows with the series length.
            An integer fixes the lag instead; pass ``hac_lags=4`` to reproduce results
            from pyTOST versions before 0.17.0, which used a fixed lag of 4 regardless
            of sample size.
        require_unique_times : bool
            When True, raise ValueError if any time value appears more than once.
        """
        self.y = y
        self.time = time
        self.hac_lags = _validate_lag_spec(hac_lags)
        # When True, raise ValueError if any time value appears more than once in the
        # data passed to fit().  On the publication path (one observation per ordered
        # time point) set this to True to prevent accidental use of raw, un-aggregated data.
        self.require_unique_times = require_unique_times

    def _resolve_lags(self, n: int) -> tuple[int, str]:
        """Return the truncation lag to use and how it was chosen."""
        if self.hac_lags == "auto":
            return auto_hac_lags(n), "auto"
        return int(self.hac_lags), "fixed"

    def _hac(self, df, alpha):
        df2 = df.sort_values(self.time)
        if self.require_unique_times:
            dup = df2[self.time].duplicated()
            if dup.any():
                raise ValueError(
                    f"TemporalTOST: duplicate time values detected (require_unique_times=True). "
                    f"Duplicated times: {df2.loc[dup, self.time].unique().tolist()}.  "
                    "Aggregate to one observation per time point before fitting."
                )
        X = np.ones((len(df2),1))
        lags, how = self._resolve_lags(len(df2))
        fit = sm.OLS(df2[self.y].to_numpy(float), X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
        mu = float(fit.params[0]); se = float(fit.bse[0])
        zcrit = stats.norm.ppf(1-alpha)  # HAC is asymptotic
        suffix = ", auto" if how == "auto" else ""
        label = f"IID mean with Newey–West HAC (lags={lags}{suffix})"
        return mu, (mu - zcrit*se, mu + zcrit*se), np.inf, label

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


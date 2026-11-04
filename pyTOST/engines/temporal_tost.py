"""
engines/temporal_tost.py
========================
Temporal-aware TOST for serial correlation when observations are ordered in time.

Implements
---------
(a) ``method="glsar"`` (default) --- AR(p) GLS for the mean with two small-sample
    corrections: a Marriott--Pope/Kendall bias correction for the estimated AR
    coefficients, and an effective-sample-size t critical value.
(b) ``method="hac"`` --- intercept-only OLS with a Newey--West HAC standard error and
    a standard normal critical value. Nonparametric and asymptotically valid for
    general stationary dependence, but materially anticonservative in the sample sizes
    typical of validation work.

Why GLSAR is the default
------------------------
Monte Carlo coverage against a nominal 0.900 (see ``tests/test_ci_coverage.py`` and
``docs/review_register.md``, RR-003) shows HAC undercovering badly as persistence
rises, while the corrected GLSAR path stays far closer to nominal and, where it errs
on short series, errs conservatively. Under AR(1) with rho = 0.7 and n = 120, HAC
covers 0.72 and corrected GLSAR covers 0.89. GLSAR also dominated HAC on every
misspecified process tested, including AR(2), MA(1) and ARMA(1,1).

References
----------
- Newey & West (1987) Econometrica 55:703-708.
- Marriott & Pope (1954) Biometrika 41:390-402.
- Kendall (1954) Biometrika 41:403-404.
- Bartlett (1935) Supplement to the Journal of the Royal Statistical Society 2:225-232.
- Box & Jenkins (1970) Time Series Analysis.
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from typing import List, Union

LagSpec = Union[int, str]

METHODS = ("glsar", "hac")


def _effective_sample_size(n: int, persistence: float) -> float:
    """Bartlett effective sample size for a mean under AR persistence.

    For an AR(1) process with coefficient ``rho`` the variance of the sample mean is
    inflated by roughly ``(1 + rho) / (1 - rho)``, so the series carries the
    information of about ``n * (1 - rho) / (1 + rho)`` independent observations. For
    AR(p) the sum of the coefficients plays the role of ``rho``, since it determines
    the long-run variance through ``1 / (1 - sum(phi)) ** 2``.
    """
    return n * (1.0 - persistence) / (1.0 + persistence)


def _bias_corrected_persistence(persistence: float, n: int) -> float:
    """Marriott--Pope/Kendall small-sample correction for AR persistence.

    Least-squares AR coefficient estimates are biased toward zero by order ``1 / n``,
    which makes a series look less persistent, and therefore its mean look more
    precisely estimated, than it is. The classical correction adds
    ``(1 + 3 * rho) / n``.
    """
    return float(min(persistence + (1.0 + 3.0 * persistence) / max(n, 1), 0.995))


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
        method: str = "glsar",
        ar_order: int = 1,
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
            When False (default) and times are tied, the data are aggregated to one
            observation per time point before fitting, which is what makes a temporal
            model applicable at all.
        method : {"glsar", "hac"}, default "glsar"
            ``"glsar"`` fits an AR(``ar_order``) GLS mean with small-sample
            corrections. ``"hac"`` uses the Newey--West HAC standard error with a
            normal critical value; it is nonparametric but anticonservative at the
            sample sizes typical of validation work. Versions before 0.18.0 always
            used ``"hac"``.
        ar_order : int, default 1
            AR order for ``method="glsar"``. Order 1 is the best default across the
            processes tested; order 2 helps when the series has genuine second-order
            structure but is worse under MA-type dependence.
        """
        self.y = y
        self.time = time
        self.hac_lags = _validate_lag_spec(hac_lags)
        method = str(method).strip().lower()
        if method not in METHODS:
            raise ValueError(f"method={method!r} is not valid; use one of {METHODS}.")
        self.method = method
        if isinstance(ar_order, bool) or not isinstance(ar_order, (int, np.integer)):
            raise ValueError(f"ar_order={ar_order!r} is not valid; use an integer >= 1.")
        if ar_order < 1:
            raise ValueError(f"ar_order={ar_order!r} is not valid; ar_order must be >= 1.")
        self.ar_order = int(ar_order)
        # When True, raise ValueError if any time value appears more than once in the
        # data passed to fit().  On the publication path (one observation per ordered
        # time point) set this to True to prevent accidental use of raw, un-aggregated data.
        self.require_unique_times = require_unique_times

    def _resolve_lags(self, n: int) -> tuple[int, str]:
        """Return the truncation lag to use and how it was chosen."""
        if self.hac_lags == "auto":
            return auto_hac_lags(n), "auto"
        return int(self.hac_lags), "fixed"

    def _ordered(self, df: pd.DataFrame) -> pd.DataFrame:
        """Order observations for sequential (HAC/AR) estimation.

        Rows are ordered by time. Rows sharing a time value are further ordered by
        response value, which makes the resulting sequence independent of the order in
        which the caller supplied the rows: two rows tied on both time and response are
        interchangeable and leave the residual sequence unchanged. Without this,
        identical data in a different row order produced different autocovariances and
        therefore a different confidence interval (see docs/review_register.md, RR-001).
        """
        return df.sort_values([self.time, self.y], kind="mergesort")

    def _prepare(self, df: pd.DataFrame) -> tuple[np.ndarray, str]:
        """Return the time-ordered series to model, collapsing replicated time points.

        A temporal model describes how one observation relates to the next in time. If
        several observations share a time value the data are a panel, not a series, and
        treating them as consecutive misrepresents the autocorrelation entirely (see
        ``docs/review_register.md``, RR-001). Averaging within each time point turns the
        panel into the series the model actually assumes. For a balanced panel the mean
        of the per-time means equals the overall mean, so the estimand is unchanged; for
        an unbalanced panel it weights each time point equally, which is a different
        quantity, so that case is warned about explicitly.
        """
        df2 = self._ordered(df)
        dup = df2[self.time].duplicated()

        if not dup.any():
            return df2[self.y].to_numpy(float), ""

        tied = df2.loc[dup, self.time].unique().tolist()
        shown = ", ".join(repr(t) for t in tied[:5])
        if len(tied) > 5:
            shown += f", ... ({len(tied)} tied time values in total)"

        if self.require_unique_times:
            raise ValueError(
                "TemporalTOST: duplicate time values detected "
                f"(require_unique_times=True). Duplicated times: {shown}. "
                "Aggregate to one observation per time point before fitting."
            )

        grouped = df2.groupby(self.time, sort=True)[self.y]
        counts = grouped.size()
        series = grouped.mean().to_numpy(float)

        if counts.nunique() > 1:
            warnings.warn(
                "TemporalTOST: time points carry unequal numbers of observations "
                f"(between {int(counts.min())} and {int(counts.max())}). The series was "
                "averaged within each time point, which weights every time point "
                "equally, so the estimate is the mean of per-time means rather than the "
                "mean of all observations. Use the cluster or spatiotemporal engine to "
                "model cross-sectional replication directly.",
                UserWarning,
                stacklevel=4,
            )

        return series, f", {len(df2)} obs averaged to {len(series)} time points"

    def _hac(self, values, alpha, note):
        n = len(values)
        X = np.ones((n, 1))
        lags, how = self._resolve_lags(n)
        fit = sm.OLS(values, X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
        mu = float(fit.params[0]); se = float(fit.bse[0])
        zcrit = stats.norm.ppf(1 - alpha)  # HAC is asymptotic
        suffix = ", auto" if how == "auto" else ""
        label = f"IID mean with Newey–West HAC (lags={lags}{suffix}{note})"
        return mu, (mu - zcrit * se, mu + zcrit * se), np.inf, label

    def _glsar(self, values, alpha, note):
        """AR(p) GLS mean with small-sample persistence and degrees-of-freedom corrections."""
        n = len(values)
        order = min(self.ar_order, max(n - 2, 1))
        X = np.ones((n, 1))

        model = sm.GLSAR(values, X, rho=order)
        res = model.iterative_fit(maxiter=10)
        mu = float(res.params[0]); se = float(res.bse[0])

        phi = np.atleast_1d(np.asarray(model.rho, dtype=float))
        persistence = float(np.clip(np.nansum(phi), 0.0, 0.99))
        corrected = _bias_corrected_persistence(persistence, n)

        # Widen the interval by the ratio of long-run variance inflation implied by the
        # corrected persistence to that implied by the raw estimate.
        raw_inflation = (1.0 + persistence) / (1.0 - persistence)
        corrected_inflation = (1.0 + corrected) / (1.0 - corrected)
        se *= float(np.sqrt(corrected_inflation / max(raw_inflation, 1e-12)))

        dfree = float(max(_effective_sample_size(n, corrected) - 1.0, 1.0))
        tcrit = stats.t.ppf(1 - alpha, dfree)

        label = (
            f"AR({order}) GLS mean, bias-corrected persistence "
            f"{corrected:.2f}, effective df {dfree:.1f}{note}"
        )
        return mu, (mu - tcrit * se, mu + tcrit * se), dfree, label

    def fit(self, df: pd.DataFrame, alpha: float, margins: List[float]) -> pd.DataFrame:
        values, note = self._prepare(df)
        estimator = self._glsar if self.method == "glsar" else self._hac
        mu, ci, dfree, label = estimator(values, alpha, note)
        out = []
        for d in margins:
            out.append(dict(delta=float(d), mu_hat=mu,
                            ci_low=ci[0], ci_high=ci[1],
                            equivalent=(ci[0] > -d and ci[1] < d),
                            method=label, df=dfree))
        return pd.DataFrame(out)


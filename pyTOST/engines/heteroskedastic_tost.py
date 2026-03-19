"""
engines/heteroskedastic_tost.py
===============================
Heteroskedasticity-aware TOST engine with:
  - HC3 robust SE (no clusters)
  - Cluster-robust (CR2-ish) SE when clusters provided
  - Wild *cluster* bootstrap CIs (Rademacher multipliers) for validation/publication

Why
---
When variance is not constant across observations (heteroskedasticity) or clusters,
t-based IID CIs can be misleading. HC and cluster-robust SEs correct first-order
effects; wild cluster bootstrap further improves small-sample accuracy.

References
----------
- MacKinnon & White (1985) J Econometrics (HC SEs).
- Bell & McCaffrey (2002) Survey Methodology (CR2 small-sample correction idea).
- Cameron, Gelbach & Miller (2008) Review of Economics and Statistics (wild bootstrap).
- Pustejovsky & Tipton (2018) J. Bus. & Econ. Stats. (small-sample CR inference).

API
---
HeteroskedasticTOST(y, cluster=None, wild_B=999, wild_type="rademacher")
  .fit(df, alpha, margins) -> DataFrame
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from typing import List, Optional, Tuple


def _percentile_ci(arr: np.ndarray, alpha: float) -> Tuple[float, float]:
    """Compute a percentile confidence interval from bootstrap draws.

    Parameters
    ----------
    arr : numpy.ndarray
        One-dimensional array of bootstrap replicates for the target mean.
    alpha : float
        One-sided TOST significance level. The returned interval has nominal
        coverage ``1 - 2 * alpha``.

    Returns
    -------
    tuple of float
        Lower and upper percentile interval bounds.
    """
    # For equivalence at α one-sided, we use a 100*(1-2α)% CI for μ.
    lo = np.quantile(arr, 2 * alpha / 2.0)  # α lower for two tails
    hi = np.quantile(arr, 1 - 2 * alpha / 2.0)
    return float(lo), float(hi)


class HeteroskedasticTOST:
    """TOST engine with heteroskedasticity-robust inference.

    Parameters
    ----------
    y : str
        Column containing paired differences.
    cluster : str or None, optional
        Column identifying clusters. If provided, clustered inference and wild
        cluster bootstrap are used.
    wild_B : int, default=999
        Number of wild bootstrap replicates.
    wild_type : {"rademacher"}, default="rademacher"
        Multiplier distribution used for the wild bootstrap.
    seed : int, default=42
        Random-number seed for bootstrap resampling.
    """

    def __init__(self, y: str, cluster: Optional[str] = None, wild_B: int = 999, wild_type: str = "rademacher", seed: int = 42):
        """Initialize the heteroskedasticity-aware TOST engine.

        Parameters
        ----------
        y : str
            Column containing paired differences.
        cluster : str or None, optional
            Column identifying clusters. If provided, clustered inference and
            wild cluster bootstrap are used.
        wild_B : int, default=999
            Number of wild bootstrap replicates.
        wild_type : {"rademacher"}, default="rademacher"
            Multiplier distribution used for the wild bootstrap.
        seed : int, default=42
            Random-number seed for bootstrap resampling.
        """
        self.y = y
        self.cluster = cluster
        self.wild_B = int(wild_B)
        self.wild_type = wild_type
        self.seed = int(seed)

    # --- core estimators ---
    def _hc3(self, df: pd.DataFrame, alpha: float):
        """Fit an intercept-only model with HC3 robust standard errors.

        Parameters
        ----------
        df : pandas.DataFrame
            Input analysis table.
        alpha : float
            One-sided TOST significance level.

        Returns
        -------
        tuple
            Mean estimate, two-sided confidence interval with nominal coverage
            ``1 - 2 * alpha``, and a method label.
        """
        X = np.ones((len(df), 1))
        fit = sm.OLS(df[self.y].to_numpy(float), X).fit(cov_type="HC3")
        mu = float(fit.params[0])
        se = float(fit.bse[0])
        # conservative: large-sample normal or Student with N-1
        tcrit = stats.t.ppf(1 - alpha, df=len(df) - 1)
        return mu, (mu - tcrit * se, mu + tcrit * se), "OLS (HC3)"

    def _cluster_robust(self, df: pd.DataFrame, alpha: float):
        """Fit an intercept-only model with cluster-robust covariance.

        Parameters
        ----------
        df : pandas.DataFrame
            Input analysis table.
        alpha : float
            One-sided TOST significance level.

        Returns
        -------
        tuple
            Mean estimate, two-sided confidence interval with nominal coverage
            ``1 - 2 * alpha``, and a method label.
        """
        X = np.ones((len(df), 1))
        fit = sm.OLS(df[self.y].to_numpy(float), X).fit(
            cov_type="cluster",
            cov_kwds={"groups": df[self.cluster].to_numpy()},
        )
        mu = float(fit.params[0])
        se = float(fit.bse[0])
        dfree = max(df[self.cluster].nunique() - 1, 1)
        tcrit = stats.t.ppf(1 - alpha, df= dfree)
        return mu, (mu - tcrit * se, mu + tcrit * se), "Cluster-robust OLS (CR)"

    # --- wild cluster bootstrap ---
    def _wild_cluster_bootstrap_ci(self, df: pd.DataFrame, alpha: float) -> Tuple[float, float]:
        """
        Percentile CI for μ using wild *cluster* bootstrap with Rademacher multipliers.
        For an intercept-only model, μ̂ is the sample mean of bootstrap pseudo-responses.

        Steps (Cameron et al., 2008):
          1) Fit OLS to get residuals u_i and cluster means \bar u_g.
          2) Generate multiplier v_g ∈ {−1, +1}, iid across clusters.
          3) Form y*_i = μ̂ + v_g * (u_i - \bar u_g) for i in cluster g.
          4) Recompute μ̂* and store.
        """
        rng = np.random.default_rng(self.seed)
        y = df[self.y].to_numpy(float)
        X = np.ones((len(df), 1))
        ols = sm.OLS(y, X).fit()
        mu_hat = float(ols.params[0])
        resid = y - mu_hat

        # center residuals within cluster to respect cluster structure
        g = df[self.cluster].to_numpy()
        uniq = np.unique(g)
        r_centered = resid.copy()
        for cl in uniq:
            idx = (g == cl)
            r_centered[idx] = resid[idx] - resid[idx].mean()

        boots = []
        for _ in range(self.wild_B):
            # rademacher multipliers at cluster level
            m = rng.choice([-1.0, 1.0], size=len(uniq))
            y_star = y.copy()
            for j, cl in enumerate(uniq):
                idx = (g == cl)
                y_star[idx] = mu_hat + m[j] * r_centered[idx]
            boots.append(float(y_star.mean()))
        arr = np.asarray(boots, float)
        return _percentile_ci(arr, alpha)

    # --- fit ---
    def fit(self, df: pd.DataFrame, alpha: float, margins: List[float]) -> pd.DataFrame:
        """Run heteroskedasticity-aware TOST inference.

        Parameters
        ----------
        df : pandas.DataFrame
            Input analysis table.
        alpha : float
            One-sided TOST significance level.
        margins : list of float
            Equivalence margins to evaluate.

        Returns
        -------
        pandas.DataFrame
            Result table with one row per equivalence margin and columns such as
            ``delta``, ``mu_hat``, ``ci_low``, ``ci_high``, ``equivalent``, and
            ``method``.
        """
        have_cluster = self.cluster is not None and self.cluster in df.columns

        if have_cluster:
            mu, ci_cr, label = self._cluster_robust(df, alpha)
            # validate/optionally replace with wild cluster bootstrap if wider
            try:
                ci_boot = self._wild_cluster_bootstrap_ci(df, alpha)
                ci_low = min(ci_cr[0], ci_boot[0])
                ci_high = max(ci_cr[1], ci_boot[1])
                ci = (ci_low, ci_high)
                label = label + " + Wild Cluster Bootstrap (conservative)"
            except Exception:
                ci = ci_cr
        else:
            mu, ci_hc3, label = self._hc3(df, alpha)
            ci = ci_hc3

        rows = []
        for d in margins:
            rows.append(
                dict(
                    delta=float(d),
                    mu_hat=float(mu),
                    ci_low=float(ci[0]),
                    ci_high=float(ci[1]),
                    equivalent=(ci[0] > -d and ci[1] < d),
                    method=label,
                )
            )
        return pd.DataFrame(rows)


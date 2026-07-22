"""
engines/heteroskedastic_tost.py
===============================
Heteroskedasticity-aware TOST engine with:
  - HC3 robust SE (no clusters)
  - Cluster-robust (CR2) SE when clusters provided
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
from pyTOST.fewcluster import wild_cluster_bootstrap_ci as _fewcluster_wcb_ci


def _percentile_ci(arr: np.ndarray, alpha: float) -> Tuple[float, float]:
    # For equivalence at α one-sided, we use a 100*(1-2α)% CI for μ.
    lo = np.quantile(arr, 2 * alpha / 2.0)  # α lower for two tails
    hi = np.quantile(arr, 1 - 2 * alpha / 2.0)
    return float(lo), float(hi)


class HeteroskedasticTOST:
    def __init__(self, y: str, cluster: Optional[str] = None, wild_B: int = 999, wild_type: str = "rademacher", seed: int = 42):
        """
        Parameters
        ----------
        y : str
            Response column (e.g., SAV difference).
        cluster : str or None
            Cluster column (for example, cluster_id). If provided, use cluster-robust SE and wild cluster bootstrap.
        wild_B : int
            Number of wild bootstrap replicates.
        wild_type : {"rademacher"}
            Multiplier type for wild bootstrap. (Rademacher: ±1 with equal prob.)
        seed : int
            RNG seed.
        """
        self.y = y
        self.cluster = cluster
        self.wild_B = int(wild_B)
        self.wild_type = wild_type
        self.seed = int(seed)

    # --- core estimators ---
    def _hc3(self, df: pd.DataFrame, alpha: float):
        X = np.ones((len(df), 1))
        fit = sm.OLS(df[self.y].to_numpy(float), X).fit(cov_type="HC3")
        mu = float(fit.params[0])
        se = float(fit.bse[0])
        # conservative: large-sample normal or Student with N-1
        tcrit = stats.t.ppf(1 - alpha, df=len(df) - 1)
        return mu, (mu - tcrit * se, mu + tcrit * se), "OLS (HC3)"

    def _cluster_robust(self, df: pd.DataFrame, alpha: float):
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
        Percentile-t CI for μ using the tested fewcluster wild cluster bootstrap
        (Cameron et al., 2008). Cluster-level Rademacher sign flips act on raw
        (non-recentered) residuals so bootstrap means spread around the original mean.
        """
        r = _fewcluster_wcb_ci(
            df[self.y].to_numpy(float),
            df[self.cluster].to_numpy(),
            alpha=alpha,
            B=self.wild_B,
            seed=self.seed,
            se="CR2",
        )
        return float(r["ci_low"]), float(r["ci_high"])

    # --- fit ---
    def fit(self, df: pd.DataFrame, alpha: float, margins: List[float]) -> pd.DataFrame:
        have_cluster = self.cluster is not None and self.cluster in df.columns

        if have_cluster:
            mu, _ci_cr, _label = self._cluster_robust(df, alpha)
            ci_boot = self._wild_cluster_bootstrap_ci(df, alpha)
            ci = ci_boot
            label = "Wild Cluster Bootstrap (CR2, Rademacher)"
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


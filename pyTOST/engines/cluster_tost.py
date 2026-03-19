"""engines/cluster_tost.py
=======================

Cluster-aware TOST using **cluster-robust OLS only**.

Rationale
---------
For equivalence testing of a mean difference with clustered observations (e.g., multiple
samples within buildings), a random-intercept MixedLM fit can become numerically unstable
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
    y
        Response column name (paired difference), e.g., ``diff``.
    cluster
        Cluster/group column name (e.g., ``building_id``).
    """

    def __init__(self, y: str, cluster: str):
        self.y = y
        self.cluster = cluster

    def _cluster_robust_ci(self, df: pd.DataFrame, alpha: float) -> Tuple[float, Tuple[float, float], int, str]:
        """Compute μ̂ and a two-sided (1-2α) CI using cluster-robust OLS."""
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
        """Run CI-in-TOST decisions for all margins.

        Returns
        -------
        DataFrame
            Columns:
            - delta: equivalence margin Δ
            - mu_hat: point estimate of mean difference μ̂
            - ci_low, ci_high: CI bounds for μ
            - equivalent: whether CI is inside (-Δ,+Δ)
            - method: estimator label
            - df: degrees of freedom used for the t critical value
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

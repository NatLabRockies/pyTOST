"""Few-cluster inference for the intercept-only mean paired difference.

Standard cluster-robust (CR0) sandwich SEs with a t(G-1) reference are known to
over-reject with few clusters (Cameron & Miller 2015, JHR 50:317-372). This module
adds two small-sample-robust alternatives for the mean of a paired difference:

* ``cr2_mean``: the bias-reduced (CR2 / Bell-McCaffrey) cluster-robust variance
  estimator with Satterthwaite degrees of freedom, specialised to the
  intercept-only model where it has a closed form.
* ``wild_cluster_bootstrap_ci``: an unrestricted wild cluster bootstrap-t
  percentile-t interval using cluster-level Rademacher weights.

References
----------
- Bell, R. M., & McCaffrey, D. F. (2002). Bias reduction in standard errors for
  linear regression with multi-stage samples. Survey Methodology, 28(2), 169-181.
- Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2008). Bootstrap-based
  improvements for inference with clustered errors. REStat 90(3), 414-427.
- Cameron, A. C., & Miller, D. L. (2015). A practitioner's guide to cluster-robust
  inference. Journal of Human Resources 50(2), 317-372.
- Pustejovsky, J. E., & Tipton, E. (2018). Small-sample methods for cluster-robust
  variance estimation and hypothesis testing in fixed effects models. JBES 36(4).
"""
from __future__ import annotations

from itertools import product
from typing import List, Optional
import numpy as np
import pandas as pd
from scipy import stats


def _rademacher_support(G: int) -> np.ndarray:
    """Return the full 2**G matrix of cluster-level Rademacher sign vectors."""
    return np.array(list(product((-1.0, 1.0), repeat=G)), dtype=float)


def _cluster_sums(y: np.ndarray, groups: np.ndarray):
    """Return per-cluster residual sums and sizes for the intercept-only model."""
    mu = float(y.mean())
    e = y - mu
    uniq = pd.unique(groups)
    s_raw = np.array([e[groups == g].sum() for g in uniq])
    n_g = np.array([int((groups == g).sum()) for g in uniq])
    return mu, e, uniq, s_raw, n_g


def cr2_mean(y: np.ndarray, groups: np.ndarray, alpha: float = 0.05):
    """CR2 (Bell-McCaffrey) cluster-robust CI for the intercept-only mean.

    For X = 1_N the hat block is H_gg = (1/N) J_{n_g}, so the CR2 adjustment acts
    only on the cluster mean direction with factor (1 - n_g/N)^{-1/2}. The CR2
    cluster score is therefore sum(e_g) / sqrt(1 - n_g/N).

    Returns dict with mu, se_cr2, se_cr0, se_iid, df_bm (Satterthwaite), ci.
    """
    y = np.asarray(y, float)
    groups = np.asarray(groups)
    N = len(y)
    mu, e, uniq, s_raw, n_g = _cluster_sums(y, groups)
    G = len(uniq)

    # CR0 (standard sandwich): V = sum_g s_g^2 / N^2
    v_cr0 = float((s_raw ** 2).sum()) / N ** 2
    # CR2: inflate each cluster score by (1 - n_g/N)^{-1/2}
    adj = 1.0 / np.sqrt(1.0 - n_g / N)
    s_cr2 = s_raw * adj
    v_cr2 = float((s_cr2 ** 2).sum()) / N ** 2

    # IID SE
    se_iid = float(y.std(ddof=1) / np.sqrt(N))

    # Bell-McCaffrey / Satterthwaite df via the clubSandwich P-matrix.
    # For the intercept-only model, the CR2 adjustment factor is a_g = (1 - n_g/N)^{-1/2}
    # and the working-model per-cluster contribution to Var(mu_hat) under
    # homoskedasticity is p_g = n_g * a_g^2 / N^2. The cross-cluster coupling
    # comes from the shared intercept: h_g = a_g * n_g / N^{1.5}. The G x G
    # matrix P = diag(p_g) - h h^T then gives
    #   df = trace(P)^2 / sum(P .* P)
    # (Pustejovsky & Tipton 2018; clubSandwich 0.7.0). This reduces to G-1 for
    # balanced clusters, in contrast to the previous scalar-weight approximation.
    a = adj
    p_g = n_g * a ** 2 / N ** 2
    h_g = a * n_g / N ** 1.5
    P = np.diag(p_g) - np.outer(h_g, h_g)
    df_bm = float((np.trace(P) ** 2) / (P * P).sum())
    df_bm = max(df_bm, 1.0)

    se_cr2 = np.sqrt(v_cr2)
    tcrit = stats.t.ppf(1 - alpha, df=df_bm)
    ci = (mu - tcrit * se_cr2, mu + tcrit * se_cr2)
    return {
        "mu": mu, "se_cr2": float(se_cr2), "se_cr0": float(np.sqrt(v_cr0)),
        "se_iid": se_iid, "df_bm": df_bm, "G": G,
        "ci_low": float(ci[0]), "ci_high": float(ci[1]),
    }


def wild_cluster_bootstrap_ci(y: np.ndarray, groups: np.ndarray, alpha: float = 0.05,
                              B: int = 1999, seed: int = 123, se: str = "CR2",
                              exact: Optional[bool] = None, enum_cap_G: int = 13):
    """Unrestricted wild cluster bootstrap-t (percentile-t) CI for the mean.

    Cluster-level Rademacher weights w_g in {-1,+1} are applied to residuals to
    form y*_i = mu_hat + w_{g(i)} e_i; the studentised statistic
    t*_b = (mu_hat* - mu_hat)/se* is collected and the percentile-t interval is
    ci = [mu_hat - q_{1-alpha} se, mu_hat - q_{alpha} se].

    ``se`` selects the studentising SE inside the bootstrap ("CR2" or "CR0").

    With few clusters the Rademacher weight vector has only ``2**G`` distinct
    values, so drawing ``B`` random vectors merely resamples that finite support
    with duplication. When ``exact`` is true (or ``exact`` is ``None`` and
    ``2**G <= 2**enum_cap_G``) the full ``2**G`` sign support is enumerated once
    and the reference distribution is exact rather than Monte Carlo.
    """
    y = np.asarray(y, float)
    groups = np.asarray(groups)
    N = len(y)
    mu, e, uniq, s_raw, n_g = _cluster_sums(y, groups)
    G = len(uniq)

    if exact is None:
        exact = G <= enum_cap_G

    def _se(yy):
        m, ee, _u, sr, ng = _cluster_sums(yy, groups)
        if se == "CR0":
            v = (sr ** 2).sum() / N ** 2
        else:
            adj = 1.0 / np.sqrt(1.0 - ng / N)
            v = ((sr * adj) ** 2).sum() / N ** 2
        return np.sqrt(max(v, 1e-300))

    se_hat = _se(y)
    # map cluster -> row mask once
    masks = {g: (groups == g) for g in uniq}

    if exact:
        signs = _rademacher_support(G)
        n_patterns = signs.shape[0]
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice((-1.0, 1.0), size=(B, G))
        n_patterns = B

    tstars = np.empty(n_patterns)
    for b in range(n_patterns):
        w = signs[b]
        ystar = y.copy()
        for gi, g in enumerate(uniq):
            m = masks[g]
            ystar[m] = mu + w[gi] * e[m]
        mstar = float(ystar.mean())
        tstars[b] = (mstar - mu) / _se(ystar)

    q_lo = np.quantile(tstars, alpha)
    q_hi = np.quantile(tstars, 1 - alpha)
    ci_low = mu - q_hi * se_hat
    ci_high = mu - q_lo * se_hat
    return {"mu": mu, "se": float(se_hat), "G": G,
            "B": int(n_patterns), "exact": bool(exact), "n_patterns": int(n_patterns),
            "seed": (None if exact else seed),
            "ci_low": float(ci_low), "ci_high": float(ci_high),
            "t_q_low": float(q_lo), "t_q_high": float(q_hi)}


def wild_cluster_tost(df: pd.DataFrame, y: str, cluster: str, margins: List[float],
                      alpha: float = 0.05, B: int = 1999, seed: int = 123,
                      se: str = "CR2", exact: Optional[bool] = None) -> pd.DataFrame:
    """TOST equivalence decisions using the wild cluster bootstrap-t CI."""
    r = wild_cluster_bootstrap_ci(df[y].to_numpy(float), df[cluster].to_numpy(),
                                  alpha=alpha, B=B, seed=seed, se=se, exact=exact)
    if r["exact"]:
        label = f"Wild cluster bootstrap-t ({se}, exact {r['n_patterns']} sign patterns)"
    else:
        label = f"Wild cluster bootstrap-t ({se}, B={r['n_patterns']})"
    rows = []
    for d in margins:
        d = float(d)
        rows.append(dict(delta=d, mu_hat=r["mu"], ci_low=r["ci_low"], ci_high=r["ci_high"],
                         equivalent=(r["ci_low"] > -d and r["ci_high"] < d),
                         method=label, G=r["G"]))
    return pd.DataFrame(rows)

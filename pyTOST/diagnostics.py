"""Diagnostics for assessing clustered, spatial, and temporal dependence.

This module provides lightweight summary statistics used by the workflow to
screen for clustered, spatial, and temporal dependence in paired-difference
data. The diagnostics are intended to guide engine selection and to provide
quick descriptive evidence about the likely structure of dependence.

References
----------
Moran (1950). Biometrika 37:17--23.
Cliff and Ord (1981). Spatial Processes: Models and Applications. Pion.
Cressie (1993). Statistics for Spatial Data. Wiley.
Snijders and Bosker (2012). Multilevel Analysis. Sage.
Bell and McCaffrey (2002). Survey Methodology 28:169--182.
Pustejovsky and Tipton (2018). Journal of Business and Economic Statistics
36(4):672--683.
Box and Pierce (1970); Ljung and Box (1978). Biometrika.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Optional, Tuple

# --------- utilities ---------
def _pairwise_dist(xy: np.ndarray) -> np.ndarray:
    """Compute the Euclidean pairwise distance matrix for planar coordinates.

    Parameters
    ----------
    xy : ndarray of shape (n_samples, 2)
        Array containing planar coordinates.

    Returns
    -------
    ndarray of shape (n_samples, n_samples)
        Symmetric matrix of Euclidean distances.
    """
    d = xy[:, None, :] - xy[None, :, :]
    return np.sqrt((d**2).sum(axis=2))

# --------- cluster diagnostics ---------
def intraclass_correlation(df: pd.DataFrame, y: str, cluster: str) -> float:
    """Estimate a one-way random-effects intraclass correlation coefficient.

    The estimator is a method-of-moments intraclass correlation coefficient
    based on one-way analysis-of-variance mean squares.

    Parameters
    ----------
    df : pandas.DataFrame
        Input table containing the response and cluster identifier.
    y : str
        Name of the response column.
    cluster : str
        Name of the cluster identifier column.

    Returns
    -------
    float
        Estimated intraclass correlation coefficient,
        :math:`\\tau^2 / (\\tau^2 + \\sigma^2)`.

    Notes
    -----
    The estimator uses the relationship
    :math:`MS_B = \\sigma^2 + \\bar{k}\\tau^2` and :math:`MS_W = \\sigma^2`.
    """
    g = df.groupby(cluster)[y]
    k = g.size().to_numpy()
    m = g.mean().to_numpy()
    n = int(k.sum())
    G = int(k.size)
    if G <= 1:
        return 0.0
    grand = float(df[y].mean())
    ssb = float((k * (m - grand)**2).sum())
    ssw = float(g.apply(lambda s: ((s - s.mean())**2).sum()).sum())
    dfb = G - 1
    dfw = n - G
    msb = ssb / max(dfb, 1)
    msw = ssw / max(dfw, 1)
    kbar = k.mean()
    tau2 = max((msb - msw) / max(kbar, 1e-12), 0.0)
    sigma2 = max(msw, 0.0)
    return float(tau2 / max(tau2 + sigma2, 1e-12))

def se_inflation_ratio(se_cluster: float, se_iid: float) -> float:
    """Compute the ratio of clustered to IID standard errors.

    Parameters
    ----------
    se_cluster : float
        Cluster-aware standard error.
    se_iid : float
        IID standard error.

    Returns
    -------
    float
        Ratio ``se_cluster / se_iid``. Values greater than 1 indicate that
        the IID standard error is smaller than the clustered standard error.
    """
    if se_iid <= 0:
        return np.inf
    return float(se_cluster / se_iid)

# --------- spatial diagnostics ---------
def morans_I(df: pd.DataFrame, y: str, x: str, ycoord: str, k: int = 4) -> Tuple[float, float]:
    """Compute a quick Moran's I diagnostic using k-nearest-neighbor weights.

    Parameters
    ----------
    df : pandas.DataFrame
        Input table containing the response and coordinates.
    y : str
        Name of the response column.
    x : str
        Name of the x-coordinate column.
    ycoord : str
        Name of the y-coordinate column.
    k : int, default=4
        Number of nearest neighbors used to build the row-standardized weights
        matrix.

    Returns
    -------
    I : float
        Moran's I statistic.
    p_value : float
        Two-sided p-value from a rough normal approximation.
    """
    xy = df[[x, ycoord]].to_numpy(float)
    n = xy.shape[0]
    if n < 5:
        return np.nan, np.nan
    D = _pairwise_dist(xy)
    np.fill_diagonal(D, np.inf)
    W = np.zeros((n, n), float)
    for i in range(n):
        idx = np.argsort(D[i])[:min(k, n-1)]
        W[i, idx] = 1.0
    W = np.maximum(W, W.T)  # symmetrize
    rs = W.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    W = W / rs
    z = df[y].to_numpy(float)
    zc = z - z.mean()
    s0 = W.sum()
    if s0 <= 0:
        return np.nan, np.nan
    I = (n / s0) * (zc @ (W @ zc)) / (zc @ zc + 1e-12)
    EnI, varI = -1.0/(n-1), 1.0/(n-1)  # rough
    zstat = (I - EnI) / np.sqrt(max(varI, 1e-12))
    pval = 2 * (1 - stats.norm.cdf(abs(zstat)))
    return float(I), float(pval)

# --------- temporal diagnostics ---------
def durbin_watson(resid: np.ndarray) -> float:
    """Compute the Durbin--Watson statistic for ordered residuals.

    Parameters
    ----------
    resid : ndarray
        Ordered residual sequence.

    Returns
    -------
    float
        Durbin--Watson statistic. Smaller values typically indicate positive
        serial autocorrelation.
    """
    e = np.asarray(resid, float).ravel()
    if e.size < 4:
        return np.nan
    num = np.sum(np.diff(e)**2)
    den = np.sum(e**2) + 1e-12
    return float(num / den)

def ljung_box(resid: np.ndarray, lags: int = 12) -> Tuple[float, float]:
    """Compute the Ljung--Box test statistic for autocorrelation.

    Parameters
    ----------
    resid : ndarray
        Ordered residual sequence.
    lags : int, default=12
        Maximum lag included in the test.

    Returns
    -------
    Q : float
        Ljung--Box Q statistic.
    p_value : float
        Upper-tail chi-square p-value using ``lags`` degrees of freedom.
    """
    e = np.asarray(resid, float).ravel()
    n = e.size
    if n < lags + 3:
        return np.nan, np.nan
    acfs = np.array([1.0 if h == 0 else np.corrcoef(e[:-h], e[h:])[0,1] for h in range(0, lags+1)])
    Q = n * (n + 2) * np.sum((acfs[1:]**2) / (n - np.arange(1, lags+1)))
    pval = 1 - stats.chi2.cdf(Q, df=lags)
    return float(Q), float(pval)

# --------- summary ---------
def summarize_diagnostics(df: pd.DataFrame,
                          y: str,
                          cluster: Optional[str] = None,
                          x: Optional[str] = None,
                          ycoord: Optional[str] = None,
                          time: Optional[str] = None) -> Dict:
    """Summarize dependence diagnostics available for the supplied dataset.

    Parameters
    ----------
    df : pandas.DataFrame
        Input analysis table.
    y : str
        Name of the response column.
    cluster : str, optional
        Name of the cluster identifier column.
    x : str, optional
        Name of the x-coordinate column.
    ycoord : str, optional
        Name of the y-coordinate column.
    time : str, optional
        Name of the time index column.

    Returns
    -------
    dict
        Dictionary containing any applicable diagnostics. Keys may include
        ``icc``, ``se_inflation``, ``morans_I``, ``morans_p``, ``dw``,
        ``ljung_box_Q``, and ``ljung_box_p`` depending on which columns are
        provided.
    """
    out: Dict[str, float] = {}
    if cluster and cluster in df.columns:
        try:
            out["icc"] = intraclass_correlation(df, y=y, cluster=cluster)
        except Exception:
            out["icc"] = np.nan
        # SE inflation via intercept-only OLS with/without clustering
        import statsmodels.api as sm
        X = np.ones((len(df),1))
        ols = sm.OLS(df[y].to_numpy(float), X).fit()
        se_iid = float(ols.bse[0])
        ols_cl = sm.OLS(df[y].to_numpy(float), X).fit(
            cov_type="cluster", cov_kwds={"groups": df[cluster].to_numpy()}
        )
        se_cl = float(ols_cl.bse[0])
        out["se_inflation"] = se_inflation_ratio(se_cl, se_iid)

    if x and ycoord and (x in df.columns) and (ycoord in df.columns):
        try:
            I, p = morans_I(df, y=y, x=x, ycoord=ycoord, k=4)
        except Exception:
            I, p = np.nan, np.nan
        out["morans_I"], out["morans_p"] = I, p

    if time and (time in df.columns):
        ser = df.sort_values(time)[y].to_numpy(float)
        out["dw"] = durbin_watson(ser - ser.mean())
        Q, p = ljung_box(ser - ser.mean(), lags=12)
        out["ljung_box_Q"], out["ljung_box_p"] = Q, p

    return out

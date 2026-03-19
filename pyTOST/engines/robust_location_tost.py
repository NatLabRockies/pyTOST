"""Robust-location TOST based on bootstrap confidence intervals.

This module provides a dependence-aware equivalence-testing workflow for robust
location statistics such as the median or a trimmed mean. Confidence intervals
are obtained by bootstrap resampling, with the resampling scheme chosen to
match the observed dependence structure.

Classes
-------
RobustLocationTOST
    Perform TOST using a robust location statistic and bootstrap confidence
    intervals.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple


def _statistic(arr: np.ndarray, stat: str) -> float:
    """Compute a robust location statistic for a numeric array.

    Parameters
    ----------
    arr : numpy.ndarray
        One-dimensional array of observations.
    stat : str
        Name of the statistic to compute. Supported options are ``"median"``,
        ``"trimmed_mean_20"``, and fallback mean behavior for any other value.

    Returns
    -------
    float
        Estimated location statistic.
    """
    if stat == "median":
        return float(np.median(arr))
    elif stat == "trimmed_mean_20":
        # 20% trimmed mean
        lo = int(0.2 * len(arr))
        hi = len(arr) - lo
        return float(np.mean(np.sort(arr)[lo:hi]))
    else:
        # fallback to simple mean
        return float(np.mean(arr))


def _percentile_ci(arr: np.ndarray, alpha: float) -> Tuple[float, float]:
    """Compute a two-sided percentile bootstrap confidence interval.

    Parameters
    ----------
    arr : numpy.ndarray
        Bootstrap replicates of a scalar statistic.
    alpha : float
        One-sided TOST significance level. The resulting confidence interval
        has nominal coverage ``1 - 2 * alpha``.

    Returns
    -------
    tuple of float
        Lower and upper percentile confidence limits.
    """
    ql = 2 * alpha / 2.0
    qh = 1 - 2 * alpha / 2.0
    return float(np.quantile(arr, ql)), float(np.quantile(arr, qh))


class RobustLocationTOST:
    """TOST based on a robust location statistic with bootstrap uncertainty.

    The class computes a point estimate from a robust location statistic and
    constructs a bootstrap confidence interval adapted to clustered, temporal,
    or independent data.
    """

    def __init__(
        self,
        y: str,
        cluster: Optional[str] = None,
        time: Optional[str] = None,
        block_len: int = 5,
        B: int = 2000,
        seed: int = 42,
        stat: str = "median",
    ):
        """
        Parameters
        ----------
        y : str
            Response column (e.g., SAV difference).
        cluster : str or None
            Cluster/group id (for example, cluster_id). If provided, cluster bootstrap is used.
        time : str or None
            Time column. If provided (and cluster absent), moving-block bootstrap is used.
        block_len : int
            Block length for time bootstrap (fixed, simple).
        B : int
            Bootstrap replicates.
        seed : int
            RNG seed.
        stat : {"median", "trimmed_mean_20", "mean"}
            Robust location estimator to test via TOST.
        """
        self.y = y
        self.cluster = cluster
        self.time = time
        self.block_len = int(block_len)
        self.B = int(B)
        self.seed = int(seed)
        self.stat = stat

    # --- resampling generators ---
    def _cluster_bootstrap_stats(self, df: pd.DataFrame) -> np.ndarray:
        """Generate bootstrap replicates using whole-cluster resampling.

        Parameters
        ----------
        df : pandas.DataFrame
            Input analysis table containing the response and cluster columns.

        Returns
        -------
        numpy.ndarray
            Bootstrap replicates of the selected robust location statistic.
        """
        rng = np.random.default_rng(self.seed)
        groups = df[self.cluster].unique()
        vals = []
        for _ in range(self.B):
            take = rng.choice(groups, size=len(groups), replace=True)
            boot = pd.concat([df[df[self.cluster] == g] for g in take], ignore_index=True)
            vals.append(_statistic(boot[self.y].to_numpy(float), self.stat))
        return np.asarray(vals, float)

    def _moving_block_bootstrap_stats(self, df: pd.DataFrame) -> np.ndarray:
        """Generate bootstrap replicates using a moving-block bootstrap.

        Parameters
        ----------
        df : pandas.DataFrame
            Input analysis table containing the response and time columns.

        Returns
        -------
        numpy.ndarray
            Bootstrap replicates of the selected robust location statistic.
        """
        rng = np.random.default_rng(self.seed)
        df2 = df.sort_values(self.time).reset_index(drop=True)
        n = len(df2)
        b = max(self.block_len, 1)
        vals = []
        # draw blocks with start indices uniformly (circular wrapping)
        for _ in range(self.B):
            idx = []
            while len(idx) < n:
                s = rng.integers(0, n)
                idx.extend((s + np.arange(b)) % n)
            idx = np.array(idx[:n])
            boot = df2.iloc[idx]
            vals.append(_statistic(boot[self.y].to_numpy(float), self.stat))
        return np.asarray(vals, float)

    def _iid_bootstrap_stats(self, df: pd.DataFrame) -> np.ndarray:
        """Generate bootstrap replicates under an IID resampling scheme.

        Parameters
        ----------
        df : pandas.DataFrame
            Input analysis table containing the response column.

        Returns
        -------
        numpy.ndarray
            Bootstrap replicates of the selected robust location statistic.
        """
        rng = np.random.default_rng(self.seed)
        y = df[self.y].to_numpy(float)
        vals = []
        for _ in range(self.B):
            idx = rng.integers(0, len(y), size=len(y))
            vals.append(_statistic(y[idx], self.stat))
        return np.asarray(vals, float)

    # --- API ---
    def fit(self, df: pd.DataFrame, alpha: float, margins: List[float]) -> pd.DataFrame:
        """Fit the robust-location TOST model for one or more margins.

        Parameters
        ----------
        df : pandas.DataFrame
            Input analysis table containing the response column and, when
            relevant, cluster or time identifiers.
        alpha : float
            One-sided TOST significance level. The reported confidence interval
            has nominal coverage ``1 - 2 * alpha``.
        margins : list of float
            Equivalence margins to evaluate.

        Returns
        -------
        pandas.DataFrame
            Result table with one row per equivalence margin and columns for the
            estimate, confidence interval, equivalence decision, and method
            label.
        """
        # point estimate:
        mu_hat = _statistic(df[self.y].to_numpy(float), self.stat)

        # bootstrap CI according to dependence structure:
        if self.cluster and self.cluster in df.columns:
            arr = self._cluster_bootstrap_stats(df)
            label = f"Robust {self.stat} + Cluster Bootstrap"
        elif self.time and self.time in df.columns:
            arr = self._moving_block_bootstrap_stats(df)
            label = f"Robust {self.stat} + Moving-Block Bootstrap (b={self.block_len})"
        else:
            arr = self._iid_bootstrap_stats(df)
            label = f"Robust {self.stat} + IID Bootstrap"

        ci = _percentile_ci(arr, alpha)

        rows = []
        for d in margins:
            rows.append(
                dict(
                    delta=float(d),
                    mu_hat=float(mu_hat),
                    ci_low=float(ci[0]),
                    ci_high=float(ci[1]),
                    equivalent=(ci[0] > -d and ci[1] < d),
                    method=label,
                )
            )
        return pd.DataFrame(rows)


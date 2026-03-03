"""
engines/robust_location_tost.py
===============================
Robust-location TOST based on the *median* (or other M-estimator) with bootstrap CI.

Why
---
The mean can be non-robust under heavy tails or outliers. For practical
equivalence judgments you may prefer a robust location statistic (median).
We construct a (1-2α) bootstrap CI for the median and apply the CI containment
criterion for TOST.

Dependence-aware resampling
---------------------------
- If `cluster` present: **cluster bootstrap** (resample whole clusters).
- Else if `time` present: **moving/block bootstrap** (simple fixed-length blocks).
- Else: i.i.d. bootstrap.

Notes
-----
Block bootstrap here is a light-weight fixed-size moving-block bootstrap suitable
for quick sensitivity checks. For rigorous time-series work, you may want to
swap-in a circular or stationary bootstrap.

References
----------
- Davison & Hinkley (1997) Bootstrap Methods and Their Application.
- Lahiri (2003) Resampling Methods for Dependent Data.
- Hettmansperger & Sheather (1986) Robust estimation (median properties).

API
---
RobustLocationTOST(y, cluster=None, time=None, block_len=5, B=2000, seed=42, stat="median")
  .fit(df, alpha, margins) -> DataFrame
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple


def _statistic(arr: np.ndarray, stat: str) -> float:
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
    ql = 2 * alpha / 2.0
    qh = 1 - 2 * alpha / 2.0
    return float(np.quantile(arr, ql)), float(np.quantile(arr, qh))


class RobustLocationTOST:
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
            Cluster/group id (e.g., building_id). If provided, cluster bootstrap is used.
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
        rng = np.random.default_rng(self.seed)
        groups = df[self.cluster].unique()
        vals = []
        for _ in range(self.B):
            take = rng.choice(groups, size=len(groups), replace=True)
            boot = pd.concat([df[df[self.cluster] == g] for g in take], ignore_index=True)
            vals.append(_statistic(boot[self.y].to_numpy(float), self.stat))
        return np.asarray(vals, float)

    def _moving_block_bootstrap_stats(self, df: pd.DataFrame) -> np.ndarray:
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
        rng = np.random.default_rng(self.seed)
        y = df[self.y].to_numpy(float)
        vals = []
        for _ in range(self.B):
            idx = rng.integers(0, len(y), size=len(y))
            vals.append(_statistic(y[idx], self.stat))
        return np.asarray(vals, float)

    # --- API ---
    def fit(self, df: pd.DataFrame, alpha: float, margins: List[float]) -> pd.DataFrame:
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


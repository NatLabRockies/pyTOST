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
RobustLocationTOST(y, cluster=None, time=None, block_len=5, B=2000, seed=42,
                   stat="median", trim=0.2)
  .fit(df, alpha, margins) -> DataFrame
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple

#: Location statistics accepted by :class:`RobustLocationTOST`.
VALID_STATS = ("median", "trimmed_mean", "trimmed_mean_20", "mean")

_LEGACY_TRIM = {"trimmed_mean_20": 0.2}

DEFAULT_TRIM = 0.2


def _validate_stat(stat: str) -> str:
    if stat not in VALID_STATS:
        raise ValueError(
            f"stat={stat!r} is not a supported location statistic. "
            f"Choose one of {list(VALID_STATS)}."
        )
    return stat


def _validate_trim(trim: float) -> float:
    trim = float(trim)
    if not (0.0 <= trim < 0.5):
        raise ValueError(
            f"trim={trim!r} is not valid; the trim fraction must satisfy "
            "0.0 <= trim < 0.5 (it is removed from *each* tail)."
        )
    return trim


def _statistic(arr: np.ndarray, stat: str, trim: float = DEFAULT_TRIM) -> float:
    """Compute a location statistic.

    Parameters
    ----------
    arr : ndarray
        Sample values.
    stat : {"median", "trimmed_mean", "trimmed_mean_20", "mean"}
        Statistic to compute.
    trim : float
        Fraction removed from each tail when ``stat="trimmed_mean"``. Ignored by the
        other statistics. ``"trimmed_mean_20"`` always trims 20%.
    """
    arr = np.asarray(arr, dtype=float)

    if stat == "median":
        return float(np.median(arr))

    if stat in _LEGACY_TRIM:
        trim = _LEGACY_TRIM[stat]
        stat = "trimmed_mean"

    if stat == "trimmed_mean":
        trim = _validate_trim(trim)
        n = arr.size
        if n == 0:
            return float("nan")
        # Symmetric trimming, matching scipy.stats.trim_mean semantics.
        cut = int(n * trim)
        kept = np.sort(arr)[cut : n - cut] if cut else arr
        return float(np.mean(kept))

    if stat == "mean":
        return float(np.mean(arr))

    raise ValueError(
        f"stat={stat!r} is not a supported location statistic. "
        f"Choose one of {list(VALID_STATS)}."
    )


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
        trim: float = DEFAULT_TRIM,
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
        stat : {"median", "trimmed_mean", "trimmed_mean_20", "mean"}
            Robust location estimator to test via TOST. ``"trimmed_mean"`` uses the
            ``trim`` fraction below. ``"trimmed_mean_20"`` is the fixed 20% variant
            retained for backwards compatibility.
        trim : float, default 0.2
            Fraction removed from *each* tail when ``stat="trimmed_mean"``; must satisfy
            ``0.0 <= trim < 0.5``. ``trim=0.0`` reproduces the sample mean and larger
            values approach the median, so the parameter interpolates between the two.
            Ignored for other statistics.

        Raises
        ------
        ValueError
            If ``stat`` is not recognized, or ``trim`` is outside ``[0.0, 0.5)``.
        """
        self.y = y
        self.cluster = cluster
        self.time = time
        self.block_len = int(block_len)
        self.B = int(B)
        self.seed = int(seed)
        self.stat = _validate_stat(stat)
        self.trim = _validate_trim(trim)

    def _stat(self, values: np.ndarray) -> float:
        return _statistic(values, self.stat, trim=self.trim)

    def _stat_label(self) -> str:
        if self.stat in ("trimmed_mean", "trimmed_mean_20"):
            trim = _LEGACY_TRIM.get(self.stat, self.trim)
            return f"trimmed mean (trim={trim:g})"
        return self.stat

    # --- resampling generators ---
    def _cluster_bootstrap_stats(self, df: pd.DataFrame) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        groups = df[self.cluster].unique()
        vals = []
        for _ in range(self.B):
            take = rng.choice(groups, size=len(groups), replace=True)
            boot = pd.concat([df[df[self.cluster] == g] for g in take], ignore_index=True)
            vals.append(self._stat(boot[self.y].to_numpy(float)))
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
            vals.append(self._stat(boot[self.y].to_numpy(float)))
        return np.asarray(vals, float)

    def _iid_bootstrap_stats(self, df: pd.DataFrame) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        y = df[self.y].to_numpy(float)
        vals = []
        for _ in range(self.B):
            idx = rng.integers(0, len(y), size=len(y))
            vals.append(self._stat(y[idx]))
        return np.asarray(vals, float)

    # --- API ---
    def fit(self, df: pd.DataFrame, alpha: float, margins: List[float]) -> pd.DataFrame:
        # point estimate:
        mu_hat = self._stat(df[self.y].to_numpy(float))

        # bootstrap CI according to dependence structure:
        if self.cluster and self.cluster in df.columns:
            arr = self._cluster_bootstrap_stats(df)
            label = f"Robust {self._stat_label()} + Cluster Bootstrap"
        elif self.time and self.time in df.columns:
            arr = self._moving_block_bootstrap_stats(df)
            label = f"Robust {self._stat_label()} + Moving-Block Bootstrap (b={self.block_len})"
        else:
            arr = self._iid_bootstrap_stats(df)
            label = f"Robust {self._stat_label()} + IID Bootstrap"

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


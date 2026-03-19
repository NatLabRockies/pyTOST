"""Bootstrap utilities for validation confidence intervals.

This module provides bootstrap routines for independent, clustered,
spatial, temporal, and spatiotemporal paired-difference settings used by
:mod:`pyTOST` validation workflows.

References
----------
Davison, A. C., and Hinkley, D. V. (1997).
    *Bootstrap Methods and Their Application*.
Lahiri, S. N. (2003).
    *Resampling Methods for Dependent Data*.
"""

from __future__ import annotations

import math
from typing import Callable, Dict

import numpy as np
import pandas as pd


def _grid_block_labels(
    x: np.ndarray,
    y: np.ndarray,
    block_size: float,
) -> np.ndarray:
    """Assign points to rectangular grid blocks.

    Parameters
    ----------
    x : numpy.ndarray
        X coordinates of the points.
    y : numpy.ndarray
        Y coordinates of the points.
    block_size : float
        Width and height of each rectangular grid block.

    Returns
    -------
    numpy.ndarray
        Object array of block labels with entries of the form ``"ix:iy"``.

    Notes
    -----
    Labels are built with Python string formatting rather than NumPy
    string-array addition to avoid ``UFuncTypeError`` in some NumPy and
    pandas builds.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    xmin = float(np.min(x))
    ymin = float(np.min(y))

    ix = np.floor((x - xmin) / float(block_size)).astype(int)
    iy = np.floor((y - ymin) / float(block_size)).astype(int)

    return np.asarray([f"{i}:{j}" for i, j in zip(ix, iy)], dtype=object)


def cluster_bootstrap(
    df: pd.DataFrame,
    y: str,
    cluster: str,
    fit_fn: Callable[[pd.DataFrame], float],
    B: int = 200,
    seed: int = 42,
) -> Dict:
    """Bootstrap a statistic by resampling clusters.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data frame.
    y : str
        Name of the paired-difference column.
    cluster : str
        Column identifying clusters to resample.
    fit_fn : callable
        Function applied to each bootstrap resample. It must accept a
        :class:`pandas.DataFrame` and return a scalar statistic.
    B : int, default=200
        Number of bootstrap replicates.
    seed : int, default=42
        Seed for the random number generator.

    Returns
    -------
    dict
        Dictionary containing the number of replicates, the percentile 90%
        confidence interval, and the bootstrap samples.
    """
    rng = np.random.default_rng(seed)

    values = []
    groups = df[cluster].unique()
    G = len(groups)

    for _ in range(B):
        take = rng.choice(groups, size=G, replace=True)
        out = pd.concat([df[df[cluster] == g] for g in take], axis=0, ignore_index=True)
        values.append(float(fit_fn(out)))

    arr = np.asarray(values, dtype=float)
    ci = (float(np.quantile(arr, 0.05)), float(np.quantile(arr, 0.95)))
    return {"B": int(B), "ci_perc_90": ci, "samples": arr}


# -----------------------------------------------------------------------------
# Spatial block bootstrap (for cross-building spatial dependence)
# -----------------------------------------------------------------------------

def spatial_block_bootstrap(
    df: pd.DataFrame,
    *,
    y: str,
    building_col: str,
    x_col: str,
    y_col: str,
    fit_fn: Callable[[pd.DataFrame], float],
    B: int = 200,
    seed: int = 42,
    block_size: float | None = None,
    blocks: pd.Series | None = None,
) -> Dict:
    """Run a spatial block bootstrap across clusters.

    Parameters
    ----------
    df : pandas.DataFrame
        Input paired-difference data.
    y : str
        Name of the paired-difference column.
    building_col : str
        Column identifying clusters or buildings.
    x_col : str
        Name of the x-coordinate column.
    y_col : str
        Name of the y-coordinate column.
    fit_fn : callable
        Function applied to each bootstrap resample. It must accept a
        :class:`pandas.DataFrame` and return a scalar statistic.
    B : int, default=200
        Number of bootstrap replicates.
    seed : int, default=42
        Seed for the random number generator.
    block_size : float or None, default=None
        Spatial block size. If ``None``, a heuristic based on centroid
        nearest-neighbor distances is used.
    blocks : pandas.Series or None, default=None
        Optional precomputed block labels indexed by cluster identifier.

    Returns
    -------
    dict
        Dictionary containing the number of replicates, the percentile 90%
        confidence interval, the bootstrap samples, and the block size used.

    Notes
    -----
    This routine resamples spatial blocks of cluster centroids rather than
    individual rows so that cross-cluster spatial dependence can be preserved
    within resampled blocks.
    """
    rng = np.random.default_rng(seed)

    centers = (
        df.groupby(building_col, sort=False)[[x_col, y_col]]
        .mean()
        .rename(columns={x_col: "cx", y_col: "cy"})
    )
    buildings = centers.index.to_numpy()

    if blocks is None:
        cx = centers["cx"].to_numpy(dtype=float)
        cy = centers["cy"].to_numpy(dtype=float)

        if block_size is None:
            # Heuristic: 2x median nearest-neighbor distance among centroids.
            if len(cx) < 3:
                block_size = float(np.ptp(cx) + np.ptp(cy) + 1.0)
            else:
                dx = cx[:, None] - cx[None, :]
                dy = cy[:, None] - cy[None, :]
                D = np.sqrt(dx * dx + dy * dy)
                np.fill_diagonal(D, np.inf)
                nn = np.min(D, axis=1)
                block_size = float(2.0 * np.median(nn))
                if not np.isfinite(block_size) or block_size <= 0:
                    block_size = float(max(np.ptp(cx), np.ptp(cy), 1.0))

        block_labels_arr = _grid_block_labels(cx, cy, float(block_size))
        blocks = pd.Series(block_labels_arr, index=centers.index, dtype="object")
    else:
        blocks = blocks.astype("object")

    block_labels = blocks.to_numpy(dtype=object)
    unique_blocks = np.unique(block_labels)
    nb = len(unique_blocks)

    values = []
    for b in range(B):
        take_blocks = rng.choice(unique_blocks, size=nb, replace=True)

        out_parts = []
        for j, bl in enumerate(take_blocks):
            blds = buildings[block_labels == bl]
            if len(blds) == 0:
                continue

            chunk = df[df[building_col].isin(blds)].copy()

            # Relabel buildings if this block draw repeats.
            if j > 0:
                chunk[building_col] = chunk[building_col].astype(str) + f"__bb{b}_{j}"

            out_parts.append(chunk)

        if len(out_parts) == 0:
            out = df.copy().reset_index(drop=True)
        else:
            out = pd.concat(out_parts, axis=0, ignore_index=True)

        values.append(float(fit_fn(out)))

    arr = np.asarray(values, dtype=float)
    ci = (float(np.quantile(arr, 0.05)), float(np.quantile(arr, 0.95)))
    return {
        "B": int(B),
        "ci_perc_90": ci,
        "samples": arr,
        "block_size": float(block_size) if block_size is not None else None,
    }


def spatial_within_building_block_bootstrap(
    df: pd.DataFrame,
    *,
    y: str,
    building_col: str,
    x_col: str,
    y_col: str,
    fit_fn: Callable[[pd.DataFrame], float],
    B: int = 200,
    seed: int = 42,
    block_size: float | None = None,
    min_blocks_per_building: int = 4,
    max_shrink_iters: int = 8,
) -> Dict:
    """Run a within-cluster spatial block bootstrap.

    Parameters
    ----------
    df : pandas.DataFrame
        Input paired-difference data.
    y : str
        Name of the paired-difference column.
    building_col : str
        Column identifying clusters or buildings.
    x_col : str
        Name of the x-coordinate column.
    y_col : str
        Name of the y-coordinate column.
    fit_fn : callable
        Function applied to each bootstrap resample. It must accept a
        :class:`pandas.DataFrame` and return a scalar statistic.
    B : int, default=200
        Number of bootstrap replicates.
    seed : int, default=42
        Seed for the random number generator.
    block_size : float or None, default=None
        Starting block size for within-cluster resampling. If ``None``, a
        heuristic based on within-cluster nearest-neighbor distances is used.
    min_blocks_per_building : int, default=4
        Minimum target number of unique blocks within each cluster when
        adaptively shrinking the block size.
    max_shrink_iters : int, default=8
        Maximum number of block-size halving iterations used to avoid
        degenerate single-block resamples.

    Returns
    -------
    dict
        Dictionary containing the number of replicates, the percentile 90%
        confidence interval, the bootstrap samples, and the starting block
        size used.

    Notes
    -----
    This routine resamples spatial blocks separately within each cluster so
    that local spatial dependence is retained without assuming independence
    among rows inside a cluster.
    """
    rng = np.random.default_rng(seed)

    if block_size is None:
        nn_meds = []
        for _b, d in df.groupby(building_col, sort=False):
            xy = d[[x_col, y_col]].to_numpy(dtype=float)
            if xy.shape[0] < 3:
                continue

            dx = xy[:, None, 0] - xy[None, :, 0]
            dy = xy[:, None, 1] - xy[None, :, 1]
            D = np.sqrt(dx * dx + dy * dy)
            np.fill_diagonal(D, np.inf)
            nn = np.min(D, axis=1)
            med = float(np.median(nn))
            if np.isfinite(med) and med > 0:
                nn_meds.append(med)

        if len(nn_meds) > 0:
            block_size = float(2.0 * np.median(nn_meds))
        else:
            xy = df[[x_col, y_col]].to_numpy(dtype=float)
            block_size = float(
                max(
                    np.ptp(xy[:, 0]) if xy.size else 1.0,
                    np.ptp(xy[:, 1]) if xy.size else 1.0,
                    1.0,
                )
            )

    block_size = float(block_size)
    if not np.isfinite(block_size) or block_size <= 0:
        block_size = 1.0

    blds = df[building_col].unique()
    per_building_rows: dict[str, pd.DataFrame] = {
        b: df[df[building_col] == b].copy() for b in blds
    }

    per_building_blocks: dict[str, np.ndarray] = {}
    for b in blds:
        d = per_building_rows[b]
        cx = d[x_col].to_numpy(dtype=float)
        cy = d[y_col].to_numpy(dtype=float)

        bs = block_size
        labels: np.ndarray | None = None

        for _ in range(max_shrink_iters + 1):
            labels = _grid_block_labels(cx, cy, bs)
            if len(np.unique(labels)) >= max(1, int(min_blocks_per_building)):
                break
            bs = max(bs / 2.0, 1e-9)

        assert labels is not None
        per_building_blocks[b] = labels

    values = np.empty(B, dtype=float)
    for k in range(B):
        out_parts = []

        for b in blds:
            d = per_building_rows[b]
            labels = per_building_blocks[b]
            u = np.unique(labels)
            nb = len(u)

            if nb <= 1:
                nrows = len(d)
                if nrows >= 2:
                    idx = rng.integers(0, nrows, size=nrows)
                    out_parts.append(d.iloc[idx].copy().reset_index(drop=True))
                else:
                    out_parts.append(d.copy().reset_index(drop=True))
                continue

            take = rng.choice(u, size=nb, replace=True)

            parts = []
            for bl in take:
                rows = np.where(labels == bl)[0]
                if rows.size == 0:
                    continue
                parts.append(d.iloc[rows].copy())

            if len(parts) == 0:
                parts = [d.copy()]

            out_parts.append(pd.concat(parts, axis=0, ignore_index=True))

        out = pd.concat(out_parts, axis=0, ignore_index=True)
        values[k] = float(fit_fn(out))

    arr = np.asarray(values, dtype=float)
    ci = (float(np.quantile(arr, 0.05)), float(np.quantile(arr, 0.95)))
    return {
        "B": int(B),
        "ci_perc_90": ci,
        "samples": arr,
        "block_size": float(block_size),
    }


def iid_bootstrap_ci_mean(
    df: pd.DataFrame,
    y: str,
    B: int = 800,
    alpha: float = 0.05,
    seed: int = 42,
):
    """Compute an IID bootstrap confidence interval for the mean.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data frame.
    y : str
        Name of the column whose mean is bootstrapped.
    B : int, default=800
        Number of bootstrap replicates.
    alpha : float, default=0.05
        Lower-tail probability used to form the percentile interval.
    seed : int, default=42
        Seed for the random number generator.

    Returns
    -------
    tuple of float
        Lower and upper percentile bootstrap confidence limits for the mean.
    """
    rng = np.random.default_rng(seed)
    yv = df[y].to_numpy(dtype=float)
    n = len(yv)
    boots = np.empty(B, dtype=float)

    for b in range(B):
        idx = rng.integers(0, n, size=n)
        boots[b] = float(np.mean(yv[idx]))

    lo, hi = np.quantile(boots, [alpha, 1 - alpha])
    return (float(min(lo, hi)), float(max(lo, hi)))


def moving_block_bootstrap_ci_mean(
    df: pd.DataFrame,
    y: str,
    time: str,
    B: int = 800,
    alpha: float = 0.05,
    block_len: int = 10,
    seed: int = 42,
):
    """Compute a moving-block bootstrap interval for a temporal mean.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data frame.
    y : str
        Name of the column whose mean is bootstrapped.
    time : str
        Name of the time-ordering column.
    B : int, default=800
        Number of bootstrap replicates.
    alpha : float, default=0.05
        Lower-tail probability used to form the percentile interval.
    block_len : int, default=10
        Length of each moving block.
    seed : int, default=42
        Seed for the random number generator.

    Returns
    -------
    tuple of float
        Lower and upper bootstrap confidence limits for the mean.
    """
    rng = np.random.default_rng(seed)
    df2 = df.sort_values(time).reset_index(drop=True)
    yv = df2[y].to_numpy(dtype=float)
    n = len(yv)
    b = max(int(block_len), 1)
    boots = np.empty(B, dtype=float)

    for k in range(B):
        idx = []
        while len(idx) < n:
            s = int(rng.integers(0, n))
            idx.extend(((s + np.arange(b)) % n).tolist())
        idx = np.asarray(idx[:n], dtype=int)
        boots[k] = float(np.mean(yv[idx]))

    lo, hi = np.quantile(boots, [alpha, 1 - alpha])
    return (float(min(lo, hi)), float(max(lo, hi)))


def spatiotemporal_time_block_bootstrap_ci_mean(
    df: pd.DataFrame,
    y: str,
    time: str,
    B: int = 800,
    alpha: float = 0.05,
    block_len: int = 5,
    seed: int = 42,
    circular: bool = True,
    center: bool = True,
) -> Dict:
    """Bootstrap the mean of a balanced spatiotemporal panel by time blocks.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data frame.
    y : str
        Name of the paired-difference column.
    time : str
        Name of the time-index column.
    B : int, default=800
        Number of bootstrap replicates.
    alpha : float, default=0.05
        Lower-tail probability used to form the percentile interval.
    block_len : int, default=5
        Number of consecutive time slices in each bootstrap block.
    seed : int, default=42
        Seed for the random number generator.
    circular : bool, default=True
        If ``True``, use circular wrapping when a block reaches the end of the
        time axis.
    center : bool, default=True
        If ``True``, center each time-slice group before resampling and add the
        overall mean back afterward.

    Returns
    -------
    dict
        Dictionary containing the number of replicates, the bootstrap samples,
        and percentile, basic, and symmetric confidence intervals.

    Raises
    ------
    ValueError
        If the panel is unbalanced across time slices.
    """
    rng = np.random.default_rng(seed)

    times = np.asarray(sorted(df[time].unique()))
    T = int(len(times))

    if T <= 1:
        yv = df[y].to_numpy(dtype=float)
        m = float(np.mean(yv)) if yv.size else float("nan")
        return {
            "B": int(B),
            "samples": np.asarray([m], dtype=float),
            "ci_perc": (m, m),
            "ci_basic": (m, m),
            "ci_sym": (m, m),
        }

    groups = []
    sizes = []
    for t in times:
        g = df.loc[df[time] == t, y].to_numpy(dtype=float)
        groups.append(g)
        sizes.append(int(len(g)))

    if len(set(sizes)) != 1:
        raise ValueError(
            "spatiotemporal_time_block_bootstrap_ci_mean requires a balanced panel "
            "(constant rows per time slice)."
        )

    block_len = int(max(1, min(int(block_len), T)))
    nblocks = int(math.ceil(T / block_len))

    ybar = float(np.mean(np.concatenate(groups))) if center else 0.0
    groups_c = [g - ybar for g in groups] if center else groups

    boots = np.empty(int(B), dtype=float)
    for b in range(int(B)):
        idx = []
        starts = rng.integers(0, T, size=nblocks)

        for s0 in starts:
            s0 = int(s0)
            if circular:
                idx.extend([(s0 + k) % T for k in range(block_len)])
            else:
                s1 = min(T - block_len, s0)
                idx.extend([s1 + k for k in range(block_len)])

            if len(idx) >= T:
                break

        idx = idx[:T]
        yb = np.concatenate([groups_c[i] for i in idx], axis=0)
        boots[b] = float(np.mean(yb) + (ybar if center else 0.0))

    lo, hi = np.quantile(boots, [alpha, 1.0 - alpha])
    ci_perc = (float(min(lo, hi)), float(max(lo, hi)))

    if center:
        q_lo = float(np.quantile(boots, alpha))
        q_hi = float(np.quantile(boots, 1.0 - alpha))
        ci_basic = (float(2.0 * ybar - q_hi), float(2.0 * ybar - q_lo))
        ci_basic = (float(min(ci_basic)), float(max(ci_basic)))

        dev = np.abs(boots - ybar)
        q = float(np.quantile(dev, 1.0 - alpha))
        ci_sym = (float(ybar - q), float(ybar + q))
        ci_sym = (float(min(ci_sym)), float(max(ci_sym)))
    else:
        ci_basic = ci_perc
        m0 = float(np.mean(boots))
        dev = np.abs(boots - m0)
        q = float(np.quantile(dev, 1.0 - alpha))
        ci_sym = (float(m0 - q), float(m0 + q))
        ci_sym = (float(min(ci_sym)), float(max(ci_sym)))

    return {
        "B": int(B),
        "samples": boots,
        "ci_perc": ci_perc,
        "ci_basic": ci_basic,
        "ci_sym": ci_sym,
    }

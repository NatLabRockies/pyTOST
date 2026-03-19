"""
bootstrap.py
============
Cluster bootstrap CI for validation (percentile CI by default).

References
----------
- Davison & Hinkley (1997) Bootstrap Methods and Their Application.
- Lahiri (2003) Resampling Methods for Dependent Data.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Callable, Dict

def cluster_bootstrap(df: pd.DataFrame, y: str, cluster: str,
                      fit_fn: Callable[[pd.DataFrame], float],
                      B: int = 200, seed: int = 42) -> Dict:
    """
    Resample whole clusters with replacement; compute μ̂ via fit_fn.
    Return percentile 90% CI (for α=0.05 equivalence).
    """
    rng = np.random.default_rng(seed)
    values = []
    groups = df[cluster].unique()
    G = len(groups)
    for _ in range(B):
        take = rng.choice(groups, size=G, replace=True)
        out = pd.concat([df[df[cluster] == g] for g in take], axis=0, ignore_index=True)
        values.append(float(fit_fn(out)))
    arr = np.array(values)
    ci = (np.quantile(arr, 0.05), np.quantile(arr, 0.95))
    return {"B": B, "ci_perc_90": ci, "samples": arr}


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
    """Spatial block bootstrap for paired-difference data.

    Purpose
    -------
    ``cluster_bootstrap`` is appropriate when clusters are independent. When the
    data-generating mechanism includes *cross-building* spatial dependence (e.g.
    a global field defined over building centroids), resampling buildings IID can
    understate uncertainty. This routine performs a simple *grid block bootstrap*
    over building centroids:

    1) Compute a centroid for each building.
    2) Assign each centroid to a grid cell (block).
    3) Resample blocks (cells) with replacement; include all buildings in selected
       blocks, duplicating buildings when blocks repeat.

    Notes
    -----
    - The bootstrap unit is the *block* (not the point, not the building), so the
      resample preserves within-block cross-building dependence.
    - Duplicated buildings are relabeled to avoid conflating multiple draws.

    Parameters
    ----------
    df
        Dataframe containing at least ``building_col``, ``x_col``, ``y_col``, and ``y``.
        This should typically be the paired-difference dataframe (one row per location)
        rather than the long A/B dataframe.
    y
        Outcome column to pass to ``fit_fn``.
    building_col
        Building/cluster identifier.
    x_col, y_col
        Spatial coordinates.
    fit_fn
        Function computing the statistic of interest on a resampled dataframe
        (e.g., ``lambda d: d[y].mean()``).
    B, seed
        Bootstrap size and RNG seed.
    block_size
        Grid cell width/height in coordinate units. If ``None``, a heuristic based on
        the median nearest-neighbor centroid distance is used.
    blocks
        Optional precomputed block labels (indexed like buildings). If provided, we
        skip grid assignment and use these labels.

    Returns
    -------
    dict
        ``{"B": B, "ci_perc_90": (q05, q95), "samples": arr, "block_size": ...}``
    """
    rng = np.random.default_rng(seed)

    # Building centroids
    centers = (
        df.groupby(building_col, sort=False)[[x_col, y_col]]
        .mean()
        .rename(columns={x_col: "cx", y_col: "cy"})
    )
    buildings = centers.index.to_numpy()

    # Block assignment
    if blocks is None:
        cx = centers["cx"].to_numpy()
        cy = centers["cy"].to_numpy()

        if block_size is None:
            # Heuristic: 2x median nearest-neighbor distance among centroids
            # (robust to large domains; avoids tiny blocks).
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

        xmin, ymin = float(cx.min()), float(cy.min())
        ix = np.floor((cx - xmin) / block_size).astype(int)
        iy = np.floor((cy - ymin) / block_size).astype(int)
        blocks = pd.Series(ix.astype(str) + ":" + iy.astype(str), index=centers.index)

    block_labels = blocks.to_numpy()
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

            # Relabel buildings if this block draw repeats
            # (prevents duplicate labels from being treated as the same cluster)
            if j > 0:
                chunk[building_col] = chunk[building_col].astype(str) + f"__bb{b}_{j}"

            out_parts.append(chunk)

        out = pd.concat(out_parts, axis=0, ignore_index=True)
        values.append(float(fit_fn(out)))

    arr = np.asarray(values, dtype=float)
    ci = (float(np.quantile(arr, 0.05)), float(np.quantile(arr, 0.95)))
    return {"B": B, "ci_perc_90": ci, "samples": arr, "block_size": float(block_size) if block_size is not None else None}

def iid_bootstrap_ci_mean(df: pd.DataFrame, y: str, B: int = 800, alpha: float = 0.05, seed: int = 42):
    """IID (rows) bootstrap CI for mean(y)."""
    rng = np.random.default_rng(seed)
    yv = df[y].to_numpy(float)
    n = len(yv)
    boots = np.empty(B, float)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        boots[b] = yv[idx].mean()
    lo, hi = np.quantile(boots, [alpha, 1 - alpha])
    return (float(min(lo, hi)), float(max(lo, hi)))

def moving_block_bootstrap_ci_mean(
    df: pd.DataFrame, y: str, time: str, B: int = 800, alpha: float = 0.05, block_len: int = 10, seed: int = 42
):
    """Moving-block bootstrap CI for mean(y) for serial dependence."""
    rng = np.random.default_rng(seed)
    df2 = df.sort_values(time).reset_index(drop=True)
    yv = df2[y].to_numpy(float)
    n = len(yv)
    b = max(int(block_len), 1)
    boots = np.empty(B, float)
    for k in range(B):
        idx = []
        while len(idx) < n:
            s = int(rng.integers(0, n))
            idx.extend(((s + np.arange(b)) % n).tolist())
        idx = np.asarray(idx[:n], int)
        boots[k] = yv[idx].mean()
    lo, hi = np.quantile(boots, [alpha, 1 - alpha])
    return (float(min(lo, hi)), float(max(lo, hi)))

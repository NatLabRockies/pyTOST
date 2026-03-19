"""
Synthetic data generators for TOST pipelines.

This module creates paired samples (arm A vs arm B) under several dependence structures
commonly encountered in practice:

1) IID, no groupings
2) IID with groupings (labels present; no correlation) and optional random effects
3) Spatially dependent clusters (within-cluster spatial correlation)
4) Temporally dependent series (AR(1) correlation)
5) Spatio-temporal process (separable space × time covariance)

Each generator returns a tidy pandas.DataFrame with columns:
    - sample_id : int index within a scenario
    - arm       : "A" or "B"
    - y         : observed response
    - mu        : latent mean (pre-noise, post-arm effect)
    - baseline  : baseline latent mean (shared before arm effect)
    - effect    : arm-specific shift applied to baseline (A: 0; B: delta)
    - group_id  : optional grouping label
    - x, y_sp   : spatial coordinates (if applicable)
    - t         : integer time index (if applicable)

Design goals
------------
- Deterministic reproducibility via an explicit `rng` (numpy Generator) or `seed`.
- Explicit control of the *true* between-arm effect size (constant or function of space/time/group).
- Control of marginal variance and dependence strength via interpretable params.
- Return metadata with the ground-truth settings for testing and regression checks.

Examples
--------
>>> from synthetic_tost_data import (
...     generate_iid, generate_iid_grouped, generate_spatial_clusters,
...     generate_temporal_ar1, generate_spatiotemporal
... )
>>> df, meta = generate_iid(n=1000, delta=0.2, sigma=1.0, seed=42)
>>> df.head()

Notes
-----
- All functions create *paired* samples for arms A and B under the same latent process.
- You can pass a callable `delta_fn` to create spatially/temporally varying treatment effects.
- For numerical stability, covariances include a tiny jitter (1e-8) on the diagonal.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import math

Array = np.ndarray


# -----------------------
# Utility covariance builders
# -----------------------

def rbf_cov(coords: Array, length_scale: float, variance: float = 1.0) -> Array:
    """Build a squared-exponential (RBF) covariance matrix.

    What it does
    ------------
    For coordinates ``x_1, …, x_n`` in ℝ^d, returns the matrix
    ``K[i,j] = variance * exp(-0.5 * ||x_i - x_j||^2 / length_scale^2)`` with a small
    diagonal jitter for numerical stability.

    Parameters
    ----------
    coords : array-like of shape (n, d)
        Coordinates of points.
    length_scale : float
        Positive length-scale; larger values imply smoother/long-range correlation.
    variance : float, default 1.0
        Marginal variance of the process.

    Returns
    -------
    K : ndarray of shape (n, n)
        Symmetric positive-definite covariance matrix.
    """
    coords = np.atleast_2d(coords)
    dists2 = np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=2)
    K = variance * np.exp(-0.5 * dists2 / (length_scale ** 2))
    np.fill_diagonal(K, K.diagonal() + 1e-8)
    return K


def ar1_cov(n: int, rho: float, variance: float = 1.0) -> Array:
    """Construct an AR(1) covariance matrix of size ``n``.

    What it does
    ------------
    Returns ``K[i,j] = variance * rho**|i-j|`` for a stationary AR(1) process, with a
    small diagonal jitter for stability.

    Parameters
    ----------
    n : int
        Length of the time series.
    rho : float
        Autocorrelation parameter (|rho| < 1 recommended).
    variance : float, default 1.0
        Marginal variance of the process.

    Returns
    -------
    K : ndarray of shape (n, n)
        AR(1) covariance matrix.
    """
    idx = np.arange(n)
    d = np.abs(idx[:, None] - idx[None, :])
    K = variance * (rho ** d)
    np.fill_diagonal(K, K.diagonal() + 1e-8)
    return K


def safe_cholesky(K: Array) -> Array:
    """Numerically robust Cholesky factorization.

    What it does
    ------------
    Attempts ``np.linalg.cholesky(K)``; on failure, increases diagonal jitter and retries.
    As a last resort, clamps negative eigenvalues to a minimum (1e-10) before factoring.

    Parameters
    ----------
    K : ndarray (n, n)
        Symmetric (near) positive-definite matrix.

    Returns
    -------
    L : ndarray (n, n)
        Lower-triangular factor such that ``L @ L.T ≈ K``.
    """
    jitter = 0.0
    for _ in range(6):
        try:
            return np.linalg.cholesky(K + jitter * np.eye(K.shape[0]))
        except np.linalg.LinAlgError:
            jitter = 10.0 * (jitter + 1e-10)
    # Last resort: eigen clamp
    w, V = np.linalg.eigh(K)
    w_clamped = np.clip(w, a_min=1e-10, a_max=None)
    Kp = (V * w_clamped) @ V.T
    return np.linalg.cholesky(Kp)


# -----------------------
# Core sampler (paired arms)
# -----------------------

@dataclass
class EffectSpec:
    """Specification of the true between-arm effect.

    If `delta_fn` is provided, it overrides `delta` and returns an effect for each observation.
    The effect is applied only to arm B; arm A gets effect 0.
    """
    delta: float = 0.0
    delta_fn: Optional[Callable[[pd.DataFrame], Array]] = None

    def compute(self, df_index: pd.DataFrame) -> Array:
        """Compute the effect vector for an index DataFrame.

        Parameters
        ----------
        df_index : pandas.DataFrame
            Index frame that may include columns used by ``delta_fn`` (e.g., ``x``,
            ``y_sp``, ``t``, ``group_id``). For constant effects, only the row count is used.

        Returns
        -------
        effect : ndarray of shape (len(df_index),)
            Effect applied to arm B at each observation.
        """
        if self.delta_fn is not None:
            out = np.asarray(self.delta_fn(df_index), dtype=float)
            if out.shape[0] != len(df_index):
                raise ValueError("delta_fn must return array length equal to number of rows in index df")
            return out
        return np.full(len(df_index), float(self.delta), dtype=float)

@dataclass
class ClusterEffectSpec:
    """Effect specification for paired B-A mean difference."""
    delta: float = 0.0

    def compute(self, idx_df: pd.DataFrame) -> np.ndarray:
        return np.full(len(idx_df), float(self.delta), dtype=float)


# -----------------------
# 1) IID, no groupings
# -----------------------

def generate_iid(
    n: int,
    delta: float = 0.0,
    sigma: float = 1.0,
    seed: Optional[int] = None,
    effect: Optional[EffectSpec] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """Generate IID paired samples (A vs B) with Gaussian noise.

    Parameters
    ----------
    n : number of observations *per arm*
    delta : constant shift applied to arm B (ignored if `effect` with delta_fn provided)
    sigma : noise standard deviation
    seed : RNG seed for reproducibility
    effect : optional EffectSpec with custom delta_fn
    """
    rng = np.random.default_rng(seed)
    baseline = rng.normal(loc=0.0, scale=1.0, size=n)
    idx_df = pd.DataFrame({"sample_id": np.arange(n)})

    effspec = effect or EffectSpec(delta=delta)
    eff = effspec.compute(idx_df)

    # Build tidy frame
    A = pd.DataFrame({
        "sample_id": np.arange(n),
        "arm": "A",
        "baseline": baseline,
        "effect": 0.0,
    })
    B = pd.DataFrame({
        "sample_id": np.arange(n),
        "arm": "B",
        "baseline": baseline,
        "effect": eff,
    })
    df = pd.concat([A, B], ignore_index=True)
    df["mu"] = df["baseline"] + df["effect"]
    df["y"] = df["mu"] + rng.normal(0.0, sigma, size=len(df))
    meta = {"type": "iid", "n_per_arm": n, "sigma": sigma, "effect": effspec.__dict__}
    return df, meta


# -----------------------
# 2) IID with groupings (labels; optional random effects)
# -----------------------

def generate_iid_grouped(
    n_groups: int,
    n_per_group: int,
    delta: float = 0.0,
    sigma: float = 1.0,
    group_sd: float = 0.0,
    seed: Optional[int] = None,
    effect: Optional[EffectSpec] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """IID within observations but with group labels and optional group random intercepts.

    Parameters
    ----------
    n_groups : number of groups (clusters)
    n_per_group : observations per group *per arm*
    delta : constant arm-B shift
    sigma : iid noise SD
    group_sd : SD of random intercept per group (0 disables random effects)
    seed : RNG seed
    effect : optional EffectSpec for custom per-observation delta
    """
    rng = np.random.default_rng(seed)
    group_ids = np.repeat(np.arange(n_groups), n_per_group)
    n = n_groups * n_per_group
    # Random intercepts (shared by both arms)
    b_g = rng.normal(0.0, group_sd, size=n_groups) if group_sd > 0 else np.zeros(n_groups)
    baseline = rng.normal(0.0, 1.0, size=n) + b_g[group_ids]

    idx_df = pd.DataFrame({"sample_id": np.arange(n), "group_id": group_ids})
    effspec = effect or EffectSpec(delta=delta)
    eff = effspec.compute(idx_df)

    A = pd.DataFrame({
        "sample_id": np.arange(n),
        "arm": "A",
        "group_id": group_ids,
        "baseline": baseline,
        "effect": 0.0,
    })
    B = pd.DataFrame({
        "sample_id": np.arange(n),
        "arm": "B",
        "group_id": group_ids,
        "baseline": baseline,
        "effect": eff,
    })
    df = pd.concat([A, B], ignore_index=True)
    df["mu"] = df["baseline"] + df["effect"]
    df["y"] = df["mu"] + rng.normal(0.0, sigma, size=len(df))
    meta = {
        "type": "iid_grouped",
        "n_groups": n_groups,
        "n_per_group": n_per_group,
        "sigma": sigma,
        "group_sd": group_sd,
        "effect": effspec.__dict__,
    }
    return df, meta

def generate_cluster_groups(
    n_groups: int,
    points_per_group: int,
    *,
    # Mean (true paired difference)
    delta: float = 0.0,
    seed: Optional[int] = None,
    effect: Optional[ClusterEffectSpec] = None,
    # Noise (row-level)
    nugget_sd: float = 0.1,
    # Shared baseline that cancels in diff (kept for completeness)
    baseline_sd: float = 1.0,
    baseline_global: bool = True,
    # Cluster-level arm-specific measurement structure (induces cluster dependence in diff)
    meas_group_sd: float = 0.0,
    meas_shared: float = 0.0,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Generate paired A/B observations with group-level dependence in the paired difference.

    Parameters
    ----------
    n_groups, points_per_group
        Number of clusters (groups/clusters) and points per group.
    delta
        True mean paired difference for arm B (ignored if `effect` provided).
    seed
        RNG seed.
    effect
        Optional ClusterEffectSpec. Defaults to constant delta.
    nugget_sd
        Row-level iid noise SD added separately to arm A and arm B.
    baseline_sd
        SD of a baseline term shared across arms and therefore cancels in `diff`.
    baseline_global
        If True, a single baseline value per sample (shared across arms).
        If False, baseline is still per-sample but drawn independently by group; since it cancels
        in the difference, this mostly affects the raw y but not diff.
    meas_group_sd
        SD of the *group-level* arm-specific measurement effect. Set >0 to induce dependence in diff.
    meas_shared
        Correlation between arm-specific group effects in A and B (in [0,1]).
        1.0 means identical group effects (cancels in diff); 0.0 means independent (max dependence in diff).

    Returns
    -------
    df_long, meta
        Long dataframe with arms A/B and metadata.
    """
    rng = np.random.default_rng(seed)
    n_groups = int(n_groups)
    points_per_group = int(points_per_group)
    total = n_groups * points_per_group

    group_id = np.repeat(np.arange(n_groups), points_per_group)
    sample_id = np.arange(total)

    # Simple synthetic coordinates for compatibility (not used by cluster engines)
    # Place groups on a line; jitter within group.
    x_centers = np.linspace(0.0, 1.0, n_groups)
    x = x_centers[group_id] + rng.normal(scale=0.01, size=total)
    y_sp = rng.normal(scale=0.01, size=total)

    # Baseline (shared across arms, cancels in diff)
    baseline = rng.normal(0.0, baseline_sd, size=total)

    # Effect surface
    idx_df = pd.DataFrame({"sample_id": sample_id, "group_id": group_id, "x": x, "y_sp": y_sp})
    effspec = effect or ClusterEffectSpec(delta=float(delta))
    eff = effspec.compute(idx_df)

    # Group-level arm-specific effects
    meas_shared = float(np.clip(meas_shared, 0.0, 1.0))
    eta_A = np.zeros(total, dtype=float)
    eta_B = np.zeros(total, dtype=float)
    if meas_group_sd > 0.0:
        u = rng.normal(0.0, meas_group_sd, size=n_groups)
        v = rng.normal(0.0, meas_group_sd, size=n_groups)
        etaA_g = u
        etaB_g = meas_shared * u + np.sqrt(max(0.0, 1.0 - meas_shared**2)) * v
        eta_A = etaA_g[group_id]
        eta_B = etaB_g[group_id]

    # Build long dataframe
    A = pd.DataFrame({
        "sample_id": sample_id,
        "arm": "A",
        "group_id": group_id,
        "x": x,
        "y_sp": y_sp,
        "baseline": baseline,
        "effect": 0.0,
        "meas_group": eta_A,
    })
    B = pd.DataFrame({
        "sample_id": sample_id,
        "arm": "B",
        "group_id": group_id,
        "x": x,
        "y_sp": y_sp,
        "baseline": baseline,
        "effect": eff,
        "meas_group": eta_B,
    })
    df = pd.concat([A, B], ignore_index=True)

    df["mu"] = df["baseline"] + df["effect"]
    df["y"] = df["mu"] + df["meas_group"] + rng.normal(0.0, nugget_sd, size=len(df))

    meta = {
        "type": "cluster_groups",
        "n_groups": n_groups,
        "points_per_group": points_per_group,
        "delta": float(delta),
        "nugget_sd": float(nugget_sd),
        "baseline_sd": float(baseline_sd),
        "baseline_global": bool(baseline_global),
        "meas_group_sd": float(meas_group_sd),
        "meas_shared": float(meas_shared),
        "effect": effspec.__dict__,
    }
    return df, meta


# -----------------------
# 3) Spatially dependent clusters
# -----------------------

def generate_spatial_clusters(
    n_clusters: int,
    points_per_cluster: int,
    cluster_radius: float = 1.0,
    length_scale: float = 0.5,
    field_sd: float = 1.0,
    nugget_sd: float = 0.1,
    delta: float = 0.0,
    seed: Optional[int] = None,
    effect: Optional[EffectSpec] = None,
    *,
    # --- Option A: arm-specific spatial measurement error (does NOT cancel in B-A)
    meas_field_sd: float = 0.0,
    meas_length_scale: Optional[float] = None,
    meas_shared: float = 1.0,
    # --- Option B: allow cross-cluster spatial dependence via global fields
    baseline_global: bool = False,
    meas_global: bool = False,
    # --- center-level global measurement field to induce cross-cluster dependence
    center_global_meas_field: bool = False,
    cluster_center_sd: float = 0.0,
    cluster_center_length_scale: Optional[float] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """Generate spatially dependent paired A/B data with optional arm-specific spatial error.

    This generator creates paired observations (arm A vs arm B) at the *same* spatial
    locations. The latent spatial baseline is shared across arms, which is appropriate
    for paired designs but causes the baseline to cancel in the paired difference
    ``diff = y_B - y_A``.

    To support stress-testing *spatially aware* inference on the paired difference,
    this function implements:

    Option A (arm-specific spatial measurement error)
        Adds an arm-specific spatial process term that does **not** cancel in the paired
        difference:
        ``y_A = baseline + effect_A + eta_A(x) + eps_A``
        ``y_B = baseline + effect_B + eta_B(x) + eps_B``
        so ``diff = (effect_B-effect_A) + (eta_B-eta_A) + (eps_B-eps_A)`` includes
        spatial dependence via ``eta_B-eta_A`` whenever ``meas_shared < 1``.

    Option B (cross-cluster spatial dependence)
        Allows either the baseline field and/or the arm-specific measurement fields to
        be generated as *global* spatial processes over all points (not independent by
        cluster). This produces spatial correlation **across** clusters.

    Center-level global measurement field (new)
        When ``center_global_meas_field=True`` and ``cluster_center_sd>0``, we generate
        an additional *cluster-level* latent measurement component as a GP over cluster
        centroids, then assign that value to all points within a cluster. If this component
        does not cancel between A/B (controlled by ``meas_shared``), then:
          - IID inference can be severely overconfident (many points are not independent).
          - Cluster-robust inference can still be overconfident (clusters are not independent).
          - A spatial GLS model can recover the correct uncertainty by modeling cross-cluster
            covariance.

    Returns
    -------
    df, meta
        Long dataframe with A/B rows and a metadata dictionary.
    """
    print(f'seed in generate_spatial_clusters: {seed}')
    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Coordinates (clusters placed in a shared coordinate system)
    # ------------------------------------------------------------------
    total_points = n_clusters * points_per_cluster
    cluster_ids = np.repeat(np.arange(n_clusters), points_per_cluster)

    # Cluster centroids and within-cluster jitter
    centroids = rng.uniform(-5.0, 5.0, size=(n_clusters, 2))
    jitter = rng.normal(scale=cluster_radius, size=(total_points, 2))
    coords = centroids[cluster_ids] + jitter

    # ------------------------------------------------------------------
    # Baseline latent field (shared across arms; cancels in diff)
    # ------------------------------------------------------------------
    baseline = np.zeros(total_points, dtype=float)

    if baseline_global:
        K = rbf_cov(coords, length_scale=length_scale, variance=field_sd ** 2)
        L = safe_cholesky(K)
        baseline = L @ rng.normal(size=total_points)
    else:
        for g in range(n_clusters):
            idx = np.where(cluster_ids == g)[0]
            K = rbf_cov(coords[idx, :], length_scale=length_scale, variance=field_sd ** 2)
            L = safe_cholesky(K)
            baseline[idx] = L @ rng.normal(size=len(idx))

    # ------------------------------------------------------------------
    # Effect surface (can be constant delta or spatially varying via EffectSpec)
    # ------------------------------------------------------------------
    idx_df = pd.DataFrame({
        "sample_id": np.arange(total_points),
        "group_id": cluster_ids,
        "x": coords[:, 0],
        "y_sp": coords[:, 1],
    })
    effspec = effect or EffectSpec(delta=delta)
    eff = effspec.compute(idx_df)

    # ------------------------------------------------------------------
    # Option A: arm-specific spatial measurement error fields (point-level)
    # ------------------------------------------------------------------
    meas_ls = float(meas_length_scale) if meas_length_scale is not None else float(length_scale)
    meas_shared = float(np.clip(meas_shared, 0.0, 1.0))

    meas_A = np.zeros(total_points, dtype=float)
    meas_B = np.zeros(total_points, dtype=float)

    if meas_field_sd > 0.0:
        if meas_global:
            K = rbf_cov(coords, length_scale=meas_ls, variance=meas_field_sd ** 2)
            L = safe_cholesky(K)
            u = L @ rng.normal(size=total_points)
            v = L @ rng.normal(size=total_points)
            meas_A = u
            meas_B = meas_shared * u + np.sqrt(max(0.0, 1.0 - meas_shared ** 2)) * v
        else:
            for g in range(n_clusters):
                idx = np.where(cluster_ids == g)[0]
                K = rbf_cov(coords[idx, :], length_scale=meas_ls, variance=meas_field_sd ** 2)
                L = safe_cholesky(K)
                u = L @ rng.normal(size=len(idx))
                v = L @ rng.normal(size=len(idx))
                meas_A[idx] = u
                meas_B[idx] = meas_shared * u + np.sqrt(max(0.0, 1.0 - meas_shared ** 2)) * v

    # ------------------------------------------------------------------
    # Centroid-level global measurement component (constant within each cluster)
    # ------------------------------------------------------------------
    cen_A = np.zeros(n_clusters, dtype=float)
    cen_B = np.zeros(n_clusters, dtype=float)
    if center_global_meas_field and (cluster_center_sd > 0.0):
        cen_ls = float(cluster_center_length_scale) if cluster_center_length_scale is not None else float(meas_ls)
        Kc = rbf_cov(centroids, length_scale=cen_ls, variance=cluster_center_sd ** 2)
        Lc = safe_cholesky(Kc)
        u = Lc @ rng.normal(size=n_clusters)
        v = Lc @ rng.normal(size=n_clusters)
        cen_A = u
        cen_B = meas_shared * u + np.sqrt(max(0.0, 1.0 - meas_shared ** 2)) * v
        meas_A = meas_A + cen_A[cluster_ids]
        meas_B = meas_B + cen_B[cluster_ids]

    # ------------------------------------------------------------------
    # Build tidy dataframe with paired A/B rows
    # ------------------------------------------------------------------
    A = pd.DataFrame({
        "sample_id": np.arange(total_points),
        "arm": "A",
        "group_id": cluster_ids,
        "x": coords[:, 0],
        "y_sp": coords[:, 1],
        "baseline": baseline,
        "effect": 0.0,
        "meas_sp": meas_A,
        "meas_center": cen_A[cluster_ids] if center_global_meas_field else 0.0,
    })
    B = pd.DataFrame({
        "sample_id": np.arange(total_points),
        "arm": "B",
        "group_id": cluster_ids,
        "x": coords[:, 0],
        "y_sp": coords[:, 1],
        "baseline": baseline,
        "effect": eff,
        "meas_sp": meas_B,
        "meas_center": cen_B[cluster_ids] if center_global_meas_field else 0.0,
    })
    df = pd.concat([A, B], ignore_index=True)

    df["mu"] = df["baseline"] + df["effect"]
    df["y"] = df["mu"] + df["meas_sp"] + rng.normal(0.0, nugget_sd, size=len(df))

    meta = {
        "type": "spatial_clusters",
        "n_clusters": n_clusters,
        "points_per_cluster": points_per_cluster,
        "cluster_radius": cluster_radius,
        "length_scale": length_scale,
        "field_sd": field_sd,
        "nugget_sd": nugget_sd,
        "effect": effspec.__dict__,
        "meas_field_sd": meas_field_sd,
        "meas_length_scale": meas_ls,
        "meas_shared": meas_shared,
        "baseline_global": baseline_global,
        "meas_global": meas_global,
        "center_global_meas_field": bool(center_global_meas_field),
        "cluster_center_sd": float(cluster_center_sd),
        "cluster_center_length_scale": float(cluster_center_length_scale) if cluster_center_length_scale is not None else None,
    }
    return df, meta



# -----------------------
# 4) Temporally dependent data (AR1)
# -----------------------

def generate_temporal_ar1(
    n_time: int,
    series_per_arm: int = 1,
    rho: float = 0.7,
    process_sd: float = 1.0,
    obs_sd: float = 0.1,
    delta: float = 0.0,
    seed: Optional[int] = None,
    effect: Optional[EffectSpec] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """Generate AR(1) time series for each arm (optionally multiple parallel series).

    Parameters
    ----------
    n_time : length of the series
    series_per_arm : how many independent series per arm (e.g., subjects)
    rho : AR(1) correlation (|rho|<1)
    process_sd : SD of the latent AR(1) process
    obs_sd : iid observation noise SD
    delta : constant shift for arm B (or use `effect` for time-varying)
    seed : RNG seed
    effect : optional EffectSpec; can depend on time index `t`
    """
    rng = np.random.default_rng(seed)
    Kt = ar1_cov(n_time, rho=rho, variance=process_sd ** 2)
    Lt = safe_cholesky(Kt)

    total_series = series_per_arm
    # Draw baseline latent process for each series (shared across arms to enable paired inference)
    baseline_all = Lt @ rng.normal(size=(n_time, total_series))  # shape (n_time, series)

    # Build index for effect function
    idx_df = pd.DataFrame({
        "sample_id": np.arange(n_time * total_series),
        "t": np.tile(np.arange(n_time), total_series),
        "series_id": np.repeat(np.arange(total_series), n_time),
    })
    effspec = effect or EffectSpec(delta=delta)
    eff_vec = effspec.compute(idx_df)
    eff_mat = eff_vec.reshape(n_time, total_series)

    # Construct tidy output
    records = []
    for s in range(total_series):
        baseline = baseline_all[:, s]
        for arm in ("A", "B"):
            eff = 0.0 if arm == "A" else eff_mat[:, s]
            mu = baseline + eff
            y = mu + rng.normal(0.0, obs_sd, size=n_time)
            for t in range(n_time):
                records.append({
                    "sample_id": s * n_time + t,
                    "arm": arm,
                    "series_id": s,
                    "t": t,
                    "baseline": baseline[t],
                    "effect": eff[t] if isinstance(eff, np.ndarray) else eff,
                    "mu": mu[t],
                    "y": y[t],
                })
    df = pd.DataFrame.from_records(records)
    meta = {
        "type": "temporal_ar1",
        "n_time": n_time,
        "series_per_arm": series_per_arm,
        "rho": rho,
        "process_sd": process_sd,
        "obs_sd": obs_sd,
        "effect": effspec.__dict__,
    }
    return df, meta


# -----------------------
# 5) Spatio-temporal (separable GP × AR1)
# -----------------------


def generate_spatiotemporal(
    n_space: int,
    n_time: int,
    length_scale: float = 0.7,
    rho: float = 0.6,
    spatial_sd: float = 1.0,
    obs_sd: float = 0.1,
    domain: Tuple[float, float, float, float] = (-2, 2, -2, 2),
    delta: float = 0.0,
    seed: Optional[int] = None,
    effect: Optional[EffectSpec] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """Generate a separable spatio-temporal process for paired A/B with *non-canceling* dependence in the difference.

    Overview
    --------
    This generator creates paired observations at the same (space, time) index for arms A and B.

    A key requirement for stress-testing dependence-aware TOST engines is that the paired
    difference ``diff = y_B - y_A`` exhibits *dependence* (spatial, temporal, or spatio-temporal).
    If the entire latent field is shared across arms, it cancels in ``diff`` and the paired
    differences become iid (up to observation noise), making it impossible to construct
    examples where dependence-aware methods disagree.

    To address this, we keep a shared baseline field (paired design realism) but also add an
    **arm-specific spatio-temporal measurement field** that does *not* cancel in the paired
    difference unless the two arms are perfectly correlated.

    Model
    -----
    Let ``b(s,t)`` be a shared baseline GP×AR(1) field and ``η_A(s,t)``, ``η_B(s,t)`` be arm-specific
    spatio-temporal measurement fields with the same separable covariance structure.

        y_A(s,t) = b(s,t) + 0 + η_A(s,t) + ε_A(s,t)
        y_B(s,t) = b(s,t) + δ(s,t) + η_B(s,t) + ε_B(s,t)

    Then

        diff(s,t) = δ(s,t) + (η_B(s,t) - η_A(s,t)) + (ε_B(s,t) - ε_A(s,t)).

    Arm-to-arm correlation in the measurement field
    ------------------------------------------------
    The correlation between ``η_A`` and ``η_B`` is not exposed as an extra parameter (signature
    must remain unchanged). Instead, we derive a deterministic value from ``rho``:

        meas_shared = clip(1 - 0.9*|rho|, 0.05, 0.95)

    so that high temporal persistence (large |rho|) tends to yield lower cross-arm sharing and
    therefore stronger dependence in the paired differences.

    Parameters
    ----------
    n_space, n_time, length_scale, rho, spatial_sd, obs_sd, domain, delta, seed, effect
        Same meaning as before. Note that ``spatial_sd`` now governs the marginal SD of both
        the shared baseline and the arm-specific measurement field.

    Returns
    -------
    df, meta
        Long dataframe with A/B rows and metadata.
    """
    rng = np.random.default_rng(seed)

    # Spatial coordinates
    xmin, xmax, ymin, ymax = domain
    coords = np.column_stack([
        rng.uniform(xmin, xmax, size=n_space),
        rng.uniform(ymin, ymax, size=n_space),
    ])

    # Separable covariance factors
    Ks = rbf_cov(coords, length_scale=length_scale, variance=spatial_sd ** 2)
    Ls = safe_cholesky(Ks)
    Lt = safe_cholesky(ar1_cov(n_time, rho=rho, variance=1.0))

    # Shared baseline spatio-temporal field (paired; cancels in diff)
    Z0 = rng.normal(size=(n_space, n_time))
    baseline_mat = Ls @ Z0 @ Lt.T  # shape (n_space, n_time)

    # Effect function index
    idx_df = pd.DataFrame({
        "sample_id": np.arange(n_space * n_time),
        "x": np.repeat(coords[:, 0], n_time),
        "y_sp": np.repeat(coords[:, 1], n_time),
        "t": np.tile(np.arange(n_time), n_space),
    })
    effspec = effect or EffectSpec(delta=delta)
    eff_vec = effspec.compute(idx_df)
    eff_mat = eff_vec.reshape(n_space, n_time)

    # Arm-specific spatio-temporal measurement fields (do NOT cancel in diff unless perfectly shared)
    meas_shared = float(np.clip(1.0 - 0.9 * abs(float(rho)), 0.05, 0.95))
    Z1 = rng.normal(size=(n_space, n_time))
    Z2 = rng.normal(size=(n_space, n_time))
    eta_A = Ls @ Z1 @ Lt.T
    eta_ind = Ls @ Z2 @ Lt.T
    eta_B = meas_shared * eta_A + math.sqrt(max(0.0, 1.0 - meas_shared ** 2)) * eta_ind

    # Build tidy frame
    records = []
    for i in range(n_space):
        x_i, y_i = coords[i]
        for t in range(n_time):
            base = baseline_mat[i, t]
            eA = eta_A[i, t]
            eB = eta_B[i, t]
            sid = i * n_time + t

            muA = base
            muB = base + eff_mat[i, t]

            yA = muA + eA + rng.normal(0.0, obs_sd)
            yB = muB + eB + rng.normal(0.0, obs_sd)

            records.append({
                "sample_id": sid, "arm": "A", "x": x_i, "y_sp": y_i, "t": t,
                "baseline": base, "effect": 0.0, "mu": muA, "y": yA,
                "meas_st": eA, "meas_shared": meas_shared,
            })
            records.append({
                "sample_id": sid, "arm": "B", "x": x_i, "y_sp": y_i, "t": t,
                "baseline": base, "effect": eff_mat[i, t], "mu": muB, "y": yB,
                "meas_st": eB, "meas_shared": meas_shared,
            })

    df = pd.DataFrame.from_records(records)

    meta = {
        "type": "spatiotemporal",
        "n_space": n_space,
        "n_time": n_time,
        "length_scale": float(length_scale),
        "rho": float(rho),
        "spatial_sd": float(spatial_sd),
        "obs_sd": float(obs_sd),
        "domain": domain,
        "meas_shared": meas_shared,
        "effect": effspec.__dict__,
        "notes": "Includes arm-specific spatio-temporal measurement field so diff exhibits dependence.",
    }
    return df, meta

# -----------------------
# Convenience effect helpers
# -----------------------

def step_effect(delta_small: float, delta_large: float, threshold: float = 0.0) -> EffectSpec:
    """Piecewise-constant effect based on the ``x`` coordinate.

    Parameters
    ----------
    delta_small : float
        Effect when ``x < threshold``.
    delta_large : float
        Effect when ``x ≥ threshold``.
    threshold : float, default 0.0
        Split point along the x-axis.

    Returns
    -------
    EffectSpec
        Use as ``effect=`` in generators that include an ``x`` column.
    """
    def _fn(df: pd.DataFrame) -> Array:
        x = df.get("x")
        if x is None:
            raise ValueError("step_effect requires 'x' column in index df")
        return np.where(x.values >= threshold, delta_large, delta_small)
    return EffectSpec(delta_fn=_fn)


def radial_decay_effect(delta0: float, center: Tuple[float, float] = (0.0, 0.0), scale: float = 1.0) -> EffectSpec:
    """Radially decaying effect from a spatial center.

    Parameters
    ----------
    delta0 : float
        Peak effect at the center.
    center : tuple of floats, default (0.0, 0.0)
        (x, y) center of the decay.
    scale : float, default 1.0
        Length-scale of the radial decay.

    Returns
    -------
    EffectSpec
        Use as ``effect=`` in spatial/spatio-temporal generators.
    """
    def _fn(df: pd.DataFrame) -> Array:
        x = df.get("x")
        y = df.get("y_sp")
        if (x is None) or (y is None):
            raise ValueError("radial_decay_effect requires 'x' and 'y_sp' columns")
        r2 = (x.values - center[0]) ** 2 + (y.values - center[1]) ** 2
        return delta0 * np.exp(-0.5 * r2 / (scale ** 2))
    return EffectSpec(delta_fn=_fn)


def seasonal_effect(amplitude: float, period: float) -> EffectSpec:
    """Sinusoidal effect over the integer time index ``t``.

    Parameters
    ----------
    amplitude : float
        Peak-to-baseline amplitude of the sinusoid.
    period : float
        Period of the cycle in time steps.

    Returns
    -------
    EffectSpec
        Use as ``effect=`` in temporal/spatio-temporal generators.
    """
    def _fn(df: pd.DataFrame) -> Array:
        t = df.get("t")
        if t is None:
            raise ValueError("seasonal_effect requires 't' column")
        return amplitude * np.sin(2 * np.pi * t.values / period)
    return EffectSpec(delta_fn=_fn)


# -----------------------
# CLI: write scenario CSVs + manifest for pipeline tests
# -----------------------
#import argparse, json, os
#from pathlib import Path
#
#
#def write_scenarios(
#    out_dir: str,
#    # IID
#    iid_n: int = 1000,
#    iid_sigma: float = 0.5,
#    iid_delta: float = 0.2,
#    iid_seed: int = 101,
#    # Grouped
#    grp_groups: int = 10,
#    grp_per_group: int = 50,
#    grp_sigma: float = 0.3,
#    grp_group_sd: float = 0.2,
#    grp_delta: float = 0.1,
#    grp_seed: int = 202,
#    # Spatial clusters
#    sp_clusters: int = 6,
#    sp_points_per: int = 40,
#    sp_radius: float = 0.8,
#    sp_len: float = 0.6,
#    sp_field_sd: float = 1.0,
#    sp_nugget_sd: float = 0.1,
#    sp_delta: float = 0.0,
#    sp_seed: int = 303,
#    # Temporal AR(1)
#    tm_len: int = 120,
#    tm_series: int = 3,
#    tm_rho: float = 0.8,
#    tm_proc_sd: float = 1.0,
#    tm_obs_sd: float = 0.1,
#    tm_delta: float = 0.0,
#    tm_seed: int = 404,
#    # Spatio-temporal
#    st_space: int = 80,
#    st_time: int = 24,
#    st_len: float = 0.7,
#    st_rho: float = 0.7,
#    st_spatial_sd: float = 1.0,
#    st_obs_sd: float = 0.1,
#    st_delta: float = 0.0,
#    st_seed: int = 505,
#    # Output options
#    prefix: str = "scenario_",
#    include_examples: bool = True,
#) -> Dict:
#    """Generate all scenarios, write CSVs, and emit a manifest JSON.
#
#    What it does
#    ------------
#    Calls each generator with the provided parameters, writes a CSV per scenario, and
#    compiles a ``manifest.json`` summarizing row counts and metadata.
#
#    Parameters
#    ----------
#    out_dir : str
#        Directory to create/overwrite with outputs.
#    iid_* / grp_* / sp_* / tm_* / st_* : various
#        Parameters forwarded to each scenario generator (see function docstrings).
#    prefix : str, default "scenario_"
#        Filename prefix for all outputs.
#    include_examples : bool, default True
#        If True, uses illustrative varying effects (radial/seasonal). If False, uses
#        constant deltas provided in ``*_delta``.
#
#    Returns
#    -------
#    manifest : dict
#        Dict with ``version``, ``notes``, and a list ``files`` of entries with
#        ``name``, ``rows``, and per-scenario ``meta``.
#    """
#    out_path = Path(out_dir)
#    out_path.mkdir(parents=True, exist_ok=True)
#
#    manifest = {
#        "version": 1,
#        "notes": "Synthetic datasets for TOST pipeline QA",
#        "files": [],
#    }
#
#    # 1) IID
#    df_iid, meta_iid = generate_iid(n=iid_n, delta=iid_delta, sigma=iid_sigma, seed=iid_seed)
#    f_iid = out_path / f"{prefix}iid.csv"
#    df_iid.to_csv(f_iid, index=False)
#    manifest["files"].append({"name": f_iid.name, "rows": len(df_iid), "meta": meta_iid})
#
#    # 2) Grouped IID (with random intercepts)
#    df_grp, meta_grp = generate_iid_grouped(
#        n_groups=grp_groups, n_per_group=grp_per_group, delta=grp_delta,
#        sigma=grp_sigma, group_sd=grp_group_sd, seed=grp_seed,
#    )
#    f_grp = out_path / f"{prefix}iid_grouped.csv"
#    df_grp.to_csv(f_grp, index=False)
#    manifest["files"].append({"name": f_grp.name, "rows": len(df_grp), "meta": meta_grp})
#
#    # 3) Spatial clusters (use a gentle radial effect so tests can detect heterogeneity)
#    eff_sp = radial_decay_effect(0.4, center=(0.0, 0.0), scale=1.2) if include_examples else EffectSpec(delta=sp_delta)
#    df_sp, meta_sp = generate_spatial_clusters(
#        n_clusters=sp_clusters, points_per_cluster=sp_points_per,
#        cluster_radius=sp_radius, length_scale=sp_len, field_sd=sp_field_sd,
#        nugget_sd=sp_nugget_sd, seed=sp_seed, effect=eff_sp,
#    )
#    f_sp = out_path / f"{prefix}spatial_clusters.csv"
#    df_sp.to_csv(f_sp, index=False)
#    manifest["files"].append({"name": f_sp.name, "rows": len(df_sp), "meta": meta_sp})
#
#    # 4) Temporal AR(1) (use seasonal effect example if requested)
#    eff_tm = seasonal_effect(amplitude=0.25, period=12) if include_examples else EffectSpec(delta=tm_delta)
#    df_tm, meta_tm = generate_temporal_ar1(
#        n_time=tm_len, series_per_arm=tm_series, rho=tm_rho,
#        process_sd=tm_proc_sd, obs_sd=tm_obs_sd, seed=tm_seed, effect=eff_tm,
#    )
#    f_tm = out_path / f"{prefix}temporal_ar1.csv"
#    df_tm.to_csv(f_tm, index=False)
#    manifest["files"].append({"name": f_tm.name, "rows": len(df_tm), "meta": meta_tm})
#
#    # 5) Spatio-temporal (seasonal example optional)
#    eff_st = seasonal_effect(amplitude=0.2, period=12) if include_examples else EffectSpec(delta=st_delta)
#    df_st, meta_st = generate_spatiotemporal(
#        n_space=st_space, n_time=st_time, length_scale=st_len, rho=st_rho,
#        spatial_sd=st_spatial_sd, obs_sd=st_obs_sd, seed=st_seed, effect=eff_st,
#    )
#    f_st = out_path / f"{prefix}spatiotemporal.csv"
#    df_st.to_csv(f_st, index=False)
#    manifest["files"].append({"name": f_st.name, "rows": len(df_st), "meta": meta_st})
#
#    # Write manifest
#    man_path = out_path / f"{prefix}manifest.json"
#    with open(man_path, "w", encoding="utf-8") as fp:
#        json.dump(manifest, fp, indent=2)
#
#    return manifest
#
#
#def _build_argparser() -> argparse.ArgumentParser:
#    """Create the command-line interface parser with helpful descriptions and defaults.
#
#    Returns
#    -------
#    argparse.ArgumentParser
#        Parser used by ``main_cli``.
#    """
#    p = argparse.ArgumentParser(
#        description="Write synthetic TOST scenarios to CSV + manifest.json",
#        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
#        epilog=(
#            "All scenarios produce paired A/B samples with a tidy schema. "
#            "Use --no-examples to disable varying effect demos (radial/seasonal)."
#        ),
#    )
#    p.add_argument(
#        "out_dir",
#        type=str,
#        help="Output directory to create (will be made if it does not exist)",
#    )
#    p.add_argument(
#        "--prefix",
#        type=str,
#        default="scenario_",
#        help="Filename prefix for all generated CSVs and the manifest",
#    )
#    p.add_argument(
#        "--no-examples",
#        action="store_true",
#        help=(
#            "Disable example varying effects for spatial/temporal settings; "
#            "use constant delta values instead"
#        ),
#    )
#
#    # IID
#    p.add_argument("--iid-n", type=int, default=1000, help="Observations per arm for IID scenario")
#    p.add_argument("--iid-sigma", type=float, default=0.5, help="IID observation noise SD")
#    p.add_argument("--iid-delta", type=float, default=0.2, help="Arm-B shift (constant) for IID scenario")
#    p.add_argument("--iid-seed", type=int, default=101, help="RNG seed for IID scenario")
#
#    # Grouped
#    p.add_argument("--grp-groups", type=int, default=10, help="Number of groups (clusters) in grouped IID scenario")
#    p.add_argument("--grp-per-group", type=int, default=50, help="Observations per group per arm in grouped IID scenario")
#    p.add_argument("--grp-sigma", type=float, default=0.3, help="IID observation noise SD for grouped IID scenario")
#    p.add_argument("--grp-group-sd", type=float, default=0.2, help="SD of group random intercepts (0 disables random effects)")
#    p.add_argument("--grp-delta", type=float, default=0.1, help="Arm-B shift (constant) for grouped IID scenario")
#    p.add_argument("--grp-seed", type=int, default=202, help="RNG seed for grouped IID scenario")
#
#    # Spatial
#    p.add_argument("--sp-clusters", type=int, default=6, help="Number of spatial clusters")
#    p.add_argument("--sp-points-per", type=int, default=40, help="Points per cluster per arm")
#    p.add_argument("--sp-radius", type=float, default=0.8, help="Cluster radius (SD of points around centroid)")
#    p.add_argument("--sp-len", type=float, default=0.6, help="Within-cluster spatial RBF length-scale")
#    p.add_argument("--sp-field-sd", type=float, default=1.0, help="Marginal SD of spatial latent field")
#    p.add_argument("--sp-nugget-sd", type=float, default=0.1, help="IID observation noise SD (nugget)")
#    p.add_argument("--sp-delta", type=float, default=0.0, help="Arm-B shift (used if --no-examples is set)")
#    p.add_argument("--sp-seed", type=int, default=303, help="RNG seed for spatial clusters scenario")
#
#    # Temporal
#    p.add_argument("--tm-len", type=int, default=120, help="Number of time points per series in AR(1) scenario")
#    p.add_argument("--tm-series", type=int, default=3, help="Number of independent series per arm")
#    p.add_argument("--tm-rho", type=float, default=0.8, help="AR(1) correlation parameter |rho|<1")
#    p.add_argument("--tm-proc-sd", type=float, default=1.0, help="SD of latent AR(1) process")
#    p.add_argument("--tm-obs-sd", type=float, default=0.1, help="IID observation noise SD for AR(1) scenario")
#    p.add_argument("--tm-delta", type=float, default=0.0, help="Arm-B shift (used if --no-examples is set)")
#    p.add_argument("--tm-seed", type=int, default=404, help="RNG seed for temporal AR(1) scenario")
#
#    # Spatio-temporal
#    p.add_argument("--st-space", type=int, default=80, help="Number of spatial locations in spatio-temporal scenario")
#    p.add_argument("--st-time", type=int, default=24, help="Number of time points in spatio-temporal scenario")
#    p.add_argument("--st-len", type=float, default=0.7, help="Spatial RBF length-scale for spatio-temporal scenario")
#    p.add_argument("--st-rho", type=float, default=0.7, help="Temporal AR(1) correlation parameter |rho|<1")
#    p.add_argument("--st-spatial-sd", type=float, default=1.0, help="Marginal SD of spatial component")
#    p.add_argument("--st-obs-sd", type=float, default=0.1, help="IID observation noise SD")
#    p.add_argument("--st-delta", type=float, default=0.0, help="Arm-B shift (used if --no-examples is set)")
#    p.add_argument("--st-seed", type=int, default=505, help="RNG seed for spatio-temporal scenario")
#    return p
#
#
#def main_cli():
#    """Entry point for the CLI.
#
#    Parses arguments, generates all scenarios, writes CSVs and a manifest, and prints
#    the manifest JSON to stdout (useful for smoke tests in CI).
#    """
#    p = _build_argparser()
#    args = p.parse_args()
#    manifest = write_scenarios(
#        out_dir=args.out_dir,
#        iid_n=args.iid_n, iid_sigma=args.iid_sigma, iid_delta=args.iid_delta, iid_seed=args.iid_seed,
#        grp_groups=args.grp_groups, grp_per_group=args.grp_per_group, grp_sigma=args.grp_sigma,
#        grp_group_sd=args.grp_group_sd, grp_delta=args.grp_delta, grp_seed=args.grp_seed,
#        sp_clusters=args.sp_clusters, sp_points_per=args.sp_points_per, sp_radius=args.sp_radius,
#        sp_len=args.sp_len, sp_field_sd=args.sp_field_sd, sp_nugget_sd=args.sp_nugget_sd,
#        sp_delta=args.sp_delta, sp_seed=args.sp_seed,
#        tm_len=args.tm_len, tm_series=args.tm_series, tm_rho=args.tm_rho,
#        tm_proc_sd=args.tm_proc_sd, tm_obs_sd=args.tm_obs_sd, tm_delta=args.tm_delta, tm_seed=args.tm_seed,
#        st_space=args.st_space, st_time=args.st_time, st_len=args.st_len, st_rho=args.st_rho,
#        st_spatial_sd=args.st_spatial_sd, st_obs_sd=args.st_obs_sd, st_delta=args.st_delta, st_seed=args.st_seed,
#        prefix=args.prefix, include_examples=(not args.no_examples),
#    )
#    print(json.dumps(manifest, indent=2))
#
#
#if __name__ == "__main__":
#    main_cli()

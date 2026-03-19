"""Synthetic data generators for dependence-aware TOST examples and tests.

This module provides utilities for constructing paired arm A and arm B datasets
under several dependence structures, including IID, grouped, spatial,
temporal, and spatiotemporal settings. The generators are intended for
benchmarking, regression testing, calibration studies, and worked examples.

Each generator returns a tidy ``pandas.DataFrame`` together with a metadata
``dict`` describing the ground-truth settings used to generate the data.
Common columns include ``sample_id``, ``arm``, ``y``, ``mu``, ``baseline``,
``effect``, and, where applicable, grouping, coordinate, or time-index
columns.
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
    """Construct a squared-exponential covariance matrix.

    Parameters
    ----------
    coords : ndarray of shape (n, d)
        Coordinates for the observation locations.
    length_scale : float
        Positive range parameter controlling correlation decay with distance.
    variance : float, default=1.0
        Marginal variance of the process.

    Returns
    -------
    ndarray of shape (n, n)
        Covariance matrix with entries defined by the radial basis function
        kernel and a small diagonal jitter for numerical stability.
    """
    coords = np.atleast_2d(coords)
    dists2 = np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=2)
    K = variance * np.exp(-0.5 * dists2 / (length_scale ** 2))
    np.fill_diagonal(K, K.diagonal() + 1e-8)
    return K


def ar1_cov(n: int, rho: float, variance: float = 1.0) -> Array:
    """Construct an AR(1) covariance matrix.

    Parameters
    ----------
    n : int
        Number of time points.
    rho : float
        AR(1) correlation parameter.
    variance : float, default=1.0
        Marginal variance of the process.

    Returns
    -------
    ndarray of shape (n, n)
        AR(1) covariance matrix with a small diagonal jitter for numerical
        stability.
    """
    idx = np.arange(n)
    d = np.abs(idx[:, None] - idx[None, :])
    K = variance * (rho ** d)
    np.fill_diagonal(K, K.diagonal() + 1e-8)
    return K


def safe_cholesky(K: Array) -> Array:
    """Compute a numerically stable Cholesky factor.

    Parameters
    ----------
    K : ndarray of shape (n, n)
        Symmetric covariance-like matrix.

    Returns
    -------
    ndarray of shape (n, n)
        Lower-triangular Cholesky factor. Additional diagonal jitter is added
        adaptively if needed, and an eigenvalue-clamped fallback is used as a
        last resort.
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
    """Specification of the true arm B minus arm A effect.

    Parameters
    ----------
    delta : float, default=0.0
        Constant effect applied to arm B when ``delta_fn`` is not provided.
    delta_fn : callable, optional
        Function that maps an index dataframe to an array of per-observation
        effects for arm B. When supplied, this overrides ``delta``.
    """
    delta: float = 0.0
    delta_fn: Optional[Callable[[pd.DataFrame], Array]] = None

    def compute(self, df_index: pd.DataFrame) -> Array:
        """Evaluate the effect specification on an index dataframe.

        Parameters
        ----------
        df_index : pandas.DataFrame
            Index dataframe passed to ``delta_fn`` when a callable effect is used.
            Typical columns include grouping labels, coordinates, or time indices.

        Returns
        -------
        ndarray of shape (len(df_index),)
            Effect values applied to arm B for each indexed observation.
        """
        if self.delta_fn is not None:
            out = np.asarray(self.delta_fn(df_index), dtype=float)
            if out.shape[0] != len(df_index):
                raise ValueError("delta_fn must return array length equal to number of rows in index df")
            return out
        return np.full(len(df_index), float(self.delta), dtype=float)

@dataclass
class ClusterEffectSpec:
    """Constant effect specification for grouped paired data.

    Parameters
    ----------
    delta : float, default=0.0
        Constant arm B minus arm A mean difference.
    """
    delta: float = 0.0

    def compute(self, idx_df: pd.DataFrame) -> np.ndarray:
        """Return a constant effect vector for the provided index dataframe.

        Parameters
        ----------
        idx_df : pandas.DataFrame
            Index dataframe used only for its number of rows.

        Returns
        -------
        ndarray of shape (len(idx_df),)
            Constant effect values equal to ``delta``.
        """
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
    """Generate IID paired arm A and arm B samples.

    Parameters
    ----------
    n : int
        Number of paired observations per arm.
    delta : float, default=0.0
        Constant shift applied to arm B when ``effect`` is not provided.
    sigma : float, default=1.0
        Observation-noise standard deviation.
    seed : int, optional
        Seed for the NumPy random number generator.
    effect : EffectSpec, optional
        Custom effect specification. When supplied, it overrides ``delta``.

    Returns
    -------
    tuple of (pandas.DataFrame, dict)
        Long-format paired dataset and metadata describing the generating
        parameters.
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
    """Generate grouped paired data with IID noise within rows.

    Parameters
    ----------
    n_groups : int
        Number of groups or clusters.
    n_per_group : int
        Number of paired observations per group and per arm.
    delta : float, default=0.0
        Constant shift applied to arm B when ``effect`` is not provided.
    sigma : float, default=1.0
        IID observation-noise standard deviation.
    group_sd : float, default=0.0
        Standard deviation of the shared group random intercept.
    seed : int, optional
        Seed for the NumPy random number generator.
    effect : EffectSpec, optional
        Custom effect specification. When supplied, it overrides ``delta``.

    Returns
    -------
    tuple of (pandas.DataFrame, dict)
        Long-format paired dataset and metadata describing the generating
        parameters.
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
    """Generate grouped paired data with cluster-level dependence.

    Parameters
    ----------
    n_groups : int
        Number of groups or clusters.
    points_per_group : int
        Number of paired observations per group.
    delta : float, default=0.0
        Constant arm B minus arm A mean difference when ``effect`` is not
        provided.
    seed : int, optional
        Seed for the NumPy random number generator.
    effect : ClusterEffectSpec, optional
        Custom effect specification. When supplied, it overrides ``delta``.
    nugget_sd : float, default=0.1
        Row-level IID noise standard deviation added independently to each arm.
    baseline_sd : float, default=1.0
        Standard deviation of the shared baseline term.
    baseline_global : bool, default=True
        Included for interface compatibility with other generators.
    meas_group_sd : float, default=0.0
        Standard deviation of the group-level arm-specific measurement effect.
    meas_shared : float, default=0.0
        Correlation between arm-specific group effects in arm A and arm B.

    Returns
    -------
    tuple of (pandas.DataFrame, dict)
        Long-format paired dataset and metadata describing the generating
        parameters.
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
    # --- center-level global measurement field to induce cross-building dependence
    center_global_meas_field: bool = False,
    building_center_sd: float = 0.0,
    building_center_length_scale: Optional[float] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """Generate paired spatial data with optional within- and cross-cluster dependence.

    Parameters
    ----------
    n_clusters : int
        Number of spatial clusters.
    points_per_cluster : int
        Number of paired observations per cluster.
    cluster_radius : float, default=1.0
        Standard deviation of the within-cluster spatial jitter.
    length_scale : float, default=0.5
        Range parameter for the shared baseline spatial field.
    field_sd : float, default=1.0
        Standard deviation of the shared baseline spatial field.
    nugget_sd : float, default=0.1
        IID observation-noise standard deviation.
    delta : float, default=0.0
        Constant arm B shift when ``effect`` is not provided.
    seed : int, optional
        Seed for the NumPy random number generator.
    effect : EffectSpec, optional
        Custom effect specification. When supplied, it overrides ``delta``.
    meas_field_sd : float, default=0.0
        Standard deviation of the arm-specific spatial measurement field.
    meas_length_scale : float, optional
        Range parameter for the arm-specific spatial measurement field. When
        omitted, ``length_scale`` is used.
    meas_shared : float, default=1.0
        Correlation between the arm-specific spatial measurement fields.
    baseline_global : bool, default=False
        If True, simulate a single baseline field over all points rather than
        independent fields by cluster.
    meas_global : bool, default=False
        If True, simulate the arm-specific measurement fields over all points
        rather than independently by cluster.
    center_global_meas_field : bool, default=False
        If True, add a cluster-centroid measurement component shared within each
        cluster.
    building_center_sd : float, default=0.0
        Standard deviation of the centroid-level measurement component.
    building_center_length_scale : float, optional
        Range parameter for the centroid-level measurement component.

    Returns
    -------
    tuple of (pandas.DataFrame, dict)
        Long-format paired dataset and metadata describing the generating
        parameters.
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
    # Centroid-level global measurement component (constant within each building)
    # ------------------------------------------------------------------
    cen_A = np.zeros(n_clusters, dtype=float)
    cen_B = np.zeros(n_clusters, dtype=float)
    if center_global_meas_field and (building_center_sd > 0.0):
        cen_ls = float(building_center_length_scale) if building_center_length_scale is not None else float(meas_ls)
        Kc = rbf_cov(centroids, length_scale=cen_ls, variance=building_center_sd ** 2)
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
        "building_center_sd": float(building_center_sd),
        "building_center_length_scale": float(building_center_length_scale) if building_center_length_scale is not None else None,
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
    """Generate paired temporal data with AR(1) dependence.

    Parameters
    ----------
    n_time : int
        Number of time points per series.
    series_per_arm : int, default=1
        Number of independent paired series.
    rho : float, default=0.7
        AR(1) correlation parameter.
    process_sd : float, default=1.0
        Standard deviation of the latent AR(1) process.
    obs_sd : float, default=0.1
        IID observation-noise standard deviation.
    delta : float, default=0.0
        Constant arm B shift when ``effect`` is not provided.
    seed : int, optional
        Seed for the NumPy random number generator.
    effect : EffectSpec, optional
        Custom effect specification. When supplied, it overrides ``delta``.

    Returns
    -------
    tuple of (pandas.DataFrame, dict)
        Long-format paired dataset and metadata describing the generating
        parameters.
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
    """Generate paired spatiotemporal data with separable dependence.

    Parameters
    ----------
    n_space : int
        Number of spatial locations.
    n_time : int
        Number of time points.
    length_scale : float, default=0.7
        Spatial range parameter for the separable covariance model.
    rho : float, default=0.6
        Temporal AR(1) correlation parameter.
    spatial_sd : float, default=1.0
        Standard deviation of the shared baseline and arm-specific
        spatiotemporal measurement fields.
    obs_sd : float, default=0.1
        IID observation-noise standard deviation.
    domain : tuple of float, default=(-2, 2, -2, 2)
        Spatial sampling domain as ``(xmin, xmax, ymin, ymax)``.
    delta : float, default=0.0
        Constant arm B shift when ``effect`` is not provided.
    seed : int, optional
        Seed for the NumPy random number generator.
    effect : EffectSpec, optional
        Custom effect specification. When supplied, it overrides ``delta``.

    Returns
    -------
    tuple of (pandas.DataFrame, dict)
        Long-format paired dataset and metadata describing the generating
        parameters.
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
    """Create a step-function effect specification over ``x``.

    Parameters
    ----------
    delta_small : float
        Effect used when ``x < threshold``.
    delta_large : float
        Effect used when ``x >= threshold``.
    threshold : float, default=0.0
        Split point along the x-axis.

    Returns
    -------
    EffectSpec
        Effect specification suitable for generators that expose an ``x``
        column in their index dataframe.
    """
    def _fn(df: pd.DataFrame) -> Array:
        x = df.get("x")
        if x is None:
            raise ValueError("step_effect requires 'x' column in index df")
        return np.where(x.values >= threshold, delta_large, delta_small)
    return EffectSpec(delta_fn=_fn)


def radial_decay_effect(delta0: float, center: Tuple[float, float] = (0.0, 0.0), scale: float = 1.0) -> EffectSpec:
    """Create a radially decaying spatial effect specification.

    Parameters
    ----------
    delta0 : float
        Peak effect at the center.
    center : tuple of float, default=(0.0, 0.0)
        Spatial center of the effect.
    scale : float, default=1.0
        Radial decay scale.

    Returns
    -------
    EffectSpec
        Effect specification suitable for spatial or spatiotemporal
        generators.
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
    """Create a sinusoidal temporal effect specification.

    Parameters
    ----------
    amplitude : float
        Amplitude of the sinusoidal effect.
    period : float
        Period of the cycle in time steps.

    Returns
    -------
    EffectSpec
        Effect specification suitable for temporal or spatiotemporal
        generators.
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

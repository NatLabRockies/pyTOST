"""Numerical cross-checks of the dependence-aware variance computations.

These guard the spatial and spatiotemporal equal-weighted variance shortcuts,
which report Var(xbar) = 1' Sigma_hat 1 / N^2, against independent references:
the Kronecker quadratic-form identity, closed-form Matern correlations, and a
brute-force entrywise reconstruction of the block-diagonal building-aware
covariance. They document that the fast assembly reproduces a dense-matrix
computation to machine precision.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from pyTOST import BuildingAwareSpatioTemporalTOST, BuildingAwareConfig
from pyTOST.engines.spatial_tost import _matern_cov, _pairwise_dists
from pyTOST.engines.spatiotemporal_tost import _ar1_corr


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_kronecker_quadratic_form_identity(seed):
    """1'(A kron B)1 == (1'A1)(1'B1) underlies the ST variance shortcut."""
    rng = np.random.default_rng(seed)
    T, M = 5, 4
    A = rng.standard_normal((T, T)); A = A @ A.T + np.eye(T)
    B = rng.standard_normal((M, M)); B = B @ B.T + np.eye(M)
    K = np.kron(A, B)
    one = np.ones(T * M)
    lhs = one @ (K @ one)
    rhs = (np.ones(T) @ A @ np.ones(T)) * (np.ones(M) @ B @ np.ones(M))
    assert lhs == pytest.approx(rhs, rel=1e-12)


def test_matern_closed_forms():
    d = np.array([[0.0, 0.5, 1.0], [0.5, 0.0, 0.75], [1.0, 0.75, 0.0]])
    rho, s2 = 1.3, 2.0
    # nu = 0.5 is the exponential correlation
    exp_ref = s2 * np.exp(-d / rho)
    assert np.allclose(_matern_cov(d, sigma2=s2, rho=rho, nu=0.5), exp_ref, rtol=1e-10)
    # nu = 1.5 closed form
    r = np.sqrt(3.0) * d / rho
    m32_ref = s2 * (1.0 + r) * np.exp(-r)
    assert np.allclose(_matern_cov(d, sigma2=s2, rho=rho, nu=1.5), m32_ref, rtol=1e-10)


def _balanced_panel(seed=3, buildings=2, M=4, T=6, mu=0.4):
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(buildings):
        xs = rng.uniform(0, 5, M); ys = rng.uniform(0, 5, M)
        for t in range(T):
            for m in range(M):
                rows.append({"diff": mu + rng.normal(0, 0.5), "building_id": b,
                             "x": xs[m], "ycoord": ys[m], "month": t})
    return pd.DataFrame(rows)


def test_building_aware_variance_matches_dense_reconstruction():
    """Equal-weighted Wald half-width equals z*sqrt(1'Sigma1)/N from a dense,
    entrywise-reconstructed block-diagonal covariance (no Kronecker shortcut)."""
    df = _balanced_panel()
    alpha, nu = 0.05, 1.5
    res = BuildingAwareSpatioTemporalTOST(
        y="diff", cluster="building_id", time="month", x="x", ycoord="ycoord",
        config=BuildingAwareConfig(nu=nu)).fit(df, alpha, [1.0])
    r = res.iloc[0]
    sb2, sst2, rho, phi, tau2 = r.sigma_b2, r.sigma_st2, r.rho, r.phi, r.tau2

    # Independent entrywise reconstruction of the full block-diagonal Sigma.
    N = len(df)
    Sigma = np.zeros((N, N))
    recs = []  # (building, t, m-index) per row, in t-major location-minor order per building
    for b, sub in df.groupby("building_id", sort=False):
        locs = sub[["x", "ycoord"]].drop_duplicates().reset_index(drop=True)
        D = _pairwise_dists(locs[["x", "ycoord"]].to_numpy(float))
        Rs = _matern_cov(D, sigma2=1.0, rho=rho, nu=nu)
        times = np.sort(sub["month"].unique())
        Rt = _ar1_corr(T=len(times), phi=phi)
        for ti in range(len(times)):
            for mi in range(len(locs)):
                recs.append((int(b), ti, mi, Rt, Rs))
    for i in range(N):
        bi, ti, mi, Rti, Rsi = recs[i]
        for j in range(N):
            bj, tj, mj, _, _ = recs[j]
            if bi != bj:
                continue
            cov = sb2 + sst2 * Rti[ti, tj] * Rsi[mi, mj]
            if i == j:
                cov += tau2
            Sigma[i, j] = cov
    one = np.ones(N)
    var_xbar = float(one @ (Sigma @ one)) / (N * N)
    z = stats.norm.ppf(1 - alpha)
    hw_ref = z * np.sqrt(var_xbar)
    hw_engine = (r.ci_high - r.ci_low) / 2.0
    assert hw_engine == pytest.approx(hw_ref, rel=1e-9)
    # point estimate is the equal-weighted (balanced) sample mean
    assert r.mu_hat == pytest.approx(df["diff"].mean(), rel=1e-12)

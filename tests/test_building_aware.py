"""Building-aware spatiotemporal engine: same estimand as the other engines
(equal-weighted sample mean), hierarchical dependence-aware variance."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyTOST import BuildingAwareSpatioTemporalTOST, BuildingAwareConfig


def _panel(G=4, M_per=6, T=8, seed=0, icc_b=0.2):
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(G):
        ab = rng.normal(0, np.sqrt(icc_b))
        xs = rng.uniform(0, 5, M_per); ys = rng.uniform(0, 5, M_per)
        for t in range(T):
            for m in range(M_per):
                rows.append({"diff": 0.3 + ab + rng.normal(0, 0.5),
                             "building_id": b, "month": t, "x": xs[m], "ycoord": ys[m]})
    return pd.DataFrame(rows)


def test_building_aware_point_is_sample_mean_and_contains():
    df = _panel(seed=1)
    res = BuildingAwareSpatioTemporalTOST(
        y="diff", cluster="building_id", time="month", x="x", ycoord="ycoord",
        config=BuildingAwareConfig(nu=1.5)).fit(df, 0.05, [1.0])
    r = res.iloc[0]
    assert float(r.mu_hat) == pytest.approx(float(df["diff"].mean()), rel=1e-9)
    assert r.ci_low < r.mu_hat < r.ci_high
    assert bool(r.converged)
    assert "Hierarchical spatiotemporal" in r.method


def test_building_aware_uses_student_t_small_sample_reference():
    """The interval uses a Student-t(G-1) reference, so with G buildings it is
    wider than the corresponding normal-reference interval and reports df=G-1."""
    from scipy import stats
    G = 5
    df = _panel(G=G, seed=3, icc_b=0.3)
    r = BuildingAwareSpatioTemporalTOST(
        y="diff", cluster="building_id", time="month", x="x", ycoord="ycoord").fit(
        df, 0.05, [1.0]).iloc[0]
    assert int(r["df"]) == G - 1
    assert int(r["n_buildings"]) == G
    half = (r.ci_high - r.ci_low) / 2
    t_crit = stats.t.ppf(0.975, G - 1)
    z_crit = stats.norm.ppf(0.975)
    # half-width recovers the reported se times the one-sided t(G-1) critical value
    assert half == pytest.approx(stats.t.ppf(0.95, G - 1) * r["se"], rel=1e-9)
    assert t_crit > z_crit  # Student-t reference is wider than normal


def test_building_aware_wider_than_ignoring_building_variance():
    """With a real building random effect, the building-aware interval should be
    at least as wide as an interval that ignores between-building variance."""
    df = _panel(seed=2, icc_b=0.4)
    res = BuildingAwareSpatioTemporalTOST(
        y="diff", cluster="building_id", time="month", x="x", ycoord="ycoord").fit(df, 0.05, [1.0])
    ba_half = (res["ci_high"].iloc[0] - res["ci_low"].iloc[0]) / 2
    naive_half = 1.645 * df["diff"].std(ddof=1) / np.sqrt(len(df))
    assert ba_half > naive_half


def test_building_aware_incomplete_building_keeps_st_structure():
    """An incomplete building (a missing cell) is handled by building the marginal
    covariance over the observed cells, not by dropping the spatial-temporal
    structure. The fit still runs and the block covariance retains off-diagonal
    spatial-temporal covariance for that building."""
    df = _panel(G=4, M_per=6, T=8, seed=7, icc_b=0.3)
    drop_idx = df.index[(df["building_id"] == 0)][3]
    df2 = df.drop(index=drop_idx).reset_index(drop=True)
    eng = BuildingAwareSpatioTemporalTOST(
        y="diff", cluster="building_id", time="month", x="x", ycoord="ycoord")
    r = eng.fit(df2, 0.05, [1.0]).iloc[0]
    assert bool(r.converged)
    assert r.ci_low < r.mu_hat < r.ci_high
    blocks = eng._blocks(df2)
    b0 = next(b for b in blocks if not b["balanced"])
    assert b0["n"] == 6 * 8 - 1
    K = eng._block_cov(b0, 0.05, float(r.sigma_st2), float(r.rho), float(r.phi),
                       float(r.tau2))
    off = K - np.diag(np.diag(K))
    assert np.any(off > 0.05 + 1e-8)

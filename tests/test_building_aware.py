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
    assert "Building-aware" in r.method


def test_building_aware_wider_than_ignoring_building_variance():
    """With a real building random effect, the building-aware interval should be
    at least as wide as an interval that ignores between-building variance."""
    df = _panel(seed=2, icc_b=0.4)
    res = BuildingAwareSpatioTemporalTOST(
        y="diff", cluster="building_id", time="month", x="x", ycoord="ycoord").fit(df, 0.05, [1.0])
    ba_half = (res["ci_high"].iloc[0] - res["ci_low"].iloc[0]) / 2
    naive_half = 1.645 * df["diff"].std(ddof=1) / np.sqrt(len(df))
    assert ba_half > naive_half

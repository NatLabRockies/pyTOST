"""Edge-case coverage for engines: single-cluster spatial input, unbalanced
spatiotemporal panels, and empty margins lists.

These cases were previously untested (JOSS readiness review, Month 1 Week 3).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyTOST import (
    ClusterTOST,
    IIDTOST,
    SpatialConfig,
    SpatialTOST,
    SpatioTemporalConfig,
    SpatioTemporalTOST,
)


def test_spatial_engine_single_cluster_input():
    """SpatialTOST should fit successfully when all rows share one cluster id.

    The Matern GLS fit does not require multiple clusters to estimate the
    spatial covariance; a single cluster is simply one spatial block.
    """
    rng = np.random.default_rng(7)
    xs, ys = np.meshgrid(np.arange(4.0), np.arange(4.0))
    df = pd.DataFrame(
        {
            "cluster_id": "only_cluster",
            "x": xs.ravel(),
            "ycoord": ys.ravel(),
        }
    )
    df["diff"] = 0.10 + rng.normal(0.0, 0.01, size=len(df))

    res = SpatialTOST(
        y="diff",
        cluster="cluster_id",
        x="x",
        ycoord="ycoord",
        config=SpatialConfig(verbose_diagnostics=False, nu_grid=(0.5,)),
    ).fit(df, alpha=0.05, margins=[0.01, 0.5])

    assert len(res) == 2
    assert res["ci_low"].notna().all()
    assert res["ci_high"].notna().all()
    assert bool(res.loc[res["delta"] == 0.5, "equivalent"].iloc[0]) is True
    assert bool(res.loc[res["delta"] == 0.01, "equivalent"].iloc[0]) is False


def test_spatiotemporal_engine_unbalanced_panel_two_time_steps():
    """SpatioTemporalTOST should fall back to the per-time IVW path when the
    panel is unbalanced (different location counts per time slice), and
    should still succeed with the minimum viable T=2 time slices.

    Each time slice needs at least `min_time_n` (default 8) rows for the
    legacy per-time fit to be attempted at all, and at least 2 successful
    per-time fits are required overall -- with only T=2, both slices must
    succeed.
    """
    rng = np.random.default_rng(11)

    rows = []
    # Time 0: a 3x3 grid (S=9 locations).
    for xi in range(3):
        for yi in range(3):
            rows.append({"time": 0, "cluster_id": "A", "x": float(xi), "ycoord": float(yi)})
    # Time 1: a different, larger 2x5 grid (S=10 locations) -- unbalanced vs time 0.
    for xi in range(2):
        for yi in range(5):
            rows.append({"time": 1, "cluster_id": "A", "x": float(xi), "ycoord": float(yi)})

    df = pd.DataFrame(rows)
    df["diff"] = 0.10 + rng.normal(0.0, 0.01, size=len(df))

    res = SpatioTemporalTOST(
        y="diff",
        cluster="cluster_id",
        time="time",
        x="x",
        ycoord="ycoord",
        config=SpatioTemporalConfig(
            verbose_diagnostics=False,
            nu_grid=(0.5,),
            min_time_n=8,
        ),
    ).fit(df, alpha=0.05, margins=[0.01, 0.5])

    assert len(res) == 2
    assert (res["method"] == "Per-time Matérn GLS (REML) aggregated via IVW + t-CI").all()
    assert res["ci_low"].notna().all()
    assert res["ci_high"].notna().all()


def test_spatiotemporal_engine_unbalanced_panel_too_few_points_raises():
    """If fewer than 2 time slices meet `min_time_n`, the per-time fallback
    should raise a clear error rather than silently returning nonsense.

    Locations differ between the two time slices (so the balanced joint ML
    path is skipped) and each slice has too few points for the per-time
    fallback's `min_time_n` threshold, so both slices are skipped and the
    engine should raise.
    """
    rng = np.random.default_rng(13)
    rows = []
    for xi in range(2):
        for yi in range(2):
            rows.append({"time": 0, "cluster_id": "A", "x": float(xi), "ycoord": float(yi)})
    # Time 1 uses a shifted grid so the panel is not balanced across (x, y).
    for xi in range(2):
        for yi in range(2):
            rows.append({"time": 1, "cluster_id": "A", "x": float(xi) + 10.0, "ycoord": float(yi) + 10.0})
    df = pd.DataFrame(rows)
    df["diff"] = 0.10 + rng.normal(0.0, 0.01, size=len(df))

    with pytest.raises(ValueError, match="at least 2 time slices"):
        SpatioTemporalTOST(
            y="diff",
            cluster="cluster_id",
            time="time",
            x="x",
            ycoord="ycoord",
            config=SpatioTemporalConfig(verbose_diagnostics=False, nu_grid=(0.5,), min_time_n=8),
        ).fit(df, alpha=0.05, margins=[0.5])


@pytest.mark.parametrize(
    "engine_factory",
    [
        lambda: IIDTOST(y="diff"),
        lambda: ClusterTOST(y="diff", cluster="cluster_id"),
    ],
)
def test_empty_margins_list_returns_empty_frame_without_error(engine_factory, basic_diff_df):
    """Passing an empty margins list should not raise; it should return an
    empty results frame since there are no equivalence decisions to report.
    """
    engine = engine_factory()
    res = engine.fit(basic_diff_df, alpha=0.05, margins=[])
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 0


def test_empty_margins_list_spatial_and_spatiotemporal(spatial_df, spatiotemporal_df):
    """The spatial and spatiotemporal engines still perform their (possibly
    expensive) model fit once even with no margins, and return zero rows.
    """
    spatial_res = SpatialTOST(
        y="diff",
        cluster="cluster_id",
        x="x",
        ycoord="ycoord",
        config=SpatialConfig(verbose_diagnostics=False, nu_grid=(0.5,)),
    ).fit(spatial_df, alpha=0.05, margins=[])
    assert isinstance(spatial_res, pd.DataFrame)
    assert len(spatial_res) == 0

    st_res = SpatioTemporalTOST(
        y="diff",
        cluster="cluster_id",
        time="time",
        x="x",
        ycoord="ycoord",
        config=SpatioTemporalConfig(
            verbose_diagnostics=False,
            nu_grid=(0.5,),
            mu_bootstrap_B=20,
            mu_bootstrap_seed=123,
        ),
    ).fit(spatiotemporal_df, alpha=0.05, margins=[])
    assert isinstance(st_res, pd.DataFrame)
    assert len(st_res) == 0

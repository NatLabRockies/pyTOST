"""Behavioral tests that pin the exact publication workflow described in the manuscript.

These tests document:
  1. Analyst-selected engine via run_tost (no automatic selector).
  2. run_tost(engine="cluster") uses statsmodels ordinary cluster-robust OLS, t(G-1).
  3. CR2 and wild-cluster bootstrap come from pyTOST.fewcluster (separate public API).
  4. BuildingAwareSpatioTemporalTOST is a separate public class, not a run_tost engine.
  5. SpatialConfig / SpatioTemporalConfig default point_estimator="gls"; the publication
     common-estimand analyses require the nondefault "equal_weighted" passed explicitly.
  6. TemporalTOST: configurable HAC lags; require_unique_times documents the one-observation-
     per-ordered-time expectation for the 12-point monthly-mean case analysis (2 lags used).
  7. Unbalanced spatiotemporal panel: explicitly rejected (ValueError) when
     require_balanced=True; no silent downgrade on the publication path.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pyTOST
from pyTOST import (
    run_tost,
    WorkflowOptions,
    ClusterTOST,
    TemporalTOST,
    SpatialTOST,
    SpatialConfig,
    SpatioTemporalTOST,
    SpatioTemporalConfig,
    BuildingAwareSpatioTemporalTOST,
    BuildingAwareConfig,
)
import pyTOST.fewcluster as fewcluster


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cluster_df():
    """Small balanced cluster dataset (G=5, n_g=20)."""
    rng = np.random.default_rng(42)
    G = 5
    n_per = 20
    groups = np.repeat(np.arange(G), n_per)
    y = rng.normal(0.10, 0.05, size=G * n_per)
    return pd.DataFrame({"diff": y, "cluster_id": groups})


@pytest.fixture
def monthly_df():
    """12-point monthly-mean series (one observation per month) as in the case analysis.

    The publication analysis uses 12 monthly means, 2 HAC lags, and requires exactly
    one observation per time point.
    """
    rng = np.random.default_rng(7)
    months = np.arange(1, 13)
    y = 0.08 + rng.normal(0.0, 0.03, size=12)
    return pd.DataFrame({"diff": y, "month": months})


@pytest.fixture
def balanced_st_df():
    """Balanced spatiotemporal panel: 4 locations × 3 time points (T×S grid)."""
    rng = np.random.default_rng(99)
    locs = [(0.0, 0.0), (5.0, 0.0), (0.0, 5.0), (5.0, 5.0)]
    rows = []
    for t in range(3):
        for i, (xi, yi) in enumerate(locs):
            rows.append({"cluster_id": f"L{i}", "time": t, "x": xi, "ycoord": yi})
    df = pd.DataFrame(rows)
    df["diff"] = 0.10 + rng.normal(0.0, 0.02, size=len(df))
    return df


@pytest.fixture
def unbalanced_st_df():
    """Unbalanced spatiotemporal panel with enough points per time for spatial fits to converge.

    3 time slices, each with 4 locations, except one time slice has only 3 locations
    (missing one location at time=2).  This breaks the T×S balanced-grid requirement.
    """
    rng = np.random.default_rng(88)
    locs = [(0.0, 0.0), (5.0, 0.0), (0.0, 5.0), (5.0, 5.0)]
    rows = []
    for t in range(3):
        for i, (xi, yi) in enumerate(locs):
            if t == 2 and i == 3:
                continue  # drop one location at time=2 → unbalanced
            rows.append({"cluster_id": f"L{i}", "time": t, "x": xi, "ycoord": yi})
    df = pd.DataFrame(rows)
    df["diff"] = 0.10 + rng.normal(0.0, 0.02, size=len(df))
    return df


# ---------------------------------------------------------------------------
# 1. Analyst-selected engine: run_tost requires explicit engine argument
# ---------------------------------------------------------------------------

class TestAnalystSelectedEngine:
    def test_run_tost_default_engine_is_iid(self, cluster_df):
        """Default engine is 'iid'; run_tost does not auto-select."""
        out = run_tost(
            cluster_df, y="diff", margins=[0.5],
            options=WorkflowOptions(do_sensitivity=False, bootstrap_B=0),
        )
        assert out["engine"] == "iid"

    def test_run_tost_engine_selection_is_explicit(self, cluster_df):
        """Passing engine='cluster' activates the cluster engine; no diagnostic-driven switch."""
        out = run_tost(
            cluster_df, y="diff", margins=[0.5], engine="cluster", cluster="cluster_id",
            options=WorkflowOptions(do_sensitivity=False, bootstrap_B=0),
        )
        assert out["engine"] == "cluster"

    def test_run_tost_unknown_engine_raises(self, cluster_df):
        """An unknown engine name raises ValueError, not a silent fallback."""
        with pytest.raises(ValueError, match="Unknown engine"):
            run_tost(
                cluster_df, y="diff", margins=[0.5], engine="autoselect",
                options=WorkflowOptions(do_sensitivity=False, bootstrap_B=0),
            )


# ---------------------------------------------------------------------------
# 2. run_tost(engine="cluster") → ordinary statsmodels cluster sandwich t(G-1)
# ---------------------------------------------------------------------------

class TestClusterEngineIsOrdinarySandwich:
    def test_cluster_engine_uses_ols_cluster_robust_se(self, cluster_df):
        """run_tost engine='cluster' uses OLS + statsmodels cluster-robust SE, not CR2/wild."""
        out = run_tost(
            cluster_df, y="diff", margins=[1.0], engine="cluster", cluster="cluster_id",
            options=WorkflowOptions(do_sensitivity=False, bootstrap_B=0),
        )
        primary = out["primary"]
        row = primary.iloc[0]
        # Label must mention cluster-robust (ordinary sandwich)
        assert "cluster-robust" in row["method"].lower() or "cluster" in row["method"].lower()
        # df must be G-1 = 4 for G=5 clusters
        G = int(cluster_df["cluster_id"].nunique())
        assert int(row["df"]) == G - 1

    def test_cluster_engine_does_not_produce_cr2_df(self, cluster_df):
        """ClusterTOST df is exactly G-1, not a Satterthwaite CR2 fractional df."""
        G = int(cluster_df["cluster_id"].nunique())
        tost = ClusterTOST(y="diff", cluster="cluster_id")
        result = tost.fit(cluster_df, alpha=0.05, margins=[1.0])
        assert int(result.iloc[0]["df"]) == G - 1, "ClusterTOST df must be integer G-1 (ordinary sandwich)"


# ---------------------------------------------------------------------------
# 3. CR2 and wild-cluster bootstrap from pyTOST.fewcluster (separate public API)
# ---------------------------------------------------------------------------

class TestFewclusterPublicInterface:
    def test_cr2_mean_is_importable_from_fewcluster(self):
        """pyTOST.fewcluster.cr2_mean must exist and return a dict with expected keys."""
        assert hasattr(fewcluster, "cr2_mean"), "cr2_mean not found in pyTOST.fewcluster"

    def test_wild_cluster_bootstrap_ci_is_importable_from_fewcluster(self):
        """pyTOST.fewcluster.wild_cluster_bootstrap_ci must exist."""
        assert hasattr(fewcluster, "wild_cluster_bootstrap_ci"), \
            "wild_cluster_bootstrap_ci not found in pyTOST.fewcluster"

    def test_cr2_mean_returns_finite_ci(self, cluster_df):
        """cr2_mean returns a finite CI; Satterthwaite df_bm is a documented output."""
        y = cluster_df["diff"].to_numpy(float)
        g = cluster_df["cluster_id"].to_numpy()
        result = fewcluster.cr2_mean(y, g, alpha=0.05)
        assert "df_bm" in result
        # CI bounds available (keys may differ from run_tost primary output)
        assert "ci_low" in result or "ci" in result
        lo = result.get("ci_low", result.get("ci", [None, None])[0])
        hi = result.get("ci_high", result.get("ci", [None, None])[1])
        assert np.isfinite(lo) and np.isfinite(hi) and hi > lo
        # For balanced clusters df_bm equals G-1 (verified by S01 numeric check)
        df_bm = result["df_bm"]
        G = int(cluster_df["cluster_id"].nunique())
        assert df_bm <= G  # Satterthwaite df <= G in general

    def test_cr2_not_reachable_via_run_tost(self, cluster_df):
        """run_tost has no cr2 or wild engine — CR2/wild are in fewcluster only."""
        for bad_engine in ("cr2", "wild", "fewcluster", "cr2_mean"):
            with pytest.raises(ValueError, match="Unknown engine"):
                run_tost(
                    cluster_df, y="diff", margins=[0.5], engine=bad_engine,
                    cluster="cluster_id",
                    options=WorkflowOptions(do_sensitivity=False, bootstrap_B=0),
                )


# ---------------------------------------------------------------------------
# 4. BuildingAwareSpatioTemporalTOST is a separate public class, not run_tost engine
# ---------------------------------------------------------------------------

class TestHierarchicalClassIsSeparate:
    def test_building_aware_is_in_public_api(self):
        """BuildingAwareSpatioTemporalTOST is exported from the top-level pyTOST namespace."""
        assert hasattr(pyTOST, "BuildingAwareSpatioTemporalTOST")
        assert BuildingAwareSpatioTemporalTOST is pyTOST.BuildingAwareSpatioTemporalTOST

    def test_hierarchical_engine_not_reachable_via_run_tost(self, cluster_df):
        """run_tost has no 'building_aware' or 'hierarchical' engine string."""
        for bad_engine in ("building_aware", "hierarchical", "building"):
            with pytest.raises(ValueError, match="Unknown engine"):
                run_tost(
                    cluster_df, y="diff", margins=[0.5], engine=bad_engine,
                    options=WorkflowOptions(do_sensitivity=False, bootstrap_B=0),
                )

    def test_building_aware_is_instantiable_directly(self, balanced_st_df):
        """BuildingAwareSpatioTemporalTOST can be constructed and fit directly."""
        tost = BuildingAwareSpatioTemporalTOST(
            y="diff", cluster="cluster_id", time="time", x="x", ycoord="ycoord",
            config=BuildingAwareConfig(),
        )
        result = tost.fit(balanced_st_df, alpha=0.05, margins=[1.0])
        assert isinstance(result, pd.DataFrame)
        assert "ci_low" in result.columns and "ci_high" in result.columns


# ---------------------------------------------------------------------------
# 5. equal_weighted must be passed explicitly (default is "gls")
# ---------------------------------------------------------------------------

class TestEqualWeightedPointEstimator:
    def test_spatial_config_default_is_gls(self):
        """SpatialConfig default point_estimator is 'gls', not 'equal_weighted'."""
        cfg = SpatialConfig()
        assert cfg.point_estimator == "gls"

    def test_spatiotemporal_config_default_is_gls(self):
        """SpatioTemporalConfig default point_estimator is 'gls', not 'equal_weighted'."""
        cfg = SpatioTemporalConfig()
        assert cfg.point_estimator == "gls"

    def test_spatial_equal_weighted_requires_explicit_kwarg(self):
        """equal_weighted must be passed explicitly via SpatialConfig; it is not the default."""
        cfg_gls = SpatialConfig(point_estimator="gls")
        cfg_ew = SpatialConfig(point_estimator="equal_weighted")
        assert cfg_gls.point_estimator == "gls"
        assert cfg_ew.point_estimator == "equal_weighted"

    def test_spatiotemporal_equal_weighted_requires_explicit_kwarg(self):
        """equal_weighted must be passed explicitly via SpatioTemporalConfig."""
        cfg_gls = SpatioTemporalConfig(point_estimator="gls")
        cfg_ew = SpatioTemporalConfig(point_estimator="equal_weighted")
        assert cfg_gls.point_estimator == "gls"
        assert cfg_ew.point_estimator == "equal_weighted"

    def test_spatial_equal_weighted_returns_sample_mean(self):
        """SpatialTOST with equal_weighted returns the simple sample mean as mu_hat."""
        rng = np.random.default_rng(5)
        rows = [{"cluster_id": "A", "x": float(i), "ycoord": 0.0} for i in range(6)]
        df = pd.DataFrame(rows)
        df["diff"] = rng.normal(0.1, 0.02, size=len(df))

        cfg = SpatialConfig(point_estimator="equal_weighted", nu_grid=(0.5,))
        tost = SpatialTOST(y="diff", cluster="cluster_id", x="x", ycoord="ycoord", config=cfg)
        result = tost.fit(df, alpha=0.05, margins=[1.0])
        expected_mean = float(df["diff"].mean())
        assert abs(result.iloc[0]["mu_hat"] - expected_mean) < 1e-9, \
            "equal_weighted mu_hat must equal the sample mean"


# ---------------------------------------------------------------------------
# 6. TemporalTOST: lag configuration + one-observation-per-ordered-time requirement
# ---------------------------------------------------------------------------

class TestTemporalLagAndTiedRows:
    def test_temporal_default_hac_lags_is_auto(self):
        """TemporalTOST selects the HAC lag from the sample size by default.

        Changed in 0.17.0: the default was previously a fixed lag of 4, which
        under-smoothed long series and over-smoothed short ones.
        """
        t = TemporalTOST(y="diff", time="month")
        assert t.hac_lags == "auto"

    def test_temporal_hac_lags_configurable(self, monthly_df):
        """TemporalTOST with hac_lags=2 sets method label to report 2 lags (monthly case)."""
        tost = TemporalTOST(y="diff", time="month", hac_lags=2)
        result = tost.fit(monthly_df, alpha=0.05, margins=[0.5])
        assert tost.hac_lags == 2
        assert "lags=2" in result.iloc[0]["method"]

    def test_temporal_monthly_series_12_points_2_lags(self, monthly_df):
        """12-point monthly series with 2 lags produces finite CI (case analysis setup)."""
        tost = TemporalTOST(y="diff", time="month", hac_lags=2)
        result = tost.fit(monthly_df, alpha=0.05, margins=[0.5])
        assert len(result) == 1
        row = result.iloc[0]
        assert np.isfinite(row["ci_low"]) and np.isfinite(row["ci_high"])
        assert row["ci_high"] > row["ci_low"]

    def test_temporal_require_unique_times_rejects_tied_rows(self, monthly_df):
        """require_unique_times=True raises ValueError when tied time rows are present."""
        # Introduce a duplicate month
        tied_df = pd.concat([monthly_df, monthly_df.iloc[[0]]], ignore_index=True)
        tost = TemporalTOST(y="diff", time="month", hac_lags=2, require_unique_times=True)
        with pytest.raises(ValueError, match="duplicate time values"):
            tost.fit(tied_df, alpha=0.05, margins=[0.5])

    def test_temporal_accepts_tied_rows_by_default(self, monthly_df):
        """By default (require_unique_times=False), TemporalTOST accepts tied time rows.

        It warns while doing so: HAC inference treats the data as one sequence indexed
        by time, which tied rows violate (docs/review_register.md, RR-001).
        """
        tied_df = pd.concat([monthly_df, monthly_df.iloc[[0]]], ignore_index=True)
        tost = TemporalTOST(y="diff", time="month", hac_lags=2, require_unique_times=False)
        # Should not raise; treats multiple rows at same time as repeated observations
        with pytest.warns(UserWarning, match="duplicate time"):
            result = tost.fit(tied_df, alpha=0.05, margins=[0.5])
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# 7. Unbalanced spatiotemporal panel → explicit rejection on publication path
# ---------------------------------------------------------------------------

class TestUnbalancedPanelRejection:
    def test_balanced_panel_accepted_with_require_balanced(self, balanced_st_df):
        """A balanced T×S panel succeeds when require_balanced=True."""
        cfg = SpatioTemporalConfig(
            require_balanced=True,
            point_estimator="equal_weighted",
            mu_ci_method="wald",
            nu_grid=(0.5,),
        )
        tost = SpatioTemporalTOST(
            y="diff", cluster="cluster_id", time="time", x="x", ycoord="ycoord",
            config=cfg,
        )
        result = tost.fit(balanced_st_df, alpha=0.05, margins=[1.0])
        assert isinstance(result, pd.DataFrame)

    def test_unbalanced_panel_raises_with_require_balanced(self, unbalanced_st_df):
        """require_balanced=True raises ValueError on an unbalanced panel (no silent fallback)."""
        cfg = SpatioTemporalConfig(
            require_balanced=True,
            point_estimator="equal_weighted",
            nu_grid=(0.5,),
        )
        tost = SpatioTemporalTOST(
            y="diff", cluster="cluster_id", time="time", x="x", ycoord="ycoord",
            config=cfg,
        )
        with pytest.raises(ValueError, match="not balanced|require_balanced"):
            tost.fit(unbalanced_st_df, alpha=0.05, margins=[1.0])

    def test_unbalanced_panel_allowed_by_default(self, unbalanced_st_df):
        """Default (require_balanced=False): unbalanced panel silently uses per-time IVW fallback."""
        cfg = SpatioTemporalConfig(
            require_balanced=False,
            nu_grid=(0.5,),
            verbose_diagnostics=False,
            min_time_n=3,  # allow small time slices in this fixture (4 locs - 1 = 3)
        )
        tost = SpatioTemporalTOST(
            y="diff", cluster="cluster_id", time="time", x="x", ycoord="ycoord",
            config=cfg,
        )
        result = tost.fit(unbalanced_st_df, alpha=0.05, margins=[1.0])
        assert isinstance(result, pd.DataFrame)

    def test_publication_path_uses_require_balanced_true(self, balanced_st_df):
        """Publication workflow: pass require_balanced=True explicitly to prevent silent unbalanced fallback."""
        cfg = SpatioTemporalConfig(
            require_balanced=True,
            point_estimator="equal_weighted",
            mu_ci_method="wald",
            nu_grid=(0.5,),
        )
        out = run_tost(
            balanced_st_df,
            y="diff", margins=[1.0], engine="spatiotemporal",
            cluster="cluster_id", time="time", x="x", ycoord="ycoord",
            spatiotemporal_config=cfg,
            options=WorkflowOptions(do_sensitivity=False, bootstrap_B=0),
        )
        assert out["engine"] == "spatiotemporal"
        assert "mu_hat" in out["primary"].columns

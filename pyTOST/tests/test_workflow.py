
from __future__ import annotations

import pytest

from pyTOST import WorkflowOptions, run_tost, SpatialConfig, SpatioTemporalConfig


def test_workflow_cluster_returns_primary_and_bootstrap(basic_diff_df):
    out = run_tost(
        basic_diff_df,
        y="diff",
        margins=[0.5],
        engine="cluster",
        cluster="cluster_id",
        options=WorkflowOptions(do_sensitivity=False, bootstrap_B=10, seed=123),
    )

    assert out["engine"] == "cluster"
    assert "primary" in out
    assert "bootstrap" in out
    assert out["bootstrap"]["method"] == "cluster_bootstrap"
    assert len(out["primary"]) == 1


def test_workflow_validates_required_columns(basic_diff_df):
    with pytest.raises(ValueError, match="requires `cluster`"):
        run_tost(
            basic_diff_df,
            y="diff",
            margins=[0.5],
            engine="cluster",
            options=WorkflowOptions(do_sensitivity=False, bootstrap_B=0),
        )


def test_workflow_spatial_smoke(spatial_df):
    out = run_tost(
        spatial_df,
        y="diff",
        margins=[0.5],
        engine="spatial",
        cluster="cluster_id",
        x="x",
        ycoord="ycoord",
        spatial_config=SpatialConfig(verbose_diagnostics=False, nu_grid=(0.5,)),
        options=WorkflowOptions(do_sensitivity=False, bootstrap_B=10, seed=123),
    )

    assert out["engine"] == "spatial"
    assert "primary" in out
    assert "bootstrap" in out
    assert out["bootstrap"]["method"] in {"cluster_bootstrap", "spatial_within_building_block_bootstrap", "spatial_block_bootstrap"}


def test_workflow_spatiotemporal_smoke(spatiotemporal_df):
    out = run_tost(
        spatiotemporal_df,
        y="diff",
        margins=[0.5],
        engine="spatiotemporal",
        cluster="cluster_id",
        time="time",
        x="x",
        ycoord="ycoord",
        spatiotemporal_config=SpatioTemporalConfig(
            verbose_diagnostics=False,
            nu_grid=(0.5,),
            mu_bootstrap_B=20,
            mu_bootstrap_seed=123,
        ),
        options=WorkflowOptions(do_sensitivity=False, bootstrap_B=10, seed=123),
    )

    assert out["engine"] == "spatiotemporal"
    assert "primary" in out
    assert "bootstrap" in out


def test_workflow_sensitivity_branch_returns_expected_outputs(basic_diff_df):
    out = run_tost(
        basic_diff_df,
        y="diff",
        margins=[0.5],
        engine="cluster",
        cluster="cluster_id",
        options=WorkflowOptions(
            do_sensitivity=True,
            bootstrap_B=10,
            robust_location_B=25,
            robust_location_block_len=3,
            robust_location_stat="median",
            seed=123,
        ),
    )

    assert out["engine"] == "cluster"
    assert "primary" in out
    assert "bootstrap" in out
    assert "sensitivity" in out
    assert set(out["sensitivity"].keys()) == {"Heteroskedastic", "Robust Location"}
    robust = out["sensitivity"]["Robust Location"]
    assert len(robust) == 1
    assert robust.loc[0, "method"] == "Robust median + Cluster Bootstrap"

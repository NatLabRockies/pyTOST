from __future__ import annotations

import math

import pytest

from pyTOST import SpatioTemporalConfig, SpatialConfig, WorkflowOptions, run_tost
from pyTOST.data_gen.synthetic_tost_data import (
    generate_cluster_groups,
    generate_iid,
    generate_spatial_clusters,
    generate_spatiotemporal,
    generate_temporal_ar1,
)
from tests.conftest import long_to_diff


BASELINE = {
    "iid": {
        "mu_hat": 0.09696327538598477,
        "ci_low": -0.010774662394107196,
        "ci_high": 0.20470121316607676,
        "equivalent": True,
        "df": 23.0,
    },
    "cluster": {
        "mu_hat": -0.0014375918686247303,
        "ci_low": -0.15502894407928147,
        "ci_high": 0.15215376034203199,
        "equivalent": True,
        "df": 7.0,
    },
    "temporal": {
        "mu_hat": 0.14365583946437474,
        "ci_low": 0.10239980673992613,
        "ci_high": 0.18491187218882335,
        "equivalent": True,
        "df": math.inf,
    },
    "spatial": {
        "mu_hat": 0.1415157128366212,
        "ci_low": 0.05836657831386505,
        "ci_high": 0.2246154343061697,
        "equivalent": True,
        "df": None,
    },
    "spatiotemporal": {
        "mu_hat": 0.053390076751734246,
        "ci_low": -0.11864130960849933,
        "ci_high": 0.2404631545944979,
        "equivalent": True,
        "df": None,
    },
}


def _first_row(primary):
    row = primary.iloc[0]
    return {
        "mu_hat": float(row["mu_hat"]),
        "ci_low": float(row["ci_low"]),
        "ci_high": float(row["ci_high"]),
        "equivalent": bool(row["equivalent"]),
        "df": float(row["df"]) if "df" in primary.columns else None,
    }


def test_deterministic_engine_regression_baseline():
    # IID
    iid_long, _ = generate_iid(n=24, delta=0.15, sigma=0.2, seed=2026)
    iid_df = long_to_diff(iid_long, index_cols=["sample_id"])
    iid = run_tost(
        iid_df,
        y="diff",
        margins=[0.25],
        engine="iid",
        options=WorkflowOptions(do_sensitivity=False, bootstrap_B=0),
    )

    # Cluster
    cluster_long, _ = generate_cluster_groups(
        n_groups=8,
        points_per_group=5,
        delta=0.15,
        seed=2026,
        nugget_sd=0.2,
        meas_group_sd=0.15,
    )
    cluster_df = long_to_diff(
        cluster_long, index_cols=["sample_id", "group_id", "x", "y_sp"]
    ).rename(columns={"group_id": "cluster_id", "y_sp": "ycoord"})
    cluster = run_tost(
        cluster_df,
        y="diff",
        margins=[0.25],
        engine="cluster",
        cluster="cluster_id",
        options=WorkflowOptions(do_sensitivity=False, bootstrap_B=0),
    )

    # Temporal
    temporal_long, _ = generate_temporal_ar1(
        n_time=40,
        series_per_arm=3,
        rho=0.6,
        process_sd=0.8,
        obs_sd=0.2,
        delta=0.15,
        seed=2026,
    )
    temporal_df = long_to_diff(
        temporal_long, index_cols=["sample_id", "series_id", "t"]
    ).rename(columns={"t": "time"})
    temporal = run_tost(
        temporal_df,
        y="diff",
        margins=[0.25],
        engine="temporal",
        time="time",
        options=WorkflowOptions(do_sensitivity=False, bootstrap_B=0),
    )

    # Spatial
    spatial_long, _ = generate_spatial_clusters(
        n_clusters=10,
        points_per_cluster=6,
        length_scale=0.5,
        field_sd=0.8,
        nugget_sd=0.2,
        delta=0.15,
        seed=2026,
    )
    spatial_df = long_to_diff(
        spatial_long, index_cols=["sample_id", "group_id", "x", "y_sp"]
    ).rename(columns={"group_id": "cluster_id", "y_sp": "ycoord"})
    spatial = run_tost(
        spatial_df,
        y="diff",
        margins=[0.25],
        engine="spatial",
        cluster="cluster_id",
        x="x",
        ycoord="ycoord",
        spatial_config=SpatialConfig(verbose_diagnostics=False, nu_grid=(0.5,)),
        options=WorkflowOptions(
            do_sensitivity=False, bootstrap_B=0, cross_cluster_dependence=False
        ),
    )

    # Spatiotemporal
    st_long, _ = generate_spatiotemporal(
        n_space=18,
        n_time=8,
        length_scale=0.6,
        rho=0.5,
        spatial_sd=0.8,
        obs_sd=0.2,
        delta=0.15,
        seed=2026,
    )
    st_df = long_to_diff(st_long, index_cols=["sample_id", "x", "y_sp", "t"]).rename(
        columns={"y_sp": "ycoord", "t": "time"}
    )
    coord_pairs = list(zip(st_df["x"].round(8), st_df["ycoord"].round(8)))
    coord_to_cluster = {}
    st_df["cluster_id"] = [
        coord_to_cluster.setdefault(pair, len(coord_to_cluster)) for pair in coord_pairs
    ]
    spatiotemporal = run_tost(
        st_df,
        y="diff",
        margins=[0.25],
        engine="spatiotemporal",
        cluster="cluster_id",
        time="time",
        x="x",
        ycoord="ycoord",
        spatiotemporal_config=SpatioTemporalConfig(
            verbose_diagnostics=False,
            nu_grid=(0.5,),
            mu_bootstrap_B=30,
            mu_bootstrap_seed=2026,
        ),
        options=WorkflowOptions(do_sensitivity=False, bootstrap_B=0),
    )

    observed = {
        "iid": _first_row(iid["primary"]),
        "cluster": _first_row(cluster["primary"]),
        "temporal": _first_row(temporal["primary"]),
        "spatial": _first_row(spatial["primary"]),
        "spatiotemporal": _first_row(spatiotemporal["primary"]),
    }

    for engine, expected in BASELINE.items():
        got = observed[engine]
        assert got["mu_hat"] == pytest.approx(expected["mu_hat"], abs=1e-6)
        assert got["ci_low"] == pytest.approx(expected["ci_low"], abs=1e-6)
        assert got["ci_high"] == pytest.approx(expected["ci_high"], abs=1e-6)
        assert got["equivalent"] is expected["equivalent"]
        if expected["df"] is None:
            assert got["df"] is None
        elif math.isinf(expected["df"]):
            assert math.isinf(got["df"])
        else:
            assert got["df"] == pytest.approx(expected["df"])

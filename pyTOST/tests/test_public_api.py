import pandas as pd

from pyTOST import (
    ClusterTOST,
    HeteroskedasticTOST,
    IIDTOST,
    RobustLocationTOST,
    SpatialConfig,
    SpatialTOST,
    SpatioTemporalConfig,
    SpatioTemporalTOST,
    TemporalTOST,
    WorkflowOptions,
    run_tost,
)
from pyTOST.data_gen.synthetic_tost_data import (
    generate_cluster_groups,
    generate_iid,
    generate_iid_grouped,
    generate_spatial_clusters,
    generate_spatiotemporal,
    generate_temporal_ar1,
)
from pyTOST.data_gen.params_io import load_params


def test_public_imports_from_readme_exist():
    assert callable(run_tost)
    assert WorkflowOptions is not None
    assert IIDTOST is not None
    assert ClusterTOST is not None
    assert TemporalTOST is not None
    assert SpatialTOST is not None
    assert SpatialConfig is not None
    assert SpatioTemporalTOST is not None
    assert SpatioTemporalConfig is not None
    assert HeteroskedasticTOST is not None
    assert RobustLocationTOST is not None


def test_readme_quickstart_cluster_example_runs():
    df = pd.DataFrame(
        {
            "diff": [0.10, 0.18, 0.05, 0.12, 0.08, 0.15],
            "cluster_id": ["A", "A", "B", "B", "C", "C"],
        }
    )

    res = run_tost(
        df,
        y="diff",
        margins=[0.5],
        alpha=0.05,
        engine="cluster",
        cluster="cluster_id",
        options=WorkflowOptions(
            do_sensitivity=True,
            bootstrap_B=50,
            robust_location_B=30,
            seed=42,
        ),
    )

    primary = res["primary"]
    assert list(primary[["delta", "mu_hat", "ci_low", "ci_high", "equivalent"]].columns) == [
        "delta",
        "mu_hat",
        "ci_low",
        "ci_high",
        "equivalent",
    ]
    assert res["engine"] == "cluster"
    assert "sensitivity" in res
    assert "bootstrap" in res


def test_readme_data_gen_imports_exist():
    assert callable(generate_iid)
    assert callable(generate_iid_grouped)
    assert callable(generate_cluster_groups)
    assert callable(generate_spatial_clusters)
    assert callable(generate_temporal_ar1)
    assert callable(generate_spatiotemporal)
    assert callable(load_params)

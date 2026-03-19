
from __future__ import annotations

def test_public_imports_smoke():
    from pyTOST import (
        run_tost,
        WorkflowOptions,
        IIDTOST,
        ClusterTOST,
        TemporalTOST,
        SpatialTOST,
        SpatialConfig,
        SpatioTemporalTOST,
        SpatioTemporalConfig,
        HeteroskedasticTOST,
        RobustLocationTOST,
    )

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


def test_data_gen_imports_smoke():
    from pyTOST.data_gen.synthetic_tost_data import (
        generate_iid,
        generate_iid_grouped,
        generate_cluster_groups,
        generate_spatial_clusters,
        generate_temporal_ar1,
        generate_spatiotemporal,
    )
    from pyTOST.data_gen.params_io import load_params, kwargs_for, validate_required_kwargs

    assert callable(generate_iid)
    assert callable(generate_iid_grouped)
    assert callable(generate_cluster_groups)
    assert callable(generate_spatial_clusters)
    assert callable(generate_temporal_ar1)
    assert callable(generate_spatiotemporal)
    assert callable(load_params)
    assert callable(kwargs_for)
    assert callable(validate_required_kwargs)

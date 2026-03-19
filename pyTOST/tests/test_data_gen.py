
from __future__ import annotations

import json

import pandas as pd
import pytest

from pyTOST.data_gen.params_io import kwargs_for, load_params, validate_required_kwargs
from pyTOST.data_gen.synthetic_tost_data import (
    generate_cluster_groups,
    generate_iid,
    generate_iid_grouped,
    generate_spatial_clusters,
    generate_spatiotemporal,
    generate_temporal_ar1,
)


def test_generate_iid_is_reproducible():
    df1, meta1 = generate_iid(n=10, delta=0.2, sigma=0.1, seed=123)
    df2, meta2 = generate_iid(n=10, delta=0.2, sigma=0.1, seed=123)

    pd.testing.assert_frame_equal(df1, df2)
    assert meta1 == meta2


@pytest.mark.parametrize(
    "generator, kwargs, required_cols",
    [
        (generate_iid, {"n": 6, "seed": 1}, {"sample_id", "arm", "baseline", "effect", "mu", "y"}),
        (generate_iid_grouped, {"n_groups": 2, "n_per_group": 3, "seed": 1}, {"sample_id", "arm", "group_id", "baseline", "effect", "mu", "y"}),
        (generate_cluster_groups, {"n_groups": 2, "points_per_group": 3, "seed": 1}, {"sample_id", "arm", "group_id", "x", "y_sp", "mu", "y"}),
        (generate_spatial_clusters, {"n_clusters": 2, "points_per_cluster": 3, "seed": 1}, {"sample_id", "arm", "group_id", "x", "y_sp", "mu", "y"}),
        (generate_temporal_ar1, {"n_time": 5, "seed": 1}, {"sample_id", "arm", "series_id", "t", "mu", "y"}),
        (generate_spatiotemporal, {"n_space": 4, "n_time": 3, "seed": 1}, {"sample_id", "arm", "x", "y_sp", "t", "mu", "y"}),
    ],
)
def test_generator_schemas(generator, kwargs, required_cols):
    df, meta = generator(**kwargs)
    assert required_cols.issubset(df.columns)
    assert isinstance(meta, dict)
    assert len(df) > 0


def test_params_io_load_and_filter(tmp_path):
    path = tmp_path / "params.json"
    payload = {"best": {"params": {"n": 10, "delta": 0.1, "sigma": 0.2, "_meta": {"note": "drop me"}}}}
    path.write_text(json.dumps(payload), encoding="utf-8")

    params = load_params(path)
    assert params == {"n": 10, "delta": 0.1, "sigma": 0.2}

    filtered = kwargs_for(generate_iid, params)
    assert filtered == {"n": 10, "delta": 0.1, "sigma": 0.2}
    validate_required_kwargs(generate_iid, filtered)

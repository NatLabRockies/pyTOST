
from __future__ import annotations

from pyTOST import SpatialConfig, SpatialTOST


def test_spatial_engine_smoke_and_decisions(spatial_df):
    res = SpatialTOST(
        y="diff",
        cluster="cluster_id",
        x="x",
        ycoord="ycoord",
        config=SpatialConfig(verbose_diagnostics=False, nu_grid=(0.5,)),
    ).fit(spatial_df, alpha=0.05, margins=[0.01, 0.5])

    assert set(["delta", "mu_hat", "ci_low", "ci_high", "equivalent", "method"]).issubset(res.columns)
    assert len(res) == 2
    assert bool(res.loc[res["delta"] == 0.5, "equivalent"].iloc[0]) is True
    assert bool(res.loc[res["delta"] == 0.01, "equivalent"].iloc[0]) is False
    assert "Matérn GLS" in res["method"].iloc[0]

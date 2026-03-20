
from __future__ import annotations

from pyTOST import SpatioTemporalConfig, SpatioTemporalTOST


def test_spatiotemporal_engine_smoke_and_decisions(spatiotemporal_df):
    res = SpatioTemporalTOST(
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
    ).fit(spatiotemporal_df, alpha=0.05, margins=[0.01, 0.5])

    assert set(["delta", "mu_hat", "ci_low", "ci_high", "equivalent", "method"]).issubset(res.columns)
    assert len(res) == 2
    assert bool(res.loc[res["delta"] == 0.5, "equivalent"].iloc[0]) is True
    assert bool(res.loc[res["delta"] == 0.01, "equivalent"].iloc[0]) is False
    assert res["method"].notna().all()

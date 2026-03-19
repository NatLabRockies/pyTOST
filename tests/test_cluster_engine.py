
from __future__ import annotations

from pyTOST import ClusterTOST


def test_cluster_engine_result_schema_and_decisions(basic_diff_df):
    res = ClusterTOST("diff", "cluster_id").fit(basic_diff_df, alpha=0.05, margins=[0.01, 0.5])

    assert set(["delta", "mu_hat", "ci_low", "ci_high", "equivalent", "method", "df"]).issubset(res.columns)
    assert len(res) == 2
    assert bool(res.loc[res["delta"] == 0.5, "equivalent"].iloc[0]) is True
    assert bool(res.loc[res["delta"] == 0.01, "equivalent"].iloc[0]) is False
    assert "cluster-robust" in res["method"].iloc[0]

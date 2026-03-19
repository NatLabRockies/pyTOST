
from __future__ import annotations

from pyTOST import IIDTOST


def test_iid_engine_result_schema_and_decisions(basic_diff_df):
    res = IIDTOST("diff").fit(basic_diff_df, alpha=0.05, margins=[0.01, 0.5])

    assert list(res.columns) == ["delta", "mu_hat", "ci_low", "ci_high", "equivalent", "method", "df"]
    assert len(res) == 2
    assert bool(res.loc[res["delta"] == 0.5, "equivalent"].iloc[0]) is True
    assert bool(res.loc[res["delta"] == 0.01, "equivalent"].iloc[0]) is False
    assert "IID" in res["method"].iloc[0]

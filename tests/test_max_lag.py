"""Tests for user control over the Newey-West HAC lag used by the temporal engine."""

import numpy as np
import pandas as pd
import pytest

from pyTOST import WorkflowOptions, run_tost
from pyTOST.engines.temporal_tost import TemporalTOST, auto_hac_lags


@pytest.fixture
def series_df():
    rng = np.random.default_rng(11)
    # n chosen so the automatic rule resolves to a lag other than the default of 4,
    # which keeps the auto-selection tests discriminating.
    n = 400
    noise = rng.normal(0.0, 0.05, size=n)
    # AR(1) structure so the lag choice actually matters
    for i in range(1, n):
        noise[i] += 0.6 * noise[i - 1]
    return pd.DataFrame({"diff": 0.10 + noise, "time": np.arange(n)})


class TestAutoLagRule:
    def test_rule_matches_newey_west_1994_formula(self):
        for n in (10, 50, 100, 250, 1000):
            assert auto_hac_lags(n) == int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))

    def test_rule_grows_with_sample_size(self):
        assert auto_hac_lags(50) <= auto_hac_lags(500) <= auto_hac_lags(5000)

    def test_rule_is_non_negative_for_tiny_samples(self):
        assert auto_hac_lags(1) >= 0
        assert auto_hac_lags(0) >= 0


class TestEngineLagSelection:
    def test_default_lag_is_unchanged(self):
        """The documented default must stay at 4 so existing results are stable."""
        assert TemporalTOST("diff", "time").hac_lags == 4

    def test_auto_resolves_from_the_data_length(self, series_df):
        tost = TemporalTOST("diff", "time", hac_lags="auto")
        result = tost.fit(series_df, alpha=0.05, margins=[0.5])

        expected = auto_hac_lags(len(series_df))
        assert f"lags={expected}" in result.iloc[0]["method"]
        assert expected != 4, "fixture should exercise a lag different from the default"

    def test_explicit_integer_is_honored(self, series_df):
        result = TemporalTOST("diff", "time", hac_lags=9).fit(
            series_df, alpha=0.05, margins=[0.5]
        )
        assert "lags=9" in result.iloc[0]["method"]

    def test_auto_label_reports_that_selection_was_automatic(self, series_df):
        result = TemporalTOST("diff", "time", hac_lags="auto").fit(
            series_df, alpha=0.05, margins=[0.5]
        )
        assert "auto" in result.iloc[0]["method"].lower()

    def test_more_lags_widen_the_interval_under_positive_autocorrelation(self, series_df):
        narrow = TemporalTOST("diff", "time", hac_lags=0).fit(
            series_df, alpha=0.05, margins=[0.5]
        )
        wide = TemporalTOST("diff", "time", hac_lags=12).fit(
            series_df, alpha=0.05, margins=[0.5]
        )

        narrow_width = narrow.iloc[0]["ci_high"] - narrow.iloc[0]["ci_low"]
        wide_width = wide.iloc[0]["ci_high"] - wide.iloc[0]["ci_low"]
        assert wide_width > narrow_width

    def test_negative_lag_is_rejected(self):
        with pytest.raises(ValueError, match="max_lag|hac_lags"):
            TemporalTOST("diff", "time", hac_lags=-1)

    def test_unknown_string_is_rejected(self):
        with pytest.raises(ValueError, match="auto"):
            TemporalTOST("diff", "time", hac_lags="sometimes")


class TestWorkflowOption:
    def _run(self, df, **options):
        return run_tost(
            df,
            y="diff",
            margins=[0.5],
            alpha=0.05,
            engine="temporal",
            time="time",
            options=WorkflowOptions(do_sensitivity=False, bootstrap_B=0, **options),
        )

    def test_default_workflow_preserves_the_engine_default(self, series_df):
        res = self._run(series_df)
        assert "lags=4" in res["primary"].iloc[0]["method"]

    def test_max_lag_overrides_the_engine_default(self, series_df):
        res = self._run(series_df, max_lag=10)
        assert "lags=10" in res["primary"].iloc[0]["method"]

    def test_max_lag_auto_is_supported_through_the_workflow(self, series_df):
        res = self._run(series_df, max_lag="auto")
        expected = auto_hac_lags(len(series_df))
        assert f"lags={expected}" in res["primary"].iloc[0]["method"]

    def test_max_lag_changes_the_confidence_interval(self, series_df):
        tight = self._run(series_df, max_lag=0)["primary"].iloc[0]
        loose = self._run(series_df, max_lag=12)["primary"].iloc[0]

        assert (loose["ci_high"] - loose["ci_low"]) > (tight["ci_high"] - tight["ci_low"])

    def test_max_lag_is_ignored_by_non_temporal_engines(self, series_df):
        """Setting max_lag must not break engines that do not use HAC variance."""
        res = run_tost(
            series_df,
            y="diff",
            margins=[0.5],
            alpha=0.05,
            engine="iid",
            options=WorkflowOptions(do_sensitivity=False, bootstrap_B=0, max_lag=7),
        )
        assert res["engine"] == "iid"

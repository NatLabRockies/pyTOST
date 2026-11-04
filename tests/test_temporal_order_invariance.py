"""Regression tests for RR-001: temporal results must not depend on input row order.

`TemporalTOST` builds Newey--West autocovariances from the sequence obtained by
ordering rows on the time column. When rows share a time value their relative order
used to be whatever the caller supplied, so the same data in a different order produced
a different confidence interval.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from pyTOST import WorkflowOptions, run_tost
from pyTOST.engines.temporal_tost import TemporalTOST

ALPHA = 0.05
MARGINS = [0.25]


@pytest.fixture
def panel_df():
    """A panel: several series observed at the same 40 time points."""
    rng = np.random.default_rng(2026)
    rows = []
    for series in range(3):
        level = rng.normal(0.15, 0.2)
        for t in range(40):
            rows.append({"time": t, "diff": level + rng.normal(0, 0.3), "series": series})
    return pd.DataFrame(rows)


@pytest.fixture
def series_df():
    """A single series: one observation per time point, no ties."""
    rng = np.random.default_rng(7)
    return pd.DataFrame({"time": np.arange(60), "diff": rng.normal(0.1, 0.2, size=60)})


def _fit(df, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return TemporalTOST("diff", "time", **kwargs).fit(df, ALPHA, MARGINS).iloc[0]


class TestOrderInvariance:
    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 17])
    def test_shuffling_rows_does_not_change_the_interval(self, panel_df, seed):
        reference = _fit(panel_df)
        shuffled = _fit(panel_df.sample(frac=1.0, random_state=seed).reset_index(drop=True))

        assert shuffled["ci_low"] == pytest.approx(reference["ci_low"], abs=1e-12)
        assert shuffled["ci_high"] == pytest.approx(reference["ci_high"], abs=1e-12)

    def test_shuffling_rows_does_not_change_the_point_estimate(self, panel_df):
        reference = _fit(panel_df)
        shuffled = _fit(panel_df.sample(frac=1.0, random_state=99).reset_index(drop=True))

        assert shuffled["mu_hat"] == pytest.approx(reference["mu_hat"], abs=1e-12)

    def test_a_non_default_index_does_not_change_the_result(self, panel_df):
        reference = _fit(panel_df)
        reindexed = panel_df.copy()
        reindexed.index = np.arange(1000, 1000 + len(reindexed))[::-1]

        assert _fit(reindexed)["ci_low"] == pytest.approx(reference["ci_low"], abs=1e-12)

    def test_unique_time_series_is_also_order_invariant(self, series_df):
        reference = _fit(series_df)
        shuffled = _fit(series_df.sample(frac=1.0, random_state=5).reset_index(drop=True))

        assert shuffled["ci_low"] == pytest.approx(reference["ci_low"], abs=1e-12)

    def test_ordering_by_time_is_still_respected(self, series_df):
        """Order invariance must not come from discarding the time ordering.

        Permuting the time labels (rather than reversing them, which HAC
        autocovariances are invariant to) must change the estimated dependence.
        """
        sorted_fit = _fit(series_df)
        rng = np.random.default_rng(3)
        scrambled = series_df.assign(
            time=rng.permutation(series_df["time"].to_numpy())
        )

        assert _fit(scrambled)["ci_low"] != pytest.approx(sorted_fit["ci_low"])


class TestPanelAggregation:
    """Replicated time points are averaged into a series (RR-001).

    Treating cross-sectional replicates as consecutive time points misrepresents the
    autocorrelation completely, so the engine reduces a panel to one observation per
    time point before fitting.
    """

    def test_a_balanced_panel_is_averaged_to_one_point_per_time(self, panel_df):
        result = TemporalTOST("diff", "time").fit(panel_df, ALPHA, MARGINS)

        assert "averaged to 40 time points" in result.iloc[0]["method"]

    def test_a_balanced_panel_does_not_warn(self, panel_df):
        """Equal replication per time point leaves the estimand unchanged."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            TemporalTOST("diff", "time").fit(panel_df, ALPHA, MARGINS)

    def test_aggregating_matches_fitting_the_averaged_series_directly(self, panel_df):
        collapsed = (
            panel_df.groupby("time", as_index=False)["diff"].mean().sort_values("time")
        )

        from_panel = TemporalTOST("diff", "time").fit(panel_df, ALPHA, MARGINS).iloc[0]
        from_series = TemporalTOST("diff", "time").fit(collapsed, ALPHA, MARGINS).iloc[0]

        assert from_panel["mu_hat"] == pytest.approx(from_series["mu_hat"])
        assert from_panel["ci_low"] == pytest.approx(from_series["ci_low"])
        assert from_panel["ci_high"] == pytest.approx(from_series["ci_high"])

    def test_an_unbalanced_panel_warns_that_time_points_are_equally_weighted(self, panel_df):
        unbalanced = panel_df.drop(panel_df.index[:2])

        with pytest.warns(UserWarning, match="unequal numbers of observations"):
            TemporalTOST("diff", "time").fit(unbalanced, ALPHA, MARGINS)

    def test_a_clean_series_does_not_warn_or_report_aggregation(self, series_df):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = TemporalTOST("diff", "time").fit(series_df, ALPHA, MARGINS)

        assert "averaged to" not in result.iloc[0]["method"]

    def test_require_unique_times_still_raises(self, panel_df):
        with pytest.raises(ValueError, match="duplicate time"):
            TemporalTOST("diff", "time", require_unique_times=True).fit(
                panel_df, ALPHA, MARGINS
            )

    def test_the_workflow_path_also_aggregates(self, panel_df):
        res = run_tost(
            panel_df,
            y="diff",
            margins=MARGINS,
            alpha=ALPHA,
            engine="temporal",
            time="time",
            options=WorkflowOptions(do_sensitivity=False, bootstrap_B=0),
        )

        assert "averaged to 40 time points" in res["primary"].iloc[0]["method"]

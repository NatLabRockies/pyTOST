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


class TestTiedTimeWarning:
    def test_tied_times_warn_that_hac_assumes_one_series(self, panel_df):
        with pytest.warns(UserWarning, match="duplicate time"):
            TemporalTOST("diff", "time").fit(panel_df, ALPHA, MARGINS)

    def test_the_warning_names_the_engine_and_suggests_alternatives(self, panel_df):
        with pytest.warns(UserWarning) as caught:
            TemporalTOST("diff", "time").fit(panel_df, ALPHA, MARGINS)

        message = str(caught[0].message)
        assert "TemporalTOST" in message
        assert "spatiotemporal" in message or "cluster" in message

    def test_a_clean_series_does_not_warn(self, series_df):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            TemporalTOST("diff", "time").fit(series_df, ALPHA, MARGINS)

    def test_require_unique_times_still_raises_instead_of_warning(self, panel_df):
        with pytest.raises(ValueError, match="duplicate time"):
            TemporalTOST("diff", "time", require_unique_times=True).fit(
                panel_df, ALPHA, MARGINS
            )

    def test_the_warning_reaches_users_of_the_workflow(self, panel_df):
        with pytest.warns(UserWarning, match="duplicate time"):
            run_tost(
                panel_df,
                y="diff",
                margins=MARGINS,
                alpha=ALPHA,
                engine="temporal",
                time="time",
                options=WorkflowOptions(do_sensitivity=False, bootstrap_B=0),
            )

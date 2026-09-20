"""Tests for the configurable trimmed-mean path in RobustLocationTOST."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from pyTOST import RobustLocationTOST, WorkflowOptions, run_tost
from pyTOST.engines.robust_location_tost import _statistic

ALPHA = 0.05
MARGINS = [1.0]


@pytest.fixture
def skewed_values():
    rng = np.random.default_rng(5)
    values = rng.normal(0.2, 1.0, size=60)
    values[:4] += 12.0  # a few strong outliers in one tail
    return values


@pytest.fixture
def frame(skewed_values):
    return pd.DataFrame({"diff": skewed_values})


class TestStatistic:
    def test_trimmed_mean_matches_scipy(self, skewed_values):
        for trim in (0.0, 0.1, 0.2, 0.25, 0.4):
            expected = stats.trim_mean(skewed_values, trim)
            assert _statistic(skewed_values, "trimmed_mean", trim=trim) == pytest.approx(
                expected
            )

    def test_zero_trim_equals_the_mean(self, skewed_values):
        assert _statistic(skewed_values, "trimmed_mean", trim=0.0) == pytest.approx(
            float(np.mean(skewed_values))
        )

    def test_heavier_trimming_moves_away_from_the_outliers(self, skewed_values):
        untrimmed = _statistic(skewed_values, "trimmed_mean", trim=0.0)
        trimmed = _statistic(skewed_values, "trimmed_mean", trim=0.25)

        assert trimmed < untrimmed

    def test_legacy_trimmed_mean_20_still_trims_twenty_percent(self, skewed_values):
        assert _statistic(skewed_values, "trimmed_mean_20") == pytest.approx(
            _statistic(skewed_values, "trimmed_mean", trim=0.2)
        )

    def test_median_is_unchanged(self, skewed_values):
        assert _statistic(skewed_values, "median") == pytest.approx(
            float(np.median(skewed_values))
        )


class TestValidation:
    def test_unknown_statistic_is_rejected_rather_than_silently_averaged(self):
        """A misspelled statistic previously fell through to the mean without warning."""
        with pytest.raises(ValueError, match="mediam|stat"):
            RobustLocationTOST(y="diff", stat="mediam")

    @pytest.mark.parametrize("trim", [-0.1, 0.5, 0.75, 1.0])
    def test_invalid_trim_fraction_is_rejected(self, trim):
        with pytest.raises(ValueError, match="trim"):
            RobustLocationTOST(y="diff", stat="trimmed_mean", trim=trim)

    @pytest.mark.parametrize("trim", [0.0, 0.1, 0.49])
    def test_valid_trim_fraction_is_accepted(self, trim):
        assert RobustLocationTOST(y="diff", stat="trimmed_mean", trim=trim).trim == trim


class TestEngine:
    def _fit(self, frame, **kwargs):
        params = dict(y="diff", B=100, seed=42, stat="trimmed_mean")
        params.update(kwargs)
        return RobustLocationTOST(**params).fit(frame, ALPHA, MARGINS)

    def test_point_estimate_uses_the_requested_trim(self, frame, skewed_values):
        result = self._fit(frame, trim=0.25)

        assert result.iloc[0]["mu_hat"] == pytest.approx(
            stats.trim_mean(skewed_values, 0.25)
        )

    def test_method_label_reports_the_trim_fraction(self, frame):
        label = self._fit(frame, trim=0.25).iloc[0]["method"]

        assert "trim" in label.lower()
        assert "0.25" in label

    def test_default_trim_is_twenty_percent(self, frame, skewed_values):
        result = self._fit(frame)

        assert result.iloc[0]["mu_hat"] == pytest.approx(
            stats.trim_mean(skewed_values, 0.2)
        )

    def test_different_trims_give_different_intervals(self, frame):
        light = self._fit(frame, trim=0.05).iloc[0]
        heavy = self._fit(frame, trim=0.40).iloc[0]

        assert light["mu_hat"] != pytest.approx(heavy["mu_hat"])

    def test_results_are_reproducible_for_a_fixed_seed(self, frame):
        first = self._fit(frame, trim=0.3)
        second = self._fit(frame, trim=0.3)

        pd.testing.assert_frame_equal(first, second)

    def test_median_path_is_unaffected_by_trim(self, frame):
        with_trim = self._fit(frame, stat="median", trim=0.4).iloc[0]
        without = self._fit(frame, stat="median").iloc[0]

        assert with_trim["mu_hat"] == pytest.approx(without["mu_hat"])


class TestWorkflowIntegration:
    def test_trim_is_configurable_through_workflow_options(self, frame, skewed_values):
        res = run_tost(
            frame,
            y="diff",
            margins=MARGINS,
            alpha=ALPHA,
            engine="iid",
            options=WorkflowOptions(
                do_sensitivity=True,
                bootstrap_B=0,
                robust_location_B=80,
                robust_location_stat="trimmed_mean",
                robust_location_trim=0.3,
                seed=42,
            ),
        )

        row = res["sensitivity"]["Robust Location"].iloc[0]
        assert row["mu_hat"] == pytest.approx(stats.trim_mean(skewed_values, 0.3))
        assert "0.3" in row["method"]

    def test_workflow_default_still_uses_the_median(self, frame, skewed_values):
        res = run_tost(
            frame,
            y="diff",
            margins=MARGINS,
            alpha=ALPHA,
            engine="iid",
            options=WorkflowOptions(
                do_sensitivity=True, bootstrap_B=0, robust_location_B=80, seed=42
            ),
        )

        row = res["sensitivity"]["Robust Location"].iloc[0]
        assert row["mu_hat"] == pytest.approx(float(np.median(skewed_values)))

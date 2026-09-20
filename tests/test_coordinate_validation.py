"""Tests for informative errors when spatial coordinates are missing or non-finite.

Without validation these inputs surface deep inside the covariance optimizer as an
opaque failure (for example ``KeyError: 'theta_log'``), which gives the user no
indication that the real problem is a bad coordinate column.
"""

import numpy as np
import pandas as pd
import pytest

from pyTOST import (
    BuildingAwareConfig,
    BuildingAwareSpatioTemporalTOST,
    SpatialConfig,
    SpatialTOST,
    SpatioTemporalConfig,
    SpatioTemporalTOST,
    WorkflowOptions,
    run_tost,
)


@pytest.fixture
def spatial_frame():
    rng = np.random.default_rng(3)
    return pd.DataFrame(
        {
            "diff": 0.10 + rng.normal(0.0, 0.01, size=12),
            "cluster_id": np.repeat(list("ABC"), 4),
            "x": np.tile([0.0, 1.0, 0.0, 1.0], 3),
            "ycoord": np.tile([0.0, 0.0, 1.0, 1.0], 3),
        }
    )


@pytest.fixture
def spatiotemporal_frame():
    rng = np.random.default_rng(3)
    rows = []
    for t in range(4):
        for cluster, (bx, by) in {"A": (0.0, 0.0), "B": (5.0, 5.0)}.items():
            for xoff, yoff in [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]:
                rows.append(
                    {
                        "cluster_id": cluster,
                        "time": t,
                        "x": bx + xoff,
                        "ycoord": by + yoff,
                    }
                )
    frame = pd.DataFrame(rows)
    frame["diff"] = 0.10 + rng.normal(0.0, 0.01, size=len(frame))
    return frame


def _spatial(frame):
    return SpatialTOST(
        y="diff", cluster="cluster_id", x="x", ycoord="ycoord", config=SpatialConfig()
    ).fit(frame, 0.05, [0.5])


def _spatiotemporal(frame):
    return SpatioTemporalTOST(
        y="diff",
        cluster="cluster_id",
        time="time",
        x="x",
        ycoord="ycoord",
        config=SpatioTemporalConfig(),
    ).fit(frame, 0.05, [0.5])


class TestSpatialEngine:
    @pytest.mark.parametrize("column", ["x", "ycoord"])
    def test_nan_coordinate_raises_value_error(self, spatial_frame, column):
        spatial_frame.loc[3, column] = np.nan

        with pytest.raises(ValueError) as excinfo:
            _spatial(spatial_frame)

        message = str(excinfo.value)
        assert "SpatialTOST" in message
        assert column in message
        assert "NaN" in message or "non-finite" in message

    def test_error_reports_how_many_rows_are_affected(self, spatial_frame):
        spatial_frame.loc[[2, 5, 7], "x"] = np.nan

        with pytest.raises(ValueError, match="3"):
            _spatial(spatial_frame)

    def test_error_identifies_the_offending_rows(self, spatial_frame):
        spatial_frame.loc[5, "ycoord"] = np.nan

        with pytest.raises(ValueError) as excinfo:
            _spatial(spatial_frame)

        assert "5" in str(excinfo.value)

    def test_infinite_coordinate_is_also_rejected(self, spatial_frame):
        spatial_frame.loc[1, "x"] = np.inf

        with pytest.raises(ValueError) as excinfo:
            _spatial(spatial_frame)

        assert "x" in str(excinfo.value)

    def test_clean_coordinates_still_fit(self, spatial_frame):
        result = _spatial(spatial_frame)

        assert len(result) == 1
        assert np.isfinite(result.iloc[0]["mu_hat"])


class TestSpatioTemporalEngine:
    @pytest.mark.parametrize("column", ["x", "ycoord"])
    def test_nan_coordinate_raises_value_error(self, spatiotemporal_frame, column):
        spatiotemporal_frame.loc[2, column] = np.nan

        with pytest.raises(ValueError) as excinfo:
            _spatiotemporal(spatiotemporal_frame)

        message = str(excinfo.value)
        assert "SpatioTemporalTOST" in message
        assert column in message

    def test_clean_coordinates_still_fit(self, spatiotemporal_frame):
        result = _spatiotemporal(spatiotemporal_frame)

        assert len(result) == 1
        assert np.isfinite(result.iloc[0]["mu_hat"])


class TestBuildingAwareEngine:
    def test_nan_coordinate_raises_value_error(self, spatiotemporal_frame):
        spatiotemporal_frame.loc[4, "x"] = np.nan

        with pytest.raises(ValueError) as excinfo:
            BuildingAwareSpatioTemporalTOST(
                y="diff",
                cluster="cluster_id",
                time="time",
                x="x",
                ycoord="ycoord",
                config=BuildingAwareConfig(),
            ).fit(spatiotemporal_frame, 0.05, [0.5])

        assert "x" in str(excinfo.value)


class TestThroughWorkflow:
    def test_run_tost_surfaces_the_informative_error(self, spatial_frame):
        spatial_frame.loc[0, "x"] = np.nan

        with pytest.raises(ValueError) as excinfo:
            run_tost(
                spatial_frame,
                y="diff",
                margins=[0.5],
                alpha=0.05,
                engine="spatial",
                cluster="cluster_id",
                x="x",
                ycoord="ycoord",
                options=WorkflowOptions(do_sensitivity=False, bootstrap_B=0),
            )

        message = str(excinfo.value)
        assert "x" in message
        assert "NaN" in message or "non-finite" in message

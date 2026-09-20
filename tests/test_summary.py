"""Tests for the human-readable equivalence decision summary."""

import pandas as pd
import pytest

from pyTOST import WorkflowOptions, run_tost
from pyTOST.results import TOSTResult


@pytest.fixture
def cluster_df():
    return pd.DataFrame(
        {
            "diff": [0.10, 0.18, 0.05, 0.12, 0.08, 0.15],
            "cluster_id": ["A", "A", "B", "B", "C", "C"],
        }
    )


def _run(df, **kwargs):
    params = dict(
        y="diff",
        margins=[0.5],
        alpha=0.05,
        engine="cluster",
        cluster="cluster_id",
        options=WorkflowOptions(
            do_sensitivity=False, bootstrap_B=0, robust_location_B=10, seed=42
        ),
    )
    params.update(kwargs)
    return run_tost(df, **params)


def test_run_tost_returns_a_mapping_that_is_still_a_dict(cluster_df):
    """Backwards compatibility: the result must keep behaving like a plain dict."""
    res = _run(cluster_df)

    assert isinstance(res, dict)
    assert isinstance(res, TOSTResult)
    assert res["engine"] == "cluster"
    assert set(res.keys()) >= {"engine", "primary"}
    assert isinstance(res["primary"], pd.DataFrame)


def test_summary_reports_engine_and_decision(cluster_df):
    res = _run(cluster_df)
    text = res.summary()

    assert isinstance(text, str)
    assert "cluster" in text
    # margin, point estimate and both CI bounds are all reported
    assert "0.5" in text
    row = res["primary"].iloc[0]
    assert f"{row['mu_hat']:.4f}" in text
    assert f"{row['ci_low']:.4f}" in text
    assert f"{row['ci_high']:.4f}" in text


def test_summary_decision_matches_equivalent_column(cluster_df):
    """A wide margin is equivalent, a tiny margin is not; the table must say so."""
    wide = _run(cluster_df, margins=[5.0])
    narrow = _run(cluster_df, margins=[0.0001])

    assert bool(wide["primary"].iloc[0]["equivalent"]) is True
    assert bool(narrow["primary"].iloc[0]["equivalent"]) is False

    assert "EQUIVALENT" in wide.summary()
    assert "NOT EQUIVALENT" in narrow.summary()


def test_summary_lists_every_margin(cluster_df):
    res = _run(cluster_df, margins=[0.25, 0.5, 1.0])
    text = res.summary()

    # margins render compactly (1.0 -> "1"), one body row per margin
    for margin in ("0.25", "0.5", "1"):
        assert margin in text
    assert text.count("EQUIVALENT") == 3


def test_summary_includes_sensitivity_and_bootstrap_when_present(cluster_df):
    res = _run(
        cluster_df,
        options=WorkflowOptions(
            do_sensitivity=True, bootstrap_B=50, robust_location_B=20, seed=42
        ),
    )
    text = res.summary()

    assert "Sensitivity" in text
    assert "Heteroskedastic" in text
    assert "Robust Location" in text
    assert "Bootstrap" in text
    assert "cluster_bootstrap" in text


def test_summary_renders_the_bootstrap_interval(cluster_df):
    """The bootstrap section must show the interval, not just the method name."""
    res = _run(
        cluster_df,
        options=WorkflowOptions(
            do_sensitivity=False, bootstrap_B=50, robust_location_B=10, seed=42
        ),
    )
    boot = res["bootstrap"]
    low, high = boot["ci_perc_90"]
    text = res.summary()

    assert "Mean CI" in text
    assert f"{float(low):.4f}" in text
    assert f"{float(high):.4f}" in text
    assert f"Replicates: {int(boot['B'])}" in text


def test_summary_omits_optional_sections_when_absent(cluster_df):
    res = _run(cluster_df)
    text = res.summary()

    assert "Sensitivity" not in text
    assert "Bootstrap" not in text


def test_summary_precision_is_configurable(cluster_df):
    res = _run(cluster_df)
    row = res["primary"].iloc[0]

    assert f"{row['mu_hat']:.2f}" in res.summary(precision=2)
    assert f"{row['mu_hat']:.6f}" in res.summary(precision=6)


def test_summary_is_reachable_from_the_package_namespace():
    import pyTOST

    assert pyTOST.TOSTResult is TOSTResult

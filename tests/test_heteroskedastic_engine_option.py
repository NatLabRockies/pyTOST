"""Tests for the heteroskedastic engine as a first-class run_tost option."""

import pandas as pd
import pytest

from pyTOST import WorkflowOptions, run_tost
from pyTOST.engines.heteroskedastic_tost import HeteroskedasticTOST


@pytest.fixture
def clustered_df():
    return pd.DataFrame(
        {
            "diff": [0.10, 0.18, 0.05, 0.12, 0.08, 0.15, 0.11, 0.09],
            "cluster_id": ["A", "A", "B", "B", "C", "C", "D", "D"],
        }
    )


@pytest.fixture
def unclustered_df():
    return pd.DataFrame({"diff": [0.10, 0.18, 0.05, 0.12, 0.08, 0.15, 0.11, 0.09]})


_NO_EXTRAS = WorkflowOptions(do_sensitivity=False, bootstrap_B=0, seed=42)


def test_heteroskedastic_is_accepted_as_a_primary_engine(unclustered_df):
    res = run_tost(
        unclustered_df,
        y="diff",
        margins=[0.5],
        alpha=0.05,
        engine="heteroskedastic",
        options=_NO_EXTRAS,
    )

    assert res["engine"] == "heteroskedastic"
    primary = res["primary"]
    assert list(primary.columns) >= ["delta", "mu_hat", "ci_low", "ci_high", "equivalent"]
    assert len(primary) == 1
    assert bool(primary.iloc[0]["equivalent"]) is True


def test_heteroskedastic_primary_matches_direct_engine_use(unclustered_df):
    """Routing through run_tost must not change the engine's numbers."""
    expected = HeteroskedasticTOST(y="diff", cluster=None, seed=42).fit(
        unclustered_df, alpha=0.05, margins=[0.5]
    )

    res = run_tost(
        unclustered_df,
        y="diff",
        margins=[0.5],
        alpha=0.05,
        engine="heteroskedastic",
        options=_NO_EXTRAS,
    )

    pd.testing.assert_frame_equal(res["primary"], expected)


def test_heteroskedastic_uses_clusters_when_supplied(clustered_df):
    """Passing `cluster` switches the engine to cluster-robust wild bootstrap."""
    res = run_tost(
        clustered_df,
        y="diff",
        margins=[0.5],
        alpha=0.05,
        engine="heteroskedastic",
        cluster="cluster_id",
        options=_NO_EXTRAS,
    )

    method = res["primary"].iloc[0]["method"]
    assert "Cluster" in method or "Wild" in method


def test_heteroskedastic_respects_the_workflow_seed(clustered_df):
    """The bootstrap path must be reproducible and seed-controlled."""
    kwargs = dict(
        y="diff",
        margins=[0.5],
        alpha=0.05,
        engine="heteroskedastic",
        cluster="cluster_id",
    )

    first = run_tost(clustered_df, **kwargs, options=WorkflowOptions(
        do_sensitivity=False, bootstrap_B=0, seed=7))
    repeat = run_tost(clustered_df, **kwargs, options=WorkflowOptions(
        do_sensitivity=False, bootstrap_B=0, seed=7))

    pd.testing.assert_frame_equal(first["primary"], repeat["primary"])


def test_heteroskedastic_engine_is_case_insensitive(unclustered_df):
    res = run_tost(
        unclustered_df,
        y="diff",
        margins=[0.5],
        alpha=0.05,
        engine="  Heteroskedastic ",
        options=_NO_EXTRAS,
    )

    assert res["engine"] == "heteroskedastic"


def test_heteroskedastic_narrow_margin_is_not_equivalent(unclustered_df):
    res = run_tost(
        unclustered_df,
        y="diff",
        margins=[0.0001],
        alpha=0.05,
        engine="heteroskedastic",
        options=_NO_EXTRAS,
    )

    assert bool(res["primary"].iloc[0]["equivalent"]) is False


def test_heteroskedastic_appears_in_the_unknown_engine_error_message(unclustered_df):
    with pytest.raises(ValueError, match="heteroskedastic"):
        run_tost(
            unclustered_df,
            y="diff",
            margins=[0.5],
            engine="not-an-engine",
            options=_NO_EXTRAS,
        )


def test_heteroskedastic_summary_renders(unclustered_df):
    res = run_tost(
        unclustered_df,
        y="diff",
        margins=[0.5],
        alpha=0.05,
        engine="heteroskedastic",
        options=_NO_EXTRAS,
    )

    assert "heteroskedastic" in res.summary()

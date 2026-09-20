"""Tests for the plot_ci equivalence visualization."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from pyTOST import WorkflowOptions, plot_ci, run_tost


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def result(basic_diff_df):
    return run_tost(
        basic_diff_df,
        y="diff",
        margins=[0.05, 0.25],
        alpha=0.05,
        engine="cluster",
        cluster="cluster_id",
        options=WorkflowOptions(
            do_sensitivity=True, bootstrap_B=0, robust_location_B=20, seed=42
        ),
    )


def _bar_labels(ax):
    return [t.get_text() for t in ax.get_yticklabels()]


def test_plot_ci_returns_figure_and_axes(result):
    fig, ax = plot_ci(result)

    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)


def test_plot_ci_draws_one_bar_per_method(result):
    fig, ax = plot_ci(result)

    labels = _bar_labels(ax)
    # primary engine plus the two sensitivity analyses
    assert "Primary (cluster)" in labels
    assert "Heteroskedastic" in labels
    assert "Robust Location" in labels
    assert len(labels) == 3


def test_plot_ci_marks_the_equivalence_margins(result):
    fig, ax = plot_ci(result, margin=0.25)

    vlines = [line.get_xdata()[0] for line in ax.lines if len(set(line.get_xdata())) == 1]
    assert pytest.approx(0.25) in vlines
    assert pytest.approx(-0.25) in vlines


def test_plot_ci_defaults_to_the_first_margin(result):
    fig, ax = plot_ci(result)

    vlines = [line.get_xdata()[0] for line in ax.lines if len(set(line.get_xdata())) == 1]
    assert pytest.approx(0.05) in vlines


def test_plot_ci_bar_spans_the_confidence_interval(result):
    fig, ax = plot_ci(result, margin=0.25)

    row = result["primary"].query("delta == 0.25").iloc[0]
    spans = [
        (min(line.get_xdata()), max(line.get_xdata()))
        for line in ax.lines
        if len(set(line.get_xdata())) > 1
    ]
    assert any(
        x0 == pytest.approx(row["ci_low"]) and x1 == pytest.approx(row["ci_high"])
        for x0, x1 in spans
    )


def test_plot_ci_accepts_a_plain_mapping_of_frames(result):
    frames = {"Only engine": result["primary"]}
    fig, ax = plot_ci(frames, margin=0.25)

    assert _bar_labels(ax) == ["Only engine"]


def test_plot_ci_rejects_a_margin_that_was_not_computed(result):
    with pytest.raises(ValueError, match="margin"):
        plot_ci(result, margin=99.0)


def test_plot_ci_draws_onto_a_supplied_axes(result):
    fig, ax = plt.subplots()
    out_fig, out_ax = plot_ci(result, ax=ax)

    assert out_ax is ax
    assert out_fig is fig


def test_plot_ci_title_is_configurable(result):
    fig, ax = plot_ci(result, title="SAV equivalence")

    assert ax.get_title() == "SAV equivalence"


def test_plot_ci_is_exported_from_the_package():
    import pyTOST

    assert callable(pyTOST.plot_ci)

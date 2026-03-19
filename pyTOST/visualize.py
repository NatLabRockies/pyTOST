"""Visualization helpers for pyTOST results.

This module provides lightweight plotting utilities for comparing confidence
intervals produced by different inference engines. The functions are intended
for exploratory summaries and demonstration notebooks rather than as a fully
featured plotting API.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_ci_comparison(summary_dict: dict):
    """Plot horizontal confidence intervals by method and equivalence margin.

    Parameters
    ----------
    summary_dict : dict
        Mapping from method name to a :class:`pandas.DataFrame` containing the
        summary results for that method. Each data frame is expected to include
        the columns ``delta``, ``mu_hat``, ``ci_low``, ``ci_high``, and
        ``equivalent``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the confidence-interval comparison plot.
    ax : matplotlib.axes.Axes
        Axes object on which the intervals are drawn.

    Notes
    -----
    Each method is plotted with a small vertical offset so that multiple
    intervals at the same equivalence margin can be compared visually.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    methods = list(summary_dict.keys())
    offsets = np.linspace(-0.2, 0.2, len(methods)) if methods else []
    for (k, df), off in zip(summary_dict.items(), offsets):
        for _, r in df.iterrows():
            ax.plot([r["ci_low"], r["ci_high"]], [r["delta"]+off]*2, marker='|')
    ax.axvline(0.0, linestyle='--', linewidth=1)
    ax.set_xlabel("μ (units)")
    ax.set_ylabel("Δ (equivalence margin)")
    ax.set_title("Confidence Intervals by Method")
    return fig, ax


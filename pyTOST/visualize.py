"""
visualize.py
============
Report-ready visualizations for equivalence results (matplotlib only).
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

__all__ = ["plot_ci", "plot_ci_comparison"]

_EQUIVALENT_COLOR = "#1b7837"
_NOT_EQUIVALENT_COLOR = "#b2182b"


def _collect_frames(result: Any) -> dict[str, pd.DataFrame]:
    """Normalize the accepted inputs into an ordered {label: DataFrame} mapping.

    Accepts either a :func:`pyTOST.run_tost` result (plotting the primary engine and
    any sensitivity analyses) or a plain mapping of label to result DataFrame.
    """
    if not isinstance(result, Mapping):
        raise TypeError(
            "plot_ci expects a run_tost result or a mapping of label -> DataFrame, "
            f"got {type(result).__name__}."
        )

    if isinstance(result.get("primary"), pd.DataFrame):
        engine = result.get("engine", "primary")
        frames = {f"Primary ({engine})": result["primary"]}
        sensitivity = result.get("sensitivity") or {}
        for name, frame in sensitivity.items():
            if isinstance(frame, pd.DataFrame) and not frame.empty:
                frames[str(name)] = frame
        return frames

    frames = {
        str(name): frame
        for name, frame in result.items()
        if isinstance(frame, pd.DataFrame) and not frame.empty
    }
    if not frames:
        raise ValueError("plot_ci found no result tables to plot.")
    return frames


def _resolve_margin(frames: Mapping[str, pd.DataFrame], margin: float | None) -> float:
    available = sorted({float(d) for f in frames.values() for d in f["delta"]})
    if not available:
        raise ValueError("plot_ci found no equivalence margins to plot.")

    if margin is None:
        return available[0]

    target = float(margin)
    if not any(np.isclose(target, value) for value in available):
        raise ValueError(
            f"margin={margin!r} was not computed. Available margins: {available}."
        )
    return target


def _row_for_margin(frame: pd.DataFrame, margin: float):
    match = frame[np.isclose(frame["delta"].astype(float), margin)]
    return None if match.empty else match.iloc[0]


def plot_ci(
    result: Any,
    *,
    margin: float | None = None,
    ax: plt.Axes | None = None,
    title: str | None = None,
):
    """Plot confidence intervals for μ across engines at one equivalence margin.

    Each engine is drawn as a horizontal CI bar with a point marker at μ̂. Dashed
    vertical lines mark the equivalence margins ``-Δ`` and ``+Δ``; a bar lying wholly
    between them is an equivalence decision. Bars are colored by that decision, which
    makes the figure directly usable in a report.

    Parameters
    ----------
    result : TOSTResult or Mapping[str, DataFrame]
        Either a :func:`pyTOST.run_tost` result — in which case the primary engine and
        any sensitivity analyses are plotted — or a mapping of label to result table.
    margin : float, optional
        The equivalence margin ``Δ`` to plot. Defaults to the smallest margin present.
    ax : matplotlib Axes, optional
        Axes to draw on. A new figure and axes are created when omitted.
    title : str, optional
        Plot title. A descriptive default is used when omitted.

    Returns
    -------
    (Figure, Axes)
        The figure and axes containing the plot.

    Raises
    ------
    ValueError
        If ``margin`` was not among the computed margins.

    Examples
    --------
    >>> res = run_tost(df, y="diff", margins=[0.5], engine="cluster",
    ...                cluster="cluster_id")            # doctest: +SKIP
    >>> fig, ax = plot_ci(res, margin=0.5)              # doctest: +SKIP
    >>> fig.savefig("equivalence.png", dpi=200, bbox_inches="tight")  # doctest: +SKIP
    """
    frames = _collect_frames(result)
    delta = _resolve_margin(frames, margin)

    labels: list[str] = []
    rows = []
    for label, frame in frames.items():
        row = _row_for_margin(frame, delta)
        if row is not None:
            labels.append(label)
            rows.append(row)

    if not rows:
        raise ValueError(f"No result table contains margin Δ={delta:g}.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 0.9 * len(rows) + 2.0))
    else:
        fig = ax.figure

    for position, row in enumerate(rows):
        equivalent = bool(row["equivalent"])
        color = _EQUIVALENT_COLOR if equivalent else _NOT_EQUIVALENT_COLOR
        ax.plot(
            [float(row["ci_low"]), float(row["ci_high"])],
            [position, position],
            color=color,
            linewidth=2.5,
            marker="|",
            markersize=10,
            solid_capstyle="butt",
        )
        ax.plot(
            [float(row["mu_hat"])],
            [position],
            marker="o",
            color=color,
            markersize=6,
            linestyle="none",
        )

    ax.axvline(0.0, color="0.5", linestyle=":", linewidth=1.0)
    for bound in (-delta, delta):
        ax.axvline(bound, color="0.2", linestyle="--", linewidth=1.2)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_ylim(-0.6, len(labels) - 0.4)
    ax.invert_yaxis()
    ax.set_xlabel("Mean paired difference μ")
    ax.set_title(title if title is not None else f"Equivalence at Δ = {delta:g}")
    ax.margins(x=0.12)
    return fig, ax


def plot_ci_comparison(summary_dict: dict):
    """
    Plot horizontal CIs for μ across methods per Δ.

    summary_dict: {method_name: DataFrame with columns [delta, mu_hat, ci_low, ci_high, equivalent]}
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


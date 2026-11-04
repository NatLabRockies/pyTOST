"""Generate the JOSS paper figure comparing engine CIs on synthetic datasets.

The figure shows why dependence-aware inference matters, in two panels.

Panel A is a spatial panel: 24 sites observed at 8 times. Four engines estimate the
same mean paired difference from identical data, but the IID engine, which ignores the
spatial dependence genuinely present, reports a materially narrower confidence interval
and so declares equivalence at a margin where the dependence-aware engines do not.

Panel B is a single time series, because that is what the temporal engine targets. A
pure-temporal model cannot be applied honestly to panel A: with one observation per time
point it would collapse 24 sites into 8 values and discard the spatial design, so it is
demonstrated on data matched to its assumptions instead. The heteroskedastic engine is
omitted from both panels because neither dataset has non-constant variance.

Run with::

    pixi run -e test python scripts/make_paper_figure.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pyTOST import (
    ClusterTOST,
    IIDTOST,
    SpatialConfig,
    SpatialTOST,
    SpatioTemporalConfig,
    SpatioTemporalTOST,
    TemporalTOST,
    plot_ci,
)
from pyTOST.data_gen.synthetic_tost_data import (
    ar1_cov,
    generate_spatiotemporal,
    safe_cholesky,
)

SEED = 20260920
N_SPACE = 24
N_TIME = 8
N_SERIES_TIME = 120
SERIES_RHO = 0.75
TRUE_SHIFT = 0.25
MARGIN = 0.60
ALPHA = 0.05
OUTPUT = Path(__file__).resolve().parents[1] / "paper_figure.png"


def build_paired_differences() -> pd.DataFrame:
    """Create a spatiotemporal panel and reduce it to paired differences."""
    long_df, _meta = generate_spatiotemporal(
        n_space=N_SPACE,
        n_time=N_TIME,
        length_scale=1.2,
        rho=0.75,
        spatial_sd=1.0,
        obs_sd=0.10,
        delta=0.25,
        seed=SEED,
    )

    wide = (
        long_df.pivot_table(
            index=["sample_id", "x", "y_sp", "t"], columns="arm", values="y"
        )
        .reset_index()
        .rename(columns={"y_sp": "ycoord", "t": "time"})
    )
    wide.columns.name = None
    wide["diff"] = wide["B"] - wide["A"]

    # Group sites into spatial clusters on a 2x2 partition of the domain.
    x_mid = wide["x"].median()
    y_mid = wide["ycoord"].median()
    wide["cluster_id"] = np.where(wide["x"] <= x_mid, "W", "E") + np.where(
        wide["ycoord"] <= y_mid, "S", "N"
    )

    return wide[["diff", "cluster_id", "x", "ycoord", "time"]]


def build_autocorrelated_series() -> pd.DataFrame:
    """Create a single series whose paired difference is itself autocorrelated.

    The temporal engine assumes one observation per time point and dependence in the
    quantity being tested. A paired difference is autocorrelated when the two arms do
    not share the same latent process, as when they are measured by different
    instruments that each drift over time.
    """
    rng = np.random.default_rng(SEED)
    cov = ar1_cov(N_SERIES_TIME, rho=SERIES_RHO, variance=1.0)
    diff = TRUE_SHIFT + safe_cholesky(cov) @ rng.normal(size=N_SERIES_TIME)

    return pd.DataFrame({"diff": diff, "time": np.arange(N_SERIES_TIME)})


def fit_all_engines(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Fit every panel-appropriate engine to the same data at a single margin.

    Only the spatial and spatiotemporal engines accept ``equal_weighted`` mode; running
    them that way makes all four plotted engines target the same estimand, the mean
    paired difference, so the comparison isolates the effect of the dependence
    assumption on CI width rather than confounding it with a change of estimand from
    GLS reweighting.

    The temporal engine is excluded: it models a single series, so on this panel it
    would average the 24 sites at each time into one value, leaving 8 observations and
    discarding the spatial structure the other engines use. See
    :func:`fit_series_engines`.
    """
    margins = [MARGIN]

    return {
        "IID (ignores dependence)": IIDTOST("diff").fit(df, ALPHA, margins),
        "Cluster-robust": ClusterTOST("diff", "cluster_id").fit(df, ALPHA, margins),
        "Spatial (Matérn)": SpatialTOST(
            y="diff",
            cluster="cluster_id",
            x="x",
            ycoord="ycoord",
            config=SpatialConfig(point_estimator="equal_weighted"),
        ).fit(df, ALPHA, margins),
        "Spatiotemporal": SpatioTemporalTOST(
            y="diff",
            cluster="cluster_id",
            time="time",
            x="x",
            ycoord="ycoord",
            config=SpatioTemporalConfig(point_estimator="equal_weighted"),
        ).fit(df, ALPHA, margins),
    }


def fit_series_engines(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Contrast IID and temporal inference on a single autocorrelated series."""
    margins = [MARGIN]

    return {
        "IID (ignores dependence)": IIDTOST("diff").fit(df, ALPHA, margins),
        "Temporal (AR GLS)": TemporalTOST("diff", "time").fit(df, ALPHA, margins),
    }


def _report(heading: str, results: dict[str, pd.DataFrame]) -> None:
    print(f"\n{heading}")
    for label, frame in results.items():
        row = frame.iloc[0]
        width = float(row["ci_high"]) - float(row["ci_low"])
        decision = "equivalent" if bool(row["equivalent"]) else "not equivalent"
        print(
            f"  {label:28s} mu={row['mu_hat']:+.3f} "
            f"CI=[{row['ci_low']:+.3f}, {row['ci_high']:+.3f}] "
            f"width={width:.3f}  {decision}"
        )


def main() -> None:
    panel = build_paired_differences()
    panel_results = fit_all_engines(panel)
    _report(f"Panel A: spatial panel, {len(panel)} observations", panel_results)

    series = build_autocorrelated_series()
    series_results = fit_series_engines(series)
    _report(f"Panel B: single series, {len(series)} time points", series_results)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8, 6.4),
        gridspec_kw={"height_ratios": [len(panel_results), len(series_results)]},
    )

    plot_ci(
        panel_results,
        margin=MARGIN,
        ax=axes[0],
        title=f"A. Spatial panel: {N_SPACE} sites × {N_TIME} times (Δ = {MARGIN:g})",
    )
    plot_ci(
        series_results,
        margin=MARGIN,
        ax=axes[1],
        title=(
            f"B. Single autocorrelated series: {N_SERIES_TIME} time points, "
            f"ρ = {SERIES_RHO:g}"
        ),
    )

    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=200, bbox_inches="tight")
    print(f"\nwrote {OUTPUT}")


if __name__ == "__main__":
    main()

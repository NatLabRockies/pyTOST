"""Generate the JOSS paper figure comparing engine CIs on one synthetic dataset.

The figure shows why dependence-aware inference matters: all five engines estimate the
same mean paired difference from identical data, but the IID engine — which ignores the
spatial and temporal dependence that is genuinely present — reports a materially
narrower confidence interval, and therefore declares equivalence at a margin where the
dependence-aware engines do not.

Run with::

    pixi run -e test python scripts/make_paper_figure.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

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
from pyTOST.data_gen.synthetic_tost_data import generate_spatiotemporal

SEED = 20260920
N_SPACE = 24
N_TIME = 8
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


def fit_all_engines(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Fit every engine to the same data at a single equivalence margin.

    The spatial and spatiotemporal engines are run in ``equal_weighted`` mode so that
    all five engines target the same estimand — the mean paired difference — and the
    comparison isolates the effect of the dependence assumption on the CI width rather
    than confounding it with a change of estimand from GLS reweighting.
    """
    margins = [MARGIN]

    return {
        "IID (ignores dependence)": IIDTOST("diff").fit(df, ALPHA, margins),
        "Cluster-robust": ClusterTOST("diff", "cluster_id").fit(df, ALPHA, margins),
        "Temporal (HAC)": TemporalTOST("diff", "time").fit(df, ALPHA, margins),
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


def main() -> None:
    df = build_paired_differences()
    results = fit_all_engines(df)

    for label, frame in results.items():
        row = frame.iloc[0]
        width = float(row["ci_high"]) - float(row["ci_low"])
        decision = "equivalent" if bool(row["equivalent"]) else "not equivalent"
        print(
            f"{label:28s} mu={row['mu_hat']:+.3f} "
            f"CI=[{row['ci_low']:+.3f}, {row['ci_high']:+.3f}] "
            f"width={width:.3f}  {decision}"
        )

    fig, ax = plot_ci(
        results,
        margin=MARGIN,
        title=f"Confidence intervals for μ by engine (Δ = {MARGIN:g})",
    )
    fig.savefig(OUTPUT, dpi=200, bbox_inches="tight")
    print(f"\nwrote {OUTPUT}")


if __name__ == "__main__":
    main()

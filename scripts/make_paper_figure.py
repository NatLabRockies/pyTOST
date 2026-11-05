"""Generate the JOSS paper figure comparing engine CIs on synthetic datasets.

The figure shows why dependence-aware inference matters, in three panels, one per
dependence structure. Every engine the package ships appears exactly once in the role it
is designed for, and the IID engine appears in all three as the common reference.

Panel A is a spatial panel: 24 sites observed at 8 times. Four engines estimate the
same mean paired difference from identical data, but the IID engine, which ignores the
spatial dependence genuinely present, reports a materially narrower confidence interval
and so declares equivalence at a margin where the dependence-aware engines do not.

Panel B is a single time series, because that is what the temporal engine targets. A
pure-temporal model cannot be applied honestly to panel A: with one observation per time
point it would collapse 24 sites into 8 values and discard the spatial design, so it is
demonstrated on data matched to its assumptions instead.

Panel C has unequal variances across clusters, which is the heteroskedastic engine's
target. It is a separate panel rather than an extra bar on panel A because panels A and
B have constant variance, where heteroskedasticity-robust intervals and IID intervals
coincide for a mean: non-constant variance alone does not bias the ordinary standard
error of an intercept-only model. Plotting the engine on constant-variance data would
therefore imply a correction it does not make. The cluster-robust engine is repeated in
panel C as a control, showing that the wild cluster bootstrap reaches a comparable
interval by a different route rather than a wider one.

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
    IIDTOST,
    ClusterTOST,
    HeteroskedasticTOST,
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
N_HET_CLUSTERS = 14
HET_PER_CLUSTER = 10
HET_SD_LOW = 0.4
HET_SD_HIGH = 2.0
HET_CLUSTER_EFFECT = 0.4
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


def build_unequal_variance_clusters() -> pd.DataFrame:
    """Create clustered data whose clusters differ markedly in variance.

    Cluster standard deviations are spread evenly over a five-fold range, and each
    cluster also carries a mean offset scaled to its own noise level, so the data are
    heteroskedastic and clustered at once. That pairing is what the heteroskedastic
    engine's cluster-robust path is for: with unequal variances the few noisiest
    clusters dominate the sampling variance of the mean, which an IID interval does not
    register.
    """
    rng = np.random.default_rng(SEED)
    sds = np.linspace(HET_SD_LOW, HET_SD_HIGH, N_HET_CLUSTERS)

    frames = []
    for cluster_id, sd in enumerate(sds):
        offset = rng.normal(0.0, sd * HET_CLUSTER_EFFECT)
        noise = rng.normal(0.0, sd, size=HET_PER_CLUSTER)
        frames.append(
            pd.DataFrame(
                {"diff": TRUE_SHIFT + offset + noise, "cluster_id": cluster_id}
            )
        )

    return pd.concat(frames, ignore_index=True)


def fit_variance_engines(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Contrast IID, cluster-robust, and wild cluster bootstrap intervals."""
    margins = [MARGIN]

    return {
        "IID (ignores dependence)": IIDTOST("diff").fit(df, ALPHA, margins),
        "Cluster-robust": ClusterTOST("diff", "cluster_id").fit(df, ALPHA, margins),
        "Heteroskedastic (wild bootstrap)": HeteroskedasticTOST(
            "diff", cluster="cluster_id", seed=SEED
        ).fit(df, ALPHA, margins),
    }


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

    variance = build_unequal_variance_clusters()
    variance_results = fit_variance_engines(variance)
    _report(
        f"Panel C: unequal cluster variances, {len(variance)} observations",
        variance_results,
    )

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(8, 9.0),
        gridspec_kw={
            "height_ratios": [
                len(panel_results),
                len(series_results),
                len(variance_results),
            ]
        },
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
    plot_ci(
        variance_results,
        margin=MARGIN,
        ax=axes[2],
        title=(
            f"C. Unequal cluster variances: {N_HET_CLUSTERS} clusters × "
            f"{HET_PER_CLUSTER}, SD {HET_SD_LOW:g}–{HET_SD_HIGH:g}"
        ),
    )

    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=200, bbox_inches="tight")
    print(f"\nwrote {OUTPUT}")


if __name__ == "__main__":
    main()

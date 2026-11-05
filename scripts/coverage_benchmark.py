"""Monte Carlo coverage study for the spatial and spatiotemporal engines.

The test suite validates coverage for the IID, cluster, and temporal engines on every
commit. The spatial engine cannot join them: each replicate fits a Gaussian process by
profile maximum likelihood, and the cost is driven by the optimizer rather than by the
sample size, so a defensible study takes minutes rather than seconds. This script runs
that study offline and writes a dated artifact so the result is auditable without
rerunning it.

Both engines are run in ``equal_weighted`` mode, which makes the estimand the mean
paired difference. The generators shift arm B by a known ``delta``, so that mean is
``delta`` exactly and coverage is the fraction of replicates whose interval contains it.

Note that the spatial generator draws from a squared-exponential covariance while the
engines fit a Matérn model. Coverage is therefore measured under a misspecified
covariance function, which is the honest test: it does not reward the estimator for
sharing a parametric form with the generator.

Run with::

    pixi run -e test python scripts/coverage_benchmark.py
    pixi run -e test python scripts/coverage_benchmark.py --replicates 50   # quick check
"""

from __future__ import annotations

import argparse
import contextlib
import io
import platform
import sys
import time
from datetime import date
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from pyTOST import (
    SpatialConfig,
    SpatialTOST,
    SpatioTemporalConfig,
    SpatioTemporalTOST,
)
from pyTOST.data_gen.synthetic_tost_data import (
    generate_spatial_clusters,
    generate_spatiotemporal,
)

ALPHA = 0.05
NOMINAL = 1 - 2 * ALPHA
MARGINS = [0.5]
TRUE_MU = 0.15
REPLICATES = 300

OUTPUT = Path(__file__).resolve().parents[1] / "docs" / "coverage_benchmark.md"

SPATIAL_DESIGN = {"n_clusters": 10, "points_per_cluster": 6, "length_scale": 0.5}
SPATIOTEMPORAL_DESIGN = {"n_space": 18, "n_time": 8, "length_scale": 0.6, "rho": 0.5}


def _long_to_diff(long_df: pd.DataFrame, index_cols: list[str]) -> pd.DataFrame:
    """Pivot a two-arm long frame into paired differences (arm B minus arm A)."""
    wide = long_df.pivot_table(index=index_cols, columns="arm", values="y").reset_index()
    wide.columns.name = None
    wide["diff"] = wide["B"] - wide["A"]
    return wide


def spatial_replicate(seed: int) -> pd.DataFrame:
    long_df, _meta = generate_spatial_clusters(
        field_sd=0.8, nugget_sd=0.2, delta=TRUE_MU, seed=seed, **SPATIAL_DESIGN
    )
    return _long_to_diff(
        long_df, ["sample_id", "group_id", "x", "y_sp"]
    ).rename(columns={"group_id": "cluster_id", "y_sp": "ycoord"})


def spatiotemporal_replicate(seed: int) -> pd.DataFrame:
    long_df, _meta = generate_spatiotemporal(
        spatial_sd=0.8, obs_sd=0.2, delta=TRUE_MU, seed=seed, **SPATIOTEMPORAL_DESIGN
    )
    df = _long_to_diff(long_df, ["sample_id", "x", "y_sp", "t"]).rename(
        columns={"y_sp": "ycoord", "t": "time"}
    )
    # Each distinct coordinate pair is one site, which is the clustering unit here.
    sites: dict[tuple[float, float], int] = {}
    df["cluster_id"] = [
        sites.setdefault(pair, len(sites))
        for pair in zip(df["x"].round(8), df["ycoord"].round(8))
    ]
    return df


def fit_spatial(df: pd.DataFrame) -> pd.Series:
    return SpatialTOST(
        y="diff",
        cluster="cluster_id",
        x="x",
        ycoord="ycoord",
        config=SpatialConfig(
            point_estimator="equal_weighted",
            verbose_diagnostics=False,
            nu_grid=(0.5,),
        ),
    ).fit(df, ALPHA, MARGINS).iloc[0]


def fit_spatiotemporal(df: pd.DataFrame) -> pd.Series:
    return SpatioTemporalTOST(
        y="diff",
        cluster="cluster_id",
        time="time",
        x="x",
        ycoord="ycoord",
        config=SpatioTemporalConfig(
            point_estimator="equal_weighted", verbose_diagnostics=False
        ),
    ).fit(df, ALPHA, MARGINS).iloc[0]


def run_study(name, build, fit, replicates):
    """Estimate coverage over ``replicates`` datasets whose true mean is TRUE_MU."""
    hits, widths = [], []
    method = ""
    started = time.perf_counter()
    for seed in range(replicates):
        with contextlib.redirect_stdout(io.StringIO()):
            row = fit(build(seed))
        method = method or str(row["method"])
        hits.append(bool(row["ci_low"] <= TRUE_MU <= row["ci_high"]))
        widths.append(float(row["ci_high"]) - float(row["ci_low"]))
        if (seed + 1) % 25 == 0:
            print(f"  {name}: {seed + 1}/{replicates}", file=sys.stderr)
    elapsed = time.perf_counter() - started

    coverage = float(np.mean(hits))
    return {
        "engine": name,
        "method": method,
        "replicates": replicates,
        "coverage": coverage,
        "mc_se": float(np.sqrt(NOMINAL * (1 - NOMINAL) / replicates)),
        "mean_width": float(np.mean(widths)),
        "seconds": elapsed,
        "seconds_per_fit": elapsed / replicates,
    }


def render(results: list[dict], replicates: int) -> str:
    lines = [
        "# Coverage benchmark: spatial and spatiotemporal engines",
        "",
        (f"Generated {date.today().strftime('%e %B %Y').strip()} by "
        "`scripts/coverage_benchmark.py`."),
        "",
        "Regenerate with:",
        "",
        "    pixi run -e test python scripts/coverage_benchmark.py",
        "",
        "## Result",
        "",
        (f"Nominal coverage is {NOMINAL:.3f}, the $(1 - 2\\alpha)$ level the TOST "
        f"decision rule uses at $\\alpha = {ALPHA}$. The true mean paired difference "
        f"is {TRUE_MU} by construction."),
        "",
        "| engine | replicates | coverage | Monte Carlo SE | mean CI width | s/fit |",
        "|--------|-----------:|---------:|---------------:|--------------:|------:|",
    ]
    for r in results:
        lines.append(
            f"| {r['engine']} | {r['replicates']} | {r['coverage']:.3f} | "
            f"{r['mc_se']:.3f} | {r['mean_width']:.3f} | {r['seconds_per_fit']:.2f} |"
        )
    lines += ["", "Estimator path exercised, taken from the `method` column:", ""]
    for r in results:
        lines.append(f"- {r['engine']}: {r['method']}")

    lines += [
        "",
        "## Design",
        "",
        f"- Replicates per engine: {replicates}, seeds `0 .. {replicates - 1}`.",
        (f"- Spatial: {SPATIAL_DESIGN['n_clusters']} clusters of "
        f"{SPATIAL_DESIGN['points_per_cluster']} points "
        f"(n = {SPATIAL_DESIGN['n_clusters'] * SPATIAL_DESIGN['points_per_cluster']}), "
        f"length scale {SPATIAL_DESIGN['length_scale']}, field SD 0.8, nugget SD 0.2."),
        (f"- Spatiotemporal: {SPATIOTEMPORAL_DESIGN['n_space']} sites x "
        f"{SPATIOTEMPORAL_DESIGN['n_time']} times "
        f"(n = {SPATIOTEMPORAL_DESIGN['n_space'] * SPATIOTEMPORAL_DESIGN['n_time']}), "
        f"length scale {SPATIOTEMPORAL_DESIGN['length_scale']}, AR(1) "
        f"rho = {SPATIOTEMPORAL_DESIGN['rho']}, spatial SD 0.8, observation SD 0.2."),
        "- Both engines run in `equal_weighted` mode, so the estimand is the mean",
        "  paired difference rather than a GLS-reweighted mean.",
        "- The spatial generator uses a squared-exponential covariance while the engines",
        "  fit a Matern model, so coverage is measured under a misspecified covariance",
        "  function rather than a matched one.",
        "- The spatial run fixes Matern smoothness at `nu_grid=(0.5,)` rather than",
        "  profiling the default `(0.5, 1.5, 2.5)`. Profiling over the grid roughly",
        "  triples the per-fit cost without changing the estimand, so the figure below",
        "  validates the exponential-smoothness configuration rather than the full",
        "  default search.",
        "",
        "## Environment",
        "",
        f"- pyTOST {version('pyTOST')}",
        (f"- Python {platform.python_version()} on {platform.system()} "
        f"{platform.machine()}"),
        f"- NumPy {np.__version__}, pandas {pd.__version__}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replicates",
        type=int,
        default=REPLICATES,
        help=f"replicates per engine (default {REPLICATES})",
    )
    parser.add_argument(
        "--no-write", action="store_true", help="print results without writing the artifact"
    )
    args = parser.parse_args()

    results = [
        run_study("spatial", spatial_replicate, fit_spatial, args.replicates),
        run_study(
            "spatiotemporal",
            spatiotemporal_replicate,
            fit_spatiotemporal,
            args.replicates,
        ),
    ]

    for r in results:
        print(
            f"{r['engine']:16s} coverage={r['coverage']:.3f} "
            f"(nominal {NOMINAL:.3f}, MC SE {r['mc_se']:.3f}) "
            f"{r['seconds']:.0f}s total"
        )

    if not args.no_write:
        OUTPUT.write_text(render(results, args.replicates))
        print(f"\nwrote {OUTPUT}")


if __name__ == "__main__":
    main()

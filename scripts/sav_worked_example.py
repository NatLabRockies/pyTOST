"""End-to-end SAV equivalence analysis on synthetic data.

This reproduces every number in ``docs/sav_worked_example.md``. The scenario is a
solar access value (SAV) method comparison: a fast proxy model (arm B) is checked for
practical interchangeability with an established ray-tracing reference (arm A) across
building sites observed over several months.

Run with::

    pixi run -e test python scripts/sav_worked_example.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

from pyTOST import WorkflowOptions, plot_ci, run_tost
from pyTOST.data_gen.synthetic_tost_data import generate_spatiotemporal

SEED = 424242
N_SITES = 30
N_MONTHS = 6
MARGIN = 1.0  # percentage points of SAV considered practically negligible
ALPHA = 0.05
FIGURE = Path(__file__).resolve().parents[1] / "docs" / "sav_worked_example.png"


def build_sav_panel() -> pd.DataFrame:
    """Synthesize a paired SAV panel with spatial and temporal dependence."""
    long_df, _meta = generate_spatiotemporal(
        n_space=N_SITES,
        n_time=N_MONTHS,
        length_scale=1.0,
        rho=0.7,
        spatial_sd=1.5,
        obs_sd=0.4,
        delta=0.8,  # true mean SAV difference, in percentage points
        seed=SEED,
    )

    panel = (
        long_df.pivot_table(index=["sample_id", "x", "y_sp", "t"], columns="arm", values="y")
        .reset_index()
        .rename(columns={"y_sp": "ycoord", "t": "month"})
    )
    panel.columns.name = None
    panel["sav_diff"] = panel["B"] - panel["A"]

    # Sites are grouped into neighborhoods; SAV is correlated within a neighborhood.
    x_mid, y_mid = panel["x"].median(), panel["ycoord"].median()
    panel["neighborhood"] = np.where(panel["x"] <= x_mid, "west", "east") + np.where(
        panel["ycoord"] <= y_mid, "_south", "_north"
    )

    return panel[["sav_diff", "neighborhood", "x", "ycoord", "month"]]


def main() -> None:
    panel = build_sav_panel()

    print("Panel shape:", panel.shape)
    print("Neighborhoods:", sorted(panel["neighborhood"].unique()))
    print("Months:", [int(m) for m in sorted(panel["month"].unique())])
    print(f"Naive mean SAV difference: {panel['sav_diff'].mean():+.3f} pp\n")

    naive = run_tost(
        panel,
        y="sav_diff",
        margins=[MARGIN],
        alpha=ALPHA,
        engine="iid",
        options=WorkflowOptions(do_sensitivity=False, bootstrap_B=0),
    )
    print("--- Naive IID analysis ---")
    print(naive.summary())

    result = run_tost(
        panel,
        y="sav_diff",
        margins=[MARGIN],
        alpha=ALPHA,
        engine="spatiotemporal",
        cluster="neighborhood",
        time="month",
        x="x",
        ycoord="ycoord",
        options=WorkflowOptions(
            do_sensitivity=True, bootstrap_B=200, robust_location_B=200, seed=SEED
        ),
    )
    print("\n--- Dependence-aware analysis ---")
    print(result.summary())

    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig, _ax = plot_ci(
        result, margin=MARGIN, title=f"SAV method comparison (Δ = {MARGIN:g} pp)"
    )
    fig.savefig(FIGURE, dpi=200, bbox_inches="tight")
    print(f"\nwrote {FIGURE}")


if __name__ == "__main__":
    main()

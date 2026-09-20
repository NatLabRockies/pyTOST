"""Executable demo for the synthetic stress-test dataset.

Loads ``synthetic_stress_test_dataset.csv`` and runs the dependence-aware
paired-equivalence engines from the companion pyTOST package on the paired
difference column, printing each two one-sided tests (TOST) confidence interval
and equivalence decision at margin Delta = 1.

Run:
    pip install pyTOST            # from the companion software archive
    python demo_analysis.py                 # fast engines (IID, cluster, temporal)
    python demo_analysis.py --with-spatial  # also fit the Matern spatial engine (slower)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pyTOST import run_tost, WorkflowOptions, SpatialConfig

DATA = Path(__file__).resolve().parent / "synthetic_stress_test_dataset.csv"
ALPHA = 0.05
MARGIN = 1.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--with-spatial", action="store_true",
                    help="also fit the Matern spatial engine (minutes, not seconds)")
    args = ap.parse_args()

    df = pd.read_csv(DATA)
    common = dict(y="diff", margins=[MARGIN], alpha=ALPHA,
                  options=WorkflowOptions(do_sensitivity=False, seed=42))

    rows = []

    def record(name, res):
        r = res["primary"].iloc[0]
        lo, hi = float(r["ci_low"]), float(r["ci_high"])
        rows.append((name, float(r["mu_hat"]), lo, hi, hi - lo,
                     (lo > -MARGIN) and (hi < MARGIN)))

    record("IID", run_tost(df, engine="iid", **common))
    record("Cluster (building-robust)",
           run_tost(df, engine="cluster", cluster="cluster_id", **common))
    record("Temporal (HAC)",
           run_tost(df, engine="temporal", time="time", **common))
    if args.with_spatial:
        record("Spatial (Matern)",
               run_tost(df, engine="spatial", cluster="cluster_id", x="x", ycoord="y",
                        spatial_config=SpatialConfig(nu_grid=(0.5, 1.5, 2.5),
                                                     per_cluster_nugget=True,
                                                     verbose_diagnostics=False),
                        **common))

    out = pd.DataFrame(rows, columns=["engine", "mu_hat", "ci_low", "ci_high",
                                      "width", "equivalent"])
    pd.set_option("display.width", 120)
    print(f"Synthetic stress-test dataset: N={len(df)}, "
          f"buildings={df['cluster_id'].nunique()}, "
          f"locations={df['loc_id'].nunique()}, times={df['time'].nunique()}")
    print(f"Equivalence margin Delta = {MARGIN}, alpha = {ALPHA}\n")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()

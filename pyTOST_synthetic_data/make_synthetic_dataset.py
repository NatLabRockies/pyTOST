"""Reproducibly generate the synthetic stress-test dataset for dependence-aware
paired-equivalence testing.

Panel dimensions: 9 buildings x 25 spatial locations x 98 time steps
(N = 2,450 paired observations).

The paired arms A and B and their spatial coordinates are drawn from a separable
AR(1) x Matern spatiotemporal process by ``pyTOST.data_gen.generate_spatiotemporal``
using the frozen parameters in ``oneshot_candidate.json`` and generation seed
15211, which reproduces the arms/coordinates to machine precision.

Grouping columns:
  - ``loc_id``     : stable spatial-location identifier (one per unique (x, y)),
                     25 levels, constant across the 98 time steps of a location.
  - ``cluster_id`` : building identifier, 9 levels, obtained by a deterministic
                     Sort-Tile-Recursive (STR) spatial partition of the 25
                     locations into 9 spatially compact buildings. It is constant
                     within a location (every time step of a location shares the
                     same building), so it is a genuine spatial grouping suitable
                     as the clustering unit for cluster-robust and spatial engines.

Writes (next to this script):
  - synthetic_stress_test_dataset.csv
  - synthetic_stress_test_meta.json
"""
from __future__ import annotations

import inspect
import json
import math
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from pyTOST.data_gen import synthetic_tost_data as gen_mod

OUT_DIR = Path(__file__).resolve().parent
ALPHA = 0.05
MARGIN = 1.0
COLUMNS = ["sample_id", "x", "y", "time", "A", "B", "diff", "cluster_id", "loc_id"]


def assign_buildings_str(loc_df: pd.DataFrame, n_buildings: int,
                         xcol: str = "x", ycol: str = "y") -> pd.Series:
    """Partition locations into ``n_buildings`` via Sort-Tile-Recursive tiling.

    Deterministic; every building is non-empty and spatially compact. Building
    ids are relabeled by ascending centroid (x, then y) for stable labeling.
    Returns a Series of building ids aligned to ``loc_df.index``.
    """
    m = len(loc_df)
    n_strips = int(round(math.sqrt(n_buildings)))
    order = loc_df.sort_values([xcol, ycol]).reset_index()  # 'index' = original row
    strips = np.array_split(np.arange(m), n_strips)         # contiguous by x
    per = [n_buildings // n_strips + (1 if s < n_buildings % n_strips else 0)
           for s in range(n_strips)]
    label = np.empty(m, dtype=int)
    b = 0
    for s, idx in enumerate(strips):
        sub = order.iloc[idx].sort_values([ycol, xcol])     # within strip by y
        for chunk in np.array_split(sub.index.values, per[s]):
            for pos in chunk:
                label[pos] = b
            b += 1
    order["building"] = label
    cent = order.groupby("building")[[xcol, ycol]].mean().sort_values([xcol, ycol])
    remap = {old: new for new, old in enumerate(cent.index)}
    order["building"] = order["building"].map(remap)
    return order.set_index("index")["building"].sort_index()


def build_dataset() -> tuple[pd.DataFrame, dict]:
    saved = json.loads((OUT_DIR / "oneshot_candidate.json").read_text())
    params = saved["params"]
    gen_seed = int(saved["seed"])
    n_buildings = int(params["n_clusters"])

    gen = gen_mod.generate_spatiotemporal
    allowed = set(inspect.signature(gen).parameters)
    call = {"seed": gen_seed}
    call.update({k: v for k, v in params.items() if k in allowed})
    df_long, _ = gen(**call)

    # Long A/B -> per-observation paired difference.
    wide = df_long.pivot(index="sample_id", columns="arm", values="y").reset_index()
    meta = (df_long[df_long["arm"] == "A"][["sample_id", "x", "y_sp", "t"]]
            .rename(columns={"y_sp": "y", "t": "time"}))
    df = meta.merge(wide, on="sample_id")
    df["diff"] = df["B"] - df["A"]

    # Stable spatial-location id (25 unique coordinate pairs).
    loc = (df[["x", "y"]].drop_duplicates().sort_values(["x", "y"])
           .reset_index(drop=True))
    loc["loc_id"] = np.arange(len(loc))
    # Building id via STR spatial partition (constant within a location).
    loc["cluster_id"] = assign_buildings_str(loc, n_buildings, "x", "y").values
    df = df.merge(loc, on=["x", "y"], how="left")
    df = df[COLUMNS].sort_values("sample_id").reset_index(drop=True)

    meta_out = {
        "generator_kwargs": call,
        "generation_seed": gen_seed,
        "n_clusters_buildings": n_buildings,
        "n_space_locations": int(params["n_space"]),
        "n_time_steps": int(params["n_time"]),
        "n_observations": int(len(df)),
        "alpha": ALPHA,
        "margin_delta": MARGIN,
        "building_assignment": (
            "Sort-Tile-Recursive spatial partition of the 25 locations into 9 "
            "buildings; constant within a location; relabeled by ascending centroid."
        ),
        "search_params": params,
        "python_version": platform.python_version(),
        "package_versions": {
            "numpy": np.__version__,
            "scipy": __import__("scipy").__version__,
            "pandas": pd.__version__,
        },
    }
    return df, meta_out


def main() -> None:
    df, meta = build_dataset()
    df.to_csv(OUT_DIR / "synthetic_stress_test_dataset.csv", index=False)
    (OUT_DIR / "synthetic_stress_test_meta.json").write_text(
        json.dumps(meta, indent=2, default=str))
    sizes = df.drop_duplicates("loc_id")["cluster_id"].value_counts().sort_index()
    print(f"Wrote dataset: N={len(df)}, buildings={df['cluster_id'].nunique()}, "
          f"locations={df['loc_id'].nunique()}, times={df['time'].nunique()}")
    print("locations per building:", sizes.to_dict())


if __name__ == "__main__":
    main()

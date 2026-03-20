
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyTOST.data_gen.synthetic_tost_data import generate_iid


def long_to_diff(df_long: pd.DataFrame, *, index_cols: list[str]) -> pd.DataFrame:
    """Convert long A/B output from data generators into paired differences."""
    keep = index_cols + ["arm", "y"]
    wide = (
        df_long[keep]
        .pivot_table(index=index_cols, columns="arm", values="y", aggfunc="first")
        .reset_index()
    )
    wide.columns.name = None
    wide["diff"] = wide["B"] - wide["A"]
    return wide


@pytest.fixture
def basic_diff_df() -> pd.DataFrame:
    rng = np.random.default_rng(123)
    return pd.DataFrame(
        {
            "diff": 0.10 + rng.normal(0.0, 0.01, size=16),
            "cluster_id": np.repeat(["A", "B", "C", "D"], 4),
            "time": np.arange(16),
        }
    )


@pytest.fixture
def spatial_df() -> pd.DataFrame:
    rng = np.random.default_rng(123)
    rows = []
    bases = {"A": (0.0, 0.0), "B": (5.0, 5.0), "C": (10.0, 0.0)}
    offsets = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
    for cluster_id, (bx, by) in bases.items():
        for xoff, yoff in offsets:
            rows.append({"cluster_id": cluster_id, "x": bx + xoff, "ycoord": by + yoff})
    df = pd.DataFrame(rows)
    df["diff"] = 0.10 + rng.normal(0.0, 0.01, size=len(df))
    return df


@pytest.fixture
def spatiotemporal_df() -> pd.DataFrame:
    rng = np.random.default_rng(123)
    rows = []
    bases = {"A": (0.0, 0.0), "B": (5.0, 5.0)}
    offsets = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
    for t in range(8):
        for cluster_id, (bx, by) in bases.items():
            for xoff, yoff in offsets:
                rows.append(
                    {
                        "cluster_id": cluster_id,
                        "time": t,
                        "x": bx + xoff,
                        "ycoord": by + yoff,
                    }
                )
    df = pd.DataFrame(rows)
    df["diff"] = 0.10 + rng.normal(0.0, 0.01, size=len(df))
    return df


@pytest.fixture
def iid_long_df() -> pd.DataFrame:
    df_long, _ = generate_iid(n=8, delta=0.2, sigma=0.05, seed=123)
    return df_long

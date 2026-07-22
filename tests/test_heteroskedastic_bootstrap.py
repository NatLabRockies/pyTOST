"""Regression tests for the heteroskedastic wild-cluster bootstrap.

P0A-S02: verifies that the bootstrap interval has strictly positive width
(nonzero spread) on clustered data with clearly nonzero per-cluster residual
sums.  The degenerate within-cluster centering bug caused the bootstrap mean
to always equal the original mean, collapsing the interval to a point.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyTOST.engines.heteroskedastic_tost import HeteroskedasticTOST


def _clustered_df(seed: int = 3, G: int = 8, per: int = 25, shift_sd: float = 1.5,
                  noise_sd: float = 0.3) -> pd.DataFrame:
    """Clustered data with nonzero per-cluster residual sums (cluster-level shifts)."""
    rng = np.random.default_rng(seed)
    rows = []
    for gi in range(G):
        shift = rng.normal(0, shift_sd)
        for _ in range(per):
            rows.append((gi, shift + rng.normal(0, noise_sd)))
    return pd.DataFrame(rows, columns=["cluster_id", "y"])


def test_bootstrap_interval_nonzero_spread():
    """Bootstrap CI width must be strictly positive when cluster residual sums are nonzero."""
    df = _clustered_df()
    h = HeteroskedasticTOST(y="y", cluster="cluster_id", wild_B=299, seed=7)
    lo, hi = h._wild_cluster_bootstrap_ci(df, alpha=0.05)
    width = hi - lo
    assert width > 1e-6, (
        f"Bootstrap interval collapsed to near-zero width={width:.2e}; "
        "within-cluster centering may have been reintroduced."
    )


def test_bootstrap_spread_consistent_across_seeds():
    """Bootstrap width should be positive for multiple seeds (not a fluke)."""
    df = _clustered_df()
    for seed in (7, 42, 99):
        h = HeteroskedasticTOST(y="y", cluster="cluster_id", wild_B=199, seed=seed)
        lo, hi = h._wild_cluster_bootstrap_ci(df, alpha=0.05)
        assert hi - lo > 1e-6, f"Collapsed bootstrap at seed={seed}"


def test_fit_returns_ci_with_nonzero_width_when_clustered():
    """fit() with cluster col returns CI with nonzero width."""
    df = _clustered_df()
    h = HeteroskedasticTOST(y="y", cluster="cluster_id", wild_B=199, seed=7)
    result = h.fit(df, alpha=0.05, margins=[2.0])
    row = result.iloc[0]
    assert row["ci_high"] - row["ci_low"] > 1e-6, "CI collapsed in fit()"
    assert "Wild Cluster Bootstrap" in row["method"]

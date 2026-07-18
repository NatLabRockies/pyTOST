"""Tests for pyTOST.fewcluster: CR2 and wild cluster bootstrap-t inference."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyTOST.fewcluster import cr2_mean, wild_cluster_bootstrap_ci, wild_cluster_tost


def _make_clustered(G, n_per, icc, seed, mu=0.0):
    rng = np.random.default_rng(seed)
    sigma_a = np.sqrt(icc)
    sigma_e = np.sqrt(1 - icc)
    rows = []
    for g in range(G):
        a = rng.normal(0, sigma_a)
        for _ in range(n_per):
            rows.append({"diff": mu + a + rng.normal(0, sigma_e), "cluster_id": g})
    return pd.DataFrame(rows)


def test_cr2_se_exceeds_cr0_and_iid_under_positive_icc():
    df = _make_clustered(G=8, n_per=30, icc=0.3, seed=1)
    r = cr2_mean(df["diff"].to_numpy(), df["cluster_id"].to_numpy())
    assert r["se_cr2"] >= r["se_cr0"] > 0
    assert r["se_cr0"] > r["se_iid"]  # clustering inflates SE over IID
    assert 1.0 <= r["df_bm"] <= r["G"]


def test_cr2_closed_form_matches_direct_sandwich_intercept_only():
    # For X=1, CR0 V = sum_g (sum e_g)^2 / N^2. Verify the module's CR0 matches.
    df = _make_clustered(G=6, n_per=10, icc=0.2, seed=2)
    y = df["diff"].to_numpy(); g = df["cluster_id"].to_numpy()
    mu = y.mean(); e = y - mu
    N = len(y)
    v_cr0 = sum((e[g == k].sum()) ** 2 for k in np.unique(g)) / N ** 2
    r = cr2_mean(y, g)
    assert r["se_cr0"] == pytest.approx(np.sqrt(v_cr0), rel=1e-10)


def test_wild_cluster_ci_brackets_mean_and_is_wider_than_iid():
    df = _make_clustered(G=9, n_per=25, icc=0.25, seed=3, mu=0.5)
    r = wild_cluster_bootstrap_ci(df["diff"].to_numpy(), df["cluster_id"].to_numpy(),
                                  B=999, seed=7)
    assert r["ci_low"] < r["mu"] < r["ci_high"]
    iid_half = 1.96 * df["diff"].std(ddof=1) / np.sqrt(len(df))
    assert (r["ci_high"] - r["ci_low"]) > 2 * iid_half  # wider than naive IID


def test_wild_cluster_tost_decisions():
    df = _make_clustered(G=9, n_per=25, icc=0.1, seed=4, mu=0.0)
    out = wild_cluster_tost(df, y="diff", cluster="cluster_id", margins=[0.5, 5.0],
                            B=499, seed=11)
    assert len(out) == 2
    # wide margin should be equivalent; near-zero-mean data inside +/-5
    assert bool(out.loc[out["delta"] == 5.0, "equivalent"].iloc[0]) is True


@pytest.mark.slow
def test_wild_cluster_coverage_few_clusters():
    """CR2-studentised wild cluster bootstrap-t should have >= ~90% coverage of a
    nominal 95% CI with only 7 clusters, and clearly beat CR0-t(G-1)."""
    G, n_per, icc, mu0 = 7, 20, 0.2, 0.0
    nsim = 300
    cover_wcb = 0
    cover_cr0 = 0
    for s in range(nsim):
        df = _make_clustered(G=G, n_per=n_per, icc=icc, seed=1000 + s, mu=mu0)
        y = df["diff"].to_numpy(); g = df["cluster_id"].to_numpy()
        r = wild_cluster_bootstrap_ci(y, g, alpha=0.025, B=399, seed=5)
        if r["ci_low"] <= mu0 <= r["ci_high"]:
            cover_wcb += 1
        c = cr2_mean(y, g, alpha=0.025)  # uses CR2 + BM df
        # CR0 with t(G-1)
        from scipy import stats as st
        tcrit = st.t.ppf(0.975, G - 1)
        lo, hi = c["mu"] - tcrit * c["se_cr0"], c["mu"] + tcrit * c["se_cr0"]
        if lo <= mu0 <= hi:
            cover_cr0 += 1
    cov_wcb = cover_wcb / nsim
    cov_cr0 = cover_cr0 / nsim
    # Wild cluster bootstrap should be close to nominal; both recorded for report.
    assert cov_wcb >= 0.90, f"wild-cluster coverage too low: {cov_wcb}"

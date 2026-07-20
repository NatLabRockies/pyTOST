"""Numerical cross-checks of pyTOST engines against reference implementations.

These guard the core interval/decision computations against independent
references: statsmodels' two-one-sided-tests for the paired mean, an analytic
Student-t interval, a from-scratch cluster sandwich, and statsmodels' HAC
covariance. They document that pyTOST reproduces established results on the
i.i.d., clustered, and temporal engines.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.weightstats import ttost_paired

from pyTOST import IIDTOST, ClusterTOST, TemporalTOST


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_iid_tost_decision_matches_statsmodels_ttost(seed):
    rng = np.random.default_rng(seed)
    diff = rng.normal(0.3, 1.0, size=60)
    df = pd.DataFrame({"diff": diff})
    alpha, delta = 0.05, 1.0
    res = IIDTOST("diff").fit(df, alpha=alpha, margins=[delta])
    pyt_eq = bool(res["equivalent"].iloc[0])
    # statsmodels reference: equivalence iff combined TOST p-value < alpha
    pval, _, _ = ttost_paired(diff, np.zeros_like(diff), -delta, delta)
    ref_eq = pval < alpha
    assert pyt_eq == ref_eq


@pytest.mark.parametrize("seed", [10, 11, 12])
def test_iid_ci_matches_analytic_t_interval(seed):
    rng = np.random.default_rng(seed)
    diff = rng.normal(0.5, 0.8, size=40)
    df = pd.DataFrame({"diff": diff})
    alpha = 0.05
    res = IIDTOST("diff").fit(df, alpha=alpha, margins=[1.0])
    mu = diff.mean(); se = diff.std(ddof=1) / np.sqrt(len(diff))
    t = stats.t.ppf(1 - alpha, len(diff) - 1)
    assert res["ci_low"].iloc[0] == pytest.approx(mu - t * se, rel=1e-9)
    assert res["ci_high"].iloc[0] == pytest.approx(mu + t * se, rel=1e-9)


def test_cluster_se_matches_manual_sandwich():
    rng = np.random.default_rng(7)
    rows = []
    for g in range(10):
        a = rng.normal(0, 0.5)
        for _ in range(rng.integers(8, 20)):
            rows.append({"diff": 0.4 + a + rng.normal(0, 0.7), "cluster_id": g})
    df = pd.DataFrame(rows)
    y = df["diff"].to_numpy(); grp = df["cluster_id"].to_numpy()
    res = ClusterTOST("diff", "cluster_id").fit(df, alpha=0.05, margins=[1.0])
    # Reference: independent statsmodels cluster-robust OLS + t(G-1) interval.
    fit = sm.OLS(y, np.ones((len(y), 1))).fit(cov_type="cluster", cov_kwds={"groups": grp})
    G = df["cluster_id"].nunique()
    mu = float(fit.params[0]); se = float(fit.bse[0]); t = stats.t.ppf(0.95, G - 1)
    assert res["ci_low"].iloc[0] == pytest.approx(mu - t * se, rel=1e-6)
    assert res["ci_high"].iloc[0] == pytest.approx(mu + t * se, rel=1e-6)


def test_temporal_hac_ci_matches_statsmodels_hac():
    rng = np.random.default_rng(3)
    T = 200
    d = rng.normal(0.2, 1.0, size=T)
    for t in range(1, T):
        d[t] += 0.3 * d[t - 1]  # serial correlation
    df = pd.DataFrame({"diff": d, "time": np.arange(T)})
    res = TemporalTOST("diff", "time", hac_lags=4).fit(df, alpha=0.05, margins=[1.0])
    fit = sm.OLS(d, np.ones((T, 1))).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
    mu = float(fit.params[0]); se = float(fit.bse[0]); z = stats.norm.ppf(0.95)
    assert res["ci_low"].iloc[0] == pytest.approx(mu - z * se, rel=1e-9)
    assert res["ci_high"].iloc[0] == pytest.approx(mu + z * se, rel=1e-9)

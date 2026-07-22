"""Equal-weighted estimand mode: spatial and spatiotemporal engines report the
sample mean as the point estimate, with a dependence-aware Wald interval whose
variance is 1' Sigma_hat 1 / N^2 under the fitted covariance."""
from __future__ import annotations

import numpy as np
import pytest

from pyTOST import SpatialTOST, SpatialConfig, SpatioTemporalTOST, SpatioTemporalConfig
from pyTOST.data_gen.synthetic_tost_data import generate_spatiotemporal
from pyTOST.data_gen.optimize_spatiotemporal_params import _make_diff_df


def _panel(n_space=16, n_time=8, seed=1):
    d = _make_diff_df(generate_spatiotemporal(n_space=n_space, n_time=n_time, delta=0.5, seed=seed)[0],
                      n_clusters=1)
    d["cluster_id"] = 0
    return d


def test_spatial_equal_weighted_point_is_sample_mean():
    d = _panel()
    d1 = d.groupby(["x", "y"], as_index=False)["diff"].mean()
    d1["cluster_id"] = 0
    res = SpatialTOST(y="diff", cluster="cluster_id", x="x", ycoord="y",
                      config=SpatialConfig(nu_grid=(1.5,), point_estimator="equal_weighted")
                      ).fit(d1, 0.05, [1.0])
    assert float(res["mu_hat"].iloc[0]) == pytest.approx(float(d1["diff"].mean()), rel=1e-9)
    assert res["ci_low"].iloc[0] < res["mu_hat"].iloc[0] < res["ci_high"].iloc[0]
    assert "equal-weighted" in res["method"].iloc[0]


def test_spatiotemporal_equal_weighted_point_is_sample_mean():
    d = _panel()
    res = SpatioTemporalTOST(y="diff", cluster="cluster_id", time="time", x="x", ycoord="y",
                             config=SpatioTemporalConfig(nu_grid=(1.5,), point_estimator="equal_weighted")
                             ).fit(d, 0.05, [1.0])
    assert float(res["mu_hat"].iloc[0]) == pytest.approx(float(d["diff"].mean()), rel=1e-9)
    assert res["ci_low"].iloc[0] < res["mu_hat"].iloc[0] < res["ci_high"].iloc[0]
    assert "equal-weighted" in res["method"].iloc[0]


def test_equal_weighted_variance_matches_quadratic_form():
    """Var(xbar) equals 1' Sigma 1 / N^2; check the reported SE against a direct
    dense computation on a small single-cluster spatial fixture."""
    d = _panel(n_space=9, n_time=2, seed=3)
    d1 = d.groupby(["x", "y"], as_index=False)["diff"].mean()
    d1["cluster_id"] = 0
    from pyTOST.engines.spatial_tost import fit_matern_profile_ml, _build_sigma_and_stacks
    dfp = d1.rename(columns={"cluster_id": "cluster_id", "diff": "diff"})
    theta = fit_matern_profile_ml(df=dfp, cluster_col="cluster_id", x_col="x", y_col="y",
                            diff_col="diff", nu_grid=(1.5,), per_cluster_nugget=True)
    Sigma, yv, ones = _build_sigma_and_stacks(dfp, "cluster_id", "x", "y", "diff",
                                              sigma2=theta["sigma2"], rho=theta["rho"],
                                              tau2=theta["tau2"], nu=theta["nu"], per_cluster_nugget=True)
    N = len(yv)
    var_direct = float((ones.T @ Sigma @ ones).item()) / N ** 2
    res = SpatialTOST(y="diff", cluster="cluster_id", x="x", ycoord="y",
                      config=SpatialConfig(nu_grid=(1.5,), point_estimator="equal_weighted")
                      ).fit(d1, 0.05, [1.0])
    from scipy import stats
    se_reported = (res["ci_high"].iloc[0] - res["mu_hat"].iloc[0]) / stats.norm.ppf(0.95)
    assert se_reported ** 2 == pytest.approx(var_direct, rel=1e-6)

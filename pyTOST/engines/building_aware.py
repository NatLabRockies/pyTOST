"""Building-aware spatiotemporal engine.

Represents all design levels of a repeated rooftop panel simultaneously: a
building-level random intercept, spatial Matern correlation among locations
within a building, temporal AR(1) correlation across the repeated slices, and an
observation nugget. The marginal covariance is block diagonal across buildings,

    Sigma_g(theta) = sigma_b^2 J_g + sigma_st^2 (R_time(phi) kron R_space(rho, nu))
                     + tau^2 I,

with J_g the all-ones block that induces equal within-building correlation from
the random intercept. Covariance parameters are fit by maximum likelihood with
the mean fixed at the equal-weighted sample mean, and the reported interval is a
Wald interval on that mean with variance 1' Sigma_hat 1 / N^2. This keeps the
estimand identical to the other engines (the equal-weighted population mean
paired difference) while using the full hierarchical uncertainty model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List
import numpy as np
import pandas as pd
from scipy import linalg, optimize, stats

from .spatial_tost import _matern_cov, _pairwise_dists
from .spatiotemporal_tost import _ar1_corr


@dataclass
class BuildingAwareConfig:
    nu: float = 1.5
    maxiter: int = 300
    verbose_diagnostics: bool = False


class BuildingAwareSpatioTemporalTOST:
    def __init__(self, y: str, cluster: str, time: str, x: str, ycoord: str,
                 config: BuildingAwareConfig | None = None):
        self.y = y
        self.cluster = cluster
        self.time = time
        self.x = x
        self.ycoord = ycoord
        self.config = config or BuildingAwareConfig()

    def _blocks(self, df: pd.DataFrame):
        """Per-building (residual-ordering, R_space, R_time, n) with (time, location) order."""
        blocks = []
        nu = self.config.nu
        for b, sub in df.groupby(self.cluster, sort=False):
            locs = sub[[self.x, self.ycoord]].drop_duplicates().reset_index(drop=True)
            times = np.sort(sub[self.time].unique())
            M, T = len(locs), len(times)
            # order rows as t-major, location-minor
            loc_key = {(round(float(r[self.x]), 9), round(float(r[self.ycoord]), 9)): i
                       for i, r in locs.iterrows()}
            sub = sub.copy()
            sub["_lk"] = [loc_key[(round(float(a), 9), round(float(c), 9))]
                          for a, c in zip(sub[self.x], sub[self.ycoord])]
            sub["_tk"] = sub[self.time].map({t: i for i, t in enumerate(times)})
            sub = sub.sort_values(["_tk", "_lk"])
            if len(sub) != M * T:
                # unbalanced building block: skip building-aware structure, treat as iid block
                yv = sub[self.y].to_numpy(float)
                blocks.append({"y": yv, "M": len(sub), "T": 1, "Rs": np.eye(len(sub)),
                               "Rt": np.array([[1.0]]), "n": len(sub), "balanced": False})
                continue
            D = _pairwise_dists(locs[[self.x, self.ycoord]].to_numpy(float))
            blocks.append({"y": sub[self.y].to_numpy(float), "M": M, "T": T,
                           "Ds": D, "n": M * T, "balanced": True})
        return blocks

    def _block_cov(self, blk, sigma_b2, sigma_st2, rho, phi, tau2):
        n = blk["n"]
        if blk["balanced"]:
            Rs = _matern_cov(blk["Ds"], sigma2=1.0, rho=rho, nu=self.config.nu)
            Rt = _ar1_corr(T=blk["T"], phi=phi)
            K = sigma_st2 * np.kron(Rt, Rs)
        else:
            K = np.zeros((n, n))
        K += sigma_b2  # random intercept: all-ones block
        K.flat[:: n + 1] += tau2
        return K

    def fit(self, df: pd.DataFrame, alpha: float, margins: List[float]) -> pd.DataFrame:
        for col in (self.y, self.cluster, self.time, self.x, self.ycoord):
            if col not in df.columns:
                raise ValueError(f"BuildingAwareSpatioTemporalTOST requires column {col!r}.")
        blocks = self._blocks(df)
        yall = df[self.y].to_numpy(float)
        N = len(yall)
        mu_hat = float(yall.mean())
        sd = float(np.std(yall) + 1e-6)

        def unpack(z):
            return (np.exp(z[0]), np.exp(z[1]), np.exp(z[2]),
                    1.0 / (1.0 + np.exp(-z[3])), np.exp(z[4]))  # sb2, sst2, rho, phi, tau2

        def nll(z):
            sb2, sst2, rho, phi, tau2 = unpack(z)
            total = 0.0
            for blk in blocks:
                r = blk["y"] - mu_hat
                K = self._block_cov(blk, sb2, sst2, rho, phi, tau2)
                n = K.shape[0]
                jitter = 1e-8 * (np.trace(K) / max(n, 1))
                try:
                    L = linalg.cholesky(K + jitter * np.eye(n), lower=True, check_finite=False)
                except linalg.LinAlgError:
                    return 1e12
                v = linalg.solve_triangular(L, r, lower=True, check_finite=False)
                logdet = 2.0 * np.sum(np.log(np.diag(L)))
                total += 0.5 * (float(v @ v) + logdet + n * np.log(2 * np.pi))
            return total

        z0 = np.array([np.log(0.5 * sd ** 2), np.log(0.5 * sd ** 2), np.log(1.0),
                       0.0, np.log(0.1 * sd ** 2)])
        bounds = [(-20, 20), (-20, 20), (-20, 20), (-12, 12), (-30, 20)]
        opt = optimize.minimize(nll, z0, method="L-BFGS-B", bounds=bounds,
                                options={"maxiter": self.config.maxiter})
        sb2, sst2, rho, phi, tau2 = unpack(opt.x)

        # Var(xbar) = sum_g 1' Sigma_g 1 / N^2
        quad = 0.0
        for blk in blocks:
            K = self._block_cov(blk, sb2, sst2, rho, phi, tau2)
            one = np.ones(K.shape[0])
            quad += float(one @ (K @ one))
        var_xbar = quad / (N * N)
        se = float(np.sqrt(max(var_xbar, 0.0)))
        z = float(stats.norm.ppf(1 - alpha))
        ci_low, ci_high = mu_hat - z * se, mu_hat + z * se

        if self.config.verbose_diagnostics:
            print(f"[BuildingAware] sb2={sb2:.4g} sst2={sst2:.4g} rho={rho:.4g} "
                  f"phi={phi:.4g} tau2={tau2:.4g} conv={opt.success} se={se:.4g}")

        rows = []
        for d in margins:
            d = float(d)
            rows.append(dict(delta=d, mu_hat=mu_hat, ci_low=float(ci_low), ci_high=float(ci_high),
                             equivalent=(ci_low > -d and ci_high < d),
                             method="Building-aware ML (random intercept + AR1 x Matérn) + equal-weighted Wald",
                             sigma_b2=sb2, sigma_st2=sst2, rho=rho, phi=phi, tau2=tau2,
                             converged=bool(opt.success)))
        return pd.DataFrame(rows)

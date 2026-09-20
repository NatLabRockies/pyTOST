"""Building-aware spatiotemporal engine.

Represents all design levels of a repeated rooftop panel simultaneously: a
building-level random intercept, spatial Matern correlation among locations
within a building, temporal AR(1) correlation across the repeated slices, and an
observation nugget. The marginal covariance is block diagonal across buildings,

    Sigma_g(theta) = sigma_b^2 J_g + sigma_st^2 (R_time(phi) kron R_space(rho, nu))
                     + tau^2 I,

with J_g the all-ones block that induces equal within-building correlation from
the random intercept. Covariance parameters are fit by maximum likelihood with
the mean fixed at the equal-weighted sample mean. The reported interval is a
small-sample interval on that mean with variance 1' Sigma_hat 1 / N^2 and a
Student-t reference with G-1 degrees of freedom, where G is the number of
buildings. The Student-t reference accounts for the small number of independent
buildings; simulation (see the operating-characteristics evaluation) shows it
brings boundary false-equivalence into the 0.04 to 0.06 range, whereas a normal
reference leaves it near 0.06 to 0.09. This keeps the estimand identical to the
other engines (the equal-weighted population mean paired difference) while using
the full hierarchical uncertainty model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List
import numpy as np
import pandas as pd
from scipy import linalg, optimize, stats

from .spatial_tost import _matern_cov, _pairwise_dists
from .spatiotemporal_tost import _ar1_corr
from ..validation import require_finite_columns


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
        """Per-building observation-level covariance ingredients.

        Every building is represented by its distinct locations (with the spatial
        distance matrix), its distinct time slices, and, for each observed row, the
        index of its location and time. The block covariance is built directly over
        the observed cells, so an incomplete building (fewer than M*T observed
        cells) keeps the full random-intercept plus AR(1) x Matern structure over
        the cells it does contain rather than being replaced by an independent
        block. The number of observed cells is retained for every building.
        """
        blocks = []
        for b, sub in df.groupby(self.cluster, sort=False):
            locs = sub[[self.x, self.ycoord]].drop_duplicates().reset_index(drop=True)
            times = np.sort(sub[self.time].unique())
            M, T = len(locs), len(times)
            loc_key = {(round(float(r[self.x]), 9), round(float(r[self.ycoord]), 9)): i
                       for i, r in locs.iterrows()}
            sub = sub.copy()
            sub["_lk"] = [loc_key[(round(float(a), 9), round(float(c), 9))]
                          for a, c in zip(sub[self.x], sub[self.ycoord])]
            sub["_tk"] = sub[self.time].map({t: i for i, t in enumerate(times)})
            sub = sub.sort_values(["_tk", "_lk"])
            Ds = _pairwise_dists(locs[[self.x, self.ycoord]].to_numpy(float))
            blocks.append({"y": sub[self.y].to_numpy(float),
                           "loc_idx": sub["_lk"].to_numpy(int),
                           "t_idx": sub["_tk"].to_numpy(int),
                           "Ds": Ds, "M": M, "T": T, "n": len(sub),
                           "balanced": len(sub) == M * T})
        return blocks

    def _block_cov(self, blk, sigma_b2, sigma_st2, rho, phi, tau2):
        n = blk["n"]
        Rs = _matern_cov(blk["Ds"], sigma2=1.0, rho=rho, nu=self.config.nu)
        Rt = _ar1_corr(T=blk["T"], phi=phi)
        li, ti = blk["loc_idx"], blk["t_idx"]
        # marginal covariance over the observed (time, location) cells
        K = sigma_st2 * Rt[np.ix_(ti, ti)] * Rs[np.ix_(li, li)]
        K += sigma_b2  # random intercept: equal within-building correlation
        K.flat[:: n + 1] += tau2
        return K

    def fit(self, df: pd.DataFrame, alpha: float, margins: List[float]) -> pd.DataFrame:
        for col in (self.y, self.cluster, self.time, self.x, self.ycoord):
            if col not in df.columns:
                raise ValueError(f"BuildingAwareSpatioTemporalTOST requires column {col!r}.")

        require_finite_columns(
            df, (self.x, self.ycoord), engine="BuildingAwareSpatioTemporalTOST"
        )
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
        n_buildings = len(blocks)
        df_ref = max(n_buildings - 1, 1)
        crit = float(stats.t.ppf(1 - alpha, df_ref))
        ci_low, ci_high = mu_hat - crit * se, mu_hat + crit * se

        if self.config.verbose_diagnostics:
            print(f"[BuildingAware] sb2={sb2:.4g} sst2={sst2:.4g} rho={rho:.4g} "
                  f"phi={phi:.4g} tau2={tau2:.4g} conv={opt.success} se={se:.4g} "
                  f"df={df_ref} crit={crit:.4g}")

        rows = []
        for d in margins:
            d = float(d)
            rows.append(dict(delta=d, mu_hat=mu_hat, ci_low=float(ci_low), ci_high=float(ci_high),
                             equivalent=(ci_low > -d and ci_high < d),
                             method="Hierarchical spatiotemporal ML (building random intercept "
                                    "+ AR1 x Matérn + nugget) + equal-weighted mean, "
                                    "Student-t(G-1) interval",
                             se=se, df=df_ref, n_buildings=n_buildings,
                             sigma_b2=sb2, sigma_st2=sst2, rho=rho, phi=phi, tau2=tau2,
                             converged=bool(opt.success)))
        return pd.DataFrame(rows)

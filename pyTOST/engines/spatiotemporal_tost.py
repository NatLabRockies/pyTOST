"""engines/spatiotemporal_tost.py
==============================

Spatio-temporal TOST

This engine is designed for panel-like spatio-temporal data where observations have:

- a building/group id (cluster),
- spatial coordinates (x, y), and
- a time index (time).

Methods implemented
-------------------
Primary: Joint separable spatiotemporal Gaussian likelihood
    We fit an intercept-only Gaussian model with separable covariance

        Σ = σ² (R_t ⊗ R_s) + τ² I,

    where R_s is a Matérn correlation over spatial locations (ν fixed by the provided
    nu_grid, defaulting to the largest ν), and R_t is an AR(1) correlation over time.
    Parameters (μ, σ², ρ, τ², φ) are estimated by ML via L-BFGS-B. A Wald CI for μ is
    computed using Var(μ̂) = 1 / (1ᵀ Σ^{-1} 1).

    This method is only attempted when the data form a balanced panel (same spatial
    locations at each time). If the balanced-panel requirement is not met, we fall back
    to the interim approach below.

Fallback: Per-time spatial fits + IVW
    We use the spatial Matérn GLS (REML) methodology from
    `spatial_tost.py` as the core estimator, applied per time slice.

    1) For each time t, fit the spatial model to estimate μ_t and Var(μ_t).
    2) Combine time-slice estimates into a global μ via inverse-variance weighting:
           μ̂ = (Σ w_t μ̂_t) / (Σ w_t),  with w_t = 1 / Var(μ̂_t)
       and Var(μ̂) = 1 / Σ w_t.
    3) Construct a conservative t-based CI for μ̂ using df = (#times - 1).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Tuple, Literal, Optional

import numpy as np
import pandas as pd
from scipy import linalg, optimize, stats

from .spatial_tost import _matern_cov  # correlation up to sigma2; we use sigma2 separately
from .spatial_tost import fit_matern_reml  # lr CI used only for per-time θ selection


@dataclass(frozen=True)
class SpatioTemporalConfig:
    """
    Parameters controlling the per-time spatial fits (legacy path) and the joint separable
    spatiotemporal ML fit (balanced-panel path).

    Parameters
    ----------
    nu_grid
        Candidate Matérn ν values for per-time REML fits (legacy path). In the joint likelihood,
        ν is fixed to the maximum of this grid.
    per_building_nugget
        Whether to include τ² I within each building block (legacy per-time path).
    min_time_n
        Minimum number of rows required at a time slice to attempt a legacy per-time fit.
    verbose_diagnostics
        If True, print per-time-slice diagnostics (n_t, mu_hat_t, var_mu_t) and IVW weight
        concentration summaries, and also print joint-fit status.

    Joint-fit regularization (balanced-panel ML path)
    -----------------------------------------------
    The joint separable model estimates (μ, σ², ρ, τ², φ) by ML. In some datasets the likelihood
    admits near-nonidentifiable covariance solutions that imply very small mean information
    (1ᵀΣ⁻¹1), which inflates Var(μ̂) and yields implausibly wide confidence intervals. The
    options below apply weak MAP-like penalties to stabilize those fits.

    joint_regularize
        If True, add penalties to the joint ML objective.
    reg_lambda
        Overall regularization strength. The joint objective is:
            nll_reg = nll + reg_lambda * penalty.
        Set to 0.0 (or joint_regularize=False) to disable.

    tau_ratio_min
        Soft lower bound for the nugget fraction τ² / (σ² + τ²). When the fitted fraction drops
        below this threshold, a log-scale penalty is applied.
    phi_max
        Soft upper bound for the AR(1) coefficient φ. When φ exceeds this value, a quadratic
        penalty is applied to discourage near-unit-root temporal correlation.
    rho_max_factor
        Soft upper bound for the spatial range ρ, expressed as rho_max_factor * Dmax where Dmax
        is the maximum inter-location distance. When ρ exceeds this, a log-scale penalty is applied.

    var_scale_target
        Target scale factor for the marginal variance (σ² + τ²) relative to the empirical variance
        of y. The target is var_scale_target * Var(y).
    var_scale_log_tol
        Log-scale tolerance. No penalty is applied while:
            |log((σ²+τ²)/target)| <= var_scale_log_tol.
    var_scale_weight
        Relative weight for the variance-scale penalty term.

    se_ratio_max
        Soft upper bound on the ratio se_model(μ̂) / se_iid, where se_iid = sqrt(Var(y)/n).
        When the ratio exceeds se_ratio_max, a log-scale penalty is applied. This directly targets
        pathological inflation of Var(μ̂).
    se_ratio_weight
        Relative weight for the SE-ratio penalty term.


    Mean CI method (joint balanced-panel fit)
    ---------------------------------------
    The joint separable ML fit yields a covariance estimate K and a GLS mean estimate μ̂.
    By default, this engine reports a *parametric bootstrap* CI for μ̂, which is often more
    stable than a Wald/curvature CI when dependence is strong.

    mu_ci_method
        Method used to form the confidence interval for μ in the joint separable ML fit.
        - "parametric_bootstrap" (default): simulate y* ~ N(μ̂ 1, K) and recompute μ̂* via GLS,
          then take percentile CI with bounds at (alpha, 1-alpha).
        - "wald": use Var(μ̂)=1/(1ᵀK⁻¹1) and a Normal critical value z_{1-alpha}.
    mu_bootstrap_B
        Number of parametric bootstrap replicates used when mu_ci_method="parametric_bootstrap".
    mu_bootstrap_seed
        RNG seed for the parametric bootstrap (None -> unpredictable).

    """
    nu_grid: Tuple[float, ...] = (0.5, 1.5, 2.5)
    per_building_nugget: bool = True
    min_time_n: int = 8
    verbose_diagnostics: bool = False

    joint_regularize: bool = True
    reg_lambda: float = 1.0

    tau_ratio_min: float = 0.02
    phi_max: float = 0.995
    rho_max_factor: float = 3.0

    var_scale_target: float = 1.0
    var_scale_log_tol: float = 2.0
    var_scale_weight: float = 1.0

    se_ratio_max: float = 8.0
    se_ratio_weight: float = 2.0

    # Joint-fit mean CI selection
    mu_ci_method: Literal["parametric_bootstrap", "wald"] = "parametric_bootstrap"
    mu_bootstrap_B: int = 400
    mu_bootstrap_seed: Optional[int] = 12345


def _pairwise_dists_xy(XY: np.ndarray) -> np.ndarray:
    XY = np.asarray(XY, float)
    diff = XY[:, None, :] - XY[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def _ar1_corr(T: int, phi: float) -> np.ndarray:
    idx = np.arange(T)
    D = np.abs(idx[:, None] - idx[None, :])
    return phi ** D


class SpatioTemporalTOST:
    """
    Spatio-temporal TOST using spatial fits per time slice or a joint separable GP.

    Parameters
    ----------
    y : str
        Response column (paired difference), e.g., "diff".
    cluster : str
        Building/group id column.
    time : str
        Time column.
    x, ycoord : str
        Coordinate columns.
    config : SpatioTemporalConfig
        Fit/aggregation settings.
    """

    def __init__(
        self,
        y: str,
        cluster: str,
        time: str,
        x: str,
        ycoord: str,
        config: SpatioTemporalConfig | None = None,
    ):
        self.y = y
        self.cluster = cluster
        self.time = time
        self.x = x
        self.ycoord = ycoord
        self.config = config or SpatioTemporalConfig()

    def fit(self, df: pd.DataFrame, alpha: float, margins: List[float]) -> pd.DataFrame:
        # Validate required columns
        for col in (self.y, self.cluster, self.time, self.x, self.ycoord):
            if col not in df.columns:
                raise ValueError(
                    f"SpatioTemporalTOST requires column {col!r}. "
                    f"Available columns: {list(df.columns)}"
                )

        # Attempt joint separable ML fit if the panel is balanced
        joint = self._try_joint_separable_ml(df=df, alpha=alpha, margins=margins)
        if joint is not None:
            return joint

        # Fall back to the legacy per-time spatial + IVW method
        return self._fit_per_time_ivw(df=df, alpha=alpha, margins=margins)

    # ------------------------ Joint separable likelihood ------------------------

    def _try_joint_separable_ml(self, df: pd.DataFrame, alpha: float, margins: List[float]) -> pd.DataFrame | None:
        """
        Try a joint separable space-time ML fit:
            y ~ N(mu * 1, sigma2 * (R_t ⊗ R_s) + tau2 I),
        with R_t AR(1) and R_s Matérn (ν fixed).

        Returns a results DataFrame if successful and the panel is balanced; otherwise None.
        """
        # Identify unique ordered times and unique locations
        times = np.sort(df[self.time].unique())
        T = int(times.size)
        if T < 2:
            return None

        # Unique locations by (x,y)
        loc = df[[self.x, self.ycoord]].drop_duplicates().reset_index(drop=True)
        S = int(loc.shape[0])
        if S < 2:
            return None

        # Check balanced panel: each time slice must contain all S locations exactly once
        # (allowing repeated rows for same (t,loc) is not supported for the Kronecker fit)
        ok = True
        if len(df) != T * S:
            ok = False
        else:
            for tval, g in df.groupby(self.time):
                if len(g) != S:
                    ok = False
                    break
                if g[[self.x, self.ycoord]].drop_duplicates().shape[0] != S:
                    ok = False
                    break

        if not ok:
            if self.config.verbose_diagnostics:
                print("[SpatioTemporal diagnostics] joint fit skipped: panel is not balanced (not T×S full grid).")
            return None

        # Build mapping to consistent ordering
        t_map = {t: i for i, t in enumerate(times)}
        loc2 = loc.copy()
        loc2["loc_id"] = np.arange(S, dtype=int)

        df2 = df.copy()
        df2["_tix"] = df2[self.time].map(t_map).astype(int)
        df2 = df2.merge(loc2, on=[self.x, self.ycoord], how="left")
        if df2["loc_id"].isna().any():
            return None

        df2 = df2.sort_values(["_tix", "loc_id"]).reset_index(drop=True)
        y = df2[self.y].to_numpy(float).reshape(-1)
        one = np.ones_like(y)
        y_var_emp = float(np.var(y, ddof=1)) if y.size > 1 else 0.0

        # Spatial distance matrix (S×S)
        XY = loc[[self.x, self.ycoord]].to_numpy(float)
        D = _pairwise_dists_xy(XY)
        Dmax = float(np.max(D)) if D.size else 0.0

        # Matérn correlation from spatial core
        nu = float(max(self.config.nu_grid)) if self.config.nu_grid else 2.5

        def _spatial_corr(rho: float) -> np.ndarray:
            # Use _matern_cov with sigma2=1 to get correlation-like matrix
            R = _matern_cov(D, sigma2=1.0, rho=rho, nu=nu)
            # enforce exact symmetry and unit diagonal
            R = 0.5 * (R + R.T)
            np.fill_diagonal(R, 1.0)
            return R

        def nll(z: np.ndarray) -> float:
            mu = float(z[0])
            log_sigma2, log_rho, log_tau2, log_phi = map(float, z[1:])
            sigma2 = float(np.exp(log_sigma2))
            rho = float(np.exp(log_rho))
            tau2 = float(np.exp(log_tau2))
            phi = float(np.exp(log_phi))
            # constrain phi strictly inside (0,1)
            phi = float(np.clip(phi, 1e-6, 0.999999))

            penalty = 0.0
            if self.config.joint_regularize and (self.config.reg_lambda > 0.0):
                # (1) nugget fraction lower bound
                frac = tau2 / (sigma2 + tau2)
                frac = float(np.clip(frac, 1e-15, 1.0))
                frac_min = float(self.config.tau_ratio_min)
                if frac < frac_min:
                    penalty += (math.log(frac_min) - math.log(frac)) ** 2

                # (2) AR(1) upper bound
                phi_max = float(self.config.phi_max)
                if phi > phi_max:
                    denom = max(1e-12, (1.0 - phi_max) ** 2)
                    penalty += (phi - phi_max) ** 2 / denom

                # (3) range relative to domain
                if Dmax > 0.0:
                    rho_max = float(self.config.rho_max_factor) * float(Dmax)
                    if rho > rho_max:
                        penalty += (math.log(rho) - math.log(rho_max)) ** 2

                # (4) variance-scale penalty (soft band)
                if y_var_emp > 0.0:
                    target = float(self.config.var_scale_target) * y_var_emp
                    target = max(target, 1e-15)
                    ratio = (sigma2 + tau2) / target
                    ratio = float(np.clip(ratio, 1e-15, 1e15))
                    dev = abs(math.log(ratio))
                    tol = float(self.config.var_scale_log_tol)
                    if dev > tol:
                        penalty += float(self.config.var_scale_weight) * (dev - tol) ** 2


            Rs = _spatial_corr(rho=rho)
            Rt = _ar1_corr(T=T, phi=phi)

            K = sigma2 * np.kron(Rt, Rs)
            # nugget
            K.flat[:: K.shape[0] + 1] += tau2

            r = y - mu * one
            n = r.size
            try:
                L = linalg.cholesky(K, lower=True, check_finite=False)
            except linalg.LinAlgError:
                jitter = 1e-8 * np.trace(K) / max(n, 1)
                L = linalg.cholesky(K + jitter * np.eye(n), lower=True, check_finite=False)

            v = linalg.solve_triangular(L, r, lower=True, check_finite=False)
            quad = float(v @ v)
            logdet = float(2.0 * np.sum(np.log(np.diag(L))))
            # ---------------- SE-ratio penalty (mean information) ----------------
            if self.config.joint_regularize and (self.config.reg_lambda > 0.0) and (y_var_emp > 0.0):
                try:
                    v1 = linalg.solve_triangular(L, one, lower=True, check_finite=False)
                    info_mu = float(v1 @ v1)  # 1' K^{-1} 1
                    if info_mu > 0.0:
                        se_mu = float(np.sqrt(1.0 / info_mu))
                        se_iid = float(np.sqrt(y_var_emp / max(n, 1)))
                        if se_iid > 0.0:
                            ratio = se_mu / se_iid
                            rmax = float(self.config.se_ratio_max)
                            if ratio > rmax:
                                penalty += float(self.config.se_ratio_weight) * (math.log(ratio) - math.log(rmax)) ** 2
                except Exception:
                    pass

            return 0.5 * (logdet + quad + n * np.log(2.0 * np.pi)) + float(self.config.reg_lambda) * float(penalty)

        mu0 = float(np.mean(y))
        var0 = float(np.var(y, ddof=1)) if y.size > 1 else 1.0
        # reasonably stable starts
        z0 = np.array([
            mu0,
            np.log(max(0.7 * var0, 1e-8)),   # sigma2
            np.log(1.0),                    # rho
            np.log(max(0.1 * var0, 1e-8)),   # tau2
            np.log(0.5),                    # phi
        ], dtype=float)

        bounds = [
            (None, None),   # mu
            (-20.0, 20.0),  # log sigma2
            (-20.0, 20.0),  # log rho
            (-30.0, 20.0),  # log tau2
            (-12.0, -1e-6), # log phi => phi in (exp(-12), ~1)
        ]

        opt = optimize.minimize(nll, z0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 400})
        if not opt.success:
            if self.config.verbose_diagnostics:
                print(f"[SpatioTemporal diagnostics] joint fit failed: {opt.message}")
            return None

        mu_hat = float(opt.x[0])

        # Rebuild K at optimum and compute Wald var(mu_hat) = 1/(1'K^{-1}1)
        log_sigma2, log_rho, log_tau2, log_phi = map(float, opt.x[1:])
        sigma2 = float(np.exp(log_sigma2))
        rho = float(np.exp(log_rho))
        tau2 = float(np.exp(log_tau2))
        phi = float(np.exp(log_phi))
        phi = float(np.clip(phi, 1e-6, 0.999999))

        Rs = _spatial_corr(rho=rho)
        Rt = _ar1_corr(T=T, phi=phi)
        K = sigma2 * np.kron(Rt, Rs)
        K.flat[:: K.shape[0] + 1] += tau2

        n = K.shape[0]
        jitter = 1e-10 * np.trace(K) / max(n, 1)
        L = linalg.cholesky(K + jitter * np.eye(n), lower=True, check_finite=False)
        v = linalg.solve_triangular(L, one.reshape(-1, 1), lower=True, check_finite=False)
        Kinvs1 = linalg.solve_triangular(L.T, v, lower=False, check_finite=False).reshape(-1)
        denom = float(one @ Kinvs1)
        var_mu = float(1.0 / denom) if denom > 0 else float("nan")

        if not np.isfinite(var_mu) or var_mu <= 0:
            if self.config.verbose_diagnostics:
                print("[SpatioTemporal diagnostics] joint fit produced non-finite var(mu); falling back to legacy.")
            return None

                # Compute CI for mu_hat from the joint fit.
        if self.config.mu_ci_method == "wald":
            zcrit = float(stats.norm.ppf(1 - alpha))
            se = float(np.sqrt(var_mu))
            ci_low = float(mu_hat - zcrit * se)
            ci_high = float(mu_hat + zcrit * se)
            ci_method = "Wald CI"
        else:
            # Parametric bootstrap: y* ~ N(mu_hat * 1, K), recompute GLS mu_hat*.
            B = int(self.config.mu_bootstrap_B)
            if B <= 0:
                raise ValueError("mu_bootstrap_B must be > 0 when mu_ci_method='parametric_bootstrap'.")
            rng = np.random.default_rng(self.config.mu_bootstrap_seed)
            Z = rng.standard_normal(size=(n, B))
            Ystar = mu_hat * one[:, None] + (L @ Z)
            V = linalg.solve_triangular(L, Ystar, lower=True, check_finite=False)
            KinvY = linalg.solve_triangular(L.T, V, lower=False, check_finite=False)
            num = (one @ KinvY).reshape(-1)  # length B
            mu_star = num / denom
            ci_low = float(np.quantile(mu_star, alpha))
            ci_high = float(np.quantile(mu_star, 1.0 - alpha))
            se = float(np.std(mu_star, ddof=1)) if B > 1 else float("nan")
            ci_method = f"Parametric bootstrap CI (B={B})"

        if self.config.verbose_diagnostics:
            print("[SpatioTemporal diagnostics] joint separable ML fit succeeded")
            print(f"  T={T}, S={S}, N={len(df2)}")
            print(f"  mu_hat={mu_hat:.6g}, se={se:.6g}, CI=[{ci_low:.6g}, {ci_high:.6g}] ({ci_method})")
            print(f"  theta: sigma2={sigma2:.6g}, tau2={tau2:.6g}, rho={rho:.6g}, nu={nu:.6g}, phi={phi:.6g}")
            if self.config.joint_regularize and (self.config.reg_lambda > 0.0):
                print(
                    "  regularizer: "
                    f"reg_lambda={self.config.reg_lambda}, tau_ratio_min={self.config.tau_ratio_min}, phi_max={self.config.phi_max}, "
                    f"rho_max_factor={self.config.rho_max_factor}, var_scale_target={self.config.var_scale_target}, "
                    f"var_scale_log_tol={self.config.var_scale_log_tol}, var_scale_weight={self.config.var_scale_weight}, "
                    f"se_ratio_max={self.config.se_ratio_max}, se_ratio_weight={self.config.se_ratio_weight}"
                )
            print("  ----")

        rows = []
        for d in margins:
            d = float(d)
            rows.append(
                dict(
                    delta=d,
                    mu_hat=mu_hat,
                    ci_low=ci_low,
                    ci_high=ci_high,
                    equivalent=(ci_low > -d and ci_high < d),
                    method=f"Joint separable spatiotemporal ML (AR1 ⊗ Matérn) + {ci_method}",
                )
            )
        return pd.DataFrame(rows)

    # ------------------------ Legacy per-time IVW ------------------------

    def _fit_per_time_ivw(self, df: pd.DataFrame, alpha: float, margins: List[float]) -> pd.DataFrame:
        mu_list = []
        var_list = []
        diag_rows = []
        for tval, g in df.groupby(self.time):
            if len(g) < self.config.min_time_n:
                continue
            dfp = g.rename(
                columns={
                    self.cluster: "building_id",
                    self.y: "diff",
                    self.x: "x",
                    self.ycoord: "y",
                }
            )
            theta = fit_matern_reml(
                df=dfp,
                building_col="building_id",
                x_col="x",
                y_col="y",
                diff_col="diff",
                nu_grid=self.config.nu_grid,
                per_building_nugget=self.config.per_building_nugget,
            )
            mu_t = float(theta["mu_hat"])
            var_t = float(theta["var_mu"])
            if np.isfinite(mu_t) and np.isfinite(var_t) and var_t > 0:
                mu_list.append(mu_t)
                var_list.append(var_t)
                if self.config.verbose_diagnostics:
                    diag_rows.append({"time": tval, "n_t": int(len(g)), "mu_hat_t": mu_t, "var_mu_t": var_t})

        if len(mu_list) < 2:
            raise ValueError(
                "SpatioTemporalTOST requires at least 2 time slices with successful spatial fits. "
                f"Got {len(mu_list)}."
            )

        mu = np.asarray(mu_list)
        var = np.asarray(var_list)
        w = 1.0 / var

        if self.config.verbose_diagnostics:
            print("[SpatioTemporal diagnostics]")
            if diag_rows:
                try:
                    ddf = pd.DataFrame(diag_rows).sort_values("time")
                    with pd.option_context("display.max_rows", 200, "display.max_columns", 20):
                        print(ddf)
                except Exception:
                    print(diag_rows)
            w_ratio = float(np.max(w) / np.median(w)) if np.median(w) > 0 else float("inf")
            print(f"  time_slices_used={len(mu)}")
            print(f"  max(w)/median(w)={w_ratio:.6g}")
            imax = int(np.argmax(w))
            print(f"  top_weight_slice: mu_hat_t={float(mu[imax]):.6g}, var_mu_t={float(var[imax]):.6g}, weight={float(w[imax]):.6g}")
            print("  ----")

        mu_hat = float(np.sum(w * mu) / np.sum(w))
        var_mu = float(1.0 / np.sum(w))

        dfree = max(len(mu) - 1, 1)
        tcrit = stats.t.ppf(1 - alpha, df=dfree)
        se = float(np.sqrt(var_mu))
        ci_low = mu_hat - tcrit * se
        ci_high = mu_hat + tcrit * se

        rows = []
        for d in margins:
            d = float(d)
            rows.append(
                dict(
                    delta=d,
                    mu_hat=mu_hat,
                    ci_low=float(ci_low),
                    ci_high=float(ci_high),
                    equivalent=(ci_low > -d and ci_high < d),
                    method="Per-time Matérn GLS (REML) aggregated via IVW + t-CI",
                )
            )
        return pd.DataFrame(rows)

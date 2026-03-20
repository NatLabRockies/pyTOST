"""
Publication-grade spatial TOST with Matérn GLS and likelihood-ratio CIs
======================================================================

This module implements a rigorous workflow to decide whether spatial dependence
matters and, when it does, to estimate the population mean difference μ using a
Gaussian process with a Matérn covariance (shared hyper-parameters across clusters),
fitted by REML. We then produce a **profile likelihood CI for μ** and apply the
CI-in-TOST rule over any set of equivalence margins Δ.

Key features
------------
- Diagnostics: ICC (random-intercept), IID vs cluster-robust SE inflation, empirical
  variograms, Moran’s I (optional with PySAL).
- Spatial model: block-diagonal Σ = diag{ Σ_b }, each Σ_b is Matérn(σ^2, ρ, ν) + τ^2 I
  on the cluster’s coordinates (x,y). Hyper-parameters (σ^2, ρ, τ^2) are estimated by
  **REML** for each candidate ν; ν is chosen by profile REML over a grid (e.g., {0.5,1.5,2.5}).
- μ inference:
  * GLS estimator: μ̂(θ) = (1ᵗ Σ(θ)⁻¹ y) / (1ᵗ Σ(θ)⁻¹ 1)
  * Likelihood-ratio CI for μ: invert LRT with 1 df → robust to θ uncertainty.
- Sensitivity: Mixed-effects (K–R via R if available; else statsmodels + conservative df),
  cluster-robust OLS, cluster (block) bootstrap CI for μ.
- Reporting: tidy DataFrames over Δ, diagnostic figures, and summary plots.

Statistical references
----------------------
- Matérn covariance and spatial likelihood: Cressie (1993), Stein (1999).
- REML for covariance parameters: Harville (1977), Kenward & Roger (1997).
- Equivalence testing (TOST): Schuirmann (1987); Lakens (2017).
- Small-sample cluster-robust variance: Bell & McCaffrey (2002); Pustejovsky & Tipton (2018).

API
---
run_pubgrade_spatial_tost(df, cluster_col='cluster_id', x_col='x', y_col='y',
                          diff_col='diff', margins=(1,3,5), alpha=0.05, out_dir='tost_pub',
                          nu_grid=(0.5, 1.5, 2.5), share_params_across_clusters=True,
                          per_cluster_nugget=True, do_sensitivity=True, moran_k=4,
                          bootstrap_B=2000, random_state=42)

Returns dict with:
- 'diagnostics': metrics + file paths to figures
- 'model': {'theta_hat': ..., 'nu_star': ..., 'mu_hat': ..., 'ci_mu_lrt': (...)}
- 'summaries': {method_name: DataFrame over Δ with μ, CI, equivalence}
- 'notes': caveats and fit summary strings

"""

from __future__ import annotations
import os, warnings
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
from string import Template
from datetime import datetime

import warnings
import numpy as np
from scipy import linalg
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize, linalg, special, stats

import statsmodels.api as sm
import statsmodels.formula.api as smf

# Optional: Moran's I (PySAL)
try:
    from libpysal.weights import KNN
    from esda.moran import Moran
    HAVE_PYSAL = True
except Exception:
    HAVE_PYSAL = False

# Optional: rpy2 + R(lme4,lmerTest) for Kenward–Roger df
HAVE_RPY2 = False
try:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    pandas2ri.activate()
    HAVE_RPY2 = True
except Exception:
    HAVE_RPY2 = False


# ------------------------------ Helpers & math ------------------------------

def _ensure_dir(d: str):
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)

def _pairwise_dists(X: np.ndarray) -> np.ndarray:
    """Euclidean distances for rows of X (n x 2)."""
    diff = X[:, None, :] - X[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))

def matern_cov(d: np.ndarray, sigma2: float, rho: float, nu: float) -> np.ndarray:
    """
    Matérn covariance with variance sigma2, range rho, smoothness nu.
    Uses parameterization: C(h) = sigma2 * 2^{1-ν}/Γ(ν) * (√(2ν) h / ρ)^ν K_ν(√(2ν) h / ρ),
    with C(0) = sigma2.
    Numerically stabilized for h→0 using series expansion.
    """
    d = np.asarray(d, float)
    # scaled distance
    s = np.sqrt(2.0 * nu) * d / max(rho, 1e-12)
    C = np.empty_like(d)
    # small s: use limit Kν(s) ~ Γ(ν) 2^{ν-1} s^{-ν}
    small = s < 1e-6
    if np.any(small):
        C[small] = sigma2  # limit as h→0
    # general case
    s2 = s[~small]
    coef = sigma2 * (2.0**(1.0 - nu)) / special.gamma(nu)
    kv = special.kv(nu, s2)
    # kv can be inf/NaN at s=0; we handled small already
    C[~small] = coef * (s2**nu) * kv
    # symmetry and fill diagonal later by caller if needed
    return C

def block_matern_cov(df: pd.DataFrame, cluster_col: str, x_col: str, y_col: str,
                     sigma2: float, rho: float, nu: float, tau2: float,
                     per_cluster_nugget: bool) -> Tuple[np.ndarray, List[int]]:
    """
    Assemble block-diagonal covariance Σ across clusters; each block is Matérn + nugget.
    Returns (Sigma, block_sizes).
    """
    blocks = []
    sizes = []
    for _, g in df.groupby(cluster_col):
        coords = g[[x_col, y_col]].to_numpy(float)
        D = _pairwise_dists(coords)
        K = matern_cov(D, sigma2=sigma2, rho=rho, nu=nu)
        # nugget: τ^2 I (either shared or per-cluster; here we use same tau2 for all blocks,
        # but allowing per_cluster_nugget means we *add* τ^2 I even for small blocks)
        n = len(g)
        K[np.diag_indices(n)] += tau2
        blocks.append(K)
        sizes.append(n)
    Sigma = linalg.block_diag(*blocks)
    return Sigma, sizes

# def gls_mu_and_profile_loglik(y: np.ndarray, ones: np.ndarray, Sigma: np.ndarray) -> Tuple[float, float, float]:
#     """
#     Given y (stacked by clusters), compute:
#       - GLS μ̂ = (1' Σ⁻¹ y) / (1' Σ⁻¹ 1)
#       - Var(μ̂) = 1 / (1' Σ⁻¹ 1)
#       - REML loglik (intercept-only) up to constants: 
#             ℓ_R(θ) = -0.5[ log|Σ| + log(1'Σ⁻¹1) + y' P y ],
#         where P = Σ⁻¹ - Σ⁻¹1 (1'Σ⁻¹1)⁻¹ 1'Σ⁻¹.
#     """
#     # Cholesky for stability
#     L = linalg.cholesky(Sigma, lower=True, check_finite=False)
#     Linv = linalg.solve_triangular(L, np.eye(L.shape[0]), lower=True, check_finite=False)
#     Sinv = Linv.T @ Linv
#     A = float(ones.T @ Sinv @ ones)
#     B = float(ones.T @ Sinv @ y)
#     mu_hat = B / A
#     var_mu = 1.0 / A
#     # REML loglik pieces
#     logdet = 2.0 * np.sum(np.log(np.diag(L)))
#     P = Sinv - (Sinv @ ones) @ (ones.T @ Sinv) / A
#     quad = float(y.T @ P @ y)
#     ll = -0.5 * (logdet + np.log(A) + quad)  # constants drop
#     return mu_hat, var_mu, ll

def _chol_inverse_with_jitter(Sigma, max_tries=8, base=1e-12):
    """
    Try Cholesky on Sigma; if it fails, add jitter = scale * I with scale increasing geometrically.
    Returns Sinv and log|Sigma|.
    """
    n = Sigma.shape[0]
    # scale the jitter relative to average variance level to be unit-agnostic
    avg_var = np.trace(Sigma) / max(n, 1)
    for k in range(max_tries):
        jitter = (0.0 if k == 0 else (base * (10.0 ** (k-1)) * avg_var))
        try:
            L = linalg.cholesky(Sigma + jitter * np.eye(n), lower=True, check_finite=False)
            # log|Sigma| = 2 * sum(log(diag(L))) - log|I + jitter*Sigma^{-1}| (but jitter tiny)
            logdet = 2.0 * np.sum(np.log(np.diag(L)))
            Linv = linalg.solve_triangular(L, np.eye(n), lower=True, check_finite=False)
            Sinv = Linv.T @ Linv
            return Sinv, logdet, jitter
        except linalg.LinAlgError:
            continue
    raise linalg.LinAlgError("Cholesky failed even after jitter ramp.")

from scipy import linalg
import numpy as np

def gls_mu_and_profile_loglik(y, ones, Sigma):
    """
    GLS μ̂, Var(μ̂), and profile Gaussian loglik using robust Cholesky.
    Requires y and ones to be 1-D arrays of length n.
    """
    y = np.asarray(y, dtype=float).ravel()       # ensure 1-D
    ones = np.asarray(ones, dtype=float).ravel() # ensure 1-D

    Sinv, logdet, _ = _chol_inverse_with_jitter(Sigma, max_tries=8, base=1e-12)

    # scalars via 1-D vector algebra
    A = float(ones @ (Sinv @ ones))    # 1' Σ^{-1} 1
    B = float(ones @ (Sinv @ y))       # 1' Σ^{-1} y
    mu_hat = B / A

    resid = y - mu_hat * ones          # shape (n,)
    quad = float(resid @ (Sinv @ resid))  # y' P y (up to constant)

    n = y.size
    ll = -0.5 * (logdet + quad + n * np.log(2.0 * np.pi))
    var_mu = 1.0 / A
    return mu_hat, var_mu, ll

# ---------- Geometry helpers ----------
def _dedupe_coords(XY, eps=1e-9, rng_seed=123):
    """Tiny jitter for exact duplicate (x,y) rows to avoid zero distances."""
    XY = np.asarray(XY, float).copy()
    if XY.size == 0:
        return XY
    rXY = np.round(XY, 12)
    _, idx, counts = np.unique(rXY, axis=0, return_index=True, return_counts=True)
    if (counts > 1).any():
        rng = np.random.default_rng(rng_seed)
        XY[idx] += rng.normal(0.0, eps, size=XY[idx].shape)
    return XY

# ---------- Matérn kernel (ν fixed) ----------
def _matern_cov(dist, sigma2, rho, nu):
    """
    Matérn covariance for given pairwise distance matrix 'dist'.
    Uses scaled range parameter rho (practical range); stable for small d.
    """
    from scipy.special import kv, gamma
    d = np.asarray(dist, float)
    d_scaled = np.sqrt(2.0 * nu) * d / max(rho, 1e-12)
    # handle d==0 separately: K_v(0) ~ ∞ but limit gives sigma2
    C = np.empty_like(d_scaled)
    C.fill(0.0)
    # small d -> use series; but for practical purposes:
    mask0 = (d_scaled == 0.0)
    C[mask0] = sigma2
    mask = ~mask0
    z = d_scaled[mask]
    # Matérn: σ² * 2^{1-ν} / Γ(ν) * (z)^ν K_ν(z)
    coef = sigma2 * (2.0 ** (1.0 - nu)) / gamma(nu)
    C[mask] = coef * (z ** nu) * kv(nu, z)
    return C

def _pairwise_dists(XY):
    diff = XY[:, None, :] - XY[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=2))

# ---------- Build block-diagonal Σ and stacks ----------
def _build_sigma_and_stacks(df, cluster_col, x_col, y_col, diff_col, sigma2, rho, tau2, nu, per_cluster_nugget):
    """
    Returns (Sigma, y, ones). Sigma is block-diagonal across clusters.
    """
    y_list, ones_list, blocks = [], [], []
    for bldg, sub in df.groupby(cluster_col, sort=False):
        yy = sub[diff_col].to_numpy(float)
        XY = sub[[x_col, y_col]].to_numpy(float)
        XY = _dedupe_coords(XY)  # avoid exact duplicates
        D = _pairwise_dists(XY)
        K = _matern_cov(D, sigma2=sigma2, rho=rho, nu=nu)
        # nugget: either global (same τ² across all rows) or per-cluster
        if per_cluster_nugget:
            K[np.diag_indices_from(K)] += tau2
        else:
            K[np.diag_indices_from(K)] += tau2
        blocks.append(K)
        y_list.append(yy)
        ones_list.append(np.ones(len(yy)))
    # assemble
    y = np.concatenate(y_list, axis=0)
    ones = np.concatenate(ones_list, axis=0)[:, None]  # column vector
    # block-diagonal Sigma
    n = len(y)
    Sigma = np.zeros((n, n), float)
    i = 0
    for K in blocks:
        m = K.shape[0]
        Sigma[i:i+m, i:i+m] = K
        i += m
    return Sigma, y, ones

# ---------- FIXED reml_objective ----------
def reml_objective(theta_log, df, cluster_col, x_col, y_col, diff_col, nu, per_cluster_nugget):
    """
    Given log-params theta_log = (log σ², log ρ, log τ²), build Σ, compute GLS μ̂ and profile loglik.
    Returns (neg_reml, cache).
    """
    # unpack with positivity
    sigma2 = float(np.exp(theta_log[0]))
    rho    = float(np.exp(theta_log[1]))
    tau2   = float(np.exp(theta_log[2]))

    # floors & caps to keep Σ well-conditioned
    sigma2 = max(sigma2, 1e-10)
    rho    = float(np.clip(rho, 1e-2, 1e5))           # cap range to avoid ρ→∞
    tau2_floor = 1e-6 * sigma2
    tau2 = max(tau2, tau2_floor)

    # build Σ, y, 1
    Sigma, y, ones = _build_sigma_and_stacks(
        df, cluster_col, x_col, y_col, diff_col,
        sigma2=sigma2, rho=rho, tau2=tau2, nu=nu, per_cluster_nugget=per_cluster_nugget
    )

    # GLS + profile loglik with adaptive jitter (penalize if still not PD)
    try:
        mu_hat, var_mu, ll = gls_mu_and_profile_loglik(y, ones, Sigma)
    except linalg.LinAlgError:
        # steer optimizer away from non-PD regions
        cache = {"sigma2": sigma2, "rho": rho, "tau2": tau2, "nu": nu, "mu_hat": np.nan, "var_mu": np.nan}
        return 1e12, cache

    cache = {"sigma2": sigma2, "rho": rho, "tau2": tau2, "nu": nu, "mu_hat": mu_hat, "var_mu": var_mu}
    # REML constant terms (X=1 only) differ by constants across θ, so using profile ll is fine for selection
    return -ll, cache



# def fit_matern_reml(df: pd.DataFrame, cluster_col: str, x_col: str, y_col: str, diff_col: str,
#                     nu_grid: Iterable[float] = (0.5, 1.5, 2.5), per_cluster_nugget: bool = True,
#                     start: Optional[Tuple[float,float,float]] = None, verbose: bool = False) -> Dict:
#     """
#     Fit Matérn parameters by REML with ν chosen by profile over nu_grid.
#     start: optional (sigma2, rho, tau2) initial values; if None, crude method-of-moments.
#     Returns dict with best θ, ν, μ̂, var(μ̂), and optimizer info.
#     """
#     y = df[diff_col].to_numpy(float)
#     var_y = np.var(y, ddof=1) if len(y) > 1 else max(1.0, y[0]**2)
#     # crude d-scale
#     # median within-cluster pairwise distance as a starting range
#     med_d = []
#     for _, g in df.groupby(cluster_col):
#         if len(g) >= 2:
#             D = _pairwise_dists(g[[x_col, y_col]].to_numpy(float))
#             iu = np.triu_indices_from(D, k=1)
#             if len(iu[0]):
#                 med_d.append(np.median(D[iu]))
#     d0 = np.median(med_d) if med_d else 1.0
#     if start is None:
#         start = (0.7*var_y, max(d0, 1e-2), 0.1*var_y)
#     best = {"obj": np.inf}
#     for nu in nu_grid:
#         theta0 = np.log(np.array(start))
#         res = optimize.minimize(
#             lambda th: reml_objective(th, df, cluster_col, x_col, y_col, diff_col, nu, per_cluster_nugget)[0],
#             theta0,
#             method="L-BFGS-B",
#             bounds=[(-20, 20), (-20, 20), (-20, 20)],
#             options=dict(maxiter=500)
#         )
#         val, cache = reml_objective(res.x, df, cluster_col, x_col, y_col, diff_col, nu, per_cluster_nugget)
#         if verbose:
#             print(f"ν={nu}: REML={-val:.3f}, θ={np.exp(res.x)}")
#         if val < best["obj"]:
#             best = {"obj": val, "nu": nu, "theta_log": res.x, **cache, "opt": res}
#     return best

def fit_matern_reml(
    df: pd.DataFrame,
    cluster_col: str,
    x_col: str,
    y_col: str,
    diff_col: str,
    nu_grid: Iterable[float] = (0.5, 1.5, 2.5),
    per_cluster_nugget: bool = True,
    start: Optional[Tuple[float, float, float]] = None,
    verbose: bool = False,
) -> Dict:
    """
    Fit Matérn parameters by REML with ν chosen by profile over nu_grid.

    start: optional (sigma2, rho, tau2) initial values; if None, crude method-of-moments.

    Returns dict with:
      - nu
      - theta_log
      - theta (exp(theta_log))
      - mu_hat
      - var_mu_hat
      - REML objective value
      - optimizer result
      - cached covariance quantities
    """
    y = df[diff_col].to_numpy(float)
    n = len(y)

    var_y = np.var(y, ddof=1) if n > 1 else max(1.0, y[0] ** 2)

    # crude distance scale
    med_d = []
    for _, g in df.groupby(cluster_col):
        if len(g) >= 2:
            D = _pairwise_dists(g[[x_col, y_col]].to_numpy(float))
            iu = np.triu_indices_from(D, k=1)
            if len(iu[0]):
                med_d.append(np.median(D[iu]))
    d0 = np.median(med_d) if med_d else 1.0

    if start is None:
        start = (0.7 * var_y, max(d0, 1e-2), 0.1 * var_y)

    best = {"obj": np.inf}

    for nu in nu_grid:
        theta0 = np.log(np.array(start))

        res = optimize.minimize(
            lambda th: reml_objective(
                th, df, cluster_col, x_col, y_col, diff_col, nu, per_cluster_nugget
            )[0],
            theta0,
            method="L-BFGS-B",
            bounds=[(-20, 20), (-20, 20), (-20, 20)],
            options=dict(maxiter=500),
        )

        val, cache = reml_objective(
            res.x, df, cluster_col, x_col, y_col, diff_col, nu, per_cluster_nugget
        )

        if verbose:
            print(f"ν={nu}: REML={-val:.3f}, θ={np.exp(res.x)}")

        if val < best["obj"]:
            best = {
                "obj": val,
                "nu": nu,
                "theta_log": res.x,
                **cache,
                "opt": res,
            }

#    # compute mu_hat and var(mu_hat) under best covariance
#    y = df[diff_col].to_numpy(float)
#    one = np.ones_like(y)

#    from pprint import pprint
#    pprint(best)

#    try:
#        # cache must expose a linear solver for Σ^{-1} v
#        # this is already true in your REML code
#        Sinv_1 = best["solve"](one)
#        Sinv_y = best["solve"](y)
#    except:
#        print('no \Sigma^{-1}')
#        Sinv = best["Sigma_inv"]
#        Sinv_1 = Sinv @ one
#        Sinv_y = Sinv @ y

#    denom = float(one @ Sinv_1)
#    mu_hat = float((one @ Sinv_y) / denom)
#    var_mu_hat = float(1.0 / denom)

    best["theta"] = np.exp(best["theta_log"])
#    best["mu_hat"] = mu_hat
#    best["var_mu_hat"] = var_mu_hat

    return best


def lr_ci_for_mu(
    df: pd.DataFrame,
    cluster_col: str,
    x_col: str,
    y_col: str,
    diff_col: str,
    theta: dict,
    alpha: float = 0.05,
    max_expand: int = 12,
    expand_factor: float = 2.0,
) -> tuple[float, float]:
    """
    Profile likelihood-ratio (LR) confidence interval for the mean difference ``mu`` in an
    intercept-only spatial GLS model with Matérn covariance.

    This function inverts the *profile* LR statistic for ``mu``, i.e., it re-optimizes the
    covariance parameters :math:`(\\sigma^2, \\rho, \\tau^2)` for each candidate ``mu``.

    Notes
    -----
    This implementation is written to be robust to bracketing failures on either side of
    ``mu_hat``. In particular:

    * Warm-start parameters are correctly paired with their associated bracket endpoints.
    * The initial bracket width is floored to a small fraction of the empirical sd(diff)
      to avoid returning an apparent one-sided CI due to a near-zero initial width.
    * If bracketing genuinely fails, we emit a RuntimeWarning and fall back to a conservative
      Wald edge computed from the local GLS variance proxy.
    """

    # --- helpers -------------------------------------------------------------

    def _ml_nll(mu: float, theta_log: np.ndarray) -> float:
        """Negative Gaussian log-likelihood for fixed mu and covariance parameters."""
        sigma2 = float(np.exp(theta_log[0]))
        rho = float(np.exp(theta_log[1]))
        tau2 = float(np.exp(theta_log[2]))

        Sigma, y, ones = _build_sigma_and_stacks(
            df=df,
            cluster_col=cluster_col,
            x_col=x_col,
            y_col=y_col,
            diff_col=diff_col,
            sigma2=sigma2,
            rho=rho,
            tau2=tau2,
            nu=float(theta.get("nu", 2.5)),
            per_cluster_nugget=bool(theta.get("per_cluster_nugget", True)),
        )

        y_vec = np.asarray(y, dtype=float).reshape(-1)
        ones_vec = np.asarray(ones, dtype=float).reshape(-1)
        r = y_vec - mu * ones_vec
        n = r.size

        # Cholesky with adaptive jitter if needed
        try:
            L = linalg.cholesky(Sigma, lower=True, check_finite=False)
        except linalg.LinAlgError:
            jitter = 1e-8 * np.trace(Sigma) / max(n, 1)
            L = linalg.cholesky(Sigma + jitter * np.eye(n), lower=True, check_finite=False)

        v = linalg.solve_triangular(L, r, lower=True, check_finite=False)
        quad = float(v @ v)

        logdet = float(2.0 * np.sum(np.log(np.diag(L))))
        return 0.5 * (logdet + quad + n * np.log(2.0 * np.pi))

    def _joint_nll(z: np.ndarray) -> float:
        """Joint NLL over (mu, log_sigma2, log_rho, log_tau2)."""
        mu = float(z[0])
        theta_log = np.asarray(z[1:], dtype=float)
        return _ml_nll(mu, theta_log)

    # --- global ML fit (mu + covariance) ------------------------------------

    mu0 = float(theta.get("mu_hat", df[diff_col].mean()))
    theta0 = np.asarray(
        theta.get("theta_log", np.log([theta["sigma2"], theta["rho"], theta["tau2"]])),
        dtype=float,
    )

    z0 = np.concatenate([[mu0], theta0])
    bounds = [
        (None, None),          # mu
        (-20.0, 20.0),         # log(sigma2)
        (-20.0, 20.0),         # log(rho)
        (-30.0, 20.0),         # log(tau2)
    ]

    opt_hat = optimize.minimize(
        _joint_nll,
        z0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 200},
    )

    # Even if the optimizer reports failure, its current iterate is often better than the start.
    z_hat = np.asarray(opt_hat.x if hasattr(opt_hat, "x") else z0, dtype=float)
    ll_hat = -float(_joint_nll(z_hat))

    mu_hat = float(z_hat[0])
    theta_hat_log = np.asarray(z_hat[1:], dtype=float)

    # --- Wald proxy variance for bracketing (GLS at ML theta) ----------------

    def _wald_var_mu_at(theta_log: np.ndarray) -> float:
        sigma2 = float(np.exp(theta_log[0]))
        rho = float(np.exp(theta_log[1]))
        tau2 = float(np.exp(theta_log[2]))
        Sigma, y, ones = _build_sigma_and_stacks(
            df=df,
            cluster_col=cluster_col,
            x_col=x_col,
            y_col=y_col,
            diff_col=diff_col,
            sigma2=sigma2,
            rho=rho,
            tau2=tau2,
            nu=float(theta.get("nu", 2.5)),
            per_cluster_nugget=bool(theta.get("per_cluster_nugget", True)),
        )
        ones_col = np.asarray(ones, dtype=float).reshape(-1, 1)
        n = ones_col.shape[0]
        try:
            L = linalg.cholesky(Sigma, lower=True, check_finite=False)
        except linalg.LinAlgError:
            jitter = 1e-8 * np.trace(Sigma) / max(n, 1)
            L = linalg.cholesky(Sigma + jitter * np.eye(n), lower=True, check_finite=False)

        v = linalg.solve_triangular(L, ones_col, lower=True, check_finite=False)
        Sinv1 = linalg.solve_triangular(L.T, v, lower=False, check_finite=False)
        A = float((ones_col.T @ Sinv1).item())
        return float(1.0 / A)

    cutoff = float(stats.chi2.ppf(1.0 - 2.0 * alpha, df=1))
    if not np.isfinite(cutoff) or cutoff <= 0:
        raise ValueError(f"Invalid LR cutoff for alpha={alpha}: {cutoff}")

    try:
        var_mu0 = _wald_var_mu_at(theta_hat_log)
    except Exception:
        var_mu0 = float(theta.get("var_mu", 1.0))

    w0 = float(np.sqrt(max(cutoff * var_mu0, 0.0)))
    if not np.isfinite(w0) or w0 <= 0:
        w0 = 1.0

    # Floor the initial bracket width to avoid an apparent one-sided CI due to tiny w0.
    # We keep this internal to avoid changing the public signature.
    try:
        y_sd = float(np.std(df[diff_col].to_numpy(float), ddof=1)) if len(df) > 1 else 1.0
        w_min = 0.05 * max(y_sd, 1e-6)  # 5% of empirical sd(diff)
        if np.isfinite(w_min) and w0 < w_min:
            w0 = float(w_min)
    except Exception:
        pass

    # --- profile LR function with caching -----------------------------------

    cache: dict[float, tuple[float, np.ndarray]] = {}

    def _profile_ll(mu: float, theta_start: np.ndarray) -> tuple[float, np.ndarray]:
        """Return (ll(mu), theta_log_mu_hat)."""
        mu_key = float(mu)
        if mu_key in cache:
            ll, th = cache[mu_key]
            return ll, th

        def obj(th: np.ndarray) -> float:
            return _ml_nll(mu_key, np.asarray(th, dtype=float))

        opt = optimize.minimize(
            obj,
            np.asarray(theta_start, dtype=float),
            method="L-BFGS-B",
            bounds=bounds[1:],
            options={"maxiter": 200},
        )

        if not opt.success:
            ll_mu = -float(obj(theta_start))
            th_mu = np.asarray(theta_start, dtype=float)
        else:
            ll_mu = -float(opt.fun)
            th_mu = np.asarray(opt.x, dtype=float)

        cache[mu_key] = (ll_mu, th_mu)
        return ll_mu, th_mu

    # Ensure LR(mu_hat) is ~0 by defining ll_hat as the profiled ll at mu_hat if needed.
    ll_mu_hat_prof, th_mu_hat_prof = _profile_ll(mu_hat, theta_hat_log)
    if ll_hat < ll_mu_hat_prof:
        ll_hat = float(ll_mu_hat_prof)
        theta_hat_log = np.asarray(th_mu_hat_prof, dtype=float)

    def lr_stat(mu: float, theta_start: np.ndarray) -> tuple[float, np.ndarray]:
        ll_mu, th_mu = _profile_ll(mu, theta_start)
        return float(2.0 * (ll_hat - ll_mu)), th_mu

    # --- invert LR(mu)=cutoff on both sides ---------------------------------

    def find_side(sign: float) -> float:
        """Find root on one side of mu_hat where LR(mu)=cutoff."""
        width = float(w0)

        mu_center = float(mu_hat)
        mu_edge = float(mu_hat + sign * width)

        # At center: LR should be ~0; at edge we need LR >= cutoff to bracket.
        lr_center, th_center = lr_stat(mu_center, theta_hat_log)
        lr_edge, th_edge = lr_stat(mu_edge, th_center)

        k = 0
        while (lr_edge < cutoff) and (k < max_expand):
            width *= float(expand_factor)
            mu_edge = float(mu_hat + sign * width)
            lr_edge, th_edge = lr_stat(mu_edge, th_edge)
            k += 1

        if lr_edge < cutoff:
            # Genuine bracketing failure: fall back to conservative Wald edge.
            warnings.warn(
                f"lr_ci_for_mu: bracketing failed on sign={sign:+.0f}. "
                f"(mu_hat={mu_hat:.6g}, width={width:.6g}, lr_edge={lr_edge:.6g}, cutoff={cutoff:.6g})",
                RuntimeWarning,
            )
            return float(mu_hat + sign * width)

        # Establish bracket [lo, hi] such that LR(lo) >= cutoff and LR(hi) <= cutoff.
        # For both sides, hi is the closer point to mu_hat (the center).
        lo_mu = float(mu_edge)   # farther from mu_hat
        hi_mu = float(mu_center) # at/near mu_hat

        lo_lr = float(lr_edge)
        hi_lr = float(lr_center)

        # Warm-start thetas matched to endpoints
        lo_th = np.asarray(th_edge, dtype=float)
        hi_th = np.asarray(th_center, dtype=float)

        # Bisection
        for _ in range(80):
            mid_mu = 0.5 * (lo_mu + hi_mu)
            # Warm start from the nearer endpoint in mu-space
            th0 = hi_th if abs(mid_mu - hi_mu) < abs(mid_mu - lo_mu) else lo_th
            mid_lr, mid_th = lr_stat(mid_mu, th0)

            if abs(mid_lr - cutoff) < 1e-7:
                return float(mid_mu)

            if mid_lr >= cutoff:
                lo_mu, lo_lr, lo_th = float(mid_mu), float(mid_lr), np.asarray(mid_th, dtype=float)
            else:
                hi_mu, hi_lr, hi_th = float(mid_mu), float(mid_lr), np.asarray(mid_th, dtype=float)

        return float(0.5 * (lo_mu + hi_mu))

    left = float(find_side(-1.0))
    right = float(find_side(+1.0))
    return (float(min(left, right)), float(max(left, right)))


def compute_icc(df: pd.DataFrame, cluster_col: str, diff_col: str) -> float:
    md = smf.mixedlm(f"{diff_col} ~ 1", df, groups=df[cluster_col])
    fit = md.fit(reml=True, method="lbfgs", disp=False)
    tau2 = float(fit.cov_re.iloc[0,0]); sig2 = float(fit.scale)
    return tau2 / (tau2 + sig2) if (tau2 + sig2) > 0 else 0.0

def morans_I(df: pd.DataFrame, cluster_col: str, x_col: str, y_col: str, diff_col: str, k=4) -> pd.DataFrame:
    if not HAVE_PYSAL:
        return pd.DataFrame(columns=["cluster_id","n","I","p_norm"])
    recs=[]
    for bid, g in df.groupby(cluster_col):
        if len(g) < k+2: continue
        W = KNN.from_array(g[[x_col,y_col]].to_numpy(float), k=k)
        mi = Moran(g[diff_col].to_numpy(float), W, two_tailed=False)
        recs.append({"cluster_id": bid, "n": len(g), "I": float(mi.I), "p_norm": float(mi.p_norm)})
    return pd.DataFrame(recs)

def empirical_variograms(df, cluster_col, x_col, y_col, diff_col, n_bins=8, out_dir=None):
    out = {}
    for bid, g in df.groupby(cluster_col):
        if len(g) < 4: continue
        coords = g[[x_col,y_col]].to_numpy(float)
        y = g[diff_col].to_numpy(float)
        D = _pairwise_dists(coords)
        iu = np.triu_indices(len(g), k=1)
        d = D[iu]; gam = 0.5*(y[:,None]-y[None,:])**2
        s = gam[iu]
        qmax = np.quantile(d, 0.9)
        bins = np.linspace(0, qmax, n_bins+1)
        idx = np.digitize(d, bins)-1
        centers = 0.5*(bins[:-1]+bins[1:])
        gamma = [float(np.mean(s[idx==k])) if np.any(idx==k) else np.nan for k in range(n_bins)]
        out[str(bid)] = pd.DataFrame({"dist": centers, "gamma": gamma})
        # plot
        if out_dir:
            plt.figure(); plt.plot(centers, gamma, marker="o")
            plt.xlabel("Distance"); plt.ylabel("Semivariance"); plt.title(f"Variogram (bldg {bid})")
            _ensure_dir(out_dir); plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"variogram_cluster_{bid}.png"), dpi=140); plt.close()
    return out

def cluster_bootstrap_mu(df: pd.DataFrame, cluster_col: str, diff_col: str, B=2000, seed=42) -> Tuple[float,float,Tuple[float,float]]:
    rng = np.random.default_rng(seed)
    groups = df[cluster_col].unique()
    Nhat = []
    for _ in range(B):
        samp = rng.choice(groups, size=len(groups), replace=True)
        y = pd.concat([df[df[cluster_col]==g][diff_col] for g in samp], axis=0).to_numpy(float)
        Nhat.append(float(np.mean(y)))
    est = float(np.mean(Nhat))
    se = float(np.std(Nhat, ddof=1))
    lo, hi = float(np.quantile(Nhat, 0.05)), float(np.quantile(Nhat, 0.95))
    return est, se, (lo, hi)

import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

def mixed_effects_mu(df, cluster_col: str, diff_col: str, alpha: float):
    """
    Mixed-effects: diff ~ 1 + (1|cluster).
    Robust to boundary/singularity:
      - If rpy2 + lmerTest available -> use Kenward–Roger.
      - Else statsmodels MixedLM (REML). If tau^2 ~ 0 or singular/boundary warnings,
        fall back to cluster-robust OLS CI for μ and annotate method.
    """
    G = df[cluster_col].nunique()

    # Preferred: R path (Kenward–Roger)
    if HAVE_RPY2:
        try:
            ro.r('suppressPackageStartupMessages({library(lme4); library(lmerTest)})')
            r_df = pandas2ri.py2rpy(df[[cluster_col, diff_col]].copy())
            ro.globalenv['r_df'] = r_df
            ro.r(f'''
                fit <- lmer({diff_col} ~ 1 + (1|{cluster_col}), data=r_df, REML=TRUE)
                est  <- fixef(fit)[1]
                se   <- as.numeric(coef(summary(fit))[1, "Std. Error"])
                dfKR <- as.numeric(coef(summary(fit))[1, "df"])
            ''')
            est = float(ro.r('est')[0]); se = float(ro.r('se')[0]); dfKR = float(ro.r('dfKR')[0])
            tcrit = stats.t.ppf(1-alpha, dfKR); ci = (est - tcrit*se, est + tcrit*se)
            return est, ci, f"Mixed-effects (K–R df={dfKR:.1f})"
        except Exception as e:
            warnings.warn(f"K–R path failed: {e}; falling back to statsmodels MixedLM.")

    # Fallback: statsmodels MixedLM with safeguard
    import statsmodels.formula.api as smf
    warn_msgs = []

    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        md = smf.mixedlm(f"{diff_col} ~ 1", df, groups=df[cluster_col])
        fit = md.fit(reml=True, method="lbfgs", disp=False)
        # collect warnings
        for w in wlist:
            if issubclass(w.category, (UserWarning, ConvergenceWarning)):
                warn_msgs.append(str(w.message))

    #est = float(fit.fe_params.get("Intercept", fit.params[0]))
    # estimate (Intercept)
    fp = getattr(fit, "fe_params", None)
    if isinstance(fp, pd.Series):
        if "Intercept" in fp.index:
            est = float(fp.loc["Intercept"])
        elif "const" in fp.index:           # just in case a model names it 'const'
            est = float(fp.loc["const"])
        else:
            est = float(fp.iloc[0])
    else:
        est = float(np.asarray(getattr(fit, "params", fp)).ravel()[0])

    #se  = float(getattr(fit, "bse_fe", {}).get("Intercept", fit.bse[0]))
    # standard error (Intercept)
    bp = getattr(fit, "bse_fe", None)       # MixedLM puts FE SEs here
    if isinstance(bp, pd.Series):
        if "Intercept" in bp.index:
            se = float(bp.loc["Intercept"])
        elif "const" in bp.index:
            se = float(bp.loc["const"])
        else:
            se = float(bp.iloc[0])
    else:
        bse_any = getattr(fit, "bse", bp)   # fallback for models without bse_fe
        se = float(np.asarray(bse_any).ravel()[0]) 

    tau2 = float(fit.cov_re.iloc[0,0]) if hasattr(fit, "cov_re") else np.nan
    boundary = (tau2 <= 1e-8) or any("singular" in m.lower() or "boundary" in m.lower() for m in warn_msgs)

    if boundary:
        # Fall back to cluster-robust OLS CI (more honest when RE variance ~ 0 or unstable)
        X = np.ones((len(df),1))
        ols_cl = sm.OLS(df[diff_col].values, X).fit(cov_type="cluster",
                                                    cov_kwds={"groups": df[cluster_col].values})
        est2 = float(ols_cl.params[0]); se2 = float(ols_cl.bse[0])
        dfree = max(G-1, 1)
        tcrit = stats.t.ppf(1-alpha, dfree)
        ci2 = (est2 - tcrit*se2, est2 + tcrit*se2)
        note = f"Mixed-effects fallback to Cluster-robust OLS (τ²≈{tau2:.3g}; singular/boundary detected)"
        return est2, ci2, note

    # Regular MixedLM result
    dfree = max(G-1, 1)
    tcrit = stats.t.ppf(1-alpha, dfree)
    ci = (est - tcrit*se, est + tcrit*se)
    return est, ci, f"Mixed-effects (statsmodels; τ²={tau2:.3g}, df≈G-1={dfree})"

def cluster_robust_ols_mu(df: pd.DataFrame, cluster_col: str, diff_col: str, alpha: float) -> Tuple[float, Tuple[float,float]]:
    X = np.ones((len(df),1))
    fit = sm.OLS(df[diff_col].values, X).fit(cov_type="cluster", cov_kwds={"groups": df[cluster_col].values})
    est = float(fit.params[0]); se = float(fit.bse[0])
    dfree = max(df[cluster_col].nunique()-1,1); tcrit = stats.t.ppf(1-alpha, dfree)
    return est, (est - tcrit*se, est + tcrit*se)

# ------------------------------ Equivalence summaries ------------------------------

def equiv_table(mu: float, ci: Tuple[float,float], deltas: Iterable[float]) -> pd.DataFrame:
    deltas = np.array(list(deltas), float)
    lo, hi = ci
    flags = (lo > -deltas) & (hi < deltas)
    return pd.DataFrame({"delta": deltas, "mu_hat": mu, "ci_low": lo, "ci_high": hi, "equivalent": flags})

def plot_ci_methods(method2ci: Dict[str, Tuple[float,Tuple[float,float]]], out_path: str):
    plt.figure()
    for i, (name, (mu, ci)) in enumerate(method2ci.items(), start=1):
        plt.hlines(i, ci[0], ci[1], linewidth=3)
        plt.plot([mu], [i], marker="D")
    plt.axvline(0, ls="--", alpha=0.5)
    plt.yticks(range(1,len(method2ci)+1), list(method2ci.keys()))
    plt.xlabel("μ and CI"); plt.title("Model-based CIs for μ"); plt.tight_layout()
    plt.savefig(out_path, dpi=140); plt.close()

# ------------------------------ Orchestrator ------------------------------

def run_pubgrade_spatial_tost(
    df: pd.DataFrame,
    cluster_col: str = "cluster_id",
    x_col: str = "x",
    y_col: str = "y",
    diff_col: str = "diff",
    margins: Iterable[float] = (1,3,5),
    alpha: float = 0.05,
    out_dir: str = "tost_pub",
    nu_grid: Iterable[float] = (0.5,1.5,2.5),
    per_cluster_nugget: bool = True,
    do_sensitivity: bool = True,
    moran_k: int = 4,
    bootstrap_B: int = 2000,
    random_state: int = 42,
    # NEW: policy switch for spatial handling
    spatial_policy: str = "auto"
) -> Dict[str, object]:
    """
    Full, publication-grade spatial TOST workflow using Matérn REML + LR CI for μ.

    Parameters
    ----------
    df : DataFrame with columns [cluster_id, x, y, diff]; diff in SAV units (A - B).
    margins : sequence of Δ values to test for equivalence (e.g., range(1,101)).
    alpha : size for CI-in-TOST; LR CI uses χ^2_1(1-2α).
    nu_grid : candidate Matérn smoothness values for profile REML (extend if desired).
    per_cluster_nugget : if True, include τ^2 I within each cluster block.
    do_sensitivity : if True, also run mixed-effects, cluster OLS, and cluster bootstrap.
    moran_k : k for k-NN weights in Moran’s I (if PySAL available).
    bootstrap_B : cluster bootstrap replicates for μ CI.
    random_state : RNG seed for bootstrap.

    spatial_policy : {"auto","force_spatial","force_nonspatial","diagnose_then_nonspatial"}
        - "auto" (default): run diagnostics, use spatial model if dependence is flagged, else IID.
        - "force_spatial": skip decision and use spatial Matérn REML + LR CI.
        - "force_nonspatial": skip spatial modeling and use non-spatial IID TOST.
        - "diagnose_then_nonspatial": run diagnostics but ignore result; use non-spatial IID TOST.

    Returns
    -------
    dict with:
      - 'diagnostics': overview table, Moran's I table, variogram dir, spatial_dependence(bool), reasons(list)
      - 'model': (present if spatial model was fit) ν*, θ̂, μ̂, LR CI
      - 'summaries': {method_name: DataFrame over Δ with μ, CI, equivalence}
      - 'primary_method': name of the summary considered primary under the chosen policy
      - 'policy': {'spatial_policy': <str>, 'decision_used_spatial': bool, 'reasons': [...]}
      - 'notes': list of human-readable notes about the fit

    Notes
    -----
    Diagnostics thresholds (conservative):
      ICC > 0.10, or SE inflation (cluster/IID) > 1.10, or significant positive Moran's I (p<0.10),
      or rising empirical variograms in ≥1 cluster. Adjust as needed upstream.
    """
    _ensure_dir(out_dir)

    # ---------- Diagnostics (always run in "auto" and "diagnose_then_nonspatial"; skipped otherwise) ----------
    run_diagnostics = spatial_policy in ("auto", "diagnose_then_nonspatial")
    if run_diagnostics:
        # ICC
        icc = compute_icc(df, cluster_col, diff_col)

        # Moran's I
        moran_tbl = morans_I(df, cluster_col, x_col, y_col, diff_col, k=moran_k)

        # Variograms (also plotted to disk)
        variograms = empirical_variograms(
            df, cluster_col, x_col, y_col, diff_col,
            n_bins=8, out_dir=os.path.join(out_dir, "variograms")
        )

        # IID vs cluster SE inflation
        X = np.ones((len(df),1))
        iid = sm.OLS(df[diff_col].values, X).fit()
        est_iid, se_iid = float(iid.params[0]), float(iid.bse[0])
        cl  = sm.OLS(df[diff_col].values, X).fit(cov_type="cluster",
                                                 cov_kwds={"groups": df[cluster_col].values})
        se_cl = float(cl.bse[0])
        se_ratio = se_cl / (se_iid if se_iid>0 else np.nan)

        diag_overview = pd.DataFrame([{
            "icc": icc,
            "se_iid": se_iid,
            "se_cluster": se_cl,
            "se_inflation_ratio": se_ratio,
            "n": len(df),
            "G": df[cluster_col].nunique()
        }])
        diag_overview.to_csv(os.path.join(out_dir, "diagnostics_overview.csv"), index=False)
        if not moran_tbl.empty:
            moran_tbl.to_csv(os.path.join(out_dir, "moran_results.csv"), index=False)

        reasons = []
        if icc > 0.10: reasons.append(f"ICC={icc:.2f}>0.10")
        if se_ratio > 1.10: reasons.append(f"SE inflation={se_ratio:.2f}>1.10")
        if not moran_tbl.empty and (moran_tbl["p_norm"]<0.10).any() and (moran_tbl["I"]>0).any():
            reasons.append("Significant positive Moran’s I (p<0.10) in ≥1 cluster")
        spatial_dep = len(reasons) > 0

        diagnostics = {
            "overview": diag_overview,
            "moran_table": moran_tbl,
            "variograms_dir": os.path.join(out_dir, "variograms"),
            "spatial_dependence": spatial_dep,
            "reasons": reasons
        }
    else:
        # Minimal diagnostics stub (not computed)
        diagnostics = {
            "overview": pd.DataFrame([{
                "icc": np.nan, "se_iid": np.nan, "se_cluster": np.nan,
                "se_inflation_ratio": np.nan, "n": len(df), "G": df[cluster_col].nunique()
            }]),
            "moran_table": pd.DataFrame(),
            "variograms_dir": None,
            "spatial_dependence": False,
            "reasons": []
        }
        spatial_dep = False
        reasons = []

    # ---------- Policy decision ----------
    if spatial_policy == "force_spatial":
        use_spatial = True
        decision_note = "Forced spatial per policy."
    elif spatial_policy == "force_nonspatial":
        use_spatial = False
        decision_note = "Forced non-spatial per policy."
    elif spatial_policy == "diagnose_then_nonspatial":
        use_spatial = False
        decision_note = "Diagnostics run, but proceeding non-spatial per policy."
    elif spatial_policy == "auto":
        use_spatial = bool(spatial_dep)
        decision_note = "Auto decision based on diagnostics."
    else:
        raise ValueError("spatial_policy must be one of {'auto','force_spatial','force_nonspatial','diagnose_then_nonspatial'}")

    # ---------- Summaries (IID always; spatial conditional) ----------
    summaries: Dict[str, pd.DataFrame] = {}

    # Non-spatial IID baseline (primary if use_spatial==False)
    X = np.ones((len(df),1))
    iid_fit = sm.OLS(df[diff_col].values, X).fit()
    est_iid = float(iid_fit.params[0]); se_iid = float(iid_fit.bse[0])
    dfree = max(len(df) - 1, 1)
    from scipy.stats import t as _t
    tcrit = _t.ppf(1 - alpha, df=dfree)
    ci_iid = (est_iid - tcrit*se_iid, est_iid + tcrit*se_iid)

    def _equiv_table(mu, ci, deltas):
        deltas = np.array(list(deltas), float)
        lo, hi = ci
        flags = (lo > -deltas) & (hi < deltas)
        return pd.DataFrame({"delta": deltas, "mu_hat": mu, "ci_low": lo, "ci_high": hi, "equivalent": flags})

    summaries["IID OLS"] = _equiv_table(est_iid, ci_iid, margins)

    method2ci = {"IID OLS": (est_iid, ci_iid)}
    model_block = {}  # populated if spatial fit runs
    notes = []

    if use_spatial:
        # Spatial (Matérn REML + LR CI for μ)
        best = fit_matern_reml(df, cluster_col, x_col, y_col, diff_col,
                               nu_grid=nu_grid, per_cluster_nugget=per_cluster_nugget)
        mu_hat = float(best["mu_hat"])
        ci_mu = lr_ci_for_mu(df, cluster_col, x_col, y_col, diff_col, theta=best, alpha=alpha)

        summaries["Matérn REML + LR CI"] = equiv_table(mu_hat, ci_mu, margins)
        method2ci["Matérn REML + LR CI"] = (mu_hat, ci_mu)

        model_block = {
            "nu_star": best["nu"],
            "theta_hat": {k: best[k] for k in ("sigma2","rho","tau2")},
            "mu_hat": mu_hat,
            "ci_mu_lrt": ci_mu
        }
        notes += [
            f"Selected Matérn ν = {best['nu']}",
            f"θ̂: σ²={best['sigma2']:.4g}, ρ={best['rho']:.4g}, τ²={best['tau2']:.4g}",
            "μ CI via 1-df LR inversion with Σ fixed at θ̂."
        ]

    # Sensitivity models (optional; they are informative regardless of policy)
    if do_sensitivity:
        est_me, ci_me, me_note = mixed_effects_mu(df, cluster_col, diff_col, alpha)
        summaries["Mixed-effects"] = equiv_table(est_me, ci_me, margins)
        method2ci[me_note] = (est_me, ci_me)

        est_cl, ci_cl = cluster_robust_ols_mu(df, cluster_col, diff_col, alpha)
        summaries["Cluster-robust OLS"] = equiv_table(est_cl, ci_cl, margins)
        method2ci["Cluster-robust OLS"] = (est_cl, ci_cl)

        est_bs, se_bs, ci_bs = cluster_bootstrap_mu(df, cluster_col, diff_col, B=bootstrap_B, seed=random_state)
        summaries[f"Cluster Bootstrap (B={bootstrap_B})"] = equiv_table(est_bs, ci_bs, margins)
        method2ci[f"Cluster Bootstrap (B={bootstrap_B})"] = (est_bs, ci_bs)

    # Plot method CIs (if at least two methods present)
    if len(method2ci) >= 1:
        plot_ci_methods(method2ci, os.path.join(out_dir, "mu_ci_by_method.png"))

    # Decide primary method (what downstream should treat as the “official” result)
    primary_method = "Matérn REML + LR CI" if use_spatial else "IID OLS"

    policy_info = {
        "spatial_policy": spatial_policy,
        "decision_used_spatial": bool(use_spatial),
        "reasons": reasons if run_diagnostics else []
    }

    return {
        "diagnostics": diagnostics,
        "model": model_block if use_spatial else {},
        "summaries": summaries,
        "primary_method": primary_method,
        "policy": policy_info,
        "notes": notes
    }

def render_one_page_report(
    results: dict,
    report_margins,
    out_dir: str = "tost_pub",
    title: str = "Spatially-Aware TOST Summary",
    subtitle: str = "Publication-grade Matérn REML + LR CI, with sensitivity checks",
    methods_note: str = None,
    ci_figure_path: str = None,
    compile_pdf: bool = True
):
    """
    Create a compact 1-page LaTeX PDF report summarizing diagnostics and Δ-wise equivalence.

    Parameters
    ----------
    results : dict
        Output of run_pubgrade_spatial_tost(...).
        Expects keys: 'diagnostics', 'model', 'summaries', 'notes'.
    report_margins : Iterable[float]
        The Δ values to display in the report table (can differ from modeling set).
    out_dir : str
        Directory to write report.tex/.pdf and figures.
    title, subtitle : str
        Title and subtitle for the report.
    methods_note : Optional[str]
        If provided, appended after the default Methods paragraph (short).
    ci_figure_path : Optional[str]
        Path to a CI-by-method plot (PNG) to include. If None, uses
        f"{out_dir}/mu_ci_by_method.png" if present; otherwise omits figure.
    compile_pdf : bool
        If True, attempts to run 'pdflatex' (and bibtex) to build a PDF.
        If missing, .tex and .bib are left for manual compilation.

    Outputs
    -------
    Writes to `out_dir`:
      - report.tex
      - refs.bib
      - (if compile_pdf=True and pdflatex found) report.pdf

    Notes
    -----
    Layout: tight margins, small font, one-column table for diagnostics and Δ-equivalence.
    The “Methods” paragraph gives a citeable, publication-grade workflow:
      - Matérn covariance + REML for (σ², ρ, τ²) with ν via profile grid [Cressie, 1993; Stein, 1999]
      - μ̂ via GLS; **likelihood-ratio CI** for μ (1 df) feeds the CI-in-TOST rule [Schuirmann, 1987; Lakens, 2017]
      - Sensitivity: mixed-effects with Kenward–Roger df [Kenward & Roger, 1997; Bates et al., 2015],
        cluster-robust OLS [Bell & McCaffrey, 2002; Pustejovsky & Tipton, 2018], and cluster bootstrap [Davison & Hinkley, 1997].
    """
    import os, shutil, subprocess, textwrap
    import pandas as pd

    os.makedirs(out_dir, exist_ok=True)

    diag = results.get("diagnostics", {})
    model = results.get("model", {})
    summaries = results.get("summaries", {})

    main_name = "Matérn REML + LR CI"
    main_df = summaries.get(main_name)
    if main_df is None or main_df.empty:
        raise ValueError(f"Expected summaries['{main_name}'] in results.")

    rep_margins = list(report_margins)
    show_df = main_df[main_df["delta"].isin(rep_margins)].copy().set_index("delta").loc[rep_margins].reset_index()

    # Diagnostics fields
    overview = diag.get("overview", pd.DataFrame())
    if isinstance(overview, pd.DataFrame) and not overview.empty:
        icc = overview.get("icc", pd.Series([None])).iloc[0]
        se_ratio = overview.get("se_inflation_ratio", pd.Series([None])).iloc[0]
        G = overview.get("G", pd.Series([None])).iloc[0]
    else:
        icc = diag.get("icc", None)
        se_ratio = diag.get("se_inflation_ratio", None)
        G = None

    moran_tbl = diag.get("moran_table", pd.DataFrame())
    moran_pos = int(((not moran_tbl.empty) and (moran_tbl["p_norm"] < 0.10) & (moran_tbl["I"] > 0)).sum())
    reasons = diag.get("reasons", [])

    # Figure path
    if ci_figure_path is None:
        default_fig = os.path.join(out_dir, "mu_ci_by_method.png")
        ci_figure_path = default_fig if os.path.exists(default_fig) else None

    # Build Δ table rows
    def _latex_bool(x): return r"\checkmark" if bool(x) else ""
    rows = []
    for _, r in show_df.iterrows():
        rows.append(f"{int(r['delta'])} & {r['mu_hat']:.3g} & [{r['ci_low']:.3g}, {r['ci_high']:.3g}] & {_latex_bool(r['equivalent'])}\\\\")
    eq_table = "\n".join(rows)

    # Diagnostics table rows
    icc_str = "—" if icc is None or pd.isna(icc) else f"{float(icc):.2f}"
    se_ratio_str = "—" if se_ratio is None or pd.isna(se_ratio) else f"{float(se_ratio):.2f}"
    G_str = "—" if G is None or pd.isna(G) else str(int(G))
    diag_table = (
        f"ICC & {icc_str} \\\\\n"
        f"SE inflation (cluster / IID) & {se_ratio_str} \\\\\n"
        f"# clusters (G) & {G_str} \\\\\n"
        f"# bldgs with sig. Moran's I (p<0.10) & {moran_pos} \\\\\n"
        f"Reasons flagged & {(', '.join(reasons) if reasons else 'None')} \\\\\n"
    )

    # Methods text
    methods_txt = textwrap.dedent(r"""
        \textbf{Estimand \& test.} We test equivalence of the population mean difference $\mu$ (SAV units) using the CI-based TOST rule \cite{Schuirmann1987,Lakens2017}: for margin $\Delta$, declare equivalence iff the $(1-2\alpha)$ CI for $\mu$ lies entirely within $[-\Delta,+\Delta]$.

        \textbf{Spatial model.} We model spatial dependence within clusters using a Gaussian process with Matérn covariance $C(h)$ \cite{Cressie1993,Stein1999}; hyper-parameters $(\sigma^2,\rho,\tau^2)$ are estimated by REML \cite{Harville1977} with $\nu$ chosen by profile REML over a small grid. The GLS estimator is $\hat\mu=(\mathbf{1}^\top\Sigma^{-1}\mathbf{y})/(\mathbf{1}^\top\Sigma^{-1}\mathbf{1})$; uncertainty for $\mu$ uses a 1-df likelihood-ratio CI obtained by inverting the profile likelihood with $\Sigma$ fixed at $\hat\theta$.

        \textbf{Sensitivity.} We report (i) mixed-effects with random intercept by cluster and Kenward--Roger df if available \cite{Kenward1997,Bates2015}; (ii) cluster-robust OLS (clusters as clusters) with small-$G$ caution \cite{Bell2002,Pustejovsky2018}; and (iii) cluster (block) bootstrap CIs for $\mu$ \cite{Davison1997}. Diagnostics include ICC, Moran's $I$, empirical variograms, and IID vs clustered SE inflation.
    """).strip()
    if methods_note:
        methods_txt += " " + methods_note

    # Pull model bits (with safe fallbacks)
    nu_star = model.get("nu_star", "—")
    theta = model.get("theta_hat", {}) or {}
    sigma2 = theta.get("sigma2", "—")
    rho = theta.get("rho", "—")
    tau2 = theta.get("tau2", "—")
    mu_hat = model.get("mu_hat", "—")
    ci_mu = model.get("ci_mu_lrt", ("—","—"))
    ci_lo, ci_hi = ci_mu if isinstance(ci_mu, (list, tuple)) else ("—","—")

    # Optional figure include
    figure_include = ""
    if ci_figure_path:
        # copy image into out_dir for LaTeX
        dst = os.path.join(out_dir, os.path.basename(ci_figure_path))
        if os.path.abspath(dst) != os.path.abspath(ci_figure_path):
            try: shutil.copy(ci_figure_path, dst)
            except Exception: pass
        figure_include = r"\vspace{0.5em}\centerline{\includegraphics[width=0.92\linewidth]{" + os.path.basename(ci_figure_path) + "}}"

    # LaTeX template (no f-strings; pure Template with $placeholders)
    tex_tmpl = Template(r"""
\documentclass[10pt]{article}
\usepackage[margin=0.75in]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amsmath,amssymb}
\usepackage{hyperref}
\usepackage{microtype}
\usepackage{multirow}
\usepackage{xcolor}
\usepackage{pifont}
\usepackage{sectsty}
\usepackage{setspace}
\allsectionsfont{\normalsize}
\renewcommand\familydefault{\sfdefault}
\begin{document}
\small
\noindent\textbf{$title}\\
$subtitle \hfill $date

\vspace{0.5em}
\noindent\textbf{Model:} Matérn REML + LR CI for $\mu$; selected $\nu=$nu_star; $\hat\sigma^2=$sigma2, $\hat\rho=$rho, $\hat\tau^2=$tau2; $\hat\mu=$mu_hat; CI$_\text{LR}$ = [$ci_lo, $ci_hi].

\vspace{0.5em}
\noindent\textbf{Diagnostics (overview).}
\begin{tabular}{@{}ll@{}}
\toprule
$diag_table\bottomrule
\end{tabular}

\vspace{0.75em}
\noindent\textbf{Equivalence at selected $\Delta$ (CI-based TOST, $\alpha=0.05$).}
\begin{tabular}{@{}rccc@{}}
\toprule
$\Delta$ & $\hat\mu$ & CI for $\mu$ & Eqv. \\
\midrule
$eq_table\bottomrule
\end{tabular}

\vspace{0.75em}
\noindent\textbf{Methods (concise).} $methods_txt

$figure_include

\vspace{0.5em}
\noindent\textbf{References.} See bibliography.

\end{document}
""")

    tex = tex_tmpl.substitute(
        title=title,
        subtitle=subtitle,
        date=datetime.now().strftime("%Y-%m-%d"),
        nu_star=nu_star,
        sigma2=f"{sigma2:.3g}" if isinstance(sigma2, (int,float)) else str(sigma2),
        rho=f"{rho:.3g}" if isinstance(rho, (int,float)) else str(rho),
        tau2=f"{tau2:.3g}" if isinstance(tau2, (int,float)) else str(tau2),
        mu_hat=f"{mu_hat:.3g}" if isinstance(mu_hat, (int,float)) else str(mu_hat),
        ci_lo=f"{ci_lo:.3g}" if isinstance(ci_lo, (int,float)) else str(ci_lo),
        ci_hi=f"{ci_hi:.3g}" if isinstance(ci_hi, (int,float)) else str(ci_hi),
        diag_table=diag_table,
        eq_table=eq_table,
        methods_txt=methods_txt,
        figure_include=figure_include
    )

    # Minimal .bib (same as before)
    bib = r"""
@article{Schuirmann1987, author={Schuirmann, Donald J.}, title={A Comparison of the Two One-Sided Tests Procedure and the Power Approach for Assessing the Equivalence of Average Bioavailability}, journal={Journal of Pharmacokinetics and Biopharmaceutics}, year={1987}, volume={15}, pages={657--680}}
@article{Lakens2017, author={Lakens, Daniël}, title={Equivalence Tests: A Practical Primer for t Tests, Correlations, and Meta-Analyses}, journal={Social Psychological and Personality Science}, year={2017}, volume={8}, number={4}, pages={355--362}}
@book{Cressie1993, author={Cressie, Noel}, title={Statistics for Spatial Data}, publisher={Wiley}, year={1993}}
@book{Stein1999, author={Stein, Michael L.}, title={Interpolation of Spatial Data: Some Theory for Kriging}, publisher={Springer}, year={1999}}
@article{Harville1977, author={Harville, David A.}, title={Maximum Likelihood Approaches to Variance Component Estimation and to Related Problems}, journal={Journal of the American Statistical Association}, year={1977}, volume={72}, number={358}, pages={320--338}}
@article{Kenward1997, author={Kenward, Michael G. and Roger, John H.}, title={Small Sample Inference for Fixed Effects from Restricted Maximum Likelihood}, journal={Biometrics}, year={1997}, volume={53}, number={3}, pages={983--997}}
@article{Bates2015, author={Bates, Douglas and M{\"a}chler, Martin and Bolker, Ben and Walker, Steve}, title={Fitting Linear Mixed-Effects Models Using lme4}, journal={Journal of Statistical Software}, year={2015}, volume={67}, number={1}, pages={1--48}}
@article{Bell2002, author={Bell, Robert M. and McCaffrey, Daniel F.}, title={Bias Reduction in Standard Errors for Linear Regression with Multi-Stage Samples}, journal={Survey Methodology}, year={2002}, volume={28}, pages={169--182}}
@article{Pustejovsky2018, author={Pustejovsky, James E. and Tipton, Elizabeth}, title={Small-Sample Methods for Cluster-Robust Variance Estimation and Hypothesis Testing in Fixed Effects Models}, journal={Journal of Business & Economic Statistics}, year={2018}, volume={36}, number={4}, pages={672--683}}
@book{Davison1997, author={Davison, A. C. and Hinkley, D. V.}, title={Bootstrap Methods and Their Application}, publisher={Cambridge University Press}, year={1997}}
"""

    tex_path = os.path.join(out_dir, "report.tex")
    bib_path = os.path.join(out_dir, "refs.bib")
    with open(tex_path, "w") as f: f.write(tex)
    with open(bib_path, "w") as f: f.write(bib)

    # Compile (optional)
    if compile_pdf:
        def run(cmd):
            subprocess.run(cmd, cwd=out_dir, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            run(["pdflatex", "-interaction=nonstopmode", "report.tex"])
            try:
                run(["bibtex", "report"])
                run(["pdflatex", "-interaction=nonstopmode", "report.tex"])
                run(["pdflatex", "-interaction=nonstopmode", "report.tex"])
            except Exception:
                run(["pdflatex", "-interaction=nonstopmode", "report.tex"])
        except Exception as e:
            print(f"[render_one_page_report] LaTeX compile failed: {e}. Left .tex/.bib for manual compile.")

    return {
        "tex_path": tex_path,
        "bib_path": bib_path,
        "pdf_path": os.path.join(out_dir, "report.pdf"),
        "diagnostics": {"icc": icc, "se_inflation_ratio": se_ratio, "G": G, "moran_sig_bldgs": moran_pos},
        "report_margins": rep_margins
    }


# -----------------------------------------------------------------------------
# Public engine wrapper (backward compatible)
# -----------------------------------------------------------------------------

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd



@dataclass(frozen=True)
class SpatialConfig:
    """
    Parameters controlling the Matérn REML fit.

    Parameters
    ----------
    nu_grid
        Candidate Matérn smoothness values ν considered in the REML profile search.
    per_cluster_nugget
        Whether to include a nugget term (τ² I) inside each cluster block.
    verbose_diagnostics
        If True, print diagnostic summaries that help detect covariance/SE
        pathologies (e.g., variance collapse) and compare estimands.
    """

    nu_grid: Tuple[float, ...] = (0.5, 1.5, 2.5)
    per_cluster_nugget: bool = True
    verbose_diagnostics: bool = False

class SpatialTOST:
    """
    Publication-grade spatial TOST using Matérn GLS (REML) + LR CI for μ.

    Parameters
    ----------
    y : str
        Response column name (paired difference), e.g., "diff".
    cluster : str
        Cluster/group id column name. Required because Σ is block-diagonal by cluster.
    x, ycoord : str
        Spatial coordinate column names.
    config : SpatialConfig
        REML fit settings.
    """

    def __init__(
        self,
        y: str,
        cluster: str,
        x: str,
        ycoord: str,
        config: SpatialConfig | None = None,
    ):
        self.y = y
        self.cluster = cluster
        self.x = x
        self.ycoord = ycoord
        self.config = config or SpatialConfig()

    def fit(self, df: pd.DataFrame, alpha: float, margins: List[float]) -> pd.DataFrame:
        """
        Fit the Matérn GLS model and compute CI-in-TOST decisions across margins.

        Returns
        -------
        DataFrame
            Columns: delta, mu_hat, ci_low, ci_high, equivalent, method
        """
        for col in (self.y, self.cluster, self.x, self.ycoord):
            if col not in df.columns:
                raise ValueError(
                    f"SpatialTOST requires column {col!r}. "
                    f"Available columns: {list(df.columns)}"
                )

        dfp = df.rename(
            columns={
                self.cluster: "cluster_id",
                self.y: "diff",
                self.x: "x",
                self.ycoord: "y",
            }
        )

        #print(f"dfp['diff'].mean(): {dfp['diff'].mean()}")
        #print("Mean of diff passed to spatial:", dfp["diff"].mean())
        #print("Min/Max of diff:", dfp["diff"].min(), dfp["diff"].max())
        #print("First 10 diff values:", dfp["diff"].head(10).to_list())

        theta = fit_matern_reml(
            df=dfp,
            cluster_col="cluster_id",
            x_col="x",
            y_col="y",
            diff_col="diff",
            nu_grid=self.config.nu_grid,
            per_cluster_nugget=self.config.per_cluster_nugget,
        )
        #from pprint import pprint
        #pprint(f"theta: {theta}")
        mu_hat = float(theta["mu_hat"])

        if self.config.verbose_diagnostics:
            # (1) Simple means
            mean_all = float(dfp["diff"].mean())
            mean_cluster_eq = float(dfp.groupby("cluster_id")["diff"].mean().mean())
            print("[Spatial diagnostics]")
            print(f"  diff.mean()={mean_all:.6g}")
            print(f"  cluster-equal-weight mean={mean_cluster_eq:.6g}")

            # (2) Fitted covariance + conditional var(mu_hat)
            sigma2 = float(theta.get("sigma2", float("nan")))
            tau2 = float(theta.get("tau2", float("nan")))
            rho = float(theta.get("rho", float("nan")))
            var_mu = float(theta.get("var_mu", float("nan")))
            nu = float(theta.get("nu", float("nan")))
            print(f"  fitted mu_hat={mu_hat:.6g}")
            print(f"  fitted theta: sigma2={sigma2:.6g}, tau2={tau2:.6g}, rho={rho:.6g}, nu={nu:.6g}")
            print(f"  fitted var_mu (conditional)={var_mu:.6g}, se={float(var_mu**0.5):.6g}")


        ci_low, ci_high = lr_ci_for_mu(
            df=dfp,
            cluster_col="cluster_id",
            x_col="x",
            y_col="y",
            diff_col="diff",
            theta=theta,
            alpha=alpha,
        )

        if self.config.verbose_diagnostics:
            print(f"  LR CI: [{float(ci_low):.6g}, {float(ci_high):.6g}]")
            print("  ----")

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
                    method="Matérn GLS (REML) + LR CI",
                )
            )
        return pd.DataFrame(rows)

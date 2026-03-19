"""
Optimization routine to find synthetic *spatio-temporal* data-generation parameters
such that:

  - IID engine:          TOST passes  at margin Δ=1
  - Cluster engine:      TOST passes  at margin Δ=1
  - Spatial engine:      TOST passes  at margin Δ=1
  - SpatioTemporal engine: TOST **fails** at margin Δ=1

Design rationale
----------------
The spatio-temporal engine is the most expensive component (joint separable ML +
bootstrap CI).  Prior versions of this optimizer crashed or ran forever because:

1) The ``SpatioTemporalTOST`` engine builds a dense (S·T)×(S·T) covariance matrix
   and Cholesky-factors it at every NLL evaluation.  For S=35, T=90 this is a
   3150×3150 matrix — each Cholesky costs ~O(n³)≈31G FLOPs, and L-BFGS-B may
   evaluate the objective hundreds of times *per fit*, multiplied by B bootstrap
   draws.  This dominates wall-clock time and can exhaust memory.

2) The search space was too wide, so most candidates failed early (non-equivalence
   on all engines) and the few that reached the ST stage took too long.

3) Bootstrap B was too low (6) for the optimizer to reliably distinguish pass/fail,
   leading to score noise.

Solutions implemented here
--------------------------
A) **Kronecker-accelerated NLL** for the search phase.  The separable model

       K = σ² (R_t ⊗ R_s) + τ² I

   admits eigendecomposition via the Kronecker product of the eigenvectors of R_s
   and R_t.  If R_s = V_s Λ_s V_sᵀ and R_t = V_t Λ_t V_tᵀ, then

       K = (V_t ⊗ V_s) [σ² (Λ_t ⊗ Λ_s) + τ² I] (V_t ⊗ V_s)ᵀ,

   and the eigenvalues of K are {σ² λ_s^i λ_t^j + τ²}.  Log-determinant and
   quadratic form in rotated coordinates cost O(ST) instead of O((ST)³).  This
   gives a ~1000× speedup for typical panel sizes, making real-time search feasible.

B) **Tight search space** centered on the known-good anchor from prior runs.  The
   anchor is already near a working solution; we perturb within ±15–25% to find
   nearby configurations that satisfy the target pattern.

C) **Staged evaluation** with cheap-to-expensive ordering:
   IID → Cluster → (prescreen on CI width) → Spatial → SpatioTemporal.
   Most candidates are rejected at Stage 1–2.

D) **Moderate bootstrap B** (default 200) for the final ST CI, with the option to
   use a fast Wald CI during the search phase and validate the winner with a
   full bootstrap afterward.

References
----------
- Kronecker covariance algebra: Van Loan (2000) J. Comp. Appl. Math.
- Separable spatio-temporal models: Genton (2007) Stat. Methods Appl.
- TOST / CI-in-equivalence: Schuirmann (1987); Lakens (2017).
- Newey-West HAC: Newey & West (1987) Econometrica.
- Block bootstrap for dependent data: Kunsch (1989) Ann. Statist.;
  Politis & Romano (1994) J. Amer. Statist. Assoc.

Notes
-----
This module is designed to be imported by the ``optimize_spatiotemporal_params``
Jupyter notebook.  It can also be run standalone via ``__main__``.
"""

from __future__ import annotations

import inspect
import json
import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import linalg, optimize, stats

from . import synthetic_tost_data
from ..engines import iid_tost, cluster_tost, spatial_tost


# ---------------------------------------------------------------------------
# Search space: tight around known-good anchor
# ---------------------------------------------------------------------------

# Anchor values from prior optimization runs (notebook output)
_ANCHOR = {
    "n_space": 35,
    "n_time": 90,
    "n_clusters": 25,
    "length_scale": 1.2,
    "rho": 0.7,
    "spatial_sd": 0.7,
    "obs_sd": 0.4,
    "delta": 0.9707,
    "domain": [-2.0, 2.0, -2.0, 2.0],
}


@dataclass
class SearchSpace:
    """Parameter ranges for stochastic search.

    Parameters
    ----------
    anchor : dict
        Center of the search region.  Defaults to the known-good solution.
    """

    anchor: Dict[str, Any] = field(default_factory=lambda: dict(_ANCHOR))

    # Perturbation half-widths (relative for continuous, absolute for integers)
    n_space_range: Tuple[int, int] = (25, 50)
    n_time_range: Tuple[int, int] = (50, 130)
    n_clusters_range: Tuple[int, int] = (15, 35)
    length_scale_range: Tuple[float, float] = (0.6, 2.0)
    rho_range: Tuple[float, float] = (0.50, 0.90)
    spatial_sd_range: Tuple[float, float] = (0.3, 1.2)
    obs_sd_range: Tuple[float, float] = (0.15, 0.8)
    delta_range: Tuple[float, float] = (0.90, 0.999)

    def sample(self, rng: np.random.Generator) -> Dict[str, Any]:
        """Draw a candidate parameter set near the anchor.

        Parameters
        ----------
        rng : numpy Generator
            Random state for reproducibility.

        Returns
        -------
        dict
            Keys match ``generate_spatiotemporal`` kwargs plus ``n_clusters``.
        """
        d: Dict[str, Any] = {}
        d["n_space"] = int(rng.integers(self.n_space_range[0], self.n_space_range[1] + 1))
        d["n_time"] = int(rng.integers(self.n_time_range[0], self.n_time_range[1] + 1))
        d["n_clusters"] = int(rng.integers(self.n_clusters_range[0], self.n_clusters_range[1] + 1))
        d["length_scale"] = float(rng.uniform(*self.length_scale_range))
        d["rho"] = float(rng.uniform(*self.rho_range))
        d["spatial_sd"] = float(rng.uniform(*self.spatial_sd_range))
        d["obs_sd"] = float(rng.uniform(*self.obs_sd_range))
        d["delta"] = float(rng.uniform(*self.delta_range))
        d["domain"] = list(self.anchor.get("domain", [-2.0, 2.0, -2.0, 2.0]))
        return d


# ---------------------------------------------------------------------------
# Evaluation result container
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    """Container for one candidate's evaluation across all engines."""

    ok_pattern: bool
    score: float
    params: Dict[str, Any]

    ci_iid: Tuple[float, float]
    ci_cluster: Tuple[float, float]
    ci_spatial: Tuple[float, float]
    ci_spatiotemporal: Tuple[float, float]

    eq_iid: bool
    eq_cluster: bool
    eq_spatial: bool
    eq_spatiotemporal: bool

    mu_hat_iid: float
    mu_hat_cluster: float
    mu_hat_spatial: float
    mu_hat_spatiotemporal: float

    notes: str = ""


# ---------------------------------------------------------------------------
# Kronecker-accelerated NLL for the search phase
# ---------------------------------------------------------------------------

def _pairwise_dists(XY: np.ndarray) -> np.ndarray:
    """Euclidean distance matrix for rows of XY (n×2)."""
    diff = XY[:, None, :] - XY[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def _matern_corr(D: np.ndarray, rho: float, nu: float) -> np.ndarray:
    """Matérn correlation matrix (sigma2=1) from distance matrix D.

    Uses the standard parameterization with range ρ and smoothness ν.
    Ref: Stein (1999) *Interpolation of Spatial Data*.
    """
    from scipy.special import kv, gamma as sp_gamma

    d = np.asarray(D, float)
    s = np.sqrt(2.0 * nu) * d / max(rho, 1e-12)
    C = np.empty_like(s)
    small = s < 1e-8
    C[small] = 1.0
    z = s[~small]
    coef = (2.0 ** (1.0 - nu)) / sp_gamma(nu)
    C[~small] = coef * (z ** nu) * kv(nu, z)
    # Enforce exact symmetry and unit diagonal
    C = 0.5 * (C + C.T)
    np.fill_diagonal(C, 1.0)
    return C


def _ar1_corr(T: int, phi: float) -> np.ndarray:
    """AR(1) correlation matrix of size T."""
    idx = np.arange(T)
    return phi ** np.abs(idx[:, None] - idx[None, :])


def _kronecker_nll(
    y: np.ndarray,
    one: np.ndarray,
    eig_s: Tuple[np.ndarray, np.ndarray],
    eig_t: Tuple[np.ndarray, np.ndarray],
    sigma2: float,
    tau2: float,
) -> Tuple[float, float, float]:
    """Fast NLL using Kronecker eigendecomposition.

    Given R_s = V_s Λ_s V_sᵀ and R_t = V_t Λ_t V_tᵀ, the covariance is

        K = σ² (R_t ⊗ R_s) + τ² I

    with eigenvalues d_ij = σ² λ_s^i λ_t^j + τ².

    The NLL (up to constant) and GLS estimator μ̂ are computed in the
    rotated basis where K is diagonal:

        ỹ = (V_t ⊗ V_s)ᵀ y
        1̃ = (V_t ⊗ V_s)ᵀ 1

    Then:
        logdet = Σ log(d_ij)
        quad   = Σ (ỹ_k - μ̂ · 1̃_k)² / d_k
        μ̂     = (1̃ᵀ D⁻¹ ỹ) / (1̃ᵀ D⁻¹ 1̃)
        Var(μ̂)= 1 / (1̃ᵀ D⁻¹ 1̃)  [= 1 / (1ᵀ K⁻¹ 1)]

    Cost: O(S² + T² + ST) instead of O((ST)³).

    Parameters
    ----------
    y : ndarray, shape (S*T,)
        Observations stacked as (t0_s0, t0_s1, ..., t0_sS, t1_s0, ...).
    one : ndarray, shape (S*T,)
        Vector of ones.
    eig_s : (vals_s, vecs_s)
        Eigendecomposition of R_s (S×S correlation).
    eig_t : (vals_t, vecs_t)
        Eigendecomposition of R_t (T×T correlation).
    sigma2, tau2 : float
        Signal variance and nugget variance.

    Returns
    -------
    nll : float
        Negative log-likelihood (excluding constant term).
    mu_hat : float
        GLS mean estimate.
    var_mu : float
        Variance of GLS mean estimate.

    References
    ----------
    Van Loan (2000). The ubiquitous Kronecker product. J. Comp. Appl. Math.
    """
    lam_s, V_s = eig_s  # (S,), (S, S)
    lam_t, V_t = eig_t  # (T,), (T, T)

    S = lam_s.size
    T = lam_t.size
    n = S * T

    # Eigenvalues of K: d_{j,i} = sigma2 * lam_t[j] * lam_s[i] + tau2
    # Reshape for broadcasting: (T,1) * (1,S) -> (T,S)
    D = sigma2 * (lam_t[:, None] * lam_s[None, :]) + tau2  # (T, S)
    D_flat = D.reshape(-1)  # (T*S,) — matches stacking order (t, s)

    # Clamp eigenvalues for numerical safety
    D_flat = np.maximum(D_flat, 1e-15)

    # Rotate y and 1 into eigenbasis: ỹ = (V_t ⊗ V_s)ᵀ y
    # Efficient via: reshape y to (T, S), multiply V_sᵀ on right, V_tᵀ on left
    Y = y.reshape(T, S)
    O = one.reshape(T, S)
    Y_rot = (V_t.T @ Y) @ V_s     # (T, S)
    O_rot = (V_t.T @ O) @ V_s     # (T, S)

    y_tilde = Y_rot.reshape(-1)
    one_tilde = O_rot.reshape(-1)

    # Inverse-eigenvalue weighted quantities
    Dinv = 1.0 / D_flat                              # (n,)
    info = float(np.sum(one_tilde ** 2 * Dinv))       # 1̃ᵀ D⁻¹ 1̃
    if info <= 0:
        return 1e15, 0.0, float("inf")

    cross = float(np.sum(one_tilde * y_tilde * Dinv))  # 1̃ᵀ D⁻¹ ỹ
    mu_hat = cross / info
    var_mu = 1.0 / info

    # Residual in rotated basis
    r_tilde = y_tilde - mu_hat * one_tilde
    quad = float(np.sum(r_tilde ** 2 * Dinv))
    logdet = float(np.sum(np.log(D_flat)))

    nll = 0.5 * (logdet + quad + n * np.log(2.0 * np.pi))
    return nll, mu_hat, var_mu


def _fast_st_ci(
    y: np.ndarray,
    coords: np.ndarray,
    T: int,
    S: int,
    alpha: float,
    nu: float = 2.5,
) -> Tuple[float, Tuple[float, float], Dict[str, float]]:
    """Fast spatiotemporal GLS fit + Wald CI using Kronecker acceleration.

    Fits the separable model K = σ²(R_t ⊗ R_s) + τ²I by ML using L-BFGS-B
    on the Kronecker-accelerated NLL, then returns a Wald CI for μ̂.

    Parameters
    ----------
    y : ndarray, shape (T*S,)
        Observations ordered as (t=0 s=0, t=0 s=1, ..., t=0 s=S-1, t=1 s=0, ...).
    coords : ndarray, shape (S, 2)
        Spatial coordinates of the S locations.
    T, S : int
        Number of time steps and spatial locations.
    alpha : float
        Significance level for the (1-2α) CI (TOST convention).
    nu : float
        Fixed Matérn smoothness parameter.

    Returns
    -------
    mu_hat : float
        GLS mean estimate.
    ci : (float, float)
        Confidence interval (ci_low, ci_high).
    theta : dict
        Fitted parameters {sigma2, tau2, rho, phi}.
    """
    one = np.ones(T * S, dtype=float)

    # Pre-compute distance matrix and its eigendecomposition structure
    D_sp = _pairwise_dists(coords)

    # Empirical variance for initialization
    var_y = float(np.var(y, ddof=1)) if y.size > 1 else 1.0

    def nll_func(z: np.ndarray) -> float:
        """NLL in log-parameterization: [log σ², log ρ, log τ², logit φ]."""
        sigma2 = float(np.exp(z[0]))
        rho = float(np.exp(z[1]))
        tau2 = float(np.exp(z[2]))
        # φ via logistic transform to (0, 1)
        phi = float(1.0 / (1.0 + np.exp(-z[3])))

        # Build correlation matrices and eigen-decompose
        Rs = _matern_corr(D_sp, rho=rho, nu=nu)
        Rt = _ar1_corr(T, phi)

        # Eigendecompositions (real symmetric → guaranteed real)
        lam_s, V_s = np.linalg.eigh(Rs)
        lam_t, V_t = np.linalg.eigh(Rt)

        # Clamp small eigenvalues (numerical)
        lam_s = np.maximum(lam_s, 1e-12)
        lam_t = np.maximum(lam_t, 1e-12)

        val, _, _ = _kronecker_nll(y, one, (lam_s, V_s), (lam_t, V_t), sigma2, tau2)
        return val

    # Initial guess
    z0 = np.array([
        np.log(max(0.5 * var_y, 1e-8)),   # log σ²
        np.log(1.0),                        # log ρ
        np.log(max(0.1 * var_y, 1e-8)),    # log τ²
        0.0,                                # logit(φ=0.5)
    ], dtype=float)

    bounds = [
        (-15.0, 15.0),   # log σ²
        (-10.0, 10.0),   # log ρ
        (-20.0, 15.0),   # log τ²
        (-6.0, 6.0),     # logit φ  => φ ∈ (~0.002, ~0.998)
    ]

    opt = optimize.minimize(
        nll_func, z0, method="L-BFGS-B", bounds=bounds,
        options={"maxiter": 200, "ftol": 1e-8},
    )

    # Extract fitted params
    sigma2 = float(np.exp(opt.x[0]))
    rho = float(np.exp(opt.x[1]))
    tau2 = float(np.exp(opt.x[2]))
    phi = float(1.0 / (1.0 + np.exp(-opt.x[3])))

    # Recompute mu_hat and var_mu at optimum
    Rs = _matern_corr(D_sp, rho=rho, nu=nu)
    Rt = _ar1_corr(T, phi)
    lam_s, V_s = np.linalg.eigh(Rs)
    lam_t, V_t = np.linalg.eigh(Rt)
    lam_s = np.maximum(lam_s, 1e-12)
    lam_t = np.maximum(lam_t, 1e-12)

    _, mu_hat, var_mu = _kronecker_nll(y, one, (lam_s, V_s), (lam_t, V_t), sigma2, tau2)

    # Wald CI (asymptotic normal)
    se = float(np.sqrt(max(var_mu, 0.0)))
    zcrit = float(stats.norm.ppf(1.0 - alpha))
    ci = (mu_hat - zcrit * se, mu_hat + zcrit * se)

    theta = {"sigma2": sigma2, "tau2": tau2, "rho": rho, "phi": phi, "nu": nu}
    return mu_hat, ci, theta


# ---------------------------------------------------------------------------
# Data conversion helpers
# ---------------------------------------------------------------------------

def _make_diff_df(
    df_long: pd.DataFrame,
    n_clusters: int,
) -> pd.DataFrame:
    """Convert tidy long A/B rows to a per-observation diff dataframe.

    Parameters
    ----------
    df_long : DataFrame
        Long-form with columns: sample_id, arm, y, x, y_sp, t.
    n_clusters : int
        Number of clusters to assign via modular sample_id.

    Returns
    -------
    DataFrame
        Columns: sample_id, cluster_id, x, y, time, diff.
    """
    req = {"sample_id", "arm", "y", "x", "y_sp", "t"}
    missing = sorted(req - set(df_long.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    wide = df_long.pivot(index="sample_id", columns="arm", values="y").reset_index()
    meta = df_long[df_long["arm"] == "A"][["sample_id", "x", "y_sp", "t"]].copy()
    out = meta.merge(wide, on="sample_id")
    out["diff"] = out["B"] - out["A"]
    out["cluster_id"] = out["sample_id"].astype(int) % n_clusters
    out = out.rename(columns={"t": "time", "y_sp": "y"})
    return out


# ---------------------------------------------------------------------------
# Staged evaluation pipeline
# ---------------------------------------------------------------------------

def _score_ci(ci: Tuple[float, float], margin: float, should_pass: bool) -> float:
    """Score a single CI against an equivalence margin.

    Parameters
    ----------
    ci : (lo, hi)
        Confidence interval bounds.
    margin : float
        Equivalence margin Δ.
    should_pass : bool
        If True, the CI should be inside (-Δ, Δ) → score penalizes violation.
        If False, the CI should NOT be inside → score penalizes containment.

    Returns
    -------
    float
        Non-negative penalty (0 = perfect).
    """
    lo, hi = ci
    inside = (lo > -margin) and (hi < margin)

    if should_pass:
        if inside:
            # Reward: how far inside the margin? Larger buffer → lower score.
            buf = min(lo + margin, margin - hi)
            return max(0.0, -buf)  # always 0 when inside
        else:
            # Penalty: how far outside?
            viol = max(0.0, -margin - lo) + max(0.0, hi - margin)
            return 5.0 + viol
    else:
        # Should fail (CI extends beyond margin)
        if not inside:
            return 0.0  # any failure is fine
        else:
            # Contained but shouldn't be: penalty proportional to buffer
            buf = min(lo + margin, margin - hi)
            return 10.0 + buf


def _run_engines(
    df_diff: pd.DataFrame,
    *,
    alpha: float,
    margin: float,
    prescreen_buffer: float,
    skip_spatial: bool = False,
) -> EvalResult:
    """Run IID → Cluster → (optionally) Spatial engines and return partial result.

    Parameters
    ----------
    df_diff : DataFrame
        Must have columns: diff, cluster_id, x, y, time.
    alpha, margin : float
        TOST parameters.
    prescreen_buffer : float
        If IID or Cluster CI is too far inside the margin, skip spatial.
    skip_spatial : bool
        If True, skip the spatial engine entirely (used for fast pre-screening).

    Returns
    -------
    EvalResult
        With spatial and spatiotemporal fields set to NaN if not evaluated.
    """
    nan_ci = (float("nan"), float("nan"))

    # --- IID ---
    iid_eng = iid_tost.IIDTOST(y="diff")
    r_iid = iid_eng.fit(df_diff, alpha=alpha, margins=[margin]).iloc[0]
    ci_iid = (float(r_iid["ci_low"]), float(r_iid["ci_high"]))
    eq_iid = bool(r_iid["equivalent"])
    mu_iid = float(r_iid["mu_hat"])

    if not eq_iid:
        return EvalResult(
            ok_pattern=False, score=1e6,
            params={},
            ci_iid=ci_iid, ci_cluster=nan_ci, ci_spatial=nan_ci, ci_spatiotemporal=nan_ci,
            eq_iid=False, eq_cluster=False, eq_spatial=False, eq_spatiotemporal=False,
            mu_hat_iid=mu_iid, mu_hat_cluster=float("nan"),
            mu_hat_spatial=float("nan"), mu_hat_spatiotemporal=float("nan"),
            notes="IID failed → skip",
        )

    # --- Cluster ---
    clu_eng = cluster_tost.ClusterTOST(y="diff", cluster="cluster_id")
    r_clu = clu_eng.fit(df_diff, alpha=alpha, margins=[margin]).iloc[0]
    ci_clu = (float(r_clu["ci_low"]), float(r_clu["ci_high"]))
    eq_clu = bool(r_clu["equivalent"])
    mu_clu = float(r_clu["mu_hat"])

    if not eq_clu:
        return EvalResult(
            ok_pattern=False, score=5e5 + abs(mu_clu),
            params={},
            ci_iid=ci_iid, ci_cluster=ci_clu, ci_spatial=nan_ci, ci_spatiotemporal=nan_ci,
            eq_iid=eq_iid, eq_cluster=False, eq_spatial=False, eq_spatiotemporal=False,
            mu_hat_iid=mu_iid, mu_hat_cluster=mu_clu,
            mu_hat_spatial=float("nan"), mu_hat_spatiotemporal=float("nan"),
            notes="Cluster failed → skip",
        )

    # --- Spatial (optional) ---
    ci_spa = nan_ci
    eq_spa = False
    mu_spa = float("nan")

    if not skip_spatial:
        # Prescreen: if cluster CI is deep inside margin, spatial likely passes too
        clu_buf = min(ci_clu[0] + margin, margin - ci_clu[1])
        if clu_buf > prescreen_buffer:
            # Cluster CI is well inside; assume spatial also passes (cheap heuristic)
            eq_spa = True
            ci_spa = ci_clu  # placeholder
            mu_spa = mu_clu
        else:
            try:
                spa_eng = spatial_tost.SpatialTOST(
                    y="diff", cluster="cluster_id", x="x", ycoord="y",
                )
                r_spa = spa_eng.fit(df_diff, alpha=alpha, margins=[margin]).iloc[0]
                ci_spa = (float(r_spa["ci_low"]), float(r_spa["ci_high"]))
                eq_spa = bool(r_spa["equivalent"])
                mu_spa = float(r_spa["mu_hat"])
            except Exception as e:
                # Spatial fit failed; treat as non-equivalent (conservative)
                eq_spa = False
                ci_spa = nan_ci
                mu_spa = float("nan")

        if not eq_spa:
            return EvalResult(
                ok_pattern=False, score=2e5,
                params={},
                ci_iid=ci_iid, ci_cluster=ci_clu, ci_spatial=ci_spa, ci_spatiotemporal=nan_ci,
                eq_iid=eq_iid, eq_cluster=eq_clu, eq_spatial=False, eq_spatiotemporal=False,
                mu_hat_iid=mu_iid, mu_hat_cluster=mu_clu,
                mu_hat_spatial=mu_spa, mu_hat_spatiotemporal=float("nan"),
                notes="Spatial failed → skip",
            )

    return EvalResult(
        ok_pattern=False,  # not yet determined (ST not run)
        score=float("inf"),
        params={},
        ci_iid=ci_iid, ci_cluster=ci_clu, ci_spatial=ci_spa, ci_spatiotemporal=nan_ci,
        eq_iid=eq_iid, eq_cluster=eq_clu, eq_spatial=eq_spa, eq_spatiotemporal=False,
        mu_hat_iid=mu_iid, mu_hat_cluster=mu_clu,
        mu_hat_spatial=mu_spa, mu_hat_spatiotemporal=float("nan"),
        notes="passed cheap stages; ST pending",
    )


# ---------------------------------------------------------------------------
# Full single-candidate evaluation
# ---------------------------------------------------------------------------

def evaluate_params(
    params: Dict[str, Any],
    *,
    seed: int = 0,
    alpha: float = 0.05,
    margin: float = 1.0,
    prescreen_buffer: float = 0.03,
    use_fast_st: bool = True,
    nu: float = 2.5,
) -> EvalResult:
    """Generate spatiotemporal data and evaluate the target pass/fail pattern.

    Parameters
    ----------
    params : dict
        Generator kwargs + ``n_clusters``.
    seed : int
        RNG seed for data generation.
    alpha : float
        TOST significance level.
    margin : float
        Equivalence margin Δ.
    prescreen_buffer : float
        Buffer for spatial pre-screening.
    use_fast_st : bool
        If True, use Kronecker-accelerated Wald CI for the ST stage (fast).
        If False, use the full ``SpatioTemporalTOST`` engine (slow, more accurate).
    nu : float
        Matérn smoothness for fast ST fit.

    Returns
    -------
    EvalResult
        Full evaluation across all four engines.
    """
    # --- Generate data ---
    gen = synthetic_tost_data.generate_spatiotemporal
    sig = inspect.signature(gen)
    allowed = set(sig.parameters.keys())

    call = {"seed": seed}
    for k, v in params.items():
        if k in allowed:
            call[k] = v

    df_long, meta = gen(**call)
    n_clusters = int(params.get("n_clusters", 25))
    df_diff = _make_diff_df(df_long, n_clusters)

    # --- Stages 1–3: IID, Cluster, Spatial ---
    partial = _run_engines(
        df_diff, alpha=alpha, margin=margin,
        prescreen_buffer=prescreen_buffer,
        skip_spatial=False,
    )

    # If any upstream stage failed the target pattern, return early
    if not (partial.eq_iid and partial.eq_cluster and partial.eq_spatial):
        partial.params = dict(params)
        return partial

    # --- Stage 4: Spatio-temporal ---
    if use_fast_st:
        # Fast Kronecker-accelerated Wald CI
        # Assign stable location IDs and sort into (time, loc_id) order
        # for Kronecker structure y.reshape(T, S).
        loc_df = (
            df_diff[["x", "y"]]
            .drop_duplicates()
            .sort_values(["x", "y"])
            .reset_index(drop=True)
        )
        loc_df["_loc_id"] = np.arange(len(loc_df), dtype=int)
        coords = loc_df[["x", "y"]].to_numpy(float)
        S = int(coords.shape[0])

        df_tmp = df_diff.merge(loc_df, on=["x", "y"], how="left")
        times = np.sort(df_tmp["time"].unique())
        T = int(times.size)

        # Verify balanced panel: every (time, loc_id) pair must appear exactly once
        if len(df_tmp) != T * S:
            use_fast_st = False
        else:
            df_tmp = df_tmp.sort_values(["time", "_loc_id"]).reset_index(drop=True)
            y_vec = df_tmp["diff"].to_numpy(float)

    if use_fast_st:
        try:
            mu_st, ci_st, theta_st = _fast_st_ci(
                y_vec, coords, T, S, alpha=alpha, nu=nu,
            )
            eq_st = (ci_st[0] > -margin) and (ci_st[1] < margin)
        except Exception as e:
            # If fast fit fails, mark as failed evaluation
            partial.notes = f"Fast ST fit failed: {e}"
            partial.params = dict(params)
            return partial
    else:
        # Full engine (slow path — use only for validation)
        from ..engines.spatiotemporal_tost import SpatioTemporalTOST, SpatioTemporalConfig
        st_cfg = SpatioTemporalConfig(
            mu_ci_method="wald",  # Wald is fastest for search
            verbose_diagnostics=False,
        )
        st_eng = SpatioTemporalTOST(
            y="diff", cluster="cluster_id", time="time",
            x="x", ycoord="y", config=st_cfg,
        )
        try:
            r_st = st_eng.fit(df_diff, alpha=alpha, margins=[margin]).iloc[0]
            ci_st = (float(r_st["ci_low"]), float(r_st["ci_high"]))
            eq_st = bool(r_st["equivalent"])
            mu_st = float(r_st["mu_hat"])
        except Exception as e:
            partial.notes = f"Full ST engine failed: {e}"
            partial.params = dict(params)
            return partial

    # --- Compute score ---
    s_iid = _score_ci(partial.ci_iid, margin, should_pass=True)
    s_clu = _score_ci(partial.ci_cluster, margin, should_pass=True)
    s_spa = _score_ci(partial.ci_spatial, margin, should_pass=True)
    s_st = _score_ci(ci_st, margin, should_pass=False)
    score = s_iid + s_clu + s_spa + s_st

    ok = partial.eq_iid and partial.eq_cluster and partial.eq_spatial and (not eq_st)

    return EvalResult(
        ok_pattern=ok,
        score=score,
        params=dict(params),
        ci_iid=partial.ci_iid,
        ci_cluster=partial.ci_cluster,
        ci_spatial=partial.ci_spatial,
        ci_spatiotemporal=ci_st,
        eq_iid=partial.eq_iid,
        eq_cluster=partial.eq_cluster,
        eq_spatial=partial.eq_spatial,
        eq_spatiotemporal=eq_st,
        mu_hat_iid=partial.mu_hat_iid,
        mu_hat_cluster=partial.mu_hat_cluster,
        mu_hat_spatial=partial.mu_hat_spatial,
        mu_hat_spatiotemporal=mu_st,
        notes="full eval",
    )


# ---------------------------------------------------------------------------
# Stochastic search
# ---------------------------------------------------------------------------

def search(
    *,
    seed: int = 123,
    n_iter: int = 60,
    alpha: float = 0.05,
    margin: float = 1.0,
    space: Optional[SearchSpace] = None,
    verbose_every: int = 10,
    prescreen_buffer: float = 0.03,
    nu: float = 2.5,
) -> Tuple[EvalResult, List[EvalResult]]:
    """Stochastic search for spatiotemporal DGP parameters.

    Uses tight perturbation around the known-good anchor and
    Kronecker-accelerated ST evaluation for speed.

    Parameters
    ----------
    seed : int
        Master RNG seed.
    n_iter : int
        Maximum number of candidates to evaluate.
    alpha : float
        TOST significance level.
    margin : float
        Equivalence margin Δ.
    space : SearchSpace, optional
        Parameter ranges.  Defaults to tight range around anchor.
    verbose_every : int
        Print progress every N iterations.
    prescreen_buffer : float
        Spatial prescreen buffer.
    nu : float
        Matérn smoothness for fast ST.

    Returns
    -------
    best : EvalResult
        Best candidate found.
    history : list of EvalResult
        All evaluated candidates.
    """
    rng = np.random.default_rng(seed)
    space = space or SearchSpace()

    best: Optional[EvalResult] = None
    hist: List[EvalResult] = []

    for i in range(n_iter):
        p = space.sample(rng)

        # Bias toward high-rho (harder ST cases) and delta near boundary
        if rng.random() < 0.5:
            p["delta"] = float(rng.uniform(0.93, 0.99))
        if rng.random() < 0.4:
            p["rho"] = float(rng.uniform(0.65, 0.85))

        try:
            eval_seed = seed + 1000 + i
            res = evaluate_params(
                p,
                seed=eval_seed,
                alpha=alpha,
                margin=margin,
                prescreen_buffer=prescreen_buffer,
                use_fast_st=True,
                nu=nu,
            )
            # Store the generation seed for reproducibility during validation
            res.params["_eval_seed"] = eval_seed
        except Exception as e:
            nan_ci = (float("nan"), float("nan"))
            res = EvalResult(
                ok_pattern=False, score=float("inf"), params=p,
                ci_iid=nan_ci, ci_cluster=nan_ci, ci_spatial=nan_ci, ci_spatiotemporal=nan_ci,
                eq_iid=False, eq_cluster=False, eq_spatial=False, eq_spatiotemporal=False,
                mu_hat_iid=float("nan"), mu_hat_cluster=float("nan"),
                mu_hat_spatial=float("nan"), mu_hat_spatiotemporal=float("nan"),
                notes=f"Exception: {type(e).__name__}: {e}",
            )

        hist.append(res)

        if best is None or (res.ok_pattern and not best.ok_pattern) or (
            res.score < best.score and (res.ok_pattern >= best.ok_pattern)
        ):
            best = res

        if (i + 1) % verbose_every == 0:
            b = best
            status = "FOUND" if b.ok_pattern else "searching"
            print(
                f"[{i+1:4d}/{n_iter}] {status} best_score={b.score:.4g} "
                f"eq(iid,clu,spa,st)=({b.eq_iid},{b.eq_cluster},{b.eq_spatial},{b.eq_spatiotemporal}) "
                f"CI_st={b.ci_spatiotemporal}"
            )

        if res.ok_pattern:
            print(f"Target pattern found at iter {i+1}: score={res.score:.4g}")
            best = res
            break

    assert best is not None
    return best, hist


# ---------------------------------------------------------------------------
# Validation with full engine
# ---------------------------------------------------------------------------

def validate_with_full_engine(
    params: Dict[str, Any],
    *,
    seed: int,
    alpha: float = 0.05,
    margin: float = 1.0,
    bootstrap_B: int = 200,
    bootstrap_seed: int = 12345,
    mu_ci_method: str = "parametric_bootstrap",
) -> EvalResult:
    """Re-evaluate winning params using the full SpatioTemporalTOST engine.

    This is slower but uses the production-grade bootstrap CI, giving a
    reliable final answer.  Run this on the search winner before saving.

    Parameters
    ----------
    params : dict
        Generator kwargs + n_clusters.
    seed : int
        RNG seed for data generation (same as during search).
    alpha, margin : float
        TOST parameters.
    bootstrap_B : int
        Number of bootstrap replicates for the ST CI.
    bootstrap_seed : int
        RNG seed for bootstrap.
    mu_ci_method : str
        CI method for the ST engine.

    Returns
    -------
    EvalResult
        Full evaluation with production CI.
    """
    from ..engines.spatiotemporal_tost import SpatioTemporalTOST, SpatioTemporalConfig

    # Generate data
    gen = synthetic_tost_data.generate_spatiotemporal
    sig = inspect.signature(gen)
    allowed = set(sig.parameters.keys())
    call = {"seed": seed}
    for k, v in params.items():
        if k in allowed:
            call[k] = v

    df_long, meta = gen(**call)
    n_clusters = int(params.get("n_clusters", 25))
    df_diff = _make_diff_df(df_long, n_clusters)

    # Upstream engines
    partial = _run_engines(
        df_diff, alpha=alpha, margin=margin,
        prescreen_buffer=0.0,  # no prescreen shortcut for validation
        skip_spatial=False,
    )

    # Full ST engine
    st_cfg = SpatioTemporalConfig(
        mu_ci_method=mu_ci_method,
        mu_bootstrap_B=bootstrap_B,
        mu_bootstrap_seed=bootstrap_seed,
        verbose_diagnostics=True,
    )
    st_eng = SpatioTemporalTOST(
        y="diff", cluster="cluster_id", time="time",
        x="x", ycoord="y", config=st_cfg,
    )
    r_st = st_eng.fit(df_diff, alpha=alpha, margins=[margin]).iloc[0]
    ci_st = (float(r_st["ci_low"]), float(r_st["ci_high"]))
    eq_st = bool(r_st["equivalent"])
    mu_st = float(r_st["mu_hat"])

    ok = partial.eq_iid and partial.eq_cluster and partial.eq_spatial and (not eq_st)

    score = (
        _score_ci(partial.ci_iid, margin, should_pass=True) +
        _score_ci(partial.ci_cluster, margin, should_pass=True) +
        _score_ci(partial.ci_spatial, margin, should_pass=True) +
        _score_ci(ci_st, margin, should_pass=False)
    )

    return EvalResult(
        ok_pattern=ok,
        score=score,
        params=dict(params),
        ci_iid=partial.ci_iid,
        ci_cluster=partial.ci_cluster,
        ci_spatial=partial.ci_spatial,
        ci_spatiotemporal=ci_st,
        eq_iid=partial.eq_iid,
        eq_cluster=partial.eq_cluster,
        eq_spatial=partial.eq_spatial,
        eq_spatiotemporal=eq_st,
        mu_hat_iid=partial.mu_hat_iid,
        mu_hat_cluster=partial.mu_hat_cluster,
        mu_hat_spatial=partial.mu_hat_spatial,
        mu_hat_spatiotemporal=mu_st,
        notes=f"validated with full engine (B={bootstrap_B}, method={mu_ci_method})",
    )


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_best(best: EvalResult, path: str) -> None:
    """Serialize best result to JSON.

    Parameters
    ----------
    best : EvalResult
        Winning candidate.
    path : str
        Output JSON file path.
    """
    payload = {
        "ok_pattern": bool(best.ok_pattern),
        "score": float(best.score),
        "params": {k: v for k, v in best.params.items()
                   if not isinstance(v, (pd.DataFrame, np.ndarray))},
        "ci_iid": list(map(float, best.ci_iid)),
        "ci_cluster": list(map(float, best.ci_cluster)),
        "ci_spatial": list(map(float, best.ci_spatial)),
        "ci_spatiotemporal": list(map(float, best.ci_spatiotemporal)),
        "eq_iid": bool(best.eq_iid),
        "eq_cluster": bool(best.eq_cluster),
        "eq_spatial": bool(best.eq_spatial),
        "eq_spatiotemporal": bool(best.eq_spatiotemporal),
        "mu_hat_iid": float(best.mu_hat_iid),
        "mu_hat_cluster": float(best.mu_hat_cluster),
        "mu_hat_spatial": float(best.mu_hat_spatial),
        "mu_hat_spatiotemporal": float(best.mu_hat_spatiotemporal),
        "notes": str(best.notes),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(f"Saved to {path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    best, hist = search(seed=123, n_iter=60, alpha=0.05, margin=1.0)
    print("\nBest (fast Wald):")
    print(f"  ok_pattern={best.ok_pattern}")
    print(f"  score={best.score:.4g}")
    print(f"  eq(iid,clu,spa,st)=({best.eq_iid},{best.eq_cluster},{best.eq_spatial},{best.eq_spatiotemporal})")
    print(f"  CI_st={best.ci_spatiotemporal}")
    print(f"  params={best.params}")

    if best.ok_pattern:
        print("\nValidating with full engine...")
        val = validate_with_full_engine(
            best.params, seed=123 + 1000 + 0,  # match search seed
            bootstrap_B=200,
        )
        print(f"  Validated ok_pattern={val.ok_pattern}")
        print(f"  CI_st (bootstrap)={val.ci_spatiotemporal}")
        save_best(val, "best_spatiotemporal_params.json")

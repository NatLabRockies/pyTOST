"""pyTOST.data_gen.optimize_spatiotemporal_params

Speed-first (but statistically defensible) optimizer for *spatiotemporal* synthetic data.

Goal
----
Find generator parameters for ``synthetic_tost_data.generate_spatiotemporal`` such that,
at equivalence margin Δ=1:

  - naive / partially-correct engines (IID, Cluster, Spatial, Temporal) produce
    *incorrectly narrow* confidence intervals and typically conclude equivalence; while
  - the fully spatiotemporal engine (AR(1) ⊗ Matérn) and a commensurate spatiotemporal
    block-bootstrap validation CI agree with each other and are wider (often flipping
    the decision to non-equivalence).

This mirrors the pattern used by the spatial/temporal optimizers, but targets
spatiotemporal dependence.

Performance strategy
--------------------
The expensive step is spatiotemporal fitting + bootstrapping. We therefore use a
two-stage evaluation:

  1) cheap screen: IID/Cluster/Temporal + a single spatiotemporal fit (Wald, reg)
  2) expensive confirm: small-B parametric bootstrap + small-B time-block bootstrap
     (centered + symmetric CI) only for promising candidates.

The final ``save_best`` re-validates the winning candidate with larger B and (optionally)
refit-based time-block bootstrap.

All methods used here are common in practice:
  - HAC (Newey–West) and moving-block bootstrap for temporal dependence
  - Matérn GLS (REML) for spatial dependence
  - AR(1) ⊗ Matérn for separable spatiotemporal covariance
  - time-block bootstrap of full spatial snapshots for ST validation

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List
import inspect
import json
import math
import time
import os
import multiprocessing as mp

import numpy as np
import pandas as pd

from . import synthetic_tost_data
from ..engines import iid_tost
from ..engines import cluster_tost
from ..engines import temporal_tost
from ..workflow import run_tost, WorkflowOptions
from ..engines.spatial_tost import SpatialConfig
from ..engines.spatiotemporal_tost import SpatioTemporalConfig


# -----------------------------------------------------------------------------
# Search space
# -----------------------------------------------------------------------------


@dataclass
class SearchSpace:
    # Size controls (keep moderate for speed)
    n_space_min: int = 20
    n_space_max: int = 80
    n_time_min: int = 40
    n_time_max: int = 120

    # How we chunk locations into buildings for cluster/spatial engines
    n_buildings_min: int = 5
    n_buildings_max: int = 14

    # Dependence knobs
    rho_min: float = 0.55
    rho_max: float = 0.90
    length_scale_frac_min: float = 0.15  # as fraction of domain diameter
    length_scale_frac_max: float = 0.70

    # Variance knobs
    spatial_sd_min: float = 0.40
    spatial_sd_max: float = 2.50
    obs_sd_min: float = 0.10
    obs_sd_max: float = 1.50

    # True mean difference (B - A)
    delta_true_min: float = 0.88
    delta_true_max: float = 0.99

    # Domain: fixed square by default (simplifies identifiability)
    domain_halfwidth: float = 2.0

    def sample(self, rng: np.random.Generator) -> Dict[str, Any]:
        n_space = int(rng.integers(self.n_space_min, self.n_space_max + 1))
        n_time = int(rng.integers(self.n_time_min, self.n_time_max + 1))
        n_buildings = int(rng.integers(self.n_buildings_min, self.n_buildings_max + 1))

        # Concentrate rho toward the high end (to make naive CIs too tight) but avoid near-unit-root.
        u = float(rng.uniform(0.0, 1.0))
        rho = float(self.rho_min + (self.rho_max - self.rho_min) * (u**0.45))

        # Domain diameter for scaling
        D = 2.0 * self.domain_halfwidth
        length_scale = float(D * rng.uniform(self.length_scale_frac_min, self.length_scale_frac_max))

        # Variances in log space
        spatial_sd = float(10 ** rng.uniform(math.log10(self.spatial_sd_min), math.log10(self.spatial_sd_max)))
        obs_sd = float(10 ** rng.uniform(math.log10(self.obs_sd_min), math.log10(self.obs_sd_max)))

        delta = float(rng.uniform(self.delta_true_min, self.delta_true_max))

        return {
            "n_space": n_space,
            "n_time": n_time,
            "n_buildings": n_buildings,
            "rho": rho,
            "length_scale": length_scale,
            "spatial_sd": spatial_sd,
            "obs_sd": obs_sd,
            "domain": (-self.domain_halfwidth, self.domain_halfwidth, -self.domain_halfwidth, self.domain_halfwidth),
            "delta": delta,
        }


@dataclass
class EvalResult:
    ok_pattern: bool
    score: float
    params: Dict[str, Any]

    ci_iid: Tuple[float, float]
    ci_cluster: Tuple[float, float]
    ci_spatial: Tuple[float, float]
    ci_temporal: Tuple[float, float]
    ci_spatiotemporal: Tuple[float, float]
    ci_st_validate: Tuple[float, float]

    eq_iid: bool
    eq_cluster: bool
    eq_spatial: bool
    eq_temporal: bool
    eq_spatiotemporal: bool

    mu_hat: float
    notes: str = ""


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _make_panel(df_long: pd.DataFrame, *, n_buildings: int) -> pd.DataFrame:
    """Convert long A/B output into a balanced diff panel.

    Returns a DataFrame with columns: sample_id, building_id, x, y_sp, t, diff
    """
    wide = df_long.pivot(index="sample_id", columns="arm", values="y").reset_index()
    meta = df_long[df_long["arm"] == "A"]["sample_id x y_sp t".split()].copy()
    meta = meta.merge(wide, on="sample_id", how="inner")
    meta["diff"] = meta["B"] - meta["A"]

    # Derive space index assuming sample_id = space_idx * n_time + t (generator convention)
    n_time = int(meta["t"].max() + 1)
    space_idx = (meta["sample_id"] // max(n_time, 1)).astype(int)
    space_n = int(space_idx.nunique())
    n_buildings = int(max(1, min(int(n_buildings), space_n)))
    per = int((space_n + n_buildings - 1) / n_buildings)
    meta["building_id"] = (space_idx // max(per, 1)).astype(int)

    return meta[["sample_id", "building_id", "x", "y_sp", "t", "diff"]].copy()


def _ci_from_primary(res: Dict[str, Any], margin: float) -> Tuple[Tuple[float, float], bool, float]:
    p = res["primary"]
    r = p.loc[p["delta"] == float(margin)].iloc[0]
    ci = (float(r["ci_low"]), float(r["ci_high"]))
    eq = bool(r["equivalent"])
    mu = float(r["mu_hat"])
    return ci, eq, mu


def _inside_pen(ci: Tuple[float, float], margin: float) -> float:
    lo, hi = ci
    return max(0.0, -margin - lo) + max(0.0, hi - margin)


def _width(ci: Tuple[float, float]) -> float:
    return float(ci[1] - ci[0])


# -----------------------------------------------------------------------------
# Evaluation (single candidate)
# -----------------------------------------------------------------------------


def evaluate_params(
    params: Dict[str, Any],
    *,
    seed: int = 0,
    alpha: float = 0.05,
    margin: float = 1.0,
    hac_lags: int = 8,
    # screening bootstrap sizes
    st_param_B: int = 20,
    st_block_B: int = 20,
    # validation bootstrap sizes (expensive)
    st_param_B_confirm: int = 60,
    st_block_B_confirm: int = 60,
    confirm: bool = False,
    # time-block bootstrap details
    block_len: Optional[int] = None,
    block_center: bool = True,
    # choose symmetric CI for stability (defensible for dependent resampling)
    st_block_ci_style: str = "symmetric",
    # whether to refit covariance each time-block draw (very expensive)
    st_block_refit_cov: bool = False,
    st_block_refit_maxiter: int = 25,
    # spatial engine config (keep default)
    spatial_config: Optional[SpatialConfig] = None,
    st_config_base: Optional[SpatioTemporalConfig] = None,
    prescreen_buffer: float = 0.12,
) -> EvalResult:
    """Generate one dataset and evaluate the demo pattern.

    Performance notes
    -----------------
    ``confirm=False`` avoids refit-based block bootstrap and uses small B.
    ``confirm=True`` uses the larger B values and (optionally) refit bootstrap.
    """

    gen = synthetic_tost_data.generate_spatiotemporal
    sig = inspect.signature(gen)
    allowed = set(sig.parameters.keys())

    n_buildings = int(params.get("n_buildings", 10))
    call = {"seed": int(seed)}
    for k, v in params.items():
        if k in allowed:
            call[k] = v

    df_long, _meta = gen(**call)
    df = _make_panel(df_long, n_buildings=n_buildings)

    # ----- cheap engines -----------------------------------------------------
    # IID
    iid = iid_tost.IIDTOST(y="diff")
    r_iid = iid.fit(df, alpha=alpha, margins=[margin]).iloc[0]
    ci_iid = (float(r_iid["ci_low"]), float(r_iid["ci_high"]))
    eq_iid = bool(r_iid["equivalent"])
    mu_hat = float(r_iid["mu_hat"])

    # Cluster
    clu = cluster_tost.ClusterTOST(y="diff", cluster="building_id")
    r_clu = clu.fit(df, alpha=alpha, margins=[margin]).iloc[0]
    ci_clu = (float(r_clu["ci_low"]), float(r_clu["ci_high"]))
    eq_clu = bool(r_clu["equivalent"])

    # Temporal
    tmp = temporal_tost.TemporalTOST(y="diff", time="t", hac_lags=hac_lags)
    r_tmp = tmp.fit(df, alpha=alpha, margins=[margin]).iloc[0]
    ci_tmp = (float(r_tmp["ci_low"]), float(r_tmp["ci_high"]))
    eq_tmp = bool(r_tmp["equivalent"])

    # Early reject: if naive methods already fail equivalence, this candidate doesn't
    # demonstrate "incorrectly narrow" inference.
    if (not eq_iid) or (not eq_clu):
        score = 1e6 + _inside_pen(ci_iid, margin) + _inside_pen(ci_clu, margin)
        return EvalResult(
            ok_pattern=False,
            score=float(score),
            params=dict(params),
            ci_iid=ci_iid,
            ci_cluster=ci_clu,
            ci_spatial=(float("nan"), float("nan")),
            ci_temporal=ci_tmp,
            ci_spatiotemporal=(float("nan"), float("nan")),
            ci_st_validate=(float("nan"), float("nan")),
            eq_iid=eq_iid,
            eq_cluster=eq_clu,
            eq_spatial=False,
            eq_temporal=eq_tmp,
            eq_spatiotemporal=False,
            mu_hat=mu_hat,
            notes="early-exit: iid/cluster not equivalent",
        )

    # Spatial (moderate cost): only run if cluster CI is close enough to boundary
    inner_lo = -float(margin) + float(prescreen_buffer)
    inner_hi = float(margin) - float(prescreen_buffer)
    if (ci_clu[0] > inner_lo) and (ci_clu[1] < inner_hi):
        # Skip spatial when deep inside; it rarely changes the story.
        ci_spa = (float("nan"), float("nan"))
        eq_spa = True
        notes_spa = "screened: skipped spatial"
    else:
        res_spa = run_tost(
            df,
            y="diff",
            margins=[margin],
            alpha=alpha,
            engine="spatial",
            cluster="building_id",
            x="x",
            ycoord="y_sp",
            spatial_config=(spatial_config or SpatialConfig()),
            options=WorkflowOptions(do_sensitivity=False, bootstrap_B=0, seed=seed),
        )
        ci_spa, eq_spa, _ = _ci_from_primary(res_spa, margin)
        notes_spa = "ran spatial"

    # ----- spatiotemporal (screen) ------------------------------------------
    # Use regularized Wald CI for a single-fit diagnostic.
    st_cfg = st_config_base or SpatioTemporalConfig(
        mu_ci_method="wald",
        joint_regularize=True,
        reg_lambda=1e-6,
        mu_timeblock_ci_style="symmetric",
        mu_timeblock_center=block_center,
    )

    res_st_wald = run_tost(
        df,
        y="diff",
        margins=[margin],
        alpha=alpha,
        engine="spatiotemporal",
        cluster="building_id",
        time="t",
        x="x",
        ycoord="y_sp",
        spatiotemporal_config=st_cfg,
        options=WorkflowOptions(do_sensitivity=False, bootstrap_B=0, seed=seed),
    )
    ci_st_wald, eq_st_wald, _ = _ci_from_primary(res_st_wald, margin)

    # If ST Wald CI is not meaningfully wider than naive, no need to pay for bootstraps.
    w_naive = max(_width(ci_iid), _width(ci_clu), _width(ci_tmp))
    if _width(ci_st_wald) < max(0.14, 2.2 * w_naive):
        score = 100.0 + abs(_width(ci_st_wald) - 0.22)
        return EvalResult(
            ok_pattern=False,
            score=float(score),
            params=dict(params),
            ci_iid=ci_iid,
            ci_cluster=ci_clu,
            ci_spatial=ci_spa,
            ci_temporal=ci_tmp,
            ci_spatiotemporal=ci_st_wald,
            ci_st_validate=(float("nan"), float("nan")),
            eq_iid=eq_iid,
            eq_cluster=eq_clu,
            eq_spatial=eq_spa,
            eq_temporal=eq_tmp,
            eq_spatiotemporal=eq_st_wald,
            mu_hat=mu_hat,
            notes=f"screened: st wald too narrow ({_width(ci_st_wald):.3f}) [{notes_spa}]",
        )

    # ----- expensive confirm: ST model-based + ST validation bootstrap -------
    Bp = int(st_param_B_confirm if confirm else st_param_B)
    Bb = int(st_block_B_confirm if confirm else st_block_B)

    # Model-based: parametric bootstrap under fitted ST model
    st_cfg_param = SpatioTemporalConfig(
        mu_ci_method="parametric_bootstrap",
        joint_regularize=True,
        reg_lambda=1e-6,
        mu_bootstrap_B=Bp,
        mu_timeblock_ci_style="symmetric",
        mu_timeblock_center=block_center,
    )
    res_st_param = run_tost(
        df,
        y="diff",
        margins=[margin],
        alpha=alpha,
        engine="spatiotemporal",
        cluster="building_id",
        time="t",
        x="x",
        ycoord="y_sp",
        spatiotemporal_config=st_cfg_param,
        options=WorkflowOptions(do_sensitivity=False, bootstrap_B=0, seed=seed),
    )
    ci_st, eq_st, _ = _ci_from_primary(res_st_param, margin)

    # Validation: time-block bootstrap over full spatial snapshots
    # We prefer a symmetric, centered interval for dependent bootstraps.
    # For speed, refit_cov is usually False during search; save_best can rerun with refit.
    st_cfg_block = SpatioTemporalConfig(
        mu_ci_method="time_block_bootstrap",
        joint_regularize=True,
        reg_lambda=1e-6,
        mu_bootstrap_B=Bb,
        mu_timeblock_L=block_len,
        mu_timeblock_center=block_center,
        mu_timeblock_ci_style=st_block_ci_style,
        mu_timeblock_refit_cov=bool(st_block_refit_cov and confirm),
        mu_timeblock_refit_maxiter=int(st_block_refit_maxiter),
    )
    res_st_block = run_tost(
        df,
        y="diff",
        margins=[margin],
        alpha=alpha,
        engine="spatiotemporal",
        cluster="building_id",
        time="t",
        x="x",
        ycoord="y_sp",
        spatiotemporal_config=st_cfg_block,
        options=WorkflowOptions(do_sensitivity=False, bootstrap_B=0, seed=seed),
    )
    ci_st_val, _eq_val, _ = _ci_from_primary(res_st_block, margin)

    # Pattern we want for demo: naive equivalent, ST NOT equivalent
    ok = (eq_iid and eq_clu and eq_tmp and eq_spa and (not eq_st))

    # Score: enforce agreement between ST model and validation widths, and keep ST wide.
    w_model = _width(ci_st)
    w_val = _width(ci_st_val)
    ratio = w_val / max(w_model, 1e-12)

    # Penalize mismatch in widths; target ratio near 1.
    ratio_pen = abs(math.log(max(ratio, 1e-6)))
    # Encourage being near the boundary so equivalence can flip.
    edge_pen = min(abs(ci_st[0] + margin), abs(ci_st[1] - margin))
    # Penalize if naive methods are not comfortably inside equivalence (we want them "wrongly" confident)
    naive_pen = _inside_pen(ci_iid, margin) + _inside_pen(ci_clu, margin)
    # Penalize if ST isn't wide enough
    wide_pen = max(0.0, 0.20 - w_model) + max(0.0, 0.20 - w_val)
    # Penalize if ST is absurdly wide (usually conditioning trouble)
    absurd_pen = max(0.0, w_model - 0.85) + max(0.0, w_val - 0.85)
    # Penalize if spatial/temporal already fail
    other_pen = (0.0 if eq_spa else 5.0) + (0.0 if eq_tmp else 5.0)

    score = (
        5.0 * ratio_pen
        + 1.0 * edge_pen
        + 20.0 * naive_pen
        + 3.0 * wide_pen
        + 2.0 * absurd_pen
        + other_pen
        + (0.0 if ok else 10.0)
    )

    return EvalResult(
        ok_pattern=bool(ok),
        score=float(score),
        params=dict(params),
        ci_iid=ci_iid,
        ci_cluster=ci_clu,
        ci_spatial=ci_spa,
        ci_temporal=ci_tmp,
        ci_spatiotemporal=ci_st,
        ci_st_validate=ci_st_val,
        eq_iid=eq_iid,
        eq_cluster=eq_clu,
        eq_spatial=eq_spa,
        eq_temporal=eq_tmp,
        eq_spatiotemporal=eq_st,
        mu_hat=mu_hat,
        notes=f"confirm={confirm} Bp={Bp} Bb={Bb} ratio={ratio:.2f} [{notes_spa}]",
    )


# -----------------------------------------------------------------------------
# Search
# -----------------------------------------------------------------------------


def _eval_in_subproc(args: Dict[str, Any]) -> EvalResult:
    return evaluate_params(**args)


def _subproc_worker(q_: "mp.Queue", a: Dict[str, Any]) -> None:
    """Top-level worker so it is picklable under the 'spawn' start method.

    Jupyter + macOS/conda environments commonly use 'spawn', which requires
    the process target to be picklable. Nested/local functions are not.
    """
    try:
        res = _eval_in_subproc(a)
    except Exception as e:
        res = EvalResult(
            ok_pattern=False,
            score=float("inf"),
            params=dict(a.get("params", {})),
            ci_iid=(float("nan"), float("nan")),
            ci_cluster=(float("nan"), float("nan")),
            ci_spatial=(float("nan"), float("nan")),
            ci_temporal=(float("nan"), float("nan")),
            ci_spatiotemporal=(float("nan"), float("nan")),
            ci_st_validate=(float("nan"), float("nan")),
            eq_iid=False,
            eq_cluster=False,
            eq_spatial=False,
            eq_temporal=False,
            eq_spatiotemporal=False,
            mu_hat=float("nan"),
            notes=f"Exception in subprocess: {type(e).__name__}: {e}",
        )
    q_.put(res)


def _run_with_timeout(args: Dict[str, Any], timeout_s: int) -> EvalResult:
    """Evaluate a candidate in a subprocess with a hard timeout.

    Notes
    -----
    On macOS + Jupyter, the default multiprocessing start method is typically "spawn".
    Under "spawn", the parent process' current working directory is captured during
    process start-up. If the notebook's CWD is in a protected location (or otherwise
    not accessible to subprocesses), `p.start()` can fail with `PermissionError:
    [Errno 1] Operation not permitted` originating from `os.getcwd()` inside
    multiprocessing's spawn preparation.

    As a practical, defensible workaround, we temporarily `chdir()` to a safe
    directory (default: /tmp) for the duration of `p.start()`.
    """
    # Prefer spawn for portability; allow override via env var.
    start_method = os.environ.get("PYTOST_MP_START", "spawn")
    try:
        ctx = mp.get_context(start_method)
    except ValueError:
        ctx = mp.get_context("spawn")

    q: mp.Queue = ctx.Queue(maxsize=1)

    prev_cwd = None
    try:
        prev_cwd = os.getcwd()
    except Exception:
        prev_cwd = None

    safe_cwd = os.environ.get("PYTOST_SUBPROC_CWD", "/tmp")
    try:
        os.chdir(safe_cwd)
    except Exception:
        # If changing directories fails, proceed; subprocess start may still work.
        pass

    p = ctx.Process(target=_subproc_worker, args=(q, args), daemon=True)
    p.start()

    # Restore parent working directory immediately after spawning.
    if prev_cwd is not None:
        try:
            os.chdir(prev_cwd)
        except Exception:
            pass

    p.join(timeout=float(timeout_s))
    if p.is_alive():
        p.terminate()
        p.join(timeout=1.0)
        return EvalResult(
            ok_pattern=False,
            score=float("inf"),
            params=dict(args.get("params", {})),
            ci_iid=(float("nan"), float("nan")),
            ci_cluster=(float("nan"), float("nan")),
            ci_spatial=(float("nan"), float("nan")),
            ci_temporal=(float("nan"), float("nan")),
            ci_spatiotemporal=(float("nan"), float("nan")),
            ci_st_validate=(float("nan"), float("nan")),
            eq_iid=False,
            eq_cluster=False,
            eq_spatial=False,
            eq_temporal=False,
            eq_spatiotemporal=False,
            mu_hat=float("nan"),
            notes=f"timeout after {timeout_s}s",
        )
    return q.get() if not q.empty() else EvalResult(
        ok_pattern=False,
        score=float("inf"),
        params=dict(args.get("params", {})),
        ci_iid=(float("nan"), float("nan")),
        ci_cluster=(float("nan"), float("nan")),
        ci_spatial=(float("nan"), float("nan")),
        ci_temporal=(float("nan"), float("nan")),
        ci_spatiotemporal=(float("nan"), float("nan")),
        ci_st_validate=(float("nan"), float("nan")),
        eq_iid=False,
        eq_cluster=False,
        eq_spatial=False,
        eq_temporal=False,
        eq_spatiotemporal=False,
        mu_hat=float("nan"),
        notes="subprocess returned no result",
    )


def search(
    *,
    seed: int = 0,
    n_iter: int = 600,
    alpha: float = 0.05,
    margin: float = 1.0,
    hac_lags: int = 8,
    space: Optional[SearchSpace] = None,
    verbose_every: int = 25,
    # speed controls
    st_param_B: int = 20,
    st_block_B: int = 20,
    prescreen_buffer: float = 0.12,
    use_subprocess: bool = True,
    candidate_timeout_seconds: int = 120,
    # optional debug
    debug_stage_prints: bool = False,
) -> Tuple[EvalResult, List[EvalResult]]:
    """Stochastic search for a good spatiotemporal demo parameter set."""
    rng = np.random.default_rng(seed)
    space = space or SearchSpace()

    best: Optional[EvalResult] = None
    hist: List[EvalResult] = []

    for i in range(int(n_iter)):
        p = space.sample(rng)
        p["seed"] = int(seed + 1000 + i)

        args = dict(
            params=p,
            seed=p["seed"],
            alpha=alpha,
            margin=margin,
            hac_lags=hac_lags,
            st_param_B=st_param_B,
            st_block_B=st_block_B,
            confirm=False,
            prescreen_buffer=prescreen_buffer,
            # validation CI settings (fast mode)
            st_block_refit_cov=False,
            st_block_ci_style="symmetric",
        )

        t0 = time.time()
        if use_subprocess:
            res = _run_with_timeout(args, int(candidate_timeout_seconds))
        else:
            try:
                res = evaluate_params(**args)
            except Exception as e:
                res = EvalResult(
                    ok_pattern=False,
                    score=float("inf"),
                    params=p,
                    ci_iid=(float("nan"), float("nan")),
                    ci_cluster=(float("nan"), float("nan")),
                    ci_spatial=(float("nan"), float("nan")),
                    ci_temporal=(float("nan"), float("nan")),
                    ci_spatiotemporal=(float("nan"), float("nan")),
                    ci_st_validate=(float("nan"), float("nan")),
                    eq_iid=False,
                    eq_cluster=False,
                    eq_spatial=False,
                    eq_temporal=False,
                    eq_spatiotemporal=False,
                    mu_hat=float("nan"),
                    notes=f"Exception: {type(e).__name__}: {e}",
                )
        dt = time.time() - t0

        hist.append(res)

        if best is None or (res.ok_pattern and not best.ok_pattern) or (res.score < best.score and (res.ok_pattern == best.ok_pattern)):
            best = res

        if debug_stage_prints:
            print(f"iter {i+1:4d}: score={res.score:.3g} ok={res.ok_pattern} dt={dt:.2f}s notes={res.notes}")

        if (i + 1) % int(verbose_every) == 0:
            b = best
            status = "FOUND" if b.ok_pattern else "searching"
            print(
                f"[{i+1:4d}/{n_iter}] {status} best_score={b.score:.4g} "
                f"eq(iid,clu,tmp,spa,st)=({b.eq_iid},{b.eq_cluster},{b.eq_temporal},{b.eq_spatial},{b.eq_spatiotemporal}) "
                f"CI_st={b.ci_spatiotemporal} CI_val={b.ci_st_validate}"
            )

        if res.ok_pattern:
            print(f"Pattern found at iter {i+1}: score={res.score:.4g}")
            best = res
            break

    assert best is not None
    return best, hist


def save_best(
    best: EvalResult,
    path: str,
    *,
    alpha: float = 0.05,
    margin: float = 1.0,
    hac_lags: int = 8,
    # stronger validation settings
    st_param_B: int = 120,
    st_block_B: int = 120,
    st_block_refit_cov: bool = True,
) -> None:
    """Re-run the winning candidate with stronger bootstrap settings and save params."""
    if "seed" not in best.params:
        raise ValueError("Refusing to save: best.params must include an integer 'seed'.")

    seed = int(best.params["seed"])
    rerun = evaluate_params(
        dict(best.params),
        seed=seed,
        alpha=alpha,
        margin=margin,
        hac_lags=hac_lags,
        confirm=True,
        st_param_B_confirm=int(st_param_B),
        st_block_B_confirm=int(st_block_B),
        st_block_refit_cov=bool(st_block_refit_cov),
        st_block_ci_style="symmetric",
    )

    payload = {
        "ok_pattern": bool(rerun.ok_pattern),
        "score": float(rerun.score),
        "params": dict(best.params),
        "ci_iid": tuple(map(float, rerun.ci_iid)),
        "ci_cluster": tuple(map(float, rerun.ci_cluster)),
        "ci_spatial": tuple(map(float, rerun.ci_spatial)),
        "ci_temporal": tuple(map(float, rerun.ci_temporal)),
        "ci_spatiotemporal": tuple(map(float, rerun.ci_spatiotemporal)),
        "ci_st_validate": tuple(map(float, rerun.ci_st_validate)),
        "eq_iid": bool(rerun.eq_iid),
        "eq_cluster": bool(rerun.eq_cluster),
        "eq_spatial": bool(rerun.eq_spatial),
        "eq_temporal": bool(rerun.eq_temporal),
        "eq_spatiotemporal": bool(rerun.eq_spatiotemporal),
        "mu_hat": float(rerun.mu_hat),
        "notes": rerun.notes,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    best, hist = search(seed=123, n_iter=600, verbose_every=10)
    print("Best:")
    print(best)
    save_best(best, "best_spatiotemporal_params.json")
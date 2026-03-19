
"""Utilities for tuning spatial synthetic-data parameters for demonstration cases.

This module searches over spatial data-generation settings to find examples where
IID and clustered equivalence tests declare equivalence at a target margin while
a spatial dependence-aware analysis does not. The routines are intended for
development, benchmarking, and example construction rather than routine package
workflows.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple, List
import inspect
import json
import hashlib
import math
import numpy as np
import pandas as pd

from . import synthetic_tost_data
from ..engines import iid_tost
from ..engines import cluster_tost
from ..engines import spatial_tost


@dataclass
class SearchSpace:
    """Parameter bounds and sampling rules for the spatial search routine.

    Attributes
    ----------
    n_clusters_min, n_clusters_max : int
        Inclusive bounds on the number of clusters to generate.
    points_per_cluster_min, points_per_cluster_max : int
        Inclusive bounds on the number of locations generated within each cluster.
    target_total_n : int
        Approximate total sample size used to bias the search toward tractable
        problem sizes.
    target_total_n_jitter : int
        Random perturbation applied around ``target_total_n`` when sampling
        candidate configurations.
    cluster_radius_min, cluster_radius_max : float
        Bounds on within-cluster spatial extent.
    length_scale_min, length_scale_max : float
        Bounds on the spatial correlation length scale.
    field_sd_min, field_sd_max : float
        Bounds on the latent spatial field standard deviation.
    nugget_sd_min, nugget_sd_max : float
        Bounds on the nugget-scale noise standard deviation.
    delta_true_min, delta_true_max : float
        Bounds on the true mean paired difference used in the generator.
    meas_field_sd_min, meas_field_sd_max : float
        Bounds on arm-specific measurement field variability.
    meas_shared_min, meas_shared_max : float
        Bounds on the shared measurement component between arms.
    baseline_global_prob, meas_global_prob : float
        Probabilities for enabling global baseline and measurement-field effects.
    center_global_meas_field_prob : float
        Probability of enabling a cluster-center global measurement field when
        supported by the generator signature.
    building_center_sd_min, building_center_sd_max : float
        Bounds on the standard deviation of cluster-center perturbations when
        supported by the generator signature.
    """
    # Integers
    n_clusters_min: int = 4
    n_clusters_max: int = 12
    points_per_cluster_min: int = 5
    points_per_cluster_max: int = 25

    # Target total N ≈ n_clusters * points_per_cluster (bias search toward fast regimes)
    target_total_n: int = 62
    target_total_n_jitter: int = 6

    # Continuous
    cluster_radius_min: float = 0.10
    cluster_radius_max: float = 2.00

    length_scale_min: float = 0.50
    length_scale_max: float = 2.00

    field_sd_min: float = 0.0
    field_sd_max: float = 0.80

    nugget_sd_min: float = 0.01
    nugget_sd_max: float = 0.50

    # Mean difference (true)
    delta_true_min: float = 0.95
    delta_true_max: float = 0.999

    # Arm-specific measurement spatial error
    meas_field_sd_min: float = 0.0
    meas_field_sd_max: float = 2.5
    meas_shared_min: float = 0.0
    meas_shared_max: float = 1.0

    # Global flags
    baseline_global_prob: float = 0.50
    meas_global_prob: float = 0.50

    # Optional new knobs (if present in generator signature)
    center_global_meas_field_prob: float = 0.50
    building_center_sd_min: float = 0.0
    building_center_sd_max: float = 10.0

    def sample(self, rng: np.random.Generator) -> Dict[str, Any]:
        """Draw one candidate parameter dictionary from the search space.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator used to sample a candidate configuration.

        Returns
        -------
        dict of str to Any
            Parameter dictionary compatible with
            ``synthetic_tost_data.generate_spatial_clusters`` when filtered to
            supported keyword arguments.
        """
        d: Dict[str, Any] = {}
                # Keep total N moderate to keep SpatialTOST iterations fast.
        n_clusters = int(rng.integers(self.n_clusters_min, self.n_clusters_max + 1))
        # Choose points-per-cluster to roughly hit target_total_n, with a bit of jitter.
        ppc_base = int(round(self.target_total_n / max(1, n_clusters)))
        ppc_jitter = int(rng.integers(-self.target_total_n_jitter, self.target_total_n_jitter + 1))
        points_per_cluster = int(ppc_base + ppc_jitter)
        points_per_cluster = int(max(self.points_per_cluster_min, min(self.points_per_cluster_max, points_per_cluster)))

        d["n_clusters"] = n_clusters
        d["points_per_cluster"] = points_per_cluster
        d["cluster_radius"] = float(rng.uniform(self.cluster_radius_min, self.cluster_radius_max))
        d["length_scale"] = float(10 ** rng.uniform(math.log10(self.length_scale_min), math.log10(self.length_scale_max)))
        d["field_sd"] = float(rng.uniform(self.field_sd_min, self.field_sd_max))
        d["nugget_sd"] = float(10 ** rng.uniform(math.log10(self.nugget_sd_min), math.log10(self.nugget_sd_max)))
        d["delta"] = float(rng.uniform(self.delta_true_min, self.delta_true_max))

        d["meas_field_sd"] = float(10 ** rng.uniform(math.log10(max(self.meas_field_sd_min, 1e-6)), math.log10(self.meas_field_sd_max)))
        # allow exact 0 sometimes
        if rng.random() < 0.25:
            d["meas_field_sd"] = 0.0

        d["meas_shared"] = float(rng.uniform(self.meas_shared_min, self.meas_shared_max))
        d["baseline_global"] = bool(rng.random() < self.baseline_global_prob)
        d["meas_global"] = bool(rng.random() < self.meas_global_prob)

        # Optional knobs (only used if generator supports them)
        d["center_global_meas_field"] = bool(rng.random() < self.center_global_meas_field_prob)
        d["building_center_sd"] = float(rng.uniform(self.building_center_sd_min, self.building_center_sd_max))
        return d


@dataclass
class EvalResult:
    """Container for the evaluation of one sampled parameter configuration.

    Attributes
    ----------
    ok_pattern : bool
        Whether the target decision pattern was achieved.
    score : float
        Optimization score, where smaller values are preferred.
    params : dict of str to Any
        Generator parameters used for this evaluation.
    ci_iid, ci_cluster, ci_spatial : tuple of float
        Confidence interval bounds from the IID, clustered, and spatial engines.
    eq_iid, eq_cluster, eq_spatial : bool
        Equivalence decisions from the IID, clustered, and spatial engines.
    mu_hat_iid, mu_hat_cluster, mu_hat_spatial : float
        Estimated mean paired differences from each engine.
    df_fingerprint : str, optional
        Hash fingerprint for the derived difference data frame.
    module_versions : dict of str to str, optional
        Module file paths recorded for reproducibility diagnostics.
    notes : str, optional
        Free-form note about screening, early exit, or evaluation errors.
    """
    ok_pattern: bool
    score: float
    params: Dict[str, Any]

    ci_iid: Tuple[float, float]
    ci_cluster: Tuple[float, float]
    ci_spatial: Tuple[float, float]

    eq_iid: bool
    eq_cluster: bool
    eq_spatial: bool

    mu_hat_iid: float
    mu_hat_cluster: float
    mu_hat_spatial: float

    # Repro/diagnostics
    df_fingerprint: str = ""
    module_versions: Dict[str, str] = None
    notes: str = ""


def _make_diff_df(df_long: pd.DataFrame) -> pd.DataFrame:
    """Convert long-format paired observations into a difference data frame.

    Parameters
    ----------
    df_long : pandas.DataFrame
        Long-format table containing paired observations with columns including
        ``sample_id``, ``arm``, ``y``, cluster identifiers, and spatial
        coordinates.

    Returns
    -------
    pandas.DataFrame
        Table with columns ``sample_id``, ``group_id``, ``x``, ``y``, and
        ``diff`` suitable for downstream TOST engines.

    Raises
    ------
    ValueError
        If the required identifier, arm, response, cluster, or coordinate
        columns are not present.
    """
    if "sample_id" not in df_long.columns or "arm" not in df_long.columns or "y" not in df_long.columns:
        raise ValueError("df_long missing required columns: sample_id, arm, y")

    # Coordinates + group from arm A (shared locations)
    coord_cols = ["sample_id"]
    for c in ["group_id", "building_id", "cluster_id"]:
        if c in df_long.columns:
            coord_cols.append(c)
            group_col = c
            break
    else:
        raise ValueError("df_long must contain a cluster column (e.g., group_id or building_id).")

    # spatial coords
    if "x" not in df_long.columns:
        raise ValueError("df_long missing x coordinate.")
    ycol = "y_sp" if "y_sp" in df_long.columns else ("y" if "y" in df_long.columns else None)
    if ycol is None:
        raise ValueError("df_long missing y_sp/y coordinate.")

    A = df_long[df_long["arm"] == "A"][coord_cols + ["x", ycol]].copy()

    wide = df_long.pivot(index="sample_id", columns="arm", values="y").reset_index()
    if not {"A", "B"}.issubset(set(wide.columns)):
        raise ValueError("Could not pivot long data into A/B columns; check arms.")

    out = A.merge(wide, on="sample_id", how="inner")
    out["diff"] = out["B"] - out["A"]
    out = out.rename(columns={ycol: "y", group_col: "group_id"})
    return out[["sample_id", "group_id", "x", "y", "diff"]]



def _run_one(
    df_diff: pd.DataFrame,
    *,
    seed: int,
    alpha: float,
    margin: float,
    spatial_config: Optional["spatial_tost.SpatialConfig"] = None,
    prescreen_buffer: float = 0.15,
) -> EvalResult:
    """Evaluate one generated spatial dataset against the target decision pattern.

    Parameters
    ----------
    df_diff : pandas.DataFrame
        Difference table with columns ``diff``, ``group_id``, ``x``, and ``y``.
    seed : int
        Random seed forwarded to workflow components that support stochastic
        validation steps.
    alpha : float
        Significance level used to construct the TOST confidence interval.
    margin : float
        Single equivalence margin to evaluate.
    spatial_config : spatial_tost.SpatialConfig, optional
        Spatial configuration forwarded to the workflow spatial engine.
    prescreen_buffer : float, default=0.15
        Buffer inside the equivalence region used to skip expensive spatial fits
        when the clustered confidence interval is already comfortably inside the
        equivalence bounds.

    Returns
    -------
    EvalResult
        Evaluation summary for the dataset, including confidence intervals,
        equivalence decisions, and screening notes.
    """
    # IID
    iid = iid_tost.IIDTOST(y="diff")
    r_iid = iid.fit(df_diff, alpha=alpha, margins=[margin]).iloc[0]
    ci_iid = (float(r_iid["ci_low"]), float(r_iid["ci_high"]))
    eq_iid = bool(r_iid["equivalent"])
    mu_iid = float(r_iid["mu_hat"])

    # Early exit: SpatialTOST is expensive; only run downstream methods when upstream criteria are met.
    if not eq_iid:
        return EvalResult(
            ok_pattern=False,
            score=1e6 + abs(mu_iid),
            params={},
            ci_iid=ci_iid,
            ci_cluster=(float("nan"), float("nan")),
            ci_spatial=(float("nan"), float("nan")),
            eq_iid=eq_iid,
            eq_cluster=False,
            eq_spatial=False,
            mu_hat_iid=mu_iid,
            mu_hat_cluster=float("nan"),
            mu_hat_spatial=float("nan"),
            notes="early-exit: iid failed",
        )

    # Cluster
    clu = cluster_tost.ClusterTOST(y="diff", cluster="group_id")
    r_clu = clu.fit(df_diff, alpha=alpha, margins=[margin]).iloc[0]
    ci_clu = (float(r_clu["ci_low"]), float(r_clu["ci_high"]))
    eq_clu = bool(r_clu["equivalent"])
    mu_clu = float(r_clu["mu_hat"])

    if not eq_clu:
        return EvalResult(
            ok_pattern=False,
            score=1e5 + abs(mu_clu),
            params={},
            ci_iid=ci_iid,
            ci_cluster=ci_clu,
            ci_spatial=(float("nan"), float("nan")),
            eq_iid=eq_iid,
            eq_cluster=eq_clu,
            eq_spatial=False,
            mu_hat_iid=mu_iid,
            mu_hat_cluster=mu_clu,
            mu_hat_spatial=float("nan"),
            notes="early-exit: cluster failed",
        )

    # Pre-screen: if cluster CI is comfortably inside the equivalence region,
    # it's unlikely (though not impossible) that the spatial LR CI will flip to non-equivalence.
    inner_lo = -float(margin) + float(prescreen_buffer)
    inner_hi =  float(margin) - float(prescreen_buffer)
    if (ci_clu[0] > inner_lo) and (ci_clu[1] < inner_hi):
        # Treat as "not ok pattern" but also not "bad": we just didn't pay for the spatial fit.
        # Score encourages being near the boundary, so we don't waste time deep inside.
        dist_to_edge = min(ci_clu[0] - (-margin), margin - ci_clu[1])
        score = 10.0 + dist_to_edge  # >10 marks "screened"
        return EvalResult(
            ok_pattern=False,
            score=float(score),
            params={},
            ci_iid=ci_iid,
            ci_cluster=ci_clu,
            ci_spatial=(float("nan"), float("nan")),
            eq_iid=eq_iid,
            eq_cluster=eq_clu,
            eq_spatial=False,
            mu_hat_iid=mu_iid,
            mu_hat_cluster=mu_clu,
            mu_hat_spatial=float("nan"),
            notes=f"screened: cluster CI inside (-Δ+{prescreen_buffer}, Δ-{prescreen_buffer}); skipped spatial",
        )

    # Spatial (heavy): run through workflow.run_tost for mechanical consistency with demo notebooks.
    from ..workflow import run_tost, WorkflowOptions
    from ..engines.spatial_tost import SpatialConfig

    res = run_tost(
        df_diff,
        y="diff",
        margins=[margin],
        alpha=alpha,
        engine="spatial",
        cluster="group_id",
        x="x",
        ycoord="y",
        spatial_config=(spatial_config or SpatialConfig()),
        options=WorkflowOptions(do_sensitivity=False, bootstrap_B=0, seed=seed),
    )
    r_spa = res["primary"].iloc[0]
    ci_spa = (float(r_spa["ci_low"]), float(r_spa["ci_high"]))
    eq_spa = bool(r_spa["equivalent"])
    mu_spa = float(r_spa["mu_hat"])

    ok = (eq_iid and eq_clu and (not eq_spa))

    # Continuous score: smaller is better.
    # Penalize if iid/cluster fail to be inside [-margin, margin].
    def inside_pen(ci: tuple[float, float]) -> float:
        lo, hi = ci
        return max(0.0, -margin - lo) + max(0.0, hi - margin)

    # Penalize if spatial succeeds (we want it to fail).
    # If it fails, reward a moderate violation beyond the margin.
    def spatial_pen(ci: tuple[float, float], eq: bool) -> float:
        lo, hi = ci
        if eq:
            return 5.0 + inside_pen(ci)
        viol = max(max(0.0, -margin - lo), max(0.0, hi - margin))
        return abs(viol - 0.25)

    score = inside_pen(ci_iid) + inside_pen(ci_clu) + spatial_pen(ci_spa, eq_spa)

    return EvalResult(
        ok_pattern=ok,
        score=float(score),
        params={},
        ci_iid=ci_iid,
        ci_cluster=ci_clu,
        ci_spatial=ci_spa,
        eq_iid=eq_iid,
        eq_cluster=eq_clu,
        eq_spatial=eq_spa,
        mu_hat_iid=mu_iid,
        mu_hat_cluster=mu_clu,
        mu_hat_spatial=mu_spa,
        notes="full eval (spatial run)",
    )


def evaluate_params(
    params: Dict[str, Any],
    *,
    seed: int = 0,
    alpha: float = 0.05,
    margin: float = 1.0,
    prescreen_buffer: float = 0.15,
    spatial_config: Optional["spatial_tost.SpatialConfig"] = None,
) -> EvalResult:
    """Generate a spatial dataset and evaluate the target equivalence pattern.

    Parameters
    ----------
    params : dict of str to Any
        Candidate generator parameters. Unsupported keys are ignored when
        calling the spatial synthetic-data generator.
    seed : int, default=0
        Seed used for data generation and downstream stochastic components.
    alpha : float, default=0.05
        Significance level used for TOST inference.
    margin : float, default=1.0
        Equivalence margin used to define the target decision pattern.
    prescreen_buffer : float, default=0.15
        Buffer used to decide whether the expensive spatial fit should be
        skipped after clustered prescreening.
    spatial_config : spatial_tost.SpatialConfig, optional
        Spatial engine configuration passed through to the workflow.

    Returns
    -------
    EvalResult
        Evaluation result populated with reproducibility metadata and the
        engine-specific inference summaries.
    """
    gen = synthetic_tost_data.generate_spatial_clusters
    sig = inspect.signature(gen)

    # Effective seed: prefer explicit arg; fall back to params["seed"] for backwards compatibility.
    eff_seed = int(seed if seed is not None else 0)
    if ("seed" in (params or {})) and (seed is None or seed == 0):
        try:
            eff_seed = int(params["seed"])
        except Exception:
            pass

    call: Dict[str, Any] = {"seed": eff_seed}
    allowed = set(sig.parameters.keys())
    for k, v in (params or {}).items():
        if k == "seed":
            continue
        if k in allowed:
            call[k] = v

    df_long, _meta = gen(**call)
    df_diff = _make_diff_df(df_long)

    # Fingerprint to detect accidental non-repro (row order, values)
    df_fingerprint = ""
    try:
        canon = df_diff[["group_id", "x", "y", "diff"]].copy()
        canon = canon.sort_values(["group_id", "x", "y"]).reset_index(drop=True)
        h = pd.util.hash_pandas_object(canon, index=False).values
        df_fingerprint = hashlib.sha256(h.tobytes()).hexdigest()[:16]
    except Exception:
        pass

    # Module file paths for debugging mismatches
    module_versions: Dict[str, str] = {}
    for mod in (iid_tost, cluster_tost, spatial_tost):
        module_versions[mod.__name__] = str(getattr(mod, "__file__", ""))

    try:
        from .. import workflow as _wf  # type: ignore
        module_versions[_wf.__name__] = str(getattr(_wf, "__file__", ""))
    except Exception:
        pass

    res = _run_one(
        df_diff,
        seed=seed,
        alpha=alpha,
        margin=margin,
        spatial_config=spatial_config,
        prescreen_buffer=prescreen_buffer,
    )
    res.params = dict(params or {})
    res.params['df_diff'] = df_diff
    res.df_fingerprint = df_fingerprint
    res.module_versions = module_versions
    return res
def search(
    *,
    seed: int = 0,
    n_iter: int = 400,
    alpha: float = 0.05,
    margin: float = 1.0,
    space: Optional[SearchSpace] = None,
    verbose_every: int = 25,
) -> Tuple[EvalResult, List[EvalResult]]:
    """Run a stochastic search for a useful spatial demonstration scenario.

    Parameters
    ----------
    seed : int, default=0
        Seed for the search-level random number generator.
    n_iter : int, default=400
        Maximum number of candidate configurations to evaluate.
    alpha : float, default=0.05
        Significance level used for each TOST evaluation.
    margin : float, default=1.0
        Equivalence margin used in the optimization target.
    space : SearchSpace, optional
        Search-space specification. If omitted, a default ``SearchSpace`` is
        used.
    verbose_every : int, default=25
        Frequency, in iterations, for printing progress updates.

    Returns
    -------
    best : EvalResult
        Best candidate found during the search.
    history : list of EvalResult
        Evaluation history in the order candidates were assessed.
    """
    rng = np.random.default_rng(seed)
    space = space or SearchSpace()

    best: Optional[EvalResult] = None
    hist: List[EvalResult] = []

    for i in range(n_iter):
        params = space.sample(rng)

        # Encourage challenging cases:
        # - ensure some cross-building dependence via global fields often
        # - keep true delta near boundary (harder)
        if rng.random() < 0.50:
            params["delta"] = float(rng.uniform(0.85, 0.99))
        if rng.random() < 0.60:
            params["meas_shared"] = float(rng.uniform(0.0, 0.3))
        if rng.random() < 0.60:
            params["meas_field_sd"] = float(10 ** rng.uniform(math.log10(0.5), math.log10(space.meas_field_sd_max)))

        try:
            params['seed'] = int(seed + 1000 + i)
            res = evaluate_params(params, seed=params['seed'], alpha=alpha, margin=margin)
        except Exception as e:
            # treat failures as very bad
            res = EvalResult(
                ok_pattern=False,
                score=float("inf"),
                params=params,
                ci_iid=(float("nan"), float("nan")),
                ci_cluster=(float("nan"), float("nan")),
                ci_spatial=(float("nan"), float("nan")),
                eq_iid=False, eq_cluster=False, eq_spatial=False,
                mu_hat_iid=float("nan"), mu_hat_cluster=float("nan"), mu_hat_spatial=float("nan"),
                notes=f"Exception: {type(e).__name__}: {e}",
            )

        hist.append(res)
        if best is None or res.score < best.score or (res.ok_pattern and not best.ok_pattern):
            best = res

        if (i + 1) % verbose_every == 0:
            b = best
            status = "FOUND" if b.ok_pattern else "searching"
            print(f"[{i+1:4d}/{n_iter}] {status} best_score={b.score:.4g} "
                  f"eq(iid,clu,spa)=({b.eq_iid},{b.eq_cluster},{b.eq_spatial}) "
                  f"CI_spa={b.ci_spatial}")

        # early stop if exact pattern found
        if res.ok_pattern:
            print(f"Pattern found at iter {i+1}: score={res.score:.4g}")
            best = res
            break

    assert best is not None
    return best, hist


def save_best(best: EvalResult, path: str, *, alpha: float = 0.05, margin: float = 1.0) -> None:
    """Save the best result to JSON after a reproducibility check.

    Parameters
    ----------
    best : EvalResult
        Best evaluation result to serialize.
    path : str
        Output path for the JSON payload.
    alpha : float, default=0.05
        Significance level used when re-running the reproducibility check.
    margin : float, default=1.0
        Equivalence margin used when re-running the reproducibility check.

    Raises
    ------
    ValueError
        If the result does not contain a reproducible integer seed or if the
        reproduced confidence intervals do not match within tolerance.
    """
    # Require a deterministic seed to be present for reproducibility checks.
    seed = None
    if isinstance(best.params, dict) and "seed" in best.params:
        try:
            seed = int(best.params["seed"])
        except Exception:
            seed = None
    if seed is None:
        raise ValueError(
            "Refusing to save: best.params does not contain a reproducible integer 'seed'. "
            "Update the optimizer to store the evaluation seed in params."
        )

    rerun = evaluate_params(best.params, seed=seed, alpha=alpha, margin=margin)

    def _close(a: tuple[float, float], b: tuple[float, float], tol: float = 1e-6) -> bool:
        return (abs(a[0] - b[0]) <= tol) and (abs(a[1] - b[1]) <= tol)

    # IID/cluster should always reproduce exactly; spatial may be numerically noisier but should be very close.
    if not _close(best.ci_iid, rerun.ci_iid, tol=1e-6):
        raise ValueError(f"Refusing to save: IID CI not reproducible. saved={best.ci_iid} rerun={rerun.ci_iid}")
    if not _close(best.ci_cluster, rerun.ci_cluster, tol=1e-6):
        raise ValueError(f"Refusing to save: Cluster CI not reproducible. saved={best.ci_cluster} rerun={rerun.ci_cluster}")
    if (not np.isnan(best.ci_spatial[0])) and (not np.isnan(rerun.ci_spatial[0])):
        if not _close(best.ci_spatial, rerun.ci_spatial, tol=1e-4):
            raise ValueError(f"Refusing to save: Spatial CI not reproducible. saved={best.ci_spatial} rerun={rerun.ci_spatial}")

    payload = {
        "ok_pattern": bool(best.ok_pattern),
        "score": float(best.score),
        "params": dict(best.params),
        "ci_iid": tuple(map(float, best.ci_iid)),
        "ci_cluster": tuple(map(float, best.ci_cluster)),
        "ci_spatial": tuple(map(float, best.ci_spatial)),
        "eq_iid": bool(best.eq_iid),
        "eq_cluster": bool(best.eq_cluster),
        "eq_spatial": bool(best.eq_spatial),
        "mu_hat_iid": float(best.mu_hat_iid),
        "mu_hat_cluster": float(best.mu_hat_cluster),
        "mu_hat_spatial": float(best.mu_hat_spatial),
        "df_fingerprint": best.df_fingerprint,
        "module_versions": best.module_versions or {},
        "notes": best.notes,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)



if __name__ == "__main__":
    best, hist = search(seed=0, n_iter=400, alpha=0.05, margin=1.0)
    print("Best:")
    print(best)
    save_best(best, "best_params.json")

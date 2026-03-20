
"""
Optimization routine to find synthetic spatial data-generation parameters that yield:
  - IID engine: TOST passes at margin delta=1
  - Cluster engine: TOST passes at margin delta=1
  - Spatial(pubgrade) engine: TOST fails at margin delta=1

This module is designed to work with the *current* local versions of:
  - synthetic_tost_data.generate_spatial_clusters
  - iid_tost.IIDTOST
  - cluster_tost.ClusterTOST
  - spatial_tost.SpatialTOST

It performs a stochastic search (random + adaptive narrowing) and returns the best
parameter set found, plus diagnostic summaries.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List
import inspect
import json
import hashlib
import math
import numpy as np
import pandas as pd

from . import synthetic_tost_data
from .params_io import save_best
from ..engines import iid_tost
from ..engines import cluster_tost
from ..engines import spatial_tost


@dataclass
class SearchSpace:
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
    cluster_center_sd_min: float = 0.0
    cluster_center_sd_max: float = 10.0

    def sample(self, rng: np.random.Generator) -> Dict[str, Any]:
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
        d["cluster_center_sd"] = float(rng.uniform(self.cluster_center_sd_min, self.cluster_center_sd_max))
        return d


@dataclass
class EvalResult:
    """Container for one candidate's evaluation results."""
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
    """
    Convert tidy long A/B rows to a per-location diff dataframe.

    Expects:
      - columns: sample_id, arm in {'A','B'}, y, group_id, x, y_sp (or y)
    Returns:
      - columns: sample_id, group_id, x, y, diff
    """
    if "sample_id" not in df_long.columns or "arm" not in df_long.columns or "y" not in df_long.columns:
        raise ValueError("df_long missing required columns: sample_id, arm, y")

    # Coordinates + group from arm A (shared locations)
    coord_cols = ["sample_id"]
    for c in ["group_id", "cluster_id", "cluster_id"]:
        if c in df_long.columns:
            coord_cols.append(c)
            group_col = c
            break
    else:
        raise ValueError("df_long must contain a cluster column (e.g., group_id or cluster_id).")

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
    """
    Evaluate a single generated dataset.

    Design goal: keep the expensive spatial fit (Matérn REML + LR CI) rare.

    Screening:
      1) IID must pass equivalence at Δ=margin
      2) Cluster must pass equivalence at Δ=margin
      3) Only then, run spatial *if* the cluster CI is "near" the equivalence boundary.

    Parameters
    ----------
    df_diff
        Wide-ish diff frame with columns: diff, group_id, x, y.
    alpha, margin
        TOST settings for a single Δ=margin.
    spatial_config
        Optional SpatialConfig forwarded to workflow.run_tost(engine="spatial").
    prescreen_buffer
        If the cluster CI lies wholly inside (-margin+buffer, margin-buffer),
        we skip the spatial fit because spatial is unlikely to flip the decision.
        Smaller -> more spatial fits (safer but slower).
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
    """
    Generate a spatial synthetic dataset and evaluate the target decision pattern.

    Target pattern at Δ = `margin`
    ------------------------------
      - IID: equivalent (pass)
      - Cluster: equivalent (pass)
      - Spatial: NOT equivalent (fail)

    Performance strategy
    --------------------
    Spatial (Matérn REML + LR CI) is expensive. We therefore:
      1) run IID and Cluster first (cheap),
      2) only attempt Spatial if those pass and the cluster CI is close to the
         equivalence boundary (controlled by `prescreen_buffer`).
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
        module_versions[mod.__name__] = Path(str(getattr(mod, "__file__", ""))).name

    try:
        from .. import workflow as _wf  # type: ignore
        module_versions[_wf.__name__] = Path(str(getattr(_wf, "__file__", ""))).name
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
    """
    Stochastic search over generator parameters.

    Returns
    -------
    best, history
    """
    rng = np.random.default_rng(seed)
    space = space or SearchSpace()

    best: Optional[EvalResult] = None
    hist: List[EvalResult] = []

    for i in range(n_iter):
        params = space.sample(rng)

        # Encourage challenging cases:
        # - ensure some cross-cluster dependence via global fields often
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



if __name__ == "__main__":
    best, hist = search(seed=0, n_iter=400, alpha=0.05, margin=1.0)
    print("Best:")
    print(best)
    save_best(best, "best_params.json", validator=evaluate_params, alpha=0.05, margin=1.0, compare_tol=1e-4)


from __future__ import annotations
"""
Optimization routine to find synthetic *clustered* data-generation parameters for
`generate_cluster_groups(...)` that yield the pattern:

  - IID engine: TOST passes at margin Δ=1
  - Cluster engine: TOST fails at margin Δ=1.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import json
import numpy as np
import pandas as pd
from scipy import optimize as _opt

from pyTOST.data_gen.synthetic_tost_data import generate_cluster_groups
from pyTOST.engines import iid_tost, cluster_tost


def _make_diff_df(df_long: pd.DataFrame) -> pd.DataFrame:
    need = {"sample_id", "arm", "y", "group_id", "x", "y_sp"}
    missing = need - set(df_long.columns)
    if missing:
        raise ValueError(f"df_long missing required columns: {sorted(missing)}")

    A = df_long[df_long["arm"] == "A"][["sample_id", "group_id", "x", "y_sp"]].copy()
    wide = df_long.pivot(index="sample_id", columns="arm", values="y").reset_index()
    out = A.merge(wide, on="sample_id", how="inner")
    out["diff"] = out["B"] - out["A"]
    out = out.rename(columns={"y_sp": "y"})
    return out[["sample_id", "group_id", "x", "y", "diff"]]


def _tost_iid_cluster(
    df_diff: pd.DataFrame, *, alpha: float, margin: float
) -> Tuple[Tuple[float, float], bool, float, Tuple[float, float], bool, float]:
    iid = iid_tost.IIDTOST(y="diff")
    r_iid = iid.fit(df_diff, alpha=alpha, margins=[margin]).iloc[0]
    ci_iid = (float(r_iid["ci_low"]), float(r_iid["ci_high"]))
    eq_iid = bool(r_iid["equivalent"])
    mu_iid = float(r_iid["mu_hat"])

    clu = cluster_tost.ClusterTOST(y="diff", cluster="group_id")
    r_clu = clu.fit(df_diff, alpha=alpha, margins=[margin]).iloc[0]
    ci_clu = (float(r_clu["ci_low"]), float(r_clu["ci_high"]))
    eq_clu = bool(r_clu["equivalent"])
    mu_clu = float(r_clu["mu_hat"])

    return ci_iid, eq_iid, mu_iid, ci_clu, eq_clu, mu_clu


@dataclass(frozen=True)
class Bounds:
    n_groups_min: int = 4
    n_groups_max: int = 30

    points_per_group_min: int = 10
    points_per_group_max: int = 300

    log10_nugget_sd_min: float = -2.2
    log10_nugget_sd_max: float = -0.2

    baseline_sd_min: float = 0.0
    baseline_sd_max: float = 2.0

    log10_meas_group_sd_min: float = -1.0
    log10_meas_group_sd_max: float = 1.0

    meas_shared_min: float = 0.0
    meas_shared_max: float = 0.9

    delta_true_min: float = 0.80
    delta_true_max: float = 0.99

    baseline_global: bool = True

    def as_de_bounds(self) -> list[tuple[float, float]]:
        return [
            (float(self.n_groups_min), float(self.n_groups_max)),
            (float(self.points_per_group_min), float(self.points_per_group_max)),
            (self.log10_nugget_sd_min, self.log10_nugget_sd_max),
            (self.baseline_sd_min, self.baseline_sd_max),
            (self.log10_meas_group_sd_min, self.log10_meas_group_sd_max),
            (self.meas_shared_min, self.meas_shared_max),
            (self.delta_true_min, self.delta_true_max),
        ]


def _decode(x: np.ndarray, b: Bounds) -> Dict[str, Any]:
    n_groups = int(np.clip(int(round(float(x[0]))), b.n_groups_min, b.n_groups_max))
    points_per_group = int(np.clip(int(round(float(x[1]))), b.points_per_group_min, b.points_per_group_max))
    nugget_sd = float(10 ** float(x[2]))
    baseline_sd = float(x[3])
    meas_group_sd = float(10 ** float(x[4]))
    meas_shared = float(np.clip(float(x[5]), 0.0, 1.0))
    delta = float(x[6])

    return {
        "n_groups": n_groups,
        "points_per_group": points_per_group,
        "delta": delta,
        "nugget_sd": nugget_sd,
        "baseline_sd": baseline_sd,
        "baseline_global": bool(b.baseline_global),
        "meas_group_sd": meas_group_sd,
        "meas_shared": meas_shared,
    }


def evaluate(
    gen_kwargs: Dict[str, Any],
    *,
    alpha: float = 0.05,
    margin: float = 1.0,
    seed: int = 123,
) -> Dict[str, Any]:
    df_long, _meta = generate_cluster_groups(seed=seed, **gen_kwargs)
    df_diff = _make_diff_df(df_long)
    ci_iid, eq_iid, mu_iid, ci_clu, eq_clu, mu_clu = _tost_iid_cluster(df_diff, alpha=alpha, margin=margin)

    # Compute boundary violation beyond margin
    def violation(ci: Tuple[float, float]) -> float:
        lo, hi = ci
        return float(max(max(0.0, -margin - lo), max(0.0, hi - margin)))

    ok = bool(eq_iid and (not eq_clu))

    # Scoring: enforce IID pass, enforce cluster fail
    score = 0.0
    if not eq_iid:
        score += 100.0 + violation(ci_iid)

    if eq_clu:
        score += 100.0 + max(0.0, margin - abs(mu_clu))

    # If cluster fails, prefer a *modest* failure, not extreme.
    if not eq_clu:
        v = violation(ci_clu)
        target = 0.20
        score += abs(v - target)

    # Keep mean near boundary for demo value
    score += 0.15 * max(0.0, (margin - abs(mu_iid)) - 0.05)

    return {
        "score": float(score),
        "ok_pattern": ok,
        "ci_iid": [float(ci_iid[0]), float(ci_iid[1])],
        "ci_cluster": [float(ci_clu[0]), float(ci_clu[1])],
        "eq_iid": bool(eq_iid),
        "eq_cluster": bool(eq_clu),
        "mu_iid": float(mu_iid),
        "mu_cluster": float(mu_clu),
    }


def optimize(
    *,
    alpha: float = 0.05,
    margin: float = 1.0,
    seed: int = 123,
    bounds: Optional[Bounds] = None,
    maxiter: int = 80,
    popsize: int = 16,
    polish: bool = False,
    rng_seed: int = 0,
    workers: int = 1,
    verbose: bool = True,
) -> Dict[str, Any]:
    b = bounds or Bounds()
    de_bounds = b.as_de_bounds()

    best_score = float("inf")
    best_kwargs: Dict[str, Any] = {}
    best_diag: Dict[str, Any] = {}

    def obj(x: np.ndarray) -> float:
        nonlocal best_score, best_kwargs, best_diag
        kwargs = _decode(x, b)
        try:
            diag = evaluate(kwargs, alpha=alpha, margin=margin, seed=seed)
        except Exception:
            return 1e6

        if diag["score"] < best_score:
            best_score = diag["score"]
            best_kwargs = kwargs
            best_diag = diag
            if verbose:
                print(
                    f"best score={best_score:.3f} "
                    f"IID(eq={diag['eq_iid']}, ci={diag['ci_iid']}) "
                    f"CL(eq={diag['eq_cluster']}, ci={diag['ci_cluster']}) "
                    f"kwargs={kwargs}"
                )
        return float(diag["score"])

    result = _opt.differential_evolution(
        obj,
        bounds=de_bounds,
        maxiter=int(maxiter),
        popsize=int(popsize),
        polish=bool(polish),
        seed=int(rng_seed),
        updating="deferred" if workers != 1 else "immediate",
        workers=int(workers),
        disp=False,
    )

    if not best_kwargs:
        best_kwargs = _decode(np.asarray(result.x), b)
        best_diag = evaluate(best_kwargs, alpha=alpha, margin=margin, seed=seed)

    return {
        "best_kwargs": best_kwargs,
        "diagnostics": {
            **best_diag,
            "optimizer_success": bool(result.success),
            "optimizer_message": str(result.message),
            "nit": int(getattr(result, "nit", -1)),
            "nfev": int(getattr(result, "nfev", -1)),
        },
    }


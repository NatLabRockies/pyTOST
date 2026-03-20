"""
Optimization routine to find synthetic *temporal* data-generation parameters that yield:

  - IID engine:       TOST passes at margin delta=1
  - Cluster engine:   TOST passes at margin delta=1
  - Temporal engine:  TOST fails at margin delta=1

Assumptions / conventions
-------------------------
- The paired design is represented in long form with `arm in {"A","B"}` and `sample_id`.
- We evaluate engines on the paired difference: diff = y_B - y_A.
- We treat `series_id` (if present) as the cluster label for ClusterTOST.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List
import inspect
import json
import math

import numpy as np
import pandas as pd

from . import synthetic_tost_data
from .params_io import save_best
from ..engines import iid_tost
from ..engines import cluster_tost
from ..engines import temporal_tost


@dataclass
class SearchSpace:
    # Discrete / integer
    n_time_min: int = 60
    n_time_max: int = 480
    series_min: int = 1
    series_max: int = 12

    # Continuous
    rho_min: float = 0.60
    rho_max: float = 0.995

    process_sd_min: float = 0.05
    process_sd_max: float = 4.0

    obs_sd_min: float = 0.02
    obs_sd_max: float = 2.0

    # True mean difference (B - A)
    delta_true_min: float = 0.70
    delta_true_max: float = 0.995

    def sample(self, rng: np.random.Generator) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        d["n_time"] = int(rng.integers(self.n_time_min, self.n_time_max + 1))
        d["series_per_arm"] = int(rng.integers(self.series_min, self.series_max + 1))
        # concentrate rho near 1 in logit space (more likely to create HAC inflation)
        u = float(rng.uniform(0.0, 1.0))
        d["rho"] = float(self.rho_min + (self.rho_max - self.rho_min) * (u**0.35))
        d["process_sd"] = float(10 ** rng.uniform(math.log10(self.process_sd_min), math.log10(self.process_sd_max)))
        d["obs_sd"] = float(10 ** rng.uniform(math.log10(self.obs_sd_min), math.log10(self.obs_sd_max)))
        d["delta"] = float(rng.uniform(self.delta_true_min, self.delta_true_max))
        return d


@dataclass
class EvalResult:
    ok_pattern: bool
    score: float
    params: Dict[str, Any]
    ci_iid: Tuple[float, float]
    ci_cluster: Tuple[float, float]
    ci_temporal: Tuple[float, float]
    eq_iid: bool
    eq_cluster: bool
    eq_temporal: bool
    mu_hat_iid: float
    mu_hat_cluster: float
    mu_hat_temporal: float
    notes: str = ""


def _make_diff_df(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Convert tidy long A/B rows to a per-observation diff dataframe.

    Expects columns: sample_id, arm in {'A','B'}, y, t
    Optional: series_id (used as cluster id for ClusterTOST)
    """
    req = {"sample_id", "arm", "y", "t"}
    missing = sorted(req - set(df_long.columns))
    if missing:
        raise ValueError(f"Temporal diff conversion missing required columns: {missing}")

    wide = df_long.pivot(index="sample_id", columns="arm", values="y").reset_index()
    if not {"A", "B"}.issubset(set(wide.columns)):
        raise ValueError("Could not pivot long data into A/B columns; check arms.")

    # attach time + series_id from one arm (paired design shares these)
    meta_cols = ["sample_id", "t"]
    if "series_id" in df_long.columns:
        meta_cols.append("series_id")
    meta = df_long[df_long["arm"] == "A"][meta_cols].copy()

    out = meta.merge(wide, on="sample_id", how="inner")
    out["diff"] = out["B"] - out["A"]
    if "series_id" in out.columns:
        out = out.rename(columns={"series_id": "group_id"})
    else:
        out["group_id"] = 0
    return out[["sample_id", "group_id", "t", "diff"]]


def _run_one(df_diff: pd.DataFrame, alpha: float, margin: float, hac_lags: int) -> EvalResult:
    # IID
    iid = iid_tost.IIDTOST(y="diff")
    r_iid = iid.fit(df_diff, alpha=alpha, margins=[margin]).iloc[0]
    ci_iid = (float(r_iid["ci_low"]), float(r_iid["ci_high"]))
    eq_iid = bool(r_iid["equivalent"])
    mu_iid = float(r_iid["mu_hat"])

    # Cluster (cluster on group_id = series_id)
    clu = cluster_tost.ClusterTOST(y="diff", cluster="group_id")
    r_clu = clu.fit(df_diff, alpha=alpha, margins=[margin]).iloc[0]
    ci_clu = (float(r_clu["ci_low"]), float(r_clu["ci_high"]))
    eq_clu = bool(r_clu["equivalent"])
    mu_clu = float(r_clu["mu_hat"])

    # Temporal (HAC)
    tmp = temporal_tost.TemporalTOST(y="diff", time="t", hac_lags=hac_lags)
    r_tmp = tmp.fit(df_diff, alpha=alpha, margins=[margin]).iloc[0]
    ci_tmp = (float(r_tmp["ci_low"]), float(r_tmp["ci_high"]))
    eq_tmp = bool(r_tmp["equivalent"])
    mu_tmp = float(r_tmp["mu_hat"])

    ok = (eq_iid and eq_clu and (not eq_tmp))

    # ---- scoring (smaller is better) --------------------------------------
    def inside_pen(ci):
        lo, hi = ci
        return max(0.0, -margin - lo) + max(0.0, hi - margin)

#     def fail_pen(ci, eq):
#         lo, hi = ci
#         if not eq:
#             # reward a "clear but not insane" failure: target violation around 0.15–0.4
#             viol = max(max(0.0, -margin - lo), max(0.0, hi - margin))
#             return abs(viol - 0.25)
#         return 5.0 + inside_pen(ci)

    def fail_pen(ci, eq):
        if not eq:
            return 0.0  # accept any failure
        return 10.0 + inside_pen(ci)

    score = inside_pen(ci_iid) + inside_pen(ci_clu) + fail_pen(ci_tmp, eq_tmp)

    return EvalResult(
        ok_pattern=ok,
        score=float(score),
        params={},
        ci_iid=ci_iid,
        ci_cluster=ci_clu,
        ci_temporal=ci_tmp,
        eq_iid=eq_iid,
        eq_cluster=eq_clu,
        eq_temporal=eq_tmp,
        mu_hat_iid=mu_iid,
        mu_hat_cluster=mu_clu,
        mu_hat_temporal=mu_tmp,
    )


def evaluate_params(
    params: Dict[str, Any],
    *,
    seed: int = 0,
    alpha: float = 0.05,
    margin: float = 1.0,
    hac_lags: int = 8,
) -> EvalResult:
    """
    Generate AR(1) temporal data via `synthetic_tost_data.generate_temporal_ar1` and
    evaluate the desired pass/fail pattern at the given margin.
    """
    gen = synthetic_tost_data.generate_temporal_ar1
    sig = inspect.signature(gen)
    allowed = set(sig.parameters.keys())

    call = {"seed": seed}
    for k, v in params.items():
        if k in allowed:
            call[k] = v

    # Ensure required keys exist
    for required in ["n_time"]:
        if required not in call:
            raise ValueError(f"Missing required parameter: {required}")

    df_long, meta = gen(**call)
    df_diff = _make_diff_df(df_long)

    res = _run_one(df_diff, alpha=alpha, margin=margin, hac_lags=hac_lags)
    res.params = dict(call)
    res.params["_meta"] = {k: meta.get(k) for k in ["n_time", "series_per_arm", "rho", "process_sd", "obs_sd", "effect"]}
    res.params["_hac_lags"] = int(hac_lags)
    return res


def search(
    *,
    seed: int = 0,
    n_iter: int = 600,
    alpha: float = 0.05,
    margin: float = 1.0,
    hac_lags: int = 8,
    space: Optional[SearchSpace] = None,
    verbose_every: int = 25,
) -> Tuple[EvalResult, List[EvalResult]]:
    """
    Stochastic search over temporal generator parameters.

    Returns
    -------
    best, history
    """
    rng = np.random.default_rng(seed)
    space = space or SearchSpace()

    best: Optional[EvalResult] = None
    hist: List[EvalResult] = []

    for i in range(n_iter):
        p = space.sample(rng)

        # Encourage "hard" cases: delta near the Δ=1 boundary + strong autocorrelation.
        if rng.random() < 0.60:
            p["delta"] = float(rng.uniform(0.88, 0.995))
        if rng.random() < 0.70:
            u = float(rng.uniform(0.0, 1.0))
            p["rho"] = float(0.85 + 0.145 * (u**0.35))  # concentrate close to 1

        try:
            res = evaluate_params(p, seed=seed + 1000 + i, alpha=alpha, margin=margin, hac_lags=hac_lags)
        except Exception as e:
            res = EvalResult(
                ok_pattern=False,
                score=float("inf"),
                params=p,
                ci_iid=(float("nan"), float("nan")),
                ci_cluster=(float("nan"), float("nan")),
                ci_temporal=(float("nan"), float("nan")),
                eq_iid=False, eq_cluster=False, eq_temporal=False,
                mu_hat_iid=float("nan"), mu_hat_cluster=float("nan"), mu_hat_temporal=float("nan"),
                notes=f"Exception: {type(e).__name__}: {e}",
            )

        hist.append(res)
        if best is None or (res.ok_pattern and not best.ok_pattern) or (res.score < best.score and (res.ok_pattern == best.ok_pattern)):
            best = res

        if (i + 1) % verbose_every == 0:
            b = best
            status = "FOUND" if b.ok_pattern else "searching"
            print(
                f"[{i+1:4d}/{n_iter}] {status} best_score={b.score:.4g} "
                f"eq(iid,clu,tmp)=({b.eq_iid},{b.eq_cluster},{b.eq_temporal}) "
                f"CI_tmp={b.ci_temporal}"
            )

        if res.ok_pattern:
            print(f"Pattern found at iter {i+1}: score={res.score:.4g}")
            best = res
            break

    assert best is not None
    return best, hist


if __name__ == "__main__":
    best, hist = search(seed=0, n_iter=600, alpha=0.05, margin=1.0, hac_lags=8)
    print("Best:")
    print(best)
    save_best(best, "best_temporal_params.json")

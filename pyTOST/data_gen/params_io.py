"""Utilities for loading optimized synthetic-data parameters.

Key behavior
------------
Optimization outputs in this project appear in a few shapes:

1) Plain params dict (keys are generator kwargs)
2) {"best": <EvalResult-like dict>} or {"best": {"params": {...}}}
3) Full EvalResult-like dict with a top-level "params" key
4) Any of the above with an auxiliary "_meta" block inside the params

`load_params` normalizes all of these to a single params dict suitable
to pass into the synthetic data generators (after optional filtering via
`kwargs_for`).
"""

from __future__ import annotations

import json
import inspect
from pathlib import Path
from typing import Any, Callable, Mapping


def _unwrap_params(obj: Any) -> dict[str, Any]:
    """Best-effort extraction of a params dict from common result shapes."""
    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict-like JSON at top level; got {type(obj)}")

    cur: Any = obj

    # Repeatedly descend through common wrappers.
    for _ in range(4):
        if isinstance(cur, dict) and "best" in cur and isinstance(cur["best"], dict):
            cur = cur["best"]
            continue
        if isinstance(cur, dict) and "params" in cur and isinstance(cur["params"], dict):
            cur = cur["params"]
            continue
        break

    if not isinstance(cur, dict):
        raise TypeError(f"Could not unwrap params dict; ended at type {type(cur)}")

    return dict(cur)


def load_params(path: str | Path) -> dict[str, Any]:
    """Load a JSON file produced by the optimization routines.

    Parameters
    ----------
    path
        Path to a JSON file. Supported shapes include:
        - a plain dict of parameters
        - {"params": {...}} (EvalResult-like)
        - {"best": {...}} or {"best": {"params": {...}}}

    Returns
    -------
    dict
        Parameters with the private ``_meta`` block removed.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    params = _unwrap_params(raw)
    params.pop("_meta", None)
    return params


def kwargs_for(func: Callable[..., Any], params: Mapping[str, Any]) -> dict[str, Any]:
    """Filter a params mapping down to kwargs accepted by ``func``."""
    sig = inspect.signature(func)
    allowed = set(sig.parameters.keys())
    return {k: params[k] for k in params.keys() if k in allowed}


def validate_required_kwargs(func: Callable[..., Any], kwargs: Mapping[str, Any]) -> None:
    """Raise a readable error if any required parameters for `func` are missing."""
    sig = inspect.signature(func)
    missing = []
    for name, p in sig.parameters.items():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        if p.default is inspect._empty and name not in kwargs:
            missing.append(name)
    if missing:
        raise TypeError(
            f"{func.__name__} missing required args: {missing}. "
            f"Provided keys: {sorted(kwargs.keys())}"
        )

# cluster
def save_best_json(best_kwargs: Dict[str, Any], path: str) -> None:
    # Save ONLY generator kwargs (no extra wrapper keys)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dict(best_kwargs), f, indent=2, sort_keys=True)

# spatial
def save_best_spatial(best: EvalResult, path: str) -> None:
    payload = {
        "ok_pattern": best.ok_pattern,
        "score": best.score,
        "params": best.params,
        "ci_iid": best.ci_iid,
        "ci_cluster": best.ci_cluster,
        "ci_spatial": best.ci_spatial,
        "eq_iid": best.eq_iid,
        "eq_cluster": best.eq_cluster,
        "eq_spatial": best.eq_spatial,
        "mu_hat_iid": best.mu_hat_iid,
        "mu_hat_cluster": best.mu_hat_cluster,
        "mu_hat_spatial": best.mu_hat_spatial,
        "notes": best.notes,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


# temporal
def save_best(best: EvalResult, path: str) -> None:
    payload = {
        "ok_pattern": best.ok_pattern,
        "score": best.score,
        "params": best.params,
        "ci_iid": best.ci_iid,
        "ci_cluster": best.ci_cluster,
        "ci_temporal": best.ci_temporal,
        "eq_iid": best.eq_iid,
        "eq_cluster": best.eq_cluster,
        "eq_temporal": best.eq_temporal,
        "mu_hat_iid": best.mu_hat_iid,
        "mu_hat_cluster": best.mu_hat_cluster,
        "mu_hat_temporal": best.mu_hat_temporal,
        "notes": best.notes,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

# spatio-temporal
def save_best(best: EvalResult, path: str) -> None:
    payload = {
        "ok_pattern": best.ok_pattern,
        "score": best.score,
        "params": best.params,
        "ci_iid": best.ci_iid,
        "ci_cluster": best.ci_cluster,
        "ci_spatial": best.ci_spatial,
        "ci_spatiotemporal": best.ci_spatiotemporal,
        "eq_iid": best.eq_iid,
        "eq_cluster": best.eq_cluster,
        "eq_spatial": best.eq_spatial,
        "eq_spatiotemporal": best.eq_spatiotemporal,
        "mu_hat_iid": best.mu_hat_iid,
        "mu_hat_cluster": best.mu_hat_cluster,
        "mu_hat_spatial": best.mu_hat_spatial,
        "mu_hat_spatiotemporal": best.mu_hat_spatiotemporal,
        "notes": best.notes,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)




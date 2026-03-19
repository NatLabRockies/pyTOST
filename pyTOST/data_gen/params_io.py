"""Utilities for loading and saving synthetic-data parameter specifications.

This module provides helpers for normalizing parameter JSON produced by the
synthetic-data calibration scripts and for filtering parameter mappings to the
keyword arguments accepted by a target generator function. It also includes
lightweight JSON writers used by several optimization scripts.
"""

from __future__ import annotations

import json
import inspect
from pathlib import Path
from typing import Any, Callable, Mapping


def _unwrap_params(obj: Any) -> dict[str, Any]:
    """Extract a parameter mapping from common optimization-result shapes.

    Parameters
    ----------
    obj : Any
        Parsed JSON-like object to normalize. Expected inputs are typically
        dictionaries produced by the synthetic-data calibration scripts.

    Returns
    -------
    dict of str to Any
        Unwrapped parameter dictionary.

    Raises
    ------
    TypeError
        If ``obj`` is not dictionary-like at the top level or if repeated
        unwrapping does not terminate in a dictionary.

    Notes
    -----
    The function descends through several wrapper patterns that occur in saved
    optimization results, including top-level ``"best"`` and ``"params"``
    keys.
    """
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
    """Load a JSON file and return a normalized parameter dictionary.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to a JSON file. Supported top-level shapes include a plain
        parameter dictionary, ``{"params": {...}}``, ``{"best": {...}}``,
        and ``{"best": {"params": {...}}}``.

    Returns
    -------
    dict of str to Any
        Parameter dictionary suitable for passing to a synthetic-data generator.
        Private metadata stored in an ``"_meta"`` key is removed if present.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    params = _unwrap_params(raw)
    params.pop("_meta", None)
    return params



def kwargs_for(func: Callable[..., Any], params: Mapping[str, Any]) -> dict[str, Any]:
    """Filter a parameter mapping to the keyword arguments accepted by a function.

    Parameters
    ----------
    func : callable
        Target callable whose signature defines the accepted keyword arguments.
    params : mapping of str to Any
        Candidate parameter mapping.

    Returns
    -------
    dict of str to Any
        Subset of ``params`` whose keys are present in the signature of
        ``func``.
    """
    sig = inspect.signature(func)
    allowed = set(sig.parameters.keys())
    return {k: params[k] for k in params.keys() if k in allowed}



def validate_required_kwargs(func: Callable[..., Any], kwargs: Mapping[str, Any]) -> None:
    """Validate that a mapping contains all required keyword arguments.

    Parameters
    ----------
    func : callable
        Target callable whose signature is used for validation.
    kwargs : mapping of str to Any
        Candidate keyword-argument mapping.

    Raises
    ------
    TypeError
        If one or more required parameters in the signature of ``func`` are not
        present in ``kwargs``.
    """
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
    """Write a plain parameter dictionary to JSON.

    Parameters
    ----------
    best_kwargs : dict of str to Any
        Best-performing generator keyword arguments.
    path : str
        Output path for the JSON file.

    Notes
    -----
    This writer stores only the generator keyword arguments and does not wrap
    them in any additional metadata structure.
    """
    # Save ONLY generator kwargs (no extra wrapper keys)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dict(best_kwargs), f, indent=2, sort_keys=True)

# spatial
def save_best_spatial(best: EvalResult, path: str) -> None:
    """Write a spatial optimization result to JSON.

    Parameters
    ----------
    best : EvalResult
        Best spatial optimization result object.
    path : str
        Output path for the JSON file.
    """
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
    """Write a temporal optimization result to JSON.

    Parameters
    ----------
    best : EvalResult
        Best temporal optimization result object.
    path : str
        Output path for the JSON file.
    """
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
    """Write a spatiotemporal optimization result to JSON.

    Parameters
    ----------
    best : EvalResult
        Best spatiotemporal optimization result object.
    path : str
        Output path for the JSON file.
    """
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

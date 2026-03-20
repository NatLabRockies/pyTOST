"""Utilities for loading and saving optimized synthetic-data parameters.

This module provides helpers for working with JSON artifacts produced by the
synthetic-data optimization routines. It supports two main workflows:

1. Loading parameter payloads from several historical JSON shapes and
   normalizing them to a generator-kwargs mapping.
2. Saving optimizer results through a single shared ``save_best`` function that
   produces JSON-safe, machine-portable artifacts.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping


def _unwrap_params(obj: Any) -> dict[str, Any]:
    """Best-effort extraction of a params dict from common result shapes.

    Parameters
    ----------
    obj : Any
        JSON-decoded object.

    Returns
    -------
    dict of str to Any
        Unwrapped parameter mapping.

    Raises
    ------
    TypeError
        If the object cannot be reduced to a parameter dictionary.
    """
    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict-like JSON at top level; got {type(obj)}")

    cur: Any = obj
    for _ in range(4):
        if isinstance(cur, dict) and "best" in cur and isinstance(cur["best"], dict):
            cur = cur["best"]
            continue
        if isinstance(cur, dict) and "params" in cur and isinstance(cur["params"], dict):
            cur = cur["params"]
            continue
        if isinstance(cur, dict) and "best_kwargs" in cur and isinstance(cur["best_kwargs"], dict):
            cur = cur["best_kwargs"]
            continue
        break

    if not isinstance(cur, dict):
        raise TypeError(f"Could not unwrap params dict; ended at type {type(cur)}")

    return dict(cur)


def load_params(path: str | Path) -> dict[str, Any]:
    """Load a JSON file produced by the optimization routines.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to a JSON file.

    Returns
    -------
    dict of str to Any
        Parameters with any private ``_meta`` block removed.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    params = _unwrap_params(raw)
    params.pop("_meta", None)
    return params


def kwargs_for(func: Callable[..., Any], params: Mapping[str, Any]) -> dict[str, Any]:
    """Filter a params mapping down to kwargs accepted by ``func``.

    Parameters
    ----------
    func : callable
        Target callable.
    params : mapping
        Candidate parameter mapping.

    Returns
    -------
    dict of str to Any
        Subset of ``params`` accepted by ``func``.
    """
    sig = inspect.signature(func)
    allowed = set(sig.parameters.keys())
    return {k: params[k] for k in params.keys() if k in allowed}


def validate_required_kwargs(func: Callable[..., Any], kwargs: Mapping[str, Any]) -> None:
    """Raise a readable error if required parameters for ``func`` are missing.

    Parameters
    ----------
    func : callable
        Target callable.
    kwargs : mapping
        Candidate kwargs mapping.

    Raises
    ------
    TypeError
        If one or more required arguments are missing.
    """
    sig = inspect.signature(func)
    missing: list[str] = []
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


def _looks_like_path(value: str) -> bool:
    """Return ``True`` if a string appears to encode a filesystem path."""
    return os.path.isabs(value) or "/" in value or "\\" in value


def _sanitize_path_string(value: str) -> str:
    """Strip machine-specific path prefixes from a path-like string."""
    if not _looks_like_path(value):
        return value
    return Path(value).name or value


def _json_ready(obj: Any, *, parent_key: str | None = None) -> Any:
    """Convert optimizer results to JSON-safe, machine-portable objects.

    Parameters
    ----------
    obj : Any
        Object to convert.
    parent_key : str, optional
        Parent mapping key used for context-sensitive sanitization.

    Returns
    -------
    Any
        JSON-safe object.
    """
    if dataclasses.is_dataclass(obj):
        return _json_ready(dataclasses.asdict(obj), parent_key=parent_key)
    if isinstance(obj, Mapping):
        out: dict[str, Any] = {}
        for key, value in obj.items():
            if key == "module_versions" and isinstance(value, Mapping):
                out[str(key)] = {str(k): _sanitize_path_string(str(v)) for k, v in value.items()}
            else:
                out[str(key)] = _json_ready(value, parent_key=str(key))
        return out
    if isinstance(obj, (list, tuple)):
        return [_json_ready(v, parent_key=parent_key) for v in obj]
    if isinstance(obj, Path):
        return _sanitize_path_string(str(obj))
    if isinstance(obj, str) and parent_key in {"module_versions", "path", "file", "source_file"}:
        return _sanitize_path_string(obj)
    if hasattr(obj, "item") and callable(getattr(obj, "item")):
        try:
            return obj.item()
        except Exception:
            pass
    return obj


def _common_payload(best: Any) -> dict[str, Any]:
    """Extract a canonical payload from an optimizer result object."""
    if isinstance(best, Mapping):
        if "best_kwargs" in best and isinstance(best["best_kwargs"], Mapping):
            return dict(best["best_kwargs"])
        return {str(k): v for k, v in best.items()}
    if dataclasses.is_dataclass(best):
        return dataclasses.asdict(best)
    if hasattr(best, "__dict__"):
        return {k: v for k, v in vars(best).items() if not k.startswith("_")}
    raise TypeError(f"Unsupported object type for save_best: {type(best)}")


def _compare_common_ci_fields(saved: Mapping[str, Any], rerun: Mapping[str, Any], tol: float = 1e-6) -> None:
    """Compare shared CI fields between two result payloads.

    Parameters
    ----------
    saved : mapping
        Original payload.
    rerun : mapping
        Recomputed payload.
    tol : float, default=1e-6
        Absolute tolerance for CI endpoint comparisons.

    Raises
    ------
    ValueError
        If any shared finite CI field differs by more than ``tol``.
    """
    ci_keys = sorted(set(saved.keys()) & set(rerun.keys()))
    ci_keys = [k for k in ci_keys if k.startswith("ci_")]
    for key in ci_keys:
        a = saved[key]
        b = rerun[key]
        if not (isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)) and len(a) == 2 and len(b) == 2):
            continue
        try:
            a0, a1 = float(a[0]), float(a[1])
            b0, b1 = float(b[0]), float(b[1])
        except Exception:
            continue
        if not all(map(lambda x: x == x, [a0, a1, b0, b1])):
            continue
        if abs(a0 - b0) > tol or abs(a1 - b1) > tol:
            raise ValueError(f"Refusing to save: {key} not reproducible. saved={(a0, a1)} rerun={(b0, b1)}")


def save_best(
    best: Any,
    path: str | Path,
    *,
    validator: Callable[..., Any] | None = None,
    compare_tol: float = 1e-6,
    **validator_kwargs: Any,
) -> Path:
    """Save an optimizer result or parameter mapping as JSON.

    Parameters
    ----------
    best : Any
        Optimizer result object, dataclass instance, or plain parameter mapping.
    path : str or pathlib.Path
        Output JSON path.
    validator : callable, optional
        Optional callable used to rerun the winning parameters before saving.
        The callable must accept the saved parameter mapping as its first
        argument and return an object compatible with ``save_best`` payload
        extraction.
    compare_tol : float, default=1e-6
        Absolute tolerance used when comparing shared CI fields during
        validator-based reproducibility checks.
    **validator_kwargs : Any
        Additional keyword arguments forwarded to ``validator``.

    Returns
    -------
    pathlib.Path
        Resolved output path.

    Raises
    ------
    ValueError
        If ``validator`` is provided and the rerun does not reproduce the saved
        CI fields within tolerance.
    """
    payload = _json_ready(_common_payload(best))

    if validator is not None:
        params = payload.get("params") if isinstance(payload, Mapping) else None
        if not isinstance(params, Mapping):
            raise ValueError("Validator-based save_best requires a payload with a 'params' mapping.")
        call_kwargs = dict(validator_kwargs)
        if "seed" not in call_kwargs and "seed" in params:
            call_kwargs["seed"] = params["seed"]
        rerun = validator(dict(params), **call_kwargs)
        rerun_payload = _json_ready(_common_payload(rerun))
        _compare_common_ci_fields(payload, rerun_payload, tol=compare_tol)

    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, allow_nan=True)
    return path.resolve()

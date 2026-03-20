"""Shared notebook helpers for synthetic data optimization notebooks.

These utilities keep the optimization notebooks aligned in structure and output
format without changing the optimization code itself.
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


def configure_notebook_environment(max_threads: int = 1) -> Path:
    """Configure a predictable notebook runtime and return the repo root.

    Parameters
    ----------
    max_threads : int, default=1
        Upper bound applied to common BLAS/OpenMP thread environment variables.

    Returns
    -------
    pathlib.Path
        Repository root containing the top-level ``pyTOST`` package directory.

    Raises
    ------
    RuntimeError
        If the repository root cannot be located from the current working
        directory.
    """
    for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ.setdefault(key, str(max_threads))

    here = Path.cwd().resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "pyTOST").is_dir():
            root = candidate
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            return root

    raise RuntimeError("Could not locate repository root containing the pyTOST package.")


def _get_value(obj: Any, name: str, default: Any = np.nan) -> Any:
    """Get a field from a dict-like object or attribute-bearing object."""
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _json_ready(obj: Any) -> Any:
    """Convert common scientific Python objects to JSON-serializable forms."""
    if dataclasses.is_dataclass(obj):
        return _json_ready(dataclasses.asdict(obj))
    if isinstance(obj, Mapping):
        return {str(k): _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_ready(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    return obj


def save_json(payload: Any, path: str | Path) -> Path:
    """Write a JSON payload to disk.

    Parameters
    ----------
    payload : Any
        Object that can be converted to a JSON-safe structure.
    path : str or pathlib.Path
        Output file path.

    Returns
    -------
    pathlib.Path
        Resolved output path.
    """
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_json_ready(payload), f, indent=2, sort_keys=True)
    return path.resolve()


def summarize_result(result: Any, engine_specs: Mapping[str, Mapping[str, str]]) -> pd.DataFrame:
    """Build a tidy engine-level summary table.

    Parameters
    ----------
    result : Any
        Dict-like or attribute-bearing result object.
    engine_specs : mapping
        Mapping from display engine name to field names with keys ``"ci"``,
        ``"eq"``, and ``"mu"``.

    Returns
    -------
    pandas.DataFrame
        Tidy summary with one row per engine.
    """
    rows = []
    for engine, spec in engine_specs.items():
        ci = _get_value(result, spec["ci"], (np.nan, np.nan))
        try:
            ci_low, ci_high = float(ci[0]), float(ci[1])
        except Exception:
            ci_low, ci_high = np.nan, np.nan
        rows.append(
            {
                "engine": engine,
                "equivalent": _get_value(result, spec["eq"], np.nan),
                "mu_hat": _get_value(result, spec["mu"], np.nan),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "ci_width": ci_high - ci_low if np.isfinite(ci_low) and np.isfinite(ci_high) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def history_frame(
    history: Iterable[Any],
    engine_specs: Mapping[str, Mapping[str, str]],
    param_keys: Iterable[str] = (),
) -> pd.DataFrame:
    """Convert a list of evaluated candidates into a flat data frame.

    Parameters
    ----------
    history : iterable
        Iterable of dict-like or attribute-bearing result objects.
    engine_specs : mapping
        Mapping from display engine name to field names with keys ``"ci"``,
        ``"eq"``, and ``"mu"``.
    param_keys : iterable of str, optional
        Parameter names to pull from the nested ``params`` field.

    Returns
    -------
    pandas.DataFrame
        Flat table of search history.
    """
    rows = []
    for item in history:
        row = {
            "ok_pattern": _get_value(item, "ok_pattern", np.nan),
            "score": _get_value(item, "score", np.nan),
            "notes": _get_value(item, "notes", ""),
        }
        params = _get_value(item, "params", {})
        if not isinstance(params, Mapping):
            params = {}
        for key in param_keys:
            row[key] = params.get(key, np.nan)

        for engine, spec in engine_specs.items():
            ci = _get_value(item, spec["ci"], (np.nan, np.nan))
            try:
                ci_low, ci_high = float(ci[0]), float(ci[1])
            except Exception:
                ci_low, ci_high = np.nan, np.nan
            prefix = engine.lower().replace(" ", "_")
            row[f"{prefix}_equivalent"] = _get_value(item, spec["eq"], np.nan)
            row[f"{prefix}_ci_low"] = ci_low
            row[f"{prefix}_ci_high"] = ci_high
            row[f"{prefix}_ci_width"] = ci_high - ci_low if np.isfinite(ci_low) and np.isfinite(ci_high) else np.nan
            row[f"{prefix}_mu_hat"] = _get_value(item, spec["mu"], np.nan)
        rows.append(row)

    return pd.DataFrame(rows)


def result_payload(
    *,
    params: Mapping[str, Any],
    summary_source: Any,
    engine_specs: Mapping[str, Mapping[str, str]],
    score_field: str = "score",
    ok_field: str = "ok_pattern",
    notes_field: str = "notes",
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Construct a consistent JSON payload for saved notebook outputs.

    Parameters
    ----------
    params : mapping
        Generator or search parameters to save.
    summary_source : Any
        Dict-like or attribute-bearing result object providing CI, equivalence,
        and estimate fields.
    engine_specs : mapping
        Mapping from display engine name to field names with keys ``"ci"``,
        ``"eq"``, and ``"mu"``.
    score_field : str, default="score"
        Field name for the scalar score.
    ok_field : str, default="ok_pattern"
        Field name for the boolean pattern flag.
    notes_field : str, default="notes"
        Field name for free-text notes.
    extra : mapping, optional
        Additional metadata to include.

    Returns
    -------
    dict
        JSON-ready payload with ``params`` plus engine-level fields.
    """
    payload: dict[str, Any] = {
        "params": _json_ready(dict(params)),
        "score": _json_ready(_get_value(summary_source, score_field, np.nan)),
        "ok_pattern": _json_ready(_get_value(summary_source, ok_field, np.nan)),
        "notes": _json_ready(_get_value(summary_source, notes_field, "")),
    }
    for engine, spec in engine_specs.items():
        suffix = engine.lower().replace(" ", "_")
        payload[f"ci_{suffix}"] = _json_ready(_get_value(summary_source, spec["ci"], (np.nan, np.nan)))
        payload[f"eq_{suffix}"] = _json_ready(_get_value(summary_source, spec["eq"], np.nan))
        payload[f"mu_hat_{suffix}"] = _json_ready(_get_value(summary_source, spec["mu"], np.nan))
    if extra:
        payload.update(_json_ready(dict(extra)))
    return payload

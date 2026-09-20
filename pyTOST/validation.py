"""
validation.py
=============

Shared input validation with actionable error messages.

Spatial covariance fitting fails deep inside the optimizer when coordinates are NaN or
infinite, producing errors that give no hint about the real cause. Validating up front
lets the engines name the offending column and rows instead.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

__all__ = ["require_finite_columns"]

_MAX_REPORTED_ROWS = 5


def require_finite_columns(
    df: pd.DataFrame,
    columns: Iterable[str],
    *,
    engine: str,
    role: str = "coordinate",
) -> None:
    """Raise a descriptive ``ValueError`` if any column holds a non-finite value.

    Parameters
    ----------
    df : DataFrame
        Data about to be fitted.
    columns : iterable of str
        Column names that must be entirely finite.
    engine : str
        Engine name used to prefix the error message.
    role : str, default "coordinate"
        How the columns are described in the message.

    Raises
    ------
    ValueError
        If a column contains NaN or infinite values, or is not numeric.
    """
    for column in columns:
        values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
        bad_mask = ~np.isfinite(values)
        if not bad_mask.any():
            continue

        bad_positions = np.flatnonzero(bad_mask)
        bad_labels = [repr(label) for label in df.index[bad_positions][:_MAX_REPORTED_ROWS]]
        shown = ", ".join(bad_labels)
        if len(bad_positions) > _MAX_REPORTED_ROWS:
            shown += ", ..."

        n_nan = int(np.isnan(values[bad_mask]).sum())
        n_inf = int(bad_mask.sum()) - n_nan
        parts = []
        if n_nan:
            parts.append(f"{n_nan} NaN")
        if n_inf:
            parts.append(f"{n_inf} infinite")
        breakdown = " and ".join(parts)

        raise ValueError(
            f"{engine}: {role} column {column!r} contains {bad_mask.sum()} non-finite "
            f"value(s) ({breakdown}) at row label(s) [{shown}]. "
            f"Spatial covariance estimation requires finite {role}s. "
            f"Drop these rows (for example "
            f"df.dropna(subset={list(columns)!r})) or impute the missing "
            f"{role}s before fitting."
        )

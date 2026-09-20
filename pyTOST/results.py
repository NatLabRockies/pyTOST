"""
results.py
==========

Result container for :func:`pyTOST.run_tost`.

:class:`TOSTResult` is a ``dict`` subclass, so every existing pattern such as
``result["primary"]`` continues to work unchanged. It adds :meth:`TOSTResult.summary`,
which renders the equivalence decision as a human-readable table suitable for
printing into a report or a notebook.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

__all__ = ["TOSTResult"]

_COLUMNS = ("delta", "mu_hat", "ci_low", "ci_high")
_HEADERS = ("Delta", "mu_hat", "ci_low", "ci_high", "Decision")

# Bootstrap helpers report intervals under different keys depending on the method.
_BOOTSTRAP_CI_KEYS = (
    ("ci_perc_90", "percentile, 90%"),
    ("ci_perc", "percentile"),
    ("ci_basic", "basic"),
    ("ci_sym", "symmetric"),
)


def _decision(equivalent: Any) -> str:
    return "EQUIVALENT" if bool(equivalent) else "NOT EQUIVALENT"


def _format_table(frame: pd.DataFrame, precision: int) -> list[str]:
    """Render one engine's per-margin results as aligned text rows."""
    rows: list[tuple[str, ...]] = []
    for _, record in frame.iterrows():
        rows.append(
            (
                f"{float(record['delta']):g}",
                f"{float(record['mu_hat']):.{precision}f}",
                f"{float(record['ci_low']):.{precision}f}",
                f"{float(record['ci_high']):.{precision}f}",
                _decision(record["equivalent"]),
            )
        )

    widths = [
        max(len(_HEADERS[i]), *(len(row[i]) for row in rows)) if rows else len(_HEADERS[i])
        for i in range(len(_HEADERS))
    ]

    def _line(cells: tuple[str, ...]) -> str:
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells)).rstrip()

    lines = [_line(_HEADERS), _line(tuple("-" * w for w in widths))]
    lines.extend(_line(row) for row in rows)
    return lines


def _methods(frame: pd.DataFrame) -> str:
    if "method" not in frame.columns:
        return ""
    names = list(dict.fromkeys(str(m) for m in frame["method"]))
    return ", ".join(names)


class TOSTResult(dict):
    """Mapping returned by :func:`pyTOST.run_tost`.

    Behaves exactly like the plain ``dict`` returned by earlier versions and adds a
    formatted :meth:`summary` of the equivalence decision.

    Keys
    ----
    engine : str
        Name of the primary engine that produced ``primary``.
    primary : DataFrame
        One row per equivalence margin with columns ``delta``, ``mu_hat``,
        ``ci_low``, ``ci_high``, ``equivalent``, and ``method``.
    sensitivity : dict[str, DataFrame], optional
        Present when sensitivity analyses were enabled.
    bootstrap : dict, optional
        Present when a bootstrap sanity check was run.

    Examples
    --------
    >>> result = run_tost(df, y="diff", margins=[0.5], engine="iid")  # doctest: +SKIP
    >>> print(result.summary())  # doctest: +SKIP
    """

    def summary(self, *, precision: int = 4) -> str:
        """Return a human-readable equivalence decision table.

        Parameters
        ----------
        precision : int, default 4
            Number of decimal places used for the point estimate and CI bounds.

        Returns
        -------
        str
            Multi-line report covering the primary engine and, when present, the
            sensitivity analyses and the bootstrap sanity check.
        """
        lines = ["pyTOST equivalence summary", "=" * 26, ""]

        engine = self.get("engine")
        if engine is not None:
            lines.append(f"Engine: {engine}")

        primary = self.get("primary")
        if isinstance(primary, pd.DataFrame) and not primary.empty:
            method = _methods(primary)
            if method:
                lines.append(f"Method: {method}")
            lines.extend(["", "Primary result", "-" * 14])
            lines.extend(_format_table(primary, precision))
        else:
            lines.extend(["", "Primary result", "-" * 14, "(no primary result)"])

        sensitivity = self.get("sensitivity")
        if sensitivity:
            lines.extend(["", "Sensitivity", "-" * 11])
            for name, frame in sensitivity.items():
                lines.append(f"{name}:")
                if isinstance(frame, pd.DataFrame) and not frame.empty:
                    lines.extend(f"  {line}" for line in _format_table(frame, precision))
                else:
                    lines.append("  (unavailable)")
                lines.append("")
            if lines[-1] == "":
                lines.pop()

        bootstrap = self.get("bootstrap")
        if bootstrap:
            lines.extend(["", "Bootstrap", "-" * 9])
            method = bootstrap.get("method")
            if method:
                lines.append(f"Method: {method}")
            replicates = bootstrap.get("B")
            if replicates is not None:
                lines.append(f"Replicates: {int(replicates)}")
            for key, label in _BOOTSTRAP_CI_KEYS:
                interval = bootstrap.get(key)
                if interval is None:
                    continue
                low, high = interval
                lines.append(
                    f"Mean CI ({label}): "
                    f"[{float(low):.{precision}f}, {float(high):.{precision}f}]"
                )

        return "\n".join(lines)

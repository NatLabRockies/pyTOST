"""
workflow.py
===========

Orchestration for pyTOST engines. Choose from IID, cluster, spatial, temporal, or spatiotemporal engines.

Sensitivity analysis
--------------------
The workflow can optionally compute:
  - cluster bootstrap CI for mean (sanity check)
  - heteroskedastic robust CI
  - robust-location CI (median/trimmed mean)

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

import pandas as pd

from .engines.iid_tost import IIDTOST
from .engines.cluster_tost import ClusterTOST
from .engines.temporal_tost import TemporalTOST
from .engines.spatial_tost import SpatialTOST, SpatialConfig
from .engines.spatiotemporal_tost import SpatioTemporalTOST, SpatioTemporalConfig

from .engines.heteroskedastic_tost import HeteroskedasticTOST
from .engines.robust_location_tost import RobustLocationTOST
from .bootstrap import cluster_bootstrap, spatial_block_bootstrap


@dataclass(frozen=True)
class WorkflowOptions:
    """Options controlling optional sensitivity analyses and bootstrap selection."""

    do_sensitivity: bool = True
    bootstrap_B: int = 200
    seed: int = 42

    # For spatial/spatiotemporal bootstrap selection only:
    #   - True  -> use spatial block bootstrap (allows cross-building dependence)
    #   - False -> use cluster bootstrap over building_id
    #   - None  -> infer from config if possible, else default False
    cross_building_dependence: Optional[bool] = None

    # Block size (same units as x/y) for spatial block bootstrap
    spatial_block_size: float = 1.0


def _infer_cross_building_dependence(
    *,
    options: WorkflowOptions,
    spatial_config: Optional[Any] = None,
    spatiotemporal_config: Optional[Any] = None,
) -> bool:
    """Infer cross-building dependence for bootstrap selection.

    Precedence:
      1) options.cross_building_dependence if not None
      2) config flags if present (e.g., meas_global / baseline_global)
      3) default False
    """
    if options.cross_building_dependence is not None:
        return bool(options.cross_building_dependence)

    cfgs = [c for c in (spatial_config, spatiotemporal_config) if c is not None]
    for cfg in cfgs:
        for name in (
            "cross_building_dependence",
            "cross_cluster_dependence",
            "meas_global",
            "baseline_global",
            "center_global_meas_field",
            "global_fields",
        ):
            if hasattr(cfg, name):
                try:
                    return bool(getattr(cfg, name))
                except Exception:
                    pass
    return False

def run_tost(
    df: pd.DataFrame,
    *,
    y: str,
    margins: list[float],
    alpha: float = 0.05,
    engine: str = "iid",
    cluster: Optional[str] = None,
    time: Optional[str] = None,
    x: Optional[str] = None,
    ycoord: Optional[str] = None,
    spatial_config: SpatialConfig | None = None,
    spatiotemporal_config: SpatioTemporalConfig | None = None,
    options: WorkflowOptions | None = None,
) -> dict[str, Any]:
    """
    Execute a chosen TOST engine and optional sensitivity analyses.

    Parameters
    ----------
    df : DataFrame
        Analysis dataset.
    y : str
        Response column (paired difference).
    margins : list[float]
        Equivalence margins.
    alpha : float
        One-sided alpha for TOST CI-inclusion rule.
    engine : {"iid","cluster","temporal","spatial","spatiotemporal"}
        Primary engine to run.
    cluster, time, x, ycoord : str or None
        Required depending on engine.
    spatial_config, spatiotemporal_config
        Settings for spatial/spatiotemporal engines.
    options : WorkflowOptions
        Controls sensitivity analyses.

    Returns
    -------
    dict with keys:
      - engine
      - primary (DataFrame)
      - sensitivity (dict[str, DataFrame]) if enabled
      - bootstrap (dict) if enabled and cluster provided
    """
    options = options or WorkflowOptions()
    eng = engine.lower().strip()

    if eng == "iid":
        primary = IIDTOST(y).fit(df, alpha, margins)

    elif eng == "cluster":
        if not cluster:
            raise ValueError("engine='cluster' requires `cluster` column name.")
        primary = ClusterTOST(y, cluster).fit(df, alpha, margins)

    elif eng == "temporal":
        if not time:
            raise ValueError("engine='temporal' requires `time` column name.")
        primary = TemporalTOST(y, time).fit(df, alpha, margins)

    elif eng == "spatial":
        if not (cluster and x and ycoord):
            raise ValueError("engine='spatial' requires `cluster`, `x`, and `ycoord` column names.")
        primary = SpatialTOST(
            y=y, cluster=cluster, x=x, ycoord=ycoord, config=(spatial_config or SpatialConfig())
        ).fit(df, alpha, margins)

    elif eng == "spatiotemporal":
        if not (cluster and time and x and ycoord):
            raise ValueError("engine='spatiotemporal' requires `cluster`, `time`, `x`, and `ycoord` column names.")
        primary = SpatioTemporalTOST(
            y=y,
            cluster=cluster,
            time=time,
            x=x,
            ycoord=ycoord,
            config=(spatiotemporal_config or SpatioTemporalConfig()),
        ).fit(df, alpha, margins)

    else:
        raise ValueError(f"Unknown engine={engine!r}. Must be one of iid/cluster/temporal/spatial/spatiotemporal.")

    out: dict[str, Any] = {"engine": eng, "primary": primary}

    # Optional sensitivity analyses (do not alter primary result)
    if options.do_sensitivity:
        sens: dict[str, pd.DataFrame] = {}
        sens["Heteroskedastic"] = HeteroskedasticTOST(y=y, cluster=cluster).fit(df, alpha=alpha, margins=margins)
        sens["Robust Location"] = RobustLocationTOST(y=y, cluster=cluster, time=time, block_len=5, B=2000, seed=options.seed).fit(
            df, alpha=alpha, margins=margins
        )
        out["sensitivity"] = sens

    # Optional bootstrap sanity check (cluster bootstrap for mean)
    if cluster and cluster in df.columns and options.bootstrap_B > 0:
        stat = lambda d_: d_[y].mean()
        out["bootstrap"] = cluster_bootstrap(df, y, cluster, stat, B=options.bootstrap_B, seed=options.seed)

    return out

"""Workflow orchestration for pyTOST.

This module provides the main workflow entry point for dependence-aware
Two One-Sided Tests (TOST) across IID, clustered, temporal, spatial, and
spatiotemporal settings. It also defines configuration options for optional
sensitivity analyses and validation bootstrap procedures.
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
from .bootstrap import cluster_bootstrap, spatial_block_bootstrap, spatial_within_building_block_bootstrap


@dataclass(frozen=True)
class WorkflowOptions:
    """Configuration options for workflow sensitivity analyses and bootstrap checks.

    Attributes
    ----------
    do_sensitivity : bool, default=True
        Whether to compute optional sensitivity analyses in addition to the
        primary engine result.
    bootstrap_B : int, default=200
        Number of bootstrap replicates used for the optional validation
        bootstrap.
    seed : int, default=42
        Random seed used for bootstrap-based procedures.
    robust_location_B : int, default=200
        Number of bootstrap replicates used by the robust-location
        sensitivity analysis.
    robust_location_block_len : int, default=5
        Block length used for time-dependent robust-location bootstrap
        procedures.
    robust_location_stat : str, default="median"
        Robust location statistic used in the robust-location sensitivity
        analysis.
    cross_building_dependence : bool or None, default=None
        Override controlling whether bootstrap validation for spatial or
        spatiotemporal analyses should assume dependence across clusters.
        If ``None``, the workflow attempts to infer this from the supplied
        configuration object.
    spatial_block_size : float, default=1.0
        Block size, in coordinate units, used by spatial block bootstrap
        procedures.
    """

    do_sensitivity: bool = True
    bootstrap_B: int = 200
    seed: int = 42

    # Sensitivity-analysis controls
    robust_location_B: int = 200
    robust_location_block_len: int = 5
    robust_location_stat: str = "median"

    # For spatial/spatiotemporal bootstrap selection only:
    #   - True  -> use spatial block bootstrap (allows cross-cluster dependence)
    #   - False -> use cluster bootstrap over the grouping column
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
    """Infer whether bootstrap validation should assume cross-cluster dependence.

    Parameters
    ----------
    options : WorkflowOptions
        Workflow options that may explicitly specify whether dependence
        across clusters should be assumed.
    spatial_config : object, optional
        Spatial configuration object that may contain flags describing
        cross-cluster dependence.
    spatiotemporal_config : object, optional
        Spatiotemporal configuration object that may contain flags
        describing cross-cluster dependence.

    Returns
    -------
    bool
        ``True`` if cross-cluster dependence should be assumed for
        bootstrap selection and ``False`` otherwise.

    Notes
    -----
    The inference precedence is:

    1. ``options.cross_building_dependence`` when explicitly provided.
    2. Recognized dependence flags on the spatial or spatiotemporal
       configuration object.
    3. ``False`` when no explicit signal is available.
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
    """Run dependence-aware equivalence testing and optional validation analyses.

    Parameters
    ----------
    df : pandas.DataFrame
        Input analysis table containing paired differences and any columns
        required by the selected engine.
    y : str
        Name of the column containing paired differences.
    margins : list of float
        Equivalence margins to evaluate.
    alpha : float, default=0.05
        One-sided significance level used by the TOST confidence-interval
        inclusion rule.
    engine : {"iid", "cluster", "temporal", "spatial", "spatiotemporal"}, default="iid"
        Primary dependence-aware inference engine to run.
    cluster : str or None, optional
        Name of the grouping column used by clustered, spatial, and
        spatiotemporal workflows.
    time : str or None, optional
        Name of the time-index column used by temporal and spatiotemporal
        workflows.
    x : str or None, optional
        Name of the x-coordinate column used by spatial and spatiotemporal
        workflows.
    ycoord : str or None, optional
        Name of the y-coordinate column used by spatial and spatiotemporal
        workflows.
    spatial_config : SpatialConfig or None, optional
        Configuration object used when ``engine="spatial"``.
    spatiotemporal_config : SpatioTemporalConfig or None, optional
        Configuration object used when ``engine="spatiotemporal"``.
    options : WorkflowOptions or None, optional
        Workflow options controlling sensitivity analyses and validation
        bootstrap procedures. If ``None``, default options are used.

    Returns
    -------
    dict of str to object
        Dictionary containing the selected engine label under ``"engine"``
        and the primary result table under ``"primary"``. When enabled,
        additional entries may include ``"sensitivity"`` for optional
        sensitivity-analysis results and ``"bootstrap"`` for validation
        bootstrap output.

    Raises
    ------
    ValueError
        If the selected engine is unknown or the required column names for
        that engine are not supplied.

    Notes
    -----
    The primary result is not altered by optional sensitivity analyses or
    bootstrap validation. These procedures are returned as additional
    outputs intended to help assess robustness of the equivalence decision.
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
        sens["Robust Location"] = RobustLocationTOST(
            y=y,
            cluster=cluster,
            time=time,
            block_len=options.robust_location_block_len,
            B=options.robust_location_B,
            seed=options.seed,
            stat=options.robust_location_stat,
        ).fit(df, alpha=alpha, margins=margins)
        out["sensitivity"] = sens

    # Optional bootstrap sanity check for the mean (validation CI)
    #
    # By default we use a cluster bootstrap over `cluster` (for example, sites or other grouped units).
    # For spatial/spatiotemporal engines, if the configuration indicates *cross-cluster*
    # dependence (e.g., a global field), we instead use a simple spatial block bootstrap
    # over cluster centroids to avoid understating uncertainty.
    if cluster and cluster in df.columns and options.bootstrap_B > 0:
        stat = lambda d_: d_[y].mean()

        if eng in {"spatial", "spatiotemporal"} and x and ycoord:
            use_spatial_blocks = _infer_cross_building_dependence(
                options=options,
                spatial_config=spatial_config,
                spatiotemporal_config=spatiotemporal_config,
            )

            # Spatial engine: validate within-cluster spatial dependence with a within-cluster block bootstrap.
            # If the user indicates cross-cluster dependence, switch to a block bootstrap over cluster centroids.
            if eng == "spatial" and not use_spatial_blocks:
                bb = spatial_within_building_block_bootstrap(
                    df,
                    y=y,
                    building_col=cluster,
                    x_col=x,
                    y_col=ycoord,
                    fit_fn=stat,
                    B=options.bootstrap_B,
                    seed=options.seed,
                    block_size=options.spatial_block_size,
                )
                bb["method"] = "spatial_within_building_block_bootstrap"
                out["bootstrap"] = bb

            else:
                # Cross-cluster dependence (or spatiotemporal): bootstrap blocks over cluster centroids.
                if use_spatial_blocks:
                    bb = spatial_block_bootstrap(
                        df,
                        y=y,
                        building_col=cluster,
                        x_col=x,
                        y_col=ycoord,
                        fit_fn=stat,
                        B=options.bootstrap_B,
                        seed=options.seed,
                        block_size=options.spatial_block_size,
                    )
                    bb["method"] = "spatial_block_bootstrap"
                    out["bootstrap"] = bb
                else:
                    bb = cluster_bootstrap(df, y, cluster, stat, B=options.bootstrap_B, seed=options.seed)
                    bb["method"] = "cluster_bootstrap"
                    out["bootstrap"] = bb
        else:
            bb = cluster_bootstrap(df, y, cluster, stat, B=options.bootstrap_B, seed=options.seed)
            bb["method"] = "cluster_bootstrap"
            out["bootstrap"] = bb

    return out

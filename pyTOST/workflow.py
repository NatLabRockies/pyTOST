"""
workflow.py
===========

Orchestration for pyTOST engines. Choose from IID, cluster, spatial, temporal, or spatiotemporal engines.

Sensitivity analysis
--------------------
The workflow can optionally compute:
  - bootstrap CI for the mean (sanity check)
  - heteroskedastic robust CI
  - robust-location CI (median/trimmed mean)

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Union

import pandas as pd

from .engines.iid_tost import IIDTOST
from .engines.cluster_tost import ClusterTOST
from .engines.temporal_tost import TemporalTOST
from .engines.spatial_tost import SpatialTOST, SpatialConfig
from .engines.spatiotemporal_tost import SpatioTemporalTOST, SpatioTemporalConfig

from .engines.heteroskedastic_tost import HeteroskedasticTOST
from .engines.robust_location_tost import RobustLocationTOST
from .bootstrap import cluster_bootstrap, spatial_block_bootstrap, spatial_within_cluster_block_bootstrap
from .results import TOSTResult


@dataclass(frozen=True)
class WorkflowOptions:
    """Options controlling optional sensitivity analyses and bootstrap selection."""

    do_sensitivity: bool = True
    bootstrap_B: int = 200
    seed: int = 42

    # Sensitivity-analysis controls
    robust_location_B: int = 200
    robust_location_block_len: int = 5
    robust_location_stat: str = "median"
    robust_location_trim: float = 0.2

    # For spatial/spatiotemporal bootstrap selection only:
    #   - True  -> use spatial block bootstrap (allows cross-cluster dependence)
    #   - False -> use cluster bootstrap over the grouping column
    #   - None  -> infer from config if possible, else default False
    cross_cluster_dependence: Optional[bool] = None

    # Block size (same units as x/y) for spatial block bootstrap
    spatial_block_size: float = 1.0

    # Newey–West truncation lag for the temporal engine.
    #   - None   -> use the engine default (4)
    #   - int    -> fix the truncation lag
    #   - "auto" -> select from the sample size via auto_hac_lags()
    max_lag: Optional[Union[int, str]] = None

def _infer_cross_cluster_dependence(
    *,
    options: WorkflowOptions,
    spatial_config: Optional[Any] = None,
    spatiotemporal_config: Optional[Any] = None,
) -> bool:
    """Infer cross-cluster dependence for bootstrap selection.

    Precedence:
      1) options.cross_cluster_dependence if not None
      2) config flags if present (e.g., meas_global / baseline_global)
      3) default False
    """
    if options.cross_cluster_dependence is not None:
        return bool(options.cross_cluster_dependence)

    cfgs = [c for c in (spatial_config, spatiotemporal_config) if c is not None]
    for cfg in cfgs:
        for name in (
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
) -> TOSTResult:
    """
    Execute a chosen TOST engine and optional sensitivity analyses.

    Applies the two one-sided tests (TOST) framework to paired differences. For each
    equivalence margin ``Δ``, the selected engine estimates the mean paired difference
    ``μ`` and forms a ``(1 - 2 * alpha)`` confidence interval. Equivalence is declared
    when that interval lies entirely inside ``(-Δ, Δ)``. The engines differ only in how
    they quantify uncertainty; choosing one that matches the dependence structure of
    the data is the central decision, because understating dependence produces
    intervals that are too narrow and can reverse the conclusion.

    Parameters
    ----------
    df : pandas.DataFrame
        Analysis dataset in long form. One row per observation.
    y : str
        Name of the response column holding the paired difference.
    margins : list of float
        Equivalence margins ``Δ``, in the units of ``y``. One result row is produced
        per margin. Margins should be chosen on domain grounds before the analysis.
    alpha : float, default 0.05
        One-sided significance level for the TOST CI-inclusion rule. The reported
        interval has nominal coverage ``1 - 2 * alpha`` (so 90% at the default).
    engine : {"iid", "cluster", "temporal", "spatial", "spatiotemporal", \
"heteroskedastic"}, default "iid"
        Primary inference engine, matched to the assumed dependence structure:

        - ``"iid"``: independent observations; intercept-only OLS with a Student-t CI.
        - ``"cluster"``: dependence within clusters, independence across them;
          cluster-robust sandwich variance with ``df = G - 1``. Requires ``cluster``.
        - ``"temporal"``: serial correlation; Newey–West HAC variance with a normal
          critical value. Requires ``time``. See ``WorkflowOptions.max_lag``.
        - ``"spatial"``: within-cluster spatial correlation; Matérn GLS with a profile
          likelihood-ratio CI. Requires ``cluster``, ``x``, and ``ycoord``.
        - ``"spatiotemporal"``: joint space-time dependence; separable AR(1) ⊗ Matérn
          model with a parametric-bootstrap CI for balanced panels, and an
          inverse-variance-weighted per-time fallback otherwise. Requires ``cluster``,
          ``time``, ``x``, and ``ycoord``.
        - ``"heteroskedastic"``: non-constant variance; HC3 robust inference, or
          cluster-robust wild bootstrap inference when ``cluster`` is supplied.

    cluster : str or None
        Grouping column (for example a site or neighborhood identifier). Required by
        the cluster, spatial, and spatiotemporal engines; optional for the
        heteroskedastic engine. When present, it also selects the bootstrap scheme.
    time : str or None
        Time-ordering column. Required by the temporal and spatiotemporal engines.
    x, ycoord : str or None
        Spatial coordinate columns, in consistent units. Required by the spatial and
        spatiotemporal engines. Both must be finite; non-finite values raise
        ``ValueError`` naming the offending column and rows.
    spatial_config : SpatialConfig or None
        Settings for the spatial engine, including the Matérn smoothness grid and the
        choice of point estimator. Defaults to ``SpatialConfig()``.
    spatiotemporal_config : SpatioTemporalConfig or None
        Settings for the spatiotemporal engine, including whether a balanced panel is
        required. Defaults to ``SpatioTemporalConfig()``.
    options : WorkflowOptions or None
        Controls sensitivity analyses, bootstrap replicates and scheme, the random
        seed, and the temporal HAC lag. Defaults to ``WorkflowOptions()``.

    Returns
    -------
    TOSTResult
        A ``dict`` subclass, so ordinary key access works unchanged. Keys:

        ``engine`` : str
            The normalized engine name that was run.
        ``primary`` : pandas.DataFrame
            One row per margin, with columns ``delta`` (the margin), ``mu_hat`` (the
            estimated mean paired difference), ``ci_low`` and ``ci_high`` (interval
            bounds), ``equivalent`` (bool; whether the interval lies inside the
            margins), and ``method`` (a description of the estimator). Some engines
            also report ``df``.
        ``sensitivity`` : dict of str to pandas.DataFrame, optional
            Present when ``options.do_sensitivity`` is True. Maps analysis name
            (``"Heteroskedastic"``, ``"Robust Location"``) to a table with the same
            schema as ``primary``.
        ``bootstrap`` : dict, optional
            Present when ``cluster`` is supplied and ``options.bootstrap_B > 0``.
            Contains ``B``, ``samples``, ``method``, and one or more interval keys
            such as ``ci_perc_90``.

        Call :meth:`TOSTResult.summary` for a human-readable decision table.

    Raises
    ------
    ValueError
        If ``engine`` is unknown, if a column required by the selected engine is
        missing, or if coordinate columns contain non-finite values.

    Notes
    -----
    Sensitivity analyses never alter the primary result. They are run to reveal
    whether the equivalence decision is fragile: if the primary engine and the
    sensitivity analyses disagree, the decision depends on modeling assumptions and
    should not be treated as established.

    See Also
    --------
    TOSTResult.summary : Human-readable decision table.
    pyTOST.plot_ci : Report-ready figure comparing intervals across engines.

    Examples
    --------
    A clustered analysis at two margins, with sensitivity analyses enabled:

    >>> import pandas as pd
    >>> from pyTOST import run_tost, WorkflowOptions
    >>> df = pd.DataFrame({
    ...     "diff": [0.10, 0.18, 0.05, 0.12, 0.08, 0.15],
    ...     "cluster_id": ["A", "A", "B", "B", "C", "C"],
    ... })
    >>> res = run_tost(
    ...     df,
    ...     y="diff",
    ...     margins=[0.25, 0.5],
    ...     alpha=0.05,
    ...     engine="cluster",
    ...     cluster="cluster_id",
    ...     options=WorkflowOptions(do_sensitivity=False, bootstrap_B=0),
    ... )
    >>> res["engine"]
    'cluster'
    >>> bool(res["primary"].iloc[0]["equivalent"])
    True

    The result is still a plain mapping, and ``summary()`` formats the decision:

    >>> sorted(res.keys())
    ['engine', 'primary']
    >>> print(res.summary())  # doctest: +SKIP
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
        temporal_kwargs = {} if options.max_lag is None else {"hac_lags": options.max_lag}
        primary = TemporalTOST(y, time, **temporal_kwargs).fit(df, alpha, margins)

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

    elif eng == "heteroskedastic":
        primary = HeteroskedasticTOST(
            y=y, cluster=cluster, seed=options.seed
        ).fit(df, alpha=alpha, margins=margins)

    else:
        raise ValueError(
            f"Unknown engine={engine!r}. Must be one of "
            "iid/cluster/temporal/spatial/spatiotemporal/heteroskedastic."
        )

    out: TOSTResult = TOSTResult({"engine": eng, "primary": primary})

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
            trim=options.robust_location_trim,
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
            use_spatial_blocks = _infer_cross_cluster_dependence(
                options=options,
                spatial_config=spatial_config,
                spatiotemporal_config=spatiotemporal_config,
            )

            # Spatial engine: validate within-cluster spatial dependence with a within-cluster block bootstrap.
            # If the user indicates cross-cluster dependence, switch to a block bootstrap over cluster centroids.
            if eng == "spatial" and not use_spatial_blocks:
                bb = spatial_within_cluster_block_bootstrap(
                    df,
                    y=y,
                    cluster_col=cluster,
                    x_col=x,
                    y_col=ycoord,
                    fit_fn=stat,
                    B=options.bootstrap_B,
                    seed=options.seed,
                    block_size=options.spatial_block_size,
                )
                bb["method"] = "spatial_within_cluster_block_bootstrap"
                out["bootstrap"] = bb

            else:
                # Cross-cluster dependence (or spatiotemporal): bootstrap blocks over cluster centroids.
                if use_spatial_blocks:
                    bb = spatial_block_bootstrap(
                        df,
                        y=y,
                        cluster_col=cluster,
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

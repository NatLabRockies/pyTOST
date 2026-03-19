"""pyTOST package public API.

This package provides equivalence testing (TOST) for paired differences under
multiple dependence structures. Users choose
the appropriate engine (IID / cluster / temporal / spatial / spatio-temporal)
based on their data-generating process and study design.

Primary entry point
-------------------
`run_tost(...)` in `workflow.py` executes a selected engine and (optionally)
adds sensitivity analyses and a cluster bootstrap sanity check.

Engines
-------
- IIDTOST: classic t-based CI for the mean difference (i.i.d. residuals).
- ClusterTOST: valid inference with intra-cluster correlation (e.g., buildings).
- TemporalTOST: valid inference with serial correlation (e.g., time series).
- SpatialTOST: grade Matérn GLS (REML) + likelihood-ratio CI.
- SpatioTemporalTOST: per-time spatial fits aggregated via IVW.

Sensitivity analyses
--------------------
- HeteroskedasticTOST: robust (HC / wild bootstrap) CI for mean difference.
- RobustLocationTOST: bootstrap CI for robust location (median / trimmed mean).

"""

from __future__ import annotations

# Workflow (strict, no plug-ins / no fallbacks)
from .workflow import run_tost, WorkflowOptions

# Core engines
from .engines.iid_tost import IIDTOST
from .engines.cluster_tost import ClusterTOST
from .engines.temporal_tost import TemporalTOST

# Publication-grade spatial engines
from .engines.spatial_tost import SpatialTOST, SpatialConfig
from .engines.spatiotemporal_tost import SpatioTemporalTOST, SpatioTemporalConfig

# Sensitivity engines
from .engines.heteroskedastic_tost import HeteroskedasticTOST
from .engines.robust_location_tost import RobustLocationTOST

__all__ = [
    # workflow
    "run_tost",
    "WorkflowOptions",
    # engines
    "IIDTOST",
    "ClusterTOST",
    "TemporalTOST",
    "SpatialTOST",
    "SpatialConfig",
    "SpatioTemporalTOST",
    "SpatioTemporalConfig",
    # sensitivity
    "HeteroskedasticTOST",
    "RobustLocationTOST",
]

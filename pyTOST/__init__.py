"""Public API for the pyTOST package.

pyTOST provides dependence-aware equivalence testing for paired differences
using the two one-sided tests (TOST) framework. The package exposes a
workflow-oriented entry point, dependence-specific inference engines, and
validation-oriented sensitivity-analysis engines.

Notes
-----
The recommended user-facing entry point is :func:`pyTOST.run_tost`, which
applies the selected inference engine and can optionally add sensitivity
analyses and bootstrap-based validation summaries.

Available public objects include:

- ``run_tost`` and ``WorkflowOptions`` for the top-level workflow.
- ``IIDTOST``, ``ClusterTOST``, ``TemporalTOST``, ``SpatialTOST``, and
  ``SpatioTemporalTOST`` for primary inference.
- ``SpatialConfig`` and ``SpatioTemporalConfig`` for spatial and
  spatiotemporal configuration.
- ``HeteroskedasticTOST`` and ``RobustLocationTOST`` for sensitivity
  analyses.
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

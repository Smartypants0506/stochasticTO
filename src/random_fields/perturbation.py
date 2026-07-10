"""Explicit geometry/mesh perturbation -- validation/visualization ONLY.

Master-context alignment (Section 3.2, Section 7 "Do NOT"):
    "Explicit geometry/mesh perturbation is generated only for validation and
    final visualization ensembles -- it is NOT the primary manufacturing-error
    modeling device." The primary mechanism is randomization of eta(x) via
    kl_expansion.py + threshold_transform.py.

This module is intentionally a stub for the MVP: mesh warping is deferred to
Stage 6 (Monte Carlo validation / visualization), which is out of scope until
the robust TO loop (Step 7 in the MVP roadmap) is complete.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def warp_mesh_from_eta_field(
    node_coordinates: np.ndarray,
    eta_field: np.ndarray,
    warp_scale: float = 0.0,
) -> np.ndarray:
    """Placeholder for Stage 6 visualization-only mesh warping.

    Explicitly NOT used by the robust optimization loop (Section 3.2, 3.5).
    Only invoked later by src/viz/probability_cloud.py and
    src/validation/monte_carlo.py.

    Args:
        node_coordinates: [N_nodes x spatial_dim] nominal mesh coordinates.
        eta_field: [N_nodes] sampled eta(x) realization driving the warp.
        warp_scale: Visualization-only displacement scale factor; 0.0 disables
            warping entirely (default, since this is not yet wired into any
            pipeline stage).

    Returns:
        Perturbed node coordinates (identical to input when warp_scale=0.0).
    """
    if warp_scale == 0.0:
        logger.debug("warp_mesh_from_eta_field called with warp_scale=0.0 -- no-op.")
        return node_coordinates.copy()
    raise NotImplementedError(
        "Mesh warping is a Stage 6 (validation/visualization) feature, not yet "
        "implemented. See master-context Section 3.2 and 4."
    )
"""
src/surrogate/sobol.py

Stage 5 (PCE Surrogate Construction) -- implementation-modules.md Item 13 /
fileDescription.md src/surrogate/sobol.py.

Computes first-order and total Sobol sensitivity indices analytically from
a fitted PCE (no extra FEA solves), using OpenTURNS's own
FunctionalChaosSobolIndices class, which derives indices directly from the
already-fitted PCE coefficients per the orthogonal-basis variance
decomposition. Identifies which KL modes drive compliance variance and
whether n_kl was truncated correctly (>= 99% cumulative variance target).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import openturns as ot

from src.surrogate.pce_builder import PCEBuildResult

logger = logging.getLogger(__name__)

CUMULATIVE_VARIANCE_TARGET = 0.99


@dataclass
class SobolReport:
    """First-order and total Sobol indices for each KL mode.

    Attributes:
        first_order: [n_kl] first-order index S_i for each KL mode.
        total_order: [n_kl] total-order index S_i^T for each KL mode
            (includes interaction effects with other modes).
        n_kl: Input dimensionality (number of KL modes).
        n_kl_effective: Smallest number of modes (ranked by first-order
            index, descending) whose cumulative sum reaches
            CUMULATIVE_VARIANCE_TARGET -- a candidate for re-truncating
            the KL expansion if n_kl_effective < n_kl.
    """
    first_order: np.ndarray
    total_order: np.ndarray
    n_kl: int
    n_kl_effective: int


def compute_sobol_indices(pce_result: PCEBuildResult) -> SobolReport:
    """Compute first-order and total Sobol indices from a fitted PCE.

    Args:
        pce_result: Output of pce_builder.build_pce_surrogate (should have
            already passed the Q^2 >= 0.99 gate -- Sobol indices from an
            under-fit surrogate are not meaningful, though this function
            does not re-check that gate itself).

    Returns:
        A SobolReport with per-mode first-order/total indices and the
        effective truncation dimension.

    Raises:
        ValueError: If the chaos result's input dimension does not match
            pce_result.n_kl (defensive check against a mismatched object).
    """
    sobol = ot.FunctionalChaosSobolIndices(pce_result.chaos_result)
    n_kl = pce_result.n_kl

    first_order = np.array([sobol.getSobolIndex(i) for i in range(n_kl)])
    total_order = np.array([sobol.getSobolTotalIndex(i) for i in range(n_kl)])

    if first_order.size != n_kl or total_order.size != n_kl:
        raise ValueError(
            f"Sobol index count ({first_order.size}) does not match "
            f"pce_result.n_kl ({n_kl}); chaos_result may not correspond "
            "to this PCEBuildResult."
        )

    order = np.argsort(first_order)[::-1]
    cumulative = np.cumsum(first_order[order])
    above_target = np.where(cumulative >= CUMULATIVE_VARIANCE_TARGET)[0]
    n_kl_effective = int(above_target[0] + 1) if above_target.size > 0 else n_kl

    logger.info(
        "Sobol indices computed: n_kl=%d, n_kl_effective=%d (%.1f%% cumulative "
        "first-order variance), top mode S_1=%.4g",
        n_kl, n_kl_effective, cumulative[min(n_kl_effective, n_kl) - 1] * 100,
        first_order[order[0]] if n_kl > 0 else float("nan"),
    )

    return SobolReport(
        first_order=first_order,
        total_order=total_order,
        n_kl=n_kl,
        n_kl_effective=n_kl_effective,
    )
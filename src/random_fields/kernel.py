"""Covariance kernel construction for the manufacturing-error random field.

Master-context alignment (Section 3.3, Section 7):
    k(x, x') = sigma^2 * exp(-||x - x'||^2 / (2 * l^2))

For the MVP, kernel parameters (sigma, correlation length l) are supplied
synthetically rather than fit from metrology data. The `fit_kernel_from_data`
stub is left in place so Stage 3's real-data path can be dropped in later
without changing the KL-expansion or threshold-transform interfaces.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import openturns as ot

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class KernelParams:
    """Squared-exponential covariance kernel parameters.

    Attributes:
        sigma: Marginal standard deviation of the underlying Gaussian field (dimensionless).
        length_scale: Spatial correlation length l, in the DOMAIN's own units.
            The beam_3d case (src/meshing/box_source.py) is dimensionless --
            domain 10 x 30 x 10, E = 100 -- so this is NOT metres there. An
            earlier version of config.yaml carried length_scale: 4 alongside
            docstrings claiming metres and a log line claiming millimetres;
            quote it as a fraction of the domain instead.
        spatial_dim: Dimensionality of the domain (2 or 3).
    """
    sigma: float
    length_scale: float
    spatial_dim: int = 2

    def __post_init__(self) -> None:
        if self.sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {self.sigma}")
        if self.length_scale <= 0:
            raise ValueError(f"length_scale must be > 0, got {self.length_scale}")
        if self.spatial_dim not in (2, 3):
            raise ValueError(f"spatial_dim must be 2 or 3, got {self.spatial_dim}")


def build_squared_exponential(params: KernelParams) -> ot.CovarianceModel:
    """Build an OpenTURNS squared-exponential covariance model.

    Implements k(x, x') = sigma^2 * exp(-||x - x'||^2 / (2 * l^2)) exactly as
    specified in master-context Section 3.3 / Section 7 ("Exact Mathematical
    Formulations — Do Not Deviate").

    Args:
        params: Kernel hyperparameters (sigma, length_scale, spatial_dim).

    Returns:
        An ot.SquaredExponential covariance model scaled by sigma^2.
    """
    # ot.SquaredExponential takes scale = [l]*dim and amplitude = [sigma]
    model = ot.SquaredExponential([params.length_scale] * params.spatial_dim, [params.sigma])
    logger.info(
        "Built squared-exponential kernel: sigma=%.4g, l=%.4g mm, dim=%d",
        params.sigma, params.length_scale, params.spatial_dim,
    )
    return model


def fit_kernel_from_data(
    deviation_points: np.ndarray,
    deviation_values: np.ndarray,
) -> KernelParams:
    """Fit kernel hyperparameters from Open3D-registered metrology deviations.

    NOT part of the MVP. Reserved for Stage 3's real-data path (Section 3.3:
    "maximum likelihood or variogram analysis"). Raises NotImplementedError
    until the Open3D registration.py module (Section 5.2) is built.

    Args:
        deviation_points: [N x spatial_dim] node coordinates.
        deviation_values: [N] scalar deviation field values at those nodes.

    Returns:
        Fitted KernelParams.
    """
    raise NotImplementedError(
        "Kernel fitting from metrology data requires src/metrology/registration.py "
        "(Open3D colored ICP) to be implemented first. See master-context Section 3.3."
    )
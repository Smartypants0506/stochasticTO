"""Memoryless (isoprobabilistic) marginal transform for the projection threshold.

Master-context alignment (Section 3.3, Section 7):
    eta(x) = T(G(x))

This is explicitly flagged as fully custom -- "no prebuilt package implements
this specific transform" -- and is the single most novel piece of the entire
project. For the MVP, the target marginal is a synthetic bounded Beta
distribution on [eta_min, eta_max] (calibration from real metrology data is
deferred; see fit_marginal_from_data stub).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import openturns as ot
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MarginalTransformParams:
    """Parameters of the target bounded marginal distribution for eta(x).

    Attributes:
        eta_min: Lower physical bound of the projection threshold (Section 3.3
            requires a "bounded, non-Gaussian marginal").
        eta_max: Upper physical bound of the projection threshold.
        alpha: Beta distribution shape parameter alpha (controls skew/peak).
        beta: Beta distribution shape parameter beta.
    """
    eta_min: float = 0.3
    eta_max: float = 0.7
    alpha: float = 2.0
    beta: float = 2.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.eta_min < self.eta_max <= 1.0):
            raise ValueError(
                f"Require 0 <= eta_min < eta_max <= 1, got eta_min={self.eta_min}, "
                f"eta_max={self.eta_max}"
            )
        if self.alpha <= 0 or self.beta <= 0:
            raise ValueError("Beta distribution shape parameters must be > 0.")


class ThresholdMarginalTransform:
    """Applies T(.) mapping standard-normal field values to bounded eta values.

    The transform is the classical isoprobabilistic (memoryless) mapping:
        eta(x) = F_eta^{-1}( Phi(G(x)) )
    where Phi is the standard normal CDF and F_eta is the CDF of the target
    bounded marginal (Beta on [eta_min, eta_max] for the MVP). This is
    "memoryless" because it acts pointwise on G(x) with no dependence on
    neighboring field values -- exactly matching Section 3.3's specification.
    """

    def __init__(self, params: MarginalTransformParams):
        self.params = params
        beta_std = ot.Beta(params.alpha, params.beta, 0.0, 1.0)
        self._target_marginal = ot.CompositeDistribution(
            ot.SymbolicFunction("x", f"{params.eta_min} + ({params.eta_max}-{params.eta_min})*x"),
            beta_std,
        )
        self._standard_normal = ot.Normal(0.0, 1.0)
        logger.info(
            "Initialized threshold marginal transform: eta in [%.3f, %.3f], "
            "Beta(alpha=%.2f, beta=%.2f)",
            params.eta_min, params.eta_max, params.alpha, params.beta,
        )

    def transform(self, gaussian_field_values: np.ndarray) -> np.ndarray:
        """Map G(x) realizations to eta(x) realizations, vectorized.

        Mathematically identical to the OpenTURNS-object-per-node version
        this replaces: Phi(G(x)) via the standard normal CDF, then the
        Beta(alpha, beta) quantile on [0,1], rescaled to [eta_min, eta_max].
        Implemented with scipy.stats (fully vectorized C loops) instead of
        looping in Python over ot.Distribution.computeCDF/computeQuantile
        calls per node -- that per-node Python/OT round-trip was costing
        ~1.7-1.9s per call on the full global mesh, repeated every training
        sample, and was the dominant cost in run_fea_at_samples().

        Args:
            gaussian_field_values: Array of any shape containing standard
                Gaussian field values G(x) (e.g. [n_samples x N_nodes]).

        Returns:
            Array of the same shape containing eta(x) in [eta_min, eta_max].
        """
        arr = np.asarray(gaussian_field_values, dtype=float)
        cdf_values = scipy_stats.norm.cdf(arr)
        cdf_values = np.clip(cdf_values, 1e-12, 1 - 1e-12)  # avoid quantile-tail blowup
        beta_quantiles = scipy_stats.beta.ppf(
            cdf_values, self.params.alpha, self.params.beta
        )
        return self.params.eta_min + (self.params.eta_max - self.params.eta_min) * beta_quantiles

    def validate_bounds(self, eta_values: np.ndarray) -> bool:
        """Sanity check that transformed values respect [eta_min, eta_max].

        Args:
            eta_values: Output of transform().

        Returns:
            True if all values lie within [eta_min, eta_max] (with 1e-9 slack).
        """
        eps = 1e-9
        within = np.all(
            (eta_values >= self.params.eta_min - eps)
            & (eta_values <= self.params.eta_max + eps)
        )
        if not within:
            logger.warning(
                "Transformed eta values out of bounds: min=%.4g, max=%.4g, "
                "expected [%.4g, %.4g]",
                eta_values.min(), eta_values.max(),
                self.params.eta_min, self.params.eta_max,
            )
        return bool(within)


def fit_marginal_from_data(deviation_values: np.ndarray) -> MarginalTransformParams:
    """Fit the target marginal from real Cp/Cpk / metrology deviation data.

    NOT part of the MVP. Reserved for calibration once
    src/metrology/process_stats.py (Cp/Cpk extraction) exists, per Section
    3.3: "calibrated from metrology data rather than assumed uniform."

    Args:
        deviation_values: Empirical deviation/threshold-equivalent samples.

    Returns:
        Fitted MarginalTransformParams.
    """
    raise NotImplementedError(
        "Marginal fitting from metrology data requires src/metrology/process_stats.py "
        "to be implemented first. See master-context Section 3.3 and 5.2."
    )
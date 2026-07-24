"""
src/surrogate/pce_model.py

Stage 5 (PCE Surrogate Construction) -- fileDescription.md src/surrogate/
section. Wraps a fitted PCEBuildResult (pce_builder.py) to provide the
analytic mean/variance/gradient interface that robust_objective.py and
robust_gradient.py currently compute via brute-force Monte Carlo -- their
docstrings explicitly flag that MC path as a temporary MVP stand-in to be
replaced once this module exists.

Design consistency requirement: dmu_C/drho and dsigma_C/drho are NOT
recomputed independently. They are obtained by projecting the SAME
per-sample adjoint gradients dC_i/drho (from fea_at_samples.py) onto the
identical sparse basis (same active multi-indices) that pce_builder.py's
LARS selection already fit for compliance itself. Using a different basis
or an independent regression here would make the gradient inconsistent
with the surrogate's own compliance predictions.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import openturns as ot

from src.surrogate.pce_builder import PCEBuildResult

logger = logging.getLogger(__name__)

_SIGMA_ZERO_EPS = 1e-14  # matches robust_gradient.py's degenerate-sigma guard


@dataclass
class PCEGradientModel:
    """Analytic mean/variance/gradient model built from a fitted PCE.

    The QoI *values* mu_C/sigma_C are taken from the PCE coefficients
    (masterContext Section 3.4: mu = c_0, sigma^2 = sum_{alpha!=0} c_alpha^2).
    The QoI *gradients*, however, are the DIRECT unbiased sample estimators
    (mean / centered-sample-covariance of the per-sample adjoint gradients
    dC_i/drho) rather than an lstsq projection of dC_i/drho onto the PCE
    basis. The direct estimators are exactly robust_gradient.py's
    compute_dmu_drho / compute_dsigma_drho (the TOuU stochastic-gradient
    form, masterContext Section 3.5) and are always sign-correct (dmu is an
    average of true adjoint gradients), whereas the lstsq projection could
    corrupt the constant-term row on a marginal PCE fit and flip the mean
    gradient's sign -- the cause of the observed volume-collapse divergence.

    Attributes:
        coefficients: [n_active] PCE coefficients c_alpha, orthonormal basis,
            with coefficients[0] the constant term (mu_C).
        mu_C: Analytic mean, equal to coefficients[0].
        sigma_C: Analytic standard deviation, sqrt(sum_{alpha!=0} c_alpha^2).
        dmu_drho_vec: [n_elems] direct estimate d(mu_C)/drho = mean_i(dC_i/drho).
        dsigma_drho_vec: [n_elems] direct estimate d(sigma_C)/drho, or None if
            sigma_C was numerically zero at build time (dsigma is undefined
            then; for the volume QoI it is never queried).
        n_kl: Input dimensionality (number of KL modes).
    """
    coefficients: np.ndarray
    mu_C: float
    sigma_C: float
    dmu_drho_vec: np.ndarray
    dsigma_drho_vec: np.ndarray | None
    n_kl: int
    active_kl_indices: np.ndarray | None = None
    chaos_result: ot.FunctionalChaosResult | None = None  # needed to evaluate
    # this PCE's metamodel outside this module (e.g. Stage 6's
    # compare_against_pce). Sourced from the PCEBuildResult this model was
    # built from.

    def dmu_drho(self) -> np.ndarray:
        """d(mu_C)/drho = mean_i(dC_i/drho), the direct unbiased sample estimator."""
        return self.dmu_drho_vec

    def dsigma_drho(self) -> np.ndarray:
        """d(sigma_C)/drho, the direct centered-sample estimator (see build).

        Raises:
            RuntimeError: If sigma_C was numerically zero at build time, so
                dsigma_C/drho is undefined (mirrors
                robust_gradient.compute_dsigma_drho's guard).
        """
        if self.dsigma_drho_vec is None:
            raise RuntimeError(
                f"sigma_C={self.sigma_C:.3g} is numerically zero -- cannot "
                "compute dsigma_C/drho (division by zero). This indicates "
                "the PCE found negligible eta(x)-driven variance at this "
                "design iterate."
            )
        return self.dsigma_drho_vec


def build_pce_gradient_model(
    pce_result: PCEBuildResult,
    xi_train: np.ndarray,
    dC_drho_train: np.ndarray,
    qoi_train: np.ndarray,
    active_kl_indices: np.ndarray | None = None,
) -> PCEGradientModel:
    """Build the analytic mean/variance/gradient model from a fitted PCE.

    The QoI VALUES mu_C/sigma_C come from the PCE coefficients (masterContext
    Section 3.4). The QoI GRADIENTS are the DIRECT unbiased sample estimators
    of dmu/drho and dsigma/drho (identical to robust_gradient.compute_dmu_drho
    / compute_dsigma_drho -- the TOuU stochastic-gradient form, Section 3.5):

        dmu/drho    = mean_i(dC_i/drho)
        dsigma/drho = (1/((N-1)*sigma_C)) * sum_i (C_i - Cbar) * dC_i/drho

    These are always sign-correct (dmu is an average of true adjoint
    gradients) and robust to PCE fit quality, unlike an lstsq projection of
    dC_i/drho onto the PCE basis, whose corrupted constant-term row on a
    marginal fit flipped the mean-gradient sign and drove the volume-collapse
    divergence.

    Args:
        pce_result: Output of pce_builder.build_pce_surrogate.
        xi_train: [n_train x n_kl] KL coefficients used to fit pce_result
            (kept for provenance / shape checks; not used by the direct
            gradient estimators).
        dC_drho_train: [n_train x n_elems] per-sample adjoint gradients
            dQoI_i/drho, collected by fea_at_samples.py alongside the QoI
            values used to fit pce_result.
        qoi_train: [n_train] per-sample QoI values (compliance C_i, or volume
            V_i) matching dC_drho_train row-for-row -- needed for the centered
            dsigma/drho estimator.

    Returns:
        A PCEGradientModel exposing mu_C, sigma_C, dmu_drho(), dsigma_drho().

    Raises:
        ValueError: If dC_drho_train / qoi_train sample counts do not match
            xi_train.
    """
    chaos_result = pce_result.chaos_result
    coefficients = np.array(chaos_result.getCoefficients()).ravel()

    # Verify orthonormal basis: OpenTURNS standard polynomial factories produce
    # orthonormal coefficients. Validate by checking the reduced basis norms.
    reduced_basis = chaos_result.getReducedBasis()
    test_point = ot.Point([0.0] * pce_result.n_kl)
    # Spot-check: first basis function at origin should equal 1.0 (constant term)
    phi0_val = float(reduced_basis[0](test_point)[0])
    if abs(phi0_val - 1.0) > 1e-8:
        raise RuntimeError(
            f"PCE basis function 0 at origin = {phi0_val}, expected 1.0 for "
            "orthonormal Hermite basis. Variance/gradient math requires orthonormality."
        )

    n_active = coefficients.size
    n_train = xi_train.shape[0]

    if dC_drho_train.shape[0] != n_train:
        raise ValueError(
            f"dC_drho_train has {dC_drho_train.shape[0]} rows but xi_train "
            f"has {n_train}; they must correspond to the same training "
            "samples in the same order."
        )
    if qoi_train.shape[0] != n_train:
        raise ValueError(
            f"qoi_train has {qoi_train.shape[0]} entries but xi_train has "
            f"{n_train}; they must correspond to the same training samples."
        )

    mu_C = float(coefficients[0])
    sigma_C = float(np.sqrt(np.sum(coefficients[1:] ** 2)))

    # --- direct sample-based gradient estimators (robust_gradient.py math) ---
    dmu_drho_vec = dC_drho_train.mean(axis=0)  # [n_elems]

    if sigma_C >= _SIGMA_ZERO_EPS and n_train > 1:
        centered_qoi = qoi_train - qoi_train.mean()          # [n_train]
        d_variance_drho = centered_qoi @ dC_drho_train        # [n_elems]
        dsigma_drho_vec = d_variance_drho / ((n_train - 1) * sigma_C)
    else:
        # Negligible variance (e.g. volume QoI) -- dsigma/drho is undefined and
        # never queried for that QoI. Leave None; dsigma_drho() will raise if
        # something does query it.
        dsigma_drho_vec = None

    logger.info(
        "PCE gradient model built: n_active=%d, mu_C=%.6g, sigma_C=%.6g "
        "(direct sample gradients over n_train=%d)",
        n_active, mu_C, sigma_C, n_train,
    )

    return PCEGradientModel(
        coefficients=coefficients,
        mu_C=mu_C,
        sigma_C=sigma_C,
        dmu_drho_vec=dmu_drho_vec,
        dsigma_drho_vec=dsigma_drho_vec,
        n_kl=pce_result.n_kl,
        active_kl_indices=active_kl_indices,
        chaos_result=chaos_result,
    )
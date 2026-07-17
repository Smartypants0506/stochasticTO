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

    Attributes:
        coefficients: [n_active] PCE coefficients c_alpha, orthonormal basis,
            with coefficients[0] the constant term (mu_C).
        mu_C: Analytic mean, equal to coefficients[0].
        sigma_C: Analytic standard deviation, sqrt(sum_{alpha!=0} c_alpha^2).
        dc_drho: [n_active x n_elems] gradient of each PCE coefficient
            w.r.t. the unfiltered design variable rho, obtained by
            projecting dC_i/drho training data onto the fixed active basis.
        n_kl: Input dimensionality (number of KL modes).
    """
    coefficients: np.ndarray
    mu_C: float
    sigma_C: float
    dc_drho: np.ndarray
    n_kl: int
    active_kl_indices: np.ndarray | None = None
    chaos_result: ot.FunctionalChaosResult | None = None  # NEW: needed to
    # evaluate this PCE's metamodel outside this module (e.g. Stage 6's
    # compare_against_pce). Sourced from the PCEBuildResult this model
    # was built from.

    def dmu_drho(self) -> np.ndarray:
        """d(mu_C)/drho = dc_alpha=0/drho -- the constant-term coefficient's gradient.

        Analytic counterpart to robust_gradient.compute_dmu_drho, which
        instead averages raw dC_i/drho over MC samples; this is the same
        quantity but sourced from the PCE's own coefficient projection.
        """
        return self.dc_drho[0, :]

    def dsigma_drho(self) -> np.ndarray:
        """d(sigma_C)/drho via the PCE coefficient-space identity.

        sigma_C^2 = sum_{alpha!=0} c_alpha^2 (orthonormal basis)
        => d(sigma_C^2)/drho = 2 * sum_{alpha!=0} c_alpha * dc_alpha/drho
        => d(sigma_C)/drho = d(sigma_C^2)/drho / (2*sigma_C)

        This is the exact coefficient-space analog of robust_gradient.py's
        sample-average identity (see module docstring derivation there);
        no additional approximation is introduced beyond the PCE fit itself.

        Raises:
            RuntimeError: If sigma_C is numerically zero (division by zero),
                mirroring robust_gradient.compute_dsigma_drho's guard.
        """
        if self.sigma_C < _SIGMA_ZERO_EPS:
            raise RuntimeError(
                f"sigma_C={self.sigma_C:.3g} is numerically zero -- cannot "
                "compute dsigma_C/drho (division by zero). This indicates "
                "the PCE found negligible eta(x)-driven variance at this "
                "design iterate."
            )
        c_nonconst = self.coefficients[1:]                # [n_active-1]
        dc_nonconst = self.dc_drho[1:, :]                  # [n_active-1 x n_elems]
        d_variance_drho = 2.0 * (c_nonconst @ dc_nonconst)  # [n_elems]
        return d_variance_drho / (2.0 * self.sigma_C)


def _evaluate_active_basis(chaos_result: ot.FunctionalChaosResult,
                            xi: np.ndarray) -> np.ndarray:
    """Evaluate the PCE's active (LARS-selected) basis functions at each xi row.

    Args:
        chaos_result: Fitted openturns.FunctionalChaosResult from pce_builder.py.
        xi: [n_samples x n_kl] KL coefficient matrix (same convention as
            sampler.py's SampleSet.xi).

    Returns:
        [n_samples x n_active] design matrix Psi, where column k is the
        k-th active basis polynomial evaluated at every sample -- same
        ordering as chaos_result.getCoefficients().
    """
    active_basis = chaos_result.getReducedBasis()
    n_active = len(active_basis)
    n_samples = xi.shape[0]
    psi = np.empty((n_samples, n_active))
    xi_sample = ot.Sample(xi)
    for k, phi_k in enumerate(active_basis):
        psi[:, k] = np.array(phi_k(xi_sample)).ravel()
    return psi


def build_pce_gradient_model(
    pce_result: PCEBuildResult,
    xi_train: np.ndarray,
    dC_drho_train: np.ndarray,
    active_kl_indices: np.ndarray | None = None,
) -> PCEGradientModel:
    """Build the analytic mean/variance/gradient model from a fitted PCE.

    Args:
        pce_result: Output of pce_builder.build_pce_surrogate (must have
            already passed the Q^2 >= 0.99 gate -- this function does not
            re-check that gate, callers must not bypass it).
        xi_train: [n_train x n_kl] KL coefficients used to fit pce_result
            (must be the exact same samples, in the same order, used in
            pce_builder's training call).
        dC_drho_train: [n_train x n_elems] per-sample adjoint gradients
            dC_i/drho, collected by fea_at_samples.py alongside the
            compliance values used to fit pce_result.

    Returns:
        A PCEGradientModel exposing mu_C, sigma_C, dmu_drho(), dsigma_drho().

    Raises:
        ValueError: If dC_drho_train's sample count does not match xi_train,
            or if the coefficient-space gradient regression is under-determined
            (n_train < n_active).
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

    if dC_drho_train.shape[0] != xi_train.shape[0]:
        raise ValueError(
            f"dC_drho_train has {dC_drho_train.shape[0]} rows but xi_train "
            f"has {xi_train.shape[0]}; they must correspond to the same "
            "training samples in the same order."
        )
    if xi_train.shape[0] < n_active:
        raise ValueError(
            f"n_train={xi_train.shape[0]} is smaller than n_active={n_active} "
            "active PCE basis terms; the gradient-coefficient regression is "
            "under-determined. Increase n_train or reduce hyperbolic_q."
        )

    psi = _evaluate_active_basis(chaos_result, xi_train)  # [n_train x n_active]

    # Project dC_i/drho onto the same fixed active basis via ordinary least
    # squares -- reuses the LARS-selected sparsity pattern from pce_builder.py
    # rather than re-selecting a basis independently for the gradient field.
    dc_drho, residuals, rank, _ = np.linalg.lstsq(psi, dC_drho_train, rcond=None)
    if rank < n_active:
        raise RuntimeError(
            f"Gradient-coefficient regression is rank-deficient (rank={rank}, "
            f"n_active={n_active}). dc_drho is unreliable -- increase n_train "
            f"or reduce hyperbolic_q to lower n_active."
        )

    mu_C = float(coefficients[0])
    sigma_C = float(np.sqrt(np.sum(coefficients[1:] ** 2)))

    logger.info(
        "PCE gradient model built: n_active=%d, mu_C=%.6g, sigma_C=%.6g",
        n_active, mu_C, sigma_C,
    )

    return PCEGradientModel(
        coefficients=coefficients,
        mu_C=mu_C,
        sigma_C=sigma_C,
        dc_drho=dc_drho,
        n_kl=pce_result.n_kl,
        active_kl_indices=active_kl_indices,
        chaos_result=chaos_result,
    )
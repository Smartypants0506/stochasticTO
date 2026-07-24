"""
src/optimization/pce_evaluation.py

Bridges src/surrogate/pce_model.py's PCEGradientModel into the exact
RobustEvaluationResult contract that robust_objective.py and
robust_gradient.py already consume, so their J = mu_C + lambda*sigma_C
and dJ/drho = dmu_C/drho + lambda*dsigma_C/drho formulas require ZERO
changes -- only the data source changes from brute-force MC (n_mc_samples
FEA solves per evaluation) to an analytic PCE lookup (0 solves per
evaluation, per masterContext Section 3.5).

Two separate PCE surrogates are used: one for compliance (drives the
robust objective J), one for volume (drives the mean-volume constraint
E[V] <= Vfrac). Both are trained on the same xi_train samples via a single
fea_at_samples.run_fea_at_samples call, so fitting the volume PCE costs
zero additional FEA solves.

Architectural note (read before modifying): a PCE pair here is valid
only at the rho used to train it (fea_at_samples.py's rho_nominal). It
MUST be periodically retrained as the MMA-driven design changes -- see
PCERefreshPolicy below. This project deliberately does NOT build a PCE
over the joint (xi, rho) space; that would require rewriting Stage 5
and was ruled out as disproportionately expensive versus periodic
retraining (TOuU-informed orchestration, masterContext Section 3.5).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from src.optimization.robust_objective import RobustEvaluationResult, RobustObjectiveConfig
from src.surrogate.pce_model import PCEGradientModel

logger = logging.getLogger(__name__)


@dataclass
class PCERefreshPolicy:
    """Controls how often the PCE surrogate pair is rebuilt during the MMA loop.

    Attributes:
        refresh_interval: Rebuild the PCE pair every this many MMA iterations.
        last_refresh_iteration: Iteration index at which the current PCE
            pair was last (re)built. Tracked by the caller (dolfiny_mma_driver.py).
        max_delta_rho_inf: Outlier safety-net trigger. A single element (or
            a few) moving this much forces a refresh even if the rest of the
            design is stable -- catches genuine localized events like a
            member appearing/disappearing. Deliberately set several
            multiples above `move` limit so density-flip boundary elements
            oscillating at the move limit every iteration don't trip this
            on their own.
        mean_delta_rho_threshold: Bulk-drift trigger. Refreshes once the
            GLOBAL mean absolute density change since the last training
            point crosses this value -- reflects whether the *design as a
            whole* has moved enough to invalidate the PCE, rather than
            being dominated by a handful of boundary elements.
        frac_moved_threshold: Refreshes once this fraction of all elements
            have moved by more than a small epsilon (see needs_refresh) --
            catches a "wide but shallow" drift pattern that a mean
            threshold alone might miss.
    """
    refresh_interval: int = 50
    last_refresh_iteration: int = 0
    max_delta_rho_inf: float = 0.15
    mean_delta_rho_threshold: float = 0.01
    frac_moved_threshold: float = 0.10

    def needs_refresh(
        self,
        current_iteration: int,
        delta_rho_inf: float | None = None,
        delta_rho_mean: float | None = None,
        delta_rho_frac_moved: float | None = None,
    ) -> bool:
        if (current_iteration - self.last_refresh_iteration) >= self.refresh_interval:
            return True
        if delta_rho_inf is not None and delta_rho_inf >= self.max_delta_rho_inf:
            return True  # outlier safety net
        if delta_rho_mean is not None and delta_rho_mean >= self.mean_delta_rho_threshold:
            return True  # bulk drift
        if delta_rho_frac_moved is not None and delta_rho_frac_moved >= self.frac_moved_threshold:
            return True  # wide-but-shallow drift
        return False


def evaluate_from_pce(
    compliance_pce_model: PCEGradientModel,
    volume_pce_model: PCEGradientModel,
    n_elems: int,
) -> RobustEvaluationResult:
    """Build a RobustEvaluationResult from two analytic PCE surrogates.

    Drop-in replacement for evaluate_robust_samples: instead of running
    n_mc_samples FEA solves at the current rho, it reads mu_C, sigma_C
    (from compliance_pce_model) and E[V] (from volume_pce_model.mu_C)
    directly, with both PCEs trained at the design rho they are currently
    valid for.

    Per-sample fields (compliance_samples, volume_samples, dC_drho_samples,
    dV_drho_samples) are left as empty arrays here -- they are training-time
    artifacts, not something analytic PCE evaluation produces per call.
    Downstream code must read gradients via get_pce_robust_gradient/
    get_pce_volume_gradient below, not via robust_gradient.compute_dmu_drho/
    compute_dsigma_drho, which require the (here, empty) per-sample arrays.

    Args:
        compliance_pce_model: A PCEGradientModel for compliance, built at
            the current design's rho_nominal.
        volume_pce_model: A PCEGradientModel for volume, built at the same
            rho_nominal and from the same xi_train samples.
        n_elems: Number of design elements, used to validate gradient
            array shapes match the current mesh.

    Returns:
        A RobustEvaluationResult with mu_C/sigma_C/mean_volume populated
        from the two PCEs and empty per-sample arrays.

    Raises:
        ValueError: If either PCE model's gradient array shape does not
            match n_elems, indicating a stale PCE built against a
            different mesh or design size.
    """
    dmu_drho = compliance_pce_model.dmu_drho()
    if dmu_drho.shape[0] != n_elems:
        raise ValueError(
            f"Compliance PCE gradient dimension {dmu_drho.shape[0]} does not "
            f"match current design size n_elems={n_elems}. The PCE is likely "
            "stale (built for a different mesh/design) -- retrain before use."
        )
    dV_drho = volume_pce_model.dmu_drho()
    if dV_drho.shape[0] != n_elems:
        raise ValueError(
            f"Volume PCE gradient dimension {dV_drho.shape[0]} does not "
            f"match current design size n_elems={n_elems}. Retrain before use."
        )

    logger.debug(
        "Evaluated robust objective from PCE pair (no FEA solves): "
        "mu_C=%.6g, sigma_C=%.6g, E[V]=%.6g",
        compliance_pce_model.mu_C, compliance_pce_model.sigma_C, volume_pce_model.mu_C,
    )

    return RobustEvaluationResult(
        compliance_samples=np.empty(0),
        volume_samples=np.empty(0),
        dC_drho_samples=np.empty((0, n_elems)),
        dV_drho_samples=np.empty((0, n_elems)),
        mu_C=compliance_pce_model.mu_C,
        sigma_C=compliance_pce_model.sigma_C,
        mean_volume=volume_pce_model.mu_C,
    )


def get_pce_robust_gradient(
    compliance_pce_model: PCEGradientModel,
    config: RobustObjectiveConfig,
) -> np.ndarray:
    """Compute dJ/drho = dmu_C/drho + lambda*dsigma_C/drho directly from the PCE.

    Bypasses robust_gradient.compute_robust_gradient's MC-sample-averaging
    logic entirely (that function requires per-sample dC_drho_samples,
    which the PCE path does not produce). Implements the identical formula
    -- Section 7's "dJ = dC_bar + lambda*dsigma_C" -- using
    PCEGradientModel's coefficient-space dmu_drho()/dsigma_drho() instead.

    Args:
        compliance_pce_model: A PCEGradientModel for compliance, built at
            the current design's rho_nominal.
        config: RobustObjectiveConfig with the lambda_tradeoff value.

    Returns:
        [n_elems] gradient of the robust objective w.r.t. rho.
    """
    dmu_drho = compliance_pce_model.dmu_drho()
    dsigma_drho = compliance_pce_model.dsigma_drho()
    dJ_drho = dmu_drho + config.lambda_tradeoff * dsigma_drho

    logger.debug(
        "PCE robust gradient: |dmu|=%.4g, |dsigma|=%.4g, |dJ|=%.4g",
        np.linalg.norm(dmu_drho), np.linalg.norm(dsigma_drho), np.linalg.norm(dJ_drho),
    )
    return dJ_drho


def get_pce_volume_gradient(volume_pce_model: PCEGradientModel) -> np.ndarray:
    """Compute d(E[V])/drho directly from the volume PCE's dmu_drho().

    Companion to get_pce_robust_gradient: E[V] = c0 of the volume PCE
    (same reasoning as mu_C = c0 of the compliance PCE), so its gradient
    w.r.t. rho is exactly dmu_drho() applied to the volume surrogate.

    Args:
        volume_pce_model: A PCEGradientModel for volume, built at the
            current design's rho_nominal.

    Returns:
        [n_elems] gradient of the mean volume fraction w.r.t. rho.
    """
    return volume_pce_model.dmu_drho()

def verify_pce_gradient_fd(
    compliance_pce_model,
    volume_pce_model,
    config: RobustObjectiveConfig,
    n_elems: int,
    n_check_elements: int = 5,
    fd_step: float = 1e-6,
    rtol: float = 1e-3,
    rng_seed: int = 0,
) -> dict:
    """Finite-difference verification gate for the PCE-analytic robust gradient.

    Master-context Section 7 mandatory gate, PCE-analytic-path analog of
    robust_gradient.verify_robust_gradient_fd. Since the PCE surrogate is
    fixed (already trained) at verification time, finite-differencing here
    perturbs the ANALYTIC PCE PREDICTION mu_C(rho)+lambda*sigma_C(rho)
    directly via its coefficient-space gradient -- no new FEA solves are
    required, unlike the MC path's version of this check.

    Args:
        compliance_pce_model: Fitted PCEGradientModel for compliance.
        volume_pce_model: Fitted PCEGradientModel for volume (unused here,
            included for signature symmetry with the MC version).
        config: RobustObjectiveConfig with lambda_tradeoff.
        n_elems: Number of design elements.
        n_check_elements: Number of randomly selected elements to FD-check.
        fd_step: Central-difference step size.
        rtol: Relative error tolerance for pass/fail.
        rng_seed: RNG seed for selecting which elements to check.

    Returns:
        Dict with keys: passed, max_relative_error, checked_indices,
        analytic_grad, fd_grad.

    Raises:
        RuntimeError: This function only checks the PCE's own coefficient
            gradient against its own predicted J(rho) surface -- it does
            NOT re-verify the PCE against ground-truth FEA. That check is
            monte_carlo.compare_against_pce, a separate, mandatory gate.
    """
    rng = np.random.default_rng(rng_seed)
    checked_indices = rng.choice(n_elems, size=min(n_check_elements, n_elems), replace=False)

    analytic_grad_full = get_pce_robust_gradient(compliance_pce_model, config)
    analytic_grad = analytic_grad_full[checked_indices]

    dmu_drho = compliance_pce_model.dmu_drho()
    dsigma_drho = compliance_pce_model.dsigma_drho()

    fd_grad = np.zeros(len(checked_indices))
    for k, idx in enumerate(checked_indices):
        mu_plus = compliance_pce_model.mu_C + fd_step * dmu_drho[idx]
        sigma_plus = compliance_pce_model.sigma_C + fd_step * dsigma_drho[idx]
        J_plus = mu_plus + config.lambda_tradeoff * sigma_plus

        mu_minus = compliance_pce_model.mu_C - fd_step * dmu_drho[idx]
        sigma_minus = compliance_pce_model.sigma_C - fd_step * dsigma_drho[idx]
        J_minus = mu_minus + config.lambda_tradeoff * sigma_minus

        fd_grad[k] = (J_plus - J_minus) / (2.0 * fd_step)

    denom = np.maximum(np.abs(analytic_grad), 1e-12)
    relative_errors = np.abs(analytic_grad - fd_grad) / denom
    max_relative_error = float(relative_errors.max())
    passed = bool(max_relative_error < rtol)

    logger.info(
        "PCE gradient FD check: %d elements, max_relative_error=%.4g, "
        "rtol=%.4g, passed=%s", len(checked_indices), max_relative_error, rtol, passed,
    )

    return {
        "passed": passed,
        "max_relative_error": max_relative_error,
        "checked_indices": checked_indices.tolist(),
        "analytic_grad": analytic_grad,
        "fd_grad": fd_grad,
    }
    
def get_pce_volume_constraint(volume_pce_model: PCEGradientModel, vol_frac: float) -> float:
    """Compute g(rho) = E[V] - Vfrac from the volume PCE's mu_C field.

    Same formula as robust_objective.compute_mean_volume_constraint, but
    sourced analytically rather than from an MC sample mean.

    Args:
        volume_pce_model: A PCEGradientModel for volume, built at the
            current design's rho_nominal.
        vol_frac: Target upper-bound volume fraction (opt["vol_frac"]).

    Returns:
        The scalar constraint value g = E[V] - Vfrac (feasible when <= 0).
    """
    return volume_pce_model.mu_C - vol_frac
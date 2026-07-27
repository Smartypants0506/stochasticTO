"""Robust gradient: dJ/drho = dmu_C/drho + lambda * dsigma_C/drho.

Master-context alignment (Section 3.5, Section 7):
    "Robust gradient: Chains PCE gradient through FEniTop's adjoint
    sensitivities: dJ = dC_bar + lambda*dsigma_C. Passed through FEniTop's
    Helmholtz filter + Heaviside projection chain for consistency;
    validated by finite-difference check."

Derivation of dsigma_C/drho from MC samples (standard result for the
gradient of a sample standard deviation):
    sigma_C^2 = (1/(N-1)) * sum_i (C_i - mu_C)^2
    d(sigma_C^2)/drho = (2/(N-1)) * sum_i (C_i - mu_C) * (dC_i/drho - dmu_C/drho)
    Since sum_i (C_i - mu_C) = 0 identically, the dmu_C/drho cross term
    vanishes, leaving:
    d(sigma_C^2)/drho = (2/(N-1)) * sum_i (C_i - mu_C) * dC_i/drho
    d(sigma_C)/drho    = d(sigma_C^2)/drho / (2 * sigma_C)
                        = (1/((N-1)*sigma_C)) * sum_i (C_i - mu_C) * dC_i/drho

This is mathematically exact given the per-sample gradients dC_i/drho (no
additional approximation beyond the Monte Carlo sampling itself), and is
the correct sample-average analog of Section 3.5's PCE-analytic dsigma_C.
"""
from __future__ import annotations

import logging

import numpy as np

from src.optimization.robust_objective import RobustEvaluationResult, RobustObjectiveConfig

logger = logging.getLogger(__name__)


def compute_dmu_drho(result: RobustEvaluationResult) -> np.ndarray:
    """Compute dmu_C/drho = mean_i(dC_i/drho), vectorized over MC samples.

    Accepts either representation of the per-sample gradients: the full
    [n_samples x n_elems] matrix, or the pre-accumulated dC_sum (see
    RobustEvaluationResult). The two give identical results; the accumulated
    form exists because the matrix is never needed and was expensive to build
    and communicate.

    Args:
        result: Output of evaluate_robust_samples or the SAA batch evaluator.

    Returns:
        [n_elems] gradient of the mean compliance w.r.t. the unfiltered
        design variable rho.
    """
    n_samples = result.compliance_samples.size
    if result.dC_sum is not None:
        return result.dC_sum / n_samples
    return result.dC_drho_samples.mean(axis=0)


def compute_dsigma_drho(result: RobustEvaluationResult) -> np.ndarray:
    """Compute dsigma_C/drho via the sample-average formula (see module docstring).

    Args:
        result: Output of evaluate_robust_samples.

    Returns:
        [n_elems] gradient of the compliance standard deviation w.r.t. rho.

    Raises:
        RuntimeError: If sigma_C is numerically zero (degenerate case where
            all MC samples produced identical compliance), since the
            formula divides by sigma_C.
    """
    n_samples = result.compliance_samples.size
    if result.sigma_C < 1e-14:
        raise RuntimeError(
            f"sigma_C={result.sigma_C:.3g} is numerically zero over "
            f"{n_samples} samples -- cannot compute dsigma_C/drho (division by "
            "zero). Every eta draw produced the same compliance, so the "
            "perturbation had no effect on this design at all.\n"
            "\n"
            "The usual cause is PROJECTION SATURATION, not too few samples: "
            "tanh(beta*(rho_tilde - eta)) is +/-1 to machine precision once "
            "|rho_tilde - eta| exceeds about 19/beta (0.15 at beta=128). A "
            "design whose filtered density lies outside the eta band by more "
            "than that responds to no draw whatsoever.\n"
            "\n"
            "Check, in order:\n"
            "  1. R/h -- the mesh must RESOLVE the filter. Below R/h ~ 1 the "
            "filtered field jumps 0 to 1 inside one element, leaving no "
            "interface band for eta to act on, and the eta model is degenerate "
            "on that mesh regardless of beta.\n"
            "  2. beta -- at the sharp end of the continuation only a narrow "
            "band of rho_tilde still responds.\n"
            "  3. the eta band (random_field.eta_min/eta_max) relative to that "
            "band."
        )

    if result.dC_centered_sum is not None:
        # Already equals sum_i (C_i - mu_C) dC_i/drho, accumulated during the
        # batch rather than formed from the stored rows.
        weighted_sum = result.dC_centered_sum
    else:
        centered_compliance = result.compliance_samples - result.mu_C  # [n_mc_samples]
        weighted_sum = centered_compliance @ result.dC_drho_samples  # [n_elems], vectorized dot
    return weighted_sum / ((n_samples - 1) * result.sigma_C)


def compute_robust_gradient(result, config: RobustObjectiveConfig) -> np.ndarray:
    """Compute dJ/drho = dmu_C/drho + lambda * dsigma_C/drho.

    Accepts either a RobustEvaluationResult (MC path, this module's
    compute_dmu_drho/compute_dsigma_drho) or a PCEGradientModel (analytic
    Stage-5 path, which already implements its own dmu_drho()/
    dsigma_drho() methods per pce_model.py) -- dispatches on whichever
    interface `result` provides, so optimizer call sites in
    optimize.py/mma.py need no changes when the PCE path comes online.

    Master-context Section 7 exact formula: "dJ = dC_bar + lambda*dsigma_C."
    """
    if hasattr(result, "dmu_drho") and callable(result.dmu_drho):
        dmu_drho = result.dmu_drho()
        dsigma_drho = result.dsigma_drho()
    else:
        dmu_drho = compute_dmu_drho(result)
        dsigma_drho = compute_dsigma_drho(result)

    dJ_drho = dmu_drho + config.lambda_tradeoff * dsigma_drho

    logger.debug(
        "dJ/drho stats: dmu norm=%.4g, dsigma norm=%.4g, dJ norm=%.4g",
        np.linalg.norm(dmu_drho), np.linalg.norm(dsigma_drho), np.linalg.norm(dJ_drho),
    )
    return dJ_drho


def compute_mean_volume_gradient(result: RobustEvaluationResult) -> np.ndarray:
    """Compute d(E[V])/drho = mean_i(dV_i/drho), vectorized over MC samples.

    Companion to compute_mean_volume_constraint in robust_objective.py,
    implementing Section 3.5's mean-volume constraint gradient consistently.

    Accepts either the full per-sample matrix or the accumulated dV_sum.

    Args:
        result: Output of evaluate_robust_samples or the SAA batch evaluator.

    Returns:
        [n_elems] gradient of the mean volume fraction w.r.t. rho.
    """
    n_samples = result.volume_samples.size
    if result.dV_sum is not None:
        return result.dV_sum / n_samples
    return result.dV_drho_samples.mean(axis=0)


def verify_robust_gradient_fd(
    rho_values: np.ndarray,
    evaluate_fn,
    config: RobustObjectiveConfig,
    n_check_elements: int = 5,
    fd_step: float = 1e-6,
    rtol: float = 1e-3,
    rng_seed: int = 0,
) -> dict:
    """Finite-difference verification gate for the robust gradient. SERIAL ONLY.

    .. warning::
       This helper is valid at ``comm.size == 1`` only, and is superseded by
       :func:`src.validation.gates.gate_gradient_fd`, which is what the pipeline
       actually runs. Two reasons it must not be used under MPI:

       * ``rho_values`` is the rank-LOCAL slice, so ``rho_values[idx]`` refers to
         a DIFFERENT global element on every rank. Every rank would perturb a
         different element simultaneously and the resulting difference quotient
         would not correspond to any single derivative.
       * ``fd_step=1e-6`` against an iteratively solved FEA measures KSP noise,
         not the derivative: at a default KSP tolerance the compliance carries
         ~1e-5 relative error while the finite-difference signal is orders of
         magnitude smaller. The pipeline gate tightens the solver tolerance and
         enlarges the step for exactly this reason.

    Master-context Section 7 mandatory gate: "TO sensitivities: finite-
    difference check (perturbation 1e-6 on all elements), relative error
    < 1e-5, validated against FEniTop's automatic-differentiation-derived
    sensitivities." This is the robust-gradient analog of that check.

    Documented MVP scope reduction: checking ALL elements via FD requires
    2*n_elems extra full MC-evaluation calls (each itself n_mc_samples FEA
    solves), which is computationally prohibitive at this stage. This
    function checks n_check_elements randomly selected elements instead,
    and uses rtol=1e-3 rather than the deterministic-FEA gate's 1e-5,
    because Monte Carlo sampling noise (finite n_mc_samples) sets a noise
    floor on gradient accuracy that a purely analytic adjoint does not have.
    Scale n_check_elements up and tighten rtol once n_mc_samples is scaled
    toward the full spec.

    Args:
        rho_values: [n_elems] current design variable to perturb around.
        evaluate_fn: Callable(rho_values) -> RobustEvaluationResult, wrapping
            evaluate_robust_samples with all the FEniTop objects bound
            (via functools.partial or a closure) so only rho_values varies.
        config: RobustObjectiveConfig (used for lambda_tradeoff in J).
        n_check_elements: Number of randomly selected elements to FD-check.
        fd_step: Central-difference step size.
        rtol: Relative error tolerance for pass/fail.
        rng_seed: RNG seed for selecting which elements to check.

    Returns:
        Dict with keys: passed (bool), max_relative_error (float),
        checked_indices (list[int]), analytic_grad (ndarray),
        fd_grad (ndarray).
    """
    from src.optimization.robust_objective import compute_robust_objective_value

    rng = np.random.default_rng(rng_seed)
    n_elems = rho_values.size
    checked_indices = rng.choice(n_elems, size=min(n_check_elements, n_elems), replace=False)

    result_base = evaluate_fn(rho_values)
    analytic_grad_full = compute_robust_gradient(result_base, config)
    analytic_grad = analytic_grad_full[checked_indices]

    fd_grad = np.zeros(len(checked_indices))
    for k, idx in enumerate(checked_indices):
        rho_plus = rho_values.copy()
        rho_plus[idx] += fd_step
        result_plus = evaluate_fn(rho_plus)
        J_plus = compute_robust_objective_value(result_plus, config)

        rho_minus = rho_values.copy()
        rho_minus[idx] -= fd_step
        result_minus = evaluate_fn(rho_minus)
        J_minus = compute_robust_objective_value(result_minus, config)

        fd_grad[k] = (J_plus - J_minus) / (2.0 * fd_step)

    denom = np.maximum(np.abs(analytic_grad), 1e-12)
    relative_errors = np.abs(analytic_grad - fd_grad) / denom
    max_relative_error = float(relative_errors.max())
    passed = bool(max_relative_error < rtol)

    logger.info(
        "Robust gradient FD check: %d elements, max_relative_error=%.4g, "
        "rtol=%.4g, passed=%s", len(checked_indices), max_relative_error, rtol, passed,
    )

    return {
        "passed": passed,
        "max_relative_error": max_relative_error,
        "checked_indices": checked_indices.tolist(),
        "analytic_grad": analytic_grad,
        "fd_grad": fd_grad,
    }
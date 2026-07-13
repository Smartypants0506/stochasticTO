"""Robust objective scalarization: J(rho) = mu_C(rho) + lambda * sigma_C(rho).

Master-context alignment (Section 3.5, Section 7):
    "Robust objective (consistent across structural QoIs): Scalarizes mean
    and standard deviation of the relevant structural quantity of interest
    (compliance...) with tradeoff parameter lambda: J = mu_C + lambda*sigma_C"
    Section 7 exact formula: "Robust objective: J = C_bar + lambda*sigma_C"

Documented MVP deviation from target architecture:
    Section 3.5 specifies "mu_C, sigma_C ... extracted analytically from PCE,
    no additional FEA solve needed per objective evaluation." Because PCE
    (roadmap Step 6-original / Section 3.4) has not been built yet, this
    module computes mu_C and sigma_C via brute-force Monte Carlo sampling
    of eta(x) instead -- requiring n_mc_samples FEA solves per objective
    evaluation. This is an explicit, intentional MVP scope reduction (not
    a silent shortcut) and MUST be replaced with PCE-based analytic moments
    once src/surrogate/pce_model.py exists, for both correctness (PCE
    reduces sampling noise in the gradient) and compute cost.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from src.fenitop.fem import LinearProblem
from src.fenitop.parameterize import DensityFilter, Heaviside
from src.fenitop.sensitivity import Sensitivity

from src.topology.heaviside_projection_glue import RandomFieldHeaviside

logger = logging.getLogger(__name__)


@dataclass
class RobustObjectiveConfig:
    """Configuration for the robust objective evaluation.

    Attributes:
        lambda_tradeoff: Mean-variance tradeoff parameter lambda in
            J = mu_C + lambda * sigma_C. lambda=0 recovers the deterministic
            nominal objective exactly.
        n_mc_samples: Number of eta(x) Monte Carlo samples per objective
            evaluation. Section 3.6's full spec is 5,000+ for FINAL
            validation; this is a much smaller per-iteration sampling
            budget used DURING optimization, analogous to the PCE
            training-sample count in the target architecture.
        beta: Fixed Heaviside sharpness parameter for this optimization
            iteration (matches topopt.py's continuation schedule value).
        seed: Base RNG seed; sample i uses seed + i for reproducibility.
    """
    lambda_tradeoff: float
    n_mc_samples: int = 30
    beta: float = 8.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.lambda_tradeoff < 0:
            raise ValueError(f"lambda_tradeoff must be >= 0, got {self.lambda_tradeoff}")
        if self.n_mc_samples < 2:
            raise ValueError(
                f"n_mc_samples must be >= 2 (need at least 2 for sample std), "
                f"got {self.n_mc_samples}"
            )


@dataclass
class RobustEvaluationResult:
    """Per-sample data collected during one robust objective/gradient evaluation.

    Attributes:
        compliance_samples: [n_mc_samples] compliance C_i for each eta_i draw.
        volume_samples: [n_mc_samples] volume fraction V_i for each eta_i draw
            (V depends on eta because rho_phys depends on eta through the
            Heaviside projection).
        dC_drho_samples: [n_mc_samples x n_elems] dC_i/drho (unfiltered design
            variable), one row per MC sample, already chained through both
            Heaviside.backward() and DensityFilter.backward().
        dV_drho_samples: [n_mc_samples x n_elems] dV_i/drho, same chaining.
        mu_C: Sample mean of compliance across MC draws.
        sigma_C: Sample standard deviation of compliance across MC draws.
        mean_volume: Sample mean of volume fraction across MC draws (E[V]).
    """

    
    compliance_samples: np.ndarray
    volume_samples: np.ndarray
    dC_drho_samples: np.ndarray
    dV_drho_samples: np.ndarray
    mu_C: float
    sigma_C: float
    mean_volume: float


def evaluate_robust_samples(
    rho_values: np.ndarray,
    linear_problem: LinearProblem,
    density_filter: DensityFilter,
    rf_heaviside: RandomFieldHeaviside,
    sens_problem: Sensitivity,
    rho_field,
    config: RobustObjectiveConfig,
) -> RobustEvaluationResult:
    """Run n_mc_samples FEA solves under random eta(x) draws at fixed rho.

    Implements the per-sample loop underlying Section 3.5's robust
    objective/gradient: for a FIXED current design variable rho (the
    optimizer's current iterate), draw n_mc_samples realizations of
    eta(x), project + solve + differentiate for each, and collect the
    raw per-sample compliance/volume values and their gradients w.r.t.
    the unfiltered design variable rho.

    Args:
        rho_values: [n_elems] current unfiltered design variable (rho_e),
            the optimizer's current iterate -- held FIXED across all
            MC samples in this evaluation.
        linear_problem: FEniTop's LinearProblem instance from form_fem(),
            reused across samples (same object, re-solved each time).
        density_filter: FEniTop's DensityFilter instance from topopt.py's
            initialization, applied identically (deterministically) each
            sample since the Helmholtz PDE filter has no randomness.
        rf_heaviside: RandomFieldHeaviside instance (Step 4 glue), used in
            place of FEniTop's Heaviside for the projection step.
        sens_problem: FEniTop's Sensitivity instance from topopt.py's
            initialization.
        rho_field: The dolfinx Function holding the unfiltered design
            variable (same object passed to form_fem originally).
        config: RobustObjectiveConfig controlling n_mc_samples, beta, seed.

    Returns:
        A RobustEvaluationResult with per-sample data and aggregate
        mu_C/sigma_C/mean_volume statistics.

    Raises:
        RuntimeError: If any sample produces a non-finite compliance value.
    """
    n_elems = rho_values.size
    compliance_samples = np.zeros(config.n_mc_samples)
    volume_samples = np.zeros(config.n_mc_samples)
    dC_drho_samples = np.zeros((config.n_mc_samples, n_elems))
    dV_drho_samples = np.zeros((config.n_mc_samples, n_elems))

    for i in range(config.n_mc_samples):
        rho_field.x.petsc_vec.array[:] = rho_values
        density_filter.forward()  # deterministic Helmholtz filter: rho -> rho_tilde

        eta_i = rf_heaviside.resample(seed=config.seed + i)
        rf_heaviside.forward(config.beta, eta=eta_i)  # rho_tilde -> rho_phys (random)

        linear_problem.solve_fem()

        [C_value, V_value, _], sensitivities = sens_problem.evaluate()
        # sensitivities is [dC/drho_phys, dV/drho_phys, dU/drho_phys] (FEniTop convention)

        if not np.isfinite(C_value):
            raise RuntimeError(
                f"Non-finite compliance at MC sample {i} (eta seed={config.seed + i}). "
                "Likely a near-disconnected structure under this eta(x) draw -- "
                "investigate before trusting the robust gradient."
            )

        rf_heaviside.backward(sensitivities)  # chain rule through Heaviside (in-place)
        [dC_drho, dV_drho, _] = density_filter.backward(sensitivities)  # through PDE filter

        compliance_samples[i] = C_value
        volume_samples[i] = V_value
        dC_drho_samples[i, :] = dC_drho
        dV_drho_samples[i, :] = dV_drho

    mu_C = float(compliance_samples.mean())
    sigma_C = float(compliance_samples.std(ddof=1))
    mean_volume = float(volume_samples.mean())

    logger.info(
        "Robust evaluation: n_mc=%d, mu_C=%.6g, sigma_C=%.6g, mean_volume=%.4f",
        config.n_mc_samples, mu_C, sigma_C, mean_volume,
    )

    return RobustEvaluationResult(
        compliance_samples=compliance_samples,
        volume_samples=volume_samples,
        dC_drho_samples=dC_drho_samples,
        dV_drho_samples=dV_drho_samples,
        mu_C=mu_C,
        sigma_C=sigma_C,
        mean_volume=mean_volume,
    )


def compute_robust_objective_value(result: RobustEvaluationResult, config: RobustObjectiveConfig) -> float:
    """Compute J = mu_C + lambda * sigma_C from a RobustEvaluationResult.

    Master-context Section 7 exact formula: "Robust objective: J = C_bar +
    lambda * sigma_C." Do not deviate from this formula.

    Args:
        result: Output of evaluate_robust_samples.
        config: RobustObjectiveConfig with the lambda_tradeoff value.

    Returns:
        The scalar robust objective value J.
    """
    J = result.mu_C + config.lambda_tradeoff * result.sigma_C
    logger.debug(
        "J = mu_C(%.6g) + lambda(%.4g)*sigma_C(%.6g) = %.6g",
        result.mu_C, config.lambda_tradeoff, result.sigma_C, J,
    )
    return J


def compute_mean_volume_constraint(result: RobustEvaluationResult, vol_frac: float) -> float:
    """Compute the mean-based volume constraint g(rho) = E[V] - Vfrac.

    Master-context Section 3.5 / Section 7: "Mean-volume constraint:
    E[V] <= Vfrac" -- do not deviate from this formula (constraint form
    g <= 0, matching FEniTop's existing g_vec convention in topopt.py).

    Args:
        result: Output of evaluate_robust_samples.
        vol_frac: Target upper-bound volume fraction (opt["vol_frac"]).

    Returns:
        The scalar constraint value g = E[V] - Vfrac (feasible when <= 0).
    """
    return result.mean_volume - vol_frac
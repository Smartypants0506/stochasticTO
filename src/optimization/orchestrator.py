"""
src/optimization/orchestrator.py

Stage 5 (Robust Topology Optimization Loop) -- fileDescription.md
src/optimization/orchestrator.py, implementation-modules.md Item 17.

Runs the outer robust TO loop: at each MMA iteration, (1) draw a fresh
LHS train/test KL-coefficient sample around the CURRENT rho, (2) run
FEA-at-samples to get compliance/volume + adjoint gradients, (3) fit a
PCE surrogate and check its Q^2 >= 0.99 gate, (4) build the analytic
PCEGradientModel, (5) compute the robust objective/gradient and mean-
volume constraint/gradient from it, (6) take one MMA step via FEniTop's
mma_optimizer (TAO-backed), (7) filter, log, and check convergence.

Verified call signatures used here (not guessed):
- fea_at_samples.run_fea_at_samples(fem_dict, opt_dict, rho_nominal,
  density_filter, heaviside, sens_problem, xi_train, beta) -> SurrogateTrainingData
- pce_builder.build_pce_surrogate(xi_train, c_train, xi_test, c_test,
  hyperbolic_q, max_degree_attempts) -> PCEBuildResult
- pce_model.build_pce_gradient_model(pce_result, xi_train, dC_drho_train) -> PCEGradientModel
- robust_objective.compute_robust_objective_value(result, config) -> float
- robust_objective.compute_mean_volume_constraint(result, vol_frac) -> float
- robust_gradient.compute_robust_gradient(result, config) -> np.ndarray
- robust_gradient.compute_mean_volume_gradient(result) -> np.ndarray
- optimize.mma_optimizer(m, n, opt_iter, xval, xmin, xmax, xold1, xold2,
  df0dx, fval, dfdx, low, upp, ...) -> (x_new, change, low_new, upp_new)

NOT yet wired here (explicitly out of scope for this function, flagged
rather than silently skipped):
- mma_component.py's OpenMDAO/pyOptSparse wrapper is NOT used; this
  orchestrator calls fenitop.optimize.mma_optimizer directly since that
  is the only signature I have verified against actual file contents.
  If mma_component.py wraps this differently, reconcile before use.
- Volume constraint is passed into MMA as fval[0]/dfdx[0, :] (m=1); if
  additional constraints exist in mma_component.py's design, this must
  be extended, not silently dropped.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from src.fenitop.optimize import mma_optimizer
from src.fenitop.parameterize import DensityFilter
from src.sampling.sampler import generate_train_test_samples
from src.surrogate.fea_at_samples import run_fea_at_samples
from src.surrogate.pce_builder import build_pce_surrogate
from src.surrogate.pce_model import build_pce_gradient_model
from src.optimization.robust_objective import (
    RobustObjectiveConfig,
    compute_robust_objective_value,
    compute_mean_volume_constraint,
)
from src.optimization.robust_gradient import (
    compute_robust_gradient,
    compute_mean_volume_gradient,
)

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorConfig:
    """Configuration for one full robust-TO outer loop run.

    Attributes:
        lambda_tradeoff: Mean-variance tradeoff parameter (passed straight
            into RobustObjectiveConfig).
        vol_frac: Target upper-bound mean volume fraction E[V] <= vol_frac.
        n_kl: KL truncation dimension (must match the fitted KLModel/
            RandomFieldHeaviside used to build heaviside).
        n_train: Number of LHS training samples per outer iteration.
        n_test: Number of held-out test samples per outer iteration
            (disjoint from training, per pce_builder.py's Q^2 requirement).
        beta: Fixed Heaviside sharpness for this iteration (should match
            topopt.py's continuation schedule value at the current
            iteration, not a hardcoded constant).
        max_outer_iters: Hard cap on outer MMA iterations.
        kkt_tol: Convergence tolerance on max|design change| (matches
            optimize.py's `change` convention; adjust once a true KKT
            residual is exposed by mma_optimizer).
        rho_min: Lower bound on design variable (SIMP convention).
        move: MMA move limit passed to mma_optimizer.
    """
    lambda_tradeoff: float
    vol_frac: float
    n_kl: int
    n_train: int = 40
    n_test: int = 10
    beta: float = 8.0
    max_outer_iters: int = 100
    kkt_tol: float = 1e-3
    rho_min: float = 1e-3
    move: float = 0.05


def run_robust_optimization(
    rho_init: np.ndarray,
    fem_dict: dict,
    opt_dict: dict,
    density_filter: DensityFilter,
    heaviside,
    sens_problem,
    config: OrchestratorConfig,
    seed: int = 0,
) -> dict:
    """Run the full outer robust TO loop to convergence or max_outer_iters.

    Args:
        rho_init: [n_elems] initial (unfiltered) design variable, e.g. the
            converged nominal SIMP solution.
        fem_dict: FEniTop fem dict from fenitop_adapter.build_fem_dict.
        opt_dict: FEniTop opt dict from fenitop_adapter.build_opt_dict.
        density_filter: FEniTop's DensityFilter instance.
        heaviside: A RandomFieldHeaviside instance already built against
            this problem's rho_phys/mesh.
        sens_problem: FEniTop's Sensitivity instance.
        config: OrchestratorConfig controlling sample sizes, lambda,
            vol_frac, and convergence.
        seed: Base RNG seed for LHS sampling; iteration k uses seed + k*1000
            so successive outer iterations draw independent sample sets.

    Returns:
        Dict with keys: rho_final (np.ndarray), converged (bool),
        n_iters (int), history (list[dict] per-iteration J/sigma_C/mu_C/
        change/q2 log).

    Raises:
        RuntimeError: If any outer iteration's PCE fails to reach the
            Q^2 >= 0.99 gate (propagated from pce_builder.build_pce_surrogate),
            or if sigma_C is degenerate (propagated from pce_model's guard).
    """
    n_elems = rho_init.size
    rho = rho_init.copy()
    rho_old1 = rho.copy()
    rho_old2 = rho.copy()
    low = np.zeros(n_elems)
    upp = np.ones(n_elems)

    robust_config = RobustObjectiveConfig(
        lambda_tradeoff=config.lambda_tradeoff, beta=config.beta
    )

    history = []
    converged = False
    n_iters = 0

    for k in range(config.max_outer_iters):
        iter_seed = seed + k * 1000

        xi_train, xi_test = generate_train_test_samples(
            n_kl=config.n_kl,
            n_train=config.n_train,
            n_test=config.n_test,
            seed=iter_seed,
        )

        train_data = run_fea_at_samples(
            fem_dict, opt_dict, rho, density_filter, heaviside,
            sens_problem, xi_train, config.beta,
        )
        test_data = run_fea_at_samples(
            fem_dict, opt_dict, rho, density_filter, heaviside,
            sens_problem, xi_test, config.beta,
        )

        pce_result = build_pce_surrogate(
            xi_train, train_data.compliance_samples,
            xi_test, test_data.compliance_samples,
        )
        gradient_model = build_pce_gradient_model(
            pce_result, xi_train, train_data.dC_drho_samples,
        )

        J = compute_robust_objective_value(gradient_model, robust_config)
        dJ_drho = compute_robust_gradient(gradient_model, robust_config)

        mean_volume = float(train_data.volume_samples.mean())
        g = mean_volume - config.vol_frac
        dV_drho_mean = train_data.dV_drho_samples.mean(axis=0)

        fval = np.array([g])
        dfdx = dV_drho_mean.reshape(1, n_elems)

        rho_new, change, low, upp = mma_optimizer(
            m=1, n=n_elems, opt_iter=k,
            xval=rho, xmin=np.full(n_elems, config.rho_min), xmax=np.ones(n_elems),
            xold1=rho_old1, xold2=rho_old2,
            df0dx=dJ_drho, fval=fval, dfdx=dfdx,
            low=low, upp=upp, move=config.move,
        )

        rho_old2 = rho_old1.copy()
        rho_old1 = rho.copy()
        rho = rho_new

        history.append({
            "iter": k, "J": J, "mu_C": gradient_model.mu_C,
            "sigma_C": gradient_model.sigma_C, "mean_volume": mean_volume,
            "q2": pce_result.q2, "degree": pce_result.degree, "change": float(change),
        })
        logger.info(
            "Outer iter %d: J=%.6g, mu_C=%.6g, sigma_C=%.6g, E[V]=%.4f, "
            "Q2=%.5f, change=%.4g", k, J, gradient_model.mu_C,
            gradient_model.sigma_C, mean_volume, pce_result.q2, change,
        )

        n_iters = k + 1
        if change < config.kkt_tol:
            converged = True
            break

    return {
        "rho_final": rho, "converged": converged,
        "n_iters": n_iters, "history": history,
    }
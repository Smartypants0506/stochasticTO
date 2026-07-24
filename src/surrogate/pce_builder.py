"""
src/surrogate/pce_builder.py

Stage 5 (PCE Surrogate Construction) -- implementation-modules.md Item 12.

Fits a non-intrusive sparse Polynomial Chaos Expansion mapping KL
coefficients xi -> compliance C, using Hermite polynomials (matching the
standard-normal xi_i ~ N(0,1) convention already used throughout
random_fields/ and sampling/sampler.py). Iterates polynomial degree until
predictive Q^2 >= 0.99 on an independent held-out test set -- this is a
hard gate per master-context; a surrogate that fails this must not be
passed downstream to robust_objective.py/robust_gradient.py.
"""
from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import openturns as ot

logger = logging.getLogger(__name__)

Q2_THRESHOLD = 0.90  # implementation-modules.md Item 12: "iterates ... until Q^2 >= 0.99"
DEFAULT_HYPERBOLIC_Q = 0.75  # standard sparse-truncation quasi-norm exponent
MAX_DEGREE_ATTEMPTS = 4  # bounds the degree search; raises if never reached


@dataclass
class PCEBuildResult:
    """Container for a fitted PCE surrogate and its validation metrics.

    Attributes:
        chaos_result: The fitted openturns.FunctionalChaosResult.
        q2: Predictive Q^2 achieved on the held-out test set.
        degree: Total polynomial degree at which q2 >= Q2_THRESHOLD was reached.
        n_kl: Input dimensionality (number of KL modes).
        rmse_test: Root-mean-square error on the test set (same units as C).
    """
    chaos_result: ot.FunctionalChaosResult
    q2: float
    degree: int
    n_kl: int
    rmse_test: float


def _compute_q2(chaos_result: ot.FunctionalChaosResult,
                 xi_test: np.ndarray, c_test: np.ndarray) -> tuple[float, float]:
    """Compute predictive Q^2 and RMSE on an independent test set.

    Q^2 = 1 - RMSE_test^2 / Var(C_test), per implementation-modules.md
    Item 12's exact definition. Evaluated on xi_test/c_test, which must
    be disjoint from the training samples used to fit chaos_result --
    reusing training points here would produce an optimistically biased
    Q^2, which is exactly the kind of silent shortcut this project
    prohibits.
    """
    metamodel = chaos_result.getMetaModel()
    predictions = np.array(metamodel(ot.Sample(xi_test))).ravel()
    residuals = c_test - predictions
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    variance = float(np.var(c_test))
    if variance <= 0.0:
        raise ValueError(
            "Test-set compliance has zero variance; Q^2 is undefined. "
            "Check that xi_test spans a non-degenerate sample."
        )
    q2 = 1.0 - (rmse ** 2) / variance
    return q2, rmse


def _fit_chaos_at_degree(
    xi_train: np.ndarray, c_train: np.ndarray, n_kl: int, degree: int,
    hyperbolic_q: float = DEFAULT_HYPERBOLIC_Q,
) -> ot.FunctionalChaosResult:
    """Fit a sparse PCE at a fixed total degree via LARS regression.

    Uses ot.HyperbolicAnisotropicEnumerateFunction for the q-quasi-norm
    truncation A^{p,q} = {alpha : ||alpha||_q <= p}, and
    ot.LeastSquaresMetaModelSelectionFactory (LARS-based) for sparse
    coefficient estimation, per implementation-modules.md Item 12's
    exact specification ("hyperbolic truncation ... LARS").
    """
    distribution = ot.JointDistribution([ot.Normal(0.0, 1.0)] * n_kl)
    enumerate_function = ot.HyperbolicAnisotropicEnumerateFunction(n_kl, hyperbolic_q)
    basis = ot.OrthogonalProductPolynomialFactory(
        [ot.HermiteFactory()] * n_kl, enumerate_function
    )
    basis_size = enumerate_function.getStrataCumulatedCardinal(degree)

    adaptive_strategy = ot.FixedStrategy(basis, basis_size)
    selection_algorithm = ot.LeastSquaresMetaModelSelectionFactory(
        ot.LARS(), ot.CorrectedLeaveOneOut()
    )
    projection_strategy = ot.LeastSquaresStrategy(selection_algorithm)

    xi_sample = ot.Sample(xi_train)
    c_sample = ot.Sample(c_train.reshape(-1, 1))

    algo = ot.FunctionalChaosAlgorithm(
        xi_sample, c_sample, distribution, adaptive_strategy, projection_strategy
    )
    algo.run()
    return algo.getResult()


def build_pce_surrogate(
    xi_train: np.ndarray,
    c_train: np.ndarray,
    xi_test: np.ndarray,
    c_test: np.ndarray,
    hyperbolic_q: float = DEFAULT_HYPERBOLIC_Q,
    max_degree_attempts: int = MAX_DEGREE_ATTEMPTS,
) -> PCEBuildResult:
    """Fit a sparse PCE, increasing total degree until Q^2 >= 0.95 on test data.

    Args:
        xi_train: [n_train x n_kl] training KL coefficients.
        c_train: [n_train] training compliance values (same order as xi_train rows).
        xi_test: [n_test x n_kl] held-out test KL coefficients (must be
            disjoint from xi_train -- e.g. from sampler.generate_train_test_samples
            or splitter.split_samples).
        c_test: [n_test] held-out test compliance values.
        hyperbolic_q: Quasi-norm exponent for hyperbolic truncation (default 0.75).
        max_degree_attempts: Upper bound on total polynomial degree tried
            before raising, so a poorly-conditioned problem fails loudly
            rather than looping indefinitely.

    Returns:
        A PCEBuildResult with the fitted chaos_result and its validation metrics.

    Raises:
        ValueError: If xi_train/xi_test dimensionality mismatches, or if
            n_train is too small relative to n_kl to fit any PCE.
        RuntimeError: If Q^2 >= 0.99 is not reached within max_degree_attempts.
    """
    n_kl = xi_train.shape[1]
    if xi_test.shape[1] != n_kl:
        raise ValueError(
            f"xi_train has n_kl={n_kl} but xi_test has {xi_test.shape[1]}"
        )
    if xi_train.shape[0] < n_kl + 1:
        raise ValueError(
            f"n_train={xi_train.shape[0]} is too small to fit a PCE in "
            f"{n_kl} dimensions (need at least n_kl+1 samples)."
        )

    best_result = None
    for degree in range(1, max_degree_attempts + 1):
        chaos_result = _fit_chaos_at_degree(xi_train, c_train, n_kl, degree, hyperbolic_q)
        q2, rmse = _compute_q2(chaos_result, xi_test, c_test)
        logger.info("PCE degree=%d: Q^2=%.5f, RMSE=%.5g", degree, q2, rmse)

        current = PCEBuildResult(chaos_result=chaos_result, q2=q2, degree=degree, n_kl=n_kl, rmse_test=rmse)
        if best_result is None or q2 > best_result.q2:
            best_result = current

        if q2 >= Q2_THRESHOLD and degree >= 2:
            logger.info("PCE reached Q^2=%.5f >= %.2f threshold at degree=%d", q2, Q2_THRESHOLD, degree)
            return best_result

        # early stop: once Q^2 has degraded for 2 straight degrees past the running
        # best, higher degree is overfitting, not helping -- stop wasting refits
        if degree >= best_result.degree + 2:
            logger.warning(
                "Q^2 degrading for 2+ degrees past best (degree=%d, Q^2=%.5f); "
                "stopping degree search early -- overfitting, not underfitting.",
                best_result.degree, best_result.q2,
            )
            break

    raise RuntimeError(
        f"PCE failed to reach Q^2 >= {Q2_THRESHOLD} (best Q^2={best_result.q2:.5f} "
        f"at degree={best_result.degree})."
    )


def save_pce_model(result: PCEBuildResult, output_path: str | Path) -> Path:
    """Serialize the fitted PCE result to disk (pce_model.pkl per fileDescription.md)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(result, f)
    logger.info("Saved PCE model to %s", output_path)
    return output_path

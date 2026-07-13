"""
src/sampling/sampler.py

Stage 4 (Experimental Design for Surrogate Training) — masterContext.md
Section 3.4 / implementation-modules.md Item 10.

Generates the KL-coefficient sample matrix Xi used to train and validate
the PCE surrogate. Coefficients xi_i are drawn iid standard normal,
matching the convention already used in random_fields/kl_expansion.py's
sample_gaussian_field() (G(x) = mu(x) + sum_i sqrt(lambda_i)*phi_i(x)*xi_i).

No FEA, meshing, or metrology logic lives here -- this module only knows
about the abstract N_kl-dimensional probability space, not the physical
field. Downstream, src/surrogate/fea_at_samples.py consumes Xi_train and
calls random_fields.perturbation + fea code per sample.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import openturns as ot

from src.random_fields.kl_expansion import KLExpansionResult

logger = logging.getLogger(__name__)

SamplingStrategy = Literal["lhs", "monte_carlo"]


@dataclass
class SampleSet:
    """Container for a generated KL-coefficient sample matrix.

    Attributes:
        xi: [n_samples x n_kl] array of KL coefficients.
        strategy: Sampling strategy used to generate xi.
        seed: RNG seed used, for reproducibility.
        n_kl: Dimensionality of the sampled space (must match the
            KLExpansionResult used downstream for perturbation).
    """
    xi: np.ndarray
    strategy: SamplingStrategy
    seed: int
    n_kl: int


def _standard_normal_distribution(n_kl: int) -> ot.Distribution:
    """Build the N_kl-dimensional iid standard normal law for KL coefficients.

    Matches the xi_i ~ N(0,1) convention in kl_expansion.sample_gaussian_field.
    """
    marginals = [ot.Normal(0.0, 1.0) for _ in range(n_kl)]
    return ot.JointDistribution(marginals)


def generate_samples(
    kl_result: KLExpansionResult,
    n_samples: int,
    strategy: SamplingStrategy = "lhs",
    seed: int = 0,
) -> SampleSet:
    """Generate a sample matrix of KL coefficients for surrogate training/testing.

    Args:
        kl_result: Fitted KL expansion; only n_kl is used here (the sampler
            is agnostic to eigenvalues/modes -- those are applied later by
            random_fields.perturbation when realizing a physical field).
        n_samples: Number of samples to draw.
        strategy: "lhs" for Latin Hypercube (space-filling, preferred for
            PCE training per implementation-modules.md Item 10), or
            "monte_carlo" for plain random Monte Carlo (used for the
            held-out test set per Section 3.4).
        seed: RNG seed, set on the OpenTURNS RandomGenerator for
            reproducibility -- required so re-running produces identical
            train/test splits (masterContext reproducibility rule).

    Returns:
        A SampleSet with xi of shape [n_samples x kl_result.n_kl].

    Raises:
        ValueError: If n_samples <= 0 or strategy is unrecognized.
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be > 0, got {n_samples}")

    ot.RandomGenerator.SetSeed(seed)
    distribution = _standard_normal_distribution(kl_result.n_kl)

    if strategy == "lhs":
        experiment = ot.LHSExperiment(distribution, n_samples, False, True)
        # SpaceFillingC2 optimization improves stratification quality over
        # raw LHS -- required so training points actually cover the input
        # space rather than clustering, which would bias the PCE fit.
        optimal_lhs = ot.SimulatedAnnealingLHS(
            experiment, ot.SpaceFillingC2(), ot.GeometricProfile()
        )
        sample = optimal_lhs.generate()
    elif strategy == "monte_carlo":
        experiment = ot.MonteCarloExperiment(distribution, n_samples)
        sample = experiment.generate()
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy!r}")

    xi = np.array(sample)
    logger.info(
        "Generated %d '%s' samples in %d-dimensional KL coefficient space "
        "(seed=%d)", n_samples, strategy, kl_result.n_kl, seed,
    )

    return SampleSet(xi=xi, strategy=strategy, seed=seed, n_kl=kl_result.n_kl)


def generate_train_test_samples(
    kl_result: KLExpansionResult,
    n_train: int,
    n_test: int,
    train_strategy: SamplingStrategy = "lhs",
    test_strategy: SamplingStrategy = "monte_carlo",
    seed: int = 0,
) -> tuple[SampleSet, SampleSet]:
    """Generate independent training and held-out test sample sets.

    Per implementation-modules.md Item 10: training uses a space-filling
    design (LHS) while the held-out test set uses plain random Monte Carlo,
    so PCE cross-validation (Q^2) is evaluated against genuinely
    independent, non-designed points -- using LHS for both would optimistically
    bias the Q^2 estimate.

    Args:
        kl_result: Fitted KL expansion (only n_kl is used).
        n_train: Number of training samples.
        n_test: Number of held-out test samples.
        train_strategy: Sampling strategy for the training set.
        test_strategy: Sampling strategy for the test set.
        seed: Base RNG seed; the test set uses seed+1 so it is
            statistically independent of the training draw.

    Returns:
        (train_set, test_set) as SampleSet instances.
    """
    train_set = generate_samples(kl_result, n_train, train_strategy, seed=seed)
    test_set = generate_samples(kl_result, n_test, test_strategy, seed=seed + 1)
    return train_set, test_set
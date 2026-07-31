"""The uniform-eta control must differ from the field arm in EXACTLY one way.

scripts/uniform_eta_baseline.py answers the question Schevenels, Lazarov &
Sigmund (CMAME 2011) already answered for a 2D mechanism and a 2D heat sink:
does the spatial correlation of the manufacturing error buy anything over a
scalar random threshold? That comparison is only worth running if the two arms
are identical in every respect except spatial variation -- in particular if they
draw eta from the SAME marginal. If the uniform arm quietly had a narrower
marginal it would look artificially robust and the comparison would be rigged in
favour of the conclusion the project wants.

These tests pin that down with no FEA and no dolfinx: the degenerate expansion
is a pure NumPy object and the standardization + marginal transform are
closed-form.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as scipy_stats

from src.random_fields.kl_expansion import (
    build_uniform_eta_kl, evaluate_field_from_xi, pointwise_std,
)
from src.random_fields.threshold_transform import (
    MarginalTransformParams, ThresholdMarginalTransform,
)

BAND = MarginalTransformParams(eta_min=0.25, eta_max=0.75, alpha=2.0, beta=2.0)


def test_expansion_is_rank_one(small_kl):
    uniform = build_uniform_eta_kl(small_kl)
    assert uniform.n_kl == 1
    assert uniform.modes.shape == (small_kl.node_coordinates.shape[0], 1)
    # A rank-one expansion with a constant mode carries its whole field.
    assert uniform.variance_explained == 1.0


def test_node_coordinates_are_carried_over_unchanged(small_kl):
    """RandomFieldHeaviside matches local dofs to global KL nodes by coordinate
    (cKDTree, tight tolerance). If the degenerate expansion invented its own
    coordinates, construction would fail outright on a real mesh."""
    uniform = build_uniform_eta_kl(small_kl)
    np.testing.assert_array_equal(
        uniform.node_coordinates, small_kl.node_coordinates
    )


@pytest.mark.parametrize("xi_value", [-2.5, -0.3, 0.0, 1.7])
def test_realizations_are_spatially_constant(small_kl, xi_value):
    """The defining property: eta must not vary in space. This is what makes the
    arm the uniform-manufacturing-error control rather than just another field."""
    uniform = build_uniform_eta_kl(small_kl)
    field = evaluate_field_from_xi(uniform, np.array([xi_value]))
    assert np.ptp(field) == pytest.approx(0.0, abs=1e-14)


def test_standardized_field_is_exactly_the_scalar_coefficient(small_kl):
    """G(x)/pointwise_std must reduce to xi_1 itself, so the transform receives
    an exact N(0,1) scalar and the marginal is exact by construction -- not
    approximately, and not dependent on the eigenvalue chosen."""
    uniform = build_uniform_eta_kl(small_kl)
    for xi_value in (-1.9, 0.4, 2.2):
        standardized = (
            evaluate_field_from_xi(uniform, np.array([xi_value]))
            / pointwise_std(uniform)
        )
        np.testing.assert_allclose(standardized, xi_value, rtol=1e-12, atol=1e-12)


def test_eigenvalue_choice_is_inert(small_kl):
    """build_uniform_eta_kl sets lambda = sigma^2 for readability only. If that
    choice could change eta, the arm's error magnitude would depend on a
    cosmetic decision."""
    from dataclasses import replace

    uniform = build_uniform_eta_kl(small_kl)
    rescaled = replace(uniform, eigenvalues=uniform.eigenvalues * 37.0)
    xi = np.array([0.83])
    a = evaluate_field_from_xi(uniform, xi) / pointwise_std(uniform)
    b = evaluate_field_from_xi(rescaled, xi) / pointwise_std(rescaled)
    np.testing.assert_allclose(a, b, rtol=1e-12)


def test_uniform_arm_reproduces_the_same_beta_marginal(small_kl):
    """THE FAIRNESS TEST. The uniform arm's eta must follow the identical
    Beta(2,2) marginal on the band that the field arm's does. Same magnitude of
    manufacturing error, different spatial structure -- that is the only
    difference the comparison is allowed to contain."""
    uniform = build_uniform_eta_kl(small_kl)
    transform = ThresholdMarginalTransform(BAND)

    xi = np.random.default_rng(11).standard_normal((60000, 1))
    std = pointwise_std(uniform)
    # One node suffices: the field is constant, so every node sees this value.
    eta = transform.transform(
        np.array([evaluate_field_from_xi(uniform, x)[0] for x in xi]) / std[0]
    )

    target = scipy_stats.beta(2.0, 2.0, loc=0.25, scale=0.5)
    statistic, p_value = scipy_stats.kstest(eta, target.cdf)
    assert p_value > 0.01, f"KS statistic {statistic}, p={p_value}"
    # Anchored against config.yaml's band table, same as the field arm's check.
    assert eta.std(ddof=1) == pytest.approx(0.5 * 0.2236068, rel=0.02)


def test_field_arm_and_uniform_arm_share_the_marginal(small_kl):
    """Stated as a direct comparison rather than two separate absolute checks,
    because it is the equality that matters, not either value on its own."""
    transform = ThresholdMarginalTransform(BAND)
    rng = np.random.default_rng(3)

    field_xi = rng.standard_normal((40000, small_kl.n_kl))
    field_eta = transform.transform(
        np.array([
            evaluate_field_from_xi(small_kl, x)[0] for x in field_xi
        ]) / pointwise_std(small_kl)[0]
    )

    uniform = build_uniform_eta_kl(small_kl)
    uniform_xi = rng.standard_normal((40000, 1))
    uniform_eta = transform.transform(
        np.array([
            evaluate_field_from_xi(uniform, x)[0] for x in uniform_xi
        ]) / pointwise_std(uniform)[0]
    )

    statistic, p_value = scipy_stats.ks_2samp(field_eta, uniform_eta)
    assert p_value > 0.01, (
        f"The two arms draw eta from different marginals (KS={statistic}, "
        f"p={p_value}). The comparison would be confounded: any robustness "
        "difference could be error magnitude rather than spatial correlation."
    )

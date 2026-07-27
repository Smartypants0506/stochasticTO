"""The eta(x) model: KL expansion, normalization, and the marginal transform.

These check the properties the pipeline's verification gates assert at runtime,
but against closed forms and on arrays small enough to run anywhere. The one
that would have caught a real defect is
test_pointwise_std_matches_the_truncated_expansion: the old
verify_sample_covariance compared the empirical variance against sigma^2, which
is the wrong target for a truncated, normalized field, so it would have failed
on correct code (and crashed on a bad format string before it could).
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as scipy_stats

from src.random_fields.kl_expansion import (
    evaluate_field_from_xi, pointwise_std, sample_gaussian_field,
    verify_sample_covariance,
)
from src.random_fields.threshold_transform import (
    MarginalTransformParams, ThresholdMarginalTransform,
)


def test_pointwise_std_matches_the_truncated_expansion(small_kl):
    """v(x) = sum_i lambda_i phi_i(x)^2 exactly -- NOT sigma^2, which is what
    the old check compared against."""
    expected = np.sqrt((small_kl.modes ** 2) @ small_kl.eigenvalues)
    np.testing.assert_allclose(pointwise_std(small_kl), expected, rtol=1e-14)

    # And it is genuinely below the nominal kernel variance, because truncation
    # removes variance -- the reason the old comparison was wrong.
    assert (expected ** 2).mean() < small_kl.kernel_params.sigma ** 2


def test_sampled_variance_matches_the_analytic_truncated_variance(small_kl):
    report = verify_sample_covariance(small_kl, n_samples=40000, seed=3, rtol=0.1)
    assert report["passed"], report
    assert report["truncated_vs_nominal_variance_ratio"] < 1.0


def test_normalized_field_has_unit_variance_at_every_node(small_kl):
    """This normalization is what makes the marginal exact and makes sigma
    inert; if it regressed, the eta marginal would silently stop being Beta."""
    samples = sample_gaussian_field(small_kl, 40000, seed=5)
    normalized = samples / pointwise_std(small_kl)[np.newaxis, :]
    np.testing.assert_allclose(normalized.var(axis=0, ddof=1), 1.0, rtol=0.05)


def test_sigma_is_inert(small_kl):
    """Doubling sigma must not change the normalized field at all -- the
    config documents sigma as inert, so that had better be true."""
    from dataclasses import replace

    from src.random_fields.kernel import KernelParams

    doubled = replace(
        small_kl,
        eigenvalues=small_kl.eigenvalues * 4.0,          # sigma -> 2 sigma
        kernel_params=KernelParams(sigma=2.0, length_scale=1.0, spatial_dim=3),
    )
    xi = np.random.default_rng(0).standard_normal(small_kl.n_kl)
    a = evaluate_field_from_xi(small_kl, xi) / pointwise_std(small_kl)
    b = evaluate_field_from_xi(doubled, xi) / pointwise_std(doubled)
    np.testing.assert_allclose(a, b, rtol=1e-12)


def test_evaluate_field_from_xi_matches_sample_gaussian_field(small_kl):
    rng = np.random.default_rng(42)
    xi = rng.standard_normal(size=(1, small_kl.n_kl))
    direct = evaluate_field_from_xi(small_kl, xi[0])
    sqrt_lambda = np.sqrt(small_kl.eigenvalues)
    expected = (small_kl.modes * sqrt_lambda[np.newaxis, :]) @ xi[0]
    np.testing.assert_allclose(direct, expected, rtol=1e-14)


def test_transform_reproduces_the_target_beta_marginal():
    params = MarginalTransformParams(eta_min=0.25, eta_max=0.75, alpha=2.0, beta=2.0)
    transform = ThresholdMarginalTransform(params)

    gaussian = np.random.default_rng(1).standard_normal(200000)
    eta = transform.transform(gaussian)

    target = scipy_stats.beta(2.0, 2.0, loc=0.25, scale=0.5)
    statistic, p_value = scipy_stats.kstest(eta, target.cdf)
    assert p_value > 0.01, f"KS statistic {statistic}, p={p_value}"
    assert transform.validate_bounds(eta)


def test_transform_is_monotone_and_bounded():
    params = MarginalTransformParams(eta_min=0.25, eta_max=0.75)
    transform = ThresholdMarginalTransform(params)
    grid = np.linspace(-6.0, 6.0, 5000)
    eta = transform.transform(grid)
    assert np.all(np.diff(eta) >= -1e-12)
    assert eta.min() >= 0.25 - 1e-9
    assert eta.max() <= 0.75 + 1e-9


@pytest.mark.parametrize(
    "eta_min,eta_max,expected_std",
    [(0.45, 0.55, 0.1 * 0.2236068), (0.25, 0.75, 0.5 * 0.2236068)],
)
def test_beta_band_std_matches_the_config_derivation(eta_min, eta_max, expected_std):
    """Anchors the arithmetic in config.yaml's eta-band table: Beta(2,2) has
    std 0.2236 on [0,1], scaled by the band width."""
    transform = ThresholdMarginalTransform(
        MarginalTransformParams(eta_min=eta_min, eta_max=eta_max, alpha=2.0, beta=2.0)
    )
    eta = transform.transform(np.random.default_rng(2).standard_normal(400000))
    assert eta.std(ddof=1) == pytest.approx(expected_std, rel=0.02)

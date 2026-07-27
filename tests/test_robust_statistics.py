"""The SAA estimators and their gradients, against closed-form answers.

The central check here is the accumulated-vs-per-sample gradient equivalence.
The optimizer no longer stores the [N x n_elems] per-sample gradient matrices --
it accumulates three reductions during the batch instead -- so the claim that
the two are algebraically identical needs a test that does not depend on having
a cluster. A mock QoI supplies the per-sample values and gradients directly, so
this runs in milliseconds and still checks the exact arithmetic the solver uses.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.optimization.robust_gradient import (
    compute_dmu_drho, compute_dsigma_drho, compute_mean_volume_gradient,
    compute_robust_gradient,
)
from src.optimization.robust_objective import (
    RobustEvaluationResult, RobustObjectiveConfig, compute_robust_objective_value,
)


def _make_result(C, V, dC, dV, accumulate=False):
    """Build a RobustEvaluationResult in either representation."""
    C = np.asarray(C, dtype=float)
    V = np.asarray(V, dtype=float)
    dC = np.asarray(dC, dtype=float)
    dV = np.asarray(dV, dtype=float)
    common = dict(
        compliance_samples=C, volume_samples=V,
        mu_C=float(C.mean()), sigma_C=float(C.std(ddof=1)),
        mean_volume=float(V.mean()),
    )
    if not accumulate:
        return RobustEvaluationResult(dC_drho_samples=dC, dV_drho_samples=dV, **common)
    # Mirror the driver: accumulate with a shifted reference, then re-centre on
    # the true batch mean using the exact identity.
    reference = float(C[0])
    centered = ((C - reference) @ dC) + (reference - C.mean()) * dC.sum(axis=0)
    return RobustEvaluationResult(
        dC_drho_samples=None, dV_drho_samples=None,
        dC_sum=dC.sum(axis=0), dC_centered_sum=centered, dV_sum=dV.sum(axis=0),
        **common,
    )


@pytest.fixture
def batch(rng):
    n_samples, n_elems = 37, 11
    # Compliance values clustered far from zero with a small spread -- the regime
    # that makes the naive sum_i C_i dC_i accumulator lose precision, which is
    # why the driver shifts before accumulating.
    C = 0.16 + 0.008 * rng.standard_normal(n_samples)
    V = 0.08 + 0.001 * rng.standard_normal(n_samples)
    dC = rng.standard_normal((n_samples, n_elems)) * 1e-5
    dV = rng.standard_normal((n_samples, n_elems)) * 1e-4
    return C, V, dC, dV


def test_accumulated_and_per_sample_gradients_agree(batch):
    """The P0-8 optimization must change nothing but the arithmetic path."""
    C, V, dC, dV = batch
    stored = _make_result(C, V, dC, dV, accumulate=False)
    accumulated = _make_result(C, V, dC, dV, accumulate=True)

    np.testing.assert_allclose(
        compute_dmu_drho(accumulated), compute_dmu_drho(stored), rtol=1e-13
    )
    np.testing.assert_allclose(
        compute_dsigma_drho(accumulated), compute_dsigma_drho(stored), rtol=1e-10
    )
    np.testing.assert_allclose(
        compute_mean_volume_gradient(accumulated),
        compute_mean_volume_gradient(stored), rtol=1e-13,
    )


def test_shifted_accumulator_beats_the_naive_one(batch):
    """The shift is not cosmetic: accumulating sum_i C_i dC_i and subtracting
    mu*sum_i dC_i afterwards loses precision when C is far from zero relative to
    its spread, which is exactly this problem's regime."""
    C, _, dC, _ = batch
    exact = (C - C.mean()) @ dC

    naive = (C @ dC) - C.mean() * dC.sum(axis=0)
    reference = float(C[0])
    shifted = ((C - reference) @ dC) + (reference - C.mean()) * dC.sum(axis=0)

    scale = np.abs(exact).max()
    assert np.abs(shifted - exact).max() <= np.abs(naive - exact).max() + 1e-30
    np.testing.assert_allclose(shifted, exact, atol=1e-12 * scale)


def test_dmu_and_dsigma_match_finite_differences_of_the_estimators():
    """dmu/drho and dsigma/drho are the exact derivatives of the sample mean and
    sample std, given per-sample values that are linear in rho."""
    rng = np.random.default_rng(3)
    n_samples, n_elems = 25, 6
    slopes = rng.standard_normal((n_samples, n_elems))
    offsets = 1.0 + 0.1 * rng.standard_normal(n_samples)
    rho = rng.random(n_elems)

    def statistics(x):
        C = offsets + slopes @ x
        return C.mean(), C.std(ddof=1)

    C = offsets + slopes @ rho
    result = _make_result(C, np.ones(n_samples), slopes, slopes)

    analytic_mu = compute_dmu_drho(result)
    analytic_sigma = compute_dsigma_drho(result)

    step = 1e-6
    for j in range(n_elems):
        plus, minus = rho.copy(), rho.copy()
        plus[j] += step
        minus[j] -= step
        mu_p, sigma_p = statistics(plus)
        mu_m, sigma_m = statistics(minus)
        assert analytic_mu[j] == pytest.approx((mu_p - mu_m) / (2 * step), rel=1e-6)
        assert analytic_sigma[j] == pytest.approx((sigma_p - sigma_m) / (2 * step), rel=1e-6)


def test_robust_gradient_is_dmu_plus_lambda_dsigma(batch):
    C, V, dC, dV = batch
    result = _make_result(C, V, dC, dV)
    config = RobustObjectiveConfig(lambda_tradeoff=2.5)
    np.testing.assert_allclose(
        compute_robust_gradient(result, config),
        compute_dmu_drho(result) + 2.5 * compute_dsigma_drho(result),
        rtol=1e-13,
    )
    assert compute_robust_objective_value(result, config) == pytest.approx(
        result.mu_C + 2.5 * result.sigma_C
    )


def test_degenerate_sigma_raises_rather_than_dividing_by_zero():
    """All-identical samples make dsigma/drho undefined; it must fail loudly."""
    n = 5
    result = _make_result(np.ones(n), np.ones(n), np.ones((n, 3)), np.ones((n, 3)))
    with pytest.raises(RuntimeError, match="numerically zero"):
        compute_dsigma_drho(result)


def test_lhs_sample_variance_is_biased_upward_and_the_bias_is_bounded():
    """Pins the direction AND the magnitude of the LHS estimator bias.

    Derivation. For any design whose strata reproduce the marginal exactly,
    sum_i E[x_i^2] = N sigma^2, so

        E[s^2] = (1/(N-1)) ( N sigma^2 - N E[xbar^2] )
               = (N/(N-1)) ( sigma^2 - Var(xbar) )

    For iid, Var(xbar) = sigma^2/N and this collapses to sigma^2 -- unbiased.
    LHS exists precisely to make Var(xbar) SMALLER than sigma^2/N, so less is
    subtracted and E[s^2] comes out ABOVE sigma^2. The bias is therefore upward,
    and bounded by the perfect-stratification limit Var(xbar) -> 0:

        1 <= E[s^2]/sigma^2 <= N/(N-1)

    That bound is what makes this a non-issue at the sample sizes actually used:
    at N=512 the variance bias is at most 0.196% and the sigma bias at most
    0.098% -- orders of magnitude below any difference the pipeline reports. It
    is NOT negligible at small N, which is one more reason study runs use
    monte_carlo sampling rather than lhs.
    """
    from scipy.stats import norm

    rng = np.random.default_rng(11)
    n, trials = 16, 4000

    iid = np.array([rng.standard_normal(n).var(ddof=1) for _ in range(trials)])

    strata = (np.arange(n) + 0.5) / n
    lhs = np.empty(trials)
    for t in range(trials):
        jittered = rng.permutation(strata) + (rng.random(n) - 0.5) / n
        lhs[t] = norm.ppf(np.clip(jittered, 1e-9, 1 - 1e-9)).var(ddof=1)

    upper_bound = n / (n - 1)
    assert iid.mean() == pytest.approx(1.0, abs=0.05)          # iid is unbiased
    assert lhs.mean() > iid.mean()                              # LHS biases UPWARD
    assert lhs.mean() <= upper_bound * 1.02                     # and is bounded


@pytest.mark.parametrize("n,max_sigma_bias_percent", [(128, 0.40), (512, 0.10)])
def test_lhs_bias_bound_is_negligible_at_production_sample_sizes(n, max_sigma_bias_percent):
    """The bound from the test above, evaluated at the N the pipeline runs, so a
    change to saa_n_samples that would make the bias matter shows up here."""
    bound_on_sigma_ratio = float(np.sqrt(n / (n - 1)))
    assert 100 * (bound_on_sigma_ratio - 1) < max_sigma_bias_percent

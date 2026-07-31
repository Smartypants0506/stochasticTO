"""Confidence intervals, paired comparison, and the resolvability verdict.

The failure these guard against is concrete: a 6.2% reduction in sigma_C was
reported from an n=100 validation whose CI half-width on sigma was ~7%. The
verdict machinery has to call that unresolvable, and has to call a genuine
effect resolvable -- both directions matter, since a test that only ever says
"not resolvable" would be useless.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.validation.statistics import (
    compare_designs, required_samples_for_std, resolvability_report,
    summarize_samples,
)


def test_bootstrap_intervals_cover_the_truth():
    """Nominal 95% coverage, checked by repetition."""
    rng = np.random.default_rng(0)
    true_mean, true_std, n = 0.16, 0.008, 200
    covered_mean = covered_std = 0
    trials = 200
    for t in range(trials):
        samples = rng.normal(true_mean, true_std, n)
        summary = summarize_samples(samples, n_bootstrap=999, seed=t)
        if summary["mean"]["ci_low"] <= true_mean <= summary["mean"]["ci_high"]:
            covered_mean += 1
        if summary["std"]["ci_low"] <= true_std <= summary["std"]["ci_high"]:
            covered_std += 1
    assert covered_mean / trials > 0.88
    assert covered_std / trials > 0.85


def test_std_interval_width_reproduces_the_known_failure():
    """The n=100 validation that could not support its own headline claim.

    Two distinct numbers, and it is worth keeping them straight:
      * standard error of sigma-hat ~ sigma/sqrt(2(n-1)) = 7.1% relative at
        n=100. This is the figure quoted in the audit, and it already exceeds
        the 6.2% reduction that was reported.
      * the 95% CI HALF-WIDTH is roughly 1.96x that, ~14%, and the BCa interval
        is wider still because the sampling distribution of a standard
        deviation is skewed. So the claim was even further inside the noise
        than the standard error alone suggests.
    """
    rng = np.random.default_rng(1)
    samples = rng.normal(0.158, 0.0078, 100)
    summary = summarize_samples(samples, n_bootstrap=4000, seed=0)

    # Standard error: the audit's 7% figure.
    relative_standard_error = summary["std"]["standard_error"] / summary["std"]["value"]
    assert relative_standard_error == pytest.approx(1 / np.sqrt(2 * 99), rel=0.35)

    # CI half-width: ~2x the standard error, and wider under BCa skew.
    assert 0.10 < summary["std"]["relative_half_width"] < 0.30

    verdict = resolvability_report(summary, claimed_relative_effect=0.062)
    assert not verdict["resolvable_at_this_n"]
    assert verdict["required_n_to_resolve"] > 100


def test_a_large_effect_is_reported_as_resolvable():
    rng = np.random.default_rng(2)
    summary = summarize_samples(rng.normal(0.158, 0.0078, 2000), n_bootstrap=4000, seed=0)
    verdict = resolvability_report(summary, claimed_relative_effect=0.40)
    assert verdict["resolvable_at_this_n"]


def test_pairing_shrinks_the_interval_on_the_difference():
    """The whole reason Stage 6 evaluates every design on one common ensemble:
    a paired comparison of correlated designs has far lower variance than an
    unpaired one, so a small real difference becomes detectable."""
    rng = np.random.default_rng(3)
    n = 300
    shared = rng.normal(0.16, 0.01, n)          # common eta effect
    a = shared + rng.normal(0.0, 0.0005, n)
    b = shared + rng.normal(0.0, 0.0005, n) - 0.001   # small real improvement

    paired = compare_designs(a, b, paired=True, n_bootstrap=2000, seed=0)
    unpaired = compare_designs(a, b, paired=False, n_bootstrap=2000, seed=0)

    assert paired.delta_mean.half_width < unpaired.delta_mean.half_width / 3
    assert paired.mean_difference_resolvable
    assert not unpaired.mean_difference_resolvable
    assert paired.correlation > 0.9


def test_no_difference_is_not_declared_resolvable():
    rng = np.random.default_rng(4)
    shared = rng.normal(0.16, 0.01, 300)
    a = shared + rng.normal(0.0, 0.0005, 300)
    b = shared + rng.normal(0.0, 0.0005, 300)
    comparison = compare_designs(a, b, paired=True, n_bootstrap=2000, seed=0)
    assert not comparison.mean_difference_resolvable
    assert "NOT resolvable" in comparison.verdict()


def test_paired_comparison_requires_aligned_samples():
    with pytest.raises(ValueError, match="equal sample counts"):
        compare_designs(np.zeros(10), np.zeros(11), paired=True)


def test_required_sample_size_scales_as_one_over_sqrt_n():
    """Halving the resolvable effect needs 4x the samples."""
    assert required_samples_for_std(0.07, 100, 0.035) == pytest.approx(400, rel=1e-9)
    assert required_samples_for_std(0.07, 100, 0.0175) == pytest.approx(1600, rel=1e-9)


def test_cv_is_reported_with_a_confidence_interval():
    """sigma_C/mu_C is the headline statistic (it is what the mesh study shows
    to be converged), so it must carry its own interval rather than be a bare
    float that a reader has to eyeball."""
    x = np.random.default_rng(11).lognormal(0.0, 0.5, 800)
    summary = summarize_samples(x, n_bootstrap=2000, seed=0)
    cv = summary["cv"]
    assert cv["ci_low"] < cv["value"] < cv["ci_high"]
    assert cv["value"] == pytest.approx(np.std(x, ddof=1) / np.mean(x), rel=1e-12)


def test_paired_cv_interval_is_narrower_than_naive_propagation():
    """THE reason cv is bootstrapped as one statistic instead of propagated.

    mu and sigma come from the SAME draws and are positively correlated across
    resamples. Propagating their separate intervals ignores that and inflates
    the result -- enough, on the real l_c sweep, to make neighbouring levels
    look unresolvable when they are not.
    """
    x = np.random.default_rng(12).lognormal(0.0, 0.7, 800)
    s = summarize_samples(x, n_bootstrap=3000, seed=0)

    def half_width(est):
        return (est["ci_high"] - est["ci_low"]) / 2 / est["value"]

    naive = (half_width(s["mean"]) ** 2 + half_width(s["std"]) ** 2) ** 0.5
    assert half_width(s["cv"]) < naive, (
        f"paired cv half-width {half_width(s['cv']):.4f} should be below the "
        f"naive propagation bound {naive:.4f}"
    )


def test_cv_is_nan_rather_than_infinite_for_a_zero_mean_sample():
    """A zero-mean ensemble has no meaningful coefficient of variation. Return
    NaN so it propagates visibly into JSON instead of an inf that plots."""
    from src.validation.statistics import _coefficient_of_variation
    assert np.isnan(_coefficient_of_variation(np.array([-1.0, 1.0])))

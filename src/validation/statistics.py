"""Uncertainty quantification of the Monte Carlo estimates themselves.

WHY THIS MODULE EXISTS
----------------------
The pipeline reported mu_C and sigma_C as bare point estimates and compared
designs by their raw difference. On the last completed run that produced a
headline claim of a 6.2% reduction in sigma_C, validated with n_samples = 100 --
where the standard error of a sample standard deviation is

    se(sigma_hat) ~ sigma / sqrt(2 (N - 1)) ~ 7% relative

i.e. the error bar on the measurement was larger than the effect being claimed.
Nothing in the codebase computed that error bar, so there was no way to see it.

This module supplies:

  * bootstrap confidence intervals (BCa) for mean, std, and percentiles, so
    every reported statistic ships with its uncertainty;
  * PAIRED comparison of two designs evaluated under COMMON RANDOM NUMBERS
    (the same eta(x) realizations). Pairing is what makes a small difference
    resolvable at a feasible sample size: the two designs respond to the same
    draws in strongly correlated ways, so the variance of the DIFFERENCE is far
    smaller than the variance of either estimate. Comparing two independently
    seeded runs -- as the pipeline did -- throws that away;
  * a required-sample-size estimate, so "how many samples do I need for this
    claim" has a number instead of a guess.

Nothing here is MPI-aware: it operates on the compliance sample vectors, which
run_monte_carlo_validation already reduces to world-identical arrays.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)

DEFAULT_N_BOOTSTRAP = 10000
DEFAULT_CONFIDENCE = 0.95


@dataclass(frozen=True)
class Estimate:
    """A point estimate with a bootstrap confidence interval."""

    value: float
    ci_low: float
    ci_high: float
    standard_error: float
    confidence: float

    @property
    def half_width(self) -> float:
        return 0.5 * (self.ci_high - self.ci_low)

    @property
    def relative_half_width(self) -> float:
        """CI half-width as a fraction of the estimate -- the number to quote
        when asking whether an effect of a given size is even resolvable."""
        denominator = abs(self.value)
        return self.half_width / denominator if denominator > 0 else float("inf")

    def as_dict(self) -> dict:
        return {
            "value": self.value,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "standard_error": self.standard_error,
            "confidence": self.confidence,
            "relative_half_width": self.relative_half_width,
        }

    def __str__(self) -> str:
        return (
            f"{self.value:.6g} [{self.ci_low:.6g}, {self.ci_high:.6g}] "
            f"({self.confidence:.0%} CI)"
        )


def _bootstrap(
    samples: np.ndarray,
    statistic,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = 0,
) -> Estimate:
    """BCa bootstrap CI for an arbitrary statistic of a 1-D sample.

    BCa (bias-corrected and accelerated) rather than the percentile bootstrap
    because the sampling distribution of a standard deviation is skewed at the
    sample sizes used here, and the percentile interval is visibly off-centre
    for it.
    """
    samples = np.asarray(samples, dtype=float).ravel()
    n = samples.size
    if n < 2:
        raise ValueError(f"Need at least 2 samples for a bootstrap CI, got {n}.")

    result = scipy_stats.bootstrap(
        (samples,),
        statistic,
        n_resamples=n_bootstrap,
        confidence_level=confidence,
        method="BCa",
        random_state=np.random.default_rng(seed),
        vectorized=False,
    )
    return Estimate(
        value=float(statistic(samples)),
        ci_low=float(result.confidence_interval.low),
        ci_high=float(result.confidence_interval.high),
        standard_error=float(result.standard_error),
        confidence=confidence,
    )


def summarize_samples(
    samples: np.ndarray,
    percentiles: tuple[float, float] = (5.0, 95.0),
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = 0,
) -> dict:
    """Point estimates WITH confidence intervals for one design's MC ensemble.

    Args:
        samples: [n] compliance (or any scalar QoI) values.
        percentiles: The two percentile levels to report, as percentages.
        n_bootstrap: Bootstrap resample count.
        confidence: Confidence level for every interval.
        seed: RNG seed, so the intervals are reproducible.

    Returns:
        Dict of {statistic_name: Estimate.as_dict()} plus n_samples.
    """
    samples = np.asarray(samples, dtype=float).ravel()
    p_low, p_high = percentiles

    estimates = {
        "mean": _bootstrap(samples, np.mean, n_bootstrap, confidence, seed),
        "std": _bootstrap(samples, lambda a: np.std(a, ddof=1), n_bootstrap, confidence, seed + 1),
        f"p{p_low:g}": _bootstrap(
            samples, lambda a: np.percentile(a, p_low), n_bootstrap, confidence, seed + 2
        ),
        f"p{p_high:g}": _bootstrap(
            samples, lambda a: np.percentile(a, p_high), n_bootstrap, confidence, seed + 3
        ),
    }
    out = {name: est.as_dict() for name, est in estimates.items()}
    out["n_samples"] = int(samples.size)
    return out


@dataclass(frozen=True)
class PairedComparison:
    """Result of comparing two designs under common random numbers."""

    name_a: str
    name_b: str
    n_samples: int
    paired: bool
    delta_mean: Estimate
    delta_std: Estimate
    correlation: float
    p_value_mean: float
    mean_difference_resolvable: bool
    std_difference_resolvable: bool

    def as_dict(self) -> dict:
        return {
            "design_a": self.name_a,
            "design_b": self.name_b,
            "n_samples": self.n_samples,
            "paired_common_random_numbers": self.paired,
            "delta_mean_b_minus_a": self.delta_mean.as_dict(),
            "delta_std_b_minus_a": self.delta_std.as_dict(),
            "sample_correlation_between_designs": self.correlation,
            "p_value_mean_difference": self.p_value_mean,
            "mean_difference_resolvable": self.mean_difference_resolvable,
            "std_difference_resolvable": self.std_difference_resolvable,
            "verdict": self.verdict(),
        }

    def verdict(self) -> str:
        parts = []
        for label, resolvable, est in (
            ("mean", self.mean_difference_resolvable, self.delta_mean),
            ("std", self.std_difference_resolvable, self.delta_std),
        ):
            if resolvable:
                parts.append(
                    f"{label}: {self.name_b} differs from {self.name_a} by "
                    f"{est.value:+.4g} ({est.confidence:.0%} CI "
                    f"[{est.ci_low:+.4g}, {est.ci_high:+.4g}], excludes 0)"
                )
            else:
                parts.append(
                    f"{label}: difference {est.value:+.4g} is NOT resolvable at "
                    f"n={self.n_samples} ({est.confidence:.0%} CI "
                    f"[{est.ci_low:+.4g}, {est.ci_high:+.4g}] contains 0) -- "
                    f"do not claim an improvement"
                )
        if not self.paired:
            parts.append(
                "NOTE: designs were evaluated on DIFFERENT eta draws, so this "
                "is an unpaired comparison with much wider intervals than a "
                "common-random-number comparison would give"
            )
        return "; ".join(parts)


def compare_designs(
    samples_a: np.ndarray,
    samples_b: np.ndarray,
    name_a: str = "A",
    name_b: str = "B",
    paired: bool = True,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = 0,
) -> PairedComparison:
    """Compare two designs' compliance ensembles, with intervals on the DIFFERENCE.

    Args:
        samples_a, samples_b: [n] compliance values. When ``paired`` is True
            these must be aligned sample-for-sample -- entry i of each array
            must come from the SAME eta(x) realization. That alignment is the
            entire point: it is what shrinks the variance of the difference.
        name_a, name_b: Labels for reporting.
        paired: Whether the two ensembles share their random draws. Pass False
            only if they genuinely do not; the intervals will be much wider.
        n_bootstrap: Bootstrap resample count.
        confidence: Confidence level.
        seed: RNG seed.

    Returns:
        A PairedComparison whose ``verdict()`` states plainly whether the
        difference is resolvable at this sample size.

    Raises:
        ValueError: If paired is True and the two arrays differ in length.
    """
    samples_a = np.asarray(samples_a, dtype=float).ravel()
    samples_b = np.asarray(samples_b, dtype=float).ravel()

    if paired and samples_a.size != samples_b.size:
        raise ValueError(
            f"Paired comparison requires equal sample counts, got "
            f"{samples_a.size} and {samples_b.size}. Either evaluate both "
            "designs on the same eta draws, or pass paired=False."
        )

    rng = np.random.default_rng(seed)
    n = samples_a.size

    if paired:
        differences = samples_b - samples_a
        delta_mean = _bootstrap(differences, np.mean, n_bootstrap, confidence, seed)
        # The std difference is not a statistic of a single paired vector, so it
        # is bootstrapped by resampling the PAIRS (preserving the pairing) and
        # recomputing both stds on each resample.
        idx = rng.integers(0, n, size=(n_bootstrap, n))
        boot_delta_std = (
            np.std(samples_b[idx], axis=1, ddof=1) - np.std(samples_a[idx], axis=1, ddof=1)
        )
        observed_delta_std = float(np.std(samples_b, ddof=1) - np.std(samples_a, ddof=1))
        alpha = 1.0 - confidence
        lo, hi = np.percentile(boot_delta_std, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        delta_std = Estimate(
            value=observed_delta_std,
            ci_low=float(lo),
            ci_high=float(hi),
            standard_error=float(np.std(boot_delta_std, ddof=1)),
            confidence=confidence,
        )
        correlation = float(np.corrcoef(samples_a, samples_b)[0, 1]) if n > 1 else float("nan")
        p_value_mean = float(scipy_stats.wilcoxon(differences).pvalue) if n > 1 else float("nan")
    else:
        idx_a = rng.integers(0, samples_a.size, size=(n_bootstrap, samples_a.size))
        idx_b = rng.integers(0, samples_b.size, size=(n_bootstrap, samples_b.size))
        boot_delta_mean = samples_b[idx_b].mean(axis=1) - samples_a[idx_a].mean(axis=1)
        boot_delta_std = (
            np.std(samples_b[idx_b], axis=1, ddof=1) - np.std(samples_a[idx_a], axis=1, ddof=1)
        )
        alpha = 1.0 - confidence
        pcts = [100 * alpha / 2, 100 * (1 - alpha / 2)]

        lo, hi = np.percentile(boot_delta_mean, pcts)
        delta_mean = Estimate(
            value=float(samples_b.mean() - samples_a.mean()),
            ci_low=float(lo), ci_high=float(hi),
            standard_error=float(np.std(boot_delta_mean, ddof=1)),
            confidence=confidence,
        )
        lo, hi = np.percentile(boot_delta_std, pcts)
        delta_std = Estimate(
            value=float(np.std(samples_b, ddof=1) - np.std(samples_a, ddof=1)),
            ci_low=float(lo), ci_high=float(hi),
            standard_error=float(np.std(boot_delta_std, ddof=1)),
            confidence=confidence,
        )
        correlation = float("nan")
        p_value_mean = float(scipy_stats.mannwhitneyu(samples_a, samples_b).pvalue)
        n = min(samples_a.size, samples_b.size)

    def _excludes_zero(est: Estimate) -> bool:
        return est.ci_low > 0.0 or est.ci_high < 0.0

    comparison = PairedComparison(
        name_a=name_a,
        name_b=name_b,
        n_samples=int(n),
        paired=paired,
        delta_mean=delta_mean,
        delta_std=delta_std,
        correlation=correlation,
        p_value_mean=p_value_mean,
        mean_difference_resolvable=_excludes_zero(delta_mean),
        std_difference_resolvable=_excludes_zero(delta_std),
    )
    logger.info("Paired comparison %s vs %s: %s", name_a, name_b, comparison.verdict())
    return comparison


def required_samples_for_std(
    observed_std_relative_half_width: float,
    n_observed: int,
    target_relative_half_width: float,
) -> int:
    """How many MC samples are needed to resolve a given relative effect in sigma.

    The CI half-width of a standard deviation shrinks as 1/sqrt(N), so

        N_required = N_observed * (observed_half_width / target_half_width)^2

    Args:
        observed_std_relative_half_width: CI half-width of sigma at the sample
            size actually run, as a fraction of sigma (Estimate.relative_half_width).
        n_observed: Sample size that produced it.
        target_relative_half_width: The effect size you want to resolve, e.g.
            0.02 to claim a 2% change in sigma. Use HALF the effect you intend
            to claim, so the interval separates from zero.

    Returns:
        Required sample count, rounded up.
    """
    if target_relative_half_width <= 0:
        raise ValueError("target_relative_half_width must be > 0")
    ratio = observed_std_relative_half_width / target_relative_half_width
    return int(np.ceil(n_observed * ratio ** 2))


def resolvability_report(
    summary: dict,
    claimed_relative_effect: float,
) -> dict:
    """Can a claimed relative effect in sigma be resolved at the sample size run?

    Args:
        summary: Output of summarize_samples().
        claimed_relative_effect: The relative change in sigma being claimed,
            e.g. 0.062 for "sigma dropped 6.2%".

    Returns:
        A dict stating whether the run's own resolution supports the claim, and
        the sample size that would.
    """
    std_estimate = summary["std"]
    n = summary["n_samples"]
    observed = std_estimate["relative_half_width"]
    # To separate an effect from zero the CI half-width must be below half of it.
    target = 0.5 * abs(claimed_relative_effect)
    resolvable = observed <= target
    return {
        "claimed_relative_effect_in_std": claimed_relative_effect,
        "std_ci_relative_half_width": observed,
        "n_samples": n,
        "resolvable_at_this_n": bool(resolvable),
        "required_n_to_resolve": (
            n if resolvable else required_samples_for_std(observed, n, target)
        ),
        "note": (
            "An effect is only resolvable when the confidence interval on the "
            "estimate is narrower than half the effect. Otherwise the reported "
            "difference is within the measurement's own noise and must not be "
            "presented as an improvement."
        ),
    }

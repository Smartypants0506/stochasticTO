"""
src/sampling/splitter.py

Stage 4 (Experimental Design for Surrogate Training) --
implementation-modules.md Item 10 / fileDescription.md src/sampling/ section.

Distinct from sampler.py: this module does NOT draw new samples. It takes
an already-generated pool of KL coefficient samples (e.g. from
sampler.generate_samples with a single large n_samples) and performs a
reproducible shuffle + train/test split, logging the split indices to disk
so the exact partition can be reconstructed later -- required for
reproducibility per masterContext's "reproducibility of runs" rule.

Use this instead of sampler.generate_train_test_samples when you want a
single sampling design (e.g. one LHS pool) partitioned post hoc, rather
than two independently-drawn designs.
"""

"""
NOTE: This module is retained for standalone use cases but the canonical
sampling path for the robust loop is sampler.generate_train_test_samples(),
which draws INDEPENDENT train/test sets (LHS train, MC test) rather than
partitioning a single pool. See review item #13.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SplitResult:
    """Container for a reproducible train/test partition of a sample pool.

    Attributes:
        xi_train: [n_train x n_kl] training subset.
        xi_test: [n_test x n_kl] held-out test subset.
        train_indices: Indices into the original pool selected for training.
        test_indices: Indices into the original pool selected for testing.
        seed: RNG seed used for the shuffle.
    """
    xi_train: np.ndarray
    xi_test: np.ndarray
    train_indices: np.ndarray
    test_indices: np.ndarray
    seed: int


def split_samples(
    xi: np.ndarray,
    train_fraction: float = 0.8,
    seed: int = 0,
) -> SplitResult:
    """Shuffle and split a sample pool into training and held-out test sets.

    Args:
        xi: [n_samples x n_kl] full pool of KL coefficient samples, e.g.
            from sampler.generate_samples().
        train_fraction: Fraction of samples assigned to training
            (default 0.80, i.e. an 80/20 split per fileDescription.md).
        seed: RNG seed for the shuffle -- fixed so re-running this function
            on the same xi always yields the same partition.

    Returns:
        A SplitResult with the partitioned arrays and their source indices.

    Raises:
        ValueError: If xi is empty, or train_fraction is not in (0, 1).
    """
    n_samples = xi.shape[0]
    if n_samples == 0:
        raise ValueError("xi must contain at least one sample.")
    if not (0.0 < train_fraction < 1.0):
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}")

    rng = np.random.default_rng(seed)
    shuffled_indices = rng.permutation(n_samples)

    n_train = int(round(n_samples * train_fraction))
    # Guard against a degenerate split (e.g. tiny pools rounding to 0 or
    # n_samples) which would silently produce an empty train or test set.
    n_train = min(max(n_train, 1), n_samples - 1)

    train_indices = shuffled_indices[:n_train]
    test_indices = shuffled_indices[n_train:]

    logger.info(
        "Split %d samples into %d train / %d test (fraction=%.2f, seed=%d)",
        n_samples, len(train_indices), len(test_indices), train_fraction, seed,
    )

    return SplitResult(
        xi_train=xi[train_indices],
        xi_test=xi[test_indices],
        train_indices=train_indices,
        test_indices=test_indices,
        seed=seed,
    )


def save_split_indices(split: SplitResult, output_dir: str | Path) -> Path:
    """Persist split indices to results/split_indices.json for reproducibility.

    Per fileDescription.md: "logs split indices to the results directory."
    Only indices (not the sample values themselves) are saved -- the values
    are always recoverable by re-applying these indices to the original xi
    pool, keeping the log small and avoiding duplicated data on disk.

    Args:
        split: A SplitResult from split_samples().
        output_dir: Directory to write split_indices.json into; created if
            it does not already exist.

    Returns:
        Path to the written split_indices.json file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "split_indices.json"

    payload = {
        "seed": split.seed,
        "n_train": len(split.train_indices),
        "n_test": len(split.test_indices),
        "train_indices": split.train_indices.tolist(),
        "test_indices": split.test_indices.tolist(),
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    logger.info("Wrote split indices to %s", out_path)
    return out_path
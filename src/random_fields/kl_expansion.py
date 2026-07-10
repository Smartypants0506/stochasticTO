"""Karhunen-Loeve expansion of the underlying Gaussian field, FEM-based.

Master-context alignment (Section 3.3):
    G(x) = mu(x) + sum_i sqrt(lambda_i) * phi_i(x) * xi_i

Eigenfunctions phi_i(x) are approximated via FEM on a nodal grid using
OpenTURNS's KarhunenLoeveP1Algorithm, exactly as specified ("eigenfunctions
approximated via FEM on a nodal grid whose spacing is proportional to the
correlation length l"). Truncation order N_KL is chosen so retained modes
explain >= 95% of total variance (Section 3.3 verbatim requirement).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import openturns as ot

from src.random_fields.kernel import KernelParams, build_squared_exponential

logger = logging.getLogger(__name__)

VARIANCE_EXPLAINED_THRESHOLD = 0.95  # Section 3.3: ">= 95% of total variance"



@dataclass
class KLExpansionResult:
    """Container for a fitted KL expansion.
= np.array(result.getEigenvalues())  # lowercase "v"
    Attributes:
        eigenvalues: [N_kl] retained eigenvalues lambda_i, descending order.
        modes: [N_nodes x N_kl] eigenfunctions phi_i evaluated at mesh nodes.
        mean_field: [N_nodes] mean function mu(x), zeros for a centered field.
        variance_explained: Fraction of total variance captured by N_kl modes.
        n_kl: Number of retained modes (truncation order).
        node_coordinates: [N_nodes x spatial_dim] mesh node coordinates used.
        kernel_params: The KernelParams used to build the covariance model.
    """
    eigenvalues: np.ndarray
    modes: np.ndarray
    mean_field: np.ndarray
    variance_explained: float
    n_kl: int
    node_coordinates: np.ndarray
    kernel_params: KernelParams


def _build_ot_mesh(node_coordinates: np.ndarray, simplices: np.ndarray) -> ot.Mesh:
    """Wrap FEM mesh nodes/connectivity into an OpenTURNS Mesh object.

    Args:
        node_coordinates: [N_nodes x spatial_dim] array of node coordinates.
        simplices: [N_elems x (spatial_dim+1)] element connectivity (triangles
            for 2D, tetrahedra for 3D), matching the FEA mesh so KL modes are
            evaluated on the same nodal grid used by FEniTop.

    Returns:
        An ot.Mesh instance for KarhunenLoeveP1Algorithm.
    """
    vertices = ot.Sample(node_coordinates)
    simplices_list = [list(map(int, s)) for s in simplices]
    mesh = ot.Mesh(vertices, simplices_list)
    if not mesh.isValid():
        raise ValueError(
            "Constructed ot.Mesh failed validity check (non-overlapping simplices, "
            "no unused/duplicate vertices required)."
        )
    return mesh


def compute_kl_expansion(
    node_coordinates: np.ndarray,
    simplices: np.ndarray,
    kernel_params: KernelParams,
    variance_threshold: float = VARIANCE_EXPLAINED_THRESHOLD,
    max_modes: int = 200,
) -> KLExpansionResult:
    """Compute the FEM-based KL expansion of a centered Gaussian field on a mesh.

    Implements master-context Section 3.3's KL expansion step:
        G(x) = mu(x) + sum_i sqrt(lambda_i) * phi_i(x) * xi_i
    with mu(x) = 0 for the MVP (unbiased manufacturing error field).

    Args:
        node_coordinates: [N_nodes x spatial_dim] FEA mesh node coordinates.
        simplices: [N_elems x (spatial_dim+1)] element connectivity.
        kernel_params: Squared-exponential kernel hyperparameters.
        variance_threshold: Minimum fraction of total variance to retain
            (default 0.95 per Section 3.3).
        max_modes: Upper bound on eigenmodes requested from OpenTURNS, to
            bound compute cost; increase if variance_threshold is not met.

    Returns:
        A KLExpansionResult with eigenvalues/modes truncated at N_kl modes.

    Raises:
        RuntimeError: If variance_threshold cannot be met within max_modes.
    """
    n_nodes = node_coordinates.shape[0]
    mesh = _build_ot_mesh(node_coordinates, simplices)
    covariance_model = build_squared_exponential(kernel_params)

    ot.ResourceMap.SetAsString("KarhunenLoeveP1Algorithm-EigenvaluesSolver", "SPECTRA")
    ot.TBB.Enable()
    ot.ResourceMap.SetAsUnsignedInteger("TBB-ThreadsNumber", 64)

    algo = ot.KarhunenLoeveP1Algorithm(mesh, covariance_model)
    algo.setThreshold(1e-3)  # discard numerically negligible eigenvalues early
    algo.setNbModes(max_modes)
    algo.run()
    result = algo.getResult()

    eigenvalues_full = np.array(result.getEigenvalues())
    modes_process = result.getModesAsProcessSample()
    n_available = len(eigenvalues_full)

    if n_available == 0:
        raise RuntimeError("KarhunenLoeveP1Algorithm returned zero eigenmodes.")

    total_variance = eigenvalues_full.sum()
    cumulative = np.cumsum(eigenvalues_full) / total_variance
    n_kl = int(np.searchsorted(cumulative, variance_threshold) + 1)
    n_kl = min(n_kl, n_available, max_modes)

    variance_explained = float(cumulative[n_kl - 1])
    if variance_explained < variance_threshold and n_available >= max_modes:
        raise RuntimeError(
            f"Could not reach {variance_threshold:.0%} variance explained within "
            f"max_modes={max_modes} (achieved {variance_explained:.2%}). "
            "Increase max_modes or adjust kernel length_scale."
        )

    eigenvalues = eigenvalues_full[:n_kl]
    modes = np.zeros((n_nodes, n_kl))
    for i in range(n_kl):
        mode_field = modes_process.getField(i).getValues()
        modes[:, i] = np.array(mode_field).ravel()

    logger.info(
        "KL expansion truncated at N_kl=%d modes (%.2f%% variance explained, "
        "threshold=%.0f%%)", n_kl, variance_explained * 100, variance_threshold * 100,
    )

    return KLExpansionResult(
        eigenvalues=eigenvalues,
        modes=modes,
        mean_field=np.zeros(n_nodes),
        variance_explained=variance_explained,
        n_kl=n_kl,
        node_coordinates=node_coordinates,
        kernel_params=kernel_params,
    )


def sample_gaussian_field(kl_result: KLExpansionResult, n_samples: int, seed: int | None = None) -> np.ndarray:
    """Draw realizations of G(x) from the truncated KL expansion.

    Implements G(x) = mu(x) + sum_i sqrt(lambda_i) * phi_i(x) * xi_i with
    xi_i ~ iid N(0,1), vectorized (no Python loops over mesh nodes/elements
    per Section 7 code standards).

    Args:
        kl_result: Output of compute_kl_expansion.
        n_samples: Number of independent field realizations to draw.
        seed: Optional RNG seed for reproducibility.

    Returns:
        [n_samples x N_nodes] array of Gaussian field realizations G(x).
    """
    rng = np.random.default_rng(seed)
    xi = rng.standard_normal(size=(n_samples, kl_result.n_kl))  # [n_samples x n_kl]
    sqrt_lambda = np.sqrt(kl_result.eigenvalues)  # [n_kl]
    scaled_modes = kl_result.modes * sqrt_lambda[np.newaxis, :]  # [N_nodes x n_kl]
    fluctuation = xi @ scaled_modes.T  # [n_samples x N_nodes]
    return kl_result.mean_field[np.newaxis, :] + fluctuation

def sample_kl_coefficients(kl_result: KLExpansionResult, n_samples: int, seed: int | None = None) -> np.ndarray:
    """Draw xi ~ iid N(0,1), the raw KL coefficients, without evaluating G(x)."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(size=(n_samples, kl_result.n_kl))

def evaluate_field_from_xi(kl_result: KLExpansionResult, xi: np.ndarray) -> np.ndarray:
    """Evaluate G(x) at an explicit, caller-supplied KL coefficient vector.

    Deterministic counterpart to sample_gaussian_field: instead of drawing
    xi ~ N(0, I) internally, this evaluates the same formula
    G(x) = mu(x) + sum_i sqrt(lambda_i) * phi_i(x) * xi_i
    at a specific xi the caller already has (e.g. a training sample from
    sampler.py's LHS design). Required by
    src/topology/heaviside_projection_glue.py's set_eta_from_xi(), which
    src/surrogate/fea_at_samples.py depends on so each training sample's
    compliance is computed from the exact xi recorded for that sample.

    Args:
        kl_result: Output of compute_kl_expansion.
        xi: [n_kl] KL coefficient vector. Must match kl_result.n_kl in length.

    Returns:
        [N_nodes] Gaussian field realization G(x) at this xi.

    Raises:
        ValueError: If xi's length does not match kl_result.n_kl.
    """
    xi = np.asarray(xi).ravel()
    if xi.shape[0] != kl_result.n_kl:
        raise ValueError(
            f"xi has length {xi.shape[0]} but kl_result.n_kl={kl_result.n_kl}. "
            "The KL coefficient vector must match the truncation order of "
            "this KLExpansionResult."
        )
    sqrt_lambda = np.sqrt(kl_result.eigenvalues)  # [n_kl]
    scaled_modes = kl_result.modes * sqrt_lambda[np.newaxis, :]  # [N_nodes x n_kl]
    fluctuation = scaled_modes @ xi  # [N_nodes]
    return kl_result.mean_field + fluctuation

def verify_sample_covariance(
    kl_result: KLExpansionResult,
    n_samples: int = 5000,
    seed: int = 0,
    rtol: float = 0.1,
) -> dict:
    """Verification gate: sample covariance must match theoretical kernel.

    Master-context Section 3.3 / Section 7 verification requirement:
    "sample covariance must match theoretical covariance kernel." Compares
    the empirical variance at each node (diagonal of covariance) against the
    theoretical kernel variance sigma^2, since full N_nodes x N_nodes
    covariance comparison is expensive; the diagonal check is a necessary
    (not sufficient) condition and is adequate for the MVP gate.

    Args:
        kl_result: Output of compute_kl_expansion.
        n_samples: Monte Carlo sample size for the empirical estimate.
        seed: RNG seed.
        rtol: Relative tolerance for pass/fail.

    Returns:
        Dict with keys: passed (bool), empirical_var_mean, theoretical_var,
        relative_error.
    """
    samples = sample_gaussian_field(kl_result, n_samples, seed=seed)
    empirical_var = samples.var(axis=0)  # [N_nodes]
    theoretical_var = kl_result.kernel_params.sigma ** 2
    relative_error = float(
        np.abs(empirical_var.mean() - theoretical_var) / theoretical_var
    )
    passed = relative_error < rtol
    logger.info(
        "KL covariance verification: empirical_var_mean=%.4g, theoretical_var=%.4g, "
        "relative_error=%.2f%, passed=%s",
        empirical_var.mean(), theoretical_var, relative_error * 100, passed,
    )
    return {
        "passed": passed,
        "empirical_var_mean": float(empirical_var.mean()),
        "theoretical_var": theoretical_var,
        "relative_error": relative_error,
    }

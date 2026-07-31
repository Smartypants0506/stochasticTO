"""Karhunen-Loeve expansion of the underlying Gaussian field, FEM-based.

Master-context alignment (Section 3.3):
    G(x) = mu(x) + sum_i sqrt(lambda_i) * phi_i(x) * xi_i

Eigenfunctions phi_i(x) are approximated via FEM on a nodal grid using
OpenTURNS's KarhunenLoeveP1Algorithm, exactly as specified ("eigenfunctions
approximated via FEM on a nodal grid whose spacing is proportional to the
correlation length l"). Truncation order N_KL is chosen so retained modes
explain >= 95% of total variance (Section 3.3 verbatim requirement).

MPI note: the OpenTURNS KarhunenLoeveP1Algorithm solve is performed ONLY on
rank 0 (it requires the full serial node/simplex arrays, which only rank 0
holds -- see mesher.py's mesh_serial convention). The resulting eigenvalues/
modes/node_coordinates are then broadcast to every other rank via comm.bcast,
so all ranks end up with an identical KLExpansionResult without redundantly
repeating the (expensive, single-threaded-per-rank) eigensolve.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import openturns as ot
from mpi4py import MPI

from src.random_fields.kernel import KernelParams, build_squared_exponential

import hashlib
from pathlib import Path

logger = logging.getLogger(__name__)

VARIANCE_EXPLAINED_THRESHOLD = 0.95  # Section 3.3: ">= 95% of total variance"

_KL_CACHE_DIR = Path("output/cache/kl_expansion")


def _kl_cache_key(
    node_coordinates: np.ndarray,
    simplices: np.ndarray,
    kernel_params: KernelParams,
    variance_threshold: float,
    max_modes: int,
) -> str:
    """Deterministic cache key covering every input that affects the KL solve."""
    hasher = hashlib.sha256()
    hasher.update(node_coordinates.tobytes())
    hasher.update(simplices.tobytes())
    hasher.update(
        f"{kernel_params.sigma:.17g}|{kernel_params.length_scale:.17g}|"
        f"{kernel_params.spatial_dim}|{variance_threshold:.17g}|{max_modes}".encode()
    )
    return hasher.hexdigest()[:24]


def _kl_cache_path(cache_key: str) -> Path:
    # Key the filename by the cache_key so different meshes/kernels/thresholds
    # get distinct cache files (the previous hardcoded name collided across all
    # problems, which is why the cache had to stay disabled).
    return _KL_CACHE_DIR / f"kl_{cache_key}.npz"


def _load_kl_cache(cache_key: str) -> tuple | None:
    path = _kl_cache_path(cache_key)
    if not path.exists():
        return None
    logger.info("KL expansion cache hit: loading %s", path)
    data = np.load(path)
    return (
        data["eigenvalues"],
        data["modes"],
        data["mean_field"],
        float(data["variance_explained"]),
        int(data["n_kl"]),
    )


def _save_kl_cache(cache_key: str, eigenvalues, modes, mean_field, variance_explained, n_kl) -> None:
    _KL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _kl_cache_path(cache_key)
    np.savez(
        path,
        eigenvalues=eigenvalues,
        modes=modes,
        mean_field=mean_field,
        variance_explained=variance_explained,
        n_kl=n_kl,
    )
    logger.info("KL expansion cached to %s", path)

@dataclass
class KLExpansionResult:
    """Container for a fitted KL expansion.

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


def _compute_kl_expansion_local(
    node_coordinates: np.ndarray,
    simplices: np.ndarray,
    kernel_params: KernelParams,
    variance_threshold: float,
    max_modes: int,
) -> tuple:
    """Do the actual OpenTURNS KL solve on the calling rank (no MPI logic).

    Returns:
        (eigenvalues, modes, mean_field, variance_explained, n_kl) tuple of
        plain NumPy/float values, ready to be broadcast via comm.bcast.
    """
    n_nodes = node_coordinates.shape[0]
    mesh = _build_ot_mesh(node_coordinates, simplices)
    covariance_model = build_squared_exponential(kernel_params)

    ot.ResourceMap.SetAsString("KarhunenLoeveP1Algorithm-EigenvaluesSolver", "SPECTRA")
    ot.TBB.Enable()
    # NOTE: deliberately NOT setting "TBB-ThreadsNumber" here. Leaving it at
    # OpenTURNS's default lets it size itself to the single rank's allotted
    # cores rather than hardcoding a thread count that would oversubscribe
    # the node once multiple MPI ranks are active.

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

    mean_field = np.zeros(n_nodes)

    logger.info(
        "KL expansion truncated at N_kl=%d modes (%.2f%% variance explained, "
        "threshold=%.0f%%)", n_kl, variance_explained * 100, variance_threshold * 100,
    )

    return eigenvalues, modes, mean_field, variance_explained, n_kl


def compute_kl_expansion(
    node_coordinates: np.ndarray | None,
    simplices: np.ndarray | None,
    kernel_params: KernelParams,
    variance_threshold: float = VARIANCE_EXPLAINED_THRESHOLD,
    max_modes: int = 200,
    comm: MPI.Comm | None = None,
) -> KLExpansionResult:
    """Compute the FEM-based KL expansion of a centered Gaussian field on a mesh.

    Implements master-context Section 3.3's KL expansion step:
        G(x) = mu(x) + sum_i sqrt(lambda_i) * phi_i(x) * xi_i
    with mu(x) = 0 for the MVP (unbiased manufacturing error field).

    MPI behavior: only rank 0 performs the OpenTURNS eigensolve (it is the
    only rank guaranteed to hold a non-None, fully-populated
    node_coordinates/simplices pair -- see mesher.py's mesh_serial
    convention). All other ranks may safely pass node_coordinates=None and
    simplices=None; they will receive the identical KLExpansionResult via
    broadcast. This must be called collectively by every rank in `comm`
    (default MPI.COMM_WORLD) -- it is a collective operation, not a
    rank-0-only convenience function.

    Args:
        node_coordinates: [N_nodes x spatial_dim] FEA mesh node coordinates.
            Required (non-None) on rank 0 only.
        simplices: [N_elems x (spatial_dim+1)] element connectivity.
            Required (non-None) on rank 0 only.
        kernel_params: Squared-exponential kernel hyperparameters.
        variance_threshold: Minimum fraction of total variance to retain
            (default 0.95 per Section 3.3).
        max_modes: Upper bound on eigenmodes requested from OpenTURNS, to
            bound compute cost; increase if variance_threshold is not met.
        comm: MPI communicator to broadcast the result across. Defaults to
            MPI.COMM_WORLD.

    Returns:
        A KLExpansionResult with eigenvalues/modes truncated at N_kl modes,
        identical on every rank in `comm`.

    Raises:
        RuntimeError: If variance_threshold cannot be met within max_modes,
            or if rank 0 hits any other error during the eigensolve. Raised
            identically on every rank (not just rank 0) to avoid deadlocks.
        ValueError: If rank 0 is called with node_coordinates=None or
            simplices=None.
    """
    if comm is None:
        comm = MPI.COMM_WORLD

    error_message: str | None = None
    payload = None

    if comm.rank == 0:
        if node_coordinates is None or simplices is None:
            error_message = (
                "compute_kl_expansion: rank 0 requires non-None "
                "node_coordinates and simplices."
            )
        else:
            try:
                # Cache keyed by every input that affects the eigensolve, so a
                # hit is only returned for an identical problem. The eigensolve
                # is deterministic, so a cached result is bit-identical to a
                # fresh one -- no accuracy change, only recompute avoided.
                cache_key = _kl_cache_key(
                    node_coordinates, simplices, kernel_params,
                    variance_threshold, max_modes,
                )
                cached = _load_kl_cache(cache_key)
                if cached is not None:
                    eigenvalues, modes, mean_field, variance_explained, n_kl = cached
                    logger.info(
                        "KL expansion loaded from cache: N_kl=%d modes "
                        "(%.2f%% variance explained, threshold=%.0f%%)",
                        n_kl, variance_explained * 100, variance_threshold * 100,
                    )
                else:
                    eigenvalues, modes, mean_field, variance_explained, n_kl = (
                        _compute_kl_expansion_local(
                            node_coordinates, simplices, kernel_params,
                            variance_threshold, max_modes,
                        )
                    )
                    _save_kl_cache(
                        cache_key, eigenvalues, modes, mean_field, variance_explained, n_kl
                    )
                payload = (
                    eigenvalues, modes, mean_field, variance_explained, n_kl,
                    node_coordinates,
                )
            except Exception as exc:  # noqa: BLE001 - re-raised on all ranks below
                error_message = f"{type(exc).__name__}: {exc}"

    error_message = comm.bcast(error_message, root=0)
    if error_message is not None:
        raise RuntimeError(
            f"compute_kl_expansion failed on rank 0: {error_message}"
        )

    payload = comm.bcast(payload, root=0)
    eigenvalues, modes, mean_field, variance_explained, n_kl, node_coordinates = payload

    return KLExpansionResult(
        eigenvalues=eigenvalues,
        modes=modes,
        mean_field=mean_field,
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

def pointwise_std(kl_result: KLExpansionResult, eps: float = 1e-12) -> np.ndarray:
    """Exact pointwise standard deviation of the TRUNCATED KL field at each node.

    The truncated field is G(x) = mu(x) + sum_i sqrt(lambda_i) * phi_i(x) * xi_i
    with xi_i ~ iid N(0,1), so its pointwise variance is exactly
        v(x) = sum_i lambda_i * phi_i(x)^2.
    For the FULL (untruncated) expansion this equals the stationary kernel
    variance sigma^2; at variance_threshold < 1 it is smaller and spatially
    varying (the truncation removes more variance in some regions than others).

    Returning sqrt(v(x)) lets callers standardize G(x)/std(x) to an EXACT
    unit-variance N(0,1) at every node, which is what makes the downstream
    isoprobabilistic marginal transform reproduce its target Beta marginal
    exactly regardless of sigma or truncation level (see
    src/topology/heaviside_projection_glue.py). Vectorized (no node loops),
    matching evaluate_field_from_xi's style.

    Args:
        kl_result: Output of compute_kl_expansion.
        eps: Floor on the returned std to guard the rare near-zero-variance
            node (e.g. a boundary node the truncated basis barely resolves)
            against a divide-by-zero when standardizing.

    Returns:
        [N_nodes] array of pointwise standard deviations, floored at eps.
    """
    variance = (kl_result.modes ** 2) @ kl_result.eigenvalues  # [N_nodes]
    return np.sqrt(np.maximum(variance, eps * eps))


def build_uniform_eta_kl(kl_result: KLExpansionResult) -> KLExpansionResult:
    """Degenerate one-mode expansion whose realizations are spatially CONSTANT.

    This is the uniform-manufacturing-error control of Schevenels, Lazarov &
    Sigmund (CMAME 200:3613-3627, 2011) Section 3.1: eta as a random VARIABLE
    rather than a random FIELD. Their central finding is that, for a 2D
    compliant mechanism and a 2D heat sink, the design optimized against uniform
    errors is as robust to non-uniform errors as the design optimized against
    non-uniform ones -- the spatial correlation bought nothing. Until this
    project runs that comparison in 3D compliance it cannot claim its KL field
    is doing any work, and the ~170x cost of the sample-average loop over a
    scalar threshold is unjustified. See scripts/uniform_eta_baseline.py.

    WHY THIS NEEDS NO CHANGES ANYWHERE ELSE
    ---------------------------------------
    RandomFieldHeaviside divides G(x) by pointwise_std() BEFORE the marginal
    transform (heaviside_projection_glue.py, `self._field_std`), so the transform
    always receives an exact unit-variance normal and eta is exactly
    Beta(alpha,beta) on [eta_min,eta_max] regardless of the eigenvalues, the
    modes, or the truncation level. A single mode with a CONSTANT eigenfunction
    therefore gives

        G(x)          = sqrt(lambda_1) * 1 * xi_1        (constant in x)
        pointwise_std = sqrt(lambda_1)                   (constant in x)
        G(x)/std      = xi_1                             (one scalar N(0,1))
        eta(x)        = T(xi_1)                          (one scalar Beta draw)

    -- a spatially uniform threshold from the IDENTICAL marginal as the field
    arm. The SAA driver, the batched FEA, the projection and the transform are
    all reused byte-for-byte; only the expansion handed to them differs. That is
    what makes the two arms comparable: any difference in the result is the
    spatial correlation and nothing else. tests/test_uniform_eta_baseline.py
    pins both halves of that claim.

    The eigenvalue is set to sigma^2 so pointwise_std matches the stationary
    variance of the real field. The value is in fact inert -- it cancels in the
    standardization above -- but matching it keeps the artifact readable.

    Args:
        kl_result: The real, spatially-correlated expansion for this mesh. Its
            node_coordinates MUST be carried over unchanged, or the projection
            glue's local-dof-to-global-node coordinate matching will fail.

    Returns:
        A KLExpansionResult with n_kl=1 whose realizations are constant in space.
    """
    n_nodes = kl_result.node_coordinates.shape[0]
    return KLExpansionResult(
        eigenvalues=np.array([kl_result.kernel_params.sigma ** 2], dtype=float),
        modes=np.ones((n_nodes, 1), dtype=float),
        mean_field=np.zeros(n_nodes, dtype=float),
        # A rank-one expansion with a constant mode carries its whole field.
        variance_explained=1.0,
        n_kl=1,
        node_coordinates=kl_result.node_coordinates,
        kernel_params=kl_result.kernel_params,
    )


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
    rtol: float = 0.05,
) -> dict:
    """Verification gate: sampled pointwise variance must match the TRUNCATED
    expansion's analytic variance.

    WHAT THIS CHECKS, AND WHAT IT USED TO CHECK
    -------------------------------------------
    This function previously compared the empirical variance against the nominal
    kernel variance sigma^2. That was wrong on two counts:

      * Truncating at variance_threshold < 1 removes variance, and removes it
        unevenly across the domain, so the truncated field's pointwise variance
        is v(x) = sum_i lambda_i phi_i(x)^2 -- strictly below sigma^2 and
        spatially varying. Comparing against sigma^2 measures the truncation
        level, not an implementation error, and fails for a correct expansion.
      * The pipeline no longer samples this field directly. Downstream, the
        field is normalized by pointwise_std() before the marginal transform, so
        sigma cancels identically and the realized field has unit variance
        everywhere by construction.

    So the meaningful check is: does sampling reproduce the analytic variance of
    the truncated expansion? That is an implementation check with a definite
    right answer. The separate, and genuinely interesting, question of how far
    the truncated correlation sits from the target kernel is reported by
    src/validation/gates.py's kl_correlation gate, which is the version wired
    into the pipeline.

    Args:
        kl_result: Output of compute_kl_expansion.
        n_samples: Monte Carlo sample size for the empirical estimate.
        seed: RNG seed.
        rtol: Relative tolerance on the per-node variance mismatch.

    Returns:
        Dict with keys: passed, max_relative_error, mean_empirical_var,
        mean_analytic_var, truncated_vs_nominal_variance_ratio.
    """
    samples = sample_gaussian_field(kl_result, n_samples, seed=seed)
    empirical_var = samples.var(axis=0, ddof=1)          # [N_nodes]
    analytic_var = pointwise_std(kl_result) ** 2         # [N_nodes], exact

    relative_error = np.abs(empirical_var - analytic_var) / np.maximum(analytic_var, 1e-300)
    max_relative_error = float(relative_error.max())
    passed = bool(max_relative_error < rtol)

    nominal_var = kl_result.kernel_params.sigma ** 2
    truncation_ratio = float(analytic_var.mean() / nominal_var) if nominal_var > 0 else float("nan")

    logger.info(
        "KL variance verification: max_relative_error=%.3g (rtol=%.3g), "
        "mean empirical=%.4g vs analytic=%.4g, truncated/nominal variance "
        "ratio=%.4f, passed=%s",
        max_relative_error, rtol, empirical_var.mean(), analytic_var.mean(),
        truncation_ratio, passed,
    )
    return {
        "passed": passed,
        "max_relative_error": max_relative_error,
        "mean_empirical_var": float(empirical_var.mean()),
        "mean_analytic_var": float(analytic_var.mean()),
        "truncated_vs_nominal_variance_ratio": truncation_ratio,
    }
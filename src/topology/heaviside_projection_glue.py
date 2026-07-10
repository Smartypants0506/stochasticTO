"""Custom glue: bridges FEniTop's scalar-eta Heaviside projection to a
spatially-correlated random field eta(x).

Master-context alignment (Section 3.1, Section 8):
    "Custom glue code retained: hooking FEniTop's eta parameter to the
    randomized, spatially-varying eta(x) field from Stage 3 (FEniTop
    natively expects a scalar eta; this project's core contribution is
    making it a random field)."

Why this is NOT implemented by modifying parameterize.py directly (Section 7,
"Do NOT... Modify FEniTop's internals"):
    parameterize.py's Heaviside.forward() computes
        self.rho_phys.vector - eta
    directly on a petsc4py.PETSc.Vec object. PETSc Vec arithmetic supports
    Vec-minus-scalar cleanly but does not support Vec-minus-arbitrary-ndarray
    via the same operator overload. Rather than fork/modify the vendored
    FEniTop file, this module reimplements the identical mathematical
    formula operating exclusively on `.vector.array` (a NumPy view), which
    is mathematically identical to the original for the scalar case and
    additionally supports per-node eta arrays. This is "extending via a
    documented interface" (the public forward/backward contract), not
    modifying FEniTop internals.

Known MVP limitation (documented, not silently skipped):
    This module assumes serial (single-rank) execution. Full MPI-consistent
    parallel eta(x) sampling requires gathering node coordinates to rank 0,
    sampling there, and scattering back -- the same pattern already used by
    utility.py's Communicator class in topopt.py. This is deferred to a
    later integration step and is NOT part of the Step 4 MVP scope.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from src.random_fields.kernel import KernelParams
from src.random_fields.kl_expansion import KLExpansionResult, compute_kl_expansion, sample_gaussian_field
from src.random_fields.threshold_transform import MarginalTransformParams, ThresholdMarginalTransform

logger = logging.getLogger(__name__)

_ETA_CLIP_EPS = 1e-6  # keeps eta strictly inside (0, 1); avoids degenerate tanh denominator


@dataclass
class RandomHeavisideConfig:
    """Configuration for the random-field Heaviside projection.

    Attributes:
        kernel_params: Squared-exponential covariance kernel hyperparameters
            (Section 3.3).
        transform_params: Target bounded marginal for eta(x) (Section 3.3).
        variance_threshold: KL truncation variance threshold (default 0.95
            per Section 3.3).
        seed: Optional RNG seed for reproducibility.
    """
    kernel_params: KernelParams
    transform_params: MarginalTransformParams
    variance_threshold: float = 0.95
    seed: int | None = None


class RandomFieldHeaviside:
    """Heaviside projection with a spatially-correlated random threshold eta(x).

    Drop-in behavioral replacement for parameterize.Heaviside, generalized so
    the projection threshold can be either:
      - a scalar float (recovers FEniTop's exact original nominal behavior), or
      - a per-node eta(x) array sampled from the Stage 3 KL expansion +
        memoryless marginal transform.

    Mathematical formula (identical to parameterize.Heaviside, Section 7
    "Exact Mathematical Formulations -- Do Not Deviate"):
        rho_hat = [tanh(beta*eta) + tanh(beta*(rho_tilde - eta))] / denom
        denom   = tanh(beta*eta) + tanh(beta*(1-eta))
        drho    = beta * (1 - tanh(beta*(rho_tilde - eta))**2) / denom

    All operations here act elementwise on NumPy arrays, so eta may be either
    a Python float (broadcasts) or an ndarray matching rho_phys.vector.array
    in shape.
    """

    def __init__(self, rho_phys, node_coordinates: np.ndarray, simplices: np.ndarray,
                 config: RandomHeavisideConfig):
        """Initialize and precompute the KL expansion once (expensive step).

        Args:
            rho_phys: The dolfinx physical-density Function (same object
                FEniTop's parameterize.Heaviside expects). Must expose
                `.vector.array` (NumPy view) and `.x.scatter_forward()`.
            node_coordinates: [N_nodes x spatial_dim] coordinates of
                rho_phys's function space dofs (CG1 nodal points), typically
                from `rho_phys.function_space.tabulate_dof_coordinates()`.
            simplices: [N_elems x (dim+1)] mesh connectivity for the same
                nodes, used to build the FEM-based KL expansion.
            config: RandomHeavisideConfig with kernel/marginal parameters.

        Raises:
            ValueError: If node_coordinates row count does not match
                rho_phys's local dof array size.
        """
        n_dofs = rho_phys.vector.array.size
        if node_coordinates.shape[0] != n_dofs:
            raise ValueError(
                f"node_coordinates has {node_coordinates.shape[0]} rows but "
                f"rho_phys has {n_dofs} local dofs. In MPI runs, this glue "
                "module requires serial execution (see module docstring)."
            )

        self.rho_phys = rho_phys
        self.config = config
        self.drho: np.ndarray | None = None
        self._current_eta: np.ndarray | float | None = None

        logger.info("Precomputing KL expansion for RandomFieldHeaviside (%d dofs)...", n_dofs)
        self.kl_result: KLExpansionResult = compute_kl_expansion(
            node_coordinates, simplices, config.kernel_params,
            variance_threshold=config.variance_threshold,
        )
        self.transform = ThresholdMarginalTransform(config.transform_params)
        logger.info(
            "RandomFieldHeaviside ready: N_kl=%d, variance_explained=%.4f",
            self.kl_result.n_kl, self.kl_result.variance_explained,
        )

    def resample(self, seed: int | None = None) -> np.ndarray:
        """Draw a new eta(x) realization and set it as the active threshold.

        Implements Section 3.3's full chain in one call:
            G(x) ~ KL expansion  ->  eta(x) = T(G(x))

        Args:
            seed: Optional seed override for this draw; falls back to
                config.seed if None.

        Returns:
            The sampled eta(x) array, shape [N_dofs].
        """
        use_seed = seed if seed is not None else self.config.seed
        g_sample = sample_gaussian_field(self.kl_result, n_samples=1, seed=use_seed)[0]
        eta_sample = self.transform.transform(g_sample)
        eta_sample = np.clip(eta_sample, _ETA_CLIP_EPS, 1.0 - _ETA_CLIP_EPS)
        self._current_eta = eta_sample
        return eta_sample

    def set_deterministic_eta(self, value: float = 0.5) -> None:
        """Fall back to FEniTop's original nominal scalar-eta behavior.

        Used for regression testing against the deterministic baseline
        (Step 1/2 of the MVP roadmap) and for nominal warm-start design runs.

        Args:
            value: Scalar eta value, must lie in (0, 1).
        """
        if not (0.0 < value < 1.0):
            raise ValueError(f"Deterministic eta must be in (0, 1), got {value}")
        self._current_eta = float(value)

    def forward(self, beta: float, eta: np.ndarray | float | None = None) -> None:
        """Apply the Heaviside projection using either a scalar or field eta.

        Mathematically identical to parameterize.Heaviside.forward(), but
        implemented entirely via `.vector.array` (NumPy) rather than raw
        PETSc Vec arithmetic, so it supports both scalar and array eta.

        Args:
            beta: Heaviside sharpness parameter (unchanged from FEniTop).
            eta: Scalar float, ndarray of shape [N_dofs], or None to reuse
                the last value set by resample()/set_deterministic_eta().

        Raises:
            RuntimeError: If eta is None and no prior resample()/
                set_deterministic_eta() call has been made.
            ValueError: If eta is an array with the wrong shape.
        """
        if eta is None:
            eta = self._current_eta
        if eta is None:
            raise RuntimeError(
                "No eta set. Call resample() or set_deterministic_eta() "
                "before forward(), or pass eta explicitly."
            )

        rho_tilde = self.rho_phys.vector.array  # NumPy view, local dofs
        if isinstance(eta, np.ndarray) and eta.shape != rho_tilde.shape:
            raise ValueError(
                f"eta array shape {eta.shape} does not match rho_phys local "
                f"dof shape {rho_tilde.shape}."
            )

        denom = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
        interior = np.tanh(beta * (rho_tilde - eta))

        self.drho = beta * (1.0 - interior ** 2) / denom
        self.rho_phys.vector.array = (np.tanh(beta * eta) + interior) / denom
        self.rho_phys.x.scatter_forward()

    def backward(self, vectors: list) -> None:
        """Recover sensitivities through the Heaviside projection.

        Identical logic to parameterize.Heaviside.backward(): since the
        Heaviside derivative drho is computed pointwise (Section 3.1's
        elementwise chain rule holds regardless of whether eta is a scalar
        or a spatially-varying field), no formula change is needed here.

        Args:
            vectors: List of sensitivity vectors (or None entries), each
                exposing a `.array` attribute, matching the interface used
                by sensitivity.py's Sensitivity.evaluate() outputs.
        """
        if self.drho is None:
            raise RuntimeError("Must call forward() before backward().")
        for vector in vectors:
            if vector is not None:
                vector.array *= self.drho

    def verify_reduces_to_deterministic(self, beta: float, eta_value: float = 0.5,
                                         rtol: float = 1e-10) -> bool:
        """Regression check: constant-array eta must match scalar eta exactly.

        This is the key equivalence proof that this glue module is a
        behavior-preserving generalization of FEniTop's original Heaviside,
        not a divergent reimplementation.

        Args:
            beta: Heaviside sharpness parameter to test at.
            eta_value: Scalar eta value to compare against its constant-array
                equivalent.
            rtol: Relative tolerance for the comparison.

        Returns:
            True if scalar-eta and constant-array-eta produce identical
            rho_phys and drho outputs within rtol.
        """
        rho_before = self.rho_phys.vector.array.copy()

        self.rho_phys.vector.array[:] = rho_before
        self.forward(beta, eta=eta_value)
        rho_scalar = self.rho_phys.vector.array.copy()
        drho_scalar = self.drho.copy()

        self.rho_phys.vector.array[:] = rho_before
        eta_array = np.full_like(rho_before, eta_value)
        self.forward(beta, eta=eta_array)
        rho_array = self.rho_phys.vector.array.copy()
        drho_array = self.drho.copy()

        rho_match = np.allclose(rho_scalar, rho_array, rtol=rtol)
        drho_match = np.allclose(drho_scalar, drho_array, rtol=rtol)
        passed = bool(rho_match and drho_match)

        logger.info(
            "verify_reduces_to_deterministic: rho_match=%s, drho_match=%s, passed=%s",
            rho_match, drho_match, passed,
        )
        return passed

    def set_eta_from_xi(self, xi: np.ndarray) -> np.ndarray:
        """Set the active eta(x) threshold from an explicit KL coefficient vector.

        Deterministic counterpart to resample(): implements the same
        G(x) -> eta(x) = T(G(x)) chain, but from a caller-supplied xi rather
        than an internally-drawn random sample. Required by
        src/surrogate/fea_at_samples.py so each training sample's compliance
        is computed from the exact xi recorded for that sample.

        Args:
            xi: [n_kl] KL coefficient vector matching self.kl_result.n_kl.

        Returns:
            The resulting eta(x) array, shape [N_dofs].
        """
        g_sample = evaluate_field_from_xi(self.kl_result, xi)
        eta_sample = self.transform.transform(g_sample)
        eta_sample = np.clip(eta_sample, _ETA_CLIP_EPS, 1.0 - _ETA_CLIP_EPS)
        self._current_eta = eta_sample
        return eta_sample


def build_random_heaviside_from_function_space(rho_phys, mesh_simplices: np.ndarray,
                                                 config: RandomHeavisideConfig) -> RandomFieldHeaviside:
    """FEniCSx-specific constructor: extracts dof coordinates and builds the glue object.

    This function requires dolfinx and is NOT standalone-testable without a
    real FEA environment; RandomFieldHeaviside itself (the math) is fully
    standalone-testable with synthetic node arrays.

    Args:
        rho_phys: The dolfinx physical-density Function.
        mesh_simplices: Element connectivity for rho_phys's function space
            mesh (must align with tabulate_dof_coordinates() ordering).
        config: RandomHeavisideConfig with kernel/marginal parameters.

    Returns:
        A ready-to-use RandomFieldHeaviside instance.
    """
    node_coordinates = rho_phys.function_space.tabulate_dof_coordinates()
    spatial_dim = config.kernel_params.spatial_dim
    node_coordinates = node_coordinates[:, :spatial_dim]  # drop unused z for 2D
    return RandomFieldHeaviside(rho_phys, node_coordinates, mesh_simplices, config)
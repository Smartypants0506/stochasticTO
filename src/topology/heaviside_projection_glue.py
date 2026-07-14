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


MPI design (replaces the old serial-only limitation):
    The KL expansion itself (KLExpansionResult) is computed ONCE, collectively,
    via src/random_fields/kl_expansion.py's compute_kl_expansion(), which
    already broadcasts an identical result to every rank. Given that shared
    KLExpansionResult, sampling eta(x) does NOT require any further MPI
    communication: every rank deterministically evaluates the KL field at
    ALL global (serial) mesh nodes from a shared seed/xi -- cheap, since it
    is just a matrix-vector product over n_kl modes -- and then each rank
    slices out only the rows corresponding to its own LOCAL dofs, using a
    one-time coordinate-matched index map built in __init__ (the same
    cKDTree-based matching pattern already used by utility.py's
    Communicator class). This avoids gather/scatter entirely while still
    guaranteeing every rank uses a mutually consistent eta(x) realization.
"""
from __future__ import annotations


import logging
from dataclasses import dataclass


import numpy as np
from scipy.spatial import cKDTree


from src.random_fields.kernel import KernelParams
from src.random_fields.kl_expansion import KLExpansionResult, sample_gaussian_field, evaluate_field_from_xi
from src.random_fields.threshold_transform import MarginalTransformParams, ThresholdMarginalTransform


logger = logging.getLogger(__name__)


_ETA_CLIP_EPS = 1e-6  # keeps eta strictly inside (0, 1); avoids degenerate tanh denominator
_COORD_MATCH_PRECISION = 9  # decimal places for coordinate-based dof matching
_COORD_MATCH_TOL = 1e-6  # max allowed nearest-neighbor distance (matches CAD units, meters)



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


    def __init__(self, rho_phys, local_node_coordinates: np.ndarray,
                 kl_result: KLExpansionResult, config: RandomHeavisideConfig):
        """Initialize using an already-computed, MPI-shared KL expansion.


        Args:
            rho_phys: The dolfinx physical-density Function (same object
                FEniTop's parameterize.Heaviside expects). Must expose
                `.vector.array` (NumPy view) and `.x.scatter_forward()`.
            local_node_coordinates: [n_local_dofs x spatial_dim] coordinates
                of THIS RANK's local rho_phys dofs (CG1 nodal points),
                typically from `rho_phys.function_space.tabulate_dof_coordinates()`.
                Under MPI these are a strict subset of kl_result's global
                node_coordinates.
            kl_result: A KLExpansionResult already computed (and, under MPI,
                already broadcast identically to every rank) via
                src/random_fields/kl_expansion.py's compute_kl_expansion().
                This is NOT recomputed here.
            config: RandomHeavisideConfig with kernel/marginal parameters.
                Only transform_params/seed are used now; kernel_params and
                variance_threshold are informational (the actual KL solve
                already happened when kl_result was built).


        Raises:
            ValueError: If local_node_coordinates has the wrong row count
                for rho_phys's local dofs, or if any local dof coordinate
                cannot be matched to a global KL node within tolerance
                (which would indicate a real mesh/dof mismatch, not just
                an MPI partitioning artifact).
        """
        n_dofs = rho_phys.x.petsc_vec.array.size
        if local_node_coordinates.shape[0] != n_dofs:
            raise ValueError(
                f"local_node_coordinates has {local_node_coordinates.shape[0]} "
                f"rows but rho_phys has {n_dofs} local dofs. These must match "
                "1:1 on every rank."
            )

        self.rho_phys = rho_phys
        self.config = config
        self.kl_result = kl_result
        self.drho: np.ndarray | None = None
        self._current_eta: np.ndarray | float | None = None

        kd_tree = cKDTree(kl_result.node_coordinates.round(_COORD_MATCH_PRECISION))
        distances, local_to_global_idx = kd_tree.query(
            local_node_coordinates.round(_COORD_MATCH_PRECISION), k=1
        )
        bad = distances > _COORD_MATCH_TOL
        if np.any(bad):
            n_bad = int(np.sum(bad))
            raise ValueError(
                f"{n_bad}/{n_dofs} local dof coordinates could not be matched "
                f"to a global KL node within tolerance={_COORD_MATCH_TOL:.1e}. "
                "This indicates local_node_coordinates and kl_result.node_coordinates "
                "were built from inconsistent meshes (e.g. different dof "
                "orderings or a stale mesh_serial)."
            )
        self._local_to_global_idx = local_to_global_idx

        self.transform = ThresholdMarginalTransform(config.transform_params)
        logger.info(
            "RandomFieldHeaviside ready: N_kl=%d, variance_explained=%.4f, "
            "n_local_dofs=%d matched to global KL nodes",
            self.kl_result.n_kl, self.kl_result.variance_explained, n_dofs,
        )


    def resample(self, seed: int | None = None) -> np.ndarray:
        """Draw a new eta(x) realization and set it as the active threshold.


        Implements Section 3.3's full chain in one call:
            G(x) ~ KL expansion  ->  eta(x) = T(G(x))
        Every rank must call this with the SAME seed (directly or via
        config.seed) to remain MPI-consistent -- the full global field is
        evaluated identically on every rank, then sliced to this rank's
        local dofs.


        Args:
            seed: Optional seed override for this draw; falls back to
                config.seed if None. Must be identical across all ranks.


        Returns:
            The sampled eta(x) array restricted to this rank's local dofs,
            shape [n_local_dofs].
        """
        use_seed = seed if seed is not None else self.config.seed
        g_sample_global = sample_gaussian_field(self.kl_result, n_samples=1, seed=use_seed)[0]
        eta_global = self.transform.transform(g_sample_global)
        eta_global = np.clip(eta_global, _ETA_CLIP_EPS, 1.0 - _ETA_CLIP_EPS)
        eta_local = eta_global[self._local_to_global_idx]
        self._current_eta = eta_local
        return eta_local


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


        rho_tilde = self.rho_phys.x.petsc_vec.array  # NumPy view, local dofs
        if isinstance(eta, np.ndarray) and eta.shape != rho_tilde.shape:
            raise ValueError(
                f"eta array shape {eta.shape} does not match rho_phys local "
                f"dof shape {rho_tilde.shape}."
            )


        denom = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
        interior = np.tanh(beta * (rho_tilde - eta))


        self.drho = beta * (1.0 - interior ** 2) / denom
        self.rho_phys.x.petsc_vec.array = (np.tanh(beta * eta) + interior) / denom
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
        rho_before = self.rho_phys.x.petsc_vec.array.copy()


        self.rho_phys.x.petsc_vec.array[:] = rho_before
        self.forward(beta, eta=eta_value)
        rho_scalar = self.rho_phys.x.petsc_vec.array.copy()
        drho_scalar = self.drho.copy()


        self.rho_phys.x.petsc_vec.array[:] = rho_before
        eta_array = np.full_like(rho_before, eta_value)
        self.forward(beta, eta=eta_array)
        rho_array = self.rho_phys.x.petsc_vec.array.copy()
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
        is computed from the exact xi recorded for that sample. As with
        resample(), every rank must call this with the SAME xi to remain
        MPI-consistent -- xi is evaluated over the full global KL field
        first, then sliced to this rank's local dofs.


        Args:
            xi: [n_kl] KL coefficient vector matching self.kl_result.n_kl.
                Must be identical on every rank.


        Returns:
            The resulting eta(x) array restricted to this rank's local dofs,
            shape [n_local_dofs].
        """
        g_sample_global = evaluate_field_from_xi(self.kl_result, xi)
        eta_global = self.transform.transform(g_sample_global)
        eta_global = np.clip(eta_global, _ETA_CLIP_EPS, 1.0 - _ETA_CLIP_EPS)
        eta_local = eta_global[self._local_to_global_idx]
        self._current_eta = eta_local
        return eta_local



def build_random_heaviside_from_function_space(rho_phys, kl_result: KLExpansionResult,
                                                 config: RandomHeavisideConfig) -> RandomFieldHeaviside:
    """FEniCSx-specific constructor: extracts LOCAL dof coordinates and builds the glue object.

    tabulate_dof_coordinates() returns coordinates for every dof local to
    this rank, INCLUDING ghosts (needed for assembly). rho_phys.x.petsc_vec
    only exposes OWNED dofs (index_map.size_local), since PETSc's local
    array excludes ghosts. dolfinx orders local dofs with owned dofs first,
    followed by ghosts, so slicing to the first `size_local * block_size`
    rows recovers exactly the owned-dof coordinate set that matches
    rho_phys.x.petsc_vec.array 1:1.
    """
    dofmap = rho_phys.function_space.dofmap
    block_size = dofmap.index_map_bs
    num_owned_dofs = dofmap.index_map.size_local * block_size

    local_node_coordinates = rho_phys.function_space.tabulate_dof_coordinates()
    local_node_coordinates = local_node_coordinates[:num_owned_dofs, :]

    spatial_dim = config.kernel_params.spatial_dim
    local_node_coordinates = local_node_coordinates[:, :spatial_dim]  # drop unused z for 2D
    return RandomFieldHeaviside(rho_phys, local_node_coordinates, kl_result, config)
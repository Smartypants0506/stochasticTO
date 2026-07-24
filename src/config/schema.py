"""
src/config/schema.py

Typed dataclasses for the entire pipeline configuration. Every downstream
module receives one of these objects (via ProjectConfig), never a raw dict
read from disk -- config.yaml is parsed exactly once, in loader.py.

Reconstructed from confirmed field usages across config.yaml, loader.py,
fenitop_adapter.py, mesher.py's MeshingConfig pattern, kernel.py's
KernelParams, and threshold_transform.py's MarginalTransformParams, after
the original schema.py was found to be overwritten with build_mesh_from_step.py's
content. No field below is guessed -- each traces to a real call site.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MaterialConfig:
    """Isotropic linear-elastic material properties.

    Confirmed fields: config.yaml's material: block uses youngs_modulus,
    poissons_ratio [config.yaml]; fenitop_adapter.py reads
    config.material.youngs_modulus / config.material.poissons_ratio
    [fenitop_adapter.py].
    """
    youngs_modulus: float
    poissons_ratio: float


@dataclass
class PetscConfig:
    """PETSc solver options, passed through to FEniTop's linear solve.

    Confirmed fields: config.yaml's petsc: block uses ksp_type, pc_type
    [config.yaml]; fenitop_adapter.py calls config.petsc.to_options_dict()
    [fenitop_adapter.py] -- to_options_dict() is added below as a method.
    """
    ksp_type: str = "cg"
    pc_type: str = "gamg"
    # Guardrails matching box_source.py's _PETSC_OPTIONS so the STEP path gets
    # the same iterative-solver safety net (previously these were silently
    # dropped by to_options_dict, leaving the STEP path with unbounded KSP
    # iterations and no hard failure on non-convergence).
    ksp_max_it: int = 2000
    ksp_error_if_not_converged: bool = True

    def to_options_dict(self) -> dict:
        """Convert to the PETSc options dict form form_fem() expects."""
        opts = {"ksp_type": self.ksp_type, "pc_type": self.pc_type}
        if self.ksp_max_it is not None:
            opts["ksp_max_it"] = self.ksp_max_it
        # PETSc expects the key present (value None) to enable the flag.
        opts["ksp_error_if_not_converged"] = None if self.ksp_error_if_not_converged else False
        return opts


@dataclass
class LoadCase:
    """A single (facet group, traction vector) entry within a named load case.

    NOTE ON NESTING: config.yaml's `load_cases:` is now a mapping of
    case_name -> list[LoadCase], not a flat list -- e.g.:

        load_cases:
          vertical_up:
            - group_name: "load_1"
              vector: [0.0, 0.0, 9.34e7]
          torsion:
            - group_name: "load_1"
              vector: [0.0, -2.9e7, 0.0]

    This supports the common "same facet group(s), different vector per
    case" pattern (multiple independent load scenarios applied to the same
    tagged faces) as well as the "different groups per case" pattern, since
    each case's list can reference any group_name(s) it needs. ProjectConfig
    (below) stores this as `load_cases: dict[str, list[LoadCase]]`, and
    loader.py / fenitop_adapter.py / mapper.py / topopt.py all key off case
    name so each case is solved independently and summed
    (topopt.py::form_fem_multi_case), instead of all traction BCs being
    combined into one static-equilibrium RHS.
    """
    group_name: str
    vector: tuple[float, float, float]


@dataclass
class OptimizationConfig:
    """FEniTop nominal-SIMP and robust-loop optimization parameters.

    Confirmed fields: every key here matches config.yaml's optimization:
    block exactly (max_iter, opt_tol, vol_frac, penalty, epsilon,
    filter_radius, beta_interval, beta_max, use_oc, move, opt_compliance,
    quadrature_degree, body_force) [config.yaml], plus the fields
    fenitop_adapter.py's build_opt_dict/build_fem_dict actually read off
    config.optimization [fenitop_adapter.py]. pce_refresh_interval is a
    new field added for the robust-loop PCE refresh policy.
    """
    max_iter: int = 400
    opt_tol: float = 1e-5
    vol_frac: float = 0.5
    penalty: float = 3.0
    epsilon: float = 1e-6
    filter_radius: float = 1.2
    beta_interval: int = 50
    beta_max: float = 128.0
    use_oc: bool = True
    move: float = 0.02
    opt_compliance: bool = True
    quadrature_degree: int = 2
    body_force: tuple[float, float, float] = (0.0, 0.0, 0.0)
    pce_refresh_interval: int = 5
    lambda_sweep: list[float] = field(default_factory=lambda: [0.0, 0.5, 1.0, 2.0])
    pce_n_train_refresh: int = 60
    pce_n_test_refresh: int = 20
    # Sample-parallelism: ranks per sub-communicator group for the FEA-at-samples
    # / MC loops. COMM_WORLD is split into world_size // this many groups, each
    # solving a disjoint subset of samples concurrently on its own mesh. Must
    # divide world_size; set equal to world_size (or >= it) to disable grouping
    # and run the original single-group path. ~8 keeps each ~150k-DOF CG+GAMG
    # solve in its efficient regime while maximizing sample concurrency.
    sample_parallel_ranks_per_group: int = 8
    # PCE-refresh robustness: when a surrogate refit falls below the Q^2 gate,
    # add up to this many extra sample batches (accumulating) and refit before
    # giving up, capped at pce_n_train_escalation_cap total training samples.
    # If it still fails, a mid-solve refresh keeps the last valid surrogate and
    # stops refreshing (never crashes). Set pce_max_escalations=0 to restore the
    # old single-attempt behavior.
    pce_max_escalations: int = 3
    pce_n_train_escalation_cap: int = 800
    # Divergence guard: if the TRUE robust objective (from each refreshed
    # surrogate) rises for this many consecutive refreshes, the robust solve is
    # declared diverged -- refreshes freeze and the best feasible design seen so
    # far is returned instead of the runaway final iterate.
    pce_divergence_patience: int = 3
    # --- Surrogate-free Sample Average Approximation (SAA) robust-TO path ---
    # When robust_method="saa", Stage 4/5 skips the PCE entirely and, at every
    # MMA iteration, evaluates the EXACT sample-average robust objective/gradient
    # via saa_n_samples full FEA solves at the current design over one FIXED
    # sample set (common random numbers -> deterministic objective -> clean KKT
    # convergence). Higher saa_n_samples = higher Monte-Carlo fidelity, more
    # compute. saa_seed must be DISJOINT from mc_validation.seed so Stage-6 MC
    # gives an unbiased assessment (never validate on the optimization samples).
    robust_method: str = "saa"          # "saa" (surrogate-free) or "pce"
    saa_n_samples: int = 512            # fixed SAA sample-set size (fidelity knob)
    saa_sampling_strategy: str = "lhs"  # "lhs" (space-filling) or "monte_carlo"
    saa_seed: int = 7                   # keep != mc_validation.seed
    saa_beta: float = 8.0               # Heaviside sharpness for the SAA solves


@dataclass
class RandomFieldConfig:
    """Stage 3 random-field parameters (masterContext Section 3.3).

    Field names deliberately match KernelParams(sigma, length_scale,
    spatial_dim) [kernel-17.py] and MarginalTransformParams(eta_min,
    eta_max, alpha, beta) [threshold_transform-19.py] exactly, so
    fenitop_adapter.py can construct both objects directly from this
    config with no name translation -- avoiding the eta_min/lower and
    eta_max/upper naming bug caught during review.

    Attributes:
        sigma: Kernel marginal standard deviation (dimensionless).
        length_scale: Squared-exponential kernel correlation length l (m).
        spatial_dim: 2 or 3, matching the FEA mesh dimensionality.
        eta_min: Lower physical bound of the projection threshold.
        eta_max: Upper physical bound of the projection threshold.
        alpha: Beta distribution shape parameter alpha.
        beta: Beta distribution shape parameter beta.
        variance_threshold: KL truncation variance threshold (Section 3.3
            default 0.95).
        n_kl_hint: Expected/target number of retained KL modes, used only
            to size training sample counts before the real KL expansion
            is computed; the actual n_kl comes from KLExpansionResult.n_kl.
        seed: RNG seed for reproducible eta(x) sampling.
    """
    sigma: float
    length_scale: float
    spatial_dim: int = 2
    eta_min: float = 0.3
    eta_max: float = 0.7
    alpha: float = 2.0
    beta: float = 2.0
    variance_threshold: float = 0.95
    n_kl_hint: int = 20
    seed: int | None = None

@dataclass
class KeepAliveConfig:
    """Non-designable "keep-alive" backbone connecting all attachment
    instances into one connected component, independent of the optimizer's
    stress-based judgment (masterContext Section 3.2 assembly constraint).

    A minimum spanning tree is built over the attachment-instance anchor
    points, with edge weights equal to the *in-mesh geodesic* distance
    (Dijkstra over mesh edges), NOT straight-line distance -- a straight
    segment between two anchors on a bent/non-convex bracket can pass
    through void space. Each MST edge is then realized as a piecewise chain
    of cylindrical corridors following the geodesic path's waypoints, and
    OR-ed into solid_zone so those cells are hard-fixed to density 1.

    Attributes:
        enabled: Master on/off switch. When False, build_boundary_conditions
            leaves solid_zone untouched.
        groups: Facet-group names whose instances must all be connected
            (e.g. ["fixed", "load_1"]). Every instance across every listed
            group participates in a single shared MST, so bolts from
            different groups still end up in one connected component.
        corridor_radius: Cylinder radius (m) of each keep-alive segment.
            Should be >= a couple of element sizes so the corridor is
            actually meshed as solid. Defaults to a small multiple of
            mesh_size_max at the call site if left None.
        cluster_eps: Radius (m) for single-linkage clustering of a group's
            facet points into distinct physical instances (separate holes).
            Points within cluster_eps of each other are one instance.
            Defaults to ~2x mesh_size_max at the call site if left None.
    """
    enabled: bool = False
    groups: list[str] = field(default_factory=list)
    corridor_radius: float | None = None
    cluster_eps: float | None = None


@dataclass
class SurrogateConfig:
    """Stage 4 PCE surrogate parameters (masterContext Section 3.4).

    Attributes:
        n_train: Number of LHS training samples per PCE fit.
        n_test: Number of held-out Monte Carlo test samples per PCE fit,
            disjoint from n_train, used for the Q^2 gate (pce_builder.py's
            build_pce_surrogate requires xi_test/c_test to be independent
            of the training draw).
        hyperbolic_q: Hyperbolic truncation q for the sparse PCE index set.
        max_degree: Maximum polynomial degree tried before giving up on
            the Q^2 >= q2_threshold gate.
        q2_threshold: Minimum predictive Q^2 on held-out test data
            required before deployment (Section 7 hard gate, default 0.99).
    """
    n_train: int = 200
    n_test: int = 200
    hyperbolic_q: float = 0.75
    max_degree_attempts: int = 4
    q2_threshold: float = 0.95

@dataclass
class MonteCarloValidationConfig:
    """Stage 6 full-scale MC validation parameters (masterContext Section 3.6)."""
    n_samples: int = 5000
    beta: float = 128.0  # must match config.optimization.beta_max, not an independent guess
    seed: int = 0
    percentile_low: float = 5.0
    percentile_high: float = 95.0
    output_dir: str = "output/mc_validation"

@dataclass
class BoxMeshConfig:
    """Synthetic box-mesh source (bypasses STEP import), used to validate
    this project's multi-load-case topopt() pipeline against FEniTop's own
    scripts/beam_3d.py reference case. Only cell_type is exposed as a
    knob -- domain extents / element counts / BC placement are
    intentionally hardcoded in src/meshing/box_source.py to match
    beam_3d.py bit-for-bit; making those config-driven would defeat the
    point of a fixed known-good reference case.
 
    Attributes:
        cell_type: "tetrahedron" (default) or "hexahedron".
            beam_3d.py itself uses hexahedron elements, but its topopt()
            call is single-case Stage-2-only (no random-field stage).
            This project's Stage 3+ (compute_kl_expansion /
            extract_simplices) builds an OpenTURNS FEM mesh that requires
            SIMPLICES, so "hexahedron" is only valid if you stop after
            Stage 2 -- main.py raises if mesh_source="box",
            cell_type="hexahedron", and the pipeline proceeds past the
            nominal topopt warm-start.
    """
    cell_type: str = "tetrahedron"



@dataclass
class ProjectConfig:
    """Top-level, fully validated configuration object.

    Confirmed fields: step_file, mesh_out_path, mesh_size_max, snap_tol,
    material, petsc, load_cases, optimization, color_targets,
    solid_volume_color all come directly from loader.py's load_config()
    constructor call [loader-2.py]. random_field and surrogate are new,
    required for the Stage 3/4/5 driver wiring.

    load_cases is a dict[case_name, list[LoadCase]] (see LoadCase's
    docstring) -- NOT a flat list -- so each named case can be solved as
    its own independent equilibrium problem and summed, rather than all
    traction BCs being combined into a single static-equilibrium RHS.
    """
    step_file: str
    mesh_out_path: str
    mesh_size_max: float
    snap_tol: float
    material: MaterialConfig
    petsc: PetscConfig
    load_cases: dict[str, list[LoadCase]]
    optimization: OptimizationConfig
    random_field: RandomFieldConfig
    surrogate: SurrogateConfig
    color_targets: dict = field(default_factory=dict)
    solid_volume_color: tuple[int, int, int, int] = (255, 255, 0, 255)
    mc_validation: MonteCarloValidationConfig = field(default_factory=MonteCarloValidationConfig)
    keep_alive: KeepAliveConfig = field(default_factory=KeepAliveConfig)
    mesh_source: str = "step"
    box_mesh: BoxMeshConfig = field(default_factory=BoxMeshConfig)

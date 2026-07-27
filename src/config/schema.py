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
    saa_beta: float = 8.0               # FIRST stage of the Heaviside continuation
    # --- Heaviside continuation for the robust solve ---
    # The robust loop previously ran at a FIXED saa_beta=8 while its warm start
    # had been continued to beta=128, leaving the reported design substantially
    # gray and breaking the "eta = boundary offset" interpretation, which only
    # holds in the sharp-projection limit. The solve now runs one converged
    # stage per beta in [saa_beta, 2*saa_beta, ..., saa_beta_max], splitting
    # max_iter across stages so the total FEA cost is unchanged.
    saa_beta_max: float = 128.0
    saa_beta_continuation: bool = True
    # --- robust-solve convergence ---
    # DISTINCT from opt_tol, which is the nominal Stage-2 OC loop's design-CHANGE
    # threshold. These two had been sharing one key despite meaning different
    # things. robust_opt_tol bounds the RELATIVE stationarity residual and
    # constraint_tol bounds feasibility and complementarity; all three must hold
    # before the robust solve is called converged (src/optimization/optimality.py).
    robust_opt_tol: float = 1.0e-3
    constraint_tol: float = 1.0e-4
    # Pareto sweep policy. "common" solves every lambda from the SAME nominal
    # warm start; "continuation" warm-starts each lambda from the previous one.
    # "common" is the default because continuation produced a sweep whose
    # lambda=1 point DOMINATED its lambda=0 point in BOTH mu_C and sigma_C --
    # impossible for a genuine trade-off curve, and proof that the points were
    # just successive iterates of one under-converged descent rather than
    # distinct optima. Use "continuation" only with sweep_check_dominance on.
    lambda_sweep_start: str = "common"
    # Fail the run when the sweep contains a dominated point. Leave this on: it
    # is the automatic check for exactly the defect described above.
    sweep_check_dominance: bool = True


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
    beta: float = 128.0  # must match optimization.saa_beta_max, not an independent guess
    seed: int = 0
    percentile_low: float = 5.0
    percentile_high: float = 95.0
    output_dir: str = "output/mc_validation"
    # Bootstrap settings for the confidence intervals attached to every
    # reported statistic. A bare point estimate is not reportable at these
    # sample sizes: at n=100 the CI half-width on sigma is ~7% relative, which
    # is wider than the sigma reductions this pipeline has been claiming.
    n_bootstrap: int = 10000
    confidence: float = 0.95
    bootstrap_seed: int = 0
    # Write a full-resolution per-sample VTU ensemble. Off by default: at
    # n_samples in the thousands this dominates Stage 6 wall-clock and is a
    # visualization artifact, not evidence.
    write_ensemble: bool = False
    # Fraction of MC realizations allowed to fail to solve before the run halts.
    # A failure at the eroded end of the eta band means that realization does
    # not carry load -- a robustness RESULT, always reported, but it also makes
    # the compliance statistics conditional on survival and therefore
    # optimistically biased. Default 0.0 forces the tolerance to be an explicit
    # per-config decision rather than a silent allowance.
    max_solver_failure_rate: float = 0.0


@dataclass
class ValidationConfig:
    """Mandatory verification gates (src/validation/gates.py).

    These exist because the project's verification routines -- the FD gradient
    check above all -- were present in the tree but called from nothing, so no
    published number had ever been verified. run_gates defaults to True and
    should stay there; a run with gates disabled is not reportable.
    """
    run_gates: bool = True
    gradient_fd: bool = True
    fd_n_samples: int = 8
    fd_n_elements: int = 16
    fd_step: float = 1.0e-3
    fd_rtol: float = 1.0e-3
    fd_ksp_rtol: float = 1.0e-12
    correlation_n_nodes: int = 64
    correlation_n_samples: int = 4000
    correlation_rtol: float = 0.05
    marginal_n_nodes: int = 128
    marginal_n_samples: int = 2000
    marginal_alpha: float = 0.01

@dataclass
class BoxMeshConfig:
    """Synthetic box-mesh source (bypasses STEP import), used to validate
    this project's multi-load-case topopt() pipeline against FEniTop's own
    scripts/beam_3d.py reference case. Domain extents, BC placement,
    material and SIMP parameters are intentionally hardcoded in
    src/meshing/box_source.py to match beam_3d.py bit-for-bit; making those
    config-driven would defeat the point of a fixed known-good reference case.

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
        refinement: Multiplier on beam_3d.py's [25, 75, 25] element counts.
            1.0 reproduces the reference mesh exactly (h = 0.4). 0.64 gives
            the study tier (16x48x16, h = 0.625, roughly 4x cheaper per
            solve). This is the ONLY geometric knob, and it exists so a
            mesh-convergence study is possible at all -- the element counts
            were previously hardcoded, which made the study undoable and
            forced every experiment to pay production cost.

            The DOMAIN and the FILTER RADIUS are identical at every level --
            only h changes. That is deliberate: it is what makes the study
            measure convergence to the continuum problem. Scaling the filter
            with the mesh would shrink the minimum feature size at every
            level, so the design, its compliance and its sigma_C would keep
            moving and never converge to anything. See
            box_source._FILTER_RADIUS.
        load_patch_halfwidth: Half-width of the traction patch, or None (the
            default) for the automatic per-mesh rule max(0.5, 1.5h).

            The automatic rule exists because dolfinx selects a facet only when
            ALL its vertices satisfy the predicate, so beam_3d's fixed 0.5
            half-width admits NO facet on most meshes -- the traction is then
            applied to nothing and the compliance comes back exactly zero, with
            nothing raised. The rule guarantees at least two nodes inside at any
            h, and at h=0.4 still selects beam_3d's own facets exactly.

            But the automatic rule makes the patch SHRINK as the mesh refines
            (1.25 at h=0.833 down to 0.6 at h=0.4), so the load concentrates
            progressively and each refinement level solves a slightly different
            problem. A MESH-CONVERGENCE STUDY MUST PIN THIS to one value across
            all its levels -- scripts/convergence_studies.py does so
            automatically -- for the same reason the filter radius is held
            fixed: otherwise the study converges to nothing while looking clean.
    """
    cell_type: str = "tetrahedron"
    refinement: float = 1.0
    load_patch_halfwidth: float | None = None



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

    STEP-PATH FIELDS ARE OPTIONAL. step_file / mesh_out_path / mesh_size_max /
    snap_tol / material / petsc / load_cases / color_targets /
    solid_volume_color / keep_alive are consumed ONLY when mesh_source ==
    "step". On the box path every one of them is inert -- box_source.py supplies
    its own geometry, material and load -- so loader.py REQUIRES them for the
    STEP path and REJECTS them for the box path, rather than letting a reader
    mistake an unused config value for the one that produced the results.
    """
    optimization: OptimizationConfig
    random_field: RandomFieldConfig
    surrogate: SurrogateConfig
    mesh_source: str = "step"
    box_mesh: BoxMeshConfig = field(default_factory=BoxMeshConfig)
    mc_validation: MonteCarloValidationConfig = field(default_factory=MonteCarloValidationConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    # --- STEP/CAD path only (None on the box path) ---
    step_file: str | None = None
    mesh_out_path: str | None = None
    mesh_size_max: float | None = None
    snap_tol: float | None = None
    material: MaterialConfig | None = None
    petsc: PetscConfig | None = None
    load_cases: dict[str, list[LoadCase]] = field(default_factory=dict)
    color_targets: dict = field(default_factory=dict)
    solid_volume_color: tuple[int, int, int, int] = (255, 255, 0, 255)
    keep_alive: KeepAliveConfig = field(default_factory=KeepAliveConfig)

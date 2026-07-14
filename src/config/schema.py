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

    def to_options_dict(self) -> dict:
        """Convert to the PETSc options dict form form_fem() expects."""
        return {"ksp_type": self.ksp_type, "pc_type": self.pc_type}


@dataclass
class LoadCase:
    """A single named traction load case.

    Confirmed fields: config.yaml's load_cases: list uses group_name,
    vector [config.yaml]; fenitop_adapter.py's build_fem_dict zips
    config.load_cases against bc.traction_bcs using lc.vector
    [fenitop_adapter.py].
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
class ProjectConfig:
    """Top-level, fully validated configuration object.

    Confirmed fields: step_file, mesh_out_path, mesh_size_max, snap_tol,
    material, petsc, load_cases, optimization, color_targets,
    solid_volume_color all come directly from loader.py's load_config()
    constructor call [loader-2.py]. random_field and surrogate are new,
    required for the Stage 3/4/5 driver wiring.
    """
    step_file: str
    mesh_out_path: str
    mesh_size_max: float
    snap_tol: float
    material: MaterialConfig
    petsc: PetscConfig
    load_cases: list[LoadCase]
    optimization: OptimizationConfig
    random_field: RandomFieldConfig
    surrogate: SurrogateConfig
    color_targets: dict = field(default_factory=dict)
    solid_volume_color: tuple[int, int, int, int] = (255, 255, 0, 255)
    mc_validation: MonteCarloValidationConfig = field(default_factory=MonteCarloValidationConfig)
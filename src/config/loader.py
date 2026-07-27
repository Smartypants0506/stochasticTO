"""
src/config/loader.py

Loads config/config.yaml into a validated ProjectConfig. This is the ONLY
place in the codebase that should call open()/yaml.safe_load() for pipeline
parameters -- every downstream module receives a ProjectConfig object, never
a raw dict pulled from disk, so parameter provenance stays traceable.

THE CONFIG MUST DESCRIBE THE RUN
--------------------------------
This loader now REJECTS any key that is inert on the selected mesh_source,
instead of silently accepting it.

The motivating defect: with mesh_source: "box", src/meshing/box_source.py
hardcodes the beam_3d reference physics and ignores config.yaml's
optimization.vol_frac (0.15 -> 0.08), optimization.filter_radius
(0.006 -> 0.6), optimization.opt_tol (1e-3 -> 1e-5), material.youngs_modulus
(68.9 GPa -> 100), and the six SI load_cases (-> one hardcoded tip load).
Every one of those was sitting in the config file looking authoritative, and
anyone reconstructing the study from it would have reported the wrong value for
essentially every parameter of the run.

A config value that does not affect the run is worse than an absent one, so
this is a hard error with a message naming the key and where the effective
value actually comes from.
"""
from __future__ import annotations

import logging
from pathlib import Path

import yaml

from src.config.schema import (
    BoxMeshConfig, KeepAliveConfig, LoadCase, MaterialConfig, MonteCarloValidationConfig,
    OptimizationConfig, PetscConfig, ProjectConfig, RandomFieldConfig,
    SurrogateConfig, ValidationConfig,
)

logger = logging.getLogger(__name__)

# Consumed on every path.
_COMMON_TOP_KEYS = {
    "mesh_source", "optimization", "random_field", "surrogate",
    "mc_validation", "validation",
}

# Consumed only when mesh_source == "step".
_STEP_ONLY_TOP_KEYS = {
    "step_file", "mesh_out_path", "mesh_size_max", "snap_tol", "material",
    "petsc", "load_cases", "color_targets", "solid_volume_color", "keep_alive",
}
_STEP_REQUIRED_TOP_KEYS = {
    "step_file", "mesh_out_path", "mesh_size_max", "snap_tol", "material",
    "load_cases",
}

# Consumed only when mesh_source == "box".
_BOX_ONLY_TOP_KEYS = {"box_mesh"}

# optimization.* keys that box_source.py hardcodes, mapped to the value it uses.
# Setting any of these in a box-path config is an error: the value in the file
# would not be the value in the run.
_BOX_INERT_OPTIMIZATION_KEYS = {
    "vol_frac": "box_source._VOL_FRAC",
    "penalty": "box_source._PENALTY",
    "filter_radius": "box_source._FILTER_RADIUS",
    # opt_tol here is the NOMINAL Stage-2 OC design-change threshold, frozen
    # beam_3d run control. The robust solve's stationarity tolerance is the
    # separate, config-driven optimization.robust_opt_tol.
    "opt_tol": "box_source._OPT_TOL",
    "beta_interval": "box_source._BETA_INTERVAL",
    "beta_max": "box_source._BETA_MAX",
    "use_oc": "box_source._USE_OC",
    "move": "box_source._MOVE",
    "opt_compliance": "box_source._OPT_COMPLIANCE",
    "quadrature_degree": "box_source._QUADRATURE_DEGREE",
    "body_force": "box_source._BODY_FORCE",
}


class ConfigError(ValueError):
    """Raised for a config that would not describe the run it configures."""


def _reject_unknown_and_inert(raw: dict, mesh_source: str) -> None:
    """Fail on any top-level key that is unknown, or inert on this mesh_source."""
    known = _COMMON_TOP_KEYS | _STEP_ONLY_TOP_KEYS | _BOX_ONLY_TOP_KEYS
    unknown = set(raw) - known
    if unknown:
        raise ConfigError(
            f"Unknown top-level config key(s): {sorted(unknown)}. Nothing reads "
            "them, so they cannot affect the run. Remove them (or add them to "
            "the schema if they are meant to do something). Note that the "
            "'recompute:' block was removed along with the caching it "
            "controlled -- src/mainClean.py always runs every stage."
        )

    if mesh_source == "step":
        missing = _STEP_REQUIRED_TOP_KEYS - set(raw)
        if missing:
            raise ConfigError(
                f"mesh_source='step' requires config key(s): {sorted(missing)}."
            )
        inert = _BOX_ONLY_TOP_KEYS & set(raw)
        if inert:
            raise ConfigError(
                f"Config key(s) {sorted(inert)} are only read when "
                "mesh_source='box', but mesh_source='step'. Remove them."
            )
        return

    # box path
    inert = _STEP_ONLY_TOP_KEYS & set(raw)
    if inert:
        raise ConfigError(
            f"Config key(s) {sorted(inert)} are only read when "
            "mesh_source='step', but mesh_source='box'. On the box path the "
            "geometry, material, loads and boundary conditions all come from "
            "src/meshing/box_source.py, which hardcodes the FEniTop beam_3d "
            "reference case. Leaving these in the file makes it look as though "
            "they configured the run when they did not. Remove them."
        )

    optimization = raw.get("optimization") or {}
    inert_opt = sorted(set(optimization) & set(_BOX_INERT_OPTIMIZATION_KEYS))
    if inert_opt:
        detail = "; ".join(
            f"optimization.{k} (effective value comes from {_BOX_INERT_OPTIMIZATION_KEYS[k]})"
            for k in inert_opt
        )
        raise ConfigError(
            "On the box path these optimization keys are hardcoded in "
            f"src/meshing/box_source.py and IGNORED here: {detail}. They were "
            "the source of a config that disagreed with its own run (e.g. "
            "vol_frac 0.15 in the file, 0.08 in the solver). Remove them from "
            "config.yaml and change the values in box_source.py, which is the "
            "single source of truth for the frozen beam_3d reference physics."
        )


def load_config(path: str | Path) -> ProjectConfig:
    """Parse and validate config.yaml.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    ConfigError
        If a required key is missing, unknown, or inert on the selected
        mesh_source -- fails loudly rather than silently defaulting or silently
        ignoring, since a config that does not describe its own run makes every
        reported parameter unverifiable.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ConfigError(f"{path} did not parse to a mapping.")

    mesh_source = raw.get("mesh_source", "step")
    if mesh_source not in ("step", "box"):
        raise ConfigError(
            f"config.yaml's mesh_source must be 'step' or 'box', got {mesh_source!r}"
        )

    _reject_unknown_and_inert(raw, mesh_source)

    for required in ("optimization", "random_field"):
        if required not in raw:
            raise ConfigError(f"config.yaml is missing required key: {required!r}")

    optimization = OptimizationConfig(**raw["optimization"])
    random_field = RandomFieldConfig(**raw["random_field"])
    surrogate = SurrogateConfig(**raw.get("surrogate", {}))
    mc_validation = MonteCarloValidationConfig(**raw.get("mc_validation", {}))
    validation = ValidationConfig(**raw.get("validation", {}))
    box_mesh = BoxMeshConfig(**raw.get("box_mesh", {}))

    _validate_cross_field(optimization, random_field, mc_validation)

    material = MaterialConfig(**raw["material"]) if "material" in raw else None
    petsc = PetscConfig(**raw["petsc"]) if "petsc" in raw else None

    load_cases: dict[str, list[LoadCase]] = {}
    if "load_cases" in raw:
        raw_load_cases = raw["load_cases"]
        if not isinstance(raw_load_cases, dict):
            raise ConfigError(
                "config.yaml's load_cases: must be a mapping of case_name -> "
                "list of {group_name, vector} entries (multi-load-case schema), "
                "not a flat list. Example:\n"
                "  load_cases:\n"
                "    vertical_up:\n"
                "      - group_name: \"load_1\"\n"
                "        vector: [0.0, 0.0, 9.34e7]\n"
                f"Got: {type(raw_load_cases).__name__}"
            )
        load_cases = {
            case_name: [LoadCase(**lc) for lc in entries]
            for case_name, entries in raw_load_cases.items()
        }
        if not load_cases:
            raise ConfigError(
                "config.yaml's load_cases: mapping is empty -- at least one "
                "named load case is required"
            )
        for case_name, entries in load_cases.items():
            if not entries:
                raise ConfigError(
                    f"Load case '{case_name}' has no group_name/vector entries"
                )

    return ProjectConfig(
        optimization=optimization,
        random_field=random_field,
        surrogate=surrogate,
        mesh_source=mesh_source,
        box_mesh=box_mesh,
        mc_validation=mc_validation,
        validation=validation,
        step_file=raw.get("step_file"),
        mesh_out_path=raw.get("mesh_out_path"),
        mesh_size_max=raw.get("mesh_size_max"),
        snap_tol=raw.get("snap_tol"),
        material=material,
        petsc=petsc,
        load_cases=load_cases,
        color_targets=raw.get("color_targets", {}),
        solid_volume_color=tuple(raw.get("solid_volume_color", (255, 255, 0, 255))),
        keep_alive=KeepAliveConfig(**raw.get("keep_alive", {})),
    )


def _validate_cross_field(
    optimization: OptimizationConfig,
    random_field: RandomFieldConfig,
    mc_validation: MonteCarloValidationConfig,
) -> None:
    """Consistency checks between blocks that silently corrupted results when violated."""
    if optimization.saa_seed == mc_validation.seed:
        raise ConfigError(
            f"optimization.saa_seed and mc_validation.seed are both "
            f"{mc_validation.seed}. Stage 6 would then validate the design on "
            "the very samples it was optimized against, turning the "
            "out-of-sample check into an in-sample one and hiding exactly the "
            "SAA overfitting it exists to detect. Use disjoint seeds."
        )

    if mc_validation.beta != optimization.saa_beta_max:
        raise ConfigError(
            f"mc_validation.beta ({mc_validation.beta}) != "
            f"optimization.saa_beta_max ({optimization.saa_beta_max}). The "
            "validation must evaluate the design at the SAME projection "
            "sharpness it was optimized at; validating a beta=128 design at "
            "beta=8 (or the reverse) measures a different structure than the "
            "one that was designed."
        )

    if optimization.lambda_sweep_start not in ("common", "continuation"):
        raise ConfigError(
            "optimization.lambda_sweep_start must be 'common' or "
            f"'continuation', got {optimization.lambda_sweep_start!r}"
        )

    if optimization.robust_method not in ("saa", "pce"):
        raise ConfigError(
            f"optimization.robust_method must be 'saa' or 'pce', got "
            f"{optimization.robust_method!r}"
        )

    if len(optimization.lambda_sweep) < 2:
        raise ConfigError(
            f"optimization.lambda_sweep has {len(optimization.lambda_sweep)} "
            "entry/entries. A trade-off study needs at least 2, and a curve "
            "that can be read as a Pareto front needs about 5."
        )
    elif len(optimization.lambda_sweep) < 5:
        logger.warning(
            "optimization.lambda_sweep has only %d points. That is enough to "
            "compare designs but too few to present as a Pareto front.",
            len(optimization.lambda_sweep),
        )

    if random_field.eta_min >= random_field.eta_max:
        raise ConfigError(
            f"random_field.eta_min ({random_field.eta_min}) must be < eta_max "
            f"({random_field.eta_max})."
        )

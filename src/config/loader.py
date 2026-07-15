"""
src/config/loader.py

Loads config/config.yaml into a validated ProjectConfig. This is the ONLY
place in the codebase that should call open()/yaml.safe_load() for pipeline
parameters -- every downstream module receives a ProjectConfig object, never
a raw dict pulled from disk, so parameter provenance stays traceable.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from src.config.schema import (
LoadCase, MaterialConfig, MonteCarloValidationConfig, OptimizationConfig,
PetscConfig, ProjectConfig, RandomFieldConfig, SurrogateConfig,
)

REQUIRED_TOP_KEYS = {"step_file", "mesh_out_path", "mesh_size_max", "snap_tol",
"material", "load_cases", "random_field", "surrogate", "mc_validation"}


def load_config(path: str | Path) -> ProjectConfig:
    """Parse and validate config/config.yaml.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    KeyError
        If a required top-level key is missing -- fails loudly rather than
        silently defaulting, since silent defaults on physical parameters
        (E, nu, vol_frac) are exactly the kind of hidden shortcut this
        project's rules prohibit.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    missing = REQUIRED_TOP_KEYS - raw.keys()
    if missing:
        raise KeyError(f"config/config.yaml missing required keys: {missing}")

    material = MaterialConfig(**raw["material"])
    petsc = PetscConfig(**raw.get("petsc", {}))

    raw_load_cases = raw["load_cases"]
    if not isinstance(raw_load_cases, dict):
        raise TypeError(
            "config.yaml's load_cases: must be a mapping of case_name -> "
            "list of {group_name, vector} entries (multi-load-case schema), "
            "not a flat list. Example:\n"
            "  load_cases:\n"
            "    vertical_up:\n"
            "      - group_name: \"load_1\"\n"
            "        vector: [0.0, 0.0, 9.34e7]\n"
            "    torsion:\n"
            "      - group_name: \"load_1\"\n"
            "        vector: [0.0, -2.9e7, 0.0]\n"
            f"Got: {type(raw_load_cases).__name__}"
        )
    load_cases = {
        case_name: [LoadCase(**lc) for lc in entries]
        for case_name, entries in raw_load_cases.items()
    }
    if not load_cases:
        raise KeyError("config.yaml's load_cases: mapping is empty -- at least one named load case is required")
    for case_name, entries in load_cases.items():
        if not entries:
            raise KeyError(f"Load case '{case_name}' has no group_name/vector entries")

    optimization = OptimizationConfig(**raw.get("optimization", {}))
    random_field = RandomFieldConfig(**raw["random_field"])
    surrogate = SurrogateConfig(**raw.get("surrogate", {}))
    mc_validation = MonteCarloValidationConfig(**raw.get("mc_validation", {}))

    return ProjectConfig(
        step_file=raw["step_file"],
        mesh_out_path=raw["mesh_out_path"],
        mesh_size_max=raw["mesh_size_max"],
        snap_tol=raw["snap_tol"],
        material=material,
        petsc=petsc,
        load_cases=load_cases,
        optimization=optimization,
        random_field=random_field,
        surrogate=surrogate,
        color_targets=raw.get("color_targets", {}),
        mc_validation=mc_validation,
        solid_volume_color=tuple(raw.get("solid_volume_color", (255, 255, 0, 255))),
    )
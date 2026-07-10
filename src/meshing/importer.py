"""
src/meshing/importer.py

Stage 1 (CAD Import) — masterContext.md Section 3.2.
Responsible ONLY for opening a STEP/IGES/BREP file via Gmsh's OpenCASCADE
(OCC) kernel and healing the geometry. Meshing and physical-group tagging
live in mesher.py; BC membership functions live in mapper.py.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import gmsh

logger = logging.getLogger(__name__)


@dataclass
class GeometryEntities:
    """Raw OCC entity handles: (dim, tag) tuples per Gmsh convention."""
    points: list[tuple[int, int]]
    curves: list[tuple[int, int]]
    surfaces: list[tuple[int, int]]
    volumes: list[tuple[int, int]]


def open_geometry(step_path: str | Path, length_unit: str = "M") -> None:
    """Open a STEP/IGES/BREP file into the active Gmsh model.

    CAD files are frequently authored in millimetres; FEniTop/dolfinx
    expects SI units, so default target unit is metres ("M"). Override if
    your source file is already in metres.
    """
    step_path = Path(step_path)
    if not step_path.exists():
        raise FileNotFoundError(f"CAD file not found: {step_path}")

    if not gmsh.isInitialized():
        gmsh.initialize()

    gmsh.option.setString("Geometry.OCCTargetUnit", length_unit)
    gmsh.open(str(step_path))
    logger.info("Opened CAD file %s (unit=%s)", step_path, length_unit)


def heal_geometry(tolerance: float = 1e-6) -> None:
    """Remove duplicate/degenerate OCC entities before tagging/meshing.

    Must run before any physical-group tagging, since duplicate surfaces
    from CAD exports otherwise corrupt tag assignment downstream.
    """
    gmsh.model.occ.removeAllDuplicates()
    gmsh.model.occ.synchronize()
    logger.info("Geometry healed (tolerance=%.3g)", tolerance)


def extract_entities() -> GeometryEntities:
    """Extract raw OCC entity lists. Call after heal_geometry()."""
    points, curves, surfaces, volumes = (
        gmsh.model.occ.get_entities(d) for d in range(4)
    )
    logger.info(
        "Extracted entities: %d points, %d curves, %d surfaces, %d volumes",
        len(points), len(curves), len(surfaces), len(volumes),
    )
    return GeometryEntities(points=points, curves=curves,
                             surfaces=surfaces, volumes=volumes)


def import_and_heal(step_path: str | Path, length_unit: str = "M",
                     tolerance: float = 1e-6) -> GeometryEntities:
    """Convenience wrapper: open + heal + extract in one call."""
    open_geometry(step_path, length_unit=length_unit)
    heal_geometry(tolerance=tolerance)
    return extract_entities()


def finalize() -> None:
    """Finalize the Gmsh session. Call only after mesher.py has exported
    the mesh, since mesher.py reuses this same live Gmsh session."""
    if gmsh.isInitialized():
        gmsh.finalize()
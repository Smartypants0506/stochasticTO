"""
src/meshing/mesher.py

Stage 1 (Mesh Generation) — masterContext.md Section 3.2.
Consumes the healed geometry produced by importer.py, applies color-coded
physical-group tagging (ported from the prototype custom_geo_TO.py CLI
script), generates a tetrahedral mesh, and exports to XDMF/HDF5 via
dolfinx.io.gmshio for FEniTop consumption.

Convention (unchanged from the prototype):
  - GREEN  surfaces -> Dirichlet ("fixed") boundary
  - RED    surfaces -> traction load case 1
  - BLUE   surfaces -> traction load case 2 (optional)
  - YELLOW volumes  -> must-stay-solid regions (bolts/mounts), density
                       hard-fixed to 1 for the entire optimization
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import gmsh
from mpi4py import MPI

from src.meshing.importer import GeometryEntities
import numpy as np
logger = logging.getLogger(__name__)

DEFAULT_COLOR_TARGETS: dict[str, tuple[int, int, int, int]] = {
    "fixed": (0, 255, 0, 255),
    "load_1": (255, 0, 0, 255),
    "load_2": (0, 0, 255, 255),
}
DEFAULT_SOLID_VOLUME_COLOR: tuple[int, int, int, int] = (255, 255, 0, 255)
DEFAULT_COLOR_MATCH_TOL = 40


@dataclass
class MeshingConfig:
    """All meshing/tagging parameters -- no hardcoded values inside mesher.py.
    Populate this from configs/config.yaml at the call site (src/config/loader.py),
    per the project's config-driven rule."""
    mesh_size_max: float = 1e-3
    mesh_algorithm_3d: int = 10  # HXT
    mesh_size_from_curvature: bool = True
    min_elements_per_2pi: int = 8
    optimize_netgen: bool = True
    color_targets: dict[str, tuple[int, int, int, int]] = field(
        default_factory=lambda: dict(DEFAULT_COLOR_TARGETS))
    solid_volume_color: tuple[int, int, int, int] = DEFAULT_SOLID_VOLUME_COLOR
    color_match_tol: int = DEFAULT_COLOR_MATCH_TOL


@dataclass
class TaggedMesh:
    """Output of mesher.py: everything mapper.py and the FEniTop adapter need."""
    mesh: object            # dolfinx.mesh.Mesh (parallel, comm.size ranks)
    mesh_serial: object     # dolfinx.mesh.Mesh on COMM_SELF (rank 0 only)
    cell_tags: object       # dolfinx.mesh.MeshTags (volume physical groups)
    facet_tags: object      # dolfinx.mesh.MeshTags (surface physical groups)
    name_to_tag: dict[str, int]


def _color_close(c1, c2, tol: int) -> bool:
    return all(abs(a - b) <= tol for a, b in zip(c1[:3], c2[:3]))


def _extract_tags_by_color(entities, target_color, tol: int) -> list[int]:
    return [e[1] for e in entities if _color_close(gmsh.model.get_color(*e), target_color, tol)]


def tag_physical_groups(entities: GeometryEntities, config: MeshingConfig) -> None:
    """Assign Gmsh physical groups from color-coded CAD faces/volumes.

    Surfaces matching a color in config.color_targets become named 2D
    physical groups ("fixed", "load_1", "load_2", ...). Volumes matching
    config.solid_volume_color become the "solid" group (hard-fixed density);
    all remaining volumes become the "volume" (optimizable) group.

    Raises
    ------
    RuntimeError
        If no optimizable volume remains after removing solid-colored volumes
        -- this mirrors the hard failure in the original prototype script.
    """
    for name, target_color in config.color_targets.items():
        tags = _extract_tags_by_color(entities.surfaces, target_color, config.color_match_tol)
        if not tags:
            logger.warning(
                "No faces found for color group '%s' (target RGBA=%s). "
                "Check CAD coloring.", name, target_color,
            )
            continue
        gmsh.model.addPhysicalGroup(2, tags, name=name)

    solid_volume_tags = _extract_tags_by_color(
        entities.volumes, config.solid_volume_color, config.color_match_tol)
    all_volume_tags = [v[1] for v in entities.volumes]
    optimizable_volume_tags = [v for v in all_volume_tags if v not in solid_volume_tags]

    if not optimizable_volume_tags:
        raise RuntimeError(
            "No optimizable volume found. Check that at least one solid body "
            "is NOT colored as the solid_volume_color."
        )

    gmsh.model.addPhysicalGroup(3, optimizable_volume_tags, name="volume")
    if solid_volume_tags:
        gmsh.model.addPhysicalGroup(3, solid_volume_tags, name="solid")
    else:
        logger.warning(
            "No solid-colored volumes found -- no bolt/mounting region will "
            "be protected from optimization."
        )


def generate_mesh(config: MeshingConfig, comm: MPI.Comm) -> None:
    """Generate the 3D tetrahedral mesh with the configured size field.

    Must be called after tag_physical_groups() so physical group tags are
    baked into the mesh before dolfinx import.
    """
    gmsh.option.setNumber("General.NumThreads", comm.size)
    gmsh.option.setNumber("Mesh.Algorithm3D", config.mesh_algorithm_3d)
    gmsh.option.setNumber("Mesh.MeshSizeMax", config.mesh_size_max)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", int(config.mesh_size_from_curvature))
    gmsh.option.setNumber("Mesh.MinimumElementsPerTwoPi", config.min_elements_per_2pi)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", int(config.optimize_netgen))
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(3)
    logger.info("Mesh generated: max_size=%.3g", config.mesh_size_max)


def import_to_dolfinx(comm: MPI.Comm) -> TaggedMesh:
    """Convert the Gmsh model (rank 0) into dolfinx Mesh/MeshTags objects.

    Produces both a parallel-distributed mesh (for FEniTop's solver) and a
    serial COMM_SELF copy on rank 0 (for plotting, matching topopt.py's own
    `mesh_serial` convention used throughout robust_TO.py).
    """
    import dolfinx

    name_to_tag: dict[str, int] = {}
    if comm.rank == 0:
        mesh_data = dolfinx.io.gmsh.model_to_mesh(gmsh.model, comm, rank=0)
        mesh, cell_tags, facet_tags = mesh_data.mesh, mesh_data.cell_tags, mesh_data.facet_tags

        mesh_serial_data = dolfinx.io.gmsh.model_to_mesh(gmsh.model, MPI.COMM_SELF, rank=0)
        mesh_serial = mesh_serial_data.mesh

        for dim, tag in gmsh.model.getPhysicalGroups():
            name = gmsh.model.getPhysicalName(dim, tag)
            name_to_tag[name] = tag
    else:
        mesh_data = dolfinx.io.gmsh.model_to_mesh(None, comm, rank=0)
        mesh, cell_tags, facet_tags = mesh_data.mesh, mesh_data.cell_tags, mesh_data.facet_tags
        mesh_serial = None

    name_to_tag = comm.bcast(name_to_tag, root=0)
    for name in name_to_tag:
        logger.info("%s: physical group tag = %d", name, name_to_tag[name])

    return TaggedMesh(mesh=mesh, mesh_serial=mesh_serial,
                       cell_tags=cell_tags, facet_tags=facet_tags,
                       name_to_tag=name_to_tag)


def export_mesh(tagged_mesh: TaggedMesh, out_path: str | Path) -> None:
    """Export the mesh + tags to XDMF/HDF5 via meshio for reuse without
    re-running Gmsh (matches the disk2d.xdmf / shell3d.xdmf convention
    already present in the FEniTop package's meshes/ directory)."""
    from dolfinx.io import XDMFFile

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with XDMFFile(tagged_mesh.mesh.comm, str(out_path), "w") as xdmf:
        xdmf.write_mesh(tagged_mesh.mesh)
        xdmf.write_meshtags(tagged_mesh.cell_tags, tagged_mesh.mesh.geometry)
        xdmf.write_meshtags(tagged_mesh.facet_tags, tagged_mesh.mesh.geometry)
    logger.info("Mesh exported to %s", out_path)

def extract_simplices(tagged_mesh: TaggedMesh) -> "np.ndarray":
    """Extract element connectivity as [N_elems x (dim+1)] node-index array.

    Required by src/random_fields/kl_expansion.py's compute_kl_expansion,
    which needs simplices to build an ot.Mesh for the FEM-based KL
    expansion. dolfinx stores this in mesh.geometry.dofmap; this function
    converts it to the plain NumPy array shape OpenTURNS expects.

    Args:
        tagged_mesh: Output of mesh_from_geometry/import_to_dolfinx.

    Returns:
        [N_elems x (dim+1)] array of node indices per element (triangles
        for 2D, tetrahedra for 3D), using tagged_mesh.mesh_serial so the
        result is available identically on every MPI rank (mesh_serial
        is a COMM_SELF, rank-0-only mesh, matching this module's existing
        serial-execution convention for the random-field glue).
    """
    import numpy as np
    mesh_serial = tagged_mesh.mesh_serial
    return mesh_serial.geometry.dofmaps[0].reshape(
        mesh_serial.topology.index_map(mesh_serial.topology.dim).size_local, -1
    )

def mesh_from_geometry(entities: GeometryEntities, config: MeshingConfig,
                        comm: MPI.Comm) -> TaggedMesh:
    """Convenience wrapper: tag + generate + import in one call."""
    tag_physical_groups(entities, config)
    generate_mesh(config, comm)
    return import_to_dolfinx(comm)
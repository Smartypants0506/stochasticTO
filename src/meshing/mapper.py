"""
src/meshing/mapper.py

Stage 1 (BC / Zone Mapping) — masterContext.md Section 3.2.
Converts tagged mesh entities (from mesher.py) into the membership
functions FEniTop's fem.py expects for `disp_bc`, `traction_bcs`,
`solid_zone`, and `void_zone`.

This module has no Gmsh dependency -- it operates purely on the dolfinx
Mesh/MeshTags objects produced by mesher.import_to_dolfinx(), keeping the
Gmsh-specific logic isolated to mesher.py per the module-separation rule.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
from mpi4py import MPI
from scipy.spatial import cKDTree

from src.meshing.mesher import TaggedMesh

logger = logging.getLogger(__name__)

MembershipFn = Callable[[np.ndarray], np.ndarray]


def facet_group_points(tagged_mesh: TaggedMesh, group_name: str,
                        comm: MPI.Comm | None = None) -> np.ndarray:
    """Return the [N x 3] coordinates of vertices on facets tagged with the
    given physical group name (e.g. "fixed", "load_1").

    Reads from `tagged_mesh.mesh_serial`/`facet_tags_serial` (the full,
    unpartitioned COMM_SELF mesh, populated only on rank 0) rather than the
    parallel-distributed `tagged_mesh.mesh`. Small, spatially concentrated
    surfaces (e.g. a single load face) can otherwise end up with zero
    facets in whichever rank's local partition is doing the lookup, even
    though the group is correctly tagged in the global mesh. The result is
    broadcast to every rank so all ranks build identical membership
    functions from the same complete point set.
    """
    name_to_tag = tagged_mesh.name_to_tag
    if group_name not in name_to_tag:
        points = np.empty((0, 3))
        return comm.bcast(points, root=0) if comm is not None else points

    tag = name_to_tag[group_name]

    if comm is None or comm.rank == 0:
        mesh_serial = tagged_mesh.mesh_serial
        facet_tags_serial = tagged_mesh.facet_tags_serial
        fdim = mesh_serial.topology.dim - 1
        mesh_serial.topology.create_connectivity(fdim, 0)
        f_to_v = mesh_serial.topology.connectivity(fdim, 0)

        facet_indices = facet_tags_serial.find(tag)
        if len(facet_indices) == 0:
            points = np.empty((0, 3))
        else:
            vertex_ids = np.unique(np.hstack([f_to_v.links(f) for f in facet_indices]))
            points = mesh_serial.geometry.x[vertex_ids]
    else:
        points = None

    if comm is not None:
        points = comm.bcast(points, root=0)
    return points


def make_membership_fn(points: np.ndarray, tol: float) -> MembershipFn:
    """Build a `lambda x: bool_mask` membership function from a KDTree
    over reference points, matching FEniTop's expected `disp_bc`/
    `traction_bcs` callable signature (x is [3 x N])."""
    if len(points) == 0:
        return lambda x: np.full(x.shape[1], False)
    tree = cKDTree(points)

    def fn(x: np.ndarray) -> np.ndarray:
        dist, _ = tree.query(x.T, k=1)
        return dist < tol
    return fn

def make_protected_zone_from_faces(
    tagged_mesh: TaggedMesh,
    group_names: list[str],
    buffer_radius: float,
    comm: MPI.Comm | None = None,
) -> MembershipFn:
    """Build a protected_zone(x) membership function that hard-fixes density
    to 1 for any point within `buffer_radius` of any facet tagged with one of
    `group_names` (e.g. bolt faces, pin faces).

    Unlike make_solid_zone_from_cells(), this does NOT require the protected
    region to be its own separate colored volume in the CAD file -- it works
    directly off facet points, so it protects material *around* a face that
    sits on the boundary of the single optimizable volume (e.g. bolt holes,
    pin holes), not just pre-separated bolt/mount volumes.

    Reads from the full serial mesh on rank 0 and broadcasts, for the same
    partition-safety reason as facet_group_points().
    """
    all_points = []
    for name in group_names:
        pts = facet_group_points(tagged_mesh, name, comm=comm)
        if len(pts) > 0:
            all_points.append(pts)

    if not all_points:
        logger.warning(
            "No points found for any protected-zone group in %s; "
            "protected zone will be empty.", group_names,
        )
        return lambda x: np.full(x.shape[1], False)

    combined_points = np.vstack(all_points)
    tree = cKDTree(combined_points)

    def fn(x: np.ndarray) -> np.ndarray:
        dist, _ = tree.query(x.T, k=1)
        return dist < buffer_radius
    return fn

def make_solid_zone_from_cells(tagged_mesh: TaggedMesh, group_name: str,
                                comm: MPI.Comm | None = None) -> MembershipFn:
    """Build a solid_zone(x) membership function from actual tagged cell
    centroids (volume-based), not coordinate proximity -- mirrors the
    fenitop GE-bracket demo's "bolts" volume treatment exactly.

    Like facet_group_points(), reads from the serial mesh/cell_tags on
    rank 0 and broadcasts, since small solid regions (e.g. bolt volumes)
    are equally susceptible to being absent from a given rank's partition.
    """
    tag = tagged_mesh.name_to_tag.get(group_name)
    if tag is None:
        return lambda x: np.full(x.shape[1], False)

    if comm is None or comm.rank == 0:
        mesh_serial = tagged_mesh.mesh_serial
        cell_tags_serial = tagged_mesh.cell_tags_serial
        cell_indices = cell_tags_serial.find(tag)

        if len(cell_indices) == 0:
            centroids = np.empty((0, 3))
        else:
            tdim = mesh_serial.topology.dim
            mesh_serial.topology.create_connectivity(tdim, 0)
            c_to_v = mesh_serial.topology.connectivity(tdim, 0)
            centroids = np.array([
                mesh_serial.geometry.x[c_to_v.links(c)].mean(axis=0) for c in cell_indices
            ])
    else:
        centroids = None

    if comm is not None:
        centroids = comm.bcast(centroids, root=0)

    if len(centroids) == 0:
        return lambda x: np.full(x.shape[1], False)

    tree = cKDTree(centroids)

    def fn(x: np.ndarray) -> np.ndarray:
        dist, _ = tree.query(x.T, k=1)
        return dist < 1e-9  # centroids match exactly (same DG0 dof coords)
    return fn


@dataclass
class BoundaryConditions:
    """Bundled BC/zone functions ready to drop into FEniTop's fem_config
    and opt_config dicts (see fem-11.py's `form_fem` signature)."""
    disp_bc: MembershipFn
    traction_bcs: list[list]  # [[load_vector, membership_fn], ...]
    solid_zone: MembershipFn
    void_zone: MembershipFn


def build_boundary_conditions(
    tagged_mesh: TaggedMesh,
    load_vectors: dict[str, tuple[float, float, float]],
    snap_tol: float = 1e-4,
    protected_face_groups: list[str] | None = None,
    protected_buffer_radius: float = 5e-3,
    comm: MPI.Comm | None = None,
) -> BoundaryConditions:
    fixed_pts = facet_group_points(tagged_mesh, "fixed", comm=comm)
    disp_bc = make_membership_fn(fixed_pts, snap_tol)

    traction_bcs = []
    for name, vector in load_vectors.items():
        pts = facet_group_points(tagged_mesh, name, comm=comm)
        if len(pts) == 0:
            logger.warning("No points found for load group '%s'; skipping.", name)
            continue
        traction_bcs.append([vector, make_membership_fn(pts, snap_tol)])

    solid_zone_volumes = make_solid_zone_from_cells(tagged_mesh, "solid", comm=comm)

    if protected_face_groups:
        solid_zone_faces = make_protected_zone_from_faces(
            tagged_mesh, protected_face_groups, protected_buffer_radius, comm=comm)
        solid_zone = lambda x: solid_zone_volumes(x) | solid_zone_faces(x)
    else:
        solid_zone = solid_zone_volumes

    void_zone = lambda x: np.full(x.shape[1], False)

    return BoundaryConditions(
        disp_bc=disp_bc, traction_bcs=traction_bcs,
        solid_zone=solid_zone, void_zone=void_zone,
    )
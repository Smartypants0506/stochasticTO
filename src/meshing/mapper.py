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

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra, minimum_spanning_tree
import itertools

from src.meshing.mesher import TaggedMesh

logger = logging.getLogger(__name__)

MembershipFn = Callable[[np.ndarray], np.ndarray]

def cluster_points_by_mesh_connectivity(
    tagged_mesh: TaggedMesh, group_name: str, comm: MPI.Comm | None = None,
) -> list[np.ndarray]:
    """Split one tagged facet group into per-instance point clusters using
    mesh connectivity (shared edges/vertices), not spatial distance.

    This is what lets a single tag like "fixed" -- which may cover all 4
    mounting holes as one physical group -- resolve into 4 separate anchor
    points instead of one merged centroid. Two facets belong to the same
    instance iff they share a vertex; holes on the same tag but physically
    separated on the mesh never share a vertex, so they fall out as
    distinct connected components automatically.
    """
    if comm is None or comm.rank == 0:
        tag = tagged_mesh.name_to_tag.get(group_name)
        clusters: list[np.ndarray] = []
        if tag is not None:
            mesh_serial = tagged_mesh.mesh_serial
            facet_tags_serial = tagged_mesh.facet_tags_serial
            fdim = mesh_serial.topology.dim - 1
            mesh_serial.topology.create_connectivity(fdim, 0)
            f_to_v = mesh_serial.topology.connectivity(fdim, 0)

            facet_indices = facet_tags_serial.find(tag)
            if len(facet_indices) > 0:
                # Build a vertex-adjacency graph restricted to this group's facets
                vertex_ids = sorted(set(
                    v for f in facet_indices for v in f_to_v.links(f)
                ))
                local_index = {v: i for i, v in enumerate(vertex_ids)}
                rows, cols = [], []
                for f in facet_indices:
                    vs = f_to_v.links(f)
                    for a, b in itertools.combinations(vs, 2):
                        rows.append(local_index[a]); cols.append(local_index[b])
                        rows.append(local_index[b]); cols.append(local_index[a])
                n = len(vertex_ids)
                adj = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
                from scipy.sparse.csgraph import connected_components
                n_components, labels = connected_components(adj, directed=False)

                coords = mesh_serial.geometry.x[vertex_ids]
                for c in range(n_components):
                    clusters.append(coords[labels == c])
    else:
        clusters = None

    if comm is not None:
        clusters = comm.bcast(clusters, root=0)
    return clusters


def _build_mesh_vertex_graph(mesh_serial) -> csr_matrix:
    """Weighted undirected graph over mesh vertices from tet connectivity.
    Edge weight = Euclidean distance -- used for true in-mesh shortest-path
    distance (not straight-line), since the bracket geometry is
    bent/non-convex and a straight chord can cut through void space."""
    tdim = mesh_serial.topology.dim
    mesh_serial.topology.create_connectivity(tdim, 0)
    c_to_v = mesh_serial.topology.connectivity(tdim, 0)
    x = mesh_serial.geometry.x
    rows, cols, weights = [], [], []
    n_cells = mesh_serial.topology.index_map(tdim).size_local
    for c in range(n_cells):
        vs = c_to_v.links(c)
        for a, b in itertools.combinations(vs, 2):
            d = np.linalg.norm(x[a] - x[b])
            rows += [a, b]; cols += [b, a]; weights += [d, d]
    n = x.shape[0]
    return csr_matrix((weights, (rows, cols)), shape=(n, n))


def _nearest_vertex_index(mesh_serial, point: np.ndarray) -> int:
    tree = cKDTree(mesh_serial.geometry.x)
    _, idx = tree.query(point)
    return int(idx)


def compute_keep_alive_corridor_paths(
    tagged_mesh: TaggedMesh,
    mounting_groups: list[str],
    load_groups: list[str],
    comm: MPI.Comm | None = None,
) -> list[np.ndarray]:
    """Cluster all mounting/load facet groups into per-instance anchors,
    compute true in-mesh shortest-path distances between every pair, then
    take the minimum spanning tree over those anchors. Returns one polyline
    (array of mesh vertex coordinates) per retained MST edge -- the minimum
    guaranteed-connected backbone linking every bolt boss and load face to
    the main structure, leaving the optimizer free everywhere else.
    """
    if comm is None or comm.rank == 0:
        mesh_serial = tagged_mesh.mesh_serial
        graph = _build_mesh_vertex_graph(mesh_serial)

        anchor_indices: list[int] = []
        for group_name in mounting_groups + load_groups:
            clusters = cluster_points_by_mesh_connectivity(tagged_mesh, group_name, comm=None)
            for cluster_pts in clusters:
                centroid = cluster_pts.mean(axis=0)
                anchor_indices.append(_nearest_vertex_index(mesh_serial, centroid))

        k = len(anchor_indices)
        if k < 2:
            paths = []
        else:
            dist_matrix = np.full((k, k), np.inf)
            predecessors_list = []
            for i, src in enumerate(anchor_indices):
                dist, pred = dijkstra(
                    graph, directed=False, indices=src, return_predecessors=True,
                )
                predecessors_list.append(pred)
                for j, tgt in enumerate(anchor_indices):
                    dist_matrix[i, j] = dist[tgt]

            mst = minimum_spanning_tree(dist_matrix).toarray()
            paths = []
            for i in range(k):
                for j in range(k):
                    if mst[i, j] > 0:
                        pred = predecessors_list[i]
                        path_idx = [anchor_indices[j]]
                        cur = anchor_indices[j]
                        while cur != anchor_indices[i] and cur != -9999:
                            cur = pred[cur]
                            path_idx.append(cur)
                        paths.append(mesh_serial.geometry.x[path_idx])
    else:
        paths = None

    if comm is not None:
        paths = comm.bcast(paths, root=0)
    return paths


def make_line_corridor_zone(path_points: np.ndarray, radius: float) -> MembershipFn:
    """Cylindrical (capsule) membership function around a polyline defined
    by consecutive points in path_points. Uses true segment distance per
    edge of the polyline rather than a single straight-line chord, so a
    multi-segment in-mesh path around bends/voids is honored exactly."""
    segments = list(zip(path_points[:-1], path_points[1:]))

    def fn(x: np.ndarray) -> np.ndarray:
        pts = x.T  # [N x 3]
        best = np.full(pts.shape[0], np.inf)
        for a, b in segments:
            ab = b - a
            ab_len2 = np.dot(ab, ab)
            if ab_len2 < 1e-30:
                d = np.linalg.norm(pts - a, axis=1)
            else:
                t = np.clip(((pts - a) @ ab) / ab_len2, 0.0, 1.0)
                proj = a + t[:, None] * ab
                d = np.linalg.norm(pts - proj, axis=1)
            best = np.minimum(best, d)
        return best < radius
    return fn


def make_keep_alive_corridors_zone(
    tagged_mesh: TaggedMesh,
    mounting_groups: list[str],
    load_groups: list[str],
    corridor_radius: float,
    comm: MPI.Comm | None = None,
) -> MembershipFn:
    """Top-level entry point: MST-backbone keep-alive corridors, generic
    over any set of mounting (red) and load (blue/green) facet groups."""
    paths = compute_keep_alive_corridor_paths(
        tagged_mesh, mounting_groups, load_groups, comm=comm)

    if not paths:
        return lambda x: np.full(x.shape[1], False)

    corridor_fns = [make_line_corridor_zone(p, corridor_radius) for p in paths]

    def fn(x: np.ndarray) -> np.ndarray:
        mask = np.full(x.shape[1], False)
        for cfn in corridor_fns:
            mask |= cfn(x)
        return mask
    return fn


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
    and opt_config dicts (see fem-11.py's `form_fem` signature).

    traction_bcs is now dict[case_name, list[[load_vector, membership_fn]]]
    -- one independent traction_bcs list per named load case -- rather than
    a single flat list. This is what gets handed directly to
    src.fenitop.topopt.form_fem_multi_case as its `load_cases` argument;
    each entry becomes its own independently-solved equilibrium problem
    sharing one density field, instead of every load being summed into one
    combined RHS.
    """
    disp_bc: MembershipFn
    traction_bcs: dict[str, list[list]]  # {case_name: [[load_vector, membership_fn], ...]}
    solid_zone: MembershipFn
    void_zone: MembershipFn


def build_boundary_conditions(
    tagged_mesh: TaggedMesh,
    load_cases: dict[str, list[tuple[str, tuple[float, float, float]]]],
    snap_tol: float = 1e-4,
    protected_face_groups: list[str] | None = None,
    protected_buffer_radius: float = 15e-3,
    keep_alive_corridors: dict | None = None,
    comm: MPI.Comm | None = None,
) -> BoundaryConditions:
    """Build shared disp_bc/solid_zone/void_zone plus a per-case traction_bcs dict.

    load_cases: {case_name: [(group_name, vector), ...], ...} -- e.g.
    {"vertical_up": [("load_1", (0,0,9.34e7))],
     "torsion": [("load_1", (0,-2.9e7,0))]}.
    Multiple cases commonly reuse the same group_name with different
    vectors (facet lookup is cached per group_name below so that pattern
    doesn't repeat the KDTree build), but each case's list can also
    reference entirely different groups.
    """
    fixed_pts = facet_group_points(tagged_mesh, "fixed", comm=comm)
    disp_bc = make_membership_fn(fixed_pts, snap_tol)

    # Cache membership fns per facet group so cases that reuse the same
    # tagged group (the common "same group, different vector" pattern)
    # don't redo the facet lookup + KDTree build for every case.
    membership_cache: dict[str, MembershipFn | None] = {}

    def _get_membership_fn(group_name: str) -> MembershipFn | None:
        if group_name not in membership_cache:
            pts = facet_group_points(tagged_mesh, group_name, comm=comm)
            if len(pts) == 0:
                logger.warning("No points found for load group '%s'; skipping.", group_name)
                membership_cache[group_name] = None
            else:
                membership_cache[group_name] = make_membership_fn(pts, snap_tol)
        return membership_cache[group_name]

    traction_bcs: dict[str, list[list]] = {}
    for case_name, entries in load_cases.items():
        case_bcs = []
        for group_name, vector in entries:
            fn = _get_membership_fn(group_name)
            if fn is None:
                continue
            case_bcs.append([vector, fn])
        if not case_bcs:
            logger.warning(
                "Load case '%s' resolved to zero valid facet groups; it "
                "will contribute no traction (check group names/tags).",
                case_name,
            )
        traction_bcs[case_name] = case_bcs

    solid_zone_volumes = make_solid_zone_from_cells(tagged_mesh, "solid", comm=comm)

    zones = [solid_zone_volumes]
    if protected_face_groups:
        zones.append(make_protected_zone_from_faces(
            tagged_mesh, protected_face_groups, protected_buffer_radius, comm=comm))
    if keep_alive_corridors:
        zones.append(make_keep_alive_corridors_zone(
            tagged_mesh,
            mounting_groups=keep_alive_corridors["mounting_groups"],
            load_groups=keep_alive_corridors["load_groups"],
            corridor_radius=keep_alive_corridors["corridor_radius"],
            comm=comm,
        ))

    solid_zone = lambda x: np.any([z(x) for z in zones], axis=0)

    void_zone = lambda x: np.full(x.shape[1], False)

    return BoundaryConditions(
        disp_bc=disp_bc, traction_bcs=traction_bcs,
        solid_zone=solid_zone, void_zone=void_zone,
    )
"""
mainTest.py

Full diagnostic reproduction of Stage 2 (nominal SIMP topopt) using the
EXACT SAME meshing/mapping/adapter pipeline as main.py -- nothing is
swapped out for a synthetic domain. This isolates whether a failure
originates in the CAD import/mesher/mapper stack (colors, snap_tol,
solid_zone, disp_bc/traction_bcs construction, protected-face buffers)
vs. the FEniTop optimizer itself.

Diagnostics included (all run BEFORE topopt() is called):
  1. disp_bc / traction_bcs point counts (catches empty/mistagged groups)
  2. Measured load_1 face area + implied traction pressure (catches
     units/magnitude mismatches against material stiffness)
  3. Fixed-to-load face distance (catches unrealistically short load paths)
  4. solid_zone element fraction + distance-to-main-body (catches
     disconnected yellow CAD solid volumes)
  5. protected-face-zone-only distance-to-main-body (catches disconnected
     buffer shells around bolt/pin holes, isolated from #4)

Run:
    mpirun -n 8 python3 mainTest.py
"""
from __future__ import annotations
import logging

import numpy as np
import ufl
import dolfinx
from mpi4py import MPI
from scipy.spatial import cKDTree

comm = MPI.COMM_WORLD

from src.config.loader import load_config
from src.fenitop.topopt import topopt
from src.meshing.importer import import_and_heal, finalize
from src.meshing.mesher import (
    extract_simplices, MeshingConfig, tag_physical_groups,
    generate_mesh, import_to_dolfinx,
)
from src.meshing.mapper import (
    build_boundary_conditions, facet_group_points, make_solid_zone_from_cells,
)
from src.fea.fenitop_adapter import build_fenitop_dicts

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


def diagnose_boundary_conditions(tagged_mesh, bc, load_vectors, comm) -> None:
    """Dump everything about the BC/zone construction BEFORE topopt runs,
    so a mesher/mapper bug is caught here rather than inferred from a
    downstream NaN or a visually broken optimized geometry.
    """
    mesh = tagged_mesh.mesh
    centers = mesh.geometry.x

    # -- 1. disp_bc coverage --
    fixed_mask = bc.disp_bc(centers.T)
    n_fixed = comm.allreduce(int(np.sum(fixed_mask)), op=MPI.SUM)
    if comm.rank == 0:
        logger.info("disp_bc: %d matched points (global)", n_fixed)
        if n_fixed == 0:
            logger.error("disp_bc matched ZERO points -- model is completely "
                         "unconstrained. This alone can cause singular "
                         "stiffness matrices and NaN sensitivities.")

    # -- 1b. traction_bcs coverage + magnitude sanity check --
    for i, (vector, membership_fn) in enumerate(bc.traction_bcs):
        mask = membership_fn(centers.T)
        n_pts = comm.allreduce(int(np.sum(mask)), op=MPI.SUM)
        vec_mag = float(np.linalg.norm(vector))
        if comm.rank == 0:
            logger.info("traction_bcs[%d]: vector=%s |vector|=%.6g, "
                        "%d matched points (global)", i, vector, vec_mag, n_pts)
            if n_pts == 0:
                logger.error("traction_bcs[%d] matched ZERO points -- load "
                            "is not actually being applied anywhere.", i)

    # -- 2. measured face area for load_1, sanity-check load magnitude/units --
    if "load_1" in tagged_mesh.name_to_tag:
        load1_tag = tagged_mesh.name_to_tag["load_1"]
        ds = ufl.Measure("ds", domain=mesh, subdomain_data=tagged_mesh.facet_tags)
        area_form = dolfinx.fem.form(1.0 * ds(load1_tag))
        local_area = dolfinx.fem.assemble_scalar(area_form)
        global_area = comm.allreduce(local_area, op=MPI.SUM)
        if comm.rank == 0:
            logger.info("Measured load_1 face area = %.6g m^2", global_area)
            for name, vector in load_vectors.items():
                if name == "load_1":
                    implied_pressure = np.linalg.norm(vector) / max(global_area, 1e-12)
                    logger.info("load_1: implied traction pressure = %.6g Pa "
                                "(force / measured area)", implied_pressure)

    # -- 3. fixed-to-load face distance (catches unrealistically short load paths) --
    fixed_pts = facet_group_points(tagged_mesh, "fixed", comm=comm)
    load_pts = facet_group_points(tagged_mesh, "load_1", comm=comm)
    if comm.rank == 0 and len(fixed_pts) > 0 and len(load_pts) > 0:
        dist_tree = cKDTree(fixed_pts)
        face_dists, _ = dist_tree.query(load_pts, k=1)
        logger.info("fixed-to-load face distance: min=%.6g m, max=%.6g m, mean=%.6g m",
                    face_dists.min(), face_dists.max(), face_dists.mean())

    # -- 4/5. solid_zone / void_zone element counts + connectivity checks --
    tdim = mesh.topology.dim
    n_cells_local = mesh.topology.index_map(tdim).size_local
    mesh.topology.create_connectivity(tdim, 0)
    c_to_v = mesh.topology.connectivity(tdim, 0)
    cell_centroids = np.array([
        mesh.geometry.x[c_to_v.links(c)].mean(axis=0) for c in range(n_cells_local)
    ]).T  # kept for the GLOBAL element-count logging above (unchanged)

    # NEW: build centroids from the full serial mesh (rank 0 only) for the
    # connectivity checks below, since those need the COMPLETE geometry,
    # not just rank 0's shard of the distributed mesh.
    if comm.rank == 0:
        mesh_serial = tagged_mesh.mesh_serial
        tdim_s = mesh_serial.topology.dim
        n_cells_serial = mesh_serial.topology.index_map(tdim_s).size_local
        mesh_serial.topology.create_connectivity(tdim_s, 0)
        c_to_v_s = mesh_serial.topology.connectivity(tdim_s, 0)
        cell_centroids_serial = np.array([
            mesh_serial.geometry.x[c_to_v_s.links(c)].mean(axis=0)
            for c in range(n_cells_serial)
        ]).T
        solid_mask = bc.solid_zone(cell_centroids_serial)

    solid_mask = bc.solid_zone(cell_centroids)
    void_mask = bc.void_zone(cell_centroids)
    n_solid = comm.allreduce(int(np.sum(solid_mask)), op=MPI.SUM)
    n_void = comm.allreduce(int(np.sum(void_mask)), op=MPI.SUM)
    n_total = comm.allreduce(n_cells_local, op=MPI.SUM)
    if comm.rank == 0:
        logger.info("solid_zone: %d/%d elements (%.2f%%)",
                    n_solid, n_total, 100 * n_solid / max(n_total, 1))
        logger.info("void_zone: %d/%d elements (%.2f%%)",
                    n_void, n_total, 100 * n_void / max(n_total, 1))

    # -- 4. solid_zone (combined volumes+protected faces) distance to main body --
    if comm.rank == 0:
        solid_centroids_local = cell_centroids[:, solid_mask]
        main_centroids_local = cell_centroids[:, ~solid_mask]
        if solid_centroids_local.shape[1] > 0 and main_centroids_local.shape[1] > 0:
            main_tree = cKDTree(main_centroids_local.T)
            dists, _ = main_tree.query(solid_centroids_local.T, k=1)
            logger.info("solid_zone to main-body distance: min=%.6g, max=%.6g, mean=%.6g",
                        dists.min(), dists.max(), dists.mean())
        else:
            logger.warning("solid_zone distance check skipped: empty solid or main set "
                           "(solid=%d, main=%d)",
                           solid_centroids_local.shape[1], main_centroids_local.shape[1])

    # -- 5. protected FACE zone specifically, isolated from volume-based solid_zone --
    solid_zone_volumes_fn = make_solid_zone_from_cells(tagged_mesh, "solid", comm=comm)
    if comm.rank == 0:
        solid_zone_volumes_mask = solid_zone_volumes_fn(cell_centroids)
        protected_mask = solid_mask & ~solid_zone_volumes_mask
        protected_centroids_local = cell_centroids[:, protected_mask]
        main_mask = ~solid_mask
        main_centroids_local2 = cell_centroids[:, main_mask]
        if protected_centroids_local.shape[1] > 0 and main_centroids_local2.shape[1] > 0:
            main_tree2 = cKDTree(main_centroids_local2.T)
            dists2, _ = main_tree2.query(protected_centroids_local.T, k=1)
            logger.info("protected-face-zone to main-body gap: max=%.6g m", dists2.max())
        else:
            logger.info("protected-face-zone check skipped: no protected-face-only cells "
                        "found (protected=%d)", protected_centroids_local.shape[1])


def main(config_path: str = "src/config/config.yaml") -> None:
    cfg = load_config(config_path)

    if comm.rank == 0:
        logger.info("material.youngs_modulus = %.6g", cfg.material.youngs_modulus)
        logger.info("material.poissons_ratio = %.6g", cfg.material.poissons_ratio)
        entities = import_and_heal(cfg.step_file)
        mesh_cfg = MeshingConfig(mesh_size_max=cfg.mesh_size_max,
                                  color_targets=cfg.color_targets,
                                  solid_volume_color=cfg.solid_volume_color)
        tag_physical_groups(entities, mesh_cfg)
        generate_mesh(mesh_cfg, comm)

    tagged_mesh = import_to_dolfinx(comm)  # collective
    assert "fixed" in tagged_mesh.name_to_tag, "No 'fixed' faces tagged — check STEP coloring"
    assert "load_1" in tagged_mesh.name_to_tag, "No 'load_1' faces tagged — check STEP coloring"

    load_vectors = {lc.group_name: lc.vector for lc in cfg.load_cases}
    bc = build_boundary_conditions(
        tagged_mesh, load_vectors,
        snap_tol=cfg.snap_tol,
        protected_face_groups=["fixed", "load_1"],
        protected_buffer_radius=4e-3,
        comm=comm,
    )

    # -- Full diagnostic dump BEFORE handing off to topopt --
    diagnose_boundary_conditions(tagged_mesh, bc, load_vectors, comm)

    fem, opt_nominal = build_fenitop_dicts(tagged_mesh, bc, cfg)

    if comm.rank == 0:
        logger.info("Running Stage 2 (nominal SIMP topopt) ONLY -- "
                    "no random-field/Pareto/MC stages.")
        topopt(fem, opt_nominal)

    finalize()


if __name__ == "__main__":
    main()

# Run in parallel:
# mpirun -n 8 python3 mainTest.py
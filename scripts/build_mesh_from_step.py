"""
scripts/build_mesh_from_step.py

CLI entry point for Stage 1: CAD -> tagged mesh -> BC bundle -> XDMF export.
Replaces the standalone custom_geo_TO.py prototype; all logic now lives in
src/meshing/{importer,mesher,mapper}.py.

Usage:
    python3 scripts/build_mesh_from_step.py --step my_part.step \\
        --out output/meshes/my_part.xdmf
"""
from __future__ import annotations

import argparse
import logging

from mpi4py import MPI

from src.meshing.importer import import_and_heal, finalize
from src.meshing.mesher import MeshingConfig, mesh_from_geometry, export_mesh
from src.meshing.mapper import build_boundary_conditions

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", required=True, help="Path to color-coded STEP file")
    parser.add_argument("--out", required=True, help="Output XDMF path")
    parser.add_argument("--mesh-size-max", type=float, default=1e-3)
    parser.add_argument("--tol", type=float, default=1e-4,
                         help="Snap tolerance (m) for matching mesh points to colored facets")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD

    # Load vectors must eventually come from configs/config.yaml
    # (src/config/loader.py), not hardcoded here -- placeholder until that
    # config system is built (see Repo Restructuring step in the roadmap).
    load_vectors = {
        "load_1": (1_000_000.0, 0.0, 1_000_000.0),
        "load_2": (1_000_000.0, 0.0, 1_000_000.0),
    }

    if comm.rank == 0:
        entities = import_and_heal(args.step)
    else:
        entities = None
    entities = comm.bcast(entities, root=0) if comm.rank != 0 else entities

    config = MeshingConfig(mesh_size_max=args.mesh_size_max)
    tagged_mesh = mesh_from_geometry(entities, config, comm)
    export_mesh(tagged_mesh, args.out)

    bc = build_boundary_conditions(tagged_mesh, load_vectors, snap_tol=args.tol)
    logger.info(
        "BC bundle built: %d traction groups, disp_bc pts derived from 'fixed' tag",
        len(bc.traction_bcs),
    )

    if comm.rank == 0:
        finalize()


if __name__ == "__main__":
    main()
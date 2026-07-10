"""
scripts/run_nominal_simp.py

Stage 2: Nominal SIMP topology optimization.
Extends build_mesh_from_step.py's pipeline (import -> mesh -> BC) through
FEniTop's own form_fem() and topopt() -- no custom assembly, solve, or
MMA logic here. This is orchestration only; all physics/optimization
math is FEniTop's benchmarked implementation (fem.py, topopt.py, mma.py).
"""
from __future__ import annotations

import argparse
import logging

from mpi4py import MPI

from src.config.loader import load_config
from src.meshing.importer import import_and_heal, finalize
from src.meshing.mesher import mesh_from_geometry, export_mesh, MeshingConfig
from src.meshing.mapper import build_boundary_conditions
from src.fea.fenitop_adapter import build_fenitop_dicts
from src.fenitop.fem import form_fem
from src.fenitop.topopt import topopt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to configs/config.yaml")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    config = load_config(args.config)

    load_vectors = {lc.group_name: tuple(lc.vector) for lc in config.load_cases}

    if comm.rank == 0:
        entities = import_and_heal(config.step_file)
    else:
        entities = None
    entities = comm.bcast(entities, root=0) if comm.rank != 0 else entities

    mesh_cfg = MeshingConfig(mesh_size_max=config.mesh_size_max)
    tagged_mesh = mesh_from_geometry(entities, mesh_cfg, comm)
    export_mesh(tagged_mesh, config.mesh_out_path)

    bc = build_boundary_conditions(tagged_mesh, load_vectors, snap_tol=config.snap_tol)
    fem_dict, opt_dict = build_fenitop_dicts(tagged_mesh, bc, config)

    logger.info("Running nominal SIMP via FEniTop's topopt()...")
    topopt(fem_dict, opt_dict)

    if comm.rank == 0:
        finalize()


if __name__ == "__main__":
    main()
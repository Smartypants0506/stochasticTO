"""viz/enrich_ensemble_fea.py -- add real FEA fields to the MC ensemble.

THE PROBLEM THIS SOLVES
-----------------------
`output/mc_validation/ensemble/sample_XXXXX.vtu` contains exactly ONE array:

    PointData "density"   = rho_phys, the projected density of that realization

That is all src/validation/monte_carlo.py ever gathers (see the
`ensemble_grid.point_data["density"] = global_rho_phys` line). There is no
displacement, no stress, no strain energy -- the module docstring in
src/viz/probability_cloud.py advertises a `.von_mises` field, but nothing in
the pipeline has ever computed one. So you cannot colour the ensemble by an
FEA metric today; the data does not exist.

This script re-solves the SAME 100 realizations and writes the missing fields.
It is not a re-run of the optimizer: the design rho is FIXED (loaded from
disk) and only eta(x) varies, exactly as in Stage 6. Because
monte_carlo.py draws sample i as `default_rng(seed + i).standard_normal(n_kl)`,
the realizations reproduced here are bit-identical to the ones already on
disk -- the compliance printed per sample should match
stage6_validation/compliance_samples.csv to solver tolerance, and the script
checks that for you.

FIELDS WRITTEN (all CG1 nodal, so they interpolate smoothly in ParaView)
-----------------------------------------------------------------------
  density                rho_phys -- same as before, kept for masking
  eta                    the sampled manufacturing threshold field itself.
                         This is the *cause*; everything else is the effect.
                         eta > 0.5 locally = under-deposition (thinner part).
  displacement           3-vector, so ParaView's Warp By Vector works
  displacement_magnitude |u|
  von_mises              macroscopic von Mises stress of the SIMP-interpolated
                         material. NOTE: in void (rho ~ 0) this is scaled down
                         by the SIMP factor and is physically meaningless --
                         always Threshold on density > 0.5 before reading it.
  von_mises_solid        von_mises / (eps + (1-eps)*rho^p), i.e. the stress in
                         the underlying SOLID phase. This is the number that
                         matters for yielding, and it is the one that blows up
                         in the thin ligaments a non-robust design relies on.
  strain_energy_density  0.5 * sigma : eps -- where compliance is being spent.
                         Integrates to the compliance of that sample.

SMOOTHED SURFACE (`surfaces/sample_XXXXX.vtp`)
------------------------------------------------
The raw .vtu above is the FULL bounding-box tet mesh -- rho ~ 0 void nodes and
all -- so opened directly in ParaView it renders as the whole solid block
(or, after a manual Threshold, a jagged tet-faceted blob) rather than the
smooth part shape. That is the same problem viz/build_cloud_index.py solves
for the density-only ensemble: contour at rho = iso to extract the actual
manufactured boundary. We do the identical thing here, with one bonus --
VTK's contour filter interpolates EVERY point-data array present on the
input, not just the one being contoured, so the rho = iso surface comes out
already carrying eta / von_mises / von_mises_solid / strain_energy_density /
displacement / displacement_magnitude as point data. No re-solve, no extra
pass. `--decimate` (0..0.95) thins the triangles exactly as in
build_cloud_index.py.

USAGE (in the dolfinx container, from the repo root)
---------------------------------------------------
    mpirun -n 8 python viz/enrich_ensemble_fea.py

    # baseline design instead of the robust one:
    mpirun -n 8 python viz/enrich_ensemble_fea.py \
        --design output/stage2_fea/rho_converged.npy \
        --out-dir output/viz/ensemble_nominal_fea

    # quick look at the 10 worst-compliance samples only:
    mpirun -n 8 python viz/enrich_ensemble_fea.py --samples 3,17,42,88

    # lighter surfaces for a big ensemble
    mpirun -n 8 python viz/enrich_ensemble_fea.py --decimate 0.7

Cost: one linear elastic solve per sample, same as Stage 6 already paid
(~100 solves on the 25x75x25 tet beam). The gathers add a few seconds total;
the contour + decimate step (rank 0 only, pyvista) adds a similar amount.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Allow `python viz/enrich_ensemble_fea.py` from the repo root without setting
# PYTHONPATH, matching how src/mainClean.py is normally launched.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import ufl
import dolfinx
from dolfinx.fem import Function, functionspace, form, assemble_scalar, Expression
from mpi4py import MPI

from src.config.loader import load_config
from src.fenitop.fem import form_fem
from src.fenitop.parameterize import DensityFilter
from src.fenitop.utility import Communicator
from src.random_fields.kernel import KernelParams
from src.random_fields.kl_expansion import compute_kl_expansion
from src.topology.heaviside_projection_glue import (
    RandomHeavisideConfig, build_random_heaviside_from_function_space,
)

logger = logging.getLogger(__name__)
comm = MPI.COMM_WORLD


def _interpolation_points(V):
    """dolfinx 0.9 exposes this as a method; 0.10 turned it into a property."""
    pts = V.element.interpolation_points
    return pts() if callable(pts) else pts


def build_stage1(cfg, comm):
    """Rebuild the Stage-1 mesh + Stage-2 fem/opt dicts for the configured
    mesh_source. This mirrors src/mainClean.py's Stage 1 exactly; it is
    repeated rather than imported because mainClean.main() is a single
    top-to-bottom function with no reusable entry point.
    """
    if cfg.mesh_source == "box":
        from src.meshing.box_source import build_box_fenitop_dicts
        return build_box_fenitop_dicts(cfg, comm)

    if cfg.mesh_source != "step":
        raise ValueError(f"Unknown mesh_source={cfg.mesh_source!r}")

    from src.meshing.importer import import_and_heal
    from src.meshing.mesher import (MeshingConfig, tag_physical_groups,
                                    generate_mesh, import_to_dolfinx)
    from src.meshing.mapper import build_boundary_conditions
    from src.fea.fenitop_adapter import build_fenitop_dicts

    if comm.rank == 0:
        entities = import_and_heal(cfg.step_file)
        mesh_cfg = MeshingConfig(mesh_size_max=cfg.mesh_size_max,
                                 color_targets=cfg.color_targets,
                                 solid_volume_color=cfg.solid_volume_color)
        tag_physical_groups(entities, mesh_cfg)
        generate_mesh(mesh_cfg, comm)
    tagged_mesh = import_to_dolfinx(comm)

    load_cases_input = {
        name: [(lc.group_name, lc.vector) for lc in entries]
        for name, entries in cfg.load_cases.items()
    }
    ka = cfg.keep_alive
    bc = build_boundary_conditions(
        tagged_mesh, load_cases_input,
        snap_tol=cfg.snap_tol,
        protected_face_groups=["fixed", "load_1", "load_2"],
        protected_buffer_radius=4e-3,
        keep_alive_groups=ka.groups if ka.enabled else None,
        keep_alive_radius=(
            (ka.corridor_radius if ka.corridor_radius is not None
             else 2.0 * cfg.mesh_size_max) if ka.enabled else None),
        keep_alive_cluster_eps=(
            (ka.cluster_eps if ka.cluster_eps is not None
             else 2.0 * cfg.mesh_size_max) if ka.enabled else None),
        comm=comm,
    )
    fem, opt, load_cases = build_fenitop_dicts(tagged_mesh, bc, cfg)
    return tagged_mesh, fem, opt, load_cases


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="src/config/config.yaml")
    ap.add_argument("--design", type=Path,
                    default=Path("output/stage5_optimization/rho_robust_lambda_1.0.npy"),
                    help="global DG0 design array to hold fixed across samples")
    ap.add_argument("--out-dir", type=Path, default=Path("output/viz/ensemble_fea"))
    ap.add_argument("--n-samples", type=int, default=None,
                    help="defaults to cfg.mc_validation.n_samples")
    ap.add_argument("--samples", type=str, default=None,
                    help="comma-separated subset of sample indices, e.g. 0,7,42")
    ap.add_argument("--seed", type=int, default=None,
                    help="defaults to cfg.mc_validation.seed -- keep it there to "
                         "reproduce the ensemble already on disk")
    ap.add_argument("--beta", type=float, default=None,
                    help="defaults to cfg.mc_validation.beta")
    ap.add_argument("--iso", type=float, default=0.5,
                    help="density level set treated as the manufactured "
                         "boundary for the smoothed surface (default 0.5, "
                         "matches build_cloud_index.py)")
    ap.add_argument("--decimate", type=float, default=0.0,
                    help="fraction of triangles to remove per surface, "
                         "0..0.95 (default 0.0 = no decimation)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO if comm.rank == 0 else logging.ERROR,
                        force=True)
    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg.mc_validation.seed
    beta = args.beta if args.beta is not None else cfg.mc_validation.beta

    tagged_mesh, fem, opt, load_cases = build_stage1(cfg, comm)
    case_name = next(iter(load_cases))
    fem["traction_bcs"] = load_cases[case_name]

    # --- Stage 3: same KL basis the optimization and Stage 6 used ----------
    kernel_params = KernelParams(sigma=cfg.random_field.sigma,
                                 length_scale=cfg.random_field.length_scale,
                                 spatial_dim=cfg.random_field.spatial_dim)
    if comm.rank == 0:
        from src.meshing.mesher import extract_simplices
        node_coordinates = tagged_mesh.mesh_serial.geometry.x
        simplices = extract_simplices(tagged_mesh)
    else:
        node_coordinates = simplices = None
    kl_result = compute_kl_expansion(
        node_coordinates, simplices, kernel_params,
        variance_threshold=cfg.random_field.variance_threshold, comm=comm)

    # --- FEA problem on the fixed design ----------------------------------
    linear_problem, u_field, lambda_field, rho_field, rho_phys_field = form_fem(fem, opt)
    density_filter = DensityFilter(comm, rho_field, rho_phys_field,
                                   opt["filter_radius"], fem["petsc_options"])

    rho_global = np.load(args.design) if comm.rank == 0 else None
    rho_global = comm.bcast(rho_global, root=0)
    design_comm = Communicator(rho_field.function_space, fem["mesh_serial"])
    design_comm.bcast(rho_field, rho_global)
    rho_field.x.scatter_forward()
    density_filter.forward()
    rho_tilde_cached = rho_phys_field.x.petsc_vec.array.copy()

    hv_cfg = RandomHeavisideConfig(
        kernel_params=opt["kernel_params"],
        transform_params=opt["transform_params"],
        variance_threshold=opt.get("kl_variance_threshold",
                                   cfg.random_field.variance_threshold),
        seed=seed,
    )
    rf = build_random_heaviside_from_function_space(rho_phys_field, kl_result, hv_cfg)

    # --- Derived-field machinery ------------------------------------------
    # Every exported field is interpolated into S == rho_phys's own CG1 space,
    # so a SINGLE Communicator (whose cKDTree coordinate match is the expensive
    # part) serves all of them, and every gathered array lands in the same
    # nodal ordering as dolfinx.plot.vtk_mesh's points.
    mesh = fem["mesh"]
    S = rho_phys_field.function_space
    dim = mesh.geometry.dim

    E0, nu = fem["young's modulus"], fem["poisson's ratio"]
    p, eps_simp = opt["penalty"], opt["epsilon"]
    simp = eps_simp + (1 - eps_simp) * rho_phys_field ** p
    E = simp * E0
    lam_ = E * nu / (1 + nu) / (1 - 2 * nu)
    mu_ = E / (2 * (1 + nu))

    eps_u = ufl.sym(ufl.grad(u_field))
    sig = 2 * mu_ * eps_u + lam_ * ufl.tr(eps_u) * ufl.Identity(dim)
    dev = sig - (1.0 / 3.0) * ufl.tr(sig) * ufl.Identity(dim)
    vm_ufl = ufl.sqrt(1.5 * ufl.inner(dev, dev))
    sed_ufl = 0.5 * ufl.inner(sig, eps_u)

    exprs = {
        "von_mises": Expression(vm_ufl, _interpolation_points(S)),
        "von_mises_solid": Expression(vm_ufl / simp, _interpolation_points(S)),
        "strain_energy_density": Expression(sed_ufl, _interpolation_points(S)),
    }
    comp_exprs = [Expression(u_field[k], _interpolation_points(S)) for k in range(dim)]
    scratch = Function(S)
    eta_fn = Function(S)

    S_comm = Communicator(S, fem["mesh_serial"])
    compliance_form = form(opt["compliance"])

    if comm.rank == 0:
        elements, cell_types, nodes = dolfinx.plot.vtk_mesh(
            fem["mesh_serial"], fem["mesh_serial"].topology.dim)
        import pyvista as pv
        grid = pv.UnstructuredGrid(elements, cell_types, nodes)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        ens_dir = args.out_dir / "ensemble"
        ens_dir.mkdir(parents=True, exist_ok=True)
        surf_dir = args.out_dir / "surfaces"
        surf_dir.mkdir(parents=True, exist_ok=True)

    if args.samples:
        indices = [int(s) for s in args.samples.split(",")]
    else:
        n = args.n_samples if args.n_samples is not None else cfg.mc_validation.n_samples
        indices = list(range(n))

    linear_problem.enable_warm_start(True)
    linear_problem.set_reuse_preconditioner(False)

    compliances: list[float] = []
    pvd_entries: list[tuple[int, str]] = []
    surf_pvd_entries: list[tuple[int, str]] = []
    for i in indices:
        rho_phys_field.x.petsc_vec.array[:] = rho_tilde_cached
        xi = np.random.default_rng(seed + i).standard_normal(kl_result.n_kl)
        eta_local = rf.set_eta_from_xi(xi)
        rf.forward(beta)
        linear_problem.solve_fem()
        C = comm.allreduce(assemble_scalar(compliance_form), op=MPI.SUM)
        compliances.append(float(C))

        fields: dict[str, np.ndarray] = {}
        fields["density"] = S_comm.gather(rho_phys_field)
        eta_fn.x.petsc_vec.array[:] = eta_local
        fields["eta"] = S_comm.gather(eta_fn)
        for name, expr in exprs.items():
            scratch.interpolate(expr)
            fields[name] = S_comm.gather(scratch)
        u_cols = []
        for k in range(dim):
            scratch.interpolate(comp_exprs[k])
            u_cols.append(S_comm.gather(scratch))

        if comm.rank == 0:
            for name, values in fields.items():
                grid.point_data[name] = values
            u_arr = np.column_stack(u_cols)
            if dim == 2:  # pad so ParaView still sees a 3-component vector
                u_arr = np.column_stack([u_arr, np.zeros(u_arr.shape[0])])
            grid.point_data["displacement"] = u_arr
            grid.point_data["displacement_magnitude"] = np.linalg.norm(u_arr, axis=1)
            path = ens_dir / f"sample_{i:05d}.vtu"
            grid.save(str(path))
            pvd_entries.append((i, os.path.relpath(path, start=args.out_dir)))

            # The smoothed rho = iso boundary -- see the "SMOOTHED SURFACE"
            # module docstring section. contour() interpolates every point
            # array already on `grid` (eta, von_mises, ...), so the surface
            # comes out fully enriched with no extra solve.
            surf = grid.contour(isosurfaces=[args.iso], scalars="density")
            if surf.n_points == 0:
                logger.info("sample %d: empty iso-surface at rho=%g, "
                            "surface skipped", i, args.iso)
            else:
                if args.decimate > 0.0:
                    surf = surf.decimate_pro(args.decimate,
                                             preserve_topology=True)
                surf_path = surf_dir / f"sample_{i:05d}.vtp"
                surf.save(str(surf_path))
                surf_pvd_entries.append(
                    (i, os.path.relpath(surf_path, start=args.out_dir)))
            logger.info("sample %d/%d: C=%.6g", i, len(indices), C)

    if comm.rank == 0:
        def _write_pvd(pvd_path: Path, entries: list[tuple[int, str]]) -> None:
            lines = ['<?xml version="1.0"?>',
                     '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
                     "  <Collection>"]
            for idx, rel in entries:
                lines.append(f'    <DataSet timestep="{idx}" group="" part="0" file="{rel}"/>')
            lines += ["  </Collection>", "</VTKFile>"]
            pvd_path.write_text("\n".join(lines) + "\n")

        _write_pvd(args.out_dir / "ensemble.pvd", pvd_entries)
        _write_pvd(args.out_dir / "ensemble_surfaces.pvd", surf_pvd_entries)
        logger.info("Wrote %d smoothed surfaces to %s (iso=%g, decimate=%g)",
                    len(surf_pvd_entries), surf_dir, args.iso, args.decimate)

        C_arr = np.asarray(compliances)
        np.savetxt(args.out_dir / "compliance_samples.csv",
                   np.column_stack([np.asarray(indices), C_arr]),
                   delimiter=",", header="sample_index,compliance", comments="",
                   fmt=["%d", "%.10e"])

        # Reproduction check against the ensemble already on disk. A mismatch
        # means the design/seed/beta here differ from the Stage-6 run, and the
        # enriched fields do NOT belong to the same realizations.
        ref = Path("output/stage6_validation/compliance_samples.csv")
        note = "reference compliance_samples.csv not found -- skipped"
        if ref.exists():
            ref_C = np.loadtxt(ref, delimiter=",", skiprows=1)[:, 1]
            if max(indices) < ref_C.size:
                rel_err = np.abs(C_arr - ref_C[indices]) / np.abs(ref_C[indices])
                note = (f"max |rel err| vs Stage 6 = {rel_err.max():.3e} "
                        f"(expect <1e-6 when --design is the Stage-6 design)")
        logger.info("Reproduction check: %s", note)
        (args.out_dir / "provenance.json").write_text(json.dumps({
            "design": str(args.design), "seed": seed, "beta": beta,
            "n_kl": int(kl_result.n_kl), "case": case_name,
            "samples": indices, "reproduction_check": note,
        }, indent=2))
        logger.info("Wrote enriched ensemble to %s", args.out_dir)


if __name__ == "__main__":
    main()

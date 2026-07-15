"""
Authors:
- Yingqi Jia (yingqij2@illinois.edu)
- Chao Wang (chaow4@illinois.edu)
- Xiaojia Shelly Zhang (zhangxs@illinois.edu)

Sponsors:
- U.S. National Science Foundation (NSF) EAGER Award CMMI-2127134
- U.S. Defense Advanced Research Projects Agency (DARPA) Young Faculty Award
  (N660012314013)
- NSF CAREER Award CMMI-2047692
- NSF Award CMMI-2245251

Reference:
- Jia, Y., Wang, C. & Zhang, X.S. FEniTop: a simple FEniCSx implementation
  for 2D and 3D topology optimization supporting parallel computing.
  Struct Multidisc Optim 67, 140 (2024).
  https://doi.org/10.1007/s00158-024-03818-7
"""

import time

import numpy as np
from mpi4py import MPI

from src.fenitop.fem import form_fem
from src.fenitop.parameterize import DensityFilter, Heaviside
from src.fenitop.sensitivity import Sensitivity
from src.fenitop.optimize import optimality_criteria, mma_optimizer
from src.fenitop.utility import Communicator, Plotter, save_xdmf
from dataclasses import dataclass

from dolfinx.fem import locate_dofs_topological

import logging
logger = logging.getLogger(__name__)

@dataclass
class LoadCaseProblem:
    name: str
    linear_problem: object
    u_field: object
    lambda_field: object
    sens_problem: "Sensitivity"


def form_fem_multi_case(fem, opt, load_cases: dict[str, list]):
    """Build one shared design (rho_field/rho_phys_field) and one
    independent elasticity problem + Sensitivity object per load case.

    load_cases: {case_name: [[vector, membership_fn], ...]} -- i.e. the
    per-case traction_bcs list, e.g. {"vertical_up": [[[0,0,9.34e7], fn1],
    [[0,0,9.34e7], fn2]], "torsion": [[[0,-2.9e7,0], fn1], [[0,2.9e7,0], fn2]], ...}
    """
    rho_field, rho_phys_field = None, None
    problems = []
    for name, traction_bcs in load_cases.items():
        fem_case = dict(fem)
        fem_case["traction_bcs"] = traction_bcs
        # form_fem() writes opt["compliance"]/opt["f_int"]/opt["volume"]/
        # opt["total_volume"] into the SAME shared opt dict every call,
        # keyed to this iteration's own u_field. That looks like it should
        # be a bug (next iteration overwrites those keys before you'd
        # "expect" them to be read) -- it isn't, because Sensitivity(...)
        # is constructed immediately below, in this same iteration, and its
        # __init__ calls dolfinx.fem.form(...) on those UFL expressions
        # right away. form() compiles against the specific Function objects
        # referenced at that moment (this case's u_field/rho_phys_field),
        # so the compiled Form is unaffected by opt's dict entries being
        # overwritten on the next iteration. Do not "fix" this by giving
        # each case its own opt dict -- Sensitivity/DensityFilter/Heaviside
        # elsewhere rely on this opt dict staying the single shared object.
        linear_problem, u_field, lambda_field, rho_field, rho_phys_field = form_fem(
            fem_case, opt, rho_field=rho_field, rho_phys_field=rho_phys_field)
        sens_problem = Sensitivity(MPI.COMM_WORLD, opt, linear_problem,
                                    u_field, lambda_field, rho_phys_field)
        problems.append(LoadCaseProblem(name, linear_problem, u_field,
                                         lambda_field, sens_problem))
    return problems, rho_field, rho_phys_field


def topopt(fem, opt, load_cases: dict[str, list]):
    """Main function for topology optimization.

    fem: shared fem TEMPLATE dict (no "traction_bcs" key -- see
        fenitop_adapter.build_fem_dict).
    opt: opt dict, as before.
    load_cases: dict[case_name, traction_bcs_list] -- one independently
        solved equilibrium problem per case, sharing one density field.
        Their compliances and sensitivities are summed each iteration
        (see the Solve FEM section below), NOT combined into a single
        static-equilibrium RHS -- summing compliances after independent
        solves is physically correct where summing loads before solving
        is not, since compliance is quadratic in the load
        (compliance(f1+f2) != compliance(f1)+compliance(f2) in general).
    """

    # Initialization
    def _ck(msg):
        print(f"[rank {comm.rank}] CK: {msg}", flush=True)

    comm = MPI.COMM_WORLD
    
    #_ck("entered topopt")
    problems, rho_field, rho_phys_field = form_fem_multi_case(fem, opt, load_cases)
    #_ck("after form_fem_multi_case")

    num_consts = 1 if opt["opt_compliance"] else 2
    num_elems = rho_field.x.petsc_vec.array.size

    density_filter = DensityFilter(comm, rho_field, rho_phys_field,
                                opt["filter_radius"], fem["petsc_options"])
    #_ck("after DensityFilter")
    heaviside = Heaviside(rho_phys_field)
    #_ck("after Heaviside")
    S_comm = Communicator(rho_phys_field.function_space, fem["mesh_serial"])
    #_ck("after Communicator")
    if comm.rank == 0:
        plotter = Plotter(fem["mesh_serial"])
    #_ck("after Plotter (rank0 built plotter)")

    # ... your solid-mask block ...
    #_ck("after solid mask block")

    centers = rho_field.function_space.tabulate_dof_coordinates()[:num_elems].T
    #_ck("after tabulate centers")
    solid, void = opt["solid_zone"](centers), opt["void_zone"](centers)
    #_ck("after solid_zone / void_zone")
    if not opt["use_oc"]:
        rho_old1, rho_old2 = np.zeros(num_elems), np.zeros(num_elems)
        low, upp = None, None

    # --- Solid mask for rho_phys_field (CG1 / nodal) --------------------------
    # rho_phys_field is Lagrange-1 (nodal), NOT DG0 -- see fem.py. Its dofs are
    # VERTICES, so we pin the vertex dofs of every solid-tagged cell.
    #
    # MPI CORRECTNESS: every collective below (create_connectivity,
    # locate_dofs_topological, allreduce) is called UNCONDITIONALLY on all ranks.
    # Small solid regions mean some ranks legitimately own zero solid cells; we
    # must NOT guard the collectives behind `len(solid_cells) > 0` or those ranks
    # skip the collective and the others deadlock.
    solid_cell_mask = np.zeros(rho_phys_field.x.petsc_vec.array.size, dtype=bool)
    cell_tags = fem.get("cell_tags")
    solid_tag = fem.get("solid_tag")

    if cell_tags is not None and solid_tag is not None:
        Vp = rho_phys_field.function_space
        mesh_p = Vp.mesh
        tdim = mesh_p.topology.dim
        n_local_cells = mesh_p.topology.index_map(tdim).size_local

        # Collective on some backends -- call on ALL ranks, unconditionally.
        mesh_p.topology.create_connectivity(tdim, 0)
        c_to_v = mesh_p.topology.connectivity(tdim, 0)

        solid_cells = cell_tags.find(solid_tag)
        solid_cells = solid_cells[solid_cells < n_local_cells].astype(np.int32)

        # Build the local vertex list. Empty on ranks that own no solid cells --
        # that's fine, but we still fall through to the (collective) dof lookup.
        if len(solid_cells) > 0:
            solid_vertices = np.unique(
                np.concatenate([c_to_v.links(c) for c in solid_cells])
            ).astype(np.int32)
        else:
            solid_vertices = np.empty(0, dtype=np.int32)

        # locate_dofs_topological is collective in dolfinx -- ALL ranks must call
        # it, even with an empty entity list. Passing an empty array is valid and
        # returns an empty dof array on that rank.
        solid_dofs = locate_dofs_topological(Vp, 0, solid_vertices)
        solid_dofs = solid_dofs[solid_dofs < solid_cell_mask.size]
        solid_cell_mask[solid_dofs] = True

        # Collective -- ALL ranks, unconditionally.
        n_solid = comm.allreduce(int(solid_cell_mask.sum()), op=MPI.SUM)
        if comm.rank == 0:
            logger.info(
                "Hard-pinning %d nodal dofs (global) to rho_phys=1.0 every "
                "iteration (solid/bolt regions, CG1 vertex dofs).", n_solid,
            )
    else:
        if comm.rank == 0:
            logger.warning(
                "No 'solid' cell_tags/solid_tag found in fem dict -- bolt/mount "
                "regions will NOT be hard-pinned to solid density. This mirrors "
                "the exact bug that caused bolt bosses to be optimized away; "
                "check fenitop_adapter.py wires cell_tags + solid_tag through."
            )

    def _pin_solid(field) -> None:
        """Hard-set solid-tagged cells to density 1, exactly like the
        reference script's `ρ_f.x.array[V_ρ_f_bolt_dofs] = 1.0` re-pin
        after every filter application."""
        if solid_cell_mask.any():
            field.x.petsc_vec.array[solid_cell_mask] = 1.0

    # Apply passive zones
    centers = rho_field.function_space.tabulate_dof_coordinates()[:num_elems].T
    #_ck("after tabulate centers")
    solid, void = opt["solid_zone"](centers), opt["void_zone"](centers)
    #_ck("after solid_zone / void_zone")

    rho_ini = np.full(num_elems, opt["vol_frac"])
    rho_ini[solid], rho_ini[void] = 0.995, 0.005
    rho_field.x.petsc_vec.array[:] = rho_ini
    rho_min, rho_max = np.zeros(num_elems), np.ones(num_elems)
    rho_min[solid], rho_max[void] = 0.99, 0.01

    if cell_tags is not None and solid_tag is not None:
        solid_cells = cell_tags.find(solid_tag)
        if comm.rank == 0 or True:  # per-rank
            print(f"[rank {comm.rank}] num_elems={num_elems}, "
                f"mask.size={solid_cell_mask.size}, "
                f"rho_field.array.size={rho_field.x.petsc_vec.array.size}, "
                f"rho_phys.array.size={rho_phys_field.x.petsc_vec.array.size}, "
                f"cell_tags n_indices={solid_cells.size}, "
                f"solid_cells.max={solid_cells.max() if solid_cells.size else -1}, "
                f"mesh local cells="
                f"{fem['mesh'].topology.index_map(fem['mesh'].topology.dim).size_local}, "
                f"+ghosts="
                f"{fem['mesh'].topology.index_map(fem['mesh'].topology.dim).num_ghosts}",
                flush=True)

    # Start topology optimization
    opt_iter, beta, change = 0, 1, 2*opt["opt_tol"]
    while opt_iter < opt["max_iter"] and change > opt["opt_tol"]:
        opt_start_time = time.perf_counter()
        opt_iter += 1
        #_ck(f"iter {opt_iter}: top")
        density_filter.forward()
        #_ck(f"iter {opt_iter}: after filter.forward")
        _pin_solid(rho_phys_field)
        #_ck(f"iter {opt_iter}: after pin (forward)")
        if opt_iter % opt["beta_interval"] == 0 and beta < opt["beta_max"]:
            beta *= 2
            change = opt["opt_tol"] * 2
        heaviside.forward(beta)
        _pin_solid(rho_phys_field)
        #_ck(f"iter {opt_iter}: after heaviside+pin")

        # Solve FEM: each load case is its own independent equilibrium
        # problem sharing rho_phys_field. Compliances and sensitivities
        # are summed AFTER independently solving each case -- this is the
        # actual multi-load-case fix; combining all tractions into one RHS
        # before solving (the old single-case behavior) understates
        # torsional/off-axis load cases whenever they'd partially cancel
        # in a single combined equilibrium state.
        C_total = 0.0
        U_total = 0.0
        dCdrho_total = None
        dUdrho_total = None
        V_value, dVdrho = None, None  # volume is case-independent; take from case 0
        max_disp_over_cases = 0.0
        for i, lcp in enumerate(problems):
            #_ck(f"iter {opt_iter}: solving case {i} ({lcp.name})")
            lcp.linear_problem.solve_fem()
            #_ck(f"iter {opt_iter}: solved case {i} ({lcp.name})")
            (C_i, V_i, U_i), (dCdrho_i, dVdrho_i, dUdrho_i) = lcp.sens_problem.evaluate()
            C_total += C_i
            U_total += U_i
            dCdrho_total = dCdrho_i.copy() if dCdrho_total is None else (dCdrho_total + dCdrho_i)
            if dUdrho_i is not None:
                dUdrho_total = dUdrho_i.copy() if dUdrho_total is None else (dUdrho_total + dUdrho_i)
            if i == 0:
                V_value, dVdrho = V_i, dVdrho_i

            # Displacement diagnostic, per case (u_field differs per case
            # under multi-load-case solving -- there is no single shared
            # u_field to report anymore).
            u_array = lcp.u_field.x.petsc_vec.array  # local dofs, interleaved [ux,uy,uz,...]
            u_reshaped = u_array.reshape(-1, 3)  # 3D problem
            disp_mag = np.linalg.norm(u_reshaped, axis=1)
            local_max_disp = disp_mag.max() if disp_mag.size > 0 else 0.0
            max_disp_over_cases = max(max_disp_over_cases, local_max_disp)

        C_value, U_value = C_total, U_total
        global_max_disp = comm.allreduce(max_disp_over_cases, op=MPI.MAX)
        if comm.rank == 0 and opt_iter % 10 == 0:  # print every 10 iters to avoid spam
            print(f"  [diag] opt_iter {opt_iter}: max |u| over all load cases = "
                f"{global_max_disp:.6e} m ({global_max_disp*1000:.6f} mm)", flush=True)

        # Filter/project the SUMMED sensitivities exactly once (not once
        # per case): heaviside.backward and density_filter.backward are
        # linear/elementwise operators driven only by the shared
        # rho_phys_field, not by load-case-specific state, so summing
        # first and filtering once is both correct and avoids the
        # (expensive) filter adjoint solve once per case per iteration.
        sensitivities = [dCdrho_total, dVdrho, dUdrho_total]
        heaviside.backward(sensitivities)
        [dCdrho, dVdrho, dUdrho] = density_filter.backward(sensitivities)
        if opt["opt_compliance"]:
            g_vec = np.array([V_value-opt["vol_frac"]])
            dJdrho, dgdrho = dCdrho, np.vstack([dVdrho])
        else:
            g_vec = np.array([V_value-opt["vol_frac"], C_value-opt["compliance_bound"]])
            dJdrho, dgdrho = dUdrho, np.vstack([dVdrho, dCdrho])

        # Update the design variables
        rho_values = rho_field.x.petsc_vec.array.copy()
        if opt["opt_compliance"] and opt["use_oc"]:
            rho_new, change = optimality_criteria(
                rho_values, rho_min, rho_max, g_vec, dJdrho, dgdrho[0], opt["move"])
        else:
            rho_new, change, low, upp = mma_optimizer(
                num_consts, num_elems, opt_iter, rho_values, rho_min, rho_max,
                rho_old1, rho_old2, dJdrho, g_vec, dgdrho, low, upp, opt["move"])
            rho_old2 = rho_old1.copy()
            rho_old1 = rho_values.copy()
        rho_field.x.petsc_vec.array = rho_new.copy()

        # Output the histories
        opt_time = time.perf_counter() - opt_start_time
        if comm.rank == 0:
            print(f"opt_iter: {opt_iter}, opt_time: {opt_time:.3g} (s), "
                  f"beta: {beta}, C: {C_value:.3f}, V: {V_value:.3f}, "
                  f"U: {U_value:.3f}, change: {change:.3f}", flush=True)

    values = S_comm.gather(rho_phys_field)
    if comm.rank == 0:
        plotter.plot(values)
    save_xdmf(fem["mesh"], rho_phys_field)

    rho_S0_comm = Communicator(rho_field.function_space, fem["mesh_serial"])
    rho_global = rho_S0_comm.gather(rho_field)
    if comm.rank == 0:
        np.save("output/rho_converged.npy", rho_global)
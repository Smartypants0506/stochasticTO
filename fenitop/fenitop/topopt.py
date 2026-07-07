"""
Main topology optimization driver.
"""

import time

import numpy as np
from mpi4py import MPI

from fenitop.fem import form_fem
from fenitop.parameterize import DensityFilter, Heaviside
from fenitop.sensitivity import Sensitivity
from fenitop.optimize import optimality_criteria, mma_optimizer
from fenitop.utility import Communicator, Plotter, save_xdmf


def topopt(fem, opt):
    """Main function for topology optimization."""

    comm = MPI.COMM_WORLD
    linear_problem, u_field, lambda_field, rho_field, rho_phys_field = form_fem(fem, opt)
    density_filter = DensityFilter(comm, rho_field, rho_phys_field,
        opt["filter_radius"], fem["petsc_options"])
    heaviside = Heaviside(rho_phys_field)
    sens_problem = Sensitivity(comm, opt, linear_problem, u_field, lambda_field, rho_phys_field)
    S_comm = Communicator(rho_phys_field.function_space, fem["mesh_serial"])
    if comm.rank == 0:
        plotter = Plotter(fem["mesh_serial"])
    num_consts = 1 if opt["opt_compliance"] else 2
    num_elems = rho_field.x.petsc_vec.array.size

    # Always initialize -- these are read every iteration regardless of
    # whether OC or MMA is used for the design update.
    rho_old1, rho_old2 = np.zeros(num_elems), np.zeros(num_elems)
    low, upp = None, None

    centers = rho_field.function_space.tabulate_dof_coordinates()[:num_elems].T
    solid, void = opt["solid_zone"](centers), opt["void_zone"](centers)
    rho_ini = np.full(num_elems, opt["vol_frac"])
    rho_ini[solid], rho_ini[void] = 0.995, 0.005
    rho_field.x.petsc_vec.array[:] = rho_ini
    rho_min, rho_max = np.zeros(num_elems), np.ones(num_elems)
    rho_min[solid], rho_max[void] = 0.99, 0.01

    plot_interval = opt.get("plot_interval", 10)
    opt_iter, beta, change = 0, 1, 2*opt["opt_tol"]
    while opt_iter < opt["max_iter"] and change > opt["opt_tol"]:
        opt_start_time = time.perf_counter()
        opt_iter += 1

        density_filter.forward()
        if opt_iter % opt["beta_interval"] == 0 and beta < opt["beta_max"]:
            beta *= 2
            change = opt["opt_tol"] * 2
        heaviside.forward(beta)

        linear_problem.solve_fem()

        [C_value, V_value, U_value], sensitivities = sens_problem.evaluate()
        heaviside.backward(sensitivities)
        [dCdrho, dVdrho, dUdrho] = density_filter.backward(sensitivities)

        for name, arr in [("dCdrho", dCdrho), ("dVdrho", dVdrho)]:
            if arr is not None:
                local_has_nan = bool(np.isnan(arr).any())
                local_has_inf = bool(np.isinf(arr).any())
                local_min = float(np.min(arr)) if arr.size else np.inf
                local_max = float(np.max(arr)) if arr.size else -np.inf
                global_nan = comm.allreduce(local_has_nan, op=MPI.LOR)
                global_inf = comm.allreduce(local_has_inf, op=MPI.LOR)
                global_min = comm.allreduce(local_min, op=MPI.MIN)
                global_max = comm.allreduce(local_max, op=MPI.MAX)
                if comm.rank == 0:
                    print(f"{name}: nan={global_nan}, inf={global_inf}, "
                        f"min={global_min:.3e}, max={global_max:.3e}", flush=True)
        if opt["opt_compliance"]:
            g_vec = np.array([V_value-opt["vol_frac"]])
            dJdrho, dgdrho = dCdrho, np.vstack([dVdrho])
        else:
            g_vec = np.array([V_value-opt["vol_frac"], C_value-opt["compliance_bound"]])
            dJdrho, dgdrho = dUdrho, np.vstack([dVdrho, dCdrho])

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

        opt_time = time.perf_counter() - opt_start_time
        if comm.rank == 0:
            print(f"opt_iter: {opt_iter}, opt_time: {opt_time:.3g} (s), "
                  f"beta: {beta}, C: {C_value:.3f}, V: {V_value:.3f}, "
                  f"U: {U_value:.3f}, change: {change:.3f}", flush=True)

        is_last_iter = (opt_iter == opt["max_iter"]) or (change <= opt["opt_tol"])
        if opt_iter % plot_interval == 0 or is_last_iter:
            values = S_comm.gather(rho_phys_field)
            if comm.rank == 0:
                plotter.plot(values)
                save_xdmf(fem["mesh"], rho_phys_field)
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


def topopt(fem, opt):
    """Main function for topology optimization."""

    # Initialization
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
    if not opt["use_oc"]:
        rho_old1, rho_old2 = np.zeros(num_elems), np.zeros(num_elems)
        low, upp = None, None

    # Apply passive zones
    centers = rho_field.function_space.tabulate_dof_coordinates()[:num_elems].T
    solid, void = opt["solid_zone"](centers), opt["void_zone"](centers)
    rho_ini = np.full(num_elems, opt["vol_frac"])
    rho_ini[solid], rho_ini[void] = 0.995, 0.005
    rho_field.x.petsc_vec.array[:] = rho_ini
    rho_min, rho_max = np.zeros(num_elems), np.ones(num_elems)
    rho_min[solid], rho_max[void] = 0.99, 0.01

    # Start topology optimization
    opt_iter, beta, change = 0, 1, 2*opt["opt_tol"]
    while opt_iter < opt["max_iter"] and change > opt["opt_tol"]:
        opt_start_time = time.perf_counter()
        opt_iter += 1

        # Density filter and Heaviside projection
        density_filter.forward()
        if opt_iter % opt["beta_interval"] == 0 and beta < opt["beta_max"]:
            beta *= 2
            change = opt["opt_tol"] * 2
        heaviside.forward(beta)

        # Solve FEM
        linear_problem.solve_fem()

        u_array = u_field.x.petsc_vec.array  # local dofs, interleaved [ux,uy,uz,...]
        u_reshaped = u_array.reshape(-1, 3)  # 3D problem
        disp_mag = np.linalg.norm(u_reshaped, axis=1)
        local_max_disp = disp_mag.max() if disp_mag.size > 0 else 0.0
        global_max_disp = comm.allreduce(local_max_disp, op=MPI.MAX)
        if comm.rank == 0 and opt_iter % 10 == 0:  # print every 10 iters to avoid spam
            print(f"  [diag] opt_iter {opt_iter}: max |u| = {global_max_disp:.6e} m "
                f"({global_max_disp*1000:.6f} mm)", flush=True)
                
        # Compute function values and sensitivities
        [C_value, V_value, U_value], sensitivities = sens_problem.evaluate()
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

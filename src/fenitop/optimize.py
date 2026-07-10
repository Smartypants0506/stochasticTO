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

import numpy as np
from mpi4py import MPI
from scipy import sparse as sparse
from scipy.linalg import solve
from petsc4py import PETSc
from src.fenitop.mma import MMA

def optimality_criteria(rho, rho_min, rho_max, V, dCdrho, dVdrho, move=0.05):
    """Solution update scheme with optimality criteria (OC)."""
    lb, ub = 0.0, 1e6
    comm = MPI.COMM_WORLD
    while ub-lb > 1e-4:
        mid = (lb+ub) / 2.0
        rho_new = np.maximum.reduce([np.minimum.reduce(
            [rho*(-dCdrho/(dVdrho+1e-12)/mid)**0.5, rho+move, rho_max]), rho-move, rho_min])
        dV = comm.allreduce(dVdrho@(rho_new-rho), op=MPI.SUM)
        if V + dV > 0:
            lb = mid
        else:
            ub = mid
    change = comm.allreduce(np.max(np.abs(rho_new-rho), initial=0), op=MPI.MAX)
    return rho_new, change


def mma_optimizer(m, n, opt_iter, xval, xmin, xmax, xold1, xold2, df0dx, fval,
                  dfdx, low, upp, a0=1, a=None, c=None, d=None, move=0.05,
                  asyinit=0.5, asydecr=0.7, asyincr=1.2,
                  low_bnd=0.002, up_bnd=1.0, albefa=0.1, feps=1e-6):
    """Solution update scheme with the method of moving asymptotes (MMA) using TAO.
    This replaces the custom primal-dual interior-point solver with the TAO MMA class.
    """
    comm = MPI.COMM_WORLD
    
    # Create TAO solver and set MMA as the context
    tao = PETSc.TAO().create(comm)
    tao.setType(PETSc.TAO.Type.PYTHON)
    mma_ctx = MMA()
    tao.setPythonContext(mma_ctx)
    
    # We restrict TAO to exactly 1 iteration to act as a drop-in update scheme step
    tao.setMaximumIterations(1)
    
    # Set variable and bounds vectors
    x = PETSc.Vec().createMPI(n, comm=comm)
    x.setArray(xval)
    
    xl = PETSc.Vec().createMPI(n, comm=comm)
    xl.setArray(xmin)
    
    xu = PETSc.Vec().createMPI(n, comm=comm)
    xu.setArray(xmax)
    
    tao.setVariableBounds((xl, xu))
    tao.setSolution(x)
    
    # Objective and gradient callback
    def objgrad_cb(tao_obj, x_vec, g_vec):
        g_vec.setArray(df0dx)
        return 0.0  # MMA primarily uses the gradient for the subproblem
        
    g = x.duplicate()
    tao.setObjectiveGradient(objgrad_cb, g)
    
    # Inequality constraints callback
    if m > 0:
        c_vec = PETSc.Vec().createMPI(m, comm=comm)
        J_mat = PETSc.Mat().createAIJ([m, n], comm=comm)
        J_mat.setUp()
        
        def constr_cb(tao_obj, x_vec, c_out):
            c_out.setArray(fval)
            
        def jac_cb(tao_obj, x_vec, J_out, P_out):
            # Format the Jacobian matrix
            J_out.setValues(range(m), range(n), dfdx.reshape(m, n))
            J_out.assemble()
            P_out.assemble()
            
        tao.setInequalityConstraints(c_vec, constr_cb)
        tao.setJacobianInequality(J_mat, J_mat, jac_cb)
        
    # Optional parameters can be passed to the underlying options database here if desired
    # e.g., PETSc.Options().setValue("tao_mma_move_limit", move)
    tao.setFromOptions()
    
    # Solve the 1-step iteration
    tao.solve()
    
    # Extract results
    x_new = x.getArray().copy()
    change = comm.allreduce(np.max(np.abs(x_new - xval), initial=0), op=MPI.MAX)
    
    # Attempt to extract internal asymptotes for continuity in external loops
    try:
        low_new = mma_ctx._L.getArray().copy()
        upp_new = mma_ctx._U.getArray().copy()
    except AttributeError:
        # Fallback if TAO did not take a step or attributes aren't populated
        low_new = low
        upp_new = upp
        
    return x_new, change, low_new, upp_new
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

import logging

import numpy as np
from scipy.spatial import cKDTree
from petsc4py import PETSc
import dolfinx.io
from dolfinx.fem import form, Function
import dolfinx.la.petsc as la_petsc
from dolfinx.fem.petsc import (create_vector, create_matrix,
                               assemble_vector, assemble_matrix, set_bc)
from dolfinx import la

import pyvista

logger = logging.getLogger(__name__)

def build_nullspace(V):
    """Build PETSc near-nullspace for 3D elasticity (rigid body modes).
    Pattern verified against dolfinx's demo_elasticity.py, stable 0.9-0.12.dev0.
    """
    dtype = PETSc.ScalarType
    bs = V.dofmap.index_map_bs
    length0 = V.dofmap.index_map.size_local
    basis = [la.vector(V.dofmap.index_map, bs=bs, dtype=dtype) for _ in range(6)]
    b = [bi.array for bi in basis]

    dofs = [V.sub(i).dofmap.list.flatten() for i in range(3)]
    for i in range(3):
        b[i][dofs[i]] = 1.0

    x = V.tabulate_dof_coordinates()
    dofs_block = V.dofmap.list.flatten()
    x0, x1, x2 = x[dofs_block, 0], x[dofs_block, 1], x[dofs_block, 2]
    b[3][dofs[0]] = -x1; b[3][dofs[1]] = x0
    b[4][dofs[0]] = x2;  b[4][dofs[2]] = -x0
    b[5][dofs[2]] = x1;  b[5][dofs[1]] = -x2

    la.orthonormalize(basis)
    basis_petsc = [
        PETSc.Vec().createWithArray(xi[: bs * length0], bsize=3, comm=V.mesh.comm)
        for xi in b
    ]
    return PETSc.NullSpace().create(vectors=basis_petsc)

def create_mechanism_vectors(func_space, in_spring, out_spring):
    """Create vectors for compliant mechanism design."""
    index_map = func_space.dofmaps[0].index_map
    block_size = func_space.dofmaps[0].index_map_bs
    spring_vec = la.create_petsc_vector(index_map, block_size)
    l_vec = spring_vec.copy()

    local_range = index_map.local_range
    local_indices = np.arange(local_range[0], local_range[1]).astype(np.int32)
    local_size = np.ptp(local_range)
    local_nodes = func_space.tabulate_dof_coordinates()[:local_size]

    for n, (locator, direction, value) in enumerate([in_spring, out_spring]):
        ctrl_nodes = local_indices[locator(local_nodes.T)]
        offset = ["x", "y", "z"].index(direction)
        ctrl_dofs = ctrl_nodes*block_size + offset
        spring_vec.setValues(ctrl_dofs, [value,]*ctrl_dofs.size)
        if n == 1:
            l_vec.setValues(ctrl_dofs, [1.0,]*ctrl_dofs.size)
    spring_vec.assemble()
    l_vec.assemble()
    return spring_vec, l_vec


class LinearProblem:
    def __init__(self, u, lam, lhs, rhs, l_vec, spring_vec, bcs=[], petsc_options={}):
        """Initialize a linear problem."""
        # Initialization
        self.u, self.lam = u, lam
        self.u_wrap = la_petsc.create_vector_wrap(self.u.x)
        self.lam_wrap = la_petsc.create_vector_wrap(self.lam.x)
        self.lhs_form, self.rhs_form = form(lhs), form(rhs)
        self.lhs_mat = create_matrix(self.lhs_form)

        # Rigid-body near-nullspace: without this, GAMG coarsens poorly as
        # SIMP density contrast grows (eps=1e-6), and CG can stall/hang on
        # near-disconnected material rather than fail. 3D vector elasticity
        # only (gdim==3) -- this project's box/STEP meshes are both 3D.
        if self.u.function_space.mesh.geometry.dim == 3:
            self.lhs_mat.setNearNullSpace(build_nullspace(self.u.function_space))
            
        self.rhs_vec = create_vector(self.rhs_form.function_spaces[0])
        self.bcs, self.l_vec, self.spring_vec = bcs, l_vec, spring_vec

        # Construct a linear solver
        self.solver = PETSc.KSP().create(self.u.function_space.mesh.comm)
        self.solver.setOperators(self.lhs_mat)
        prefix = f"linear_solver_{id(self)}"
        self.solver.setOptionsPrefix(prefix)

        # Apply PETSc options
        opts = PETSc.Options()
        opts.prefixPush(prefix)
        for key, value in petsc_options.items():
            opts[key] = value
        opts.prefixPop()
        self.solver.setFromOptions()
        for var in [self.lhs_mat, self.rhs_vec, self.l_vec]:
            if var is not None:
                var.setOptionsPrefix(prefix)
                var.setFromOptions()

        # Override whatever petsc_options set for ksp_error_if_not_converged:
        # that option makes a non-converged KSPSolve raise via SETERRQ, which
        # exits KSPSolve_Private before the normal PCPostSolve cleanup runs and
        # leaves the PC's internal reentrancy-depth counter incremented.
        # Calling KSPSolve again on the SAME KSP after that (as solve_fem's
        # reuse-fallback retry does) then trips PETSc's "Cannot embed
        # PCPreSolve() more than twice" guard. Enforcing convergence ourselves
        # in Python (see solve_fem) preserves the exact same "never silently
        # use a non-converged solve" guarantee via a normal-returning solve +
        # explicit getConvergedReason() check, which is always safe to retry.
        self.solver.setErrorIfNotConverged(False)

        assemble_vector(self.rhs_vec, self.rhs_form)
        self.rhs_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(self.rhs_vec, self.bcs)

        # Tracks whether the PC is currently frozen (see set_reuse_preconditioner),
        # so solve_fem can tell a reuse-caused failure apart from a genuine one.
        self._reuse_pc_active = False
        self._warned_no_pc_reuse = False

    def enable_warm_start(self, flag=True):
        """Use the current contents of u (the previous solve's solution) as the
        CG initial guess instead of zero. Math-exact: CG still converges to the
        same solution at the same tolerance; only the iteration count changes.
        The Dirichlet BC here is homogeneous (Constant 0), so the retained u
        already satisfies u=0 at constrained dofs, keeping the guess admissible.
        """
        self.solver.setInitialGuessNonzero(flag)

    def set_reuse_preconditioner(self, flag=True):
        """Freeze (or unfreeze) the assembled preconditioner (e.g. the GAMG
        multigrid hierarchy) across subsequent solves. When True, solve_fem()
        still re-assembles the true matrix and CG iterates against it to the same
        tolerance -- only the (expensive) PC *setup* is skipped. Used across a
        batch of mildly-varying matrices (fixed nominal design, only eta(x)
        varies) to turn ~N GAMG setups into 1. The first solve must run with
        this False so the PC is built once.

        The reuse flag really lives on the PC (PCSetReusePreconditioner). The
        KSP-level convenience wrapper (KSP.setReusePreconditioner) only exists in
        newer petsc4py, so prefer the PC method and fall back to the KSP method,
        to work across versions.
        """
        pc = self.solver.getPC()
        if hasattr(pc, "setReusePreconditioner"):
            pc.setReusePreconditioner(flag)
            self._reuse_pc_active = bool(flag)
        elif hasattr(self.solver, "setReusePreconditioner"):
            self.solver.setReusePreconditioner(flag)
            self._reuse_pc_active = bool(flag)
        else:
            # Preconditioner reuse is a pure speed optimization. If this petsc4py
            # exposes no toggle on either PC or KSP, skip it -- the PC just
            # rebuilds every solve (the original, correct-but-slower behavior) --
            # rather than crashing the whole run over an optional speedup.
            if not self._warned_no_pc_reuse:
                logger.warning(
                    "petsc4py exposes no setReusePreconditioner on PC or KSP; "
                    "preconditioner reuse disabled (solves remain correct, the "
                    "PC is just rebuilt each solve)."
                )
                self._warned_no_pc_reuse = True
            self._reuse_pc_active = False

    def solve_fem(self):
        """Solve K*x=F for FEM."""
        self.lhs_mat.zeroEntries()
        assemble_matrix(self.lhs_mat, self.lhs_form, bcs=self.bcs)
        self.lhs_mat.assemble()
        if self.spring_vec is not None:
            self.lhs_mat.setDiagonal(self.lhs_mat.getDiagonal()+self.spring_vec)

        self.solver.solve(self.rhs_vec, self.u_wrap)
        reason = self.solver.getConvergedReason()

        if reason < 0:
            # ANY non-convergence falls back to the safest possible solve:
            # fresh preconditioner AND zero initial guess. This is the original
            # pre-warm-start, pre-reuse configuration, so if the problem is
            # solvable at all this should solve it.
            #
            # The retry used to be gated on `self._reuse_pc_active`, on the
            # reasoning that without PC reuse a failure must be a genuine
            # near-singular design. That reasoning missed the WARM START, which
            # is enabled independently (enable_warm_start) and is the other
            # thing that can wreck a solve: the previous sample's displacement
            # field is a poor starting point for a substantially different
            # matrix, and KSP then reports DIVERGED_DTOL (reason=-4) because the
            # residual grew relative to its initial value. The Monte Carlo loop
            # runs with warm start ON and reuse OFF, so the zero-initial-guess
            # retry -- the one fallback that addresses that failure mode -- was
            # unreachable in exactly the configuration that needed it.
            logger.warning(
                "KSP solve failed to converge (reason=%d, reuse_pc=%s, "
                "warm_start=%s); retrying once from a zero initial guess with a "
                "freshly built preconditioner.",
                reason, self._reuse_pc_active,
                self.solver.getInitialGuessNonzero(),
            )
            self.set_reuse_preconditioner(False)
            # The failed solve may have left u_wrap holding a poor (or
            # non-finite) iterate; a zero initial guess makes KSP ignore it
            # entirely. Restore whatever warm-start setting the caller had
            # afterwards. lhs_mat/rhs_vec are already assembled and unchanged,
            # so no reassembly is needed.
            had_warm_start = self.solver.getInitialGuessNonzero()
            if had_warm_start:
                self.solver.setInitialGuessNonzero(False)
            try:
                self.solver.solve(self.rhs_vec, self.u_wrap)
                retry_reason = self.solver.getConvergedReason()
            finally:
                if had_warm_start:
                    self.solver.setInitialGuessNonzero(True)

            if retry_reason < 0:
                # The safest configuration also failed, so this is a genuine
                # property of the system, not an artifact of an optimization:
                # under a sufficiently eroded eta(x) draw the structure can be
                # near-disconnected and K near-singular. Callers that can
                # tolerate it (the MC loop) catch this and record the sample as
                # a failure; that failure RATE is a robustness result.
                raise RuntimeError(
                    "KSP solve failed to converge even from a zero initial "
                    f"guess with a fresh preconditioner (first attempt "
                    f"reason={reason}, retry reason={retry_reason}). The "
                    "system is genuinely near-singular for this design and "
                    "eta(x) realization."
                )

        self.u.x.scatter_forward()

    def solve_adjoint(self):
        """Solve K*lambda=-L for the adjoint equation."""
        self.solver.solve(-self.l_vec, self.lam_wrap)
        self.lam.x.scatter_forward()

    def __del__(self):
        self.solver.destroy()
        self.lhs_mat.destroy()
        self.rhs_vec.destroy()
        self.u_wrap.destroy()
        self.lam_wrap.destroy()
        if self.spring_vec is not None:
            self.spring_vec.destroy()
            self.l_vec.destroy()


class Communicator():
    """Communicate information among different processes."""

    def __init__(self, func_space, mesh_serial, size=1):
        self.size = size
        self.comm = func_space.mesh.comm
        idx_map = func_space.dofmaps[0].index_map

        num_local_nodes = idx_map.size_local
        num_global_nodes = idx_map.size_global
        num_nodal_dofs = func_space.dofmaps[0].index_map_bs
        self.num_global_dofs = num_global_nodes * num_nodal_dofs

        local_nodal_range = np.asarray(idx_map.local_range, dtype=np.int32)  # [start, end]
        local_dof_range = local_nodal_range * num_nodal_dofs  # [start, end]
        local_nodes = func_space.tabulate_dof_coordinates()[:num_local_nodes]

        # Gather to Process 0
        local_nodal_range_gather = self.comm.gather(local_nodal_range, root=0)
        self.local_dof_range_gather = self.comm.gather(local_dof_range, root=0)
        local_nodes_gather = self.comm.gather(local_nodes, root=0)

        element = func_space.ufl_element()
        if self.comm.rank == 0:
            func_space_serial = dolfinx.fem.functionspace(mesh_serial, element)
            nodes_serial = func_space_serial.tabulate_dof_coordinates()

            nodes_collect = np.zeros((num_global_nodes, 3))
            for r, nodes in zip(local_nodal_range_gather, local_nodes_gather):
                nodes_collect[r[0]:r[1]] = nodes
            global_to_local_nodes = compare_matrices(nodes_serial, nodes_collect)
            local_to_global_nodes = compare_matrices(nodes_collect, nodes_serial)

            def node2dof(nodes, num_nodal_dofs):
                return (np.tile(nodes, (num_nodal_dofs, 1))*num_nodal_dofs
                        + np.arange(num_nodal_dofs).reshape(-1, 1)).ravel("F")

            global_to_local_dofs = node2dof(global_to_local_nodes, num_nodal_dofs)
            self.local_to_global_dofs = node2dof(local_to_global_nodes, num_nodal_dofs)
            self.local_to_global_dofs = (
                np.tile(self.local_to_global_dofs.reshape(-1, 1), (1, size))*size + np.arange(size)).ravel()
        else:
            global_to_local_dofs = None
        global_to_local_dofs = self.comm.bcast(global_to_local_dofs, root=0)
        self.idx = global_to_local_dofs[local_dof_range[0]:local_dof_range[1]]

    def bcast(self, func, global_values):
        """Broadcast data from Process 0 to all the other processes."""
        if func.x.petsc_vec.size != global_values.size:
            raise ValueError("Mismatched sizes.")
        func.x.petsc_vec.array = global_values[self.idx]

    def gather(self, func):
        """Gather data to Process 0 from all the other processes."""
        if type(func) is Function:
            values_gather = self.comm.gather(func.x.petsc_vec.array, root=0)
        elif type(func) is PETSc.Vec:
            values_gather = self.comm.gather(func.array, root=0)
        elif type(func) is np.ndarray:
            values_gather = self.comm.gather(func, root=0)
        else:
            raise TypeError("Unsupported func.")

        if self.comm.rank == 0:
            values_collect = np.zeros(self.num_global_dofs*self.size)
            for r, local_values in zip(self.local_dof_range_gather, values_gather):
                values_collect[r[0]*self.size:r[1]*self.size] = local_values
            global_values = values_collect[self.local_to_global_dofs]
        else:
            global_values = None
        return global_values


def compare_matrices(array1, array2, precision=12, k=1):
    """Find the "args" such that array1[args] == array2."""
    kd_tree = cKDTree(array1.round(precision))
    return kd_tree.query(array2.round(precision), k=k)[1]


class Plotter():
    def __init__(self, mesh):
        """Initialize a plotter."""
        pyvista.OFF_SCREEN = True
        self.dim = mesh.topology.dim
        elements, cell_types, nodes = dolfinx.plot.vtk_mesh(mesh, self.dim)
        self.grid = pyvista.UnstructuredGrid(elements, cell_types, nodes)

    def plot(self, density, threshold=0.49, smooth_iter=100, path=""):
        self.grid.point_data["density"] = np.hstack(density)
        if self.dim == 2:
            grid = self.grid
        else:
            grid = self.grid.threshold(threshold).extract_surface(algorithm='dataset_surface')
        empty_mesh = (self.dim == 3 and grid.n_faces == 0)

        if not empty_mesh:
            if self.dim == 3:
                grid = grid.smooth(n_iter=smooth_iter)
                grid.point_data["density"] = 0.4
            plotter = pyvista.Plotter()
            plotter.background_color = "white"
            lighting = self.dim == 3
            plotter.add_mesh(grid, clim=[0, 1], cmap="Greys", lighting=lighting,
                             show_scalar_bar=False)
            if self.dim == 2:
                plotter.view_xy()
            plotter.screenshot(path+"optimized_design.jpg", window_size=(1000, 1000))
            plotter.close()


def save_xdmf(mesh, rho, path=""):
    xdmf = dolfinx.io.XDMFFile(mesh.comm, path+"optimized_design.xdmf", "w")
    xdmf.write_mesh(mesh)
    rho.name = "density"
    xdmf.write_function(rho)

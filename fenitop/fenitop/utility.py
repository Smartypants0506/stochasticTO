"""
Modified LinearProblem implementing GPU-accelerated efficient reanalysis
(Combined Approximations / PCG) per:
Amir, O., Sigmund, O., Lazarov, B.S., Schevenels, M. (2012).
"Efficient reanalysis techniques for robust topology optimization."
Comput. Methods Appl. Mech. Engrg. 245-246, 217-231.
"""

import numpy as np
from scipy.spatial import cKDTree
from petsc4py import PETSc
import dolfinx.io
from dolfinx.fem import form, Function
from dolfinx import la
from dolfinx.fem.petsc import (create_vector, create_matrix,
    assemble_vector, assemble_matrix, set_bc)
import pyvista


def create_mechanism_vectors(func_space, in_spring, out_spring):
    """Create vectors for compliant mechanism design."""
    index_map = func_space.dofmap.index_map
    block_size = func_space.dofmap.index_map_bs
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


class _ReferencePC:
    """PCG preconditioner that applies K0^{-1} via a reusable, GPU-
    resident reference solver (Amir et al. 2012, Sec. 3.1, Eq. 10-12).
    The reference solve is deliberately cheap (few fixed iterations),
    since it is only meant to act as a preconditioner application, not
    a full accurate solve.
    """
    def __init__(self, ref_solver):
        self.ref_solver = ref_solver

    def apply(self, pc, x, y):
        self.ref_solver.solve(x, y)


class LinearProblem:
    def __init__(self, u, lam, lhs, rhs, l_vec, spring_vec, bcs=[],
                 petsc_options={}, reanalysis_options=None):
        """Initialize a linear problem.

        reanalysis_options (dict, optional): enables the Combined
        Approximations / PCG reanalysis scheme instead of a full
        solve rebuilt from scratch each iteration. Keys:
            enabled (bool)
            max_iter (int): truncated outer PCG iterations per cycle.
            rtol (float): outer PCG relative residual tolerance.
            refactor_interval (int): design cycles between refreshing
                the reference operator K0.
            ref_petsc_options (dict): PETSc options for the reference
                solve (GPU-capable: mat_type/vec_type + hypre BoomerAMG).
        """
        self.u, self.lam = u, lam
        self.u_wrap = la.create_petsc_vector_wrap(self.u.x)
        self.lam_wrap = la.create_petsc_vector_wrap(self.lam.x)
        self.lhs_form, self.rhs_form = form(lhs), form(rhs)
        self.lhs_mat = create_matrix(self.lhs_form)
        self.rhs_vec = create_vector(self.rhs_form)
        self.bcs, self.l_vec, self.spring_vec = bcs, l_vec, spring_vec
        self.comm = self.u.function_space.mesh.comm

        default_reanalysis = {
            "enabled": False,
            "max_iter": 15,
            "rtol": 1e-6,
            "refactor_interval": 50,
            "ref_petsc_options": {
                "ksp_type": "richardson",
                "pc_type": "jacobi",
                "mat_type": "aijcusparse",
                "vec_type": "cuda",
            },
        }
        self.reanalysis = dict(default_reanalysis)
        if reanalysis_options is not None:
            self.reanalysis.update(reanalysis_options)

        prefix = f"linear_solver_{id(self)}"
        self.solver = PETSc.KSP().create(self.comm)
        self.solver.setOptionsPrefix(prefix)

        if self.reanalysis["enabled"]:
            self.ref_mat = None
            ref_prefix = f"ref_solver_{id(self)}"
            self.ref_solver = PETSc.KSP().create(self.comm)
            self.ref_solver.setType(
                self.reanalysis["ref_petsc_options"].get("ksp_type", "richardson"))
            self.ref_solver.getPC().setType(
                self.reanalysis["ref_petsc_options"].get("pc_type", "hypre"))
            self.ref_solver.setOptionsPrefix(ref_prefix)

            opts = PETSc.Options()
            opts.prefixPush(ref_prefix)
            for key, value in self.reanalysis["ref_petsc_options"].items():
                opts[key] = value
            opts.prefixPop()
            self.ref_solver.setFromOptions()
            # Cheap preconditioner application: a fixed handful of
            # BoomerAMG iterations, not a full accurate solve.
            self.ref_solver.setTolerances(rtol=1e-12, max_it=2)
            self.ref_solver.setNormType(PETSc.KSP.NormType.NONE)

            self.solver.setOperators(self.lhs_mat)
            # Flexible CG tolerates the reference solve being an
            # inexact/variable preconditioner across calls.
            self.solver.setType("fcg")
            outer_pc = self.solver.getPC()
            outer_pc.setType("python")
            outer_pc.setPythonContext(_ReferencePC(self.ref_solver))
            self.solver.setInitialGuessNonzero(True)
            self.solver.setTolerances(
                max_it=self.reanalysis["max_iter"],
                rtol=self.reanalysis["rtol"])

            self._ref_ready = False
            self._iter_count = 0
        else:
            self.ref_mat = None
            self.solver.setOperators(self.lhs_mat)
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

        assemble_vector(self.rhs_vec, self.rhs_form)
        self.rhs_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(self.rhs_vec, self.bcs)

    def _assemble_lhs(self):
        self.lhs_mat.zeroEntries()
        assemble_matrix(self.lhs_mat, self.lhs_form, bcs=self.bcs)
        self.lhs_mat.assemble()
        if self.spring_vec is not None:
            self.lhs_mat.setDiagonal(self.lhs_mat.getDiagonal()+self.spring_vec)

    def _maybe_refactorize_reference(self):
        """Refresh the reference operator K0 on a schedule (Sec. 3.3,
        Amir et al. 2012: use the stiffest available design as K0 to
        guarantee convergence of the underlying series for all softer
        designs subsequently reanalyzed against it)."""
        interval = self.reanalysis["refactor_interval"]
        needs_refresh = (not self._ref_ready) or (self._iter_count % interval == 0)
        if needs_refresh:
            if self.ref_mat is None:
                self.ref_mat = self.lhs_mat.duplicate(copy=True)
                mat_type = self.reanalysis["ref_petsc_options"].get("mat_type")
                if mat_type is not None:
                    self.ref_mat.convert(mat_type)
            else:
                self.lhs_mat.copy(self.ref_mat, PETSc.Mat.Structure.SAME_NONZERO_PATTERN)
            self.ref_solver.setOperators(self.ref_mat)
            self.ref_solver.getPC().setReusePreconditioner(False)
            self._ref_ready = True
        else:
            self.ref_solver.getPC().setReusePreconditioner(True)

    def solve_fem(self):
        """Solve K*u=F, using truncated GPU-accelerated PCG reanalysis
        when enabled (Amir et al. 2012, Sec. 3.1, Eq. 10-12)."""
        self._assemble_lhs()
        if self.reanalysis["enabled"]:
            self.solver.setOperators(self.lhs_mat)
            self._maybe_refactorize_reference()
            self._iter_count += 1
            self.solver.solve(self.rhs_vec, self.u_wrap)
        else:
            self.solver.solve(self.rhs_vec, self.u_wrap)
        self.u.x.scatter_forward()

    def solve_adjoint(self):
        """Solve K*lambda=-L for the adjoint equation."""
        self.solver.solve(-self.l_vec, self.lam_wrap)
        self.lam.x.scatter_forward()

    def __del__(self):
        self.solver.destroy()
        if getattr(self, "ref_mat", None) is not None:
            self.ref_mat.destroy()
        if getattr(self, "ref_solver", None) is not None:
            self.ref_solver.destroy()
        self.lhs_mat.destroy()
        self.rhs_vec.destroy()
        self.u_wrap.destroy()
        self.lam_wrap.destroy()
        if self.spring_vec is not None:
            self.spring_vec.destroy()
        if self.l_vec is not None:
            self.l_vec.destroy()


class Communicator():
    """Communicate information among different processes."""

    def __init__(self, func_space, mesh_serial, size=1):
        self.size = size
        self.comm = func_space.mesh.comm
        idx_map = func_space.dofmap.index_map

        num_local_nodes = idx_map.size_local
        num_global_nodes = idx_map.size_global
        num_nodal_dofs = func_space.dofmap.index_map_bs
        self.num_global_dofs = num_global_nodes * num_nodal_dofs

        local_nodal_range = np.asarray(idx_map.local_range, dtype=np.int32)
        local_dof_range = local_nodal_range * num_nodal_dofs
        local_nodes = func_space.tabulate_dof_coordinates()[:num_local_nodes]

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
        if func.x.petsc_vec.size != global_values.size:
            raise ValueError("Mismatched sizes.")
        func.x.petsc_vec.array = global_values[self.idx]

    def gather(self, func):
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
    kd_tree = cKDTree(array1.round(precision))
    return kd_tree.query(array2.round(precision), k=k)[1]


class Plotter():
    def __init__(self, mesh):
        pyvista.OFF_SCREEN = True
        pyvista.start_xvfb()
        self.dim = mesh.topology.dim
        elements, cell_types, nodes = dolfinx.plot.vtk_mesh(mesh, self.dim)
        self.grid = pyvista.UnstructuredGrid(elements, cell_types, nodes)

    def plot(self, density, threshold=0.49, smooth_iter=100, path=""):
        self.grid.point_data["density"] = np.hstack(density)
        if self.dim == 2:
            grid = self.grid
        else:
            grid = self.grid.threshold(threshold).extract_surface()
        empty_mesh = (self.dim == 3 and grid.n_faces_strict == 0)

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
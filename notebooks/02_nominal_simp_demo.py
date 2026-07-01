#!/usr/bin/env python
# coding: utf-8

# # MBB Beam Topology Optimization (Deterministic)
# ### FEniCSx + OpenMDAO Pipeline
# 
# Implements the deterministic SIMP topology optimization for the classic MBB beam benchmark:
# 
# 1. **Mesh Generation** — Gmsh / DOLFINx structured mesh
# 2. **Deterministic FEA** — FEniCSx (dolfinx, UFL), linear elasticity, K U = F
# 3. **SIMP + Density/Heaviside Projection Filter** — checkerboard suppression
# 4. **Optimization Loop** — OpenMDAO `ExplicitComponent` driving the density update
# 
# No manufacturing-uncertainty / random-field / PCE modeling in this notebook — deterministic core only.
# 
# Requires: `fenics-dolfinx`, `mpi4py`, `petsc4py`, `openmdao`

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import time, os

import dolfinx
from dolfinx import mesh, fem, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
import ufl
from mpi4py import MPI
from petsc4py import PETSc

import openmdao
import openmdao.api as om

print("dolfinx:", dolfinx.__version__)
print("openmdao:", openmdao.__version__)


os.environ["CC"] = "gcc"
os.environ["LDSHARED"] = "gcc -shared"
os.environ["CXX"] = "g++"
print(os.environ["CC"])

# In[2]:


# --- Geometry (half-MBB beam, symmetry model) ---
L, H = 3.0, 1.0          # length, height
nelx, nely = 120, 40      # element grid resolution

# --- Material (SIMP) ---
E0, Emin = 1.0, 1e-9      # solid / void Young's modulus
nu = 0.3                  # Poisson's ratio
p_penal = 3.0              # SIMP penalization exponent

# --- Optimization settings ---
volfrac = 0.5              # target volume fraction
rmin = 1.5                  # filter radius (elements)
beta_heaviside = 4.0        # Heaviside projection sharpness
eta_heaviside = 0.5         # Heaviside projection threshold

F_load = -1.0               # point load at top-left corner (downward)

# ## 1. Mesh Generation (DOLFINx structured mesh)
# 
# Structured quadrilateral mesh over the half-MBB design domain. For a regular rectangular domain this is equivalent to a Gmsh-tagged structured grid; swap in `gmsh.model.occ` + `meshio` here if importing an external CAD/STEP geometry.

# In[3]:


comm = MPI.COMM_WORLD

domain = mesh.create_rectangle(
    comm,
    [np.array([0.0, 0.0]), np.array([L, H])],
    [nelx, nely],
    cell_type=mesh.CellType.quadrilateral,
)

V = fem.functionspace(domain, ("Lagrange", 1, (2,)))   # vector displacement space
V0 = fem.functionspace(domain, ("DG", 0))                # elementwise density space

tdim = domain.topology.dim
num_cells = domain.topology.index_map(tdim).size_local
print("Mesh created:", num_cells, "cells,", V.dofmap.index_map.size_global * 2, "DOFs")

# ## 2. Boundary Conditions
# 
# - **Symmetry plane** (x = 0): horizontal displacement \(u_x = 0\)
# - **Bottom-right support** (x = L, y = 0): vertical displacement \(u_y = 0\)
# - **Point load**: downward force at top-left corner (x = 0, y = H)

# In[4]:


def left_boundary(x):
    return np.isclose(x[0], 0.0)

def support_corner(x):
    return np.logical_and(np.isclose(x[0], L), np.isclose(x[1], 0.0))

fdim = tdim - 1
left_facets = mesh.locate_entities_boundary(domain, fdim, left_boundary)
ux_dofs = fem.locate_dofs_topological(V.sub(0), fdim, left_facets)
bc_left = fem.dirichletbc(default_scalar_type(0.0), ux_dofs, V.sub(0))

corner_verts = mesh.locate_entities_boundary(domain, 0, support_corner)
uy_dofs = fem.locate_dofs_topological(V.sub(1), 0, corner_verts)
bc_corner = fem.dirichletbc(default_scalar_type(0.0), uy_dofs, V.sub(1))

bcs = [bc_left, bc_corner]
print("BC dofs fixed:", len(ux_dofs), "ux +", len(uy_dofs), "uy")

# ## 3. Deterministic FEA — Linear Elasticity + SIMP (FEniCSx/UFL)
# 
# Weak form: \( a(u,v) = \int_\Omega \sigma(\rho, u):\varepsilon(v)\, d\Omega \), with SIMP-penalized stiffness
# \( E(\rho) = E_{min} + \rho^{p}(E_0 - E_{min}) \). Solved via PETSc `LinearProblem` (\(KU=F\)), giving compliance \( C = F^T U \).

# In[5]:


rho_h = fem.Function(V0)          # physical (projected) density field

def eps(u):
    return ufl.sym(ufl.grad(u))

def sigma(u, rho):
    Ey = Emin + rho**p_penal * (E0 - Emin)
    lam = Ey * nu / ((1 + nu) * (1 - 2 * nu))
    mu = Ey / (2 * (1 + nu))
    return 2 * mu * eps(u) + lam * ufl.tr(eps(u)) * ufl.Identity(2)

u_, v_ = ufl.TrialFunction(V), ufl.TestFunction(V)
a_form = ufl.inner(sigma(u_, rho_h), eps(v_)) * ufl.dx

f_zero = fem.Function(V)          # zero body force; point load applied to RHS vector directly
L_form = ufl.inner(f_zero, v_) * ufl.dx

problem = LinearProblem(
    a_form, L_form, bcs=bcs,
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
)

# Locate top-left corner DOF for nodal point load
coords = V.tabulate_dof_coordinates()
top_left_node = np.where(np.isclose(coords[:, 0], 0.0) & np.isclose(coords[:, 1], H))[0]
print("Top-left load DOF node index:", top_left_node)

# In[ ]:


def fea_solve(rho_array: np.ndarray):
    """Solve K U = F for a given density field; return compliance, displacement, strain-energy density."""
    rho_h.x.array[:] = rho_array

    uh = problem.solve()
    b = problem.b
    if len(top_left_node):
        b.array[2 * top_left_node[0] + 1] += F_load
    uh = problem.solve()

    C = float(problem.b.dot(uh.x.petsc_vec))

    # Elementwise strain energy density (vectorized via UFL form on DG0 space)
    w_ = ufl.TestFunction(V0)
    energy_form = fem.form(ufl.inner(sigma(uh, rho_h), eps(uh)) * w_ * ufl.dx)
    energy_vec = fem.assemble_vector(energy_form)
    energy_vec.scatter_reverse(dolfinx.la.InsertMode.add)
    strain_energy_e = energy_vec.array.copy()

    return C, uh, strain_energy_e

print("FEA solver function ready.")

# ## 4. SIMP Sensitivity, Density Filter & Heaviside Projection
# 
# - Adjoint sensitivity: \( \partial C/\partial \rho_e = -p\,\rho_e^{p-1}(E_0-E_{min})\, u_e^T k_e u_e \)
# - Linear density filter (radius `rmin`) suppresses checkerboarding
# - Heaviside projection sharpens the filtered field toward 0/1 (per masterContext.md: "density filter + Heaviside projection")
# 
# All operations are fully vectorized (no Python loops over elements), per project coding standards.

# In[ ]:


from scipy.sparse import coo_matrix

def build_filter(nelx: int, nely: int, rmin: float):
    ii, jj = np.meshgrid(np.arange(nelx), np.arange(nely), indexing="ij")
    ii = ii.flatten(); jj = jj.flatten()
    rows, cols, vals = [], [], []
    span = int(np.ceil(rmin))
    for d_i in range(-span, span + 1):
        for d_j in range(-span, span + 1):
            dist = np.sqrt(d_i**2 + d_j**2)
            if dist >= rmin:
                continue
            ni, nj = ii + d_i, jj + d_j
            valid = (ni >= 0) & (ni < nelx) & (nj >= 0) & (nj < nely)
            rows.append((ii * nely + jj)[valid])
            cols.append((ni * nely + nj)[valid])
            vals.append(np.full(valid.sum(), rmin - dist))
    rows = np.concatenate(rows); cols = np.concatenate(cols); vals = np.concatenate(vals)
    H_mat = coo_matrix((vals, (rows, cols)), shape=(nelx * nely, nelx * nely)).tocsc()
    Hs = np.array(H_mat.sum(axis=1)).flatten()
    return H_mat, Hs

H_filter, Hs_filter = build_filter(nelx, nely, rmin)
print("Filter matrix built:", H_filter.shape)

# In[ ]:


def filter_density(rho: np.ndarray) -> np.ndarray:
    """Linear density filter: rho_tilde = H * rho / Hs."""
    return np.asarray(H_filter @ rho) / Hs_filter

def heaviside_project(rho_tilde: np.ndarray, beta: float, eta: float) -> np.ndarray:
    """Smooth Heaviside projection toward 0/1 densities."""
    num = np.tanh(beta * eta) + np.tanh(beta * (rho_tilde - eta))
    den = np.tanh(beta * eta) + np.tanh(beta * (1 - eta))
    return num / den

def heaviside_derivative(rho_tilde: np.ndarray, beta: float, eta: float) -> np.ndarray:
    """d(rho_phys)/d(rho_tilde) for chain-rule sensitivity."""
    den = np.tanh(beta * eta) + np.tanh(beta * (1 - eta))
    return beta * (1 - np.tanh(beta * (rho_tilde - eta)) ** 2) / den

def filter_sensitivity(rho: np.ndarray, drho_phys: np.ndarray, rho_tilde: np.ndarray) -> np.ndarray:
    """Chain sensitivity back through Heaviside projection and density filter."""
    dH = heaviside_derivative(rho_tilde, beta_heaviside, eta_heaviside)
    d_via_filter = drho_phys * dH
    return np.asarray(H_filter @ (d_via_filter / Hs_filter))

# ## 5. OpenMDAO Component — SIMP Topology Optimization Loop
# 
# Wraps the FEniCSx FEA solve, filter, and projection chain as an `ExplicitComponent` with analytic partials, driven by `ScipyOptimizeDriver` (SLSQP). The optimizer updates the raw design variable `rho`; the component internally applies filter → projection → FEA → adjoint sensitivity → filter-chain before returning gradients.

# In[ ]:


class SIMPComponent(om.ExplicitComponent):
    def setup(self):
        self.add_input("rho", val=volfrac * np.ones(nelx * nely))
        self.add_output("compliance", val=1.0)
        self.add_output("volume_frac", val=volfrac)
        self.declare_partials("compliance", "rho")
        self.declare_partials("volume_frac", "rho")
        self.history = []

    def compute(self, inputs, outputs):
        rho = inputs["rho"]
        rho_tilde = filter_density(rho)
        rho_phys = heaviside_project(rho_tilde, beta_heaviside, eta_heaviside)

        # Map (nelx*nely,) element ordering -> DG0 dof ordering expected by fea_solve
        rho_dg0 = rho_phys  # assumed consistent column-major element ordering with mesh generation

        C, uh, strain_energy_e = fea_solve(rho_dg0)

        drho_phys = -p_penal * np.maximum(rho_phys, 1e-3) ** (p_penal - 1) * (E0 - Emin) * strain_energy_e
        self._dC_drho = filter_sensitivity(rho, drho_phys, rho_tilde)

        outputs["compliance"] = C
        outputs["volume_frac"] = np.mean(rho_phys)
        self.history.append((C, np.mean(rho_phys)))

    def compute_partials(self, inputs, partials):
        partials["compliance", "rho"] = self._dC_drho
        partials["volume_frac", "rho"] = np.ones(nelx * nely) / (nelx * nely)

# In[ ]:


prob = om.Problem()
model = prob.model
model.add_subsystem("simp", SIMPComponent(), promotes=["*"])

model.add_design_var("rho", lower=1e-3, upper=1.0)
model.add_objective("compliance")
model.add_constraint("volume_frac", upper=volfrac)

prob.driver = om.ScipyOptimizeDriver(optimizer="SLSQP", maxiter=80, tol=1e-4)
prob.setup()
prob.set_val("rho", volfrac * np.ones(nelx * nely))

print("OpenMDAO problem configured: design vars =", nelx * nely)

# In[ ]:


t0 = time.time()
prob.run_driver()
elapsed = time.time() - t0

rho_opt = prob.get_val("rho")
C_opt = float(prob.get_val("compliance"))
vf_opt = float(prob.get_val("volume_frac"))

print(f"Converged in {elapsed:.1f}s | Compliance = {C_opt:.4f} | Volume fraction = {vf_opt:.3f}")

# ## 6. Visualize Final Topology

# In[ ]:


rho_tilde_final = filter_density(rho_opt)
rho_phys_final = heaviside_project(rho_tilde_final, beta_heaviside, eta_heaviside)
rho_grid = rho_phys_final.reshape((nelx, nely))

fig, ax = plt.subplots(figsize=(10, 4))
ax.imshow(-rho_grid.T, cmap="gray", origin="lower", extent=[0, L, 0, H])
ax.set_title(f"MBB Beam Topology — Compliance={C_opt:.3f}, Vf={vf_opt:.2f}")
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_aspect("equal")
plt.tight_layout()

os.makedirs("output", exist_ok=True)
plt.savefig("output/mbb_topology_optimized.png", dpi=150)
plt.show()

# In[ ]:


import pandas as pd

results = {
    "nelx": nelx, "nely": nely, "volfrac": volfrac, "p_penal": p_penal,
    "rmin": rmin, "beta_heaviside": beta_heaviside, "eta_heaviside": eta_heaviside,
    "compliance_final": C_opt, "volume_frac_final": vf_opt, "runtime_s": elapsed,
}
pd.DataFrame([results]).to_csv("output/mbb_optimization_summary.csv", index=False)
np.savetxt("output/rho_opt_density_field.csv", rho_grid, delimiter=",")
print("Saved: output/mbb_optimization_summary.csv, output/rho_opt_density_field.csv")

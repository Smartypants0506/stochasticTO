import numpy as np
import openturns as ot
import matplotlib.pyplot as plt
from mpi4py import MPI
import ufl
import dolfinx
from dolfinx import mesh, fem, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
from petsc4py import PETSc

ot.RandomGenerator.SetSeed(43)
np.random.seed(43)

# %%
nelx, nely = 60, 20
L, H = 3.0, 1.0

domain = mesh.create_rectangle(
    MPI.COMM_WORLD,
    [np.array([0.0, 0.0]), np.array([L, H])],
    [nelx, nely],
    cell_type=mesh.CellType.quadrilateral,
)

V = fem.functionspace(domain, ("Lagrange", 1, (2,)))  # vector displacement space
D0 = fem.functionspace(domain, ("DG", 0))  # element-wise density space

rho = fem.Function(D0)
rho.x.array[:] = 1.0  # nominal full-material design (start point for TO)

E0, Emin, nu, penal = 1.0, 1e-9, 0.3, 3.0


def simp_E(rho_func):
    return Emin + (E0 - Emin) * rho_func**penal


def sigma(u, E):
    eps = ufl.sym(ufl.grad(u))
    mu = E / (2 * (1 + nu))
    lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
    return 2 * mu * eps + lmbda * ufl.tr(eps) * ufl.Identity(2)


# %%
def left_boundary(x):
    return np.isclose(x[0], 0.0)

def bottom_right_point(x):
    return np.logical_and(np.isclose(x[0], L), np.isclose(x[1], 0.0))

# Collapse the sub-spaces to cleanly isolate individual coordinate components
V_x, _ = V.sub(0).collapse()
V_y, _ = V.sub(1).collapse()

# Correct left boundary: Map through the collapsed tuple and unpack index [0] to get true parent X DOFs
facets_left = mesh.locate_entities_boundary(domain, domain.topology.dim - 1, left_boundary)
dofs_left_x = fem.locate_dofs_topological((V.sub(0), V_x), domain.topology.dim - 1, facets_left)[0]
bc_left = fem.dirichletbc(default_scalar_type(0.0), dofs_left_x, V.sub(0))

# Correct corner boundary: Pin the bottom right corner in Y to prevent rigid body rotation
dofs_corner_y = fem.locate_dofs_geometrical((V.sub(1), V_y), bottom_right_point)[0]
bc_corner = fem.dirichletbc(default_scalar_type(0.0), dofs_corner_y, V.sub(1))

bcs = [bc_left, bc_corner]


def fenicsx_fea(rho_array, return_field=False):
    """Solves linear elasticity on the MBB domain for a given density field.
    Returns compliance C = F^T U (objective in masterContext robust TO)."""
    rho.x.array[:] = rho_array
    E = simp_E(rho)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(sigma(u, E), ufl.sym(ufl.grad(v))) * ufl.dx

    f = fem.Function(V)
    
    # 1. Properly collapse the subspace view as done previously
    V_y, _ = V.sub(1).collapse()
    top_left_dof = fem.locate_dofs_geometrical(
        (V.sub(1), V_y), lambda x: np.logical_and(np.isclose(x[0], 0.0), np.isclose(x[1], H))
    )[0]
    
    # 2. Insert a point force value into the f vector at that DOF
    # (Assuming a downward unit force of -1.0 in the y-direction)
    f.x.array[top_left_dof] = -1.0
    
    # 3. Update the linear form to integrate this load vector
    L_form = ufl.inner(f, v) * ufl.dx

    problem = LinearProblem(
        a, L_form, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
    )
    uh = problem.solve()

    F_vec = problem.b.copy()
    compliance = F_vec.dot(uh.x.petsc_vec)
    if return_field:
        return compliance, uh
    return compliance


C_nominal = fenicsx_fea(np.ones(D0.dofmap.index_map.size_local))
print(f"Nominal full-density compliance: {C_nominal:.6f}")

# %%
nel = D0.dofmap.index_map.size_local
volfrac = 0.5
rho_vec = np.full(nel, volfrac)


def oc_update(rho_vec, dC, volfrac, move=0.2):
    """Optimality criteria density update for SIMP, standard 88-line formulation."""
    l1, l2 = 0, 1e9
    while l2 - l1 > 1e-4:
        lmid = 0.5 * (l1 + l2)
        rho_new = np.clip(
            rho_vec * np.sqrt(np.maximum(-dC / lmid, 1e-10)),
            np.maximum(rho_vec - move, 1e-3),
            np.minimum(rho_vec + move, 1.0),
        )
        if rho_new.mean() > volfrac:
            l1 = lmid
        else:
            l2 = lmid
    return rho_new


def adjoint_sensitivity(uh, rho_func):
    """Computes dC/drho_e per element using the self-adjoint compliance identity
    lambda = -u, avoiding a second linear solve. Strain energy is integrated
    exactly per element against DG0 test functions, giving the elementwise
    sensitivity vector directly via UFL/PETSc assembly."""
    w = ufl.TestFunction(D0)
    eps_u = ufl.sym(ufl.grad(uh))

    mu0 = 1.0 / (2 * (1 + nu))
    lmbda0 = nu / ((1 + nu) * (1 - 2 * nu))
    sigma0_u = 2 * mu0 * eps_u + lmbda0 * ufl.tr(eps_u) * ufl.Identity(2)
    strain_energy_density = ufl.inner(sigma0_u, eps_u)

    dE_drho = penal * rho_func ** (penal - 1) * (E0 - Emin)
    sensitivity_form = -dE_drho * strain_energy_density * w * ufl.dx

    dC_form = fem.form(sensitivity_form)
    dC_vec = fem.assemble_vector(dC_form)
    (
        dC_vec.scatter_reverse(dolfinx.la.InsertMode.add)
        if hasattr(dC_vec, "scatter_reverse")
        else None
    )

    cell_volumes = fem.assemble_vector(fem.form(w * ufl.dx)).array
    dC_dense = dC_vec.array / cell_volumes
    return dC_dense


n_iter = 15
compliance_history = []
for it in range(n_iter):
    C, uh = fenicsx_fea(rho_vec, return_field=True)
    dC = adjoint_sensitivity(uh, rho)
    rho_vec = oc_update(rho_vec, dC, volfrac)
    compliance_history.append(C)
    print(f"iter {it}: compliance = {C:.4f}, volfrac = {rho_vec.mean():.3f}")

rho_nominal_optimal = rho_vec.copy()

# %%
C0, uh0 = fenicsx_fea(rho_vec, return_field=True)
dC_adjoint = adjoint_sensitivity(uh0, rho)

check_idx = np.random.choice(nel, size=10, replace=False)
fd_eps = 1e-6
max_rel_err = 0.0
for i in check_idx:
    rho_pert = rho_vec.copy()
    rho_pert[i] += fd_eps
    C_pert = fenicsx_fea(rho_pert)
    dC_fd = (C_pert - C0) / fd_eps
    rel_err = abs(dC_fd - dC_adjoint[i]) / (abs(dC_fd) + 1e-12)
    max_rel_err = max(max_rel_err, rel_err)
    print(f"elem {i}: adjoint={dC_adjoint[i]:.6e}, FD={dC_fd:.6e}, rel_err={rel_err:.2e}")

print(f"\nMax relative error: {max_rel_err:.2e}  (gate: must be < 1e-5)")
print("PASS" if max_rel_err < 1e-5 else "FAIL: check adjoint derivation/UFL form")

# %%
NKL = 5


def perturb_density(xi, rho_base, nelx, nely):
    """Spatially smooth perturbation of the optimized density field, standing in
    for the KL-expansion of geometric manufacturing error in masterContext."""
    xx, yy = np.meshgrid(np.arange(nelx), np.arange(nely))
    modes = [
        np.sin(np.pi * xx / nelx),
        np.cos(np.pi * yy / nely),
        np.sin(2 * np.pi * xx / nelx),
        np.cos(2 * np.pi * yy / nely),
        np.sin(np.pi * xx / nelx) * np.cos(np.pi * yy / nely),
    ]
    field = sum(c * m for c, m in zip(xi, modes)).flatten()
    return np.clip(rho_base + 0.1 * field, 1e-3, 1.0)


def model_fea(xi_array):
    xi = np.asarray(xi_array)
    rho_pert = perturb_density(xi, rho_nominal_optimal, nelx, nely)
    return [fenicsx_fea(rho_pert)]


distribution = ot.JointDistribution([ot.Normal(0, 1)] * NKL)
g = ot.PythonFunction(NKL, 1, model_fea)

Ntrain = 120
xi_train = ot.LHSExperiment(distribution, Ntrain).generate()
C_train = g(xi_train)
print(
    f"Generated {Ntrain} FEniCSx training samples, mean C = {float(C_train.computeMean()[0]):.4f}"
)

# %%
selectionAlgorithm = ot.LeastSquaresMetaModelSelectionFactory(ot.LARS(), ot.CorrectedLeaveOneOut())
projectionStrategy = ot.LeastSquaresStrategy(xi_train, C_train, selectionAlgorithm)

enumerateFunction = ot.LinearEnumerateFunction(NKL)
multivariateBasis = ot.OrthogonalProductPolynomialFactory(
    [ot.HermiteFactory()] * NKL, enumerateFunction
)
max_degree = 4
totalSize = enumerateFunction.getStrataCumulatedCardinal(max_degree)
adaptiveStrategy = ot.FixedStrategy(multivariateBasis, totalSize)

chaosAlgo = ot.FunctionalChaosAlgorithm(
    xi_train, C_train, distribution, adaptiveStrategy, projectionStrategy
)
chaosAlgo.run()
pce_result = chaosAlgo.getResult()
pce_model = pce_result.getMetaModel()
print("PCE surrogate trained on FEniCSx compliance data.")

# %%
Ntest = 50
xi_test = distribution.getSample(Ntest)
C_fea_test = g(xi_test)
C_pce_test = pce_model(xi_test)

validation = ot.MetaModelValidation(C_fea_test, C_pce_test)
Q2 = validation.computeR2Score()[0]
residuals = np.array(C_fea_test) - np.array(C_pce_test)
rmse = np.sqrt(np.mean(residuals**2))
rel_error = np.abs(residuals) / np.abs(np.array(C_fea_test))

print(f"Q2 score: {Q2:.4f}  (gate: must be >= 0.99 before deployment in robust TO loop)")
print(f"RMSE: {rmse:.6f}, Max rel. error: {rel_error.max():.4%}")
print("PASS" if Q2 >= 0.99 else "FAIL: increase max_degree or Ntrain")

# %%
fea_vals = np.array(C_fea_test).flatten()
pce_vals = np.array(C_pce_test).flatten()

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(fea_vals, pce_vals, alpha=0.7, label="Held-out samples")
lims = [min(fea_vals.min(), pce_vals.min()), max(fea_vals.max(), pce_vals.max())]
ax.plot(lims, lims, "r--", label="Perfect agreement")
ax.set_xlabel("FEniCSx compliance (ground truth)")
ax.set_ylabel("PCE surrogate compliance")
ax.set_title(f"PCE vs FEniCSx FEA validation (Q2 = {Q2:.4f})")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("pce_vs_fenicsx_validation.png", dpi=150)
plt.show()
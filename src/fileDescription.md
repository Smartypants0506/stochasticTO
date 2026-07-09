# NIST Stochastic Topology Optimization — Complete File Descriptions & End-to-End Guide

---

## `src/` Directory Overview

The project source tree is organized into ten functional packages, each mapping directly to a pipeline stage. Every package below is described with its exact data inputs, internal transformations, and outputs. GPU acceleration via the NVIDIA DGX (4× A100 80 GB) is called out where parallelism is exploitable.

---

## `src/config/` — Configuration Layer

This package is the **single source of truth** for every run; all downstream modules receive a `Config` object rather than raw files.

### `loader.py`
- **In:** Path to a `config.yaml` or `config.json` file on disk containing keys for: CAD file path, metrology dataset paths, material properties (`E`, `nu`), load cases, boundary condition specifications, volume fraction `V_frac`, manufacturing process type (`CNC` | `FDM`), robust tradeoff parameter `λ`, KL truncation order `N_KL`, PCE polynomial degree, MMA solver settings, output directory, and GPU/parallelism flags
- **What happens:** Reads YAML/JSON with PyYAML/`json`, validates keys against the schema, resolves relative paths to absolute, and instantiates a `Config` dataclass
- **Out:** A fully validated, in-memory `Config` object passed by reference to all other modules

### `schema.py`
- **In:** Raw parsed dictionary from `loader.py`
- **What happens:** Enforces required field presence, type coercion (e.g., floats for `E`, integers for `N_KL`), range checks (e.g., `0 < V_frac < 1`), and raises structured `ValidationError` exceptions with field-level messages
- **Out:** Validated dictionary, or raised exception halting the run

### `structures.py`
- **In:** Validated dictionary
- **What happens:** Defines typed dataclasses `Config`, `MaterialProps`, `LoadCase`, `BoundaryCondition`, `SolverSettings`, `GPUSettings` using Python `dataclasses` or Pydantic; performs unit consistency checks (all SI)
- **Out:** Exported dataclass types used as type hints throughout the codebase

---

## `src/meshing/` — CAD Import & FE Mesh Generation

This package converts a raw STEP file into a simulation-ready mesh.

### `importer.py`
- **In:** STEP/IGES/STL file path from `Config`; optional geometry repair tolerance
- **What happens:** Uses the `OCC` (OpenCASCADE) kernel via Gmsh's Python API (`gmsh.model.occ.importShapes`) to load the CAD solid; performs geometry healing (merge coincident vertices, fix degenerate edges); extracts physical surface tags that will later anchor boundary conditions
- **Out:** Gmsh model object with loaded and healed CAD geometry; dictionary mapping CAD surface names/tags to physical group IDs

### `mesher.py`
- **In:** Gmsh model from `importer.py`; meshing parameters (`lc` characteristic element length, refinement region bounding boxes, element order)
- **What happens:** Calls `gmsh.model.mesh.generate(3)` for volumetric meshing (tetrahedral), applies size fields (`gmsh.model.mesh.field`) for local refinement near stress concentrators or design-variable boundaries; checks mesh quality metrics (aspect ratio, Jacobian determinant); exports to `.msh` format; converts to XDMF/HDF5 format compatible with FEniCSx via `meshio`
- **Out:** `dolfinx.mesh.Mesh` object; `MeshTags` for cell and facet regions; element-volume array `vol_e` (shape `[N_elements]`)

### `mapper.py`
- **In:** `dolfinx.mesh.Mesh`; CAD surface-to-physical-group dictionary
- **What happens:** Builds a Python dictionary mapping human-readable BC names (e.g., `"fixed_face"`, `"load_face"`) to FEniCSx `MeshTags` integer markers; constructs a surface-to-volume proximity mapping used later by the random field perturbation module
- **Out:** `bc_marker_dict: dict[str, int]`; `surface_node_map: np.ndarray`

---

## `src/fea/` — Deterministic Linear Elasticity FEA

This package is the physics engine. All FEA solves run on the DGX via PETSc's CUDA backend — FEniCSx/dolfinx passes `PETSc.Options` to enable GPU-accelerated sparse solvers (`-pc_type gamg -ksp_type cg` on CUDA).

### `assembler.py`
- **In:** `dolfinx.mesh.Mesh`; material properties `E` (Young's modulus, Pa), `ν` (Poisson's ratio); density field `ρ_e: np.ndarray` shape `[N_elements]`; SIMP penalty exponent `p` (default 3)
- **What happens:** Defines the UFL variational form for linear elasticity:

$$\mathbf{a}(\mathbf{u}, \mathbf{v}) = \int_\Omega \rho_e^p \, \boldsymbol{\sigma}(\mathbf{u}) : \boldsymbol{\varepsilon}(\mathbf{v}) \, d\Omega$$

  Assembles global stiffness matrix `K` (PETSc `Mat`, sparse CSR) and load vector `F` (PETSc `Vec`); applies SIMP penalization elementwise via a DG0 coefficient function; tags assembled matrix with CUDA device context for GPU solve
- **Out:** PETSc sparse stiffness matrix `K`; PETSc load vector `F`

### `boundary.py`
- **In:** FEniCSx `FunctionSpace`; `bc_marker_dict` from `mapper.py`; load case specification (face tag, force vector `[Fx, Fy, Fz]` N); Dirichlet BC specification (face tag, displacement value)
- **What happens:** Constructs `dolfinx.fem.DirichletBC` objects for zero-displacement constraints; builds Neumann BC surface integrals (`ds` measure) appended to the load vector `F`
- **Out:** List of `DirichletBC` objects; assembled Neumann contribution added to `F`

### `solver.py`
- **In:** Assembled `K`, `F`; list of `DirichletBC` objects; PETSc solver options (from `Config.SolverSettings`)
- **What happens:** Applies Dirichlet BCs via `dolfinx.fem.petsc.apply_lifting`; solves `KU = F` using PETSc CG + AMG preconditioner (GAMG) with CUDA backend on A100; monitors convergence residual; on the DGX, solver is launched across up to 4 MPI ranks (one per GPU) for large meshes via `mpirun -n 4`
- **Out:** Displacement field `u: dolfinx.fem.Function` (shape `[N_dofs × 3]`); solver convergence residual `r_norm: float`

### `postprocess.py`
- **In:** Displacement field `u`; material properties; density field `ρ_e`
- **What happens:** Computes Cauchy stress tensor `σ = C : ε(u)` via UFL projection; computes elementwise strain energy density `s_e = ½ σ : ε`; computes total compliance `C = F^T U`; assembles `ResultsBundle`
- **Out:** `stress_field: dolfinx.fem.Function`; `strain_energy_e: np.ndarray [N_elements]`; scalar `compliance: float`; `ResultsBundle` dataclass

---

## `src/topology/` — Nominal SIMP Topology Optimization

### `filters.py`
- **In:** Raw sensitivity array `dc_drho: np.ndarray [N_elements]`; density field `ρ_e`; filter radius `r_min` (m); mesh connectivity/neighbor list
- **What happens:** Applies density filter (weighted averaging of densities over a radius) to regularize the design field and prevent checkerboarding; applies Heaviside projection

$$\tilde{\rho}_e = \frac{\tanh(\beta \eta) + \tanh(\beta(\rho_e - \eta))}{\tanh(\beta \eta) + \tanh(\beta(1-\eta))}$$

  with continuation on `β`; chains filter Jacobians for adjoint-consistent sensitivity correction
- **Out:** Filtered/projected density field `ρ_tilde: np.ndarray`; filtered sensitivities `dc_drho_filtered: np.ndarray`

### `mma_simp.py`
- **In:** Filtered sensitivities `dc_drho_filtered [N_elements]`; current density field `ρ_e [N_elements]`; volume fraction `V_frac`; element volumes `vol_e [N_elements]`; previous two iterates `ρ_e^{k-1}`, `ρ_e^{k-2}` (for asymptote initialization on first call, set both equal to `ρ_e`); MMA hyperparameters `(asyinit, asydecr, asyincr, move)` from `Config.SolverSettings`
- **What happens:** Wraps `pyOptSparse` with the `ParOpt` MMA driver; at each TO iteration constructs the MMA convex separable subproblem

$$\min_{\rho} \sum_{e=1}^{N} \left( \frac{p_e}{\mathcal{U}_e - \rho_e} + \frac{q_e}{\rho_e - \mathcal{L}_e} \right) \quad \text{s.t.} \quad \sum_e v_e \rho_e \leq V_{\text{frac}} \sum_e v_e, \quad \rho_{\min} \leq \rho_e \leq 1$$

  where the moving asymptotes `𝒰_e` and `ℒ_e` are updated each iteration based on oscillation detection in the design variable history; computes `p_e` and `q_e` approximation coefficients from the filtered sensitivity and current asymptote positions; solves the subproblem (dual decomposition, one Lagrange multiplier for the single volume constraint — analytically solvable in O(N)); updates asymptotes for the next call; checks KKT residual `||∇L||_∞ < tol` as the convergence criterion
- **Out:** Updated density field `ρ_e_new [N_elements]`; current volume fraction `V_current: float`; KKT residual `kkt: float`; asymptote arrays `U_e`, `L_e` (passed back in as state on the next iteration)

---

## `src/metrology/` — Metrology Data Ingestion & Preprocessing

This package converts raw scan data into deviation fields that parameterize the manufacturing uncertainty model.

### `ingestion.py`
- **In:** Paths to CMM `.csv` files (columns: `x, y, z, dx, dy, dz`) or laser-scan point cloud files (`.ply`, `.xyz`); process type flag (`CNC` | `FDM`)
- **What happens:** Reads point clouds into NumPy arrays; performs outlier removal (IQR-based); normalizes deviation units to meters; stores raw data in a `MetrologyDataset` object alongside provenance metadata (machine ID, date, operator)
- **Out:** `MetrologyDataset`: arrays `points [N_pts × 3]`, `deviations [N_pts × 3]`; metadata dict

### `registration.py`
- **In:** `MetrologyDataset`; nominal mesh node coordinates `X_nom [N_nodes × 3]`
- **What happens:** Runs Iterative Closest Point (ICP) algorithm (via `open3d` or `scipy.spatial`) to rigidly align the measured point cloud to the nominal CAD surface; computes residual deviation field after alignment; performs best-fit datum alignment per ASME Y14.5 conventions
- **Out:** Transformation matrix `T_icp [4×4]`; aligned point cloud; residual deviation field `δ_field [N_pts × 3]`

### `deviation.py`
- **In:** Aligned deviation field from `registration.py`; nominal mesh
- **What happens:** Projects scattered deviation vectors onto the FE mesh nodes via radial basis function (RBF) interpolation (`scipy.interpolate.RBFInterpolator`); produces a continuous scalar deviation field (signed distance from nominal surface) defined at every mesh node
- **Out:** Nodal deviation field `delta_nodal [N_nodes]`; deviation statistics (mean, std, max per surface region)

### `process_stats.py`
- **In:** Multiple `MetrologyDataset` objects (replicate parts across machines/operators)
- **What happens:** Computes SPC statistics: `X̄/R` charts per feature; calculates `Cp = (USL-LSL)/(6σ)` and `Cpk = min((USL-μ)/(3σ), (μ-LSL)/(3σ))` for each key dimension; checks for statistical control (Western Electric rules); exports control charts via Matplotlib
- **Out:** `ProcessCapabilityReport`: dict of `{dimension: {Cp, Cpk, mean, std}}`; control chart figures; `sigma_process: float` (overall process standard deviation used to scale the random field variance)

---

## `src/random_fields/` — Covariance Kernel & KL Expansion

This package builds the mathematical model of spatially correlated manufacturing error.

### `kernel.py`
- **In:** Nodal deviation fields from `deviation.py` (multiple parts = multiple realizations); sampling node coordinates; process type to select kernel family
- **What happens:** Constructs empirical variogram `γ(h) = ½ Var[δ(x) - δ(x+h)]` from deviation pairs; fits squared-exponential or Matérn-5/2 kernel

$$k(x, x') = \sigma^2 \exp\!\left(-\frac{\|x-x'\|^2}{2\ell^2}\right)$$

  via maximum likelihood (scipy `minimize` on the negative log-likelihood); validates fit with leave-one-out cross-validation; FDM parts get an anisotropic Matérn kernel with separate in-plane and out-of-plane length scales
- **Out:** Fitted `KernelModel` object: `{kernel_type, sigma^2, ell, ell_z (FDM)}`; variogram plot; validation RMSE

### `kl_expansion.py`
- **In:** `KernelModel`; FE mesh (node coordinates); truncation order `N_KL` (from `Config`)
- **What happens:** Instantiates `openturns.KarhunenLoeveP1Algorithm` with the fitted kernel; solves the Fredholm integral eigenvalue problem

$$C(x, x') \phi_i(x') = \lambda_i \phi_i(x)$$

  via discretized eigendecomposition; truncates to `N_KL` modes capturing ≥ 95% of total variance; stores `λ_i`, `φ_i(x)` for all modes
- **Out:** `KLModel`: eigenpairs `{lambda_i: float, phi_i: np.ndarray [N_nodes]}` for `i = 1..N_KL`; explained variance ratio per mode; KL mode visualization VTK files

### `perturbation.py`
- **In:** Nominal mesh node coordinates; `KLModel`; a single sample of KL coefficients `ξ [N_KL]`
- **What happens:** Computes the geometric error field as

$$\delta(x, \xi) = \sum_{i=1}^{N_{KL}} \xi_i \sqrt{\lambda_i} \, \phi_i(x)$$

  Applies displacement to mesh node positions; calls `mesher.py` to remesh the perturbed geometry while preserving BCs; checks mesh quality post-perturbation and triggers local refinement if Jacobian drops below threshold
- **Out:** Perturbed `dolfinx.mesh.Mesh`; updated `MeshTags` with consistent BC markers; perturbation magnitude map (for diagnostics)

---

## `src/sampling/` — Experimental Design for Surrogate Training

### `sampler.py`
- **In:** KL coefficient probability laws (standard normal `N(0,1)` for Gaussian inputs); truncation dimension `N_KL`; total sample count `N_train + N_test`; sampling strategy (`LHS` | `sparse_grid` | `MonteCarlo`) from `Config`
- **What happens:** For LHS: uses `openturns.LHSExperiment` with Monte Carlo optimization for space-filling; for sparse grid: uses `openturns.SparseGrid` (Smolyak) appropriate for moderate `N_KL ≤ 20`; for MC: `openturns.MonteCarloExperiment`; generates full sample matrix `Ξ [N_samples × N_KL]`
- **Out:** `Ξ_train [N_train × N_KL]`, `Ξ_test [N_test × N_KL]` as NumPy arrays; sample weights if quadrature-based

### `splitter.py`
- **In:** Full sample matrix `Ξ`; train/test split ratio (default 80/20); optional stratification flag
- **What happens:** Shuffles samples with a fixed random seed (from `Config` for reproducibility); splits into training and held-out test sets; logs split indices to the results directory
- **Out:** `Ξ_train`, `Ξ_test`; `split_indices.json`

---

## `src/surrogate/` — PCE Surrogate Construction

This package converts `N_train` FEA evaluations into a fast analytic surrogate for compliance. The FEA-at-samples loop is the primary GPU bottleneck and is parallelized across all 4 A100s via MPI.

### `fea_at_samples.py`
- **In:** `Ξ_train [N_train × N_KL]`; nominal density field `ρ_e`; nominal mesh; `KLModel`; `Config` (material props, BCs)
- **What happens:** For each training sample `ξ^(j)`: calls `perturbation.py` → `assembler.py` → `solver.py` → `postprocess.py`; extracts scalar compliance `C^(j)`; the loop is parallelized: `mpirun -n 4` assigns sample batches to each GPU (A100), with each rank running its own FEniCSx instance via CUDA PETSc; collects results via `mpi4py.gather`; estimated throughput ~150–300 solves/hour per A100 for medium meshes
- **Out:** `C_train [N_train]` (compliance values at training points); optionally `sigma_vm_train [N_train × N_elements]`; sample-level perturbed mesh files (VTK, for later Monte Carlo visualization)

### `pce_builder.py`
- **In:** `Ξ_train [N_train × N_KL]`; `C_train [N_train]`; polynomial basis type (Hermite for Gaussian inputs); maximum total degree `p_max`; hyperbolic truncation `q` (default 0.75)
- **What happens:** Constructs the multi-index set for sparse PCE using hyperbolic truncation

$$\mathcal{A}^{p,q} = \left\{ \alpha \in \mathbb{N}^{N_{KL}} : \|\alpha\|_q \leq p \right\}$$

  Instantiates `openturns.FunctionalChaosAlgorithm` with LARS (Least Angle Regression) for sparse coefficient estimation; cross-validates with `Ξ_test` computing predictive $Q^2 = 1 - \text{RMSE}_{\text{test}}^2 / \text{Var}(C_{\text{test}})$; iterates on `p_max` until `Q² ≥ 0.99`
- **Out:** Fitted `openturns.FunctionalChaosResult`; PCE coefficients `c_α` array; `Q²` metric; serialized `pce_model.pkl`

### `pce_model.py`
- **In:** `FunctionalChaosResult` from `pce_builder.py`
- **What happens:** Wraps the OpenTURNS result in a lightweight `PCEModel` class with methods: `predict(Ξ)` → compliance values; `mean()` → `c_0`; `variance()` → `∑_{α≠0} c_α²`; `gradient_wrt_design(ρ_e)` → chain-rule gradient for the optimizer
- **Out:** `PCEModel` object used by `robust_objective.py` and `robust_gradient.py`

### `sobol.py`
- **In:** `FunctionalChaosResult`; KL mode labels
- **What happens:** Computes first-order Sobol indices $S_i = V_i / V$ and total Sobol indices $S_i^T$ analytically from PCE coefficients using `openturns.FunctionalChaosSobolIndices`; produces ranked bar chart of top contributing KL modes; identifies minimum `N_KL` that accounts for 99% of output variance
- **Out:** `SobolReport`: `{S_i, S_i_total}` for `i = 1..N_KL`; Matplotlib bar chart PNG; recommended `N_KL_effective` for possible truncation refinement

---

## `src/optimization/` — Robust Topology Optimization Loop

This is the core novel contribution of the NIST framework.

### `robust_objective.py`
- **In:** Current density field `ρ_e [N_elements]`; `PCEModel`; robust tradeoff parameter `λ`; volume fraction `V_frac`
- **What happens:** Evaluates the robust objective

$$J(\rho, \lambda) = \underbrace{\mu_C(\rho)}_{\text{mean compliance}} + \lambda \underbrace{\sigma_C(\rho)}_{\text{std of compliance}}$$

  where `μ_C = c_0` and `σ_C = (∑_{α≠0} c_α²)^{0.5}` are extracted analytically from PCE; evaluates volume constraint `g = V(ρ) - V_frac`; no additional FEA solve needed per evaluation
- **Out:** Scalar `J: float`; scalar `g: float`

### `robust_gradient.py`
- **In:** `PCEModel`; adjoint sensitivities `∂C/∂ρ_e [N_elements]` from `postprocess.py`; current density field
- **What happens:** Chains the PCE gradient through the adjoint:

$$\frac{\partial J}{\partial \rho_e} = \frac{\partial \mu_C}{\partial \rho_e} + \lambda \frac{\partial \sigma_C}{\partial \rho_e}$$

  Uses `PCEModel.gradient_wrt_design()` which internally differentiates the PCE w.r.t. each density variable; passes gradients through `filters.py` (chain rule) for consistency with the filtered/projected density; optionally validates with finite-difference perturbation check
- **Out:** `dJ_drho [N_elements]`; `dg_drho [N_elements]`

### `mma_driver.py`
- **In:** Objective `J`, gradient `dJ_drho`, constraint `g`, constraint gradient `dg_drho`, current `ρ_e`, bounds `[ρ_min, 1]`, previous two iterates `ρ_e^{k-1}` and `ρ_e^{k-2}` (for asymptote update), MMA hyperparameters from `Config`
- **What happens:** Wraps `pyOptSparse` with `ParOpt` solver configured for MMA; constructs and solves the convex MMA subproblem at each iteration

$$\min_{\rho} \sum_e \left(\frac{p_e}{\mathcal{U}_e - \rho_e} + \frac{q_e}{\rho_e - \mathcal{L}_e}\right)$$

  where `𝒰_e` and `ℒ_e` are the moving asymptotes updated each iteration; checks KKT residual `||∇L||_∞ < tol` for convergence
- **Out:** Updated density field `ρ_e_new [N_elements]`; convergence flag; KKT residual; iteration log CSV

### `orchestrator.py`
- **In:** `Config`; initial density field (from nominal SIMP result); `PCEModel`; `KLModel`; `mma_driver`, `robust_objective`, `robust_gradient` modules
- **What happens:** Runs the outer robust TO loop: density update → FEA solve → PCE gradient → MMA step → filter → convergence check; optionally sweeps over a list of `λ` values to build the Pareto front; logs every iteration to `results/iterations/`; saves checkpoint `ρ_e` arrays every 10 iterations for restart capability
- **Out:** `DesignState`: converged `ρ_e_robust [N_elements]` per `λ`; Pareto front data `[(μ_C, σ_C)]`; full iteration history CSV; checkpoint files

---

## `src/validation/` — Monte Carlo Ground-Truth Validation

### `monte_carlo.py`
- **In:** Converged robust design `ρ_e_robust`; `KLModel`; number of MC samples `N_mc` (default 5,000); `Config` (BCs, material)
- **What happens:** Generates `N_mc` KL coefficient samples from `openturns.MonteCarloExperiment`; for each sample runs `perturbation.py` + `fea/solver.py` + `fea/postprocess.py`; parallelized across 4 A100s via MPI (each GPU handles `N_mc / 4 = 1,250` solves); aggregates `C_mc [N_mc]`; computes empirical mean, variance, 5th/95th percentile, full CDF
- **Out:** `MCReport`: `{mean_C, var_C, p05_C, p95_C, C_mc_array}`; CDF plot PNG; comparison table of PCE predictions vs MC ground truth

### `comparator.py`
- **In:** `MCReport`; `PCEModel` predictions for the same `N_mc` samples
- **What happens:** Computes PCE vs MC validation metrics: RMSE, relative error on mean/variance, Q² on the full sample set; flags any PCE underprediction of tail quantiles (safety-critical for manufacturing); generates scatter plot of PCE vs FEA compliance
- **Out:** `ValidationReport`: `{Q2, RMSE, mean_error_pct, var_error_pct}`; scatter plot PNG; pass/fail flag for Q² ≥ 0.99

---

## `src/viz/` — Visualization & CAVE XR Pipeline

### `ensemble_generator.py`
- **In:** `ρ_e_robust`; `KLModel`; `N_vis` sample KL coefficients (default 500–5,000 for CAVE); `Config`
- **What happens:** Generates `N_vis` perturbed VTK meshes with scalar fields (compliance, coefficient of variation per element, Von Mises stress); writes each as `ensemble/sample_{i:04d}.vtu`; assembles a `ParaView` `.pvd` collection file; on the DGX, uses all 4 GPUs for batch mesh generation; computes per-element occurrence probability for opacity mapping
- **Out:** Directory `results/ensemble/` with `N_vis` `.vtu` files; `ensemble.pvd` collection; `probability_weights.npy [N_vis]`

### `probability_cloud.py`
- **In:** `ensemble.pvd`; `probability_weights.npy`; CAVE/display configuration (resolution, stereo mode, display dimensions)
- **What happens:** Loads ensemble in PyVista; maps opacity of each mesh to `P(sample)` weight — lower-probability (extreme) geometries render more transparent; composites all meshes into a single semi-transparent volume rendering; for CAVE XR export: writes a `ParaView` Python script (`pvbatch`) to drive the immersive display at full wall resolution; leverages ParaView's CAVE mode configuration for the XR wall with head-tracking support; optionally exports CAVE-ready `.pvd` + state file
- **Out:** Local PNG preview; `cave_render.py` script for ParaView CAVE XR; `probability_cloud.vtp` (composited)

### `pareto_plot.py`
- **In:** Pareto front data `[(μ_C, σ_C, λ)]` from `orchestrator.py`; MC validation statistics per design point
- **What happens:** Generates the mean–variance Pareto frontier plot using Matplotlib; annotates each point with its `λ` value; overlays MC-validated mean/variance with PCE predictions (error bars); highlights the recommended operating point; exports as publication-quality PDF and PNG
- **Out:** `pareto_front.pdf`, `pareto_front.png`; `pareto_data.csv`

---

## End-to-End Guide: Nominal STEP → Topologically Optimized STEP + Statistics

#### Stage 1: CAD Import & Meshing
`src/meshing/importer.py` → `mesher.py` → `mapper.py`

**INPUT:** `my_part.step`  

**OUTPUT:** `mesh.xdmf, mesh.h5, bc_markers.json`
→ N_elements (e.g., ~250,000 tets for a 100mm part at 5mm lc)

---

#### Stage 2: Nominal SIMP (Deterministic Baseline)
`src/fea/assembler.py + solver.py + postprocess.py` → `src/topology/filters.py + mma_simp.py`

**INPUT:** mesh.xdmf, material props, BCs, V_frac=0.40

**OUTPUT:**  
rho_nominal.npy — element density field [N_elements]  
compliance_nominal.txt — scalar, e.g., 0.847 J  
nominal_design.vtu — visualization file  
convergence_nominal.csv — compliance, KKT residual vs iteration  

At each iteration the loop is: **FEA solve → adjoint sensitivity → density filter + Heaviside projection → MMA subproblem (`mma_simp.py`) → convergence check on KKT residual**. Using MMA here ensures the nominal warm-start design and the robust loop share the same solver, making Pareto comparisons consistent — there is no algorithmic discontinuity when transitioning from Stage 2 to Stage 5. This run executes on a single GPU; typical convergence is 100–300 MMA iterations.

---

#### Stage 3: Metrology Ingestion & Random Field Fitting
`src/metrology/` → `src/random_fields/kernel.py + kl_expansion.py`  

**INPUT:** cmm_scan_batch1.csv, cmm_scan_batch2.csv
(columns: x, y, z, dx, dy, dz in meters)  

**OUTPUT:** kernel_model.pkl — fitted σ²=8.1e-8 m², ℓ=12.3 mm (example)  
process_capability.csv — Cp/Cpk per feature  
kl_model.pkl — 20 eigenpairs (λ_i, φ_i)  
kl_modes.vtu — first 10 spatial modes for inspection  
`variogram_fit.png`  

---

#### Stage 4: Surrogate Training (PCE)
`src/sampling/sampler.py` → `src/surrogate/fea_at_samples.py` → `pce_builder.py` → `sobol.py`  

**INPUT:** rho_nominal.npy, kl_model.pkl, N_train=400 LHS samples  

**OUTPUT:**  
`C_train.npy` — 400 compliance values  
`pce_model.pkl` — PCE with Q²=0.993  
`sobol_indices.csv` — first-order & total indices per KL mode  
`sobol_bar_chart.png`  
`surrogate_accuracy.png` — PCE vs FEA scatter, test set  

All 400 FEA solves are distributed: 100 per A100. Wallclock ~25–40 minutes on the DGX for medium meshes.

---

#### Stage 5: Robust Optimization (Pareto Sweep)
`src/optimization/orchestrator.py` (calling `robust_objective.py`, `robust_gradient.py`, `mma_driver.py`)

For each `λ ∈ {0.0, 0.5, 1.0, 2.0, 5.0}`:  

**INPUT:** `rho_nominal.npy` (warm start), `pce_model.pkl`, `kl_model.pkl`  

**OUTPUT:**  
`rho_robust_lambda_{λ}.npy` — converged robust density per λ  
`iteration_log_lambda_{λ}.csv` — J, μ_C, σ_C, V, KKT vs iteration
`robust_design_lambda_{λ}.vtu`  

**COMBINED OUTPUT:**  
`pareto_data.csv` — columns: lambda, mean_C, std_C  
`pareto_front.png / .pdf`


Each `λ` run converges in ~100–300 MMA iterations. PCE evaluation per iteration costs < 1 ms (no FEA needed), so the bottleneck is just MMA subproblem solve.

---

#### Stage 6: Monte Carlo Validation
`src/validation/monte_carlo.py` → `comparator.py`

**INPUT:** `rho_robust_lambda_1.0.npy` (selected design), N_mc=5000  

**OUTPUT:**  
`mc_compliance_samples.npy` — 5000 compliance values  
`mc_report.json` — {mean_C, var_C, p05, p95}  
`compliance_cdf.png`  
`pce_vs_mc_scatter.png`  
`validation_report.json` — {Q2=0.994, RMSE, mean_err_pct}


5,000 FEA solves: 1,250 per A100. Estimated wallclock ~3–5 hours on DGX.

---

#### Stage 7: STEP Export of Optimized Geometry

After the robust density field `rho_robust_lambda_1.0.npy` is converged:

```bash
robust_topo export-step \
  --density rho_robust_lambda_1.0.npy \
  --mesh mesh.xdmf \
  --threshold 0.5 \
  --smooth-iterations 3 \
  --output results/run_001/optimized_part.step
```

Internally this:
1. Thresholds `ρ_e > 0.5` to extract the solid domain (marching cubes via `pyvista.threshold`)
2. Applies Laplacian smoothing (`n=3`) to de-staircase the surface
3. Exports to `.stl` → passes through Gmsh's `gmsh.model.occ` CAD kernel for surface reconstruction and STEP export via `gmsh.write("optimized_part.step")`

**OUTPUT:** `optimized_part.step` — manufacturable, uncertainty-robust topology


---

#### Stage 8: CAVE XR Visualization
`src/viz/ensemble_generator.py` → `probability_cloud.py`

```bash
# On the DGX, generate 5000-geometry probability cloud
mpirun -n 4 python -m src.viz.ensemble_generator \
  --density rho_robust_lambda_1.0.npy \
  --n-vis 5000 \
  --output results/run_001/ensemble/

```
**OUTPUT:** ensemble/ (5000 .vtu files)  
`ensemble.pvd`  
`probability_cloud.vtp` — opacity-mapped composite  


---

### Final Output Summary

| File | Description |
|---|---|
| `optimized_part.step` | Topologically optimized, uncertainty-robust solid geometry |
| `rho_robust_lambda_{λ}.npy` | Element density field for each Pareto point |
| `pareto_data.csv` | `λ, μ_C, σ_C` for all Pareto designs |
| `pareto_front.png` | Mean–variance Pareto frontier plot |
| `mc_report.json` | `mean_C, var_C, p05_C, p95_C` from MC ground truth |
| `compliance_cdf.png` | Empirical CDF of compliance over 5,000 manufactured variants |
| `validation_report.json` | PCE vs MC accuracy: `Q², RMSE, mean_err_pct` |
| `sobol_indices.csv` | Which KL modes (geometric error patterns) drive compliance variance |
| `process_capability.csv` | `Cp, Cpk` per feature from metrology |
| `kl_modes.vtu` | Spatial visualization of top manufacturing error modes |
| `ensemble.pvd` + `cave_render.py` | 5,000-geometry probability cloud for CAVE XR |
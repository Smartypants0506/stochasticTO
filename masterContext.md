# NIST Stochastic Topology Optimization Under Manufacturing Uncertainty
### Master Project Context File — AI-Readable Reference

---

## 1. Problem Statement

Topology optimization produces the most structurally efficient shape possible for a given design space, load cases, and material budget. The fundamental flaw: every resulting design is optimized for a **perfect, exact geometry that no factory can actually build** [file:5]. Thin members, precise geometric relationships, and tight spatial tolerances make topology-optimized parts especially sensitive to manufacturing variation. A drilled hole meant to be at position *x* lands at *x ± δ*; a 3 mm wall comes out at 2.85 mm; a fillet radius drifts across a production run.

**The standards gap this project fills:** No standards body — including NIST — currently provides quantitative guidance on how manufacturing process capability should constrain topology optimization [file:2]. This project builds the **first robust topology optimization framework** grounded in real process metrology data, treating geometric manufacturing variation as a statistically rigorous, spatially correlated first-class design input [file:5].

**Novelty:** Robust topology optimization exists in the literature, but exclusively for uncertainty in loads or material properties — never for spatially correlated **geometric** manufacturing error modeled from real metrology data [file:1].

---

## 2. Solution Architecture — Pipeline Overview

The framework is a six-stage pipeline. Every stage maps to a specific tool from the approved open-source stack [file:1]:

```text
CAD / STEP File
│
▼
┌─────────────────────────────┐
│ Stage 1: Mesh Generation │ ← Gmsh (Python API)
│ CAD import, mesh, BC tags │
└────────────┬────────────────┘
│
▼
┌─────────────────────────────┐
│ Stage 2: Deterministic FEA │ ← FEniCSx (dolfinx / UFL)
│ SIMP Topology Optimization │
│ Nominal optimal design │
└────────────┬────────────────┘
│
▼
┌─────────────────────────────┐
│ Stage 3: Random Field │ ← OpenTURNS
│ Metrology data ingestion │
│ Covariance kernel fitting │
│ KL Expansion (truncated) │
└────────────┬────────────────┘
│
▼
┌─────────────────────────────┐
│ Stage 4: PCE Surrogate │ ← OpenTURNS + scikit-learn
│ Sample KL coefficients │
│ Non-intrusive sparse PCE │
│ Sobol sensitivity indices │
└────────────┬────────────────┘
│
▼
┌─────────────────────────────┐
│ Stage 5: Robust TO Loop │ ← OpenMDAO / PyOptSparse / ParOpt (MMA)
│ Robust objective J=μ+λσ │
│ Adjoint gradients │
│ MMA density update │
│ Pareto front sweep (λ) │
└────────────┬────────────────┘
│
▼
┌─────────────────────────────┐
│ Stage 6: MC Validation & │ ← OpenTURNS + FEniCSx + PyVista + ParaView
│ Visualization │
│ 5,000-sample MC ensemble │
│ Probability cloud CAVE XR │
└─────────────────────────────┘

```


---

## 3. Method of Solution

### 3.1 Deterministic FEA Core — `src/fea/` + `src/topology/`
**Tool:** FEniCSx (dolfinx, UFL variational forms), GPU-accelerated via PETSc CUDA backend

**What it does:**
- Solves linear elasticity PDE in weak/Galerkin form; assembles global stiffness matrix **K** and load vector **F**; solves **KU = F**
- Computes total compliance **C = F^T U** and elementwise strain energy densities
- Implements SIMP topology optimization: stiffness penalization `E(ρ) = E₀ · ρᵖ`, p=3, with design variable ρ ∈ [0,1] per element
- Applies density filter + Heaviside projection to prevent checkerboarding
- Computes adjoint sensitivities ∂C/∂ρₑ (must pass finite-difference verification to relative error < 1e-5)

**Inputs:** CAD mesh, material properties (E, ν), boundary conditions (Dirichlet/Neumann), load cases, volume fraction V_frac, density field ρ_e  
**Outputs:** Displacement field u, compliance C, stress field σ, elementwise strain energy, adjoint sensitivities ∂C/∂ρ_e, nominal optimal density field

**Verification required:** Cantilever beam analytical solution δ = PL³/3EI; mesh convergence study (second-order convergence on log-log plot)

---

### 3.2 Mesh Generation & Geometry Perturbation — `src/meshing/` + `src/random_fields/perturbation.py`
**Tool:** Gmsh (Python API, OpenCASCADE kernel)

**What it does:**
- Imports STEP/IGES/STL files, heals geometry, extracts physical surface tags for boundary condition anchoring
- Generates tetrahedral FE meshes with size fields for local refinement; exports to XDMF/HDF5 for FEniCSx via `meshio`
- For each random field realization: deforms nominal mesh node positions by the KL expansion sample field, maintaining positive mesh Jacobians and consistent BC tags across thousands of perturbed geometries

**Inputs:** STEP/IGES/STL CAD file, meshing parameters (element size, refinement regions), KL coefficient sample ξ, KL eigenmodes φᵢ  
**Outputs:** `dolfinx.mesh.Mesh` object, `MeshTags` for BCs, element volumes `vol_e`, perturbed mesh per random field sample

---

### 3.3 Manufacturing Uncertainty — Random Field Modeling — `src/metrology/` + `src/random_fields/`
**Tool:** OpenTURNS (`KarhunenLoeveP1Algorithm`, covariance kernels)

**What it does:**
- **Metrology ingestion:** Reads CMM point clouds and laser scan deviation fields; registers measured geometry to nominal CAD via ICP; computes Cp/Cpk process capability statistics
- **Kernel fitting:** Fits a stationary covariance kernel (squared-exponential or Matérn) to empirical spatial error fields via maximum likelihood or variogram analysis:
  `k(x, x') = σ² exp(−‖x−x'‖² / 2l²)`
- **KL Expansion:** Decomposes the random field into deterministic spatial eigenmodes φᵢ(x) and uncorrelated scalar random variables ξᵢ:
  `Z(x) = μ(x) + Σᵢ √λᵢ φᵢ(x) ξᵢ`
  Truncation order N_KL chosen so retained modes explain ≥ 95% of total variance; Sobol indices justify truncation

**Inputs:** Raw CMM/laser scan CSVs, nominal CAD geometry, manufacturing process type (CNC | FDM), Cp/Cpk statistics  
**Outputs:** Fitted covariance kernel (σ², l), KL eigenpairs (λᵢ, φᵢ), low-dimensional KL coefficient samples ξ ∈ ℝ^N_KL

**Verification required:** Sample covariance must match theoretical covariance kernel; kernel parameters validated against empirical variogram from metrology data

---

### 3.4 Surrogate Modeling — Polynomial Chaos Expansion & Sobol Indices — `src/surrogate/`
**Tool:** OpenTURNS (`FunctionalChaosAlgorithm`, LARS), scikit-learn (LASSO)

**What it does:**
- **Sampling:** Generates Latin Hypercube (or sparse grid) samples of KL coefficients ξ; runs FEA on each perturbed geometry to collect compliance training data C(ξ)
- **Sparse PCE:** Fits a non-intrusive sparse Polynomial Chaos Expansion using hyperbolic index truncation and LARS regression:
  `C(ξ) ≈ Σ_α c_α Ψ_α(ξ)` (Hermite polynomials for Gaussian ξᵢ)
  Iterates on polynomial degree until Q² ≥ 0.99 on held-out test set
- **Moment extraction (analytic, no sampling):** `μ_C = c_0`, `σ²_C = Σ_{α≠0} c_α²`
- **Sobol indices:** First-order Sᵢ and total Sᵢᵀ computed analytically from PCE coefficients via `openturns.FunctionalChaosSobolIndices`; identifies which KL modes (geometric error patterns) most drive compliance variance

**Inputs:** KL coefficient training samples Ξ_train [N_train × N_KL], FEA compliance values C_train [N_train], polynomial basis type, max degree  
**Outputs:** PCE model (c_α coefficients), analytic μ_C and σ²_C, Q² accuracy metric, Sobol indices {Sᵢ, Sᵢᵀ}, recommended N_KL_effective

**Verification required:** Q² ≥ 0.99 on held-out test set before any deployment in optimization loop

---

### 3.5 Robust Topology Optimization Loop — MMA Driver — `src/optimization/`
**Tool:** PyOptSparse / ParOpt (MMA implementation) via OpenMDAO

**What it does:**
- **Robust objective:** Scalarizes mean and standard deviation of compliance with tradeoff parameter λ:
  `J(ρ, λ) = μ_C(ρ) + λ · σ_C(ρ)`
  μ_C and σ_C extracted analytically from PCE — no additional FEA solve needed per objective evaluation
- **Robust gradient:** Chains PCE gradient through adjoint sensitivities:
  `∂J/∂ρₑ = ∂μ_C/∂ρₑ + λ · ∂σ_C/∂ρₑ`
  Passed through density filter chain for consistency; validated by finite-difference check
- **MMA update:** Solves the convex MMA subproblem with moving asymptotes 𝒰ₑ and ℒₑ; convergence check: KKT residual ‖∇L‖_∞ < tol
- **Pareto sweep:** Runs optimization for multiple λ values to build the mean/variance Pareto frontier; starts from nominal SIMP solution as warm start

**Inputs:** Current density field ρ_e, PCE model, λ, volume constraint V_frac, adjoint sensitivities, MMA hyperparameters, previous two iterates (for asymptote initialization)  
**Outputs:** Converged robust density field ρ_e_robust per λ, Pareto front data [(μ_C, σ_C)], KKT residual history, iteration log CSV

---

### 3.6 Monte Carlo Validation Engine — `src/validation/`
**Tool:** OpenTURNS (`MonteCarloExperiment`) + FEniCSx; parallelized across 4× A100 GPUs via MPI

**What it does:**
- Generates N_mc = 5,000 KL coefficient samples; for each: perturbation → FEA → compliance
- Parallelized: 1,250 solves per GPU via MPI ranks
- Computes empirical compliance distribution: mean, variance, 5th/95th percentiles, full CDF
- Validates PCE surrogate against brute-force MC ground truth: RMSE, relative error on mean/variance, Q² on full sample set; flags PCE underprediction of tail quantiles

**Inputs:** Converged ρ_e_robust, KLModel, N_mc, Config (BCs, material)  
**Outputs:** Empirical C distribution (mean, variance, percentiles), CDF plot PNG, PCE vs MC validation scatter plot, pass/fail flag (Q² ≥ 0.99)

---

## 4. Method of Visualization

**Tools:** PyVista (Python VTK wrapper), ParaView (CAVE XR immersive rendering)

### 4.1 Probability Cloud — `src/viz/probability_cloud.py`
- Generates N_vis (500–5,000) perturbed VTK meshes with scalar fields: compliance, coefficient of variation per element, Von Mises stress
- Maps opacity of each mesh to P(sample): lower-probability (extreme) geometries render more transparent
- **CAVE XR output:** Writes `cave_render.py` script for ParaView CAVE mode with stereo + head-tracking; exports `probability_cloud.vtp` and `ensemble.pvd`
- **Result:** You can visually identify which structural members are stable across the production run and which are fragile — manufacturing uncertainty becomes a geometric phenomenon

### 4.2 Pareto Frontier Plot — `src/viz/pareto_plot.py`
- Plots mean compliance μ_C vs. standard deviation σ_C for each λ value
- Overlays nominal (λ=0) design point; shows the mean/variance trade-off curve

### 4.3 Sobol Bar Chart — `src/surrogate/sobol.py`
- Ranked bar chart of first-order and total Sobol indices per KL mode
- Identifies which geometric error patterns (KL eigenmodes) most drive structural performance variance
- Used to justify KL truncation order N_KL

### 4.4 Density Field Rendering
- TO density field ρ_e rendered in PyVista/ParaView as isosurface (ρ = 0.5 threshold) and volumetric colormap
- Side-by-side comparison: nominal SIMP design vs. robust design

---

## 5. Method of Data Collection

### 5.1 Metrology Datasets (Primary Source — Use Premade Data First)
- **CMM (Coordinate Measuring Machine) scans:** Point clouds of surface deviations on CNC-machined test coupons; provides gold-standard geometric error data with datum alignment
- **Laser / structured-light scans:** Dense full-field point clouds; registered to nominal CAD via ICP algorithm to extract continuous deviation fields
- **Process capability data:** Cp/Cpk statistics for CNC and FDM dimensions from SPC records
- **Priority sources:** NIST existing metrology campaigns; Zenodo/GitHub production datasets for 5-axis CNC milling and powder bed fusion; Bosch CNC machining dataset (GitHub); KIRO LPBF Peregrine datasets

### 5.2 Data Pipeline — `src/metrology/`

| Module | Input | Output |
|---|---|---|
| `ingestion.py` | Raw CSV point clouds, scan files | Cleaned `points [N×3]` and `deviations [N]` arrays |
| `registration.py` | Measured point cloud, nominal CAD | ICP-aligned deviation field on CAD surface |
| `deviation.py` | Aligned scan, nominal mesh | RBF-interpolated deviation field over all mesh nodes |
| `process_stats.py` | Dimensional measurements | Cp, Cpk, σ_process per dimension |

### 5.3 Synthetic Training Data (for PCE)
- KL coefficient samples ξ are generated programmatically by OpenTURNS from fitted distributions
- FEA compliance values C(ξ) are computed by running FEniCSx on each perturbed geometry — this is the primary compute loop
- Latin Hypercube Sampling (LHS) for training set; random Monte Carlo for validation set

### 5.4 New Metrology Experiments (Last Resort Only)
- Only if coverage gaps exist (process or geometry not represented in existing datasets)
- Structured as DoE: full or fractional factorial across machines, operators, process conditions
- Must come from a process in statistical control (SPC X-bar/R charts verified before use)

---

## 6. Approved Tool Stack

| Function | Tool | Custom Code Scope |
|---|---|---|
| FEA (linear elasticity) | **FEniCSx** (dolfinx, UFL) | Problem setup, SIMP loop, adjoint sensitivities |
| Topology optimization (SIMP) | **FEniCSx** + Python | Density/Heaviside filters, sensitivity chaining |
| Random field modeling (KL) | **OpenTURNS** | Geometry ↔ random field mapping |
| Surrogate PCE + Sobol | **OpenTURNS** + scikit-learn | Model orchestration, Q² accuracy checks |
| Robust objective + gradients | **Python (NumPy/SciPy)** | Novel formulation — must be custom glue code |
| Optimization driver (MMA) | **PyOptSparse/ParOpt** via OpenMDAO | Component wrapper, driver config |
| Mesh generation + perturbation | **Gmsh** (Python API) | Geometry ↔ random field deformation logic |
| Monte Carlo engine | **OpenTURNS** + FEniCSx | Loop orchestration, MPI parallelization |
| Visualization | **PyVista** + **ParaView** | Opacity mapping, CAVE XR export |
| DevOps | Git, Docker/Singularity, pytest | Container specs, CI scripts |

---

## 7. Critical Constraints & Rules

### Always Use Premade Tools — NEVER Reimplement
- Use **OpenTURNS** for ALL UQ/PCE/random field operations
- Use **FEniCSx** for ALL FEA — never write custom element assembly
- Use **Gmsh Python API** for ALL meshing
- Use **PyOptSparse/ParOpt MMA** for optimization — never write custom optimizers
- Use **scikit-learn** for LASSO/GP — never write custom regression solvers
- Only write custom code for the novel robust TO formulation and integration glue

### Exact Mathematical Formulations (Do Not Deviate)
- SIMP penalization: `E(ρ) = E₀ · ρᵖ`, p=3
- Compliance: `C = U^T K U`
- Robust objective: `J = μ[C] + λ · σ[C]`
- PCE moments: `μ = c₀`, `σ² = Σ_{α≠0} c_α²`
- Covariance kernel: `k(x,x') = σ² exp(−‖x−x'‖² / 2l²)`
- KL expansion: `Z(x) = μ(x) + Σᵢ √λᵢ φᵢ(x) ξᵢ`

### Verification Gates (Never Skip)
- FEA solver: verify δ = PL³/3EI before use in TO loop
- TO sensitivities: finite-difference check (perturbation 1e-6) on all elements; relative error < 1e-5
- KL expansion: sample covariance must match theoretical kernel
- PCE surrogate: Q² ≥ 0.99 on held-out test set before deployment
- Random field: kernel parameters validated against empirical variogram
- Monte Carlo validation: 500+ full FEA samples on final robust design

### Code Standards
- Python 3.11+, type hints on all function signatures, `from __future__ import annotations`
- NumPy vectorized operations only — no Python loops over mesh elements
- Config-driven: all parameters in YAML (`config.yaml`), never hardcoded
- Logging via `logging` module — never `print()`
- Units explicit in variable names or docstrings (Pa, m, N)
- Conventional commits: `feat:`, `fix:`, `refactor:`, `docs:`, `test:`

### Do NOT
- Use scalar noise for manufacturing error — always spatially correlated random fields
- Modify FEA solver internals (use non-intrusive PCE only)
- Skip any verification step listed above
- Hardcode physical parameters
- Commit notebooks with uncleared outputs

---

## 8. Project Source Tree
```text
src/
├── config/ # loader.py, schema.py, structures.py
├── fea/ # assembler.py, boundary.py, solver.py, postprocess.py
├── meshing/ # importer.py, mesher.py, mapper.py
├── metrology/ # ingestion.py, registration.py, deviation.py, process_stats.py
├── random_fields/ # kernel.py, kl_expansion.py, perturbation.py
├── sampling/ # sampler.py, splitter.py
├── surrogate/ # fea_at_samples.py, pce_builder.py, pce_model.py, sobol.py
├── topology/ # filters.py, optimality_criteria.py
├── optimization/ # robust_objective.py, robust_gradient.py, mma_driver.py, orchestrator.py
├── validation/ # monte_carlo.py, comparator.py
└── viz/ # ensemble_generator.py, probability_cloud.py, pareto_plot.py

tests/
├── unit/ # one file per src module
├── regression/ # numerical regression tests (MBB beam, PCE coefficients)
└── integration/ # full pipeline tests, PCE vs MC consistency

configs/ # YAML problem definitions
data/metrology/ # raw CMM/scan datasets
```


**Entry point:** `python src/main.py --config configs/config.yaml`  
**Tests:** `pytest --cov=src --cov-report=term-missing`  
**Container:** `docker build -t robust-to . && docker run robust-to`

---

## 9. Key References

| Topic | Reference |
|---|---|
| TO classic | Sigmund (2001), 99-line MATLAB code, *Struct Multidisc Optim* 21:120 |
| TO theory | Bendsoe & Sigmund, *Topology Optimization* (Springer, 2003) |
| Robust TO geometry | Lazarov et al. (2012), *IJNME* 90:1321 |
| PCE foundations | Xiu & Karniadakis (2002), *SIAM J Sci Comput* 24:619 |
| Sparse PCE | Blatman & Sudret (2011), *J Comput Phys* 230:2345 |
| PCE + TO | Keshavarzzadeh et al. (2016), *CMAME* 318 |
| MMA optimizer | Svanberg (1987), *IJNME* 24:359 |
| Random fields | Vanmarcke, *Random Fields* (MIT Press, 1983/2010) |
| Sobol sensitivity | Saltelli et al., *Global Sensitivity Analysis* (Wiley, 2008) |
| CNC error modeling | Ramesh et al. (2000), *Int J Machine Tools Manuf* 40:1235 |
| FDM error modeling | Turner et al. (2015), *Rapid Prototyping J* 21:137 |
| NIST standards gap | NIST SP Measurement Science Roadmap for Metal-Based AM |
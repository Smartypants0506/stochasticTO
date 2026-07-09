# NIST Stochastic Topology Optimization Under Manufacturing Uncertainty
### Master Project Context File — AI-Readable Reference (Revision 2: Prebuilt-Software Integration)

---

## 0. Revision Note

This revision replaces four custom/from-scratch code paths in the original architecture with vetted, prebuilt open-source packages, in line with the "Always Use Premade Tools — Never Reimplement" project rule. Custom code is now reserved strictly for the genuinely novel contribution: the spatially-correlated, metrology-calibrated randomized Heaviside projection threshold and its glue logic connecting the swapped-in packages.

| Original Custom Scope | Replaced By | Repository |
|---|---|---|
| FEniCSx SIMP + density filter + Heaviside projection + adjoint sensitivities | **FEniTop** | github.com/missionlab/fenitop |
| Custom MMA optimizer / OpenMDAO+PyOptSparse+ParOpt wiring | **dolfiny MMA (native PETSc TAO algorithm)** | github.com/fenics-dolfiny/dolfiny/blob/main/src/dolfiny/mma.py |
| Custom stochastic robust-TO gradient/orchestration reference | **TOuU (Topology Optimization under Uncertainty)** | github.com/CU-UQ/TOuU |
| Custom ICP point-cloud registration for metrology ingestion | **Open3D colored point cloud registration pipeline** | open3d.org (Colored ICP tutorial) |

---

## 1. Problem Statement

Topology optimization produces the most structurally efficient shape possible for a given design space, load cases, and material budget. The fundamental flaw: every resulting design is optimized for a **perfect, exact geometry that no factory can actually build**. Thin members, precise geometric relationships, and tight spatial tolerances make topology-optimized parts especially sensitive to manufacturing variation. A drilled hole meant to be at position *x* lands at *x ± δ*; a 3 mm wall comes out at 2.85 mm; a fillet radius drifts across a production run.

**The standards gap this project fills:** No standards body — including NIST — currently provides quantitative guidance on how manufacturing process capability should constrain topology optimization. This project builds the **first robust topology optimization framework** grounded in real process metrology data, treating geometric manufacturing variation as a statistically rigorous, spatially correlated first-class design input.

**Novelty:** Robust topology optimization exists in the literature, but predominantly for uncertainty in loads or material properties. This project extends the established robust projection-based TO tradition (Chevens et al., 2011; Amir et al., 2012) — which randomizes the Heaviside projection threshold to model uniform over-/under-etching — to **spatially correlated, metrology-derived geometric manufacturing error**, closing the gap between idealized robust TO theory and real process capability data. With prebuilt FEA/TO/optimizer/registration packages now handling the commodity engineering, essentially all custom engineering effort concentrates on this novel random-field-on-threshold formulation.

---

## 2. Solution Architecture — Pipeline Overview

The framework is a six-stage pipeline. Every stage maps to a specific tool from the approved open-source stack; prebuilt packages now cover Stages 2 and 5 wholesale.

```text
CAD / STEP File
│
▼
┌─────────────────────────────┐
│ Stage 1: Mesh Generation     │ ← Gmsh (Python API)
│ CAD import, mesh, BC tags    │
└────────────┬────────────────┘
│
▼
┌─────────────────────────────┐
│ Stage 2: Deterministic FEA   │ ← FEniTop (FEniCSx-based, prebuilt)
│ SIMP + Helmholtz-PDE filter +│
│ Heaviside projection (η)     │
│ Nominal optimal design       │
└────────────┬────────────────┘
│
▼
┌─────────────────────────────┐
│ Stage 3: Random Field on η   │ ← OpenTURNS
│ Metrology data ingestion     │
│ (Open3D colored ICP)         │
│ Squared-exp covariance fit   │
│ KL Expansion (FEM basis)     │
│ Memoryless marginal transform│
└────────────┬────────────────┘
│
▼
┌─────────────────────────────┐
│ Stage 4: PCE Surrogate       │ ← OpenTURNS + scikit-learn
│ Sample KL coefficients       │
│ Non-intrusive sparse PCE     │
│ Sobol sensitivity indices    │
└────────────┬────────────────┘
│
▼
┌─────────────────────────────┐
│ Stage 5: Robust TO Loop      │ ← dolfiny MMA (PETSc TAO) +
│ Robust objective J=μ+λσ      │   TOuU-informed orchestration
│ Mean-volume constraint E[V]  │
│ Adjoint gradients (FEniTop)  │
│ MMA density update (dolfiny) │
│ Pareto front sweep (λ)       │
└────────────┬────────────────┘
│
▼
┌─────────────────────────────┐
│ Stage 6: MC Validation,      │ ← OpenTURNS + FEniTop + PyVista + ParaView
│ Geometry Perturbation &      │
│ Visualization                │
│ Explicit mesh warping here   │
│ 5,000+ sample MC ensemble    │
│ Probability cloud / CAVE XR  │
└─────────────────────────────┘
```

---

## 3. Method of Solution

### 3.1 Deterministic FEA Core — `src/fea/` + `src/topology/` (now a thin wrapper around FEniTop)
**Tool:** **FEniTop** (github.com/missionlab/fenitop), a prebuilt FEniCSx-based topology optimization package (dolfinx 0.7.3 + petsc4py backend), GPU/MPI-parallel via PETSc

**What it does (now imported, not reimplemented):**
- Solves linear elasticity PDE in weak/Galerkin form; assembles global stiffness matrix **K** and load vector **F**; solves **KU = F** via FEniTop's built-in solver wrappers over PETSc's linear/nonlinear solvers
- Computes total compliance **C = F^T U** and elementwise strain energy densities using FEniTop's native postprocessing utilities
- Implements SIMP topology optimization out of the box: stiffness penalization `E(ρ) = E₀ · ρᵖ`, p=3, with design variable ρ ∈ [0,1] per element
- Applies FEniTop's **Helmholtz-type PDE filter** (replacing a hand-written density filter) **followed by a smooth Heaviside projection with threshold η**: `ρ̃ = Filter(ρ)`, `ρ̂ = Heaviside(ρ̃; η)` — FEniTop's filter is specifically tuned for large filter radii in parallel computing, an improvement over a bespoke filter implementation
- Computes adjoint sensitivities ∂C/∂ρₑ via FEniCSx's automatic differentiation as wired through FEniTop (must still pass finite-difference verification to relative error < 1e-5 — this project-specific verification gate is retained regardless of using prebuilt FEA)
- **Custom glue code retained:** hooking FEniTop's η parameter to the randomized, spatially-varying η(x) field from Stage 3 (FEniTop natively expects a scalar η; this project's core contribution is making it a random field)

**Inputs:** CAD mesh, material properties (E, ν), boundary conditions (Dirichlet/Neumann), load cases, volume fraction V_frac, design variable ρ_e

**Outputs:** Displacement field u, compliance C, stress field σ, elementwise strain energy, adjoint sensitivities ∂C/∂ρ_e, nominal optimal density field ρ̂

**Verification required:** Cantilever beam analytical solution δ = PL³/3EI (FEniTop ships a 2D/3D cantilever beam example script — `scripts/beam_2d.py`, `scripts/beam_3d.py` — usable directly for this check); mesh convergence study (second-order convergence on log-log plot)

**Installation note:** Docker image `jiayingqi/dolfinx-fenitop` bundles FEniCSx + PyVista + Xvfb, satisfying this project's container/DevOps requirement (Section 6) without a custom Dockerfile for this stage.

---

### 3.2 Mesh Generation & Geometry Perturbation (Validation/Visualization Only) — `src/meshing/` + `src/random_fields/perturbation.py`
**Tool:** Gmsh (Python API, OpenCASCADE kernel) — unchanged, no prebuilt replacement adopted here

**What it does:**
- Imports STEP/IGES/STL files, heals geometry, extracts physical surface tags for boundary condition anchoring
- Generates tetrahedral FE meshes with size fields for local refinement; exports to XDMF/HDF5 for FEniTop/FEniCSx via `meshio`
- **Explicit geometry/mesh perturbation is generated only for validation and final visualization ensembles — it is NOT the primary manufacturing-error modeling device.** The primary mechanism is randomization of the projection threshold η (Section 3.3); mesh warping is used downstream in Stage 6 to visualize how η-driven density variation manifests as geometric deviation

**Inputs:** STEP/IGES/STL CAD file, meshing parameters (element size, refinement regions), sampled η realizations for visualization

**Outputs:** `dolfinx.mesh.Mesh` object (feeding FEniTop), `MeshTags` for BCs, element volumes `vol_e`, perturbed mesh per validation/visualization sample

---

### 3.3 Manufacturing Uncertainty — Metrology Registration & Projection-Threshold Random Field Modeling — `src/metrology/` + `src/random_fields/`
**Tools:** **Open3D** (colored point cloud registration, replacing custom ICP) + OpenTURNS (`KarhunenLoeveP1Algorithm`, covariance kernels)

**What it does:**
- **Metrology ingestion:** Reads CMM point clouds and laser scan deviation fields
- **Registration (now prebuilt):** Aligns measured geometry to nominal CAD using **Open3D's colored ICP pipeline** (`open3d.pipelines.registration`), which jointly optimizes geometric and photometric (color/intensity) residuals per Park et al.'s colored-ICP algorithm — replacing a custom ICP implementation and improving alignment robustness over geometry-only ICP when scan color/intensity channels are available; computes Cp/Cpk process capability statistics downstream of registered deviations
- **Primary random field target — projection threshold η(x):** Rather than perturbing mesh nodes directly, manufacturing error is modeled as a spatially varying Heaviside projection threshold η(x), following the robust projection-based TO convention (Chevens et al., 2011). An underlying Gaussian field is discretized via KL expansion, then passed through a **memoryless (isoprobabilistic) transform** to produce a bounded, non-Gaussian marginal for η, calibrated from metrology data rather than assumed uniform. **This remains fully custom — no prebuilt package implements this specific transform.**
- **Kernel fitting:** Fits a **squared-exponential covariance kernel** to the empirical spatial error field (now sourced from Open3D-registered deviations) via maximum likelihood or variogram analysis:
  `k(x, x') = σ² exp(−‖x−x'‖² / 2l²)`
- **KL Expansion (FEM-based):** Decomposes the underlying Gaussian field into deterministic spatial eigenmodes φᵢ(x) and uncorrelated scalar random variables ξᵢ, with eigenfunctions approximated via FEM on a nodal grid whose spacing is proportional to the correlation length l:
  `G(x) = μ(x) + Σᵢ √λᵢ φᵢ(x) ξᵢ`, then `η(x) = T(G(x))` (memoryless marginal transform)
  Truncation order N_KL chosen so retained modes explain ≥ 95% of total variance; Sobol indices justify truncation

**Inputs:** Raw CMM/laser scan point clouds (with color/intensity channels where available for colored ICP), nominal CAD geometry, manufacturing process type (CNC | FDM), Cp/Cpk statistics

**Outputs:** Open3D-registered deviation field on CAD surface, fitted covariance kernel (σ², l), KL eigenpairs (λᵢ, φᵢ), calibrated marginal transform T(·), low-dimensional KL coefficient samples ξ ∈ ℝ^N_KL, sampled η(x) realizations

**Verification required:** Registration fitness/RMSE reported by Open3D's `evaluate_registration` must meet a defined threshold before deviations are used for kernel fitting; sample covariance must match theoretical covariance kernel; kernel parameters validated against empirical variogram from metrology data; document any deviation from purely Gaussian assumptions and justify the chosen marginal transform

---

### 3.4 Surrogate Modeling — Polynomial Chaos Expansion & Sobol Indices — `src/surrogate/`
**Tool:** OpenTURNS (`FunctionalChaosAlgorithm`, LARS), scikit-learn (LASSO) — unchanged

**What it does:**
- **Sampling:** Generates Latin Hypercube (or sparse grid) samples of KL coefficients ξ; runs FEniTop FEA on each η-perturbed density field to collect compliance training data C(ξ)
- **Sparse PCE:** Fits a non-intrusive sparse Polynomial Chaos Expansion using hyperbolic index truncation and LARS regression:
  `C(ξ) ≈ Σ_α c_α Ψ_α(ξ)` (Hermite polynomials for Gaussian ξᵢ)
  Iterates on polynomial degree until Q² ≥ 0.99 on held-out test set
- **Moment extraction (analytic, no sampling):** `μ_C = c_0`, `σ²_C = Σ_{α≠0} c_α²`
- **Sobol indices:** First-order Sᵢ and total Sᵢᵀ computed analytically from PCE coefficients via `openturns.FunctionalChaosSobolIndices`; identifies which KL modes (geometric error patterns) most drive compliance variance

**Inputs:** KL coefficient training samples Ξ_train [N_train × N_KL], FEniTop compliance values C_train [N_train], polynomial basis type, max degree

**Outputs:** PCE model (c_α coefficients), analytic μ_C and σ²_C, Q² accuracy metric, Sobol indices {Sᵢ, Sᵢᵀ}, recommended N_KL_effective

**Verification required:** Q² ≥ 0.99 on held-out test set before any deployment in optimization loop

---

### 3.5 Robust Topology Optimization Loop — dolfiny MMA + TOuU-Informed Orchestration — `src/optimization/`
**Tools:** **dolfiny MMA** (github.com/fenics-dolfiny/dolfiny/blob/main/src/dolfiny/mma.py) as native PETSc TAO algorithm, paired with dolfiny's `taoproblem.py` wrapper for FEniCSx interfacing; orchestration logic informed by **CU-UQ/TOuU** (github.com/CU-UQ/TOuU)

**What it does (replaces the original OpenMDAO/PyOptSparse/ParOpt stack):**
- **MMA driver:** Uses dolfiny's MPI-parallel MMA implementation, which integrates directly as a native `TAO` algorithm inside PETSc — this removes the need for the OpenMDAO/PyOptSparse/ParOpt glue layer originally specified, since dolfiny's `taoproblem.py` wrapper interfaces natively with FEniCSx (and by extension FEniTop) objective/gradient callbacks
- **Robust objective (consistent across structural QoIs):** Scalarizes mean and standard deviation of the relevant structural quantity of interest — compliance, stiffness, heat transfer, or (as a future extension) local stress/reliability metrics — with tradeoff parameter λ:
  `J(ρ, λ) = μ_C(ρ) + λ · σ_C(ρ)`
  μ_C and σ_C extracted analytically from PCE — no additional FEA solve needed per objective evaluation
- **Volume constraint — mean-based:** Expressed in terms of mean occupied volume, `E[V(ρ)] ≤ V_frac`, consistent with stochastic robust TO formulations, passed to dolfiny's TAO problem as a PETSc constraint
- **Orchestration reference:** **TOuU's** stochastic-gradient-based approach to topology optimization under uncertainty (Struct Multidisc Optim, 62(5), 2255-2278) provides an architectural reference for how gradient noise from the stochastic η-field should be handled across MMA iterations; where TOuU's MATLAB reference logic is directly portable, it is translated to Python rather than rederived from first principles
- **Robust gradient:** Chains PCE gradient through FEniTop's adjoint sensitivities:
  `∂J/∂ρₑ = ∂μ_C/∂ρₑ + λ · ∂σ_C/∂ρₑ`
  Passed through FEniTop's Helmholtz filter + Heaviside projection chain for consistency; validated by finite-difference check
- **MMA update:** dolfiny's MMA solves the convex MMA subproblem with moving asymptotes 𝒰ₑ and ℒₑ internally; convergence check: KKT residual ‖∇L‖_∞ < tol, read from the TAO solver's convergence diagnostics
- **Pareto sweep:** Runs the dolfiny-TAO-MMA loop for multiple λ values to build the mean/variance Pareto frontier; starts from nominal FEniTop SIMP solution as warm start

**Custom glue code retained:** the robust objective/gradient scalarization (`robust_objective.py`, `robust_gradient.py`) remains project-specific, since neither dolfiny nor TOuU natively implements the μ+λσ formulation driven by a projection-threshold random field — this is the connective tissue between prebuilt packages and the project's novel contribution

**Inputs:** Current density field ρ_e (from FEniTop), PCE model, λ, mean volume constraint E[V_frac], adjoint sensitivities (FEniTop), dolfiny MMA/TAO hyperparameters, previous two iterates (for asymptote initialization)

**Outputs:** Converged robust density field ρ_e_robust per λ, Pareto front data [(μ_C, σ_C)], KKT residual history (from TAO), iteration log CSV

---

### 3.6 Monte Carlo Validation Engine (Two-Tier UQ) — `src/validation/`
**Tool:** OpenTURNS (`MonteCarloExperiment`) + FEniTop; parallelized across 4× A100 GPUs via MPI

**What it does:**
- Implements the **two-tier uncertainty propagation scheme**: PCE (or SROM) serves as the main engine for evaluating mean/variance *during* optimization; high-fidelity Monte Carlo (thousands of samples) is reserved for *final* verification
- Generates N_mc = 5,000+ η/KL coefficient samples; for each: sample η(x) → apply projection → FEniTop FEA → compliance
- Generates explicit geometry perturbations (mesh deformation) at this stage only, for validation and visualization ensembles
- Parallelized: samples split evenly per GPU via MPI ranks (FEniTop's native MPI/PETSc parallelism is reused here rather than building a separate parallelization layer)
- Computes empirical compliance distribution: mean, variance, 5th/95th percentiles, full CDF
- Validates PCE surrogate against brute-force MC ground truth: RMSE, relative error on mean/variance, Q² on full sample set; flags PCE underprediction of tail quantiles; documents discrepancies and adjusts PCE or random-field modeling as needed

**Inputs:** Converged ρ_e_robust, KLModel, marginal transform T(·), N_mc, Config (BCs, material)

**Outputs:** Empirical C distribution (mean, variance, percentiles), CDF plot PNG, PCE vs MC validation scatter plot, pass/fail flag (Q² ≥ 0.99)

---

## 4. Method of Visualization

**Tools:** PyVista (Python VTK wrapper, bundled with FEniTop's Docker image), ParaView (CAVE XR immersive rendering) — unchanged

### 4.1 Probability Cloud — `src/viz/probability_cloud.py`
- Generates N_vis (500–5,000) perturbed VTK meshes (using Stage 6 explicit geometry perturbation) with scalar fields: compliance, coefficient of variation per element, Von Mises stress
- Maps opacity of each mesh to P(sample): lower-probability (extreme) geometries render more transparent
- **CAVE XR output:** Writes `cave_render.py` script for ParaView CAVE mode with stereo + head-tracking; exports `probability_cloud.vtp` and `ensemble.pvd`
- **Result:** You can visually identify which structural members are stable across the production run and which are fragile — manufacturing uncertainty becomes a geometric phenomenon

### 4.2 Pareto Frontier Plot — `src/viz/pareto_plot.py`
- Plots mean compliance μ_C vs. standard deviation σ_C for each λ value (from the dolfiny-MMA Pareto sweep)
- Overlays nominal (λ=0) design point; shows the mean/variance trade-off curve

### 4.3 Sobol Bar Chart — `src/surrogate/sobol.py`
- Ranked bar chart of first-order and total Sobol indices per KL mode
- Identifies which geometric error patterns (KL eigenmodes) most drive structural performance variance; used to justify KL truncation order N_KL and flag dominant spatial regions for metrology follow-up

### 4.4 Reliability Maps and Density Field Rendering
- TO density field ρ_e (from FEniTop) rendered in PyVista/ParaView as isosurface (ρ = 0.5 threshold) and volumetric colormap
- Generate reliability maps highlighting regions of high sensitivity to manufacturing deviations, feeding back into design guidelines and potential standards language
- Side-by-side comparison: nominal FEniTop SIMP design vs. robust design

---

## 5. Method of Data Collection

### 5.1 Metrology Datasets (Primary Source — Use Premade Data First)
- **CMM (Coordinate Measuring Machine) scans:** Point clouds of surface deviations on CNC-machined test coupons; provides gold-standard geometric error data with datum alignment, now processed via Open3D
- **Laser / structured-light scans:** Dense full-field point clouds, ideally with color/intensity channels to leverage Open3D's colored ICP; registered to nominal CAD via Open3D to extract continuous deviation fields
- **Process capability data:** Cp/Cpk statistics for CNC and FDM dimensions from SPC records
- **Priority sources:** NIST existing metrology campaigns; Zenodo/GitHub production datasets for 5-axis CNC milling and powder bed fusion; Bosch CNC machining dataset (GitHub); KIRO LPBF Peregrine datasets

### 5.2 Data Pipeline — `src/metrology/`

| Module | Input | Output |
|---|---|---|
| `ingestion.py` | Raw CSV/PLY point clouds, scan files | Cleaned `points [N×3]`, `colors [N×3]` (if available), and `deviations [N]` arrays |
| `registration.py` | Measured point cloud, nominal CAD | **Open3D colored-ICP-aligned** deviation field on CAD surface (thin wrapper around `open3d.pipelines.registration`) |
| `deviation.py` | Open3D-aligned scan, nominal mesh | RBF-interpolated deviation field over all mesh nodes; used to calibrate η marginal/covariance |
| `process_stats.py` | Dimensional measurements | Cp, Cpk, σ_process per dimension |

### 5.3 Synthetic Training Data (for PCE)
- KL coefficient samples ξ are generated programmatically by OpenTURNS from fitted distributions on the underlying Gaussian field
- FEA compliance values C(ξ) are computed by running FEniTop on each η-projected density field — this is the primary compute loop
- Latin Hypercube Sampling (LHS) for training set; random Monte Carlo for validation set

### 5.4 New Metrology Experiments (Last Resort Only)
- Only if coverage gaps exist (process or geometry not represented in existing datasets)
- Structured as DoE: full or fractional factorial across machines, operators, process conditions; design experiments (DoE) to cover the KLE coefficient space and key design variables (e.g., use Latin Hypercube sampling and cross-validation for accuracy matching the master-context requirements)
- Must come from a process in statistical control (SPC X-bar/R charts verified before use)

---

## 6. Approved Tool Stack (Revised)

| Function | Tool | Custom Code Scope |
|---|---|---|
| FEA (linear elasticity) + SIMP + Heaviside projection | **FEniTop** (missionlab/fenitop, FEniCSx-based, prebuilt) | Config/problem setup, wiring randomized η(x) field into FEniTop's scalar-η interface |
| Topology optimization filter | **FEniTop's Helmholtz-type PDE filter** (prebuilt) | None — used as-is |
| Random field modeling (KL on η) | **OpenTURNS** | Projection-threshold ↔ random field mapping, marginal transform (fully custom, core novelty) |
| Surrogate PCE + Sobol | **OpenTURNS** + scikit-learn | Model orchestration, Q² accuracy checks; optional SROM |
| Robust objective + gradients | **Python (NumPy/SciPy)** | Novel formulation — must be custom glue code connecting FEniTop, OpenTURNS, and dolfiny MMA |
| Optimization driver (MMA) | **dolfiny MMA** (fenics-dolfiny/dolfiny, native PETSc TAO algorithm) + `taoproblem.py` wrapper (prebuilt) | TAO problem configuration only — no custom MMA math |
| Robust TO orchestration reference | **CU-UQ/TOuU** (reference architecture, MATLAB → Python translation where portable) | Stochastic-gradient handling logic adapted to this project's η-random-field context |
| Mesh generation + perturbation | **Gmsh** (Python API) | Geometry ↔ random field deformation logic (validation/viz only) |
| Metrology point-cloud registration | **Open3D** (colored ICP pipeline, prebuilt) | Thin wrapper (`registration.py`) calling Open3D API with project-specific I/O |
| Monte Carlo engine | **OpenTURNS** + FEniTop | Loop orchestration, MPI parallelization (reusing FEniTop's PETSc/MPI backend), two-tier UQ |
| Visualization | **PyVista** + **ParaView** | Opacity mapping, reliability maps, CAVE XR export |
| DevOps | Git, Docker (FEniTop's `jiayingqi/dolfinx-fenitop` image as base) / Singularity, pytest | Container specs extending FEniTop's base image, CI scripts |

---

## 7. Critical Constraints & Rules (Revised)

### Always Use Premade Tools — NEVER Reimplement
- Use **FEniTop** for ALL FEA, SIMP, filtering, and Heaviside projection — never write custom element assembly or a custom density filter
- Use **dolfiny MMA / PETSc TAO** for ALL optimization driving — never write a custom MMA solver or reintroduce OpenMDAO/PyOptSparse/ParOpt unless dolfiny MMA proves insufficient for a specific constraint type
- Use **Open3D** for ALL point cloud registration — never write a custom ICP implementation
- Use **CU-UQ/TOuU** as the architectural reference for stochastic-gradient robust TO orchestration before designing new orchestration logic from scratch
- Use **OpenTURNS** for ALL UQ/PCE/random field operations
- Use **Gmsh Python API** for ALL meshing
- Use **scikit-learn** for LASSO/GP — never write custom regression solvers
- Only write custom code for: (1) the novel spatially-correlated randomized-η formulation and its memoryless marginal transform, (2) the robust objective/gradient scalarization `J = μ + λσ`, and (3) integration glue code binding FEniTop, dolfiny MMA, Open3D, and OpenTURNS together

### Exact Mathematical Formulations (Do Not Deviate)
- SIMP penalization: `E(ρ) = E₀ · ρᵖ`, p=3 (as implemented in FEniTop)
- Density filter + Heaviside projection: `ρ̃ = Filter(ρ)`, `ρ̂ = Heaviside(ρ̃; η)` (FEniTop's Helmholtz filter + Heaviside projection, with η promoted to a random field via custom glue)
- Compliance: `C = U^T K U`
- Robust objective: `J = μ[C] + λ · σ[C]`
- Mean volume constraint: `E[V(ρ)] ≤ V_frac`
- PCE moments: `μ = c₀`, `σ² = Σ_{α≠0} c_α²`
- Covariance kernel (squared-exponential): `k(x,x') = σ² exp(−‖x−x'‖² / 2l²)`
- KL expansion (underlying Gaussian field): `G(x) = μ(x) + Σᵢ √λᵢ φᵢ(x) ξᵢ`
- Projection-threshold marginal transform: `η(x) = T(G(x))` (memoryless, calibrated from metrology data registered via Open3D)

### Verification Gates (Never Skip)
- FEA solver: verify δ = PL³/3EI before use in TO loop, using FEniTop's own cantilever example scripts as the test harness
- TO sensitivities: finite-difference check (perturbation 1e-6) on all elements; relative error < 1e-5 (validated against FEniTop's automatic-differentiation-derived sensitivities)
- Point cloud registration: Open3D `evaluate_registration` fitness/inlier RMSE must clear a defined threshold before deviation fields are accepted
- KL expansion: sample covariance must match theoretical kernel
- PCE surrogate: Q² ≥ 0.99 on held-out test set before deployment
- Random field: kernel parameters and marginal transform validated against empirical variogram/distribution from metrology data
- MMA convergence: KKT residual from dolfiny's TAO solver below tolerance before accepting a converged design
- Monte Carlo validation: 500+ full FEniTop FEA samples on final robust design; PCE-vs-MC comparison documented

### Code Standards
- Python 3.11+, type hints on all function signatures, `from __future__ import annotations`
- NumPy vectorized operations only — no Python loops over mesh elements
- Config-driven: all parameters in YAML (`config.yaml`), never hardcoded
- Logging via `logging` module — never `print()`
- Units explicit in variable names or docstrings (Pa, m, N)
- Conventional commits: `feat:`, `fix:`, `refactor:`, `docs:`, `test:`

### Do NOT
- Use scalar noise for manufacturing error — always spatially correlated random fields
- Use direct mesh/geometry warping as the *primary* manufacturing-error modeling device — reserve it for validation/visualization ensembles only
- Modify FEniTop's internals (extend via its documented η/filter interfaces, or fork only if strictly necessary and document why)
- Reimplement MMA, ICP, or FEA assembly now that dolfiny, Open3D, and FEniTop cover them
- Skip any verification step listed above
- Hardcode physical parameters
- Commit notebooks with uncleared outputs

---

## 8. Project Source Tree (Revised)
```text
src/
├── config/          # loader.py, schema.py, structures.py
├── fea/             # fenitop_adapter.py (wraps FEniTop calls), postprocess.py
├── meshing/         # importer.py, mesher.py, mapper.py
├── metrology/       # ingestion.py, registration.py (Open3D colored ICP wrapper), deviation.py, process_stats.py
├── random_fields/   # kernel.py, kl_expansion.py, perturbation.py, threshold_transform.py
├── sampling/        # sampler.py, splitter.py
├── surrogate/       # fea_at_samples.py, pce_builder.py, pce_model.py, sobol.py, srom_model.py
├── topology/        # heaviside_projection_glue.py (bridges FEniTop scalar-η to random field η(x))
├── optimization/    # robust_objective.py, robust_gradient.py, dolfiny_mma_driver.py, orchestrator.py (TOuU-informed)
├── validation/      # monte_carlo.py, comparator.py
└── viz/             # ensemble_generator.py, probability_cloud.py, pareto_plot.py, reliability_map.py

tests/
├── unit/            # one file per src module
├── regression/      # numerical regression tests (MBB beam, PCE coefficients)
└── integration/     # full pipeline tests, PCE vs MC consistency

configs/             # YAML problem definitions
data/metrology/      # raw CMM/scan datasets

external/
├── fenitop/         # vendored/pinned FEniTop repo (missionlab/fenitop)
├── dolfiny/         # vendored/pinned dolfiny repo, using src/dolfiny/mma.py + taoproblem.py
└── touu_reference/  # CU-UQ/TOuU reference code, consulted for orchestration logic (not directly executed)
```

**Entry point:** `python src/main.py --config configs/config.yaml`
**Tests:** `pytest --cov=src --cov-report=term-missing`
**Container:** `docker build -t robust-to --build-arg BASE=jiayingqi/dolfinx-fenitop . && docker run robust-to`

---

## 9. Key References (Revised, with prebuilt-software citations added)

| Topic | Reference |
|---|---|
| TO classic | Sigmund (2001), 99-line MATLAB code, *Struct Multidisc Optim* 21:120 |
| TO theory | Bendsoe & Sigmund, *Topology Optimization* (Springer, 2003) |
| **FEA + SIMP prebuilt package** | **Jia, Y., Wang, C. & Zhang, X.S. FEniTop: a simple FEniCSx implementation for 2D and 3D topology optimization supporting parallel computing. Struct Multidisc Optim 67, 140 (2024).** |
| Robust TO geometry (foundational alignment) | Chevens et al. (2011); Amir et al. (2012) — robust projection-based TO from uniform over/under-etching to spatially varying manufacturing errors via randomized Heaviside threshold |
| Robust TO geometry | Lazarov et al. (2012), *IJNME* 90:1321 |
| PCE foundations | Xiu & Karniadakis (2002), *SIAM J Sci Comput* 24:619 |
| Sparse PCE | Blatman & Sudret (2011), *J Comput Phys* 230:2345 |
| PCE + TO | Keshavarzzadeh et al. (2016), *CMAME* 318 |
| **Stochastic robust TO orchestration reference** | **CU-UQ/TOuU; companion paper: Topology Optimization under Uncertainty using a Stochastic Gradient-based Approach, Struct Multidisc Optim, 62(5), 2255-2278.** |
| SROM-based robust TO under loading uncertainty | Recent robust TO literature — flexible alternative to Monte Carlo/PCE |
| Robust TO for additive manufacturing distortion/anisotropy | Recent robust TO literature coupling process-specific distortion prediction with structural TO |
| Multi-fidelity variational autoencoders for robust TO | Recent robust TO literature on multi-fidelity surrogate modeling |
| Reliability-based robust TO under random fields | Recent robust TO literature on reliability-constrained formulations |
| **MMA optimizer (prebuilt implementation)** | Svanberg (1987), *IJNME* 24:359; implemented as native PETSc TAO algorithm in dolfiny (github.com/fenics-dolfiny/dolfiny/blob/main/src/dolfiny/mma.py) |
| Random fields | Vanmarcke, *Random Fields* (MIT Press, 1983/2010) |
| Sobol sensitivity | Saltelli et al., *Global Sensitivity Analysis* (Wiley, 2008) |
| **Point cloud registration (prebuilt implementation)** | Park, J., Zhou, Q.-Y., Koltun, V. Colored Point Cloud Registration Revisited, ICCV 2017; implemented in Open3D (open3d.org, `open3d.pipelines.registration`) |
| CNC error modeling | Ramesh et al. (2000), *Int J Machine Tools Manuf* 40:1235 |
| FDM error modeling | Turner et al. (2015), *Rapid Prototyping J* 21:137 |
| NIST standards gap | NIST SP Measurement Science Roadmap for Metal-Based AM |
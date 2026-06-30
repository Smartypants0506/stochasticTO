# NIST Stochastic Topology Optimization — Test Suite File Descriptions

---

## `tests/` Directory Overview

The test suite is organized into three sub-directories — `integration`, `regression`, and `unit` — mirroring the project's source packages. Integration tests validate full pipeline behavior end-to-end, regression tests guard against numerical drift, and unit tests cover individual module logic in isolation. A shared `conftest.py` at the root provides common fixtures and configuration for all test levels.

---

## `tests/integration/` — End-to-End Pipeline Tests

These tests exercise multiple modules together, verifying that data flows correctly across pipeline boundaries.

### `test_full_pipeline_small.py`
- **Purpose:** Runs the complete pipeline (meshing → FEA → KL expansion → PCE → robust optimization) on a small, low-resolution test geometry
- **What it checks:** That the pipeline executes without error from a STEP file input to a converged robust density field; asserts that final compliance is finite, volume constraint is satisfied within tolerance, and output files (`rho_robust.npy`, `pce_model.pkl`) are created
- **Fixtures used:** `small_step_fixture`, `test_config_small` from `conftest.py`

### `test_mc_vs_pce.py`
- **Purpose:** Validates that PCE surrogate predictions are consistent with Monte Carlo FEA evaluations on a small sample set
- **What it checks:** Computes PCE mean and variance against a small MC ensemble (~50 samples); asserts relative error on mean `< 5%` and on variance `< 10%`; acts as a fast integration-level proxy for the full `validation/comparator.py` logic

---

## `tests/regression/` — Numerical Regression Tests

These tests pin specific numerical outputs to known-good reference values, catching any unintended changes to physics, solvers, or surrogate logic between code updates.

### `test_nominal_simp_2d.py`
- **Purpose:** Runs the nominal SIMP topology optimization loop on a 2D MBB beam benchmark and compares final compliance and density field against a stored reference solution
- **What it checks:** Final compliance within `1e-4` of the reference value; density field L2 norm within `1e-3`; KKT residual at convergence below threshold

### `test_pce_accuracy.py`
- **Purpose:** Rebuilds a PCE surrogate from a fixed set of stored training samples and checks that the resulting Q² and coefficient array match the reference values
- **What it checks:** `Q² ≥ 0.99`; leading PCE coefficients within relative tolerance `1e-5` of stored reference; guards against changes in OpenTURNS version behavior

### `test_robust_vs_nominal.py`
- **Purpose:** Verifies that a robust design (`λ > 0`) yields strictly lower compliance standard deviation than the nominal design (`λ = 0`) at the cost of higher mean compliance
- **What it checks:** `σ_C(robust) < σ_C(nominal)` and `μ_C(robust) ≥ μ_C(nominal)` for at least one non-zero `λ` value; enforces the expected Pareto trade-off direction

---

## `tests/unit/` — Module-Level Unit Tests

Each file targets a single source module, using mocked or minimal fixtures to keep tests fast and isolated.

### `test_config.py`
- **Purpose:** Tests `src/config/loader.py`, `schema.py`, and `structures.py`
- **What it checks:** Valid YAML loads without error; missing required fields raise `ValidationError`; out-of-range values (e.g., `V_frac > 1`) are rejected; `Config` dataclass fields are correctly typed

### `test_fea.py`
- **Purpose:** Tests `src/fea/assembler.py`, `boundary.py`, `solver.py`, and `postprocess.py`
- **What it checks:** Stiffness matrix `K` is symmetric positive-definite on a unit cube mesh; applying a known point load yields displacement consistent with an analytic Timoshenko beam solution; compliance output is positive and finite

### `test_filters.py`
- **Purpose:** Tests `src/topology/filters.py`
- **What it checks:** Density filter produces a smoother field (reduced gradient norm) than the input; Heaviside projection is monotone and satisfies `ρ_tilde ∈ [0,1]`; filter Jacobian passes a finite-difference check

### `test_kernel.py`
- **Purpose:** Tests `src/random_fields/kernel.py`
- **What it checks:** Fitted kernel hyperparameters (`σ²`, `ℓ`) are positive; the kernel matrix is symmetric positive semi-definite; leave-one-out cross-validation RMSE is below a threshold for synthetic deviation data

### `test_kl_expansion.py`
- **Purpose:** Tests `src/random_fields/kl_expansion.py`
- **What it checks:** Eigenvalues `λ_i` are positive and sorted in descending order; eigenfunctions are orthonormal on the mesh; truncation to `N_KL` modes captures ≥ 95% of total variance

### `test_meshing.py`
- **Purpose:** Tests `src/meshing/importer.py`, `mesher.py`, and `mapper.py`
- **What it checks:** A minimal STEP fixture imports without error; generated mesh has nonzero element count and all positive Jacobians; `bc_marker_dict` contains the expected keys matching the fixture geometry's surface names

### `test_metrology.py`
- **Purpose:** Tests `src/metrology/ingestion.py`, `registration.py`, `deviation.py`, and `process_stats.py`
- **What it checks:** CSV ingestion produces correctly shaped `points` and `deviations` arrays; ICP registration reduces point-to-surface distance; RBF interpolation covers all mesh nodes; `Cp` and `Cpk` values are computed correctly against analytic reference

### `test_mma_driver.py`
- **Purpose:** Tests `src/optimization/mma_driver.py`
- **What it checks:** MMA subproblem solution satisfies the volume constraint; density update stays within move limits; KKT residual decreases monotonically over several iterations on a small synthetic objective

### `test_pce_builder.py`
- **Purpose:** Tests `src/surrogate/pce_builder.py` and `pce_model.py`
- **What it checks:** PCE fits a known analytic function (e.g., Ishigami) to within `Q² ≥ 0.99`; `PCEModel.mean()` and `PCEModel.variance()` match the analytic values; `predict()` output shape matches input sample count

### `test_perturbation.py`
- **Purpose:** Tests `src/random_fields/perturbation.py`
- **What it checks:** Perturbed mesh node coordinates differ from nominal by the expected KL expansion magnitude; mesh quality (Jacobian determinant) remains positive after perturbation; BC marker tags are preserved on the perturbed mesh

### `test_robust_gradient.py`
- **Purpose:** Tests `src/optimization/robust_gradient.py`
- **What it checks:** Gradient `dJ/dρ_e` passes a finite-difference check against the robust objective; gradient shape matches `[N_elements]`; gradient through the filter chain is consistent with the filter Jacobian

### `test_robust_objective.py`
- **Purpose:** Tests `src/optimization/robust_objective.py`
- **What it checks:** Objective `J = μ_C + λ σ_C` is computed correctly from PCE coefficients for `λ = 0`, `λ = 1`, and `λ = 5`; volume constraint `g` is zero when `V_frac` exactly matches the current density field volume

### `test_sampler.py`
- **Purpose:** Tests `src/sampling/sampler.py` and `splitter.py`
- **What it checks:** LHS sample matrix `Ξ` has correct shape `[N_samples × N_KL]` and is space-filling (minimum inter-sample distance above threshold); train/test split preserves total sample count and is reproducible with a fixed seed

### `test_simp.py`
- **Purpose:** Tests `src/topology/mma_simp.py` end-to-end on a small 2D mesh
- **What it checks:** Volume constraint is satisfied at convergence; density field converges to near-binary values (`ρ_e < 0.1` or `> 0.9` for ≥ 90% of elements); compliance decreases monotonically over the first 10 iterations

### `test_sobol.py`
- **Purpose:** Tests `src/surrogate/sobol.py`
- **What it checks:** First-order Sobol indices sum to `≤ 1`; total Sobol indices `S_i^T ≥ S_i` for all modes; indices computed from a known PCE (e.g., Sobol G-function) match analytic reference values within `1e-3`

---

## `conftest.py` — Shared Test Fixtures

- **Purpose:** Provides pytest fixtures reused across all three test levels
- **What it provides:** `test_config_small` — a minimal `Config` object pointing to a tiny STEP file and low `N_KL`/`N_train` values for fast execution; `small_mesh_fixture` — a pre-built `dolfinx.mesh.Mesh` for a unit cube; `synthetic_kl_model` — a `KLModel` with 5 analytic eigenpairs for surrogate and perturbation tests; `tmp_output_dir` — a temporary directory for output file assertions
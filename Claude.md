# CLAUDE.md

## Project Overview
NIST Stochastic Topology Optimization Under Manufacturing Uncertainty framework. Builds robust topology-optimized designs that account for spatially correlated geometric manufacturing errors (CNC/FDM) modeled as random fields from real process metrology data. Fills NIST standards gap: no existing guidance on how manufacturing process capability should constrain topology optimization.

## Tech Stack
- Python 3.11+ (NumPy, SciPy, Matplotlib, PyVista)
- FEniCSx (dolfinx) — FEA engine via UFL variational forms
- OpenTURNS — UQ, PCE, KL expansion, random field generation
- OpenMDAO / PyOptSparse — optimization driver (MMA solver)
- Gmsh — mesh generation (Python API)
- scikit-learn — LASSO for sparse PCE, GP comparisons
- ParaView — post-processing and CAVE visualization
- Docker/Singularity — reproducible environments

## Commands
- `pytest` — run all tests
- `pytest --cov=src --cov-report=term-missing` — coverage
- `python -m mypy src/` — type checking
- `docker build -t robust-to .` — build container
- `python src/main.py --config config.yaml` — run full pipeline

## Project Structure
- `src/fea/` — FEniCSx linear elasticity solver
- `src/topology/` — SIMP TO loop, adjoint sensitivities, density filtering
- `src/random_fields/` — KL expansion, kernel fitting, Cholesky sampling
- `src/pce/` — Non-intrusive PCE surrogate (OpenTURNS wrapper)
- `src/robust/` — Robust objective (μ[C] + λσ[C]), robust gradient, MMA driver
- `src/metrology/` — CMM/laser scan data ingestion, variogram fitting
- `src/visualization/` — PyVista/ParaView probability cloud rendering
- `src/utils/` — Config loading, dimensional analysis, logging
- `tests/` — mirrors src/ structure
- `data/metrology/` — raw CMM/scan datasets
- `configs/` — YAML problem definitions

## Critical Rules

### ALWAYS Prioritize Premade Solutions
- IMPORTANT: Use OpenTURNS for ALL UQ/PCE/random field operations — never reimplement KL expansion, PCE fitting, or Sobol analysis from scratch.
- IMPORTANT: Use FEniCSx for ALL FEA — never write custom element assembly.
- IMPORTANT: Use Gmsh Python API for ALL meshing — never write custom mesh generators.
- IMPORTANT: Use PyOptSparse/OpenMDAO MMA for optimization — never write custom optimizers.
- IMPORTANT: Use scikit-learn for LASSO/GP — never write custom regression solvers.
- Before writing ANY numerical algorithm, search for an existing implementation in the stack above. Only write custom code for the novel robust TO formulation integration glue.

### Mathematical Rigor Requirements
- Every numerical implementation MUST reference a specific equation from the onboarding guide or literature.
- Use established formulations exactly:
  - SIMP penalization: E(ρ) = E₀ · ρᵖ, p=3
  - Compliance: C = U^T K U
  - Robust objective: J = μ[C] + λ·σ[C]
  - PCE moments: μ = c₀, σ² = Σ(α≠0) c_α²
  - Covariance kernel: k(x,x') = σ² exp(-‖x-x'‖² / 2l²)
  - KL expansion: Z(x) = μ(x) + Σᵢ √λᵢ φᵢ(x) ξᵢ
- Adjoint sensitivities must match finite-difference verification to relative error < 1e-5.
- All solvers must demonstrate mesh convergence before use in optimization loop.
- PCE surrogate requires Q² > 0.99 on held-out test set before deployment.
- Never approximate or simplify a standard formulation without explicit justification in comments.

### Code Quality
- Type hints on ALL function signatures (parameters + return types)
- `from __future__ import annotations` in every module
- NumPy vectorized operations only — no Python loops over mesh elements
- Docstrings (Google style) on all public functions citing the equation/method source
- Units must be explicit in variable names or docstrings (Pa, m, N)
- Dimensional consistency checks at module boundaries

### Verification Before Proceeding
- FEA solver: verify against analytical solution (cantilever: δ = PL³/3EI) before using in TO loop
- TO sensitivities: finite-difference check (perturbation 1e-6) on ALL elements
- KL expansion: verify sample covariance matches theoretical covariance kernel
- PCE: report Q² on held-out set; iterate until Q² > 0.99
- Random field: verify kernel parameters against empirical variogram from metrology data
- Monte Carlo validation: 500+ full FEA samples on final robust design

## Coding Conventions
- Conventional commits: feat:, fix:, refactor:, docs:, test:
- One logical change per commit
- Config-driven design — all problem parameters in YAML, never hardcoded
- Immutable data structures where possible (frozen dataclasses)
- Logging via `logging` module — never `print()`

## Do NOT
- Do not reimplement functionality available in OpenTURNS, FEniCSx, Gmsh, or scikit-learn
- Do not use scalar noise models for manufacturing error — always use spatially correlated random fields
- Do not skip verification steps (FD checks, convergence studies, Q² validation)
- Do not hardcode physical parameters — use config files
- Do not commit notebooks with uncleared outputs
- Do not use `import *`
- Do not modify the FEA solver internals (use non-intrusive PCE only)

## Reference Materials
- @docs/onboarding-guide.md — full mathematical background and 12-week roadmap
- @docs/framework-map.md — component architecture and data flow
- @docs/implementation-modules.md — specific module I/O specifications
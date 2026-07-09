## Overall framework map 

At a high level, the system can be organized into the following pipeline: [1] 

## 1. Design & deterministic FEA core 

Inputs: design domain, boundary conditions, loads, material model, volume fraction constraint. [1] 

Component: FEA + topology optimization engine (SIMP, adjoint sensitivities). 

   - Outputs: compliance, sensitivities, optimized nominal density field. [1] 

2. Manufacturing uncertainty model (random fields) 

   - Inputs: premade metrology datasets (CMM scans, laser scans, process capability Cp/Cpk for CNC and FDM). [1] 

   - Components: random field kernel fitting, KL expansion, sampling. [1] 

   - Outputs: low-dimensional KL coefficients describing spatially correlated geometric error fields. [1] 

## 3. Surrogate modeling (PCE) 

- Inputs: KL coefficients, compliance from FEA at training points. [1] 

- Components: non-intrusive sparse Polynomial Chaos Expansion (PCE) and Sobol sensitivity analysis. [2] [1] 

- Outputs: analytic mean and variance of compliance, Sobol indices, surrogate model for robust objective. [3] [1] 

## 4. Robust topology optimization loop 

   - Inputs: current density field, PCE coefficients, KL modes, volume constraint, robust tradeoff parameter . [1] 

   - Components: robust objective , adjoint sensitivities, MMA-based update. [4] [1] 

   - Outputs: converged robust design, Pareto front across mean/variance objectives. [1] 

5. Monte Carlo validation & visualization 

   - Inputs: robust design, random field sampler, mesh generator. [1] 

   - Components: large Monte Carlo sampling (e.g., 5 000 geometries), FEA evaluations, visualization pipeline for CAVE. [1] 

   - Outputs: probability cloud of geometries, plots of compliance distributions and Pareto frontier. [1] 

## 6. Infrastructure 

- Components: version control (Git), containers (Docker/Singularity), Jupyter for exploration, CI/tests. [1] 

- Purpose: keep the product cohesive and reproducible across machines and HPC clusters. [1] 

## – Component tool mapping 

The table below shows the main parts, the preferred open-source tool, inputs/outputs, and [5] [2] [4] [1] what remains as custom glue code. 

|Function|Inputs|Outputs|Preferred tool(s)|Custom code scope|
|---|---|---|---|---|
|Deterministic FEA<br>(linear elasticity)|Domain,BCs,<br>loads,material|Displacements,<br>stresses,compliance|FEniCSx (UFL,dolfinx)|Problem setup,<br>post-processing|
|Classic topology<br>optimization<br>(SIMP)|FEA results,design<br>densities,volume<br>fraction|Nominal optimal<br>density field|FEniCSx+Python|SIMP loop,adjoint<br>sensitivities|
|Random field<br>modeling(KL,<br>kernels)|Premade<br>metrology<br>datasets,process<br>capability data|KL modes,fitted<br>covariance kernel,<br>random samples|OpenTURNS|Mapping geometry<br>↔random field<br>parameters|
|Surrogate PCE&<br>Sobol indices|KL coeficients,<br>compliance<br>samples|PCE coeficients,<br>mean/variance,Sobol<br>sensitivity|OpenTURNS +<br>scikit-learn|Model<br>orchestration,<br>accuracy checks|
|Robust objective<br>&sensitivities|PCE moments,<br>density field,<br>volume constraint|Gradient of robust<br>objective,constraint<br>gradient|Python(NumPy/SciPy)|Derivation&<br>implementation|
|Optimization<br>driver(MMA)|Objective,<br>gradients,<br>constraints|Updated design<br>variables,convergence<br>status|PyOptSparse/ParOpt<br>via OpenMDAO|Component<br>wrapper&driver<br>config|
|Mesh generation<br>&perturbation|Nominal<br>CAD/mesh,random<br>field realization|Remeshed perturbed<br>geometries for FEA|Gmsh (Python API)|Geometry↔random<br>field deformation<br>logic|
|Monte Carlo<br>engine|Random field<br>sampler,robust<br>design|Compliance samples,<br>geometry ensemble|OpenTURNS+FEniCSx|Loop orchestration,<br>parallelization|
|Visualization<br>(probability<br>cloud)|VTK meshes,scalar<br>fields(e.g.<br>coeficient of<br>variation)|CAVE wall render,local<br>plots|PyVista+ParaView|Visual mapping and<br>storytelling|
|DevOps&<br>reproducibility|Source code,<br>dependencies|Docker/Singularity<br>images,CI pipeline|Git,Docker,Singularity|Container specs and<br>CI scripts|



All of the heavy numerical pieces (FEA, random fields, PCE, MMA, meshing, visualization) are covered by mature open-source packages; custom work focuses on the novel robust TO [5] [2] [4] [1] formulation and the integration glue. 

## Why FEniCSx for FEA & topology optimization 

FEniCSx is a next-generation finite element platform that lets you define PDEs and variational forms in a high-level Unified Form Language (UFL), with automatic assembly and solving through Python and C++ back ends. Compared to alternatives like MFEM or deal.II, which have strong C++ APIs but less native Python integration, FEniCSx fits better with a Python-first stack and makes it easier to couple FEA with OpenTURNS, scikit-learn, and OpenMDAO. [6] [7] [5] [1] 

The intern onboarding guide already standardizes on FEniCSx for linear elasticity and topology optimization, emphasizing its ability to represent DG0 design variables and adjoint-based 

sensitivities within one framework. Commercial solvers (Abaqus, ANSYS, COMSOL) offer robustness but constrain scripting, licensing, and containerization, which conflicts with the open-source and reproducibility goals of the NIST project. [1] 

Given: 

- Native Python/UFL workflow for custom variational forms. [6] [5] 

- Existing project scaffolding and roadmap already built around FEniCSx. [1] 

- Easier integration into Docker/Singularity environments. [1] 

FEniCSx is the most cohesive choice for the FEA/topology optimization core. 

## Why OpenTURNS for random fields, UQ, and PCE 

- OpenTURNS is an open source uncertainty quantification library that explicitly supports random fields (via KL expansion), functional chaos (PCE), Sobol sensitivity analysis, and Monte Carlo engines in a unified API. The onboarding guide calls OpenTURNS “the most complete open-source UQ library available” for this project and highlights its dedicated classes for KL [2] [3] [1] expansion and PCE (e.g., KarhunenLoeveP1Algorithm, FunctionalChaosAlgorithm). 

## Compared to alternatives: 

- ChaosPy and similar Python libraries provide PCE but have more limited tooling for random fields and metrology-driven kernel fitting. 

- UQLab is powerful but MATLAB-based, which conflicts with the all-Python stack and [1] 

- containerized deployment. 

- Frameworks like Dakota or OpenCOSSAN focus broadly on UQ but lack the tight polynomial chaos and KL tooling that OpenTURNS offers out-of-the-box. [8] [3] 

For this project, the key advantages of OpenTURNS are: 

- Direct support for Gaussian random fields and KL expansion suitable for metrology-fitted [2] [1] 

- covariance kernels. 

- Non-intrusive sparse PCE with cross-validation tools for surrogate accuracy (Q²) and Sobol indices derived from the PCE coefficients. [3] [1] 

- Integration of Monte Carlo, quasi-Monte Carlo, and sensitivity experiments in one package, [9] [10] [1] 

- simplifying the robust TO loop and validation. 

ʼ These features align almost exactly with the project s requirements, minimizing custom implementation of random fields and PCE and keeping the UQ layer cohesive. 

## Why OpenMDAO + PyOptSparse (ParOpt) for MMA 

OpenMDAO is a Python framework for multidisciplinary optimization that provides drivers interfacing with external optimizers via pyOptSparseDriver. pyOptSparse itself is designed for - — large scale constrained sparse nonlinear problems and exposes multiple optimizers including ParOpt, which implements trust-region, interior-point, and MMA algorithms. [11] [12] [13] [4] 

## For this framework, we specifically want MMA: 

- The intern guide cites Svanbergʼs MMA as an “industrial-grade” solver for structural optimization, superior to simple Optimality Criteria (OC) for robust convergence in topology optimization. [1] 

- ParOpt offers open-source implementations of MMA integrated into pyOptSparse, meaning we can call MMA through a consistent Python interface without writing our own solver. [13] [4] 

## Compared to other options: 

- scipy.optimize (e.g., SLSQP, trust-constr) is convenient but not tailored to MMA nor to very large sparse constrained problems typical in TO, and lacks the structural optimization lineage that MMA has. [14] 

- IPOPT or SNOPT can be accessed via pyOptSparse, but they are general nonlinear optimizers; MMA is specifically configured for structural optimization with moving asymptotes and good behavior on SIMP-style density problems. [12] [11] 

- NLopt and similar libraries provide many algorithms but require additional integration effort and do not align directly with the existing OpenMDAO/pyOptSparse ecosystem. [12] 

Using OpenMDAO + pyOptSparse/ParOpt therefore: 

- Satisfies the “use MMA as the optimization solver” constraint via a mature, tested implementation. [4] [13] 

- ʼ 

- Fits the project s plan to wrap the TO loop in an OpenMDAO component, letting gradients flow cleanly and enabling future multi-disciplinary extensions. [11] [1] 

- Keeps the whole optimization stack in Python, matching the rest of the framework and simplifying deployment. [1] 

## Why Gmsh for mesh generation 

Gmsh is an open-source mesh generator widely used in the FEA community and offers a Python API for scripting mesh generation and refinement. The onboarding document explicitly chooses Gmsh for remeshing perturbed geometries while preserving boundary conditions and recommends learning its Python interface and element size control, which shows it is already [1] integrated into the planned workflow. 

Alternative meshers (e.g., Netgen, TetGen, Salome) can produce quality meshes but: 

- Do not have the same level of direct scripting integration with FEniCSx that the community commonly uses. 

- Might require additional work to maintain consistent BCs and geometry perturbation [1] 

- pipelines across thousands of random field realizations. 

Since: 

The project stack already standardizes on Gmsh for this purpose. 

[1] 

- Gmsh supports both structured and unstructured meshes, plus size fields to refine critical regions affected by manufacturing variation. [1] 

- It integrates smoothly with FEniCSx via standard mesh formats and existing workflows in the FEniCS ecosystem. [7] 

Gmsh is the cohesive choice for mesh generation and geometry perturbation. 

## Data strategy: prioritizing premade datasets 

The project is explicitly built around using real process metrology data—CMM scans, laser/structured-light scans, and statistical process control data—to parameterize the random field model. The onboarding guide references NIST metal additive manufacturing roadmaps and existing metrology datasets, and the roadmap includes tasks to analyze provided CMM [1] scans of test coupons rather than collecting entirely new data. 

To prioritize premade datasets: 

- Start from NISTʼs existing metrology campaigns and any publicly available CNC and FDM capability datasets (Cp/Cpk, geometric deviation scans) specified in the project materials. [1] 

- Use these datasets to fit covariance kernels and KL expansions in OpenTURNS, validating with leave-one-out cross-validation as outlined in the roadmap. [1] 

- Only design new metrology experiments if coverage gaps appear (e.g., a process or geometry not represented in existing datasets), and structure them as incremental [1] 

- additions rather than the primary source. 

This approach adheres to the “premade datasets first” constraint while staying faithful to the ʼ project s emphasis on physically grounded random fields derived from real measurement data. 

## Cohesive product view 

Putting everything together, the most cohesive product architecture is: 

- Language & environment: Python 3.11, Docker/Singularity, Git, CI. [1] 

- Core physics & TO: FEniCSx for FEA and topology optimization, with custom SIMP and adjoint implementations coded in Python/UFL. [6] [1] 

- Uncertainty & surrogates: OpenTURNS for random fields, KL, PCE, Sobol, Monte Carlo/Quasi-Monte Carlo. [3] [2] [1] 

- Optimization: OpenMDAO + pyOptSparse/ParOpt to run MMA on the robust objective, interfacing with FEniCSx and OpenTURNS components. [13] [11] [4] [1] 

- Geometry & visualization: Gmsh for meshing perturbed geometries; PyVista and ParaView for visualizing the probability cloud and performance metrics. [1] 

This stack minimizes bespoke numerical implementations, maximizes reuse of mature open-source libraries, respects the requirement to use MMA, and keeps all components within a cohesive Python-centric ecosystem aligned with the NIST intern roadmap. [5] [4] [2] [1] 

⁂ 

1. intern_onboarding_guide.pdf 

2. https://openturns.github.io/openturns/1.22/theory/meta_modeling/chaos_basis.html 

3. https://openturns.github.io/openturns/latest/theory/meta_modeling/functional_chaos.html 

4. https://mdolab-pyoptsparse.readthedocs-hosted.com/en/latest/optimizers/ParOpt.html 

5. https://fenicsproject.org 

6. https://jsdokken.com/dolfinx-tutorial/fem.html 

7. https://fenicsproject.discourse.group/t/comparison-between-fenicsx-and-mfem/13890 

8. https://www.academia.edu/23074629/Implementation_of_a_polynomial_chaos_toolbox_in_OpenTURNS_and_applic aations_to_structural_reliability_and_sensitivity_analyses 

9. https://openturns.github.io/openturns/1.24/auto_meta_modeling/polynomial_chaos_metamodel/plot_chaos_conditi onal_expectation.html 

10. https://openturns.discourse.group/t/chaos-vs-saltelli/70 

11. https://openmdao.org/newdocs/versions/latest/features/building_blocks/drivers/pyoptsparse_driver.html 

12. https://websites.umich.edu/~mdolaboratory/pdf/Wu2020a.pdf 

13. https://github.com/mdolab/pyoptsparse 

14. https://stackoverflow.com/questions/50386480/openmdao-what-are-the-pros-and-cons-when-comparing-scipyoptimi zedriver-to-pyopt 

15. https://github.com/fenics/basix 

16. https://www.youtube.com/watch?v=D-YcVd4-_2E 


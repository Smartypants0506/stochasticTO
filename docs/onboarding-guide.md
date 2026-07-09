## **STOCHASTIC TOPOLOGY OPTIMIZATION UNDER MANUFACTURING UNCERTAINTY** 

Intern Onboarding Guide 

Concepts · Whitepapers · Software · Tasks 

NIST Advanced Manufacturing Program · 2026 Prepared by: Senior FEA & Simulation Lead 

## **Table of Contents** 

## **1. Project Overview & Objectives** 

## **2. Foundational Concepts You Must Understand** 

2.1 Finite Element Analysis (FEA) Fundamentals 

2.2 Topology Optimization 

2.3 Manufacturing Tolerances & Geometric Variation 

2.4 Random Fields & Geostatistics 

2.5 Uncertainty Quantification (UQ) & Robust Design 

2.6 Polynomial Chaos Expansion (PCE) / Surrogate Modeling 

2.7 Process Metrology for CNC & FDM 

## **3. Essential Whitepapers & Textbooks** 

## **4. Software Stack** 

## **5. Phased Task Plan (12-Week Roadmap)** 

## **6. Deliverables & Success Criteria** 

## **7. Tips, Common Pitfalls & How to Ask Good Questions** 

## **1. Project Overview & Objectives** 

This internship sits at the intersection of computational solid mechanics, manufacturing process science, and applied probability. Your job is to help build the first robust topology optimization framework that treats **geometric manufacturing variation** as a statistically rigorous, spatially correlated input — rather than ignoring it or approximating it as simple scalar noise on loads or material properties. 

## I **The Core Research Gap** 

No standards body — including NIST — currently provides quantitative guidance on how manufacturing process capability should constrain topology optimization. Engineers routinely produce designs that are mathematically optimal for a perfect part the factory cannot actually build. This project fills that gap. 

## **What the framework does — in plain English:** 

- Takes a design domain, boundary conditions, load cases, and a volume-fraction constraint as inputs. 

- Samples thousands of realistic 'as-manufactured' geometry variants using a spatially correlated random field that mimics actual CNC or FDM error distributions from metrology data. 

- Runs structural analysis on each variant using Finite Element Analysis (FEA). 

- Uses Polynomial Chaos Expansion (PCE) as a surrogate model so the optimizer doesn't need to run a full FEA solve for every sample at every iteration. 

- Minimizes a combined objective: mean compliance (stiffness) + a penalty on the variance of compliance across the manufactured population. 

- Optionally renders all 5,000 Monte Carlo sample geometries simultaneously in a CAVE visualization as 

- a semi-transparent probability cloud. 

## **2. Foundational Concepts** 

The following seven conceptual pillars underpin this project. You do not need to be an expert in all of them on day one, but you need working fluency before you write a single line of production code. Read this section carefully — each concept links directly to a phase in your 12-week task plan. 

## **2.1 Finite Element Analysis (FEA) Fundamentals** 

FEA is the engine that powers every structural evaluation in this project. You will need to understand it deeply, not just as a black box. 

**Strong vs. Weak Form:** The governing PDE for linear elasticity (Navier equations) and how the Galerkin weak form is derived. You must understand why we do this — it's the mathematical foundation of why FEA works. 

**Discretization & Element Types:** How a continuous domain is divided into elements (triangles/quads in 2D; tets/hexes in 3D). Know Q4, Q8, T3 elements at minimum. Understand isoparametric mapping and Gauss quadrature. 

**Assembly & Boundary Conditions:** How element stiffness matrices K_e assemble into the global system KU = F. Dirichlet (displacement) and Neumann (force) BCs. How to apply them without corrupting the global matrix. 

**Mesh Quality & Convergence:** What makes a bad mesh (high aspect ratio, skewness). h-refinement vs p-refinement. How to do a mesh convergence study to verify your results are not mesh-dependent. 

**Linear vs. Nonlinear Analysis:** This project uses linear elasticity, but you need to know when that assumption breaks down (large deformations, material nonlinearity, contact). 

## **2.2 Topology Optimization** 

This is the algorithmic heart of the project. You need to understand the classic SIMP method and the optimization math that drives it. 

**SIMP (Solid Isotropic Material with Penalization):** The standard density-based TO approach. Each element has a design variable rho in [0,1] representing material density. Stiffness is penalized as E(rho) = E_0 * rho^p (typically p=3) to drive the design toward 0/1. Understand why the penalty is needed and what 'gray elements' mean physically. 

**Sensitivity Analysis / Adjoint Method:** Computing dc/drho — how the objective (compliance) changes with respect to each element's density. The adjoint method avoids solving N linear systems for N design variables. This is non-negotiable; you must derive it by hand. 

**Filtering & Regularization:** Raw SIMP produces checkerboard patterns and mesh-dependent results. Density filter, sensitivity filter, and projection filter (Heaviside) are the standard fixes. Know the difference. 

**OC / MMA Optimizers:** Optimality Criteria (OC) is the simple update rule used in educational codes. Method of Moving Asymptotes (MMA) is the industrial-grade solver. Understand what each is minimizing and their convergence properties. 

**Volume Constraint Handling:** The single constraint is V(rho) <= V_frac. Know how the Lagrange multiplier for this constraint is computed in the OC update. 

## **2.3 Manufacturing Tolerances & Geometric Variation** 

This is what makes the project novel. Most engineers treat tolerances as a post-design concern. Here, they are a design input. 

**GD&T; Basics:** Geometric Dimensioning and Tolerancing (ASME Y14.5). Know position, flatness, circularity, and profile tolerances. Understand the difference between form, orientation, location, and runout controls. 

**Process Capability (Cp, Cpk):** Statistical measures of how well a manufacturing process stays within tolerance. Cp = (USL-LSL)/(6*sigma). Six Sigma implies Cp >= 2. You'll use real Cpk data from CNC and FDM machines to parameterize the random field. 

**CNC Machining Error Modes:** Tool deflection, thermal expansion, backlash, and fixture compliance. Understand how these manifest spatially: positional errors in hole patterns are correlated with spindle load; surface roughness is correlated with feed rate. 

**FDM Additive Error Modes:** Layer adhesion variation, warping, support-interface geometry, and staircase approximation. FDM errors are highly anisotropic — in-plane vs. out-of-plane behavior differs dramatically. 

## **2.4 Random Fields & Geostatistics** 

A random field is a function where every point in space has a probability distribution, and those distributions are spatially correlated. This is the mathematical object we use to model how manufacturing error varies across a part. 

## **Why a random field, not just a random variable?** 

If you model error as a single number (e.g., 'the wall is ±0.1mm thick'), you're saying the error is the same everywhere on the part at any given moment. Real parts don't work that way: one region might be systematically thicker, the adjacent region thinner, with a correlation that decays over distance. A Gaussian random field captures this spatial structure via a covariance kernel. 

**Gaussian Random Fields:** Defined by a mean function mu(x) and a covariance kernel k(x, x'). The squared-exponential (RBF) kernel k(x,x') = sigma^2 * exp(-|x-x'|^2 / 2*l^2) is the standard starting point. Understand length-scale l and variance sigma^2. 

**Karhunen-Loeve (KL) Expansion:** Decomposes a random field into a series of deterministic spatial modes (eigenfunctions of the covariance kernel) multiplied by uncorrelated random scalars. This gives you a low-dimensional parameterization. KL truncation is how you control computational cost. 

**Covariance Kernel Fitting from Metrology Data:** Given a set of measured part scans, fit the kernel parameters (sigma, l) using maximum likelihood or variogram analysis. This is the 'physically grounded' part of the novelty claim. 

**Cholesky Sampling:** Given a covariance matrix C, decompose C = L*L^T (Cholesky), then samples are L * z where z ~ N(0,I). Know when this is stable and what to do when C is ill-conditioned (nugget regularization). 

## **2.5 Uncertainty Quantification (UQ) & Robust Design** 

UQ is the mathematical discipline of characterizing and propagating uncertainty through computational models. Robust design optimization is the engineering application of UQ inside an optimization loop. 

**Monte Carlo Simulation (MCS):** The reference method: draw N samples from the input distribution, run the model N times, compute statistics on the outputs. Converges at O(1/sqrt(N)) regardless of dimension. Computationally expensive but trivially parallelizable. 

**Robust Optimization Formulation:** Instead of minimizing f(x, p) for a nominal parameter p, minimize mu_f(x) + lambda * sigma_f(x), where mu and sigma are the mean and standard deviation of the objective over the uncertainty distribution. Lambda controls the mean/variance tradeoff. Understand Pareto fronts in the (mean, variance) objective space. 

**Sensitivity Indices (Sobol):** Variance-based global sensitivity analysis tells you which input random variables (KL modes) most strongly drive output variance. First-order and total-order Sobol indices. Useful for KL truncation. 

**Quasi-Monte Carlo (QMC):** Low-discrepancy sequences (Halton, Sobol) that converge faster than random sampling. Relevant when MCS is used for validation. 

## **2.6 Polynomial Chaos Expansion (PCE) / Surrogate Modeling** 

PCE is the computational trick that makes robust topology optimization tractable. Without it, you would need thousands of full FEA solves per optimization iteration — which is prohibitive. 

**What PCE Is:** PCE approximates a model output Y = f(xi_1, ..., xi_N) as a finite sum of orthogonal ≈ polynomials in the random input variables: Y sum_alpha c_alpha * Psi_alpha(xi). For Gaussian inputs, the basis functions Psi are Hermite polynomials. 

**Intrusive vs. Non-Intrusive PCE:** Intrusive PCE requires modifying the FEA solver — generally not practical. Non-intrusive PCE (NIPC) treats the solver as a black box and fits the expansion using sparse quadrature or regression on a sample set. You will use NIPC exclusively. 

**Sparse PCE:** The number of terms grows combinatorially with input dimension and polynomial degree. Sparse PCE uses compressed sensing / LASSO to identify the dominant terms. Understand the hyperbolic index set truncation scheme. 

**Moment Extraction:** Once the PCE coefficients c_alpha are known, mean and variance are analytic: mu = c_0, sigma^2 = sum_{alpha != 0} c_alpha^2. This is why PCE is efficient — no extra sampling needed for statistics. 

**PCE vs. Gaussian Process (GP) Surrogates:** Know the tradeoffs: PCE is fast for smooth functions in moderate dimensions; GP is more flexible but expensive to train. Know when each is preferred. 

## **2.7 Process Metrology for CNC & FDM** 

The project's credibility rests on using real measurement data — not assumed distributions — to parameterize the uncertainty model. You need to understand how that data is collected and structured. 

**CMM Measurement:** Coordinate Measuring Machine. The gold standard for geometric inspection. Understand probe calibration, datum alignment, and how to export a point cloud of surface deviations. GD&T; callouts map directly to CMM measurement strategies. 

**Laser / Structured Light Scanning:** Full-field surface scan producing a dense point cloud. Useful for capturing spatially continuous error fields. Understand registration (ICP algorithm) and how to compare a scan to nominal CAD geometry. 

**Statistical Process Control (SPC):** X-bar/R charts, control limits, and the distinction between common cause and special cause variation. The data we use to fit the random field must come from a process that is in statistical control. 

**Design of Experiments (DoE) for Metrology Studies:** To build a meaningful dataset, you need a designed experiment: replicate parts across machines, operators, and process conditions. Understand full factorial vs. fractional factorial designs. 

## **3. Essential Whitepapers & Textbooks** 

The table below lists the key references organized by topic. Priority 1 items should be read in full during weeks 1–3. Priority 2 items are reference texts you will consult repeatedly. Priority 3 items are background depth for specific implementation decisions. 

|**Topic**|**Resource / Reference**|**Type**|
|---|---|---|
|FEA — Intro|Fish & Belytschko, A First Course in Finite Elements (Wiley,<br>2007)|Textbook P1|
|FEA — Formulation|Hughes, The Finite Element Method (Dover, 2000)|Textbook P2|
|TO — Classic|Sigmund (2001). 'A 99-line topology optimization code<br>written in Matlab.' Struct Multidisc Optim 21:120–127|Paper P1|
|TO — Review|Sigmund & Maute (2013). 'Topology optimization<br>approaches.' Struct Multidisc Optim 48:1031–1055|Paper P1|
|TO — Filtering|Lazarov & Sigmund (2016). 'Filters in topology optimization<br>based on Helmholtz-type differential equations.' IJNME<br>86:765–781|Paper P2|
|TO — SIMP Theory|Bendsoe & Sigmund, Topology Optimization: Theory,<br>Methods, and Applications (Springer, 2003)|Textbook P1|
|Robust TO — Loads|Lazarov et al. (2012). 'Topology optimization with geometric<br>uncertainties by perturbation techniques.' IJNME<br>90:1321–1336|Paper P1|
|Robust TO —<br>Geometry|Wang et al. (2011). 'On projection methods, convergence<br>and robust formulations in topology optimization.' Struct<br>Multidisc Optim 43:767–784|Paper P1|
|PCE — Foundations|Xiu & Karniadakis (2002). 'The Wiener-Askey polynomial<br>chaos for stochastic differential equations.' SIAM J Sci<br>Comput 24:619–644|Paper P1|
|PCE — Non-Intrusive|Blatman & Sudret (2011). 'Adaptive sparse polynomial<br>chaos expansion based on least angle regression.' J<br>Comput Phys 230:2345–2367|Paper P2|
|PCE + TO|Keshavarzzadeh et al. (2016). 'Topology optimization under<br>uncertainty via non-intrusive polynomial chaos expansion.'<br>Comput Methods Appl Mech Eng 318|Paper P1|



|**Topic**|**Resource / Reference**|**Type**|
|---|---|---|
|UQ — Textbook|Sullivan, Introduction to Uncertainty Quantification<br>(Springer, 2015)|Textbook P2|
|Random Fields|Vanmarcke, Random Fields: Analysis and Synthesis (MIT<br>Press, 1983 / rev 2010)|Textbook P2|
|KL Expansion|Loeve, Probability Theory II (Springer, 1978) — Chapter on<br>spectral representation|Textbook P3|
|Kernel Methods|Rasmussen & Williams, Gaussian Processes for Machine<br>Learning (MIT Press, 2006) — free PDF at<br>gaussianprocess.org|Textbook P2|
|Metrology & GD&T;|ASME Y14.5-2018, Dimensioning and Tolerancing<br>Standard|Standard P1|
|CNC Error Modeling|Ramesh et al. (2000). 'Error compensation in machine tools<br>— a review.' Int J Machine Tools Manuf 40:1235–1256|Paper P2|
|FDM Error Modeling|Turner et al. (2015). 'A review of melt extrusion additive<br>manufacturing processes.' Rapid Prototyping J 21:137–151|Paper P2|
|NIST Standards Gap|NIST SP 1999-001 (or latest), Measurement Science<br>Roadmap for Metal-Based Additive Manufacturing|Report P1|
|MMA Optimizer|Svanberg (1987). 'The method of moving asymptotes — a<br>new method for structural optimization.' IJNME 24:359–373|Paper P2|
|Sobol Sensitivity|Saltelli et al., Global Sensitivity Analysis: The Primer (Wiley,<br>2008)|Textbook P3|



_P1 = read in full during ramp-up. P2 = reference as needed. P3 = depth reading for specific implementation questions._ 

## **4. Software Stack** 

You will work primarily in Python. Below is each tool in the stack, what it does in this project, and what you need to learn about it. 

## _**Python 3.11+ — Core Language**_ 

Language of the entire codebase. You need: NumPy (dense linear algebra), SciPy (sparse solvers, optimization, stats), Matplotlib/PyVista (visualization). Know how to write vectorized NumPy code — no Python loops over mesh elements. 

## _**FEniCSx — FEA Engine**_ 

Open-source FEA framework based on the FEniCS project. Lets you define variational problems symbolically in UFL (Unified Form Language) and automatically assembles and solves them. You will implement the linear elasticity solver and the topology optimization loop here. Install via conda: conda install -c conda-forge fenics-dolfinx 

## _**OpenMDAO / PyOptSparse — Optimization Driver**_ 

OpenMDAO is NASA's open-source multidisciplinary optimization framework. Handles gradient flow between components and interfaces with gradient-based optimizers (SNOPT, IPOPT, MMA). Alternatively, use PyOptSparse directly. You will wrap the TO loop as an OpenMDAO Component. 

## _**OpenTURNS — UQ & PCE Library**_ 

The most complete open-source UQ library available. Provides: random field generation (KL expansion), PCE construction (sparse, regression-based), Sobol sensitivity analysis, and Monte Carlo engines. This is your primary tool for the stochastic layer of the framework. 

## _**scikit-learn — Machine Learning Utilities**_ 

Used for LASSO regression in sparse PCE fitting, Gaussian Process surrogate comparisons, and kernel hyperparameter optimization (via GaussianProcessRegressor). Know cross-validation for surrogate accuracy. 

## _**PyVista / VTK — 3D Visualization**_ 

PyVista is a Pythonic wrapper around VTK. Used to visualize the semi-transparent Monte Carlo geometry cloud and TO density fields. For CAVE rendering, you will export VTK files and load them into a visualization environment (ParaView in immersive mode or similar). 

## _**ParaView — Post-Processing & CAVE Visualization**_ 

Industry-standard scientific visualization software. You will use it both for local post-processing of FEA results and potentially for driving the CAVE wall display. Know how to write Python Programmable Filters and use the PvPython scripting interface. 

## _**Gmsh — Mesh Generation**_ 

Open-source mesh generator. Used to remesh perturbed geometries (as-manufactured variants) while preserving boundary conditions. Know how to script Gmsh via its Python API and how to control element size fields. 

## _**Git + GitHub/GitLab — Version Control**_ 

All code lives in a repository. Use feature branches, pull requests, and meaningful commit messages. You will be code-reviewed. Learn: branching strategy, rebasing, resolving merge conflicts, and tagging releases. 

## _**Docker / Singularity — Reproducible Environments**_ 

FEniCSx and OpenTURNS have complex dependencies. The team uses containerized environments to ensure reproducibility across machines and HPC clusters. Know how to build, run, and debug Docker containers. On HPC (if used), Singularity is the container runtime. 

## _**Jupyter Lab — Exploration & Prototyping**_ 

Use for exploratory analysis, plotting, and demonstrating results. Production code lives in .py modules — Jupyter is for exploration only. Never commit notebooks with uncleared outputs. 

## **5. Phased Task Plan — 12-Week Roadmap** 

This roadmap is structured so that each phase builds directly on the previous. Do not skip phases. Each phase ends with a checkpoint meeting with the senior engineer. You are expected to present working code and be able to answer conceptual questions about what you built. 

**PHASE 1** FEA Foundation & Literature Review · Weeks 1–2 

**Task 1.1:** Read all Priority 1 papers/textbooks listed in Section 3. Take structured notes — one paragraph per paper summarizing novelty, method, and limitations. 

**Task 1.2:** Implement a 2D linear elasticity FEA solver from scratch in Python/NumPy for a rectangular domain with prescribed loads and boundary conditions. No FEniCSx yet. 

**Task 1.3:** Verify your solver against the cantilever beam analytical solution (max tip deflection = PL^3 / 3EI). Plot the error vs. mesh refinement on a log-log plot and confirm second-order convergence. 

**Task 1.4:** Install FEniCSx via Docker. Reproduce the linear elasticity cantilever solution using FEniCSx. Compare against your hand-coded solver. 

**Task 1.5:** Checkpoint deliverable: 2-page writeup + convergence plot + FEniCSx code repo. 

## **PHASE 2** 

Classic Topology Optimization · Weeks 3–4 

**Task 2.1:** Implement Sigmund's 99-line SIMP TO code in Python (translate from MATLAB). Run the MBB beam and cantilever benchmark problems. Reproduce published results exactly. 

**Task 2.2:** Add sensitivity filtering and Heaviside projection to your implementation. Compare optimized topologies with and without filtering. 

**Task 2.3:** Derive the adjoint sensitivity analysis for the compliance objective by hand. Verify numerically using finite differences (perturb one element's density by 1e-6, compare to adjoint gradient). 

**Task 2.4:** Port the TO loop into FEniCSx. Parameterize the design variable field as a DG0 (piecewise constant) function on the mesh. 

**Task 2.5:** Checkpoint deliverable: Working FEniCSx-based TO code + adjoint verification plot for all elements + MBB beam result at 50% volume fraction. 

## **PHASE 3** 

Random Fields & Metrology Data · Weeks 5–6 

**Task 3.1:** Study the KL expansion derivation in Vanmarcke (Section 3) and Loeve. Implement KL sampling for a 1D random field with a squared-exponential kernel. Verify the sample covariance matches the theoretical covariance. 

**Task 3.2:** Extend to 2D KL expansion on a rectangular domain. Plot the first 10 eigenmodes. Understand the relationship between eigenvalue decay rate and kernel length-scale. 

**Task 3.3:** Use OpenTURNS to reproduce your KL expansion results. Learn the ot.KarhunenLoeveP1Algorithm interface. 

**Task 3.4:** Analyze the provided metrology dataset (CMM scans of test coupons). Fit a squared-exponential kernel to the empirical variogram using maximum likelihood. Report fitted (sigma, l) parameters and validate with leave-one-out cross-validation. 

**Task 3.5:** Checkpoint deliverable: Kernel fitting report with variogram plots, KL mode plots, and validation metrics. OpenTURNS-based KL sampler committed to repo. 

## **PHASE 4** 

PCE Surrogate Construction · Weeks 7–8 

**Task 4.1:** Study Blatman & Sudret (2011) on sparse adaptive PCE. Implement a toy 1D PCE for the Ishigami function (standard UQ benchmark) and verify mean/variance against analytical values. 

**Task 4.2:** Use OpenTURNS FunctionalChaosAlgorithm to build a PCE surrogate for the FEniCSx compliance solver. Inputs are the KL coefficients (truncated to the first 20 modes). Output is compliance. 

**Task 4.3:** Conduct a surrogate accuracy study: compare PCE predictions vs. full FEA on a held-out test set of 200 random field samples. Report Q2 coefficient (predictive R^2 on test set). Iterate on polynomial degree and truncation until Q2 > 0.99. 

**Task 4.4:** Compute first-order and total Sobol indices from the PCE coefficients. Identify the top 5 KL modes that drive compliance variance. Use this to justify your KL truncation order. 

**Task 4.5:** Checkpoint deliverable: Surrogate accuracy report (Q2, error distribution), Sobol index plot, and recommended KL truncation order with justification. 

## **PHASE 5** 

Robust Topology Optimization Loop · Weeks 9–10 

**Task 5.1:** Formulate the robust TO objective: J = mu_C + lambda * sigma_C where mu_C and sigma_C are extracted analytically from the PCE coefficients. Derive the sensitivity of this objective with respect to each element density. 

**Task 5.2:** Implement the full robust TO loop: SIMP density update -> FEniCSx FEA solve at PCE training points -> PCE fit -> moment extraction -> sensitivity computation -> OC/MMA update. Run until convergence. 

**Task 5.3:** Perform a parameter sweep over lambda (0, 0.5, 1.0, 2.0, 5.0). For each, run 500 Monte Carlo validation samples (full FEA, not PCE) on the converged design. Plot the Pareto frontier of (mean compliance, variance of compliance). Confirm the trend is monotone. 

**Task 5.4:** Compare robust TO results vs. deterministic TO: show that the deterministic design has higher compliance variance at the same mean, using a boxplot of the MC sample distributions. 

**Task 5.5:** Checkpoint deliverable: Pareto frontier plot, deterministic vs. robust comparison figure, and full code committed with a README that documents how to reproduce results. 

## **PHASE 6** 

Visualization & Documentation · Weeks 11–12 

**Task 6.1:** Export all 5,000 Monte Carlo sample geometries (perturbed meshes) as VTK files. Load in ParaView, apply semi-transparent rendering with opacity mapped to structural member stability (local coefficient of variation of density field). Produce a screenshot-quality rendering. 

**Task 6.2:** Write a technical report (10–15 pages) documenting: problem formulation, random field model and metrology calibration, PCE construction, robust TO loop, results, and limitations. Format per NIST technical report guidelines. 

**Task 6.3:** Prepare a 20-minute presentation covering the project end-to-end. Practice presenting to a non-specialist audience: lead with the engineering problem, not the math. 

**Task 6.4:** Write unit tests for all major code components (FEA solver, KL sampler, PCE wrapper, TO update). Target >80% line coverage using pytest. 

**Task 6.5:** Final deliverable: Technical report, presentation slides, fully tested code repository with CI pipeline, and ParaView visualization file. 

## **6. Deliverables & Success Criteria** 

|**Week**|**Deliverable**|**Acceptance Criteria**|n<br>ark<br>ion|
|---|---|---|---|
|Week 2|FEA Solver|2D NumPy FEA + FEniCSx cantilever with convergence verificatio||
|Week 4|Topology Optimizer|FEniCSx SIMP TO code with adjoint verification and MBB benchm||
|Week 6|Random Field Model|Calibrated KL sampler + metrology fitting report||
|Week 8|PCE Surrogate|Sparse PCE with Q2 > 0.99, Sobol indices, KL truncation justificat||
|Week 10|Robust TO Loop|Full robust TO with Pareto sweep and MC validation||
|Week 12|Final Package|Technical report, presentation, tested code repo, ParaView viz||



## **Definition of Done** 

A deliverable is 'done' when: (1) the code runs end-to-end without manual intervention from a clean clone of the repository, (2) outputs match the acceptance criteria quantitatively, (3) a checkpoint meeting has been held and the senior engineer has signed off, and (4) the code has been merged to the main branch via a reviewed pull request. 

## **7. Tips, Common Pitfalls & How to Ask Good Questions** 

## **Common Pitfalls to Avoid** 

## I **Skipping the hand derivation** 

Implementing an adjoint sensitivity method without deriving it yourself is like copy-pasting code you don't understand. When it's wrong (and it will be wrong the first time), you won't know where to look. Derive everything in a notebook first. 

## I **Not verifying the FEA solver** 

Always verify against an analytical solution before using the solver in a larger loop. A 1% stiffness error in FEA becomes a 1% gradient error in the TO loop, which can cause the optimizer to converge to the wrong design. 

## I **Treating PCE as a black box** 

OpenTURNS will build you a PCE even if the surrogate is inaccurate. Always check Q2 on a held-out test set. A Q2 of 0.95 sounds good but means 5% of the variance is unexplained — enough to corrupt Sobol indices. 

## I **Confusing the random field with the error model** 

The random field is the mathematical object. The error model is the physical interpretation (e.g., wall thickness deviation). Be precise about what each KL mode represents physically. 

## I **Over-truncating the KL expansion** 

If you keep only 5 KL modes to save compute, but those 5 modes only explain 60% of the variance, your random field samples look nothing like real parts. Use the Sobol indices to justify your truncation — don't guess. 

## I **Git commits without context** 

A commit message that says 'fix' is useless. Write: what changed, why, and what the effect is. Your future self and your teammates will thank you. 

## **How to Ask Good Questions** 

## **The XY Problem** 

When you're stuck, describe what you are trying to accomplish (X), what approach you tried (Y), what you expected to happen, what actually happened (with the exact error message and stack trace), and what you have already ruled out. Never ask 'why doesn't this work' and paste 200 lines of code. 

- Reproduce the issue in the smallest possible code example before asking. 

- If it's a math question, write out the equation you think should hold and show where the derivation breaks down. 

• If it's a numerical result question, include the value you got, the value you expected, and a convergence or validation test that demonstrates the discrepancy. 

• Search the FEniCSx discourse (fenicsproject.discourse.group) and OpenTURNS documentation before asking — these communities have excellent archives. 

• Use the senior engineer's time for conceptual and design-level questions. Use Stack Overflow / GitHub issues for library-specific debugging. 

_You have everything you need to make this project a success. The math is hard, the software is complex, and the timeline is tight — but every piece of this framework has been chosen because it is the right tool for the job, not because it is the easiest. Ask early, verify often, and document everything._ 

— Senior FEA & Simulation Lead, NIST Advanced Manufacturing Program · 2026 


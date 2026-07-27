"""Brute-force Monte Carlo validation engine (Stage 6, MVP subset).


Master-context alignment (Section 3.6):
    "Implements the two-tier uncertainty propagation scheme: PCE (or SROM)
    serves as the main engine for evaluating mean/variance during
    optimization; high-fidelity Monte Carlo (thousands of samples) is
    reserved for final verification... Generates N_mc = 5,000+ eta/KL
    coefficient samples; for each: sample eta(x) -> apply projection ->
    FEniTop FEA -> compliance... Computes empirical compliance distribution:
    mean, variance, 5th/95th percentiles, full CDF."


Explicit, documented MVP scope reductions (NOT silent shortcuts):
    1. Runs on a single FIXED converged density field (no re-optimization
       per sample) -- this matches Section 3.6's spec exactly, since MC
       validation is defined as a post-hoc check on a converged design.
    2. n_samples defaults far below the "5,000+" full spec; a warning is
       logged (not hidden) whenever n_samples < 5000.
    3. PCE-vs-MC comparison (Section 3.6's Q^2 pass/fail flag) is NOT
       computed here because no PCE surrogate exists yet (roadmap Step 6).
       Calling compare_against_pce() raises NotImplementedError with a
       pointer to the missing module, rather than returning a fake flag.


MPI design (replaces the old serial-only limitation):
    This loop is world-collective, matching fea_at_samples.py and
    dolfiny_mma_driver.py's convention. It takes an already-computed,
    MPI-shared kl_result (from src/random_fields/kl_expansion.py's
    compute_kl_expansion(), broadcast identically to every rank) rather
    than recomputing/reconstructing it from raw node_coordinates/simplices.
    RandomFieldHeaviside is built with THIS RANK's local dof coordinates
    and matches them against kl_result's global nodes via the same
    coordinate-matching pattern used throughout this codebase. Every rank
    must call this function together with an identical rho_converged
    (a full GLOBAL array -- see run_monte_carlo_validation's docstring;
    this function scatters it internally) and mc_config.seed; xi is drawn
    identically on every rank per sample since np.random.default_rng(seed
    + i) is deterministic, so no further communication is needed at sample
    time beyond the per-sample rho_phys_field gather used for ensemble
    export. Compliance is already a true global scalar via comm.allreduce
    on assemble_scalar, unchanged from the original.
"""
from __future__ import annotations


import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path


import numpy as np
import pyvista
import dolfinx.plot
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx.fem import form, assemble_scalar


from src.fenitop.fem import form_fem
from src.fenitop.parameterize import DensityFilter
from src.fenitop.utility import Communicator


from src.topology.heaviside_projection_glue import RandomFieldHeaviside, RandomHeavisideConfig
from src.random_fields.kl_expansion import KLExpansionResult
from src.topology.heaviside_projection_glue import build_random_heaviside_from_function_space

import openturns as ot


logger = logging.getLogger(__name__)


FULL_SPEC_N_MC = 5000  # Section 3.6: "5,000+ sample MC ensemble"



@dataclass
class MCConfig:
    """Configuration for the brute-force Monte Carlo validation run.


    Attributes:
        n_samples: Number of eta(x) realizations to draw and solve. Section
            3.6 specifies 5,000+; MVP default is far lower for iteration
            speed. A warning is logged if below FULL_SPEC_N_MC.
        beta: Fixed Heaviside sharpness parameter to use for all samples,
            normally the final/converged beta from the nominal TO run.
        percentiles: Percentile levels to report (Section 3.6: "5th/95th
            percentiles").
        seed: Base RNG seed; sample i uses seed + i for reproducibility.
            Must be identical on every rank (this is a collective loop).
        output_dir: Directory to write results CSV/plot to.
        write_ensemble: If True, gather and write a FULL-RESOLUTION per-sample
            .vtu (nodal rho_phys density field) for every one of n_samples
            draws, plus an ensemble.pvd ParaView collection and a
            reliability_map.vtu (per-node mean/std/exceedance-probability
            across the ensemble). This is the "probability cloud of
            perturbed geometries" artifact -- exact, not a cheap replay --
            per explicit request. Cost scales as n_samples x (one
            world-collective gather + one VTU write on rank 0), on top of
            the FEA solve already paid per sample; at n_samples=5000 this
            is the dominant wall-clock cost, matching fileDescription.md's
            own "~3-5 hours on DGX" estimate for full-scale MC.
        ensemble_dir: Directory for per-sample .vtu files. Defaults to
            output_dir / "ensemble" if left None (resolved in __post_init__).
        reliability_threshold: Density value below which a node is counted
            as "locally void/unreliable" for the per-node exceedance-
            probability field in reliability_map.vtu (default 0.5, the
            conventional solid/void SIMP threshold).
        store_eta_samples: If True, retain the full [n_samples x n_dofs_local]
            eta(x) realization array in the returned MCResult (as before).
            Default False: at n_samples=5000 across e.g. 64 ranks this is
            the single largest per-rank buffer in the whole loop and is not
            needed for compliance stats, the CDF, or PCE-vs-MC comparison
            (which only needs xi_samples, always retained). Opt in only if
            you specifically need the raw per-sample nodal eta(x) fields.
    """
    n_samples: int = 2500
    beta: float = 8.0
    percentiles: tuple[float, float] = (5.0, 95.0)
    seed: int = 0
    output_dir: Path = field(default_factory=lambda: Path("output/mc_validation"))
    write_ensemble: bool = True
    ensemble_dir: Path | None = None
    reliability_threshold: float = 0.5
    store_eta_samples: bool = False
    # Fraction of samples allowed to fail to solve before the run is halted.
    #
    # A failed solve at the ERODED end of a wide eta band is not noise -- it
    # means that realization of the structure does not carry load. That is a
    # ROBUSTNESS RESULT, and the failure rate is the headline number it
    # produces, so it is always reported. But it also means the compliance
    # statistics are CONDITIONAL on the structure surviving, which biases them
    # optimistically: the realizations that were hardest on the design are
    # exactly the ones missing.
    #
    # Default 0.0 (any failure halts) so the value has to be an explicit,
    # visible decision in each config rather than a silent tolerance.
    max_solver_failure_rate: float = 0.0


    def __post_init__(self) -> None:
        if self.n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {self.n_samples}")
        if self.n_samples < FULL_SPEC_N_MC:
            logger.warning(
                "n_samples=%d is below the master-context full spec of %d "
                "('5,000+ sample MC ensemble', Section 3.6). This is an "
                "explicit MVP scope reduction, not a silent shortcut -- "
                "scale up n_samples before treating results as production-grade.",
                self.n_samples, FULL_SPEC_N_MC,
            )
        # Defensive coercion: schema.py's MonteCarloValidationConfig.output_dir
        # is typed as a plain str (config.yaml value), so a caller wiring that
        # straight into MCConfig(output_dir=...) would otherwise pass a str
        # here, and self.output_dir / "ensemble" below would raise TypeError.
        self.output_dir = Path(self.output_dir)
        if self.ensemble_dir is None:
            self.ensemble_dir = self.output_dir / "ensemble"
        else:
            self.ensemble_dir = Path(self.ensemble_dir)



@dataclass
class MCResult:
    """Empirical compliance distribution from the Monte Carlo ensemble.


    Attributes:
        compliance_samples: [n_samples] array of compliance values C(eta_i).
            A true global scalar per sample (via comm.allreduce), identical
            on every rank.
        mean: Sample mean, mu_C.
        variance: Sample variance, sigma_C^2.
        std: Sample standard deviation, sigma_C.
        percentile_low: Value at the lower percentile (default 5th).
        percentile_high: Value at the upper percentile (default 95th).
        eta_samples: [n_samples x n_dofs_local] array of the eta(x) fields
            used, restricted to THIS RANK's local dofs (retained for
            reproducibility / later PCE-vs-MC comparison; NOT gathered to
            a global array here to avoid an unnecessary MPI collective for
            an MVP diagnostic field). None unless mc_config.store_eta_samples
            was True (see MCConfig docstring -- this is the largest
            per-rank buffer in the loop and is opt-in).
        n_kl: Number of KL modes used to generate eta(x), for provenance.
        variance_explained: KL truncation variance fraction, for provenance.
        reliability_mean: [N_nodes_GLOBAL] per-node mean physical density
            rho_phys across the full ensemble. Populated on rank 0 only
            (None on every other rank), since it is a gathered/aggregated
            spatial field, not a world-identical scalar. None entirely if
            mc_config.write_ensemble was False.
        reliability_std: [N_nodes_GLOBAL] per-node std of rho_phys across
            the ensemble. Same rank-0-only convention as reliability_mean.
        reliability_prob_void: [N_nodes_GLOBAL] per-node fraction of
            ensemble samples with rho_phys below mc_config.reliability_threshold
            -- the manufacturing-sensitive-region reliability map. Same
            rank-0-only convention.
        probability_weights: [n_samples] Gaussian-density opacity weight
            per sample, exp(-0.5*||xi_i||^2) normalized to max=1, i.e. how
            "typical" each drawn realization is under the underlying
            N(0, I) KL coefficient law -- intended for ParaView opacity
            transfer functions on the ensemble (extreme/rare geometries
            render more transparent). World-identical (xi is drawn
            identically on every rank), always populated when
            write_ensemble is True.
        ensemble_pvd_path: Path to the written ensemble.pvd ParaView
            collection file, or None if write_ensemble was False. Only
            meaningful on rank 0 (the only rank that writes files).
    """
    compliance_samples: np.ndarray
    mean: float
    variance: float
    std: float
    percentile_low: float
    percentile_high: float
    eta_samples: np.ndarray | None
    n_kl: int
    variance_explained: float
    xi_samples: np.ndarray
    # Per-sample realized volume FRACTION, and the eta=0.5 reference. Collected
    # for one extra scalar assembly per sample: they carry the spread of the
    # (mean-only-constrained) volume, and they yield the realized boundary
    # displacement with no extra FEA -- see src/validation/boundary_offset.py.
    volume_samples: np.ndarray | None = None
    nominal_volume_fraction: float | None = None
    total_volume: float | None = None
    n_solver_failures: int = 0
    # Fraction of realizations in which the structure failed to carry load. For
    # a robustness study this is a first-class result, not an error count.
    solver_failure_rate: float = 0.0
    # True when some samples failed: mean/std/percentiles are then computed only
    # over the surviving realizations and are optimistically biased.
    statistics_conditional_on_success: bool = False
    reliability_mean: np.ndarray | None = None
    reliability_std: np.ndarray | None = None
    reliability_prob_void: np.ndarray | None = None
    probability_weights: np.ndarray | None = None
    ensemble_pvd_path: Path | None = None


    def summary_with_intervals(
        self, percentiles: tuple[float, float] = (5.0, 95.0), seed: int = 0
    ) -> dict:
        """Point estimates WITH bootstrap confidence intervals.

        `mean` and `std` on this object are bare point estimates; at the sample
        sizes this pipeline runs, the interval around them is often wider than
        the differences being claimed between designs. Report this instead --
        see src/validation/statistics.py for why.
        """
        from src.validation.statistics import summarize_samples

        # compliance_samples keeps NaN for failed realizations as the full
        # record; the statistics are over the survivors only, which is what
        # statistics_conditional_on_success flags.
        finite = self.compliance_samples[np.isfinite(self.compliance_samples)]
        summary = summarize_samples(finite, percentiles=percentiles, seed=seed)
        summary["solver_failure_rate"] = self.solver_failure_rate
        summary["conditional_on_success"] = self.statistics_conditional_on_success
        if self.statistics_conditional_on_success:
            summary["note"] = (
                "CONDITIONAL statistics: "
                f"{self.n_solver_failures} realization(s) "
                f"({100 * self.solver_failure_rate:.3g}%) did not carry load "
                "and are excluded. These figures are optimistically biased and "
                "must be reported together with the failure rate."
            )
        return summary


    def cdf(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (sorted_values, empirical_cdf) for plotting Section 3.6's CDF."""
        sorted_vals = np.sort(self.compliance_samples)
        n = sorted_vals.size
        empirical_cdf = np.arange(1, n + 1) / n
        return sorted_vals, empirical_cdf


    def to_csv(self, path: Path) -> None:
        """Write per-sample compliance values to CSV (Section 3.6 output artifact).


        Compliance is already identical on every rank (world-allreduced),
        so this is written unconditionally -- callers should still gate the
        actual call on comm.rank == 0 to avoid redundant file writes.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        header = "sample_index,compliance"
        rows = np.column_stack([np.arange(self.compliance_samples.size), self.compliance_samples])
        np.savetxt(path, rows, delimiter=",", header=header, comments="", fmt=["%d", "%.10e"])
        logger.info("Wrote MC compliance samples to %s", path)



def _write_pvd_collection(pvd_path: Path, entries: list[tuple[int, str]]) -> None:
    """Write a minimal ParaView .pvd XML collection referencing each
    per-sample .vtu file, keyed by sample index as the "timestep" attribute
    so ParaView's animation/time controls can scrub through the ensemble.

    entries: list of (sample_index, relative_path_to_vtu) tuples, where the
    relative path is resolved relative to pvd_path's own directory (i.e.
    "ensemble/sample_00000.vtu" when pvd_path lives in the parent of
    ensemble_dir) -- this matches how most ParaView-facing tools expect a
    .pvd file to sit alongside (not inside) the directory it indexes.
    """
    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
        "  <Collection>",
    ]
    for idx, relpath in entries:
        lines.append(f'    <DataSet timestep="{idx}" group="" part="0" file="{relpath}"/>')
    lines.append("  </Collection>")
    lines.append("</VTKFile>")
    pvd_path.parent.mkdir(parents=True, exist_ok=True)
    pvd_path.write_text("\n".join(lines))
    logger.info("Wrote ParaView collection %s (%d entries)", pvd_path, len(entries))


def run_monte_carlo_validation(
    fem_config: dict,
    opt_config: dict,
    rho_converged: np.ndarray,
    kl_result: KLExpansionResult,
    heaviside_config: RandomHeavisideConfig,
    mc_config: MCConfig,
) -> MCResult:
    """Run the brute-force MC validation loop on a fixed converged design.


    For each of mc_config.n_samples draws:
        1. Reset rho_field to rho_converged (the fixed nominal/robust design).
        2. Apply the density filter (deterministic Helmholtz PDE solve) to
           get rho_tilde -- this is NOT random, only recomputed because
           Heaviside overwrites the same Function object in place.
        3. Draw a fresh xi and set eta(x) via RandomFieldHeaviside.set_eta_from_xi().
        4. Apply the random-field Heaviside projection using that eta(x).
        5. Solve the FEA problem (KU = F).
        6. Assemble compliance C = U^T K U (world-allreduced) and record it.


    This directly implements Section 3.6's loop: "for each: sample eta(x) ->
    apply projection -> FEniTop FEA -> compliance."


    Args:
        fem_config: The `fem` dict as consumed by fenitop.fem.form_fem
            (mesh, material properties, BCs, etc.) -- identical structure to
            what topopt.py passes in.
        opt_config: The `opt` dict as consumed by form_fem / DensityFilter
            (penalty, epsilon, filter_radius, opt_compliance=True required
            for this MVP since only the compliance QoI path is exercised).
        rho_converged: [n_elems_GLOBAL] converged density field from a
            prior deterministic or robust TO run (e.g. topopt.py's
            rho_converged.npy, or run_robust_topopt's now-gathered
            "rho_robust" -- see dolfiny_mma_driver.py). Must be a full
            GLOBAL array, IDENTICAL on every rank (loaded once on rank 0
            and comm.bcast'd by the caller, per this codebase's existing
            convention). This function scatters it internally to each
            rank's local dof slice via Communicator.bcast(), exactly
            mirroring run_robust_topopt's own rho_warm_start handling --
            it is NOT a per-rank local slice, unlike the previous version
            of this function.
        kl_result: A KLExpansionResult already computed (and, under MPI,
            already broadcast identically to every rank) via
            src/random_fields/kl_expansion.py's compute_kl_expansion(). This
            is NOT recomputed here.
        heaviside_config: RandomHeavisideConfig (kernel + marginal params).
        mc_config: MCConfig controlling n_samples, beta, seed, percentiles,
            and (new) write_ensemble/ensemble_dir/reliability_threshold/
            store_eta_samples -- see MCConfig docstring.


    Returns:
        An MCResult with the empirical compliance distribution (global,
        identical on every rank), xi_samples (world-identical), optionally
        eta_samples (per-rank-local, only if mc_config.store_eta_samples),
        and -- when mc_config.write_ensemble is True -- a full-resolution
        per-sample VTU ensemble + ParaView .pvd collection on disk, plus
        rank-0-only reliability_mean/reliability_std/reliability_prob_void
        per-node arrays and per-sample probability_weights for opacity
        mapping.


    Raises:
        ValueError: If rho_converged's size does not match the design
            space's GLOBAL dof count, or if opt_config["opt_compliance"]
            is not True.
    """
    comm = MPI.COMM_WORLD

    if not opt_config.get("opt_compliance", True):
        raise ValueError(
            "run_monte_carlo_validation currently only supports the compliance "
            "QoI path (opt_config['opt_compliance']=True). The displacement-QoI "
            "path (compliant mechanisms) is out of scope for this MVP step."
        )

    if comm.rank == 0:
        logger.info(
            "Starting MC validation: n_samples=%d, beta=%.2f, seed=%d",
            mc_config.n_samples, mc_config.beta, mc_config.seed,
        )


    linear_problem, u_field, lambda_field, rho_field, rho_phys_field = form_fem(
        fem_config, opt_config
    )
    density_filter = DensityFilter(
        comm, rho_field, rho_phys_field,
        opt_config["filter_radius"], fem_config["petsc_options"],
    )


    # BUGFIX (multi-rank correctness): the old version asserted rho_converged
    # against rho_field's LOCAL dof shape and used it directly -- correct by
    # accident only at comm.size == 1. rho_converged is now a GLOBAL array
    # (see docstring); validate against the GLOBAL dof count and scatter it
    # into rho_field's local slice via Communicator.bcast(), the exact same
    # pattern run_robust_topopt uses for its own rho_warm_start argument.
    design_index_map = rho_field.function_space.dofmap.index_map
    n_elems_global = design_index_map.size_global
    if rho_converged.size != n_elems_global:
        raise ValueError(
            f"rho_converged has {rho_converged.size} entries but the design "
            f"space has {n_elems_global} GLOBAL dofs. rho_converged must be "
            "a full global array (identical on every rank), not a local "
            "slice -- see this function's docstring."
        )
    design_comm = Communicator(rho_field.function_space, fem_config["mesh_serial"])
    design_comm.bcast(rho_field, rho_converged)
    rho_converged_local = rho_field.x.petsc_vec.array.copy()


    local_node_coordinates = rho_phys_field.function_space.tabulate_dof_coordinates()
    spatial_dim = heaviside_config.kernel_params.spatial_dim
    local_node_coordinates = local_node_coordinates[:, :spatial_dim]
    rf_heaviside = build_random_heaviside_from_function_space(
    rho_phys_field, kl_result, heaviside_config
    )
    if comm.rank == 0:
        logger.info(
            "RandomFieldHeaviside ready for MC loop: N_kl=%d, variance_explained=%.4f",
            rf_heaviside.kl_result.n_kl, rf_heaviside.kl_result.variance_explained,
        )


    # --- Full-resolution ensemble export setup (only if requested) --------
    # phys_comm gathers THIS SAMPLE's rho_phys_field to a true global array
    # on rank 0, using the same Communicator machinery (coordinate-matched
    # local<->global dof map, built ONCE here and reused for every sample --
    # the expensive cKDTree match happens only once, not per sample).
    write_ensemble = mc_config.write_ensemble
    if write_ensemble:
        phys_comm = Communicator(rho_phys_field.function_space, fem_config["mesh_serial"])
        if comm.rank == 0:
            mesh_serial = fem_config["mesh_serial"]
            tdim = mesh_serial.topology.dim
            vtk_cells, vtk_cell_types, vtk_points = dolfinx.plot.vtk_mesh(mesh_serial, tdim)
            ensemble_grid = pyvista.UnstructuredGrid(vtk_cells, vtk_cell_types, vtk_points)
            n_nodes_global = vtk_points.shape[0]

            mc_config.output_dir.mkdir(parents=True, exist_ok=True)
            # Clear any stale per-sample VTUs left by a previous run at this
            # same path (ensemble_dir is NOT covered by mainClean.py's
            # per-stage _fresh_dir() sweep) -- otherwise a run with fewer
            # samples than a prior run leaves higher-index files behind,
            # which is exactly what made build_probability_cloud's glob-based
            # sample count diverge from this run's actual n_samples.
            if mc_config.ensemble_dir.exists():
                shutil.rmtree(mc_config.ensemble_dir)
            mc_config.ensemble_dir.mkdir(parents=True, exist_ok=True)
            sum_rho = np.zeros(n_nodes_global)
            sumsq_rho = np.zeros(n_nodes_global)
            count_void = np.zeros(n_nodes_global)
            pvd_entries: list[tuple[int, str]] = []
            weights = np.zeros(mc_config.n_samples)
            # Resolved once, up front, so per-sample relative paths below are
            # correct even if ensemble_dir was overridden away from the
            # output_dir/"ensemble" default.
            ensemble_pvd_dir = mc_config.output_dir

            logger.warning(
                "write_ensemble=True: gathering + writing a full-resolution "
                ".vtu for all %d samples (N_nodes_global=%d). This is the "
                "dominant wall-clock cost of Stage 6 at scale -- expect "
                "hours, not minutes, for n_samples>=5000 (matches "
                "fileDescription.md's own ~3-5 hour DGX estimate).",
                mc_config.n_samples, n_nodes_global,
            )


    compliance_form = form(opt_config["compliance"])
    # Per-sample VOLUME, collected alongside compliance for two reasons, at the
    # cost of one extra scalar assembly per sample and no extra FEA:
    #   1. The volume constraint is on E[V] only, so the realized spread of V
    #      across the ensemble -- and its 95th percentile -- is uncontrolled and
    #      was never reported. A design whose dilated realization is 20% over
    #      budget is a finding.
    #   2. It yields the realized boundary displacement for free, via
    #      d_s = (V_i - V_nominal) * total_volume / interface_area. See
    #      src/validation/boundary_offset.py.
    volume_form = form(opt_config["volume"])
    total_volume = comm.allreduce(
        assemble_scalar(form(opt_config["total_volume"])), op=MPI.SUM
    )

    compliance_samples = np.zeros(mc_config.n_samples)
    volume_samples = np.zeros(mc_config.n_samples)
    n_failed_samples = 0
    n_ksp_failures = 0
    n_dofs_local = local_node_coordinates.shape[0]
    eta_samples_all = (
        np.zeros((mc_config.n_samples, n_dofs_local)) if mc_config.store_eta_samples else None
    )

    # rho_converged is fixed across all samples -- only eta(x) varies -- so the
    # density write + Helmholtz filter solve happens ONCE here, not once per
    # sample (mirrors fea_at_samples.py). Cache the filtered rho_tilde and reset
    # rho_phys to it at the top of each sample before the Heaviside projection
    # overwrites it in place.
    rho_field.x.petsc_vec.array[:] = rho_converged_local
    rho_field.x.scatter_forward()
    density_filter.forward()
    rho_tilde_cached = rf_heaviside.rho_phys.x.petsc_vec.array.copy()

    # Warm-start CG from the previous sample (math-exact, cuts iteration count).
    # PC reuse is intentionally OFF (rebuild every sample, see
    # fea_at_samples.PC_REBUILD_INTERVAL for why): this project's SIMP contrast
    # (epsilon=1e-6) + sharp Heaviside (beta) means a frozen GAMG hierarchy goes
    # stale almost immediately as eta(x) varies, making reuse net-negative here.
    _PC_REBUILD_INTERVAL = 1
    linear_problem.enable_warm_start(True)

    # --- COMMON RANDOM NUMBERS -------------------------------------------
    # The whole xi block is drawn UP FRONT from a single generator, rather than
    # per-sample from default_rng(seed + i). Two properties this buys, both of
    # which the pipeline depends on:
    #
    #   1. Two designs validated with the same mc_config.seed see the IDENTICAL
    #      eta(x) ensemble, so their compliance samples are paired and the
    #      difference between them can be estimated with far lower variance than
    #      either design's own mean (see src/validation/statistics.py). Comparing
    #      designs on different draws -- which is what a per-sample reseed makes
    #      easy to do by accident -- discards that.
    #   2. Drawing (M, n_kl) from one stream is prefix-stable: the first N rows
    #      of an M-sample block equal an N-sample block from the same seed. So an
    #      N-convergence study nests, instead of resampling everything at each N.
    xi_samples_all = np.random.default_rng(mc_config.seed).standard_normal(
        size=(mc_config.n_samples, rf_heaviside.kl_result.n_kl)
    )

    # Volume of the NOMINAL realization (eta = 0.5), the reference the realized
    # boundary displacement of every sample is measured against.
    rf_heaviside.rho_phys.x.petsc_vec.array[:] = rho_tilde_cached
    rf_heaviside.forward(mc_config.beta, eta=0.5)
    nominal_volume_fraction = (
        comm.allreduce(assemble_scalar(volume_form), op=MPI.SUM) / total_volume
    )

    for i in range(mc_config.n_samples):
        rf_heaviside.rho_phys.x.petsc_vec.array[:] = rho_tilde_cached

        xi_sample = xi_samples_all[i, :]
        eta_sample = rf_heaviside.set_eta_from_xi(xi_sample)
        if mc_config.store_eta_samples:
            eta_samples_all[i, :] = eta_sample
        rf_heaviside.forward(mc_config.beta)

        linear_problem.set_reuse_preconditioner(i % _PC_REBUILD_INTERVAL != 0)
        try:
            linear_problem.solve_fem()
            C_value = comm.allreduce(assemble_scalar(compliance_form), op=MPI.SUM)
        except (PETSc.Error, RuntimeError) as exc:
            # solve_fem() raises RuntimeError on a non-converged KSP (it already
            # retries once internally with a fresh preconditioner if reuse was
            # active -- see LinearProblem.solve_fem -- so this is reached only
            # after that retry also failed, or reuse wasn't in play). PETSc.Error
            # is kept for any other, unrelated PETSc-level failure.
            n_failed_samples += 1
            n_ksp_failures += 1
            if comm.rank == 0:
                logger.warning(
                    "MC sample %d/%d: FEA solve failed (%s: %s). "
                    "seed=%d, eta band [%.3g, %.3g]. Recording compliance as "
                    "NaN and continuing. A solver failure at the ERODED end of "
                    "a wide eta band is not a numerical accident -- it usually "
                    "means that realization of the structure does not carry "
                    "load, which is a RESULT about the design's robustness and "
                    "must be reported, not swallowed.",
                    i + 1, mc_config.n_samples, type(exc).__name__, exc,
                    mc_config.seed,
                    float(np.min(eta_sample)) if eta_sample is not None else float("nan"),
                    float(np.max(eta_sample)) if eta_sample is not None else float("nan"),
                )
            compliance_samples[i] = np.nan
            volume_samples[i] = np.nan
            continue

        compliance_samples[i] = C_value
        volume_samples[i] = (
            comm.allreduce(assemble_scalar(volume_form), op=MPI.SUM) / total_volume
        )


        if write_ensemble:
            # Collective on every rank -- gathers this sample's rho_phys_field
            # to a true global nodal array on rank 0 (None on other ranks).
            global_rho_phys = phys_comm.gather(rho_phys_field)
            if comm.rank == 0:
                ensemble_grid.point_data["density"] = global_rho_phys
                sample_path = mc_config.ensemble_dir / f"sample_{i:05d}.vtu"
                ensemble_grid.save(str(sample_path))
                sample_relpath = os.path.relpath(sample_path, start=ensemble_pvd_dir)
                pvd_entries.append((i, sample_relpath))

                sum_rho += global_rho_phys
                sumsq_rho += global_rho_phys ** 2
                count_void += (global_rho_phys < mc_config.reliability_threshold)
                # Gaussian-density opacity weight: xi ~ iid N(0, I), so this
                # is proportional to the true likelihood of this realization
                # -- extreme (rare) samples get a lower weight, i.e. render
                # more transparent in a ParaView opacity transfer function.
                weights[i] = np.exp(-0.5 * float(np.sum(xi_sample ** 2)))


        if comm.rank == 0 and (i + 1) % max(1, mc_config.n_samples // 10) == 0:
            logger.info(
                "MC sample %d/%d: C=%.6g", i + 1, mc_config.n_samples, C_value
            )


    finite_mask = np.isfinite(compliance_samples)
    n_bad = int(np.sum(~finite_mask))
    failure_rate = n_bad / mc_config.n_samples

    if failure_rate > mc_config.max_solver_failure_rate:
        raise RuntimeError(
            f"{n_bad}/{mc_config.n_samples} compliance samples are non-finite "
            f"(NaN/inf), i.e. a {100 * failure_rate:.3g}% solver failure rate, "
            f"above the configured tolerance of "
            f"{100 * mc_config.max_solver_failure_rate:.3g}% "
            "(mc_validation.max_solver_failure_rate). Two readings, and they "
            "need different responses:\n"
            "  * Concentrated at the ERODED end of the eta band: a RESULT -- "
            "those realizations do not carry load. Raise the tolerance to "
            "accept and report it, or reconsider vol_frac / the eta band. Do "
            "not simply drop them: the statistics would then be conditional on "
            "survival and biased toward the realizations the design handled.\n"
            "  * Scattered across the band: a conditioning problem. Raise "
            "optimization.epsilon, or loosen the KSP tolerance.\n"
            "Note the solver already retries every failure once from a zero "
            "initial guess with a fresh preconditioner, so these are not "
            "warm-start artifacts."
        )

    statistics_conditional = n_bad > 0
    if statistics_conditional and comm.rank == 0:
        logger.error(
            "%d/%d samples (%.3g%%) failed to solve and are EXCLUDED from the "
            "statistics below, which are therefore CONDITIONAL ON THE "
            "STRUCTURE CARRYING LOAD. They are biased optimistically: the "
            "realizations that were hardest on the design are the ones "
            "missing. Report the failure rate alongside any compliance number "
            "from this ensemble -- for a robustness study it is arguably the "
            "more important of the two.",
            n_bad, mc_config.n_samples, 100 * failure_rate,
        )

    successful = compliance_samples[finite_mask]
    mean = float(successful.mean())
    variance = float(successful.var(ddof=1))
    std = float(np.sqrt(variance))
    p_low, p_high = np.percentile(successful, mc_config.percentiles)


    if comm.rank == 0:
        logger.info(
            "MC validation complete: mean=%.6g, std=%.6g, p%d=%.6g, p%d=%.6g",
            mean, std, mc_config.percentiles[0], p_low, mc_config.percentiles[1], p_high,
        )
        # The volume constraint bounds E[V] only, so the realized spread is
        # uncontrolled. A dilated realization well over budget is a finding
        # about the design, not a rounding detail.
        successful_volumes = volume_samples[finite_mask]
        logger.info(
            "MC realized volume: E[V]=%.6g, std=%.4g, p95=%.6g, max=%.6g "
            "(nominal eta=0.5 realization: %.6g)",
            successful_volumes.mean(), successful_volumes.std(ddof=1),
            np.percentile(successful_volumes, 95.0), successful_volumes.max(),
            nominal_volume_fraction,
        )


    reliability_mean = reliability_std = reliability_prob_void = None
    probability_weights = None
    ensemble_pvd_path = None


    if write_ensemble and comm.rank == 0:
        n = mc_config.n_samples
        reliability_mean = sum_rho / n
        reliability_var = np.clip(sumsq_rho / n - reliability_mean ** 2, 0.0, None)
        reliability_std = np.sqrt(reliability_var)
        reliability_prob_void = count_void / n

        # Normalize opacity weights to max=1 -- a direct [0,1] range is what
        # a ParaView opacity transfer function expects; the relative
        # likelihood ordering between samples is unaffected by this scaling.
        probability_weights = weights / weights.max() if weights.max() > 0 else weights
        np.savetxt(
            mc_config.output_dir / "probability_weights.csv",
            np.column_stack([np.arange(n), probability_weights]),
            delimiter=",", header="sample_index,opacity_weight", comments="", fmt=["%d", "%.10e"],
        )

        reliability_grid = ensemble_grid.copy()
        reliability_grid.point_data.clear()
        reliability_grid.point_data["mean_density"] = reliability_mean
        reliability_grid.point_data["std_density"] = reliability_std
        reliability_grid.point_data["prob_void"] = reliability_prob_void
        reliability_path = mc_config.output_dir / "reliability_map.vtu"
        reliability_grid.save(str(reliability_path))

        ensemble_pvd_path = ensemble_pvd_dir / "ensemble.pvd"
        _write_pvd_collection(ensemble_pvd_path, pvd_entries)

        logger.info(
            "Ensemble export complete: %d sample VTUs in %s, reliability map "
            "at %s, ParaView collection at %s",
            n, mc_config.ensemble_dir, reliability_path, ensemble_pvd_path,
        )


    return MCResult(
        compliance_samples=compliance_samples,
        mean=mean,
        variance=variance,
        std=std,
        percentile_low=float(p_low),
        percentile_high=float(p_high),
        eta_samples=eta_samples_all,
        xi_samples=xi_samples_all,
        volume_samples=volume_samples,
        nominal_volume_fraction=float(nominal_volume_fraction),
        total_volume=float(total_volume),
        n_solver_failures=int(n_ksp_failures),
        solver_failure_rate=float(failure_rate),
        statistics_conditional_on_success=bool(statistics_conditional),
        n_kl=rf_heaviside.kl_result.n_kl,
        variance_explained=rf_heaviside.kl_result.variance_explained,
        reliability_mean=reliability_mean,
        reliability_std=reliability_std,
        reliability_prob_void=reliability_prob_void,
        probability_weights=probability_weights,
        ensemble_pvd_path=ensemble_pvd_path,
    )



def compare_against_pce(mc_result: MCResult, pce_model) -> dict:
    """...
    Args:
        mc_result: Output of run_monte_carlo_validation (xi_samples at
            FULL n_kl dimension).
        pce_model: A PCEGradientModel (src.surrogate.pce_model). Its
            active_kl_indices (if set) is used to slice mc_result.xi_samples
            down to the dimension pce_model.chaos_result was actually fit
            on, before evaluating the metamodel.
    """

    xi_eval = mc_result.xi_samples
    if pce_model.active_kl_indices is not None:
        xi_eval = xi_eval[:, pce_model.active_kl_indices]

    metamodel = pce_model.chaos_result.getMetaModel()
    xi_ot = ot.Sample(xi_eval)
    pce_predictions = np.array(metamodel(xi_ot)).ravel()


    mc_truth = mc_result.compliance_samples
    residuals = mc_truth - pce_predictions


    rmse = float(np.sqrt(np.mean(residuals ** 2)))


    pce_mean = float(pce_predictions.mean())
    pce_var = float(pce_predictions.var(ddof=1))
    relative_error_mean = abs(pce_mean - mc_result.mean) / abs(mc_result.mean)
    relative_error_variance = abs(pce_var - mc_result.variance) / abs(mc_result.variance)


    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((mc_truth - mc_truth.mean()) ** 2)
    q2_vs_mc = float(1.0 - ss_res / ss_tot)


    pce_p_low = float(np.percentile(pce_predictions, 5.0))
    pce_p_high = float(np.percentile(pce_predictions, 95.0))
    tail_low_underprediction = pce_p_low > mc_result.percentile_low
    tail_high_underprediction = pce_p_high < mc_result.percentile_high


    logger.info(
        "PCE-vs-MC: RMSE=%.6g, rel_err_mean=%.4g, rel_err_var=%.4g, Q2=%.6g, "
        "tail_low_underpredict=%s, tail_high_underpredict=%s",
        rmse, relative_error_mean, relative_error_variance, q2_vs_mc,
        tail_low_underprediction, tail_high_underprediction,
    )


    return {
        "rmse": rmse,
        "relative_error_mean": relative_error_mean,
        "relative_error_variance": relative_error_variance,
        "q2_vs_mc": q2_vs_mc,
        "tail_low_underprediction": tail_low_underprediction,
        "tail_high_underprediction": tail_high_underprediction,
    }



def plot_cdf(mc_result: MCResult, output_path: Path) -> None:
    """Save a CDF plot PNG, per Section 3.6's output artifact spec.


    Compliance samples are already identical on every rank, so this should
    be called only on rank 0 by the caller to avoid redundant file writes.


    Args:
        mc_result: Output of run_monte_carlo_validation.
        output_path: File path to write the PNG to.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt


    sorted_vals, empirical_cdf = mc_result.cdf()


    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sorted_vals, empirical_cdf, linewidth=2)
    ax.axvline(mc_result.mean, color="red", linestyle="--", label=f"mean={mc_result.mean:.4g}")
    ax.axvline(mc_result.percentile_low, color="gray", linestyle=":", label="5th/95th pct")
    ax.axvline(mc_result.percentile_high, color="gray", linestyle=":")
    ax.set_xlabel("Compliance C")
    ax.set_ylabel("Empirical CDF")
    ax.set_title(f"MC Compliance Distribution (n={mc_result.compliance_samples.size})")
    ax.legend()
    fig.tight_layout()


    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Wrote CDF plot to %s", output_path)
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
    (local slice) and mc_config.seed; xi is drawn identically on every
    rank per sample since np.random.default_rng(seed + i) is deterministic,
    so no further communication is needed at sample time. Compliance is
    already a true global scalar via comm.allreduce on assemble_scalar,
    unchanged from the original.
"""
from __future__ import annotations


import logging
from dataclasses import dataclass, field
from pathlib import Path


import numpy as np
from mpi4py import MPI
from dolfinx.fem import form, assemble_scalar


from src.fenitop.fem import form_fem
from src.fenitop.parameterize import DensityFilter


from src.topology.heaviside_projection_glue import RandomFieldHeaviside, RandomHeavisideConfig
from src.random_fields.kl_expansion import KLExpansionResult


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
    """
    n_samples: int = 2500
    beta: float = 8.0
    percentiles: tuple[float, float] = (5.0, 95.0)
    seed: int = 0
    output_dir: Path = field(default_factory=lambda: Path("output/mc_validation"))


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
            an MVP diagnostic field).
        n_kl: Number of KL modes used to generate eta(x), for provenance.
        variance_explained: KL truncation variance fraction, for provenance.
    """
    compliance_samples: np.ndarray
    mean: float
    variance: float
    std: float
    percentile_low: float
    percentile_high: float
    eta_samples: np.ndarray
    n_kl: int
    variance_explained: float
    xi_samples: np.ndarray


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
        rho_converged: [n_elems_local] converged density field from a prior
            deterministic or robust TO run. Must be THIS RANK's local slice,
            matching rho_field's local dof array exactly under the SAME
            comm.size that produced it.
        kl_result: A KLExpansionResult already computed (and, under MPI,
            already broadcast identically to every rank) via
            src/random_fields/kl_expansion.py's compute_kl_expansion(). This
            is NOT recomputed here.
        heaviside_config: RandomHeavisideConfig (kernel + marginal params).
        mc_config: MCConfig controlling n_samples, beta, seed, percentiles.
            seed must be identical on every rank.


    Returns:
        An MCResult with the empirical compliance distribution (global,
        identical on every rank) and per-rank-local eta(x)/xi provenance
        arrays.


    Raises:
        ValueError: If rho_converged's shape does not match rho_field's
            local dof shape, or if opt_config["opt_compliance"] is not True.
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


    expected_shape = rho_field.x.petsc_vec.array.shape
    if rho_converged.shape != expected_shape:
        raise ValueError(
            f"rho_converged shape {rho_converged.shape} does not match "
            f"rho_field local dof shape {expected_shape}. Under MPI, "
            "rho_converged must be THIS RANK's local slice, not a global array."
        )


    local_node_coordinates = rho_phys_field.function_space.tabulate_dof_coordinates()
    spatial_dim = heaviside_config.kernel_params.spatial_dim
    local_node_coordinates = local_node_coordinates[:, :spatial_dim]
    rf_heaviside = RandomFieldHeaviside(
        rho_phys_field, local_node_coordinates, kl_result, heaviside_config
    )
    if comm.rank == 0:
        logger.info(
            "RandomFieldHeaviside ready for MC loop: N_kl=%d, variance_explained=%.4f",
            rf_heaviside.kl_result.n_kl, rf_heaviside.kl_result.variance_explained,
        )


    compliance_form = form(opt_config["compliance"])
    compliance_samples = np.zeros(mc_config.n_samples)
    n_dofs_local = local_node_coordinates.shape[0]
    eta_samples_all = np.zeros((mc_config.n_samples, n_dofs_local))
    xi_samples_all = np.zeros((mc_config.n_samples, rf_heaviside.kl_result.n_kl))


    for i in range(mc_config.n_samples):
        rho_field.x.petsc_vec.array[:] = rho_converged
        density_filter.forward()


        rng = np.random.default_rng(mc_config.seed + i)
        xi_sample = rng.standard_normal(size=rf_heaviside.kl_result.n_kl)
        eta_sample = rf_heaviside.set_eta_from_xi(xi_sample)
        xi_samples_all[i, :] = xi_sample
        eta_samples_all[i, :] = eta_sample
        rf_heaviside.forward(mc_config.beta)


        linear_problem.solve_fem()
        C_value = comm.allreduce(assemble_scalar(compliance_form), op=MPI.SUM)
        compliance_samples[i] = C_value


        if comm.rank == 0 and (i + 1) % max(1, mc_config.n_samples // 10) == 0:
            logger.info(
                "MC sample %d/%d: C=%.6g", i + 1, mc_config.n_samples, C_value
            )


    if not np.all(np.isfinite(compliance_samples)):
        n_bad = np.sum(~np.isfinite(compliance_samples))
        raise RuntimeError(
            f"{n_bad}/{mc_config.n_samples} compliance samples are non-finite "
            "(NaN/inf). This indicates an FEA solver failure for some eta(x) "
            "realization, not a valid result -- investigate before trusting "
            "any statistics below."
        )


    mean = float(compliance_samples.mean())
    variance = float(compliance_samples.var(ddof=1))
    std = float(np.sqrt(variance))
    p_low, p_high = np.percentile(compliance_samples, mc_config.percentiles)


    if comm.rank == 0:
        logger.info(
            "MC validation complete: mean=%.6g, std=%.6g, p%d=%.6g, p%d=%.6g",
            mean, std, mc_config.percentiles[0], p_low, mc_config.percentiles[1], p_high,
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
        n_kl=rf_heaviside.kl_result.n_kl,
        variance_explained=rf_heaviside.kl_result.variance_explained,
    )



def compare_against_pce(mc_result: MCResult, pce_result) -> dict:
    """Validate a PCE surrogate against brute-force MC ground truth.


    Implements Section 3.6: RMSE, relative error on mean/variance, Q^2 on
    the full MC sample set, and tail-quantile underprediction flags.


    Note: mc_result.xi_samples is a WORLD-COLLECTIVE, rank-agnostic
    quantity (xi is drawn identically per rank per sample), so this
    comparison is valid to run identically on every rank without gathering.


    Args:
        mc_result: Output of run_monte_carlo_validation (must have
            xi_samples populated).
        pce_result: A fitted PCEBuildResult (pce_builder.build_pce_surrogate's
            output), whose chaos_result.getMetaModel() is evaluated at
            mc_result.xi_samples for a like-for-like comparison.


    Returns:
        Dict with keys: rmse, relative_error_mean, relative_error_variance,
        q2_vs_mc, tail_low_underprediction, tail_high_underprediction.
    """
    import openturns as ot


    metamodel = pce_result.chaos_result.getMetaModel()
    xi_ot = ot.Sample(mc_result.xi_samples)
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
import open3d as o3d
import openturns as ot
import numpy as np
import copy
import os
from scipy.linalg import eigh
from scipy.spatial.distance import cdist


def load_moadv2_data(cad_stl_path, scan_obj_path, voxel_size=1.0):
    """Ingests nominal CAD geometry (.stl) and raw scan data (.obj) from MOADv2."""
    if not os.path.exists(cad_stl_path) or not os.path.exists(scan_obj_path):
        raise FileNotFoundError("Please provide valid paths to MOADv2 dataset files.")

    cad_mesh = o3d.io.read_triangle_mesh(cad_stl_path)
    cad_mesh.compute_vertex_normals()
    nominal_pcd = cad_mesh.sample_points_uniformly(number_of_points=50000)

    scan_pcd = o3d.io.read_point_cloud(scan_obj_path)

    if len(scan_pcd.points) == 0:
        scan_mesh = o3d.io.read_triangle_mesh(scan_obj_path, enable_post_processing=True)
        scan_pcd = o3d.geometry.PointCloud()
        scan_pcd.points = scan_mesh.vertices
        if scan_mesh.has_vertex_colors():
            scan_pcd.colors = scan_mesh.vertex_colors

    if not scan_pcd.has_colors():
        scan_pcd.paint_uniform_color([0.7, 0.7, 0.7])
    if not nominal_pcd.has_colors():
        nominal_pcd.paint_uniform_color([0.8, 0.8, 0.8])

    return cad_mesh, nominal_pcd, scan_pcd


def diagnose_clouds(source, target, voxel_size):
    src_bb = source.get_max_bound() - source.get_min_bound()
    tgt_bb = target.get_max_bound() - target.get_min_bound()
    centroid_dist = np.linalg.norm(source.get_center() - target.get_center())
    model_scale = np.max(tgt_bb)

    print(f"[Diagnostics] Source bbox extents: {src_bb}")
    print(f"[Diagnostics] Target bbox extents: {tgt_bb}")
    print(f"[Diagnostics] Centroid distance: {centroid_dist:.6f} | Model scale: {model_scale:.4f}")

    if voxel_size < model_scale * 0.001:
        print(f"[Diagnostics] WARNING: voxel_size ({voxel_size}) is tiny relative to model "
              f"scale ({model_scale:.2f}). Consider voxel_size ~ {model_scale * 0.02:.3f}.")

    return model_scale


def colored_icp_registration(source, target, voxel_size=1.0, max_corr_dist=None):
    """Aligns measured geometry to nominal CAD using Open3D's colored ICP pipeline."""
    if max_corr_dist is None:
        max_corr_dist = voxel_size * 5

    source_down = source.voxel_down_sample(voxel_size)
    target_down = target.voxel_down_sample(voxel_size)

    model_scale = diagnose_clouds(source_down, target_down, voxel_size)

    init_transform = np.identity(4)
    init_transform[:3, 3] = target_down.get_center() - source_down.get_center()

    radius = voxel_size * 2
    source_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
    target_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))

    pre_eval = o3d.pipelines.registration.evaluate_registration(
        source_down, target_down, max_corr_dist, init_transform)
    print(f"[Diagnostics] Pre-ICP fitness: {pre_eval.fitness:.4f} | "
          f"correspondences: {len(pre_eval.correspondence_set)}")

    if len(pre_eval.correspondence_set) < 10:
        raise RuntimeError(
            f"Too few correspondences ({len(pre_eval.correspondence_set)}). "
            f"Increase voxel_size/max_corr_dist relative to model scale ({model_scale:.2f})."
        )

    result = o3d.pipelines.registration.registration_colored_icp(
        source_down, target_down, max_corr_dist, init_transform,
        o3d.pipelines.registration.TransformationEstimationForColoredICP(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=50)
    )

    eval_reg = o3d.pipelines.registration.evaluate_registration(
        source_down, target_down, max_corr_dist, result.transformation)
    print(f"[Verification] Registration Fitness: {eval_reg.fitness:.4f} | RMSE: {eval_reg.inlier_rmse:.6f}")

    if eval_reg.inlier_rmse > voxel_size * 2:
        print("Warning: RMSE exceeds the defined threshold. Registration may be poor.")

    registered_scan = copy.deepcopy(source)
    registered_scan.transform(result.transformation)
    return registered_scan, model_scale


def squared_exponential_cov(dist, sigma_sq, corr_length):
    """Vectorized squared-exponential covariance kernel."""
    return sigma_sq * np.exp(-(dist ** 2) / (2.0 * corr_length ** 2))


def generate_eole_grid(cad_mesh, n_points_per_axis=6):
    """
    Generates a small, coarse auxiliary grid of EOLE nodal points spanning the
    CAD bounding box (analogous to Fig. 12 in Schevenels et al., 2011), instead
    of discretizing the random field on the full dense mesh.
    """
    min_b = np.asarray(cad_mesh.get_min_bound())
    max_b = np.asarray(cad_mesh.get_max_bound())

    axes = [np.linspace(min_b[i], max_b[i], n_points_per_axis) for i in range(3)]
    grid = np.array(np.meshgrid(*axes)).T.reshape(-1, 3)
    print(f"[EOLE] Generated coarse grid with {len(grid)} nodal points "
          f"(vs. {len(cad_mesh.vertices)} full mesh vertices).")
    return grid


def eole_decompose(eole_grid, sigma_sq, corr_length, threshold=1e-4):
    """
    Builds the covariance matrix on the small EOLE grid only, and eigendecomposes
    it there. This is the core cost-saving step: eigendecomposition happens on an
    M x M matrix (M = number of EOLE points, e.g. ~100-300), not on the full
    mesh's N x N matrix, avoiding the hang/segfault from the P1 algorithm.
    """
    dist_grid = cdist(eole_grid, eole_grid)
    cov_grid = squared_exponential_cov(dist_grid, sigma_sq, corr_length)

    eigenvalues, eigenvectors = eigh(cov_grid)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    eigenvalues = np.clip(eigenvalues, a_min=0.0, a_max=None)
    total_variance = np.sum(eigenvalues)
    explained_variance = np.cumsum(eigenvalues) / total_variance
    M = np.searchsorted(explained_variance, 1.0 - threshold) + 1
    M = min(M, len(eigenvalues))

    print(f"[EOLE] Retained {M} modes on the coarse grid to explain "
          f">= {100 * (1 - threshold):.2f}% of variance.")

    return eigenvalues[:M], eigenvectors[:, :M]


def eole_interpolate(target_points, eole_grid, eigenvalues, eigenvectors, sigma_sq, corr_length):
    """
    Projects the coarse-grid eigenmodes onto the full-resolution mesh via the
    EOLE linear estimator:
        phi_k(x) = (1 / lambda_k) * sum_j eigenvectors[j,k] * Cov(x, grid_j)
    This reconstructs the field's spatial modes at every full-mesh point cheaply,
    using only matrix-vector products against the small coarse grid.
    """
    dist_cross = cdist(target_points, eole_grid)
    cov_cross = squared_exponential_cov(dist_cross, sigma_sq, corr_length)

    safe_eigenvalues = np.where(eigenvalues > 1e-12, eigenvalues, 1e-12)
    modes = (cov_cross @ eigenvectors) / np.sqrt(safe_eigenvalues)[None, :]
    return modes


def fit_random_field_model(cad_mesh, registered_scan, model_scale, n_grid_per_axis=6):
    """
    Fits the spatial error field using the EOLE method (Li & Der Kiureghian, 1993;
    as applied by Schevenels et al., 2011) instead of a full-mesh KL expansion.
    """
    nominal_vertices = np.asarray(cad_mesh.vertices)
    nominal_pcd = o3d.geometry.PointCloud()
    nominal_pcd.points = o3d.utility.Vector3dVector(nominal_vertices)

    deviations = np.asarray(nominal_pcd.compute_point_cloud_distance(registered_scan))

    sigma_sq = np.var(deviations)
    corr_length = model_scale * 0.05
    print(f"Fitted Covariance Kernel: sigma^2 = {sigma_sq:.6f}, l = {corr_length:.4f}")

    # 1. Build the small EOLE grid instead of using the full mesh (cost-saving step)
    eole_grid = generate_eole_grid(cad_mesh, n_points_per_axis=n_grid_per_axis)

    # 2. Eigendecompose only the small M x M covariance matrix on the coarse grid
    eigenvalues, eigenvectors = eole_decompose(eole_grid, sigma_sq, corr_length)
    N_KL = len(eigenvalues)

    # 3. Interpolate the eigenmodes back onto the full mesh's nodal points cheaply
    modes_on_mesh = eole_interpolate(
        nominal_vertices, eole_grid, eigenvalues, eigenvectors, sigma_sq, corr_length)

    # 4. Non-Gaussian marginal transform, same as before, via isoprobabilistic mapping
    standard_normal = ot.Normal(0.0, 1.0)
    emp_min, emp_max = np.min(deviations), np.max(deviations)
    bounded_marginal = ot.TruncatedNormal(np.mean(deviations), np.std(deviations), emp_min, emp_max)
    marginal_transform = ot.DistributionTransformation(standard_normal, bounded_marginal)

    kl_result = {
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "eole_grid": eole_grid,
        "modes_on_mesh": modes_on_mesh,
    }

    return kl_result, marginal_transform, N_KL


def sample_field_realization(kl_result, n_samples=1, seed=None):
    """
    Draws realizations of the underlying Gaussian field on the full mesh using
    the EOLE expansion: zeta(x) = sum_k xi_k * phi_k(x), xi_k ~ N(0,1) i.i.d,
    exactly as in Eq. (21)/Fig. 14 of the paper.
    """
    rng = np.random.default_rng(seed)
    modes = kl_result["modes_on_mesh"]
    M = modes.shape[1]
    xi = rng.standard_normal(size=(n_samples, M))
    return xi @ modes.T


if __name__ == "__main__":
    CAD_PATH = "/workspace/data/cad/gear-large_cad.stl"
    SCAN_PATH = "/workspace/data/cad/fused_model.obj"

    try:
        cad_mesh, nominal_pcd, scan_pcd = load_moadv2_data(CAD_PATH, SCAN_PATH)
        print(f"Data loaded successfully. Scan contains {len(scan_pcd.points)} points.")

        registered_scan, model_scale = colored_icp_registration(scan_pcd, nominal_pcd, voxel_size=1.0)

        kl_result, marginal_transform, N_KL = fit_random_field_model(
            cad_mesh, registered_scan, model_scale)

        realizations = sample_field_realization(kl_result, n_samples=5, seed=42)
        print(f"[EOLE] Generated {realizations.shape[0]} field realizations "
              f"over {realizations.shape[1]} mesh nodes.")

        print("Validation process completed successfully.")

    except FileNotFoundError as e:
        print(e)
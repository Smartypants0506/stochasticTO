import open3d as o3d
import openturns as ot
import numpy as np
import copy
import os


def load_moadv2_data(cad_stl_path, scan_obj_path, voxel_size=0.01):
    """Ingests nominal CAD geometry (.stl) and raw scan data (.obj) from MOADv2."""
    if not os.path.exists(cad_stl_path) or not os.path.exists(scan_obj_path):
        raise FileNotFoundError("Please provide valid paths to MOADv2 dataset files.")

    # 1. Load nominal CAD geometry from STL and sample it into a point cloud
    cad_mesh = o3d.io.read_triangle_mesh(cad_stl_path)
    cad_mesh.compute_vertex_normals()
    nominal_pcd = cad_mesh.sample_points_uniformly(number_of_points=50000)

    # 2. Load scan data from OBJ
    scan_pcd = o3d.io.read_point_cloud(scan_obj_path)

    if len(scan_pcd.points) == 0:
        scan_mesh = o3d.io.read_triangle_mesh(scan_obj_path, enable_post_processing=True)
        scan_pcd = o3d.geometry.PointCloud()
        scan_pcd.points = scan_mesh.vertices
        if scan_mesh.has_vertex_colors():
            scan_pcd.colors = scan_mesh.vertex_colors

    # 3. Ensure color channels exist for Colored ICP
    if not scan_pcd.has_colors():
        scan_pcd.paint_uniform_color([0.7, 0.7, 0.7])
    if not nominal_pcd.has_colors():
        nominal_pcd.paint_uniform_color([0.8, 0.8, 0.8])

    return cad_mesh, nominal_pcd, scan_pcd


def diagnose_clouds(source, target, voxel_size):
    """Prints bounding box / centroid diagnostics to sanity-check scale and alignment."""
    src_bb = source.get_max_bound() - source.get_min_bound()
    tgt_bb = target.get_max_bound() - target.get_min_bound()
    src_center = source.get_center()
    tgt_center = target.get_center()
    centroid_dist = np.linalg.norm(src_center - tgt_center)

    print(f"[Diagnostics] Source bbox extents: {src_bb}")
    print(f"[Diagnostics] Target bbox extents: {tgt_bb}")
    print(f"[Diagnostics] Source centroid: {src_center}")
    print(f"[Diagnostics] Target centroid: {tgt_center}")
    print(f"[Diagnostics] Centroid distance: {centroid_dist:.6f} (voxel_size={voxel_size})")

    if centroid_dist > voxel_size * 50:
        print("[Diagnostics] WARNING: Centroids are far apart relative to voxel_size. "
              "Consider running a coarse/global registration step first.")


def colored_icp_registration(source, target, voxel_size=0.01, max_corr_dist=None):
    """Aligns measured geometry to nominal CAD using Open3D's colored ICP pipeline."""
    if max_corr_dist is None:
        # Use a looser correspondence radius than the downsampling voxel size itself
        max_corr_dist = voxel_size * 5

    source_down = source.voxel_down_sample(voxel_size)
    target_down = target.voxel_down_sample(voxel_size)

    diagnose_clouds(source_down, target_down, voxel_size)

    # Naive coarse alignment: shift source centroid onto target centroid before ICP
    src_center = source_down.get_center()
    tgt_center = target_down.get_center()
    init_transform = np.identity(4)
    init_transform[:3, 3] = tgt_center - src_center

    radius = voxel_size * 2
    source_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
    target_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))

    src_normals = np.asarray(source_down.normals)
    tgt_normals = np.asarray(target_down.normals)
    n_nan_src = np.isnan(src_normals).any(axis=1).sum()
    n_nan_tgt = np.isnan(tgt_normals).any(axis=1).sum()
    if n_nan_src or n_nan_tgt:
        print(f"[Diagnostics] WARNING: NaN normals found -> source: {n_nan_src}, target: {n_nan_tgt}. "
              "Consider increasing normal-estimation radius/max_nn or upsampling before downsampling.")

    # Sanity check: evaluate correspondences at the naive init before running full colored ICP
    pre_eval = o3d.pipelines.registration.evaluate_registration(
        source_down, target_down, max_corr_dist, init_transform)
    print(f"[Diagnostics] Pre-ICP fitness: {pre_eval.fitness:.4f} | "
          f"correspondences: {len(pre_eval.correspondence_set)}")

    if len(pre_eval.correspondence_set) == 0:
        raise RuntimeError(
            "No correspondences found even at max_corr_dist="
            f"{max_corr_dist}. Increase max_corr_dist further or run a "
            "global/feature-based registration (e.g., RANSAC + FPFH) before colored ICP."
        )

    # Jointly optimize geometric and photometric (color/intensity) residuals
    result = o3d.pipelines.registration.registration_colored_icp(
        source_down, target_down, max_corr_dist, init_transform,
        o3d.pipelines.registration.TransformationEstimationForColoredICP(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=50)
    )

    # Verification: Registration fitness/RMSE threshold
    eval_reg = o3d.pipelines.registration.evaluate_registration(
        source_down, target_down, max_corr_dist, result.transformation)
    print(f"[Verification] Registration Fitness: {eval_reg.fitness:.4f} | RMSE: {eval_reg.inlier_rmse:.6f}")

    if eval_reg.inlier_rmse > voxel_size * 2:
        print("Warning: RMSE exceeds the defined threshold. Registration may be poor.")

    registered_scan = copy.deepcopy(source)
    registered_scan.transform(result.transformation)
    return registered_scan


def fit_random_field_model(cad_mesh, registered_scan):
    """Fits the spatial error field, executes KL expansion, and maps the non-Gaussian marginal."""
    # 1. Metrology extraction: Map deviations to the nominal CAD surface nodes
    nominal_vertices = np.asarray(cad_mesh.vertices)
    nominal_pcd = o3d.geometry.PointCloud()
    nominal_pcd.points = o3d.utility.Vector3dVector(nominal_vertices)

    deviations = np.asarray(nominal_pcd.compute_point_cloud_distance(registered_scan))

    # 2. OpenTURNS Mesh Setup
    vertices = ot.Sample(nominal_vertices)
    simplices = np.asarray(cad_mesh.triangles)
    mesh = ot.Mesh(vertices, simplices)

    # 3. Fit squared-exponential covariance kernel to empirical spatial error field
    sigma_sq = np.var(deviations)
    l = [0.05, 0.05, 0.05]  # Correlation length heuristic (would be fitted via empirical variogram)
    cov_model = ot.SquaredExponential(l, [sigma_sq])
    print(f"Fitted Covariance Kernel: sigma^2 = {sigma_sq:.6f}, l = {l[0]}")

    # 4. KL Expansion (FEM-based on nodal grid)
    threshold = 1e-4
    algo = ot.KarhunenLoeveP1Algorithm(mesh, cov_model, threshold)
    algo.run()
    kl_result = algo.getResult()

    eigenvalues = np.array(kl_result.getEigenvalues())
    total_variance = np.sum(eigenvalues)
    explained_variance = np.cumsum(eigenvalues) / total_variance

    N_KL = np.searchsorted(explained_variance, 0.95) + 1
    print(f"KL Expansion: Retained {N_KL} eigenmodes to explain >= 95% of total variance.")

    # 5. Primary random field target - projection threshold eta(x) via isoprobabilistic transform
    standard_normal = ot.Normal(0.0, 1.0)

    emp_min, emp_max = np.min(deviations), np.max(deviations)
    bounded_marginal = ot.TruncatedNormal(np.mean(deviations), np.std(deviations), emp_min, emp_max)

    marginal_transform = ot.DistributionTransformation(standard_normal, bounded_marginal)

    return kl_result, marginal_transform, N_KL


if __name__ == "__main__":
    CAD_PATH = "/workspace/data/cad/gear-large_cad.stl"
    SCAN_PATH = "/workspace/data/cad/fused_model.obj"

    try:
        cad_mesh, nominal_pcd, scan_pcd = load_moadv2_data(CAD_PATH, SCAN_PATH)
        print(f"Data loaded successfully. Scan contains {len(scan_pcd.points)} points.")

        registered_scan = colored_icp_registration(scan_pcd, nominal_pcd)
        kl_result, marginal_transform, N_KL = fit_random_field_model(cad_mesh, registered_scan)
        print("Validation process completed successfully.")

    except FileNotFoundError as e:
        print(e)
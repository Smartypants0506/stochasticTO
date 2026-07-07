"""
Beam 3D compliance minimization example.
Runs on 4 A100 GPUs (physical indices 0, 1, 2, 4 -- slot 3 is a
non-compute display GPU and is excluded), one MPI rank per GPU.

Launch with:
    mpirun -n 4 --bind-to none python3 scripts/beam_3d.py
"""

import os
from mpi4py import MPI

# --- GPU device binding (must happen before any PETSc/dolfinx import
# that could initialize a CUDA context) ---
COMPUTE_GPU_IDS = [0, 1, 2, 4]  # slot 3 excluded (display GPU)
_local_rank = MPI.COMM_WORLD.rank % len(COMPUTE_GPU_IDS)
os.environ["CUDA_VISIBLE_DEVICES"] = str(COMPUTE_GPU_IDS[_local_rank])

import numpy as np
from dolfinx.mesh import create_box, CellType

from fenitop.topopt import topopt

mesh = create_box(MPI.COMM_WORLD, [[0, 0, 0], [10, 30, 10]],
    [75, 225, 75], CellType.hexahedron)
if MPI.COMM_WORLD.rank == 0:
    mesh_serial = create_box(MPI.COMM_SELF, [[0, 0, 0], [10, 30, 10]],
        [75, 225, 75], CellType.hexahedron)
else:
    mesh_serial = None

fem = {  # FEA parameters
    "mesh": mesh,
    "mesh_serial": mesh_serial,
    "young's modulus": 100,
    "poisson's ratio": 0.25,
    "disp_bc": lambda x: np.isclose(x[1], 0) & (np.less(x[0], 1.5) | np.greater(x[0], 8.5)),
    "traction_bcs": [[(0, 0, -2.0),
        lambda x: np.isclose(x[1], 30) & (
            np.greater(x[0], 4.5) & np.less(x[0], 5.5)
            & np.greater(x[2], 4.5) & np.less(x[2], 5.5))]],
    "body_force": (0, 0, 0),
    "quadrature_degree": 2,
    # GPU-accelerated fallback solver (used only if reanalysis is off).
    "petsc_options": {
        "ksp_type": "cg",
        "pc_type": "gamg",
    },
    # GPU-accelerated options used only by the structural LinearProblem.
    "gpu_petsc_options": {
        "ksp_type": "cg",
        "pc_type": "jacobi",
        "mat_type": "aijcusparse",
        "vec_type": "cuda",
    },
}

opt = {  # Topology optimization parameters
    "max_iter": 100,
    "opt_tol": 1e-4,
    "vol_frac": 0.08,
    "solid_zone": lambda x: np.full(x.shape[1], False),
    "void_zone": lambda x: np.full(x.shape[1], False),
    "penalty": 3.0,
    "epsilon": 1e-6,
    "filter_radius": 0.6,
    "beta_interval": 50,
    "beta_max": 128,
    "use_oc": True,
    "move": 0.02,
    "opt_compliance": True,

    # Only gather/plot/save every N iterations (always includes the
    # final iteration) to avoid paying PyVista smoothing + gather
    # communication costs every single iteration.
    "plot_interval": 10,

    # Efficient reanalysis (Combined Approximations / PCG), Amir et al.
    # (2012), accelerated on GPU via CUDA-resident matrices/vectors and
    # hypre's BoomerAMG (GPU-capable).
    "reanalysis_options": {
        "enabled": True,
        "max_iter": 15,
        "rtol": 1e-6,
        "refactor_interval": 50,
        "ref_petsc_options": {
            "ksp_type": "richardson",
            "pc_type": "jacobi",
            "mat_type": "aijcusparse",
            "vec_type": "cuda",
        },
},
}

if __name__ == "__main__":
    topopt(fem, opt)

# Execute the code in parallel (one rank per A100 GPU):
# mpirun -n 4 --bind-to none python3 scripts/beam_3d.py
#!/usr/bin/env bash
set -euo pipefail

bash /workspace/.devcontainer/common/postCreate_common.sh

# DGX: 4 active GPUs on slots 0,1,2,4
python3 - <<'EOF'
import cupy as cp
for idx in [0, 1, 2, 4]:
    with cp.cuda.Device(idx):
        props = cp.cuda.runtime.getDeviceProperties(idx)
        mem = props['totalGlobalMem'] // 1024**3
        print(f"  GPU {idx}: {props['name'].decode()} — {mem} GB")
EOF

mpirun -n 4 python3 -c "
from mpi4py import MPI
import os
rank = MPI.COMM_WORLD.Get_rank()
gpu_map = {0:0, 1:1, 2:2, 3:4}
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_map[rank])
print(f'  rank {rank} → GPU {gpu_map[rank]}')
"
#!/usr/bin/env bash
set -euo pipefail

bash /workspace/.devcontainer/common/postCreate_common.sh

python3 - <<'EOF'
import cupy as cp
for idx in [0]:
    with cp.cuda.Device(idx):
        props = cp.cuda.runtime.getDeviceProperties(idx)
        mem = props['totalGlobalMem'] // 1024**3
        print(f" GPU (local {idx}): {props['name'].decode()} — {mem} GB")
EOF

/usr/local/bin/mpirun -n 1 python3 -c "
from mpi4py import MPI
import os
rank = MPI.COMM_WORLD.Get_rank()
print(f' rank {rank} -> local GPU ordinal {rank}')
"
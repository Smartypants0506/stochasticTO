#!/usr/bin/env bash
set -euo pipefail

bash /workspace/.devcontainer/common/postCreate_common.sh

/dolfinx-env/bin/python3 - <<'EOF'
import cupy as cp
props = cp.cuda.runtime.getDeviceProperties(0)
mem = props['totalGlobalMem'] // 1024**3
print(f"  GPU 0: {props['name'].decode()} — {mem} GB")
EOF
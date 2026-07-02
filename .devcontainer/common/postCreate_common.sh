#!/usr/bin/env bash
set -euo pipefail

PIP=/dolfinx-env/bin/pip

# UID/GID now match host at build time — no chown needed on /workspace bind mount.
# Only /home/devuser is container-local, safe to normalize if ever needed.

if [ -f /workspace/pyproject.toml ]; then
  # Clear stale build artifacts before an editable install. setuptools' egg_info
  # step tries to update the mtime of an existing <pkg>.egg-info directory, and
  # fails hard ("Cannot update time stamp of directory ...") if that directory is
  # left over from an earlier/failed devcontainer build — e.g. created under a
  # different UID, or on a bind-mount where the UID/GID-matches-host assumption
  # above didn't hold for that particular artifact. These are regenerated on
  # every install anyway, so it's always safe to drop them first.
  find /workspace -maxdepth 1 -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true
  rm -rf /workspace/build

  if ! sudo "$PIP" install -e /workspace; then
    echo "pip install -e failed — retrying once after a permission fix on egg-info/build artifacts."
    find /workspace -maxdepth 1 -name '*.egg-info' -exec chown -R "$(id -u):$(id -g)" {} + 2>/dev/null || true
    find /workspace -maxdepth 1 -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true
    rm -rf /workspace/build
    sudo "$PIP" install -e /workspace
  fi
fi

if [ -f /workspace/.pre-commit-config.yaml ]; then
  cd /workspace && "$PIP" show pre-commit >/dev/null 2>&1 && pre-commit install
fi

mkdir -p /workspace/meshes /workspace/logs /workspace/checkpoints /workspace/results

# ── GPU-aware MPI sanity check ──────────────────────────────────────────────
# Confirms the base image's CUDA-aware Open MPI is still what's active (i.e.
# nothing on the host/devcontainer side has shadowed it) and that the runtime
# env vars that actually activate CUDA transport are set. Informational only —
# doesn't fail postCreate, since some hosts (no GPU attached) legitimately
# won't show CUDA support.
echo "── MPI / GPU sanity check ──────────────────────────────────────────"
echo "mpirun: $(which mpirun 2>/dev/null || echo 'NOT FOUND')"
if command -v ompi_info >/dev/null 2>&1; then
  CUDA_AWARE=$(ompi_info --parsable --all 2>/dev/null | grep -i 'mpi_built_with_cuda_support:value' || true)
  echo "Open MPI CUDA build flag: ${CUDA_AWARE:-unknown}"
  case "$CUDA_AWARE" in
    *true*) echo "  -> OK: Open MPI reports CUDA-aware." ;;
    *) echo "  -> WARNING: Open MPI does NOT report CUDA-aware. Check that nothing" \
            "(apt, pip, a custom image layer) installed a second, non-CUDA MPI." ;;
  esac
fi
echo "OMPI_MCA_opal_cuda_support=${OMPI_MCA_opal_cuda_support:-<unset>}"
echo "PETSC_OPTIONS=${PETSC_OPTIONS:-<unset>}"
echo "────────────────────────────────────────────────────────────────────"
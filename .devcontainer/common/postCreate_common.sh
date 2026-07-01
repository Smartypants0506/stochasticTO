#!/usr/bin/env bash
set -euo pipefail

PIP=/dolfinx-env/bin/pip

# UID/GID now match host at build time — no chown needed on /workspace bind mount.
# Only /home/devuser is container-local, safe to normalize if ever needed.

if [ -f /workspace/pyproject.toml ]; then
  "$PIP" install -e /workspace
fi

if [ -f /workspace/.pre-commit-config.yaml ]; then
  cd /workspace && "$PIP" show pre-commit >/dev/null 2>&1 && pre-commit install
fi

mkdir -p /workspace/meshes /workspace/logs /workspace/checkpoints /workspace/results
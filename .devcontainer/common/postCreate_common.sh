#!/usr/bin/env bash
set -euo pipefail

PIP=/dolfinx-env/bin/pip

sudo chown -R devuser:devuser /home/devuser
sudo chown -R devuser:devuser /workspace 2>/tmp/chown_errors.log || {
  echo "Warning: chown failed for some paths under /workspace (continuing):"
  cat /tmp/chown_errors.log
}

if [ -f /workspace/pyproject.toml ]; then
  sudo "$PIP" install -e /workspace
fi

if [ -f /workspace/.pre-commit-config.yaml ]; then
  cd /workspace && "$PIP" show pre-commit >/dev/null 2>&1 && pre-commit install
fi

mkdir -p /workspace/{meshes,logs,checkpoints,results}
sudo chown -R devuser:devuser /workspace/{meshes,logs,checkpoints,results} 2>/dev/null || true
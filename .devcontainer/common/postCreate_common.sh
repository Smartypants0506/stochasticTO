#!/usr/bin/env bash
set -euo pipefail

if [ -f /workspace/pyproject.toml ]; then
  pip install --user -e /workspace
fi

if [ -f /workspace/.pre-commit-config.yaml ]; then
  cd /workspace && pre-commit install
fi

mkdir -p /workspace/{meshes,logs,checkpoints,results}
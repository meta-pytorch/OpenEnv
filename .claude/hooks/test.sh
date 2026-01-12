#!/bin/bash
# Test runner for OpenEnv
# Runs pytest excluding environments that need special setup

set -e

# Check for required tools
if ! command -v uv &> /dev/null; then
    echo "Error: 'uv' is not installed or not in PATH"
    echo "Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "=== Running tests ==="
PYTHONPATH=src:envs uv run pytest tests/ \
    --ignore=tests/envs/test_browsergym_environment.py \
    --ignore=tests/envs/test_dipg_environment.py \
    --ignore=tests/envs/test_websearch_environment.py \
    -v \
    --tb=short

echo "=== Tests completed ==="

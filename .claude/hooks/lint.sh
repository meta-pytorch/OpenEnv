#!/bin/bash
# Lint check for OpenEnv
# Runs ruff format check and ruff check (linting rules) on src/ and tests/

set -e

# Check for required tools
if ! command -v uv &> /dev/null; then
    echo "Error: 'uv' is not installed or not in PATH"
    echo "Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "=== Running format check ==="
uv run ruff format src/ tests/ --check

echo "=== Running lint rules check ==="
uv run ruff check src/ tests/

echo "=== Lint check passed ==="

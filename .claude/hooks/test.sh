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
# Note: Using timeout to prevent hanging tests from blocking indefinitely (5 min max)
# Some websocket protocol tests can hang when fixtures fail to start servers
PYTHONPATH=src:envs timeout 300 uv run pytest tests/ \
    --ignore=tests/envs/test_browsergym_environment.py \
    --ignore=tests/envs/test_dipg_environment.py \
    --ignore=tests/envs/test_websearch_environment.py \
    --ignore=tests/envs/test_websockets.py \
    -v \
    --tb=short

TEST_EXIT_CODE=$?
if [ $TEST_EXIT_CODE -eq 124 ]; then
    echo "ERROR: Tests timed out after 5 minutes"
    exit 1
elif [ $TEST_EXIT_CODE -ne 0 ]; then
    echo "=== Tests failed ==="
    exit $TEST_EXIT_CODE
fi

echo "=== Tests completed ==="

#!/bin/bash
# Run the wildfire environment server from the monorepo

# Get the OpenEnv root directory (3 levels up from this script)
OPENENV_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"

# Run from monorepo root with proper PYTHONPATH
cd "$OPENENV_ROOT"
PYTHONPATH=src python -m envs.wildfire_env.server.app "$@"

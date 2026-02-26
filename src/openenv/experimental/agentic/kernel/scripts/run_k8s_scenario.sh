#!/usr/bin/env bash
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# Run an agentkernel example scenario against the k8s cluster.
#
# Usage:
#   ./agentkernel/scripts/run_k8s_scenario.sh                       # team_scenario (default)
#   ./agentkernel/scripts/run_k8s_scenario.sh simple_agent           # simple_agent
#   ./agentkernel/scripts/run_k8s_scenario.sh team_scenario          # explicit
#   LLM_API_KEY=... ./agentkernel/scripts/run_k8s_scenario.sh      # provide key inline
#
# Run from the agentkernel root.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG="$REPO_ROOT/agentkernel/examples/agentkernel.yaml"

SCENARIO="${1:-team_scenario}"

# Validate
MODULE="agentkernel.examples.${SCENARIO}"
SCENARIO_FILE="$REPO_ROOT/agentkernel/examples/${SCENARIO}.py"
if [ ! -f "$SCENARIO_FILE" ]; then
    echo "Error: scenario '$SCENARIO' not found at $SCENARIO_FILE" >&2
    echo "Available scenarios:" >&2
    ls "$REPO_ROOT/agentkernel/examples/"*.py 2>/dev/null | xargs -I{} basename {} .py | grep -v __pycache__ >&2
    exit 1
fi

# Check API key
if [ -z "${LLM_API_KEY:-}" ] && [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "Error: LLM_API_KEY or OPENAI_API_KEY must be set" >&2
    exit 1
fi

echo "==> Running $MODULE against k8s"
echo "    config: $CONFIG"
echo ""

uv run python -m "$MODULE" --config "$CONFIG"

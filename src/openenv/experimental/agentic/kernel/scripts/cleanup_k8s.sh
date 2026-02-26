#!/usr/bin/env bash
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# Clean up agentkernel resources from the k8s cluster.
#
# Usage:
#   ./agentkernel/scripts/cleanup_k8s.sh              # clean up everything
#   ./agentkernel/scripts/cleanup_k8s.sh --dry-run     # show what would be deleted
#   ./agentkernel/scripts/cleanup_k8s.sh --pods-only   # only delete pods
#
# Reads namespace and kubeconfig from agentkernel.yaml if present.
# Run from the agentkernel root.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG="$REPO_ROOT/agentkernel/examples/agentkernel.yaml"

# Read config
NAMESPACE="agentkernel-0"
KUBECONFIG_PATH=""
if [ -f "$CONFIG" ]; then
    NAMESPACE=$(grep '^namespace:' "$CONFIG" 2>/dev/null | sed 's/^namespace:[[:space:]]*//' || echo "agentkernel-0")
    KUBECONFIG_PATH=$(grep '^kubeconfig:' "$CONFIG" 2>/dev/null | sed 's/^kubeconfig:[[:space:]]*//' || true)
    # Expand ~
    KUBECONFIG_PATH="${KUBECONFIG_PATH/#\~/$HOME}"
fi

KUBECTL="kubectl"
if [ -n "$KUBECONFIG_PATH" ]; then
    KUBECTL="kubectl --kubeconfig $KUBECONFIG_PATH"
fi

DRY_RUN=""
PODS_ONLY=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN="--dry-run=client"; shift ;;
        --pods-only)
            PODS_ONLY=true; shift ;;
        --help|-h)
            echo "Usage: $0 [--dry-run] [--pods-only]"
            exit 0 ;;
        *)
            echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

echo "==> Namespace: $NAMESPACE"
if [ -n "$DRY_RUN" ]; then
    echo "    (dry run — nothing will be deleted)"
fi
echo ""

# Show current state
echo "--- Current resources ---"
$KUBECTL -n "$NAMESPACE" get pods,svc,cm 2>&1 | grep -v '^$' || echo "(none)"
echo ""

if [ "$PODS_ONLY" = true ]; then
    echo "==> Deleting pods..."
    $KUBECTL -n "$NAMESPACE" delete pods --all --grace-period=5 $DRY_RUN 2>&1
else
    # Delete agentkernel-managed resources (pods, services, configmaps with agent labels)
    echo "==> Deleting agentkernel pods..."
    $KUBECTL -n "$NAMESPACE" delete pods -l agentkernel/agent-id --grace-period=5 $DRY_RUN 2>&1 || true

    echo "==> Deleting agentkernel services..."
    $KUBECTL -n "$NAMESPACE" delete svc -l agentkernel/agent-id $DRY_RUN 2>&1 || true

    echo "==> Deleting agentkernel configmaps..."
    $KUBECTL -n "$NAMESPACE" delete cm -l agentkernel/agent-id $DRY_RUN 2>&1 || true
fi

echo ""
echo "--- Remaining resources ---"
$KUBECTL -n "$NAMESPACE" get pods,svc,cm 2>&1 | grep -v '^$' || echo "(none)"
echo ""
echo "Done."

#!/usr/bin/env bash
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# Build and push the agentkernel image (includes agentbus by default).
#
# Usage:
#   ./agentkernel/scripts/build_base_image.sh                    # build + push (arm64, with agentbus)
#   ./agentkernel/scripts/build_base_image.sh --no-push          # build only
#   ./agentkernel/scripts/build_base_image.sh --tag myregistry/img:v2
#   ./agentkernel/scripts/build_base_image.sh --platform amd64   # build for amd64
#   ./agentkernel/scripts/build_base_image.sh --no-agentbus      # base image only (no agentbus)
#   ./agentkernel/scripts/build_base_image.sh --force-base       # rebuild base before agentbus
#
# Valid platforms: arm64 (default), amd64
#
# Uses --network=host for network access during build.
# The agentbus overlay cross-compiles the Rust extension from the host arch,
# avoiding QEMU emulation which causes rustc zombie processes.
#
# Reads defaults from agentkernel.yaml if present (base_image, registry_url).
# Run from the agentkernel root.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DOCKERFILE="$REPO_ROOT/agentkernel/kernel/backends/kubernetes/Dockerfile"
BUILD_DIR="${AGENTKERNEL_BUILD_DIR:-/tmp/agentkernel-build}"

# Defaults
TAG=""
PUSH=true
PLATFORM="linux/arm64"
WITH_AGENTBUS=true
FORCE_BASE=false

# Try to read defaults from agentkernel.yaml (simple grep, no yaml dep)
CONFIG="$REPO_ROOT/agentkernel/examples/agentkernel.yaml"
if [ -f "$CONFIG" ]; then
    DEFAULT_TAG=$(grep '^base_image:' "$CONFIG" 2>/dev/null | sed 's/^base_image:[[:space:]]*//' || true)
    if [ -n "$DEFAULT_TAG" ]; then
        TAG="$DEFAULT_TAG"
    fi
fi

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --tag)
            TAG="$2"; shift 2 ;;
        --no-push)
            PUSH=false; shift ;;
        --no-agentbus)
            WITH_AGENTBUS=false; shift ;;
        --force-base)
            FORCE_BASE=true; shift ;;
        --platform)
            case "$2" in
                arm64|aarch64) PLATFORM="linux/arm64" ;;
                amd64|x86_64) PLATFORM="linux/amd64" ;;
                *) echo "Error: invalid platform '$2'. Valid: arm64, amd64" >&2; exit 1 ;;
            esac
            shift 2 ;;
        --help|-h)
            echo "Usage: $0 [--tag IMAGE:TAG] [--no-push] [--no-agentbus] [--force-base] [--platform arm64|amd64]"
            exit 0 ;;
        *)
            echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [ -z "$TAG" ]; then
    echo "Error: no tag specified and could not read base_image from agentkernel.yaml" >&2
    echo "Usage: $0 --tag REGISTRY/IMAGE:TAG" >&2
    exit 1
fi

echo "==> Staging build context to $BUILD_DIR"
mkdir -p "$BUILD_DIR"
# Sync only what the Dockerfile needs
cp "$REPO_ROOT/pyproject.toml" "$BUILD_DIR/"
cp "$REPO_ROOT/README.md" "$BUILD_DIR/"
cp "$DOCKERFILE" "$BUILD_DIR/Dockerfile"
rsync -aL --delete "$REPO_ROOT/agentkernel/" "$BUILD_DIR/agentkernel/"
# Note: if your agent runtime requires additional packages, copy them here:
# rsync -aL --safe-links --delete "$REPO_ROOT/my_runtime/" "$BUILD_DIR/my_runtime/"

BUILD_ARGS=()
if [ "$WITH_AGENTBUS" = true ]; then
    # Build the base image if missing or wrong architecture.
    BASE_TAG="${TAG}-base"
    BASE_ARCH=$(podman inspect "$BASE_TAG" --format '{{.Architecture}}' 2>/dev/null \
             || podman inspect "localhost/$BASE_TAG" --format '{{.Architecture}}' 2>/dev/null \
             || true)
    if [ "$FORCE_BASE" = true ] || [ "$BASE_ARCH" != "${PLATFORM#linux/}" ]; then
        echo "==> Building base image ($BASE_TAG)"
        "$0" --tag "$BASE_TAG" --no-push --no-agentbus --platform "${PLATFORM#linux/}"
        echo ""
    fi

    AGENTBUS_ROOT="$REPO_ROOT/../../../agentbus"
    if [ ! -d "$AGENTBUS_ROOT" ]; then
        echo "Error: agentbus not found at $AGENTBUS_ROOT" >&2
        echo "Expected agentbus/ relative to repo root." >&2
        exit 1
    fi

    # Stage agentbus Python package
    rsync -aL --delete "$AGENTBUS_ROOT/python/" "$BUILD_DIR/agentbus_python/"

    # Stage Rust workspace (server_py_ext + its workspace dependencies).
    # We copy source from the main tree, then overlay Cargo.toml files from
    # public_autocargo/ which replace internal path deps (e.g. fbinit) with
    # git deps that can be fetched during the Docker build.
    mkdir -p "$BUILD_DIR/agentbus_src"
    cp "$AGENTBUS_ROOT/Cargo.lock" "$BUILD_DIR/agentbus_src/Cargo.lock"
    for item in proto api core server_py_ext; do
        mkdir -p "$BUILD_DIR/agentbus_src/$item"
        rsync -aL --delete "$AGENTBUS_ROOT/$item/" "$BUILD_DIR/agentbus_src/$item/"
    done
    # impls/ - server_py_ext depends on impls/writeonce; sync the whole dir
    # since impls/simple is also a workspace member referenced by other crates.
    rsync -aL --delete "$AGENTBUS_ROOT/impls/" "$BUILD_DIR/agentbus_src/impls/"

    # Overlay public_autocargo Cargo.toml files (replaces internal path deps
    # like fbinit with git deps from github.com/facebookexperimental/rust-shed).
    PUBLIC_CARGO="$AGENTBUS_ROOT/public_autocargo"
    for item in api core server_py_ext; do
        cp "$PUBLIC_CARGO/$item/Cargo.toml" "$BUILD_DIR/agentbus_src/$item/Cargo.toml"
    done
    for item in simple writeonce; do
        cp "$PUBLIC_CARGO/impls/$item/Cargo.toml" "$BUILD_DIR/agentbus_src/impls/$item/Cargo.toml"
    done

    # Write a trimmed workspace Cargo.toml with only the staged members.
    cat > "$BUILD_DIR/agentbus_src/Cargo.toml" <<'CARGO_EOF'
[workspace]
members = [
  "api",
  "core",
  "impls/simple",
  "impls/writeonce",
  "proto",
  "server_py_ext",
]
resolver = "2"

[workspace.package]
license = "BSD-3-Clause"
CARGO_EOF

    # Use the agentbus Dockerfile overlay instead of the base Dockerfile
    cp "$REPO_ROOT/agentkernel/kernel/backends/kubernetes/Dockerfile.agentbus" \
       "$BUILD_DIR/Dockerfile"

    # Pass base image to the agentbus Dockerfile.
    # Prefix short names with localhost/ so podman doesn't prompt for a registry.
    if [[ "$BASE_TAG" == *.* ]]; then
        BUILD_ARGS+=(--build-arg "BASE_IMAGE=$BASE_TAG")
    else
        BUILD_ARGS+=(--build-arg "BASE_IMAGE=localhost/$BASE_TAG")
    fi
    echo "==> Building agentbus image $TAG (base: $BASE_TAG)"
fi

echo "==> Building $TAG (platform: $PLATFORM)"
cd "$BUILD_DIR"
# Resolve forward proxy IP if available (for networks requiring a proxy).
# Set PROXY_HOST to override (e.g. PROXY_HOST=myproxy.example.com).
ADD_HOST_ARGS=()
PROXY_HOST="${PROXY_HOST:-}"
if [ -n "$PROXY_HOST" ] && command -v getent &>/dev/null; then
    PROXY_IP=$(getent hosts "$PROXY_HOST" 2>/dev/null | awk '{print $1}' || true)
    if [ -n "$PROXY_IP" ]; then
        ADD_HOST_ARGS=(--add-host "$PROXY_HOST:$PROXY_IP")
    fi
fi
podman build \
    --network=host \
    --platform "$PLATFORM" \
    "${ADD_HOST_ARGS[@]+"${ADD_HOST_ARGS[@]}"}" \
    "${BUILD_ARGS[@]+"${BUILD_ARGS[@]}"}" \
    -t "$TAG" \
    .

ARCH=$(podman inspect "$TAG" --format '{{.Architecture}}')
echo "==> Built $TAG (arch: $ARCH)"

if [ "$PUSH" = true ]; then
    # Extract registry hostname and bypass proxy (registry is often directly reachable)
    REGISTRY_HOST=$(echo "$TAG" | cut -d/ -f1)
    echo "==> Pushing $TAG"
    no_proxy="${no_proxy:-},$REGISTRY_HOST" podman push "$TAG"
    echo "==> Pushed $TAG"
else
    echo "==> Skipping push (--no-push)"
fi

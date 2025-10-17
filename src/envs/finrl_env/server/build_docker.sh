#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Script to build the FinRL environment Docker image
# Usage: ./build_docker.sh [tag]
#
# Note: Requires envtorch-base:latest to be built first.
# Build with: docker build -t envtorch-base:latest -f src/core/containers/images/Dockerfile .

set -e

TAG="${1:-latest}"
IMAGE_NAME="finrl-env:${TAG}"

echo "🐳 Building FinRL Environment Docker Image"
echo "=============================================="
echo "Image: $IMAGE_NAME"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Navigate to OpenEnv root (4 levels up from server/)
OPENENV_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

echo "📁 OpenEnv root: $OPENENV_ROOT"
echo ""

# Check if base image exists
if ! docker images | grep -q "envtorch-base.*latest"; then
    echo "⚠️  Base image 'envtorch-base:latest' not found!"
    echo ""
    echo "Building base image first..."
    cd "$OPENENV_ROOT"
    docker build -t envtorch-base:latest -f src/core/containers/images/Dockerfile .

    if [ $? -ne 0 ]; then
        echo ""
        echo "❌ Failed to build base image"
        exit 1
    fi
    echo ""
fi

# Build FinRL environment image
echo "⏳ Building FinRL environment image..."
docker build \
    -f "$SCRIPT_DIR/Dockerfile" \
    -t "$IMAGE_NAME" \
    "$OPENENV_ROOT"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Build successful!"
    echo ""
    echo "📊 Image info:"
    docker images "$IMAGE_NAME" --format "table {{.Repository}}:{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
    echo ""
    echo "🚀 Usage examples:"
    echo ""
    echo "  # Basic usage (default sample data)"
    echo "  docker run -p 8000:8000 $IMAGE_NAME"
    echo ""
    echo "  # With custom initial amount"
    echo "  docker run -p 8000:8000 -e FINRL_INITIAL_AMOUNT=50000 $IMAGE_NAME"
    echo ""
    echo "  # With custom configuration file"
    echo "  docker run -p 8000:8000 \\"
    echo "    -v \$(pwd)/config.json:/config/config.json \\"
    echo "    -e FINRL_CONFIG_PATH=/config/config.json \\"
    echo "    $IMAGE_NAME"
    echo ""
    echo "  # With custom data and configuration"
    echo "  docker run -p 8000:8000 \\"
    echo "    -v \$(pwd)/data:/data \\"
    echo "    -v \$(pwd)/config.json:/config/config.json \\"
    echo "    -e FINRL_CONFIG_PATH=/config/config.json \\"
    echo "    -e FINRL_DATA_PATH=/data/stock_data.csv \\"
    echo "    $IMAGE_NAME"
    echo ""
    echo "  # With different log level"
    echo "  docker run -p 8000:8000 -e FINRL_LOG_LEVEL=DEBUG $IMAGE_NAME"
    echo ""
    echo "📚 Environment Variables:"
    echo "  FINRL_CONFIG_PATH    - Path to JSON config file"
    echo "  FINRL_DATA_PATH      - Path to stock data CSV"
    echo "  FINRL_INITIAL_AMOUNT - Starting capital (default: 100000)"
    echo "  FINRL_STOCK_DIM      - Number of stocks (default: 1)"
    echo "  FINRL_HMAX           - Max shares per trade (default: 100)"
    echo "  FINRL_LOG_LEVEL      - Logging level (default: INFO)"
    echo ""
    echo "🔗 Next steps:"
    echo "  1. Start the server"
    echo "  2. Test with: curl http://localhost:8000/health"
    echo "  3. Get config: curl http://localhost:8000/config"
    echo "  4. Run example: python ../../../examples/finrl_simple.py"
    echo ""
else
    echo ""
    echo "❌ Build failed!"
    echo ""
    echo "💡 Troubleshooting:"
    echo "  - Ensure Docker is running"
    echo "  - Check if envtorch-base:latest exists"
    echo "  - Verify you're in the OpenEnv root directory"
    echo "  - Check Docker logs: docker logs <container-id>"
    echo ""
    exit 1
fi

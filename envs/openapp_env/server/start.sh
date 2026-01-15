#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Startup script for OpenApp Environment Docker container
# This script starts both the OpenApps server and the FastAPI environment server

set -e

echo "Starting OpenApp Environment..."

# Start OpenApps server in the background
echo "Starting OpenApps server on port ${OPENAPPS_PORT:-5001}..."
cd /app/openapps
# Run launch.py directly - it uses Hydra and needs the config directory
# Redirect OpenApps output to a log file so we can debug if needed
python launch.py > /tmp/openapps.log 2>&1 &
OPENAPPS_PID=$!

# Wait for OpenApps server to be ready
echo "Waiting for OpenApps server to be ready..."
for i in {1..60}; do
    # Check if OpenApps server is responding using curl
    if curl -sf http://localhost:${OPENAPPS_PORT:-5001} >/dev/null 2>&1; then
        echo "OpenApps server is ready on port ${OPENAPPS_PORT:-5001}!"
        break
    fi
    if [ $i -eq 60 ]; then
        echo "ERROR: OpenApps server failed to start within 60 seconds"
        echo "OpenApps log output:"
        cat /tmp/openapps.log || echo "No log file found"
        kill $OPENAPPS_PID 2>/dev/null || true
        exit 1
    fi
    sleep 1
done

# Start the FastAPI environment server
echo "Starting FastAPI environment server on port 8000..."
cd /app/env
exec uvicorn openapp_env.server.app:app --host 0.0.0.0 --port 8000

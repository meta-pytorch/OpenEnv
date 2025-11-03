# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Use the standard openenv base image
# Built from: docker build -t openenv-base:latest -f src/core/containers/images/Dockerfile .
# In GitHub Actions, this is overridden to use the GHCR base image
ARG BASE_IMAGE=openenv-base:latest
FROM ${BASE_IMAGE}

# Install dependencies
COPY src/envs/coding_env/server/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Copy only what's needed for this environment
COPY src/core/ /app/src/core/
COPY src/envs/coding_env/ /app/src/envs/coding_env/

# Copy README for web interface documentation
COPY src/envs/coding_env/README.md /app/README.md

# Set default port (can be overridden via environment variable)
ENV PORT=8000

# Health check (uses PORT env variable)
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Run the FastAPI server (uses PORT env variable)
CMD sh -c "uvicorn envs.coding_env.server.app:app --host 0.0.0.0 --port ${PORT}"

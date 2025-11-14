# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Use the pre-built OpenEnv base image
# Built from: docker build -t openenv-base:latest -f src/core/Dockerfile.openenv-base .
# In CI, this can be overridden to use GHCR or other registries
ARG OPENENV_BASE_IMAGE=openenv-base:latest
FROM ${OPENENV_BASE_IMAGE}

# Install Python dependencies that all environments need
RUN pip install --no-cache-dir \
    numpy>=2.3.4 \
    matplotlib>=3.10.7

# Set working directory
WORKDIR /app

# Copy OpenEnv core (already expected in base image but ensure updated)
COPY src/core/ /app/src/core/

# Copy Maze environment
COPY src/envs/maze_env/ /app/src/envs/maze_env/

# Copy README for web interface documentation
COPY src/envs/maze_env/README.md /app/README.md

# Extend Python path for OpenEnv (base image sets PYTHONPATH=/app/src)
# We prepend Maze paths
ENV PYTHONPATH=/app/src


# Health check (curl provided by openenv-base)
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose default port
EXPOSE 8000

# Run the FastAPI server (uvicorn installed by openenv-base)
CMD ["uvicorn", "envs.maze_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]

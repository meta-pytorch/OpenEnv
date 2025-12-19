# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Dockerfile for Coding Environment
# Supports both in-repo and standalone builds
#
# In-repo build (from repo root):
#   docker build -t coding-env:latest -f src/envs/coding_env/server/Dockerfile .
#
# Standalone build (from coding_env directory with openenv-core on PyPI):
#   docker build -t coding-env:latest -f server/Dockerfile --build-arg BUILD_MODE=standalone .

FROM python:3.11-slim

# Build argument to control mode
ARG BUILD_MODE=in-repo

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy files based on build mode
# For in-repo: copy entire src/ directory from repo root
# For standalone: copy current directory (coding_env package)
COPY ${BUILD_MODE:+src/} ./

# Install dependencies
RUN if [ "$BUILD_MODE" = "in-repo" ]; then \
        # In-repo: install core dependencies directly \
        pip install --no-cache-dir \
            'fastapi>=0.104.0' \
            'pydantic>=2.0.0' \
            'uvicorn[standard]>=0.24.0' \
            'requests>=2.25.0' \
            'smolagents>=1.22.0,<2'; \
    else \
        # Standalone: install from pyproject.toml (includes openenv-core from PyPI) \
        pip install --no-cache-dir -e .; \
    fi

# Convert ARG to ENV so it's available at runtime
ENV BUILD_MODE=${BUILD_MODE}
ENV PYTHONUNBUFFERED=1
ENV ENABLE_WEB_INTERFACE=true

# Set PYTHONPATH based on build mode (evaluated at build time)
RUN if [ "$BUILD_MODE" = "in-repo" ]; then \
        echo "export PYTHONPATH=/app/src" >> /etc/environment; \
    fi

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the server with correct module path based on build mode
# In-repo: envs.coding_env.server.app:app (with PYTHONPATH=/app/src)
# Standalone: coding_env.server.app:app (with package installed)
CMD if [ "$BUILD_MODE" = "in-repo" ]; then \
        export PYTHONPATH=/app/src && uvicorn envs.coding_env.server.app:app --host 0.0.0.0 --port 8000; \
    else \
        uvicorn coding_env.server.app:app --host 0.0.0.0 --port 8000; \
    fi

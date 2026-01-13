# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# NLE requires build dependencies, so we start from a base image with build tools
# Using Python 3.11 for kw_only dataclass support (required by OpenEnv core)
FROM python:3.11-slim

# Install system dependencies needed for NLE
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libbz2-dev \
    flex \
    bison \
    libncurses5-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
# NLE requires cmake to be available during installation
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    requests \
    pydantic \
    numpy

# Install NLE (this will compile NetHack from source)
# Pin gym version for reproducibility and API compatibility
RUN pip install --no-cache-dir nle gym==0.26.2

# Copy OpenEnv core
COPY src/core/ /app/src/core/

# Copy NLE environment implementation
COPY src/envs/nle_env/ /app/src/envs/nle_env/

# Copy README for web interface documentation
COPY src/envs/nle_env/README.md /app/README.md

# Set PYTHONPATH so imports work
ENV PYTHONPATH=/app/src

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the FastAPI server
# Note: workers=1 because NLE uses C extension with global state
CMD ["uvicorn", "envs.nle_env.server.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

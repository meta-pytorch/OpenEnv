# Copyright (c) Yogesh Singla, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Use the standard openenv base image
# Built from: docker build -t openenv-base:latest -f src/core/containers/images/Dockerfile .
# In GitHub Actions, this is overridden to use the GHCR base image

# Use the standard openenv base image
ARG BASE_IMAGE=openenv-base:latest
FROM ${BASE_IMAGE}

# Install Julia using juliaup (official installer - more reliable in Docker)
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install juliaup and Julia
RUN curl -fsSL https://install.julialang.org | sh -s -- --yes --default-channel 1.10

# Add Julia to PATH
ENV PATH="/root/.juliaup/bin:${PATH}"

# Verify Julia installation
RUN julia --version

# Precompile commonly used Julia packages (Test is built-in, but precompile it)
RUN julia -e 'using Test; println("Julia Test module ready")'

# Install Python dependencies
COPY src/envs/julia_env/server/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Environment variable to enable Julia process pool (optional - can be set at runtime)
# Set to "1" to enable process pool, "0" to use standard execution
ENV JULIA_USE_PROCESS_POOL=1
ENV JULIA_POOL_SIZE=64
ENV JULIA_MAX_WORKERS=64
# Copy only what's needed for the Julia environment
COPY src/core/ /app/src/core/
COPY src/envs/julia_env/ /app/src/envs/julia_env/

# Environment variables for port and workers with defaults
ENV PORT=8000
ENV NUM_WORKER=4

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Run the FastAPI server
CMD ["sh", "-c", "python -m uvicorn envs.julia_env.server.app:app --host 0.0.0.0 --port ${PORT} --workers ${NUM_WORKER}"]

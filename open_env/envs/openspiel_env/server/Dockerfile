# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Multi-stage build for OpenSpiel + OpenEnv
# Stage 1: Build OpenSpiel C++ bindings
# Using Python 3.11 to match envtorch-base
FROM python:3.11 AS openspiel-builder

# Avoid interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    clang \
    cmake \
    curl \
    git \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Set up OpenSpiel build directory
RUN mkdir /repo
WORKDIR /repo

# Clone OpenSpiel
RUN git clone https://github.com/google-deepmind/open_spiel.git .

# Run OpenSpiel's installation script (downloads C++ dependencies)
RUN ./install.sh

# Install Python dependencies
RUN pip3 install --no-cache-dir --upgrade setuptools testresources importlib_metadata
RUN pip3 install --no-cache-dir --upgrade -r requirements.txt cmake

# Build OpenSpiel with Python 3.11
RUN mkdir -p build
WORKDIR /repo/build
RUN cmake -DPython3_EXECUTABLE=$(which python3) -DCMAKE_CXX_COMPILER=$(which clang++) ../open_spiel
RUN make -j$(nproc) pyspiel

# Stage 2: Runtime image using published openenv-base
# Uses the standardized base image from GitHub Container Registry
# See: https://github.com/meta-pytorch/OpenEnv/pkgs/container/openenv-base
FROM ghcr.io/meta-pytorch/openenv-base:latest

# Copy OpenSpiel build artifacts from builder
RUN mkdir -p /repo
COPY --from=openspiel-builder /repo /repo

# Install OpenSpiel Python requirements in runtime
WORKDIR /repo
RUN pip3 install --no-cache-dir --upgrade -r requirements.txt

# Set Python path for OpenSpiel
ENV PYTHONPATH=/repo:/repo/build/python:${PYTHONPATH}

# Copy OpenEnv core (base image already set WORKDIR=/app)
WORKDIR /app
COPY src/core/ /app/src/core/

# Copy OpenSpiel environment
COPY src/envs/openspiel_env/ /app/src/envs/openspiel_env/

# Copy README for web interface documentation
COPY src/envs/openspiel_env/README.md /app/README.md

# Extend Python path for OpenEnv (base image set PYTHONPATH=/app/src)
# We prepend OpenSpiel paths
ENV PYTHONPATH=/repo:/repo/build/python:/app/src

# OpenSpiel-specific environment variables (can be overridden at runtime)
ENV OPENSPIEL_GAME=catch
ENV OPENSPIEL_AGENT_PLAYER=0
ENV OPENSPIEL_OPPONENT_POLICY=random

# Health check (curl is provided by envtorch-base)
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Note: EXPOSE 8000 already set by envtorch-base

# Run the FastAPI server (uvicorn installed by envtorch-base)
CMD ["uvicorn", "envs.openspiel_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]

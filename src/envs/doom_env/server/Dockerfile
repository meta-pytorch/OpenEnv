# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Dockerfile for Doom Environment
# This image provides ViZDoom for visual RL research

# Configurable base image - defaults to local build, can be overridden for CI/CD
# Base image provides: fastapi, uvicorn, requests, curl, PYTHONPATH=/app/src
#
# Local build: docker build -t openenv-base:latest -f src/core/containers/images/Dockerfile .
#              docker build -f src/envs/doom_env/server/Dockerfile -t doom-env:latest .
#
# CI/CD build: docker build --build-arg BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest \
#              -f src/envs/doom_env/server/Dockerfile -t doom-env:latest .
ARG BASE_IMAGE=openenv-base:latest
FROM ${BASE_IMAGE}

# Install system dependencies for ViZDoom
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libboost-all-dev \
    libsdl2-dev \
    libfreetype6-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libpng-dev \
    libjpeg-dev \
    libbz2-dev \
    libfluidsynth-dev \
    libgme-dev \
    libopenal-dev \
    zlib1g-dev \
    timidity \
    tar \
    nasm && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY src/envs/doom_env/server/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Copy OpenEnv core (base image already set WORKDIR=/app)
COPY src/core/ /app/src/core/

# Copy Doom environment code
COPY src/envs/doom_env/ /app/src/envs/doom_env/

# Copy README for web interface documentation
COPY src/envs/doom_env/README.md /app/README.md

# Doom-specific environment variables (can be overridden at runtime)
ENV DOOM_SCENARIO=basic
ENV DOOM_SCREEN_RESOLUTION=RES_160X120
ENV DOOM_SCREEN_FORMAT=RGB24
ENV DOOM_WINDOW_VISIBLE=false

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the FastAPI server
CMD ["uvicorn", "envs.doom_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]

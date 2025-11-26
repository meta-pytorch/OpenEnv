# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Dockerfile for Doom Environment
# This image provides ViZDoom for visual RL research with OpenEnv
#
# This Dockerfile works standalone with public images, suitable for:
# - CI/CD pipelines
# - GitHub deployments
# - HuggingFace Spaces
#
# Build from project root:
#   docker build -t doom-env:latest -f src/envs/doom_env/server/Dockerfile .
#
# Run with web interface:
#   docker run -p 8000:8000 -e ENABLE_WEB_INTERFACE=true doom-env:latest

FROM python:3.11-slim

# Set metadata
LABEL maintainer="OpenEnv Team"
LABEL description="Doom RL Environment with ViZDoom and OpenEnv"
LABEL org.opencontainers.image.source="https://github.com/meta-pytorch/OpenEnv"

# Set working directory
WORKDIR /app

# Install system dependencies
# - curl: for healthcheck
# - ViZDoom build dependencies: build-essential, cmake, boost, SDL2, etc.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
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

# Set environment variables
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

# Copy OpenEnv core (required for server functionality)
COPY src/core/ /app/src/core/

# Copy entire doom_env directory
COPY src/envs/doom_env/ /app/src/envs/doom_env/

# Install doom_env and all its dependencies from pyproject.toml
# This includes: openenv-core, fastapi, uvicorn, pydantic, requests, vizdoom, numpy
RUN pip install --no-cache-dir -e /app/src/envs/doom_env

# Doom-specific environment variables (can be overridden at runtime)
ENV DOOM_SCENARIO=basic
ENV DOOM_SCREEN_RESOLUTION=RES_160X120
ENV DOOM_SCREEN_FORMAT=RGB24
ENV DOOM_WINDOW_VISIBLE=false

# Enable web interface by default (set to false to disable)
ENV ENABLE_WEB_INTERFACE=true

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the FastAPI server
# Web interface will be available at /web if ENABLE_WEB_INTERFACE=true
# API documentation available at /docs
CMD ["uvicorn", "envs.doom_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]

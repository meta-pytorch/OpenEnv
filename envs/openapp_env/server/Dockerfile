# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Dockerfile for OpenApp Environment
# This image provides OpenApps web application simulation for UI agent training
#
# This Dockerfile works for both local builds and HuggingFace Spaces deployment:
# - Local build from project root: docker build -t openapp-env:latest -f src/envs/openapp_env/server/Dockerfile src/envs/openapp_env
# - HuggingFace: Automatically deployed via `openenv push`
#
# Run with web interface:
#   docker run -p 8000:8000 -e ENABLE_WEB_INTERFACE=true openapp-env:latest

FROM python:3.11-slim

# Set metadata
LABEL maintainer="OpenEnv Team"
LABEL description="OpenApp Environment with BrowserGym for UI agent training"
LABEL org.opencontainers.image.source="https://github.com/meta-pytorch/OpenEnv"

# Set working directory
WORKDIR /app/env

# Install system dependencies
# - git: required to clone OpenApps from GitHub
# - curl: for healthcheck
# - Playwright/BrowserGym dependencies: fonts, libraries for browser automation
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    wget \
    gnupg \
    # Playwright/Chromium dependencies
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libdbus-1-3 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    libpango-1.0-0 \
    libcairo2 \
    libatspi2.0-0 \
    libxshmfence1 \
    fonts-liberation \
    libappindicator3-1 \
    xdg-utils && \
    rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app/env

# Copy environment files
COPY . .

# Install openapp_env and all its dependencies from pyproject.toml
# This includes: openenv-core, fastapi, uvicorn, pydantic, requests, browsergym, playwright
RUN pip install --no-cache-dir -e .

# Verify installation
RUN python -c "import openapp_env; print('✓ openapp_env installed')" && \
    python -c "import openapp_env.server.app; print('✓ openapp_env.server.app importable')"

# Install Playwright browsers (Chromium for BrowserGym)
# We already installed system dependencies above, so just install the browser
RUN playwright install chromium

# Clone and install OpenApps server
# OpenApps will run as a background service inside the container
WORKDIR /app
RUN git clone https://github.com/facebookresearch/OpenApps.git openapps && \
    cd openapps && \
    pip install --no-cache-dir -e .

# Verify OpenApps installation
RUN python -c "import open_apps; print('✓ OpenApps installed')"

# Copy startup script
WORKDIR /app/env
COPY server/start.sh /app/start.sh
RUN chmod +x /app/start.sh

# OpenApp-specific environment variables (can be overridden at runtime)
ENV OPENAPPS_URL=http://localhost:5001
ENV OPENAPPS_PORT=5001
ENV OPENAPP_HEADLESS=true
ENV OPENAPP_MAX_STEPS=50

# Hydra requires USER environment variable
ENV USER=root

# Enable web interface by default (set to false to disable)
ENV ENABLE_WEB_INTERFACE=true

# Expose ports (8000 for FastAPI, 5001 for OpenApps)
EXPOSE 8000 5001

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the startup script that launches both OpenApps server and FastAPI server
# Web interface will be available at /web if ENABLE_WEB_INTERFACE=true
# API documentation available at /docs
CMD ["/app/start.sh"]

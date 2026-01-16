# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Dockerfile for Coding Environment
# Build from repo root:
#   docker build -t coding-env:latest -f envs/coding_env/server/Dockerfile .

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy core and environment packages
COPY src/ ./src/
COPY envs/coding_env/ ./envs/coding_env/

# Install dependencies
RUN pip install --no-cache-dir \
    'fastapi>=0.104.0' \
    'pydantic>=2.0.0' \
    'uvicorn[standard]>=0.24.0' \
    'requests>=2.25.0' \
    'smolagents>=1.22.0,<2'

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV ENABLE_WEB_INTERFACE=true
ENV PYTHONPATH=/app/src:/app/envs

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the server
CMD ["uvicorn", "coding_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]

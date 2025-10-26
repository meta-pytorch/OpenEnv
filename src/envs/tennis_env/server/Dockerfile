# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Dockerfile for Tennis Environment Server
# Builds a containerized Tennis environment using ALE (Arcade Learning Environment)

FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy source code
COPY src/ /app/src/

# Install Python dependencies
RUN pip install --no-cache-dir \
    fastapi==0.104.1 \
    uvicorn==0.24.0 \
    requests==2.31.0 \
    numpy==1.24.3 \
    ale-py==0.8.1 \
    gymnasium==0.29.1

# Set Python path
ENV PYTHONPATH=/app/src

# Expose port for FastAPI server
EXPOSE 8000

# Environment variables with defaults
ENV TENNIS_MODE=""
ENV TENNIS_DIFFICULTY=""
ENV TENNIS_REPEAT_ACTION_PROB="0.25"
ENV TENNIS_FRAMESKIP="4"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=5)"

# Run the Tennis environment server
CMD ["python", "-m", "uvicorn", "envs.tennis_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]

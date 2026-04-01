# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE} AS builder

WORKDIR /app

# Ensure git is available
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Build argument to control whether we are building standalone or in-repo
ARG BUILD_MODE=in-repo
ARG ENV_NAME=minesweeper

# Copy openenv-core source (assumes build context is OpenEnv root)
COPY src/pyproject.toml /app/openenv-core/pyproject.toml
COPY src/core /app/openenv-core/core
COPY src/openenv_cli /app/openenv-core/openenv_cli
COPY src/__init__.py /app/openenv-core/__init__.py

# Copy the environment source code
COPY src/envs/minesweeper_env /app/env

# For in-repo builds, openenv-core is already in the pyproject.toml dependencies
# For standalone builds, openenv-core will be installed from PyPI via pyproject.toml
WORKDIR /app/env

# Ensure uv is available
RUN if ! command -v uv > /dev/null 2>&1; then \
        curl -LsSf https://astral.sh/uv/install.sh | sh && \
        mv /root/.local/bin/uv /usr/local/bin/uv && \
        mv /root/.local/bin/uvx /usr/local/bin/uvx; \
    fi

# Install dependencies and create virtual environment
RUN uv venv /app/env/.venv && \
    uv pip install --python /app/env/.venv/bin/python --no-cache /app/openenv-core && \
    uv pip install --python /app/env/.venv/bin/python --no-cache .

# Final runtime stage
FROM ${BASE_IMAGE} AS runtime

WORKDIR /app

# Copy the built environment from the builder stage
COPY --from=builder /app/env/.venv /app/.venv

# Copy the environment source code
COPY --from=builder /app/env /app/env

# Set PATH to use the virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# Set PYTHONPATH
ENV PYTHONPATH="/app/env:$PYTHONPATH"

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s CMD curl -f http://localhost:8000/health || exit 1

# Run the FastAPI server
CMD ["/app/.venv/bin/uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]

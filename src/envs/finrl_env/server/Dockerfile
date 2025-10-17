# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#
# FinRL Environment Docker Image
#
# This image wraps FinRL's StockTradingEnv in the OpenEnv HTTP API.
# It supports runtime configuration via environment variables for flexibility.
#

# Use the standard envtorch base image
# Built from: docker build -t envtorch-base:latest -f src/core/containers/images/Dockerfile .
# TODO: Once published, use: FROM ghcr.io/meta-pytorch/openenv-base:latest
FROM envtorch-base:latest

# Install FinRL and its dependencies with pinned versions for reproducibility
RUN pip install --no-cache-dir \
    finrl==0.3.6 \
    yfinance==0.2.28 \
    pandas==2.0.3 \
    numpy==1.24.3 \
    gymnasium==0.29.1 \
    stable-baselines3==2.1.0 \
    matplotlib==3.7.2 \
    ta==0.11.0 \
    stockstats==0.6.2

# Copy core framework (base image set WORKDIR=/app)
COPY src/core/ /app/src/core/

# Copy FinRL environment
COPY src/envs/finrl_env/ /app/src/envs/finrl_env/

# Set working directory for the application
WORKDIR /app/src

# Set Python path explicitly (redundant with base but clear)
ENV PYTHONPATH=/app/src:${PYTHONPATH}

# FinRL runtime configuration via environment variables
# These can be overridden at runtime with -e flags
ENV FINRL_CONFIG_PATH="" \
    FINRL_DATA_PATH="" \
    FINRL_INITIAL_AMOUNT=100000 \
    FINRL_STOCK_DIM=1 \
    FINRL_HMAX=100 \
    FINRL_LOG_LEVEL=INFO

# Document the exposed port
EXPOSE 8000

# Health check (curl is provided by envtorch-base)
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the FastAPI server (uvicorn installed by envtorch-base)
CMD ["uvicorn", "envs.finrl_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Use the pre-built OpenSpiel base image
# Built from: docker build -t openspiel-base:latest -f envs/openspiel_env/server/Dockerfile.openspiel-base .
# In GitHub Actions, this is overridden to use the GHCR base image
ARG OPENSPIEL_BASE_IMAGE=openspiel-base:latest
FROM ${OPENSPIEL_BASE_IMAGE}

# Copy OpenEnv core (base image already set WORKDIR=/app)
WORKDIR /app
COPY src/core/ /app/src/core/

# Copy OpenSpiel environment
COPY envs/openspiel_env/ /app/envs/openspiel_env/

# Copy README for web interface documentation
COPY envs/openspiel_env/README.md /app/README.md

# Extend Python path for OpenEnv (base image set PYTHONPATH=/app/src)
# We prepend OpenSpiel paths
ENV PYTHONPATH=/repo:/repo/build/python:/app/src

# OpenSpiel-specific environment variables (can be overridden at runtime)
ENV OPENSPIEL_GAME=catch
ENV OPENSPIEL_AGENT_PLAYER=0
ENV OPENSPIEL_OPPONENT_POLICY=random

# Health check (curl is provided by openenv-base)
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Note: EXPOSE 8000 already set by openenv-base

# Run the FastAPI server (uvicorn installed by openenv-base)
CMD ["uvicorn", "envs.openspiel_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]

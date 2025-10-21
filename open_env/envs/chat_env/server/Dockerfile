# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Use the standard openenv base image
# Built from: docker build -t openenv-base:latest -f src/core/containers/images/Dockerfile .
# In GitHub Actions, this is overridden to use the GHCR base image
ARG BASE_IMAGE=openenv-base:latest
FROM ${BASE_IMAGE}

# Install additional dependencies for ChatEnvironment
RUN pip install torch transformers

# Copy only what's needed for this environment
COPY src/core/ /app/src/core/
COPY src/envs/chat_env/ /app/src/envs/chat_env/

# Copy README for web interface documentation
COPY src/envs/chat_env/README.md /app/README.md

# Environment variables that can be overridden at runtime
ENV TOKENIZER_NAME=gpt2
ENV SYSTEM_PROMPT="You are a helpful AI assistant."

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the FastAPI server
CMD ["uvicorn", "envs.chat_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]

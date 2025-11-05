# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Use the standard openenv base image
ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE}

# Copy only what's needed for this environment
COPY . /app/src/envs/__ENV_NAME__/

# Install dependencies
RUN pip install --no-cache-dir -r /app/src/envs/__ENV_NAME__/server/requirements.txt && rm /app/src/envs/__ENV_NAME__/server/requirements.txt

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the FastAPI server
CMD ["uvicorn", "envs.__ENV_NAME__.server.app:app", "--host", "0.0.0.0", "--port", "8000"]


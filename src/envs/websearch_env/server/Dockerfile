# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Use the shared OpenEnv base image (Python 3.11)
ARG BASE_IMAGE=openenv-base:latest
FROM ${BASE_IMAGE}

# Install Python dependencies
RUN pip install --no-cache-dir \
    chardet==5.2.0

# Copy OpenEnv core and WebSearch environment sources
COPY src/core/ /app/src/core/
COPY src/envs/websearch_env/ /app/src/envs/websearch_env/

# Optional: health check to ensure server responsiveness
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the WebSearch FastAPI server
CMD ["uvicorn", "envs.websearch_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]

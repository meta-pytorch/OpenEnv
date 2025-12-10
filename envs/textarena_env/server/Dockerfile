# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Use the shared OpenEnv base image (Python 3.11)
ARG BASE_IMAGE=openenv-base:latest
FROM ${BASE_IMAGE}

# Install system libraries required by TextArena (cv2 needs libGL, glib)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install TextArena and Python dependencies
RUN pip install --no-cache-dir \
    textarena==0.6.1 \
    nltk==3.9.2

# Copy OpenEnv core and TextArena environment sources
COPY src/core/ /app/src/core/
COPY envs/textarena_env/ /app/envs/textarena_env/

# Optional: health check to ensure server responsiveness
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the TextArena FastAPI server
CMD ["uvicorn", "envs.textarena_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]


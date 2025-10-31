# Dockerfile for Atari Environment
# This image provides Atari 2600 games via the Arcade Learning Environment (ALE)

# Configurable base image - defaults to local build, can be overridden for CI/CD
# Base image provides: fastapi, uvicorn, requests, curl, PYTHONPATH=/app/src
#
# Local build: docker build -t envtorch-base:latest -f src/core/containers/images/Dockerfile .
#              docker build -f src/envs/atari_env/server/Dockerfile -t atari-env:latest .
#
# CI/CD build: docker build --build-arg BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest \
#              -f src/envs/atari_env/server/Dockerfile -t atari-env:latest .
ARG BASE_IMAGE=openenv-base:latest
FROM ${BASE_IMAGE}

# Install dependencies
COPY src/envs/atari_env/server/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Copy OpenEnv core (base image already set WORKDIR=/app)
COPY src/core/ /app/src/core/

# Copy Atari environment code
COPY src/envs/atari_env/ /app/src/envs/atari_env/

# Copy README for web interface documentation
COPY src/envs/atari_env/README.md /app/README.md

# Atari-specific environment variables (can be overridden at runtime)
ENV ATARI_GAME=pong
ENV ATARI_OBS_TYPE=rgb
ENV ATARI_FULL_ACTION_SPACE=false
ENV ATARI_REPEAT_ACTION_PROB=0.0
ENV ATARI_FRAMESKIP=4

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the FastAPI server
CMD ["uvicorn", "envs.atari_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]

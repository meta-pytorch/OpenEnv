# Dockerfile for ARE Environment
# This image provides the Agents Research Environment (ARE) for OpenEnv

# Configurable base image - defaults to local build, can be overridden for CI/CD
# Base image provides: fastapi, uvicorn, requests, curl, PYTHONPATH=/app/src
#
# Local build: docker build -t openenv-base:latest -f src/core/containers/images/Dockerfile .
#              docker build -f src/envs/are_env/server/Dockerfile -t are-env:latest .
#
# CI/CD build: docker build --build-arg BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest \
#              -f src/envs/are_env/server/Dockerfile -t are-env:latest .
ARG BASE_IMAGE=openenv-base:latest
FROM ${BASE_IMAGE}

# Phase 3.1: Install ARE dependencies for real integration
RUN pip install --no-cache-dir meta-agents-research-environments

# Copy OpenEnv core (base image already set WORKDIR=/app)
COPY src/core/ /app/src/core/

# Copy ARE environment code
COPY src/envs/are_env/ /app/src/envs/are_env/

# Copy README for web interface documentation
COPY src/envs/are_env/README.md /app/README.md

# Phase 1: No environment variables needed
# Phase 2: Add ARE-specific environment variables

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the FastAPI server
CMD ["uvicorn", "envs.are_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]

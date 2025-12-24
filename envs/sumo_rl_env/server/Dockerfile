# Dockerfile for SUMO-RL Environment
# This image provides traffic signal control via SUMO (Simulation of Urban MObility)

# Configurable base image - defaults to local build, can be overridden for CI/CD
# Base image provides: fastapi, uvicorn, requests, curl, PYTHONPATH=/app/src
#
# Local build: docker build -t envtorch-base:latest -f src/core/containers/images/Dockerfile .
#              docker build -f envs/sumo_rl_env/server/Dockerfile -t sumo-rl-env:latest .
#
# CI/CD build: docker build --build-arg BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest \
#              -f envs/sumo_rl_env/server/Dockerfile -t sumo-rl-env:latest .
ARG BASE_IMAGE=envtorch-base:latest
FROM ${BASE_IMAGE}

# Install SUMO system dependencies
# SUMO is available in Debian repositories
RUN apt-get update && apt-get install -y --no-install-recommends \
    sumo \
    sumo-tools \
    && rm -rf /var/lib/apt/lists/*

# Set SUMO_HOME environment variable
ENV SUMO_HOME=/usr/share/sumo

# Install SUMO-RL and Python dependencies
# sumo-rl includes: gymnasium, pettingzoo, numpy, pandas, sumolib, traci
RUN pip install --no-cache-dir \
    gymnasium>=0.28 \
    pettingzoo>=1.24.3 \
    numpy>=1.24.0 \
    pandas>=2.0.0 \
    sumolib>=1.14.0 \
    traci>=1.14.0 \
    sumo-rl>=1.4.5

# Copy OpenEnv core (base image already set WORKDIR=/app)
COPY src/core/ /app/src/core/

# Copy SUMO-RL environment code (includes nets/)
COPY envs/sumo_rl_env/ /app/envs/sumo_rl_env/

# Copy example network files to expected location
# Default: single-intersection (simple 4-way intersection)
COPY envs/sumo_rl_env/nets/single-intersection/ /app/nets/single-intersection/

# SUMO environment variables (can be overridden at runtime)
ENV SUMO_NET_FILE=/app/nets/single-intersection/single-intersection.net.xml
ENV SUMO_ROUTE_FILE=/app/nets/single-intersection/single-intersection.rou.xml
ENV SUMO_NUM_SECONDS=20000
ENV SUMO_DELTA_TIME=5
ENV SUMO_YELLOW_TIME=2
ENV SUMO_MIN_GREEN=5
ENV SUMO_MAX_GREEN=50
ENV SUMO_REWARD_FN=diff-waiting-time
ENV SUMO_SEED=42

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the FastAPI server
CMD ["uvicorn", "envs.sumo_rl_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]

# Build on top of the shared OpenEnv base image
ARG BASE_IMAGE=openenv-base:latest
FROM ${BASE_IMAGE}

# Install git for pip VCS installs
RUN apt-get update && apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Install environment-specific dependencies
RUN pip install --no-cache-dir "gym==0.25.2" "numpy<2.0" \
    git+https://github.com/Danielhp95/gym-connect4

# Copy the framework core plus this environment
COPY src/core/ /app/src/core/
COPY src/envs/connect4_env/ /app/src/envs/connect4_env/
COPY src/envs/connect4_env/README.md /app/README.md

# Simple health check - the web UI reuses /health
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the FastAPI server
CMD ["uvicorn", "envs.connect4_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]

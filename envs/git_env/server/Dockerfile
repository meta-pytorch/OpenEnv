# Dockerfile for Git Environment
# Connects to an external shared Gitea service for task-based isolation
# Optimized for fast resets and minimal resource usage

# Use the standard openenv base image
ARG BASE_IMAGE=openenv-base:latest
FROM ${BASE_IMAGE}

# Install git and curl (no Gitea binary needed - connects to external service)
RUN apt-get update && apt-get install -y \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create workspace directory for git operations
RUN mkdir -p /workspace && chmod 777 /workspace

# Copy core and environment code
COPY src/core/ /app/src/core/
COPY envs/git_env/ /app/envs/git_env/

# Environment variables for Gitea connection
# These MUST be provided at runtime via -e flags or --env-file
# See .env.example for required variables
ENV WORKSPACE_DIR=/workspace

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the FastAPI server
CMD ["uvicorn", "envs.git_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]

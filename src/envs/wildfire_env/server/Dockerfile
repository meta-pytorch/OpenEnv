# Build ARG for CI/CD consistency
ARG BASE_IMAGE=openenv-base:latest
FROM ${BASE_IMAGE}

# Work in app directory (already used by base image)
WORKDIR /app

# Copy OpenEnv core
COPY src/core/ /app/src/core/

# Copy Wildfire environment
COPY src/envs/wildfire_env/ /app/src/envs/wildfire_env/

# Environment variables (override at runtime if needed)
ENV PYTHONPATH=/app/src

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run FastAPI server
CMD ["uvicorn", "envs.wildfire_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
